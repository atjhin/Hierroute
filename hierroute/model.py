from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from .extra_functions import validate_checkpoint
from .hierarchy import Hierarchy

class FocalLoss(nn.Module):
    """
    Focal loss for binary cross-entropy on probability inputs with one-hot targets.
    Applies modulating factor (1 - pt)^gamma to down-weight easy examples.
    """
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input: [N, C], float32 — probabilities (already softmaxed/path-product)
        target: [N, C], float32 — one-hot encoded targets
        Returns: [N, C] per-element focal BCE losses (unreduced)
        """
        bce = -(target * torch.log(input) + (1 - target) * torch.log(1 - input))
        pt = target * input + (1 - target) * (1 - input)
        focal_weight = (1 - pt) ** self.gamma
        return focal_weight * bce
        

class Expert(nn.Module):
    """
    A local classifier (expert) assigned to a single internal node in the hierarchy.

    Each expert learns to route an input feature vector to one of the node's children.
    Three architectures are supported, selected via expert_type:

    - ``"linear"``: a single Linear(in_dim → num_children) layer.
    - ``"mlp"``:    two-layer MLP with a 2×in_dim hidden layer and ReLU activation.
    - ``"cnn"``:    two Conv2d layers (1×1 then 2×2) followed by a LazyLinear output;
                    expects 4D input (B, C, H, W) rather than a flat vector.

    In **soft** mode (training) the expert returns softmax probabilities over children.
    In **hard** mode (inference) it returns the argmax child index.

    Attributes:
        layer       (nn.Sequential): The classifier network.
        mode        (str):           ``"soft"`` or ``"hard"``.
        expert_type (str):           One of ``"linear"``, ``"mlp"``, ``"cnn"``.
    """

    def __init__(self, in_dim: int, num_children: int, mode: str = "soft", expert_type: str = "mlp") -> None:
        super().__init__()
        if expert_type == "linear":
            self.layer = nn.Sequential(nn.Linear(in_dim, num_children))
        elif expert_type == "mlp":
            self.layer = nn.Sequential(
                nn.Linear(in_dim, 2*in_dim),
                nn.ReLU(),
                nn.Linear(2*in_dim, num_children))
        elif expert_type == "cnn":
            self.layer = nn.Sequential(
                nn.Conv2d(in_dim, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 64, kernel_size=2),
                nn.ReLU(),
                nn.Flatten(1),
                nn.LazyLinear(num_children)) # allows different resolution images.
        else:
            raise ValueError(f"Unsupported expert_type: {expert_type}")
        self.mode = mode
        self.expert_type = expert_type

    def forward(self, x: torch.Tensor):
        """
        Run the expert on a batch of feature vectors (or feature maps for CNN).

        Args:
            x (torch.Tensor): Feature input.
                              Shape ``(B, in_dim)`` for linear/mlp; ``(B, C, H, W)`` for cnn.

        Returns:
            tuple:
                - **weights / indices** — soft probabilities ``(B, num_children)`` in soft mode;
                  argmax child indices ``(B,)`` in hard mode.
                - **logits** ``(B, num_children)`` — raw pre-softmax scores.
        """
        logits = self.layer(x)
        if self.mode == "soft":
            weights = torch.softmax(logits, dim=-1)
            return weights, logits

        elif self.mode == "hard":
            indices = torch.argmax(logits, dim=-1)
            return indices, logits

        else:
            raise ValueError("mode must be 'soft' or 'hard'")

class HierRouteNet(nn.Module):
    """
    Hierarchical Mixture-of-Experts classification network.

    One :class:`Expert` is instantiated per internal node of the taxonomy tree.
    During a forward pass each expert produces soft routing weights over its children;
    leaf probabilities are obtained by multiplying the conditional weights along every
    root-to-leaf path (path-product).  At inference time, :meth:`predict` greedily
    follows the argmax child at each node.

    Supported backbones:
        - ``"efficientnet_b0"`` — EfficientNet-B0, ImageNet pretrained (5.3 M params).
        - ``"swin_t"``          — Swin Transformer Tiny, ImageNet pretrained (28.3 M params).
        - ``"swin_s"``          — Swin Transformer Small, ImageNet pretrained (49.6 M params).

    Attributes:
        hierarchy         (Hierarchy):    Taxonomy tree used to structure routing.
        label_to_id       (dict):         Mapping from class name to integer node_id.
        seed              (int):          Random seed passed at construction.
        loss_type         (str):          ``"bce"`` or ``"focal"``.
        backbone_type     (str):          Name of the backbone used.
        expert_type       (str):          One of ``"linear"``, ``"mlp"``, ``"cnn"``.
        shared            (nn.Sequential):Backbone feature extractor (without classifier head).
        pool              (nn.Module):    Global average pooling applied after the backbone.
        feature_dim       (int):          Number of channels output by the backbone.
        local_classifiers (nn.ModuleDict):Experts keyed by ``str(node_id)`` for each internal node.
        leaf_index        (torch.Tensor): Float binary mask of shape ``(num_nodes,)``; 1 at leaf positions.
        loss              (nn.Module):    Loss module — BCELoss or FocalLoss.
    """

    def __init__(
        self,
        hierarchy: Hierarchy,
        label_to_id: dict[str, int],
        weights_directory: str | None = None,
        checkpoint_dir: str | None = None,
        seed: int = 123,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        backbone: str = "efficientnet_b0",
        freeze_backbone: bool = False,
        expert_type: str = "mlp",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.hierarchy = hierarchy
        self.seed = seed
        self.label_to_id = label_to_id
        self.loss_type = loss_type
        self.backbone_type = backbone
        self.expert_type = expert_type

        # pretrained=False skips torchvision's ImageNet download, for when a trained
        # checkpoint is about to overwrite every backbone weight anyway.
        if backbone == "efficientnet_b0":
            bb = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self.shared = bb.features
            self.pool = bb.avgpool
            self.feature_dim = bb.classifier[1].in_features
        elif backbone == "swin_t":
            bb = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
            self.shared = bb.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.feature_dim = bb.head.in_features
        elif backbone == "swin_s":
            bb = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1 if pretrained else None)
            self.shared = bb.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.feature_dim = bb.head.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for param in self.shared.parameters():
                param.requires_grad = False

        self.local_classifiers = nn.ModuleDict()
        for node_id, node in hierarchy.nodes.items():
            if len(node.children) > 0:
                self.local_classifiers[str(node_id)] = Expert(self.feature_dim, len(node.children), expert_type=expert_type)
        self.leaf_index = torch.tensor(hierarchy.get_leaf_index(), dtype=torch.float32)
        if loss_type == "focal":
            self.loss = FocalLoss(gamma=focal_gamma)
        else:
            self.loss = nn.BCELoss(reduction='none')

        if checkpoint_dir is not None:
            pt_files = list(Path(checkpoint_dir).glob("*.pt"))
            if not pt_files:
                raise FileNotFoundError(f"No .pt file found in '{checkpoint_dir}'")
            checkpoint_path = pt_files[0]
            ckpt_state = validate_checkpoint(checkpoint_path, self, backbone, expert_type)
            self.load_state_dict(ckpt_state)
            print(f"Loaded checkpoint: {checkpoint_path}")


    def forward(self, x: torch.Tensor):
        """
        Forward pass: extract features, run all experts, compute leaf probabilities.

        Uses soft routing — every expert returns softmax weights, and each leaf's
        probability is the product of conditional weights along its root-to-leaf path.

        Args:
            x (torch.Tensor): Input image batch of shape ``(B, 3, H, W)``.

        Returns:
            tuple:
                - **logit_tensor** ``(B, num_nodes)`` — path-product probability for every node;
                  only leaf positions are meaningful (internal nodes equal the cumulative
                  path probability up to that node).
                - **node_logits** ``dict[str, Tensor]`` — raw expert logits keyed by
                  ``str(node_id)`` for each internal node; shape ``(B, num_children)``.
        """
        # --- Backbone ---
        x = self.shared(x)
        if self.backbone_type.startswith("swin"):
            x = x.permute(0, 3, 1, 2)              # (B, H, W, C) -> (B, C, H, W)

        if self.expert_type == "cnn":
            feat = x                                # (B, C, H, W) — pre-pool
        else:
            feat = torch.flatten(self.pool(x), 1)   # (B, feature_dim)

        # --- Run all local classifiers in one pass ---
        # Store conditional probs for every internal node
        # node_probs[node_id] -> (B, num_children) soft probabilities
        node_probs = {}
        node_logits = {}
        for node_id, classifier in self.local_classifiers.items():
            probs, logits = classifier.forward(feat) # (B, num_children)
            node_probs[node_id] = probs
            node_logits[node_id] = logits

        # --- Compute leaf probabilities via path products ---
        # For each leaf, walk root -> leaf and multiply conditional probs
        # P(leaf) = P(child_k | node_n) * P(node_n | node_n-1) * ... * P(node_1 | root)
        leaf_probs = [-1] * len(self.hierarchy.nodes)
        for node_id, node in self.hierarchy.nodes.items():
            path = self.hierarchy.get_path_to_root(node_id)

            # Start with prob=1 and multiply down the path
            B = feat.shape[0]
            prob = torch.ones(B, device=feat.device) # (B,)

            for i in range(len(path) - 1):
                parent = path[i]
                child  = path[i + 1]

                child_idx = self.hierarchy.nodes[parent].children.index(child)

                prob = prob * node_probs[str(parent)][:, child_idx]  # (B,)
            # print(prob)
            leaf_probs[node_id] = prob  # (B,)

        logit_tensor = torch.stack(leaf_probs).transpose(0,1)

        return logit_tensor, node_logits


    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, list[list[int]]]:
        """
        Predict class labels by greedily following argmax decisions from root to leaf,
        recording the expert path taken for each sample.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            leaf_ids : LongTensor of shape (B,) — predicted leaf node_id per sample
            paths    : list of lists — paths[i] is [root, ..., leaf] for sample i,
                       reflecting the exact sequence of expert decisions made
        """
        feat = self.shared(x)
        if self.backbone_type.startswith("swin"):
            feat = feat.permute(0, 3, 1, 2)     # (B, H, W, C) -> (B, C, H, W)

        if self.expert_type == "cnn":
            pass                                 # keep (B, C, H, W) for CNN experts
        else:
            feat = torch.flatten(self.pool(feat), 1)  # (B, feature_dim)

        B = feat.shape[0]
        current_nodes = [self.hierarchy.root] * B
        paths = [[self.hierarchy.root] for _ in range(B)]

        while not all(self.hierarchy.is_leaf(n) for n in current_nodes):
            # Group sample indices by their current node
            node_to_indices = {}
            for i, node_id in enumerate(current_nodes):
                if not self.hierarchy.is_leaf(node_id):
                    node_to_indices.setdefault(node_id, []).append(i)

            for node_id, indices in node_to_indices.items():
                group_feat = feat[indices]                                  # (G, feature_dim)
                _, logits = self.local_classifiers[str(node_id)](group_feat)  # (G, num_children)
                child_idx = torch.argmax(logits, dim=-1)                    # (G,)
                children = self.hierarchy.nodes[node_id].children
                for sample_idx, cidx in zip(indices, child_idx.tolist()):
                    next_node = children[cidx]
                    current_nodes[sample_idx] = next_node
                    paths[sample_idx].append(next_node)

        leaf_ids = torch.tensor(current_nodes, dtype=torch.long, device=feat.device)
        return leaf_ids, paths


    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the masked leaf loss over a batch.

        Loss is evaluated only at leaf node positions (controlled by ``self.leaf_index``),
        summed per sample, normalised by the number of leaves, then averaged over the batch.

        Args:
            logits  (torch.Tensor): Path-product probabilities ``(B, num_nodes)``
                                    from :meth:`forward`.
            targets (torch.Tensor): One-hot soft labels ``(B, num_nodes)``
                                    from ``HierImageDataset.collate_fn``.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        leaf_index = self.leaf_index.to(logits.device)
        logits = logits.clamp(1e-7, 1 - 1e-7)
        loss = self.loss(logits, targets) * leaf_index
        loss = loss.sum(dim=1) / self.leaf_index.sum()
        return loss.mean()



