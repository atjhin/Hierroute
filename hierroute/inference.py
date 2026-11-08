"""
inference.py — Running a trained HierRouteNet over particle images.

Preprocessing matches training-time evaluation: greyscale -> Resize((R, R)) -> ToTensor ->
repeat to 3 channels, where R is the model's training resolution.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .flowcam import is_mosaic_name


def eval_transform(resolution: int) -> transforms.Compose:
    return transforms.Compose([transforms.Resize((resolution, resolution)), transforms.ToTensor()])


def find_crops(root: str | Path, labelled: bool) -> tuple[list[str], list[str | None]]:
    """
    Particle crop ``.tif`` files under ``root``.

    labelled=True:  one folder per class directly under ``root``; the folder name is the label and
                    only files directly inside it are used (so a ``CSV_Data`` or mosaic folder
                    next to the class folders is ignored).
    labelled=False: every ``.tif`` under ``root`` at any depth, except FlowCam mosaics and
                    ``_bin`` masks.

    Returns:
        (paths, labels) sorted by folder then file name; labels are None when unlabelled.
    """
    root = Path(root)
    paths, labels = [], []
    if labelled:
        for folder in sorted(os.listdir(root)):
            folder_path = root / folder
            if not folder_path.is_dir():
                continue
            for name in sorted(os.listdir(folder_path)):
                if name.lower().endswith(".tif") and (folder_path / name).is_file():
                    paths.append(str(folder_path / name))
                    labels.append(folder)
    else:
        for path in sorted(root.rglob("*")):
            name = path.name
            if name.lower().endswith(".tif") and not name.endswith("_bin.tif") and not is_mosaic_name(name):
                paths.append(str(path))
                labels.append(None)
    return paths, labels


class CropDataset(Dataset):
    """Individual crop files -> ``(image (3, R, R), index)``."""

    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("L")
        return self.transform(image).repeat(3, 1, 1), idx


@torch.no_grad()
def predict_with_confidences(model, x: torch.Tensor) -> tuple[list[list[int]], list[list[float]]]:
    """
    Greedy root -> leaf routing, recording each decision's local softmax confidence.

    Returns:
        paths       : paths[i] = [root, ..., deepest node reached]
        confidences : confidences[i][d] is the confidence of the decision that produced paths[i][d + 1]
    """
    feat = model.shared(x)
    if model.backbone_type.startswith("swin"):
        feat = feat.permute(0, 3, 1, 2)
    if model.expert_type != "cnn":
        feat = torch.flatten(model.pool(feat), 1)

    B = feat.shape[0]
    current = [model.hierarchy.root] * B
    paths = [[model.hierarchy.root] for _ in range(B)]
    confidences = [[] for _ in range(B)]

    while not all(model.hierarchy.is_leaf(n) for n in current):
        node_to_indices: dict[int, list[int]] = {}
        for i, node_id in enumerate(current):
            if not model.hierarchy.is_leaf(node_id):
                node_to_indices.setdefault(node_id, []).append(i)
        for node_id, indices in node_to_indices.items():
            weights, _ = model.local_classifiers[str(node_id)](feat[indices])
            conf, child_idx = weights.max(dim=-1)
            children = model.hierarchy.nodes[node_id].children
            for sample_idx, cidx, c in zip(indices, child_idx.tolist(), conf.tolist()):
                current[sample_idx] = children[cidx]
                paths[sample_idx].append(children[cidx])
                confidences[sample_idx].append(c)

    return paths, confidences


@dataclass
class Prediction:
    path: list[int]            # node-ID path, root -> deepest node reached
    confidences: list[float]   # per-decision local confidence
    predicted_1: str           # greedy leaf name
    predicted_2: str           # highest path-probability leaf other than predicted_1


class Predictor:
    """Batched inference for one model: greedy path, per-level confidence and a runner-up leaf."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        hierarchy = model.hierarchy
        self.id_to_name = {nid: node.name for nid, node in hierarchy.nodes.items()}
        self.leaf_ids = [nid for nid in sorted(hierarchy.nodes) if hierarchy.is_leaf(nid)]
        self.leaf_ids_tensor = torch.tensor(self.leaf_ids, device=device)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> list[Prediction]:
        images = images.to(self.device, non_blocking=True)
        paths, confidences = predict_with_confidences(self.model, images)
        leaf_probs, _ = self.model(images)
        ranked = torch.argsort(leaf_probs[:, self.leaf_ids_tensor], dim=1, descending=True).tolist()
        results = []
        for i, (path, confs) in enumerate(zip(paths, confidences)):
            predicted_1 = self.id_to_name[path[-1]]
            predicted_2 = next(self.id_to_name[self.leaf_ids[j]] for j in ranked[i]
                               if self.id_to_name[self.leaf_ids[j]] != predicted_1)
            results.append(Prediction(path, confs, predicted_1, predicted_2))
        return results


def level_columns(prediction: Prediction, id_to_name: dict[int, str], n_levels: int,
                  thresholds=None) -> dict:
    """confidence_level_d / predicted_level_d (and pass_level_d when gated) for d = 1..n_levels."""
    row = {}
    confs = prediction.confidences
    for d in range(1, n_levels + 1):
        row[f"confidence_level_{d}"] = confs[d - 1] if d <= len(confs) else np.nan
    if thresholds is not None:
        for d in range(1, n_levels + 1):
            conf = row[f"confidence_level_{d}"]
            row[f"pass_level_{d}"] = (conf >= thresholds[d - 1]) if not np.isnan(conf) else np.nan
    for d in range(1, n_levels + 1):
        row[f"predicted_level_{d}"] = id_to_name[prediction.path[d]] if d < len(prediction.path) else None
    return row
