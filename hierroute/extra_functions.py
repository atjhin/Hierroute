from __future__ import annotations

import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from typing import TYPE_CHECKING
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

if TYPE_CHECKING:
    from .model import HierRouteNet

def visualize_swin_attention(
    model: HierRouteNet,
    image_tensor: torch.Tensor,
    stage: int = 5,
    overlay: bool = True,
    show_input: bool = True,
    interp: str = 'bilinear',
    label: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualizes self-attention maps from a chosen stage of the swin_t backbone.

    Hooks each SwinTransformerBlock's attention module to capture its normalized
    spatial input (B, H, W, C), then manually applies the qkv projection and
    recomputes attention weights (softmax(QK^T / sqrt(d_head))), averaged across
    heads, and overlays the result on the original image.

    Best used with stage=5 (4x4 spatial resolution, single window) for 64x64 images.

    Args:
        model:        HierRouteNet with swin_t backbone (eval mode recommended).
        image_tensor: (1, 3, H, W) float tensor, same preprocessing as training.
        stage:        Index into model.shared (default 5 = stage 3, 4x4 patches).
        overlay:      If True (default), overlays the attention map on the input image.
                      If False, shows the raw attention heatmap without the image.
        show_input:   If True (default), includes the input image as the first panel.
                      If False, only the per-block attention maps are shown.
        interp:       Interpolation mode for upsampling the attention map to input
                      size. 'bilinear' (default) gives smooth gradients; 'nearest'
                      gives sharp token blocks — useful for seeing the raw 2×2 or
                      4×4 token structure at later stages.
        save_path:    Optional path to save the figure (e.g. 'attention.png').

    Returns:
        fig: matplotlib Figure.
    """
    import torch.nn.functional as F

    stage_module = model.shared[stage]
    n_blocks = len(stage_module)

    # Hook block.attn (ShiftedWindowAttention) to capture its normalized input (B, H, W, C).
    # Hooking attn.qkv is unreliable across torchvision versions because the windowing
    # pre-processing happens inside the module before qkv is called.
    captured = {}

    def make_hook(idx):
        def hook(module, input, output):
            captured[idx] = (input[0].detach(), module)   # (B, H, W, C), module ref
        return hook

    handles = []
    for i, block in enumerate(stage_module):
        handles.append(block.attn.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        _ = model.shared(image_tensor.to(next(model.parameters()).device))

    for h in handles:
        h.remove()

    attn_maps = []

    for idx in range(n_blocks):
        x_spatial, attn_module = captured[idx]          # (B, H, W, C)
        B, H_f, W_f, C = x_spatial.shape
        nP      = H_f * W_f
        n_heads = attn_module.num_heads
        d_head  = C // n_heads

        x_flat = x_spatial.reshape(B, nP, C)
        with torch.no_grad():
            qkv = attn_module.qkv(x_flat)               # (B, nP, 3*C)

        q, k, _ = qkv.chunk(3, dim=-1)
        q = q.reshape(B, nP, n_heads, d_head).permute(0, 2, 1, 3)
        k = k.reshape(B, nP, n_heads, d_head).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * (d_head ** -0.5)
        attn = attn.softmax(dim=-1)                     # (B, n_heads, nP, nP)

        attn_mean = attn.mean(dim=0).mean(dim=0)        # (nP, nP) — avg over batch & heads
        attended  = attn_mean.mean(dim=0)               # (nP,)    — avg incoming attention per patch

        attn_spatial = attended.reshape(H_f, W_f).cpu().numpy()
        attn_maps.append(attn_spatial)

    img_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    if img_np.shape[2] == 1:
        img_np = img_np.squeeze(-1)

    n_cols = n_blocks + (1 if show_input else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))
    if n_cols == 1:
        axes = [axes]
    title = f'Swin-T Self-Attention Maps — stage index {stage}'
    if label is not None:
        title += f' — {label}'
    fig.suptitle(title,
                 fontsize=11, fontweight='bold')

    offset = 0
    if show_input:
        axes[0].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
        axes[0].set_title('Input', fontsize=9)
        axes[0].axis('off')
        offset = 1

    for i, attn_map in enumerate(attn_maps):
        interp_kwargs = {'align_corners': False} if interp == 'bilinear' else {}
        upsampled = F.interpolate(
            torch.tensor(attn_map).unsqueeze(0).unsqueeze(0).float(),
            size=image_tensor.shape[-2:], mode=interp, **interp_kwargs
        ).squeeze().numpy()
        upsampled = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)

        if overlay:
            axes[i + offset].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
            im = axes[i + offset].imshow(upsampled, cmap='hot', alpha=0.55)
        else:
            im = axes[i + offset].imshow(upsampled, cmap='hot')
        axes[i + offset].set_title(f'Block {i}', fontsize=9)
        axes[i + offset].axis('off')

    plt.tight_layout()

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label('Attention magnitude', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_path}')

    return fig


def visualize_gradcam(
    model: HierRouteNet,
    image_tensor: torch.Tensor,
    target_leaf: str,
    weights: list[float],
    stage: int = 5,
    overlay: bool = True,
    show_input: bool = True,
    label: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualizes Grad-CAM from a chosen stage of the swin_t backbone, using a
    weighted sum of log-probabilities along the root-to-leaf hierarchy path as
    the score to differentiate.

    Args:
        model:        HierRouteNet with swin_t backbone (will be set to eval mode).
        image_tensor: (1, 3, H, W) float tensor, same preprocessing as training.
        target_leaf:  Name of the target leaf class (e.g. "Calanoid").
        weights:      List of floats, one per hierarchy edge (e.g. [1,1,1] for full
                      path, [0,0,1] for species level only, [1,0,0] for top level).
                      Length must equal the depth of target_leaf in the hierarchy.
        stage:        Index into model.shared (default 5 = stage 3, 4x4 spatial
                      resolution for 64x64 images).
        overlay:      If True (default), overlays the CAM on the input image.
                      If False, shows the raw CAM heatmap without the image.
        show_input:   If True (default), includes the input image as the first panel.
                      If False, only the Grad-CAM panel is shown.
        label:        Optional string appended to the figure title.
        save_path:    Optional path to save the figure (e.g. 'gradcam.png').

    Returns:
        fig: matplotlib Figure with one or two panels (input image + Grad-CAM).
    """
    import torch.nn.functional as F

    # --- 1. Hierarchy lookup ---
    name_to_id = {node.name: nid for nid, node in model.hierarchy.nodes.items()}
    if target_leaf not in name_to_id:
        raise ValueError(
            f"'{target_leaf}' not found in hierarchy. "
            f"Available names: {sorted(name_to_id.keys())}"
        )
    leaf_id = name_to_id[target_leaf]
    path = model.hierarchy.get_path_to_root(leaf_id)  # [root_id, ..., leaf_id] as ints

    if len(weights) != len(path) - 1:
        raise ValueError(
            f"'weights' has {len(weights)} entries but path from root to "
            f"'{target_leaf}' has {len(path) - 1} edges. "
            f"Provide exactly {len(path) - 1} weights."
        )

    # --- 2. Register hooks on the chosen stage ---
    captured_acts = {}
    captured_grads = {}

    handles = [
        model.shared[stage].register_forward_hook(
            lambda m, i, o: captured_acts.update({'value': o})
        ),
        model.shared[stage].register_full_backward_hook(
            lambda m, gi, go: captured_grads.update({'value': go[0]})
        ),
    ]

    model.eval()
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    fig = None
    try:
        # --- 3. Forward pass (no torch.no_grad() — gradients must flow) ---
        logit_tensor, node_logits = model(image_tensor)

        # --- 4. Score: weighted sum of log-probs along root-to-leaf path ---
        score = sum(
            weights[d] * F.log_softmax(node_logits[str(parent_id)], dim=-1)[0,
                model.hierarchy.nodes[parent_id].children.index(child_id)]
            for d, (parent_id, child_id) in enumerate(zip(path[:-1], path[1:]))
        )

        # --- 5. Backward pass ---
        model.zero_grad()
        score.backward()

        # --- 6. CAM computation (channels-last: B, H_f, W_f, C) ---
        acts  = captured_acts['value'].detach()   # (B, H_f, W_f, C)
        grads = captured_grads['value'].detach()  # (B, H_f, W_f, C)
        alpha = grads.mean(dim=(1, 2))            # (B, C) — pool over spatial dims
        cam   = F.relu((acts[0] * alpha[0]).sum(-1))  # (H_f, W_f)

        # --- 7. Normalise + upsample ---
        cam_np = cam.cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        cam_up = F.interpolate(
            torch.tensor(cam_np).unsqueeze(0).unsqueeze(0).float(),
            size=image_tensor.shape[-2:], mode='bilinear', align_corners=False
        ).squeeze().numpy()
        cam_up = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-8)

        # --- 8. Image prep ---
        img_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(-1)

        # --- 9. Plot ---
        n_cols = 2 if show_input else 1
        fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))
        if n_cols == 1:
            axes = [axes]
        title = f'Grad-CAM — stage {stage} — target: {target_leaf}'
        if label is not None:
            title += f' — {label}'
        fig.suptitle(title, fontsize=11, fontweight='bold')

        if show_input:
            axes[0].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
            axes[0].set_title('Input', fontsize=9)
            axes[0].axis('off')

        cam_ax = axes[1] if show_input else axes[0]
        if overlay:
            cam_ax.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
            im = cam_ax.imshow(cam_up, cmap='hot', alpha=0.55)
        else:
            im = cam_ax.imshow(cam_up, cmap='hot')
        cam_ax.set_title('Grad-CAM', fontsize=9)
        cam_ax.axis('off')

        plt.tight_layout()

        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label('Grad-CAM magnitude', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Saved → {save_path}')

    finally:
        for h in handles:
            h.remove()

    return fig


def validate_checkpoint(checkpoint_path, model, backbone: str, expert_type: str) -> dict:
    """
    Load a checkpoint and verify it is compatible with the given model.

    Compares keys and parameter shapes between the checkpoint state dict and
    the model's current state dict.  Raises a :class:`ValueError` with a
    categorised diagnostic message if any mismatch is found.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        model:           The instantiated model to validate against.
        backbone (str):  Backbone name used to build the model (included in error messages).
        expert_type (str): Expert type used to build the model (included in error messages).

    Returns:
        dict: The loaded state dict, ready to be passed to ``model.load_state_dict()``.

    Raises:
        ValueError: If the checkpoint cannot be read or is structurally incompatible.
    """
    try:
        ckpt_state = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        raise ValueError(f"Failed to read checkpoint '{checkpoint_path}': {e}") from e

    model_state = model.state_dict()
    model_keys  = set(model_state.keys())
    ckpt_keys   = set(ckpt_state.keys())

    missing      = model_keys - ckpt_keys
    unexpected   = ckpt_keys  - model_keys
    shape_errors = []
    for k in model_keys & ckpt_keys:
        try:
            if model_state[k].shape != ckpt_state[k].shape:
                shape_errors.append(
                    f"{k}: checkpoint {ckpt_state[k].shape} vs model {model_state[k].shape}"
                )
        except RuntimeError:
            pass  # skip uninitialized (lazy) parameters

    if not (missing or unexpected or shape_errors):
        return ckpt_state

    all_bad = missing | unexpected | {e.split(":")[0] for e in shape_errors}
    lines = [f"Checkpoint '{checkpoint_path}' is incompatible with the current model configuration."]

    if any(k.startswith(("shared.", "pool.")) for k in all_bad):
        lines.append(f"  • Backbone mismatch — checkpoint backbone differs from '{backbone}'.")
    if any(k.startswith("local_classifiers.") for k in all_bad):
        lines.append(f"  • Expert/hierarchy mismatch — checkpoint expert_type or hierarchy differs from current (expert_type='{expert_type}').")

    if missing:
        shown = sorted(missing)[:5]
        lines.append(f"  • Missing keys ({len(missing)}): {shown}" + (" ..." if len(missing) > 5 else ""))
    if unexpected:
        shown = sorted(unexpected)[:5]
        lines.append(f"  • Unexpected keys ({len(unexpected)}): {shown}" + (" ..." if len(unexpected) > 5 else ""))
    if shape_errors:
        lines.append(f"  • Shape mismatches ({len(shape_errors)}): {shape_errors[:3]}" + (" ..." if len(shape_errors) > 3 else ""))

    raise ValueError("\n".join(lines))


def set_seed(seed: int = 666) -> None:
    """
    Set the random seed across Python, NumPy, and PyTorch for reproducibility.

    Fixes seeds for ``random``, ``numpy``, ``torch`` (CPU and all GPUs) and
    enables ``cudnn.deterministic`` mode.

    Args:
        seed (int): Seed value to use. Defaults to 666.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _largest_connected_component(mask):

    """
    Boolean mask kept to only its largest 4-connected component, so a threshold
    mask that happens to catch a few disconnected specks of vignetting/noise
    elsewhere in the image doesn't get treated as part of the organism.

    Uses scipy.ndimage.label rather than a Python flood fill: this runs per
    image inside the training augmentation pipeline, and the pure-Python version
    was ~17x slower (0.60 vs 0.03 ms/mask), which dominated epoch time on the
    BCNN path where the DataLoader has no worker processes to hide it behind.
    Output is identical.
    """

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    labelled, n_components = ndimage.label(mask)   # 4-connectivity by default
    if n_components == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labelled.ravel())
    counts[0] = 0                                   # background label
    return labelled == counts.argmax()


def _dilate_mask(mask, dilate_px):

    """Grow a boolean mask by dilate_px on each side. 0 returns it unchanged."""

    if dilate_px <= 0 or not mask.any():
        return mask
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.MaxFilter(2 * dilate_px + 1))
    return np.array(mask_img) > 0


def _bbox_from_mask(mask, padding_frac=0.15):

    """(x0, y0, x1, y1) bounding box of a boolean mask, padded by padding_frac."""

    h, w = mask.shape
    if not mask.any():
        return (0, 0, w, h)
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    pad_x, pad_y = int((x1 - x0) * padding_frac), int((y1 - y0) * padding_frac)
    return (max(0, x0 - pad_x), max(0, y0 - pad_y), min(w, x1 + pad_x), min(h, y1 + pad_y))


def _organism_base_mask(image):

    """Undilated largest-component silhouette mask -- the one expensive step."""

    arr = np.array(image.convert('L'), dtype=float)
    mask = arr < (arr.mean() - 0.3 * arr.std())
    return _largest_connected_component(mask)


def get_organism_mask(image, dilate_px=4):

    """
    Boolean silhouette mask of the organism via brightness thresholding
    (for dark-organism-on-light-background microscopy images), restricted to
    its largest connected component and optionally dilated by dilate_px so
    callers get a small halo around the raw silhouette rather than a mask that
    hugs it exactly. The threshold is deliberately lenient (mean - 0.3*std,
    rather than a stricter cutoff) because real specimens often have faint,
    thin appendages (antennae, legs, spines) that are much lighter than the
    organism's body but still part of it -- a stricter threshold only catches
    the solid body and leaves those appendages exposed.

    Args:
        image (PIL.Image): Input image.
        dilate_px (int, optional): Pixels to grow the mask by on each side via a
            max filter. 0 disables dilation. Defaults to 4.

    Returns:
        np.ndarray: Boolean array, shape (H, W), True where the organism (plus
            halo) is.
    """

    return _dilate_mask(_organism_base_mask(image), dilate_px)


def get_organism_bbox(image, padding_frac=0.15):

    """
    Approximate the organism's bounding box from its silhouette mask (see
    get_organism_mask). Used to bias synthetic dirt/fiber placement toward the
    area surrounding the specimen, matching how real messy images tend to show
    hair/fiber debris clustering near the organism rather than scattered
    uniformly.

    Args:
        image (PIL.Image): Input image.
        padding_frac (float, optional): Fraction of the organism's extent to pad
            the bounding box by on each side. Defaults to 0.15.

    Returns:
        tuple[int, int, int, int]: (x0, y0, x1, y1) bounding box.
    """

    return _bbox_from_mask(_organism_base_mask(image), padding_frac)


def add_synthetic_dirt(image, n_specks=25, n_fibers=8, speck_darkness=(40, 110),
                        fiber_darkness=(50, 120), speck_alpha=(60, 120),
                        fiber_alpha=(55, 110), haze_strength=10,
                        fiber_near_organism_frac=0.75, exclusion_dilate_frac=0.08,
                        blur_radius=1.2, seed=None):

    """
    Overlay synthetic dirt/debris onto a clean image to mimic a messy background:
    small dark speckles (dust particles) scattered across the whole image, thin
    fiber/hair-like streaks mostly clustered near (but not on top of) the
    organism (fraction controlled by fiber_near_organism_frac), and a subtle
    low-frequency haze. The dirt (haze, speckles, fibers, all softly blurred so
    they read as out-of-focus, since background debris in real messy images is
    out of focus relative to the in-focus specimen) is built on a full copy of
    the image first, and the organism's original, untouched pixels are then
    pasted back on top via its silhouette mask (get_organism_mask, dilated by
    exclusion_dilate_frac * min(width, height) plus a feathered edge) -- this
    guarantees the organism itself is unaffected by the dirt regardless of any
    imprecision in the mask elsewhere (e.g. faint appendages the threshold
    doesn't fully catch), rather than relying on dirt never being drawn there
    in the first place. Parameters were calibrated against real messy-background
    images so background patch stats (mean brightness, mean gradient/texture)
    land close to the real distribution.

    Args:
        image (PIL.Image): Input image (any mode; converted internally).
        n_specks (int, optional): Number of dust speckles to draw. Defaults to 25.
        n_fibers (int, optional): Number of fiber/hair streaks to draw. Defaults to 8.
        speck_darkness (tuple[int, int], optional): Min/max darkness for speckles.
        fiber_darkness (tuple[int, int], optional): Min/max darkness for fibers.
        speck_alpha (tuple[int, int], optional): Min/max opacity for speckles.
        fiber_alpha (tuple[int, int], optional): Min/max opacity for fibers.
        haze_strength (int, optional): Max brightness reduction from the low-frequency haze.
        fiber_near_organism_frac (float, optional): Fraction of fibers placed near the
            organism (via get_organism_bbox) rather than uniformly at random.
        exclusion_dilate_frac (float, optional): Halo kept clear of dirt around the
            organism's silhouette, as a fraction of min(width, height) -- resolution-
            relative since raw crop sizes in this dataset vary widely. Defaults to 0.08.
        blur_radius (float, optional): Gaussian blur radius applied to the
            specks/fibers overlay, so dirt reads as out-of-focus. 0 disables it.
        seed (int, optional): Random seed for reproducibility. Defaults to None (fresh
            randomness each call, as wanted for training-time augmentation).

    Returns:
        PIL.Image: RGB image with synthetic dirt applied.
    """

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    img = image.convert('RGBA').copy()
    w, h = img.size
    # One silhouette pass shared by both uses -- the bbox and the exclusion mask
    # derive from the same base mask, so connected components runs once per image.
    base_mask = _organism_base_mask(image)
    bx0, by0, bx1, by1 = _bbox_from_mask(base_mask)
    exclusion_dilate_px = max(2, round(exclusion_dilate_frac * min(w, h)))
    exclusion_mask = _dilate_mask(base_mask, exclusion_dilate_px)

    def sample_outside_exclusion(x_range, y_range, max_tries=10):
        x, y = rng.uniform(*x_range), rng.uniform(*y_range)
        for _ in range(max_tries):
            if not exclusion_mask[min(int(y), h - 1), min(int(x), w - 1)]:
                break
            x, y = rng.uniform(*x_range), rng.uniform(*y_range)
        return x, y

    # Low-frequency haze
    noise = np_rng.rand(h, w).astype(np.float32)
    haze = Image.fromarray((noise * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=w / 6))
    haze_arr = np.array(haze, dtype=np.float32) / 255.0
    arr = np.array(img, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] -= haze_strength * haze_arr
    arr = np.clip(arr, 0, 255)
    dirty = Image.fromarray(arr.astype(np.uint8), 'RGBA')

    # Speckles (biased away from the organism, but not strictly forbidden -- the
    # final paste-back step is what actually guarantees the organism stays clean)
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(n_specks):
        x, y = sample_outside_exclusion((0, w), (0, h))
        r = rng.uniform(0.3, 1.2)
        darkness = rng.randint(*speck_darkness)
        alpha = rng.randint(*speck_alpha)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(darkness, darkness, darkness, alpha))
    # fibers (mostly clustered around, but not on top of, the organism)
    for _ in range(n_fibers):
        if rng.random() < fiber_near_organism_frac:
            x1, y1 = sample_outside_exclusion((bx0, bx1), (by0, by1))
        else:
            x1, y1 = sample_outside_exclusion((0, w), (0, h))
        length = rng.uniform(8, 35)
        angle = rng.uniform(0, 2 * np.pi)
        points = [(x1, y1)]
        for _ in range(rng.randint(2, 4)):
            angle += rng.uniform(-0.5, 0.5)
            nx = points[-1][0] + length / 3 * np.cos(angle)
            ny = points[-1][1] + length / 3 * np.sin(angle)
            points.append((nx, ny))
        darkness = rng.randint(*fiber_darkness)
        alpha = rng.randint(*fiber_alpha)
        draw.line(points, fill=(darkness, darkness, darkness, alpha), width=1)

    if blur_radius > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    dirty = Image.alpha_composite(dirty, overlay)

    # Paste the organism back from the original, untouched image, via a lightly
    # feathered version of its mask -- guarantees pixel-level fidelity within the
    # organism regardless of any haze/blur bleed or mask imprecision at its edge.
    feather_radius = max(1, exclusion_dilate_px // 2)
    paste_alpha = Image.fromarray((exclusion_mask * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=feather_radius)
    )
    return Image.composite(image.convert('RGB'), dirty.convert('RGB'), paste_alpha)


class RandomBackgroundNoise:

    """
    torchvision-Compose-compatible transform that randomly applies add_synthetic_dirt
    to an image, for use as a training-time augmentation step. Unlike calling
    add_synthetic_dirt directly with a fixed seed (for reproducible one-off
    replication), this uses fresh randomness on every call, matching the behavior
    of torchvision's own RandomXxx transforms.

    Note: add_synthetic_dirt internally composites through RGB and returns an RGB
    image. Since ImageDataset/HierImageDataset.__getitem__ apply image_transforms to
    a grayscale ('L') image and then unconditionally do image.repeat(3, 1, 1) to get
    3 channels, this transform converts the output back to the input's original mode
    before returning it, so it stays a drop-in single-channel step in that pipeline.

    Args:
        p (float, optional): Probability of applying the noise on each call. Defaults to 0.5.
        **kwargs: Passed through to add_synthetic_dirt (n_specks, n_fibers, etc.).
    """

    def __init__(self, p: float = 0.5, **kwargs):
        self.p = p
        self.kwargs = kwargs

    def __call__(self, image):
        if random.random() < self.p:
            return add_synthetic_dirt(image, seed=None, **self.kwargs).convert(image.mode)
        return image


class Visualize:
    """
    Reads saved artefacts from a model_dir (training_info.json, eval_results.json,
    predictions.npz) and produces publication-ready plots.

    Requires trainer.predict(save=True) to have been run so that per-level arrays
    are present in predictions.npz.
    """

    # Colour palette shared across methods
    _TRAIN_COL = "#4C72B0"   # blue  — train
    _VAL_COL   = "#DD8452"   # orange — validation
    _METRIC_COLS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]  # F1 / Acc / Prec / Rec

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)

        train_path = self.model_dir / "training_info.json"
        eval_path  = self.model_dir / "eval_results.json"
        pred_path  = self.model_dir / "predictions.npz"

        self.training_info = json.loads(train_path.read_text()) if train_path.exists() else None
        self.eval_results  = json.loads(eval_path.read_text())  if eval_path.exists()  else None
        self.pred_data     = dict(np.load(pred_path))           if pred_path.exists()  else None

        if self.training_info is None:
            print(f"  [Visualize] training_info.json not found in '{model_dir}' — plot_train unavailable")
        if self.eval_results is None or self.pred_data is None:
            print(f"  [Visualize] eval_results.json / predictions.npz not found — plot_pred unavailable")

    # ------------------------------------------------------------------ #
    #  Training curves                                                     #
    # ------------------------------------------------------------------ #

    def plot_train(self, max_epochs: int | None = None) -> None:
        """
        Three side-by-side line charts (Loss / Accuracy / F1),
        one line for train (blue) and one for validation (orange).
        Best validation epoch is marked with a vertical dashed line
        and a diamond marker on the validation curve.
        Saves train_metrics.png to model_dir.

        Args:
            max_epochs: If set, only the first max_epochs epochs are shown.
                        Defaults to None (show all epochs).
        """
        if self.training_info is None:
            print("training_info.json not found — skipping plot_train.")
            return

        m          = self.training_info["training_metrics"]
        best_epoch = self.training_info["results"]["best_epoch"]
        n_epochs   = len(m["train_loss"])
        if max_epochs is not None:
            n_epochs = min(n_epochs, max_epochs)
        epochs     = np.arange(1, n_epochs + 1)

        metric_cfgs = [
            ("Loss",     "train_loss", "valid_loss"),
            ("Accuracy", "train_acc",  "valid_acc"),
            ("F1",       "train_f1",   "valid_f1"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Training Metrics per Epoch", fontsize=14, fontweight="bold")

        for ax, (title, train_key, val_key) in zip(axes, metric_cfgs):
            train_vals = np.array(m[train_key])[:n_epochs]
            val_vals   = np.array(m[val_key])[:n_epochs]

            ax.plot(epochs, train_vals, color=self._TRAIN_COL, linewidth=2,
                    marker="o", markersize=4, label="Train")
            ax.plot(epochs, val_vals,   color=self._VAL_COL,   linewidth=2,
                    marker="o", markersize=4, label="Validation")

            # Mark the best validation epoch
            if best_epoch is not None and 1 <= best_epoch <= n_epochs:
                bi = best_epoch - 1
                ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.5,
                           label=f"Best ep {best_epoch}", zorder=3)
                ax.scatter([best_epoch], [val_vals[bi]], color="black",
                           marker="D", s=60, zorder=5)

            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Epoch")
            ax.set_xticks(epochs)
            ax.set_xticklabels(epochs, fontsize=7,
                               rotation=45 if n_epochs > 20 else 0)
            ax.legend(fontsize=9)
            if title != "Loss":
                ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = self.model_dir / "train_metrics.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.show()

    # ------------------------------------------------------------------ #
    #  Prediction results                                                  #
    # ------------------------------------------------------------------ #

    def plot_pred(self) -> None:
        """
        For each hierarchy level:
          1. Confusion matrix — raw counts (left) and normalised / recall (right).
             An "Other" column captures predictions that landed at the wrong depth.
          2. Per-class bar chart — Acc / F1 / Precision / Recall, n annotated.

        Also calls plot_level_comparison() for an overall cross-level summary.
        Saves PNG files to model_dir.
        """
        if self.eval_results is None or self.pred_data is None:
            print("eval_results.json or predictions.npz not found — skipping plot_pred.")
            return

        level_results = self.eval_results["level_results"]

        for depth_str in sorted(level_results, key=int):
            depth     = int(depth_str)
            per_class = level_results[depth_str]["per_class"]

            classes     = sorted(per_class.items(), key=lambda x: x[1]["node_id"])
            class_names = [c[0] for c in classes]
            node_ids    = [c[1]["node_id"] for c in classes]
            nid_to_idx  = {nid: i for i, nid in enumerate(node_ids)}
            n           = len(class_names)

            preds_key = f"level_{depth}_preds"
            trues_key = f"level_{depth}_trues"

            if preds_key not in self.pred_data:
                print(f"  Level {depth} arrays missing — re-run trainer.predict(save=True).")
                continue

            preds = self.pred_data[preds_key].astype(int)
            trues = self.pred_data[trues_key].astype(int)

            # Build (n_true × n_pred+1) confusion matrix; last col = "Other"
            cm          = np.zeros((n, n + 1), dtype=int)
            col_labels  = class_names + ["Other"]
            for t, p in zip(trues, preds):
                ri = nid_to_idx.get(t, -1)
                ci = nid_to_idx.get(p, n)   # unknown pred → "Other"
                if ri >= 0:
                    cm[ri, ci] += 1

            # Drop "Other" column if empty
            if cm[:, n].sum() == 0:
                cm         = cm[:, :n]
                col_labels = class_names
            n_cols = cm.shape[1]

            # --- Confusion matrices ---
            fig, axes = plt.subplots(1, 2, figsize=(max(10, n_cols * 1.6 + 3), max(5, n * 1.2)))
            fig.suptitle(f"Level {depth} — Confusion Matrix", fontsize=13, fontweight="bold")

            for ax, data, title, fmt, vmax in [
                (axes[0], cm,                          "Counts",                    "d",    None),
                (axes[1], cm / cm.sum(1, keepdims=True).clip(1), "Normalised (recall / row)", ".2f", 1.0),
            ]:
                im = ax.imshow(data, cmap="Blues", aspect="auto",
                               vmin=0, vmax=vmax if vmax else data.max())
                ax.set_xticks(range(n_cols))
                ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
                ax.set_yticks(range(n))
                ax.set_yticklabels(class_names, fontsize=9)
                ax.invert_yaxis()
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title(title)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                thresh = data.max() / 2.
                for i in range(n):
                    for j in range(n_cols):
                        val = data[i, j]
                        ax.text(j, i, format(val, fmt), ha="center", va="center",
                                fontsize=8, color="white" if val > thresh else "black")

            plt.tight_layout()
            out = self.model_dir / f"confusion_matrix_level{depth}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved → {out}")
            plt.show()

            # --- Per-class metric bar chart ---
            self._plot_class_metrics(depth, per_class)

        # --- Cross-level overview ---
        self.plot_level_comparison()

    def _plot_class_metrics(self, depth: int, per_class: dict) -> None:
        """Grouped bar chart: Acc / F1 / Precision / Recall per class at one level."""
        classes     = sorted(per_class.items(), key=lambda x: x[1]["node_id"])
        class_names = [c[0] for c in classes]
        n           = len(class_names)

        met_cfgs = [
            ("Accuracy",  "accuracy",  self._METRIC_COLS[0]),
            ("F1",        "f1",        self._METRIC_COLS[1]),
            ("Precision", "precision", self._METRIC_COLS[2]),
            ("Recall",    "recall",    self._METRIC_COLS[3]),
        ]
        x      = np.arange(n)
        w      = 0.18
        offset = [-1.5, -0.5, 0.5, 1.5]

        fig, ax = plt.subplots(figsize=(max(10, n * 1.6), 5))
        ax.set_title(f"Level {depth} — Per-class Metrics", fontsize=12, fontweight="bold")

        for (label, key, color), off in zip(met_cfgs, offset):
            vals = [per_class[cls][key] for cls in class_names]
            ax.bar(x + off * w, vals, w, label=label, color=color, alpha=0.85)

        # Annotate sample counts below x-axis
        n_samples = [per_class[cls]["n_samples"] for cls in class_names]
        for i, ns in enumerate(n_samples):
            ax.annotate(f"n={ns}", xy=(i, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -28), textcoords="offset points",
                        ha="center", fontsize=8, color="dimgray")

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=9, loc="lower right")

        plt.tight_layout()
        out = self.model_dir / f"class_metrics_level{depth}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.show()

    def plot_level_comparison(self) -> None:
        """
        Grouped bar chart comparing overall Acc / F1 / Precision / Recall
        across all hierarchy levels, with value labels on each bar.
        Saves level_comparison.png to model_dir.
        """
        if self.eval_results is None:
            print("eval_results.json not found — skipping plot_level_comparison.")
            return

        level_results = self.eval_results["level_results"]
        levels        = sorted(level_results, key=int)
        met_cfgs      = [
            ("Accuracy",  "accuracy",  self._METRIC_COLS[0]),
            ("F1",        "f1",        self._METRIC_COLS[1]),
            ("Precision", "precision", self._METRIC_COLS[2]),
            ("Recall",    "recall",    self._METRIC_COLS[3]),
        ]
        x      = np.arange(len(levels))
        w      = 0.18
        offset = [-1.5, -0.5, 0.5, 1.5]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Overall Metrics across Hierarchy Levels", fontsize=12, fontweight="bold")

        for (label, key, color), off in zip(met_cfgs, offset):
            vals = [level_results[l]["overall"][key] for l in levels]
            bars = ax.bar(x + off * w, vals, w, label=label, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Level {l}" for l in levels], fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score")
        ax.legend(fontsize=9)

        plt.tight_layout()
        out = self.model_dir / "level_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.show()

    def plot_class_size_vs_accuracy(self) -> None:
        """
        Scatter plot: class sample size vs accuracy for every class across all levels.
        Useful for diagnosing whether low accuracy is driven by data imbalance.
        Each level is a different colour.
        """
        if self.eval_results is None:
            print("eval_results.json not found.")
            return

        level_results = self.eval_results["level_results"]
        level_colors  = ["#4C72B0", "#55A868", "#C44E52"]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title("Class Size vs Accuracy across Hierarchy Levels",
                     fontsize=12, fontweight="bold")

        for (depth_str, color) in zip(sorted(level_results, key=int), level_colors):
            depth     = int(depth_str)
            per_class = level_results[depth_str]["per_class"]
            for cls_name, metrics in per_class.items():
                ax.scatter(metrics["n_samples"], metrics["accuracy"],
                           color=color, alpha=0.8, s=80, zorder=3)
                ax.annotate(cls_name, (metrics["n_samples"], metrics["accuracy"]),
                            textcoords="offset points", xytext=(5, 3), fontsize=7)

        # Legend patches
        handles = [mpatches.Patch(color=c, label=f"Level {d}")
                   for c, d in zip(level_colors, sorted(level_results, key=int))]
        ax.legend(handles=handles, fontsize=9)
        ax.set_xlabel("Number of test samples")
        ax.set_ylabel("Accuracy")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = self.model_dir / "size_vs_accuracy.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.show()
