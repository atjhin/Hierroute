"""
pretrained.py — Released model weights and loading trained models.

Released weights are GitHub Release assets, downloaded on first use into
``$HIERROUTE_CACHE_DIR`` (default ``~/.cache/hierroute``) and verified by sha256.

Every training run (``hierroute train``) writes ``model_config.json`` next to its
``best_model.pt``, so a self-trained run directory loads exactly like a released model::

    model, config = load_model("noise")                    # released weights
    model, config = load_model(weights="training_result/swin_t_cnn_noise_11-08")
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import HierarchySpec, load_hierarchy
from .extra_functions import validate_checkpoint
from .model import HierRouteNet

RELEASE_URL = "https://github.com/atjhin/Hierroute/releases/download/models-v1"
MODEL_CONFIG_FILE = "model_config.json"
CHECKPOINT_FILE = "best_model.pt"
DEFAULT_MODEL = "noise"


def tree_fingerprint(tree: dict[str, list[str]]) -> str:
    """Short hash of a hierarchy tree, including key order (which defines node IDs)."""
    return hashlib.sha256(json.dumps(tree).encode()).hexdigest()[:16]


# name -> release asset + what the weights were trained with. hierarchy_fingerprint pins the
# bundled hierarchy's tree (and its key order) to the one these weights were trained on.
PRETRAINED: dict[str, dict] = {
    "noise": {
        "file": "swin_t_cnn_noise_10-20.pt",
        "sha256": "a556600c664ad039b2fc8ca49475df04bc9ea3938617b6fa1e27acb22309533c",
        "backbone": "swin_t",
        "expert_type": "cnn",
        "resolution": 64,
        "hierarchy": "mnr_17leaf",
        "hierarchy_fingerprint": "ceed96146bbec71c",
        "description": "Swin-T + CNN experts, trained with synthetic background noise "
                       "(recommended for messy or dirty backgrounds)",
    },
    "baseline": {
        "file": "swin_t_cnn_baseline_10-20.pt",
        "sha256": "3bdf0d9ef2cd3240a1d3e349898264dac043739867c4cffe201a684a267324fa",
        "backbone": "swin_t",
        "expert_type": "cnn",
        "resolution": 64,
        "hierarchy": "mnr_17leaf",
        "hierarchy_fingerprint": "ceed96146bbec71c",
        "description": "Swin-T + CNN experts, trained without background noise (clean backgrounds)",
    },
}


@dataclass
class ModelConfig:
    """Everything needed to rebuild a trained HierRouteNet around its checkpoint."""
    name: str
    backbone: str
    expert_type: str
    resolution: int
    hierarchy: HierarchySpec
    loss_type: str = "bce"

    def to_dict(self) -> dict:
        return {
            "format_version": 1,
            "name": self.name,
            "backbone": self.backbone,
            "expert_type": self.expert_type,
            "resolution": self.resolution,
            "loss_type": self.loss_type,
            "hierarchy": self.hierarchy.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(
            name=data["name"],
            backbone=data["backbone"],
            expert_type=data["expert_type"],
            resolution=int(data["resolution"]),
            hierarchy=HierarchySpec.from_dict(data["hierarchy"]),
            loss_type=data.get("loss_type", "bce"),
        )

    def save(self, run_dir: str | Path) -> Path:
        path = Path(run_dir) / MODEL_CONFIG_FILE
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def cache_dir() -> Path:
    """Directory released weights are downloaded into."""
    if os.environ.get("HIERROUTE_CACHE_DIR"):
        return Path(os.environ["HIERROUTE_CACHE_DIR"]).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "hierroute"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registry_entry(name: str) -> dict:
    if name not in PRETRAINED:
        raise ValueError(f"Unknown model '{name}'. Released models: {sorted(PRETRAINED)}")
    return PRETRAINED[name]


def download(name: str, force: bool = False, progress: bool = True) -> Path:
    """Download a released checkpoint into :func:`cache_dir` (if needed) and return its path."""
    entry = _registry_entry(name)
    target = cache_dir() / entry["file"]
    if target.exists() and not force:
        if _sha256(target) == entry["sha256"]:
            return target
        print(f"Cached {target} failed its checksum -- downloading again.")

    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RELEASE_URL}/{entry['file']}"
    print(f"Downloading '{name}' weights from {url}")
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.hub.download_url_to_file(url, str(tmp_path), progress=progress)
        actual = _sha256(tmp_path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {url}: expected {entry['sha256']}, got {actual}")
        shutil.move(str(tmp_path), target)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target


def _pretrained_config(name: str) -> ModelConfig:
    entry = _registry_entry(name)
    spec = load_hierarchy(entry["hierarchy"])
    if tree_fingerprint(spec.tree) != entry["hierarchy_fingerprint"]:
        raise RuntimeError(
            f"The bundled '{entry['hierarchy']}' hierarchy no longer matches the one the '{name}' "
            "weights were trained with (nodes added, removed or reordered)."
        )
    return ModelConfig(name=name, backbone=entry["backbone"], expert_type=entry["expert_type"],
                       resolution=entry["resolution"], hierarchy=spec)


def _legacy_config(run_dir: Path) -> ModelConfig:
    """Run directories from before model_config.json existed: the 17-leaf MNR hierarchy with
    CNN experts at 64 px, backbone read from the directory name prefix."""
    name = run_dir.name
    backbone = next((b for b in ("efficientnet_b0", "swin_s", "swin_t") if name.lower().startswith(b)), "swin_t")
    warnings.warn(
        f"{run_dir} has no {MODEL_CONFIG_FILE}; assuming hierarchy=mnr_17leaf, backbone={backbone}, "
        "expert=cnn, resolution=64.",
        stacklevel=3,
    )
    return ModelConfig(name=name, backbone=backbone, expert_type="cnn", resolution=64,
                       hierarchy=load_hierarchy("mnr_17leaf"))


def resolve_weights(model: str | None = None, weights: str | Path | None = None) -> tuple[Path, ModelConfig]:
    """
    Locate a checkpoint and its config.

    Args:
        model:   Name of a released model (see :data:`PRETRAINED`). Used when ``weights`` is None.
        weights: A run directory (containing best_model.pt and, normally, model_config.json)
                 or a ``.pt`` file inside one.

    Returns:
        (checkpoint_path, ModelConfig)
    """
    if weights is None:
        name = model or DEFAULT_MODEL
        return download(name), _pretrained_config(name)

    weights = Path(weights)
    if weights.is_dir():
        run_dir, checkpoint = weights, weights / CHECKPOINT_FILE
        if not checkpoint.exists():
            candidates = sorted(weights.glob("*.pt"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"Expected {CHECKPOINT_FILE} (or exactly one .pt file) in {weights}")
            checkpoint = candidates[0]
    elif weights.is_file():
        run_dir, checkpoint = weights.parent, weights
    else:
        raise FileNotFoundError(f"Weights not found: {weights}")

    config_path = run_dir / MODEL_CONFIG_FILE
    config = ModelConfig.from_dict(json.loads(config_path.read_text())) if config_path.exists() \
        else _legacy_config(run_dir)
    return checkpoint, config


def build_model(config: ModelConfig, pretrained_backbone: bool = False) -> HierRouteNet:
    """An untrained HierRouteNet matching ``config``."""
    hierarchy, label_to_id = config.hierarchy.build()
    return HierRouteNet(
        hierarchy=hierarchy,
        label_to_id=label_to_id,
        backbone=config.backbone,
        expert_type=config.expert_type,
        loss_type=config.loss_type,
        pretrained=pretrained_backbone,
    )


def load_model(model: str | None = None, weights: str | Path | None = None,
               device: str | torch.device = "cpu") -> tuple[HierRouteNet, ModelConfig]:
    """
    Load a trained HierRouteNet in eval mode.

    Args:
        model:   Released model name (default ``"noise"``), used when ``weights`` is None.
        weights: Run directory or ``.pt`` file of a model trained with ``hierroute train``.
        device:  Device to move the model to.

    Returns:
        (model, ModelConfig)
    """
    checkpoint, config = resolve_weights(model, weights)
    net = build_model(config)
    state = validate_checkpoint(checkpoint, net, config.backbone, config.expert_type)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net, config
