"""
train.py — Train a HierRouteNet from a configuration dict (see :mod:`hierroute.config`).

The run directory gets best_model.pt, model_config.json (so ``hierroute evaluate --weights``
can load it), train_config.yaml (the resolved configuration), training_info.json, and the
held-out test split's eval_results.json / predictions.npz, scored with the best checkpoint.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import torch
import yaml
from torchvision import transforms

from .config import load_hierarchy
from .data_setup import HierImageDataset, ImageDataset
from .extra_functions import RandomBackgroundNoise, set_seed
from .model import HierRouteNet
from .pretrained import CHECKPOINT_FILE, ModelConfig
from .trainer import Trainer

BACKBONES = ("swin_t", "swin_s", "efficientnet_b0")
EXPERTS = ("cnn", "mlp", "linear")


def run_name(config: dict) -> str:
    if config["run_name"]:
        return config["run_name"]
    variant = "noise" if config["augment"]["background_noise"] else "baseline"
    return f"{config['model']['backbone']}_{config['model']['expert']}_{variant}_{date.today().strftime('%m-%d')}"


def _validate(config: dict) -> None:
    if not config["data"]["data_dir"]:
        raise ValueError("data.data_dir is required (set it in the config file or pass --data-dir)")
    if not Path(config["data"]["data_dir"]).is_dir():
        raise FileNotFoundError(f"data.data_dir not found: {config['data']['data_dir']}")
    if config["model"]["backbone"] not in BACKBONES:
        raise ValueError(f"model.backbone must be one of {BACKBONES}")
    if config["model"]["expert"] not in EXPERTS:
        raise ValueError(f"model.expert must be one of {EXPERTS}")
    split = config["data"]["split"]
    if len(split) != 3 or abs(sum(split) - 1) > 1e-6:
        raise ValueError(f"data.split must be three proportions summing to 1, got {split}")


def build_train_transforms(config: dict) -> transforms.Compose:
    augment, resolution = config["augment"], config["data"]["resolution"]
    steps = []
    if augment["background_noise"]:
        # Applied first, at native resolution: the synthetic dirt's speck/fiber sizes were
        # calibrated against real messy-background images at their original resolution.
        steps.append(RandomBackgroundNoise(p=augment["noise_p"]))
    if augment["hflip"]:
        steps.append(transforms.RandomHorizontalFlip())
    if augment["vflip"]:
        steps.append(transforms.RandomVerticalFlip())
    steps += [transforms.Resize((resolution, resolution)), transforms.ToTensor()]
    return transforms.Compose(steps)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def train(config: dict) -> Path:
    """Train according to ``config`` and return the run directory."""
    _validate(config)
    data, model_cfg, train_cfg = config["data"], config["model"], config["train"]
    spec = load_hierarchy(config["hierarchy"])
    seed = train_cfg["seed"]

    exclude_log = data["exclude_log"]
    if exclude_log and not os.path.isabs(exclude_log):
        exclude_log = os.path.join(data["data_dir"], exclude_log)

    print(f"Hierarchy: {spec.name} | backbone: {model_cfg['backbone']} | expert: {model_cfg['expert']} "
          f"| background noise: {config['augment']['background_noise']}")
    set_seed(seed)

    dataset = ImageDataset(
        data_directory=data["data_dir"],
        data_subdirectories=list(data["subdirectories"] or []),
        class_names=spec.training_classes,
        class_folder_map=spec.folder_map,
        max_class_size=data["max_class_size"],
        image_resolution=data["resolution"],
        format_file=data["format"],
        seed=seed,
        exclude_log_path=exclude_log,
        uncapped_subdirectories=list(data["uncapped_subdirectories"] or []),
    )
    hier_dataset = HierImageDataset(
        base_dataset=dataset,
        adjacency_graph=spec.tree,
        levels=spec.build()[0].max_depth,
        leaves_only=True,
    )
    hier_dataset.print_dataset_details()
    if len(hier_dataset) == 0:
        raise ValueError(f"No training images found under {data['data_dir']} for hierarchy {spec.name}")

    hier_dataset.append_image_transforms(build_train_transforms(config), replace=True)
    train_split, val_split, test_split = hier_dataset.split_train_test_val(*data["split"])
    train_loader, val_loader, test_loader = hier_dataset.create_dataloaders(
        batch_size=train_cfg["batch_size"],
        train_indices=train_split,
        val_indices=val_split,
        test_indices=test_split,
        balanced=data["balanced_sampling"],
        num_workers=train_cfg["num_workers"],
    )

    model = HierRouteNet(
        hierarchy=hier_dataset.hierarchy,
        label_to_id=hier_dataset.label_to_ids,
        backbone=model_cfg["backbone"],
        expert_type=model_cfg["expert"],
        loss_type=model_cfg["loss"],
        focal_gamma=model_cfg["focal_gamma"],
        freeze_backbone=model_cfg["freeze_backbone"],
        pretrained=model_cfg["pretrained_backbone"],
    )

    device = resolve_device(train_cfg["device"])
    trainer = Trainer(
        learning_rate=train_cfg["lr"],
        max_epochs=train_cfg["max_epochs"],
        gradient_clip_val=train_cfg["gradient_clip"],
        device=device,
        print_every=1,
        model_dir=str(Path(config["out_dir"]) / run_name(config)),
    )
    trainer.fit(model, train_loader, val_loader, scheduler=train_cfg["scheduler"] == "cosine",
                patience=train_cfg["patience"], delta=train_cfg["delta"])

    run_dir = Path(trainer.model_dir)
    model_config = ModelConfig(name=run_dir.name, backbone=model_cfg["backbone"], expert_type=model_cfg["expert"],
                               resolution=data["resolution"], hierarchy=spec, loss_type=model_cfg["loss"])
    model_config.save(run_dir)
    (run_dir / "train_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    # Score the held-out test split with the best checkpoint, not the last epoch's weights.
    model.load_state_dict(torch.load(run_dir / CHECKPOINT_FILE, map_location=device, weights_only=True))
    trainer.predict(model, test_loader, save=True)
    print(f"\nRun directory: {run_dir}\n  evaluate it with: hierroute evaluate <images> --weights {run_dir}")
    return run_dir
