"""
train_swin_cnn_phyto.py -- train HierRouteNet on the phytoplankton binary task
(phytoplankton vs not phytoplankton).

Mirrors runs/train_swin_cnn_mnr.py: a Swin-T backbone with CNN experts, trained
via Trainer, best checkpoint + metrics saved under
training_result/phytoplankton/. The only structural difference is the taxonomy
-- one non-root level, two leaves -- which is entirely handled by
phytoplankton/scripts/data_setup.py.

Run (from anywhere):
    python phytoplankton/scripts/train_swin_cnn_phyto.py

Config: edit the constants below, or set the matching env vars --
    PHYTO_NOISE_AUG=1        enable RandomBackgroundNoise augmentation (default: OFF)
    MODEL_BACKBONE=swin_s    swin_t (default) | swin_s | efficientnet_b0
    NUM_WORKERS=8
    MAX_EPOCHS=40
    DEVICE=cuda:1
    PHYTOPLANKTON_DATA_DIR=/data/Hier-Phytoplankton/Processed_data
    PHYTO_RESULTS_DIR=...    where run folders are written
                            (default: <repo>/training_result/phytoplankton)
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import torch
from dotenv import load_dotenv
from torchvision import transforms

# --- make hierroute + this folder's modules importable ------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
load_dotenv(REPO_ROOT / ".env")

from hierroute import HierRouteNet, Trainer
from hierroute.constants import SEED, RESOLUTION
from hierroute.extra_functions import set_seed, RandomBackgroundNoise

from data_setup import build_dataloaders

# --- configuration -----------------------------------------------------------
# RandomBackgroundNoise = synthetic dirt/fiber overlay (hierroute.extra_functions).
# Default OFF for phytoplankton -- enable with PHYTO_NOISE_AUG=1 once it has been
# sanity-checked against real messy phytoplankton captures (its speck/fiber sizes
# and the organism-silhouette mask were calibrated on the zooplankton dataset).
NOISE_AUGMENTATION = os.environ.get("PHYTO_NOISE_AUG", "0") == "1"
NOISE_PROB = 0.8

BACKBONE = os.environ.get("MODEL_BACKBONE", "swin_t")
EXPERT_TYPE = "cnn"
SUPPORTED_BACKBONES = ("swin_t", "swin_s", "efficientnet_b0")
if BACKBONE not in SUPPORTED_BACKBONES:
    raise ValueError(f"MODEL_BACKBONE={BACKBONE!r} not supported; choose one of {SUPPORTED_BACKBONES}")

BATCH_SIZE = 64
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 8))

# Every run folder is written under here. Pinned to the phytoplankton project's
# results directory (not the shared training_result/mnr/ the reference uses).
RESULTS_DIR = Path(
    os.environ.get("PHYTO_RESULTS_DIR", REPO_ROOT / "training_result" / "phytoplankton")
)

LEARNING_RATE = 3e-4
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 40))
PATIENCE = 10
DELTA = 0.001

DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

print(f"Backbone: {BACKBONE} | expert: {EXPERT_TYPE} | background noise: {NOISE_AUGMENTATION} | device: {DEVICE}")

set_seed(SEED)

# --- data --------------------------------------------------------------------
# RandomBackgroundNoise runs first, on the raw native-resolution grayscale crop;
# flips are label-preserving for plankton; Resize + ToTensor close the pipeline.
# Matches runs/train_swin_cnn_mnr.py.
#
# NOTE: as in that script, build_dataloaders applies this one pipeline to all
# three splits (HierImageDataset.create_dataloaders has no separate-train-pipeline
# path). The extra transforms here are label-preserving, but for a clean held-out
# number rebuild the test split with data_setup.EVAL_TRANSFORMS.
train_transforms_list = []
if NOISE_AUGMENTATION:
    train_transforms_list.append(RandomBackgroundNoise(p=NOISE_PROB))
train_transforms_list += [
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
]
train_transforms = transforms.Compose(train_transforms_list)

hier_dataset, train_loader, val_loader, test_loader = build_dataloaders(
    batch_size=BATCH_SIZE,
    train_transforms=train_transforms,
    balanced=True,
    num_workers=NUM_WORKERS,
)

# --- model ------------------------------------------------------------------
model = HierRouteNet(
    hierarchy=hier_dataset.hierarchy,
    label_to_id=hier_dataset.label_to_ids,
    backbone=BACKBONE,
    expert_type=EXPERT_TYPE,
    freeze_backbone=False,
)

# --- train + evaluate ------------------------------------------------------
run_date = date.today().strftime("%m-%d")
variant = "noise" if NOISE_AUGMENTATION else "baseline"
model_dir = str(RESULTS_DIR / f"{BACKBONE}_{EXPERT_TYPE}_{variant}_{run_date}")
print(f"Results dir: {model_dir}")

trainer = Trainer(
    learning_rate=LEARNING_RATE,
    max_epochs=MAX_EPOCHS,
    device=DEVICE,
    print_every=1,
    model_dir=model_dir,
)
trainer.fit(model, train_loader, val_loader, scheduler=True, patience=PATIENCE, delta=DELTA)

trainer.predict(model, test_loader, save=True)
