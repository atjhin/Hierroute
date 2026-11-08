"""
Dataset / DataLoader builders shared by runs/hiernet_mnr.ipynb,
runs/confidence_threshold_experiment.ipynb and standalone scripts, so
"evaluate on the held-out test split" and "evaluate on Messy_background" use one
identical, seeded construction.

    import sys; sys.path.append('scripts')              # from runs/
    from eval_loaders import build_test_loader, build_messy_loader

    test_ds,  test_loader  = build_test_loader()        # held-out Hier-Zooplankton split
    messy_ds, messy_loader = build_messy_loader()       # full Messy_background set

Both wrap the HIERNET 17-leaf taxonomy (hiernet_adjacency_graph), levels=3,
leaves_only=True, and a clean Resize+ToTensor pipeline -- NOT the training
augmentation (flips / RandomBackgroundNoise / rotation) -- so the two result sets
stay directly comparable, and every rerun is deterministic.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from hierroute.constants import (
    HIERNET_CLASSES,
    HIERNET_CLASS_FOLDER_MAP,
    hiernet_adjacency_graph,
    SEED,
    MAX_CLASS_SIZE,
    RESOLUTION,
)
from hierroute.data_setup import ImageDataset, HierImageDataset
from hierroute.extra_functions import set_seed

BATCH_SIZE = 64
LEVELS = 3
SPLIT = (0.7, 0.1, 0.2)  # train / val / test -- matches runs/train_swin_cnn_mnr.py

DATA_DIRECTORY = os.environ["ZOOPLANKTON_DATA_DIR"]
DATA_SUBDIRECTORIES = ["IssacData"]
MESSY_BACKGROUND_DIR = os.path.join(DATA_DIRECTORY, "Messy_background")
# Images physically relocated into Messy_background (and the clean images later
# re-dirtied to replace them) are logged here; passing this to the training
# ImageDataset keeps the held-out test split free of Messy_background leakage.
# See hierroute.data_setup.load_excluded_filenames.
EXCLUDE_LOG_PATH = os.path.join(MESSY_BACKGROUND_DIR, "moved_images", "log.md")

# Clean evaluation pipeline -- exactly what runs/evaluate_hierroute_messy_background.py
# and hiernet_mnr.ipynb's evaluate step feed the model.
EVAL_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor(),
    ]
)


def _hier_dataset(base_dataset: ImageDataset, image_transforms) -> HierImageDataset:
    return HierImageDataset(
        base_dataset=base_dataset,
        adjacency_graph=hiernet_adjacency_graph,
        levels=LEVELS,
        leaves_only=True,
        image_transforms=image_transforms,
    )


def _eval_loader(dataset, batch_size: int, num_workers: int, collate_fn) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def build_hier_dataset(
    image_transforms=None,
    max_class_size: int = MAX_CLASS_SIZE,
    print_details: bool = True,
) -> HierImageDataset:
    """
    The main Hier-Zooplankton dataset used for train/val (data_subdirectories=['IssacData'],
    Messy_background images excluded) -- identical construction to runs/train_swin_cnn_mnr.py.

    image_transforms=None leaves the pipeline unset (train_variant fills it in with the
    training augmentation); pass EVAL_TRANSFORMS for a clean Resize+ToTensor pipeline.
    """
    set_seed(SEED)
    base = ImageDataset(
        data_directory=DATA_DIRECTORY,
        data_subdirectories=DATA_SUBDIRECTORIES,
        class_names=HIERNET_CLASSES,
        class_folder_map=HIERNET_CLASS_FOLDER_MAP,
        max_class_size=max_class_size,
        image_resolution=RESOLUTION,
        format_file=".tif",
        seed=SEED,
        exclude_log_path=EXCLUDE_LOG_PATH,
    )
    ds = _hier_dataset(base, image_transforms)
    if print_details:
        ds.print_dataset_details()
    return ds


def train_val_test_split(hier_dataset: HierImageDataset, split=SPLIT):
    """
    Seeded 0.7/0.1/0.2 index split -> (train_idx, val_idx, test_idx).

    Deterministic on SEED (sklearn train_test_split(random_state=seed)), so a fresh
    dataset instance reproduces exactly the split runs/train_swin_cnn_mnr.py trained
    against -- the test indices line up with the released checkpoints.
    """
    return hier_dataset.split_train_test_val(*split)


def build_test_loader(
    image_transforms=EVAL_TRANSFORMS,
    split=SPLIT,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    print_details: bool = True,
):
    """
    (hier_dataset, test_loader) for the held-out test split of the main
    Hier-Zooplankton dataset -- the "same as the training workflow" evaluation
    target, lined up with the checkpoints via the seeded split.

    Builds its own hier_dataset instance (clean transforms), so mutating a training
    dataset's transform pipeline elsewhere can't corrupt this loader. Train/val
    indices are discarded.
    """
    ds = build_hier_dataset(image_transforms=image_transforms, print_details=print_details)
    _, _, test_idx = train_val_test_split(ds, split)
    print(f"Test split: {len(test_idx)} samples")
    loader = _eval_loader(Subset(ds, test_idx), batch_size, num_workers, ds.collate_fn)
    return ds, loader


def build_messy_loader(
    image_transforms=EVAL_TRANSFORMS,
    # max_class_size: int = MAX_CLASS_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    print_details: bool = True,
):
    """
    (messy_hier_dataset, messy_loader) for the full Messy_background set -- real
    messy/dirty-background images, the held-out test set hiernet_mnr.ipynb reports
    on. Same construction as runs/evaluate_hierroute_messy_background.py (no
    subdirectories, no exclude log, no per-class cap, SequentialSampler).
    """
    set_seed(SEED)
    base = ImageDataset(
        data_directory=MESSY_BACKGROUND_DIR,
        class_names=HIERNET_CLASSES,
        class_folder_map=HIERNET_CLASS_FOLDER_MAP,
        # max_class_size=max_class_size,
        image_resolution=RESOLUTION,
        format_file=".tif",
        seed=SEED,
    )
    ds = _hier_dataset(base, image_transforms)
    if print_details:
        ds.print_dataset_details()
    loader = _eval_loader(ds, batch_size, num_workers, ds.collate_fn)
    return ds, loader
