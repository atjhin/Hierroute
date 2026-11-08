"""
data_setup.py -- phytoplankton binary dataset & DataLoaders.

Builds train / val / test DataLoaders for the *phytoplankton vs not phytoplankton*
task, reusing hierroute's ``ImageDataset`` / ``HierImageDataset`` / ``Hierarchy``
the same way ``runs/scripts/eval_loaders.py`` and ``runs/train_swin_cnn_mnr.py``
do. Only the taxonomy differs: a single non-root level with two leaves.

    from data_setup import build_dataloaders

    hier_dataset, train_loader, val_loader, test_loader = build_dataloaders()

    from hierroute import HierRouteNet, Trainer
    model = HierRouteNet(hier_dataset.hierarchy, hier_dataset.label_to_ids,
                         backbone="swin_t", expert_type="cnn")
    Trainer(learning_rate=3e-4, max_epochs=40, device=device).fit(
        model, train_loader, val_loader)

On-disk layout (produced by ``phytoplankton/notebooks/pre_process_data.ipynb``):

    <PHYTOPLANKTON_DATA_DIR>/
    |-- Diatom/            |-- Zooplankton/          |-- Bubble/     ... one folder
    |-- Dinobryon/         |-- Not Phytoplankton/                    per raw class

``constants.BINARY_CLASS_FOLDERS`` pools those raw folders into the two model
classes (``ImageDataset``'s ``class_folder_map``). ``constants.IGNORED_CLASSES``
(EXCLUDE + TBD) are never listed in ``class_names``, so they are simply not
loaded.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from torchvision import transforms

# --- make the shared hierroute library and this folder's constants importable --
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
load_dotenv(REPO_ROOT / ".env")

from hierroute.data_setup import ImageDataset, HierImageDataset
from hierroute.hierarchy import Hierarchy  # noqa: F401  built by HierImageDataset; re-exported for callers
from hierroute.extra_functions import set_seed
from hierroute.constants import SEED, MAX_CLASS_SIZE, RESOLUTION

from constants import BINARY_CLASSES, BINARY_CLASS_FOLDERS, IGNORED_CLASSES  # noqa: F401

# --- configuration -----------------------------------------------------------
PHYTOPLANKTON_DATA_DIR = os.environ.get(
    "PHYTOPLANKTON_DATA_DIR", "/data/Hier-Phytoplankton/Processed_data"
)
LEVELS = 1                          # one non-root level: phytoplankton vs not
BATCH_SIZE = 64
SPLIT = (0.7, 0.1, 0.2)            # train / val / test
FORMAT_FILE = ".tif"

# --- taxonomy: one level, two leaves ---------------------------------------
# Same adjacency-graph shape hierroute uses (parent -> [children]); "root" is
# inferred by HierImageDataset as the node that never appears as a child.
BINARY_ADJACENCY_GRAPH: dict[str, list[str]] = {
    "root": list(BINARY_CLASSES),
    **{class_name: [] for class_name in BINARY_CLASSES},
}

# --- transform pipelines --------------------------------------------------
# Clean pipeline for evaluation / inference.
EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
])

# Training augmentation -- mirrors runs/train_swin_cnn_mnr.py. Flips/rotations are
# label-preserving for plankton crops; Resize + ToTensor close the pipeline.
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(180),
    # transforms.Pad(padding=5, fill=0),
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
])


def build_base_dataset(
    image_transforms=None,
    data_directory: str = PHYTOPLANKTON_DATA_DIR,
    max_class_size: int = MAX_CLASS_SIZE,
    seed: int = SEED,
) -> ImageDataset:
    """
    Flat ``ImageDataset`` over the two pooled model classes
    (``constants.BINARY_CLASSES``), reading the raw FlowCam folders under
    ``data_directory`` via ``constants.BINARY_CLASS_FOLDERS``. EXCLUDE / TBD
    folders are not included.
    """
    set_seed(seed)
    return ImageDataset(
        data_directory=data_directory,
        class_names=list(BINARY_CLASSES),
        class_folder_map=BINARY_CLASS_FOLDERS,
        max_class_size=max_class_size,
        image_resolution=RESOLUTION,
        image_transforms=image_transforms,
        format_file=FORMAT_FILE,
        seed=seed,
    )


def build_hier_dataset(
    image_transforms=None,
    data_directory: str = PHYTOPLANKTON_DATA_DIR,
    max_class_size: int = MAX_CLASS_SIZE,
    seed: int = SEED,
    print_details: bool = True,
) -> HierImageDataset:
    """
    ``HierImageDataset`` wrapping the binary taxonomy (``levels=1``,
    ``leaves_only=True``). ``hier_dataset.hierarchy`` / ``.label_to_ids`` feed
    ``HierRouteNet`` directly.
    """
    base = build_base_dataset(
        image_transforms=None,
        data_directory=data_directory,
        max_class_size=max_class_size,
        seed=seed,
    )
    hier_dataset = HierImageDataset(
        base_dataset=base,
        adjacency_graph=BINARY_ADJACENCY_GRAPH,
        levels=LEVELS,
        leaves_only=True,
        image_transforms=image_transforms,
    )
    if print_details:
        hier_dataset.print_dataset_details()
    return hier_dataset


def train_val_test_split(hier_dataset: HierImageDataset, split=SPLIT):
    """Seeded stratified 0.7 / 0.1 / 0.2 index split -> (train_idx, val_idx, test_idx)."""
    return hier_dataset.split_train_test_val(*split)


def build_dataloaders(
    hier_dataset: HierImageDataset | None = None,
    *,
    batch_size: int = BATCH_SIZE,
    split=SPLIT,
    train_transforms: transforms.Compose | None = TRAIN_TRANSFORMS,
    balanced: bool = True,
    num_workers: int = 0,
    seed: int = SEED,
    print_details: bool = True,
):
    """
    ``(hier_dataset, train_loader, val_loader, test_loader)`` for the binary task,
    built the same way as ``runs/train_swin_cnn_mnr.py``:

        1. build the ``HierImageDataset``
        2. append the transform pipeline (``replace=True``)
        3. seeded stratified split
        4. ``hier_dataset.create_dataloaders(..., balanced=balanced)``

    Args:
        hier_dataset:      Reuse an existing dataset instead of building one.
        batch_size:        DataLoader batch size.
        split:             (train, val, test) proportions.
        train_transforms:  Pipeline applied to the dataset. ``None`` falls back to
                           ``EVAL_TRANSFORMS`` (Resize + ToTensor only).
        balanced:          Train loader draws with an inverse-frequency
                           ``WeightedRandomSampler`` -- Diatom and Not
                           Phytoplankton dominate the raw counts.
        num_workers:       DataLoader worker processes.

    Note: as in the hierroute training scripts, the pipeline set here is shared by
    all three splits (``create_dataloaders`` only applies a *separate* train
    pipeline when passed ``image_transforms=``, which the ``HierImageDataset``
    path does not support). Pass ``train_transforms=EVAL_TRANSFORMS`` when you
    need an augmentation-free test loader for final held-out metrics.
    """
    if hier_dataset is None:
        hier_dataset = build_hier_dataset(seed=seed, print_details=print_details)

    hier_dataset.append_image_transforms(
        image_transforms=train_transforms or EVAL_TRANSFORMS, replace=True
    )

    train_idx, val_idx, test_idx = train_val_test_split(hier_dataset, split)

    train_loader, val_loader, test_loader = hier_dataset.create_dataloaders(
        batch_size=batch_size,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        balanced=balanced,
        num_workers=num_workers,
    )
    return hier_dataset, train_loader, val_loader, test_loader
