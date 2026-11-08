"""
Train HierNet's BCNN (hierclassifier.BCNN_Model) on exactly the same dataset and
hierarchy that runs/train_swin_cnn_mnr.py uses, so the two are directly comparable.

Everything data-side is shared with the swin script -- same ZOOPLANKTON_DATA_DIR,
same IssacData subdirectory, same HIERNET_CLASSES / HIERNET_CLASS_FOLDER_MAP,
same exclusion log, same transforms, split and balanced sampling. Only the model
differs.

Two pieces of plumbing are needed to bridge the repos:
  1. hierclassifier's ImageDataset has no class_folder_map, so hierroute's
     ImageDataset builds the sample list (it has both class_folder_map and
     exclude_log_path) and only the labels are converted for hierclassifier.
  2. hierclassifier expects the hierarchy as groups/coarse_names lists ordered
     finest -> coarsest rather than an adjacency graph, so the levels are derived
     from hiernet_adjacency_graph at runtime (build_levels) instead of being
     hardcoded -- that way this stays in sync when constants.py is edited.
"""

import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HIERNET_ROOT = Path('/home/tjhinale/repo/HierNet')
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(HIERNET_ROOT))

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, SequentialSampler, Subset, WeightedRandomSampler
from torchvision import transforms

load_dotenv()

from hierroute.constants import (
    HIERNET_CLASSES, HIERNET_CLASS_FOLDER_MAP, hiernet_adjacency_graph,
    SEED, MAX_CLASS_SIZE, RESOLUTION,
)
from hierroute.data_setup import ImageDataset, _seed_worker
from hierroute.extra_functions import set_seed, RandomBackgroundNoise

from hierclassifier.samples_setup import HierImageDataset
from hierclassifier.bcnn_setup import BCNN_Model

DATA_DIRECTORY = os.environ['ZOOPLANKTON_DATA_DIR']
DATA_SUBDIRECTORIES = ['IssacData']
EXCLUDE_LOG_PATH = os.path.join(DATA_DIRECTORY, 'Messy_background', 'moved_images', 'log.md')
ADD_BACKGROUND_NOISE = os.environ.get('BCNN_NOISE', '1') == '1'
BATCH_SIZE = 64
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 8))

MODEL_NAME = os.environ.get('BCNN_BACKBONE', 'densenet121')
SUPPORTED = ('densenet121', 'resnet50', 'vgg16')
if MODEL_NAME not in SUPPORTED:
    raise ValueError(f'BCNN_BACKBONE={MODEL_NAME!r} not supported; choose one of {SUPPORTED}')
print(f'BCNN backbone: {MODEL_NAME} | background noise: {ADD_BACKGROUND_NOISE}')

set_seed(SEED)

############################ Hierarchy ########################################


def build_levels(graph, leaf_classes):
    """
    Convert an adjacency graph into hierclassifier's groups/coarse_names format,
    ordered finest -> coarsest.

    Leaves shallower than the deepest level (e.g. 'Zoop-No', which sits directly
    under root while the zooplankton leaves are three deep) are clamped to
    themselves at every level below their own depth, which is how they end up
    carrying the same label at the fine and mid levels.
    """
    parent = {child: p for p, kids in graph.items() for child in kids}

    def path_to_root(node):
        path = [node]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        return list(reversed(path))

    max_depth = max(len(path_to_root(c)) - 1 for c in leaf_classes)

    groups, coarse_names = [], []
    for depth in range(max_depth, 0, -1):
        buckets = {}
        for cls in leaf_classes:
            path = path_to_root(cls)
            ancestor = path[depth] if depth < len(path) else path[-1]
            buckets.setdefault(ancestor, []).append(cls)
        names = list(buckets)
        groups.append([buckets[n] for n in names])
        coarse_names.append(names)
    return groups, coarse_names


groups, coarse_names = build_levels(hiernet_adjacency_graph, HIERNET_CLASSES)
LEVELS = len(groups)

dim_outputs = []
for element in coarse_names:
    dim_outputs.insert(0, len(element))

print(f'Levels: {LEVELS} | dim_outputs (coarsest -> finest): {dim_outputs}')
for lvl, names in enumerate(coarse_names):
    print(f'  level {lvl} ({len(names)}): {names}')

############################# Data ############################################

dataset = ImageDataset(
    data_directory=DATA_DIRECTORY,
    data_subdirectories=DATA_SUBDIRECTORIES,
    class_names=HIERNET_CLASSES,
    class_folder_map=HIERNET_CLASS_FOLDER_MAP,
    max_class_size=MAX_CLASS_SIZE,
    image_resolution=RESOLUTION,
    format_file='.tif',
    seed=SEED,
    exclude_log_path=EXCLUDE_LOG_PATH,
)

# hierclassifier.HierImageDataset expects integer labels; hierroute.ImageDataset stores strings
name_to_id = {name: cid for name, cid in zip(dataset.class_names, dataset.class_ids)}
dataset.labels = [name_to_id[name] for name in dataset.labels]

hier_dataset = HierImageDataset(
    base_dataset=dataset,
    groups=groups,
    coarse_names=coarse_names,
)
hier_dataset.print_dataset_details()

train_transforms_list = []
if ADD_BACKGROUND_NOISE:
    # Applied first, on the raw native-resolution image, since add_synthetic_dirt's
    # speck/fiber sizes were calibrated against real messy-background images at their
    # original resolution -- applying it after Resize would distort those proportions.
    train_transforms_list.append(RandomBackgroundNoise(p=0.8))

train_transforms_list += [
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(180),
    # transforms.Pad(padding=5, fill=0),
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
]
train_transforms = transforms.Compose(train_transforms_list)
hier_dataset.append_image_transforms(image_transforms=train_transforms, replace=True)

train_split, val_split, test_split = hier_dataset.split_train_test_val(
    train_prop=0.7, val_prop=0.1, test_prop=0.2
)

# Mirrors balanced=True in train_swin_cnn_mnr.py: hierclassifier has no `balanced`
# flag, so the inverse-frequency weights are built here from the finest level
# (hier_dataset.labels is ordered coarsest -> finest).
finest_labels = hier_dataset.labels[-1]
train_counts = Counter(finest_labels[i] for i in train_split)
train_sample_weights = torch.tensor(
    [1.0 / train_counts[finest_labels[i]] for i in train_split], dtype=torch.double
)

# hierclassifier's create_dataloaders has no num_workers, so data loading runs
# single-process and RandomBackgroundNoise (CPU-bound PIL work) dominates epoch
# time. Build the loaders here instead -- same structure create_dataloaders would
# produce for this call (transforms already baked into hier_dataset via
# append_image_transforms, weighted sampler on train, sequential on val/test),
# plus worker processes.
# worker_init_fn is hierroute's _seed_worker: RandomBackgroundNoise gates on the
# global `random` RNG, which forked workers would otherwise inherit identically
# and produce correlated noise decisions across workers.
train_dataset = Subset(hier_dataset, train_split)
val_dataset = Subset(hier_dataset, val_split)
test_dataset = Subset(hier_dataset, test_split)
loader_kwargs = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
    worker_init_fn=_seed_worker,
)

train_loader = DataLoader(
    train_dataset,
    sampler=WeightedRandomSampler(
        train_sample_weights, num_samples=len(train_sample_weights), replacement=True
    ),
    **loader_kwargs,
)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), **loader_kwargs)
test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), **loader_kwargs)
print(f'DataLoaders: num_workers={NUM_WORKERS} | train={len(train_split)} val={len(val_split)} test={len(test_split)}')

############################# Model & training ################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = BCNN_Model(
    weights_directory=str(HIERNET_ROOT / 'pre_trained_weights'),
    dim_outputs=dim_outputs,
    levels=LEVELS,
    device=device,
    model_name=MODEL_NAME,
)

HYPERPARAMETERS = {
    'loss_fn': {
        'criterion': 'CrossEntropyLoss',
        'alpha': [1 / 3, 1 / 3, 1 / 3],
        'thresholds': None,
    },
    'optimizer': 'Adam',
    'lr': 5e-4,
    'epochs': 60,
    'scheduler': {'type': 'CosineAnnealingLR', 'T_max': 50},
    'early_stopping': {'patience': 15, 'delta': 0.0005},
    'batch_size': BATCH_SIZE,
}

model.train(hyperparameters=HYPERPARAMETERS, train_loader=train_loader, val_loader=val_loader)

############################# Inference & saving ##############################

labels, probs, preds, logits = model.predict(test_loader=test_loader)

run_date = date.today().strftime('%m-%d')
run_name = f"bcnn_{MODEL_NAME}_" + ('noise' if ADD_BACKGROUND_NOISE else 'baseline') + f"_{run_date}"
model_dir = REPO_ROOT / 'training_result' / 'mnr' / run_name
model_dir.mkdir(parents=True, exist_ok=True)

torch.save(model.model.state_dict(), model_dir / 'best_model.pt')
torch.save((labels, probs, preds, logits), model_dir / 'predictions.pth')
torch.save(
    {
        'model_id': model.model_id,
        'model_name': MODEL_NAME,
        'run_name': run_name,
        'dataset': hier_dataset,
        'classes': hier_dataset.class_names,
        'class_ids': hier_dataset.class_ids,
        'groups': groups,
        'coarse_names': coarse_names,
        'dim_outputs': dim_outputs,
        'levels': LEVELS,
        'hyperparameters': HYPERPARAMETERS,
        'image_transforms': hier_dataset.image_transforms,
        'max_class_size': MAX_CLASS_SIZE,
        'add_background_noise': ADD_BACKGROUND_NOISE,
        'train_metrics': model.train_results,
    },
    model_dir / 'environment.pth',
)

print(f'\nTest-set results ({run_name}):')
summary = {}
for level in range(LEVELS):
    y_true = labels[level].cpu().numpy()
    y_pred = preds[level].cpu().numpy()
    valid = y_true != -1
    acc = float((y_true[valid] == y_pred[valid]).mean())
    summary[f'level_{level}'] = {'accuracy': acc, 'n': int(valid.sum())}
    print(f'  level {level} (n={valid.sum()}): acc={acc:.4f}')

with open(model_dir / 'training_info.json', 'w') as f:
    json.dump(
        {'run_name': run_name, 'hyperparameters': HYPERPARAMETERS,
         'test_results': summary, 'train_metrics': model.train_results},
        f, indent=2, default=str,
    )

print(f'\nSaved to {model_dir}/')
