"""
Train BcnnDensenet121 on MNR zooplankton data.
- Model  : BCNN_Model (hierclassifier) with densenet121 backbone
- Data   : same classes, transforms, splits as mnr.ipynb cells 1-4
            (leaf-only classes, balanced sampler not used — matches hierclassifier API)
- Metrics: macro/micro accuracy & F1 at leaf level; hierarchical accuracy & F1 across all levels
"""

import sys
import os
import time
from pathlib import Path

# Make hierclassifier and modular importable from HierNet
HIERNETPATH = Path(__file__).resolve().parents[1] / "HierNet" / "HierNet"
sys.path.insert(0, str(HIERNETPATH))

import torch
from torchvision import transforms
from sklearn.metrics import f1_score

from hierclassifier.extra_functions import set_seed
from hierclassifier.samples_setup import ImageDataset, HierImageDataset
from hierclassifier.bcnn_setup import BCNN_Model

# ─── Data constants (match hierroute/constants.py and mnr.ipynb) ───────

data_directory      = '/Users/alexandermichaeltjhin/Everything/Repos/Zooplankton/data'
data_subdirectories = ['Processed Data']
SEED                = 666
MAX_CLASS_SIZE      = 6000
RESOLUTION          = 64

# Leaf-only classes — mirrors leaves_only=True in mnr.ipynb
# (excludes intermediate nodes Copepoda, Cladocera, Fiber which have no images of their own)
LEAF_CLASSES = [
    'Calanoid', 'Cyclopoid', 'Harpacticoid', 'Nauplius_Copepod',  # Copepoda children
    'Bosminidae', 'Daphnia',                                        # Cladocera children
    'Rotifer',                                                      # Rotifer (leaf)
    'Bubbles', 'Exoskeleton', 'Plant_Matter',                       # Zoop-No leaves
    'Fiber_Hairlike', 'Fiber_Squiggly',                             # Fiber children
]

# ─── Hierarchy (derived from hier_adjacency_graph in constants.py) ────────────
# hierclassifier expects groups/coarse_names ordered finest → coarsest
# Each group entry lists the *leaf* class names that belong to that coarse class

# Level 3 — species (identity, one group per leaf class)
groups3 = [
    ['Calanoid'], ['Cyclopoid'], ['Harpacticoid'], ['Nauplius_Copepod'],
    ['Bosminidae'], ['Daphnia'],
    ['Rotifer'],
    ['Bubbles'], ['Exoskeleton'], ['Plant_Matter'],
    ['Fiber_Hairlike'], ['Fiber_Squiggly'],
]
coarse_names3 = [
    'Calanoid', 'Cyclopoid', 'Harpacticoid', 'Nauplius_Copepod',
    'Bosminidae', 'Daphnia',
    'Rotifer',
    'Bubbles', 'Exoskeleton', 'Plant_Matter',
    'Fiber_Hairlike', 'Fiber_Squiggly',
]

# Level 2 — order (Copepoda / Cladocera / Rotifer / Bubbles / Exoskeleton / Fiber / Plant_Matter)
groups2 = [
    ['Calanoid', 'Cyclopoid', 'Harpacticoid', 'Nauplius_Copepod'],
    ['Bosminidae', 'Daphnia'],
    ['Rotifer'],
    ['Bubbles'],
    ['Exoskeleton'],
    ['Fiber_Hairlike', 'Fiber_Squiggly'],
    ['Plant_Matter'],
]
coarse_names2 = ['Copepoda', 'Cladocera', 'Rotifer', 'Bubbles', 'Exoskeleton', 'Fiber', 'Plant_Matter']

# Level 1 — zooplankton vs non-zooplankton
groups1 = [
    ['Calanoid', 'Cyclopoid', 'Harpacticoid', 'Nauplius_Copepod', 'Bosminidae', 'Daphnia', 'Rotifer'],
    ['Bubbles', 'Exoskeleton', 'Fiber_Hairlike', 'Fiber_Squiggly', 'Plant_Matter'],
]
coarse_names1 = ['Zoop-yes', 'Zoop-No']

groups       = [groups3, groups2, groups1]   # finest → coarsest
coarse_names = [coarse_names3, coarse_names2, coarse_names1]

LEVELS = 3

# dim_outputs: insert(0, ...) reverses to [coarsest, ..., finest] = [2, 7, 12]
dim_outputs = []
for element in coarse_names:
    dim_outputs.insert(0, len(element))

# ─── Device ───────────────────────────────────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using device: CUDA GPU")
else:
    device = torch.device('cpu')
    print("Using device: CPU")

# ─── Data (mirrors mnr.ipynb cells 1–4) ────────────────────────────────

set_seed(SEED)

dataset = ImageDataset(
    data_directory      = data_directory,
    data_subdirectories = data_subdirectories,
    class_names         = LEAF_CLASSES,
    max_class_size      = MAX_CLASS_SIZE,
    image_resolution    = RESOLUTION,
    image_transforms    = None,
    format_file         = '.tif',
    seed                = SEED,
)

hier_dataset = HierImageDataset(
    base_dataset = dataset,
    groups       = groups,
    coarse_names = coarse_names,
)
hier_dataset.print_dataset_details()

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.Pad(padding=5, fill=0),
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
])

hier_dataset.append_image_transforms(image_transforms=train_transforms, replace=True)

TRAIN_PROP = 0.7
VAL_PROP   = 0.1
TEST_PROP  = 0.2
BATCH_SIZE = 64

train_split, val_split, test_split = hier_dataset.split_train_test_val(
    train_prop=TRAIN_PROP, val_prop=VAL_PROP, test_prop=TEST_PROP
)

train_loader, val_loader, test_loader = hier_dataset.create_dataloaders(
    batch_size           = BATCH_SIZE,
    train_indices        = train_split,
    val_indices          = val_split,
    test_indices         = test_split,
    image_transforms     = None,
    train_sample_weights = None,
)

# ─── Model & Training (mirrors train_bcnn_new.py) ─────────────────────────────

MODEL_NAME = 'densenet121'

model = BCNN_Model(
    weights_directory = '../HierNet/HierNet/pre_trained_weights',
    dim_outputs       = dim_outputs,
    levels            = LEVELS,
    device            = device,
    model_name        = MODEL_NAME,
)

HYPERPARAMETERS = {
    'loss_fn': {
        'criterion':  'CrossEntropyLoss',
        'alpha':      [1/3, 1/3, 1/3],
        'thresholds': None,
    },
    'optimizer':       'Adam',
    'lr':              5e-4,
    'epochs':          40,
    'scheduler':       {'type': 'CosineAnnealingLR', 'T_max': 50},
    'early_stopping':  {'patience': 15, 'delta': 0.0005},
}

start = time.time()
model.train(
    hyperparameters = HYPERPARAMETERS,
    train_loader    = train_loader,
    val_loader      = val_loader,
)
print(f"Training done in {round((time.time() - start) / 60, 2)} minutes")

# ─── Inference ────────────────────────────────────────────────────────────────

labels, probs, preds, logits = model.predict(test_loader=test_loader)

# ─── Metrics ──────────────────────────────────────────────────────────────────
# labels[i] / preds[i]: i=0 coarsest (Zoop-yes/No), i=2 finest (leaf species)

def _masked(tensor, mask):
    arr = tensor.cpu().numpy()
    return arr[mask(arr)]

level_accs, level_macro_f1s = [], []
for l in range(LEVELS):
    y_true = labels[l].cpu().numpy()
    y_pred = preds[l].cpu().numpy()
    mask   = y_true != -1
    y_true, y_pred = y_true[mask], y_pred[mask]
    level_accs.append((y_true == y_pred).mean())
    level_macro_f1s.append(float(f1_score(y_true, y_pred, average='macro', zero_division=0)))

# Leaf-level (index 2 = finest)
y_true_leaf = labels[2].cpu().numpy()
y_pred_leaf = preds[2].cpu().numpy()
leaf_mask   = y_true_leaf != -1
y_true_leaf, y_pred_leaf = y_true_leaf[leaf_mask], y_pred_leaf[leaf_mask]

unique_cls    = sorted(set(y_true_leaf))
n_per_cls     = [int((y_true_leaf == c).sum()) for c in unique_cls]
per_cls_acc   = [(y_true_leaf[y_true_leaf == c] == y_pred_leaf[y_true_leaf == c]).mean()
                 for c in unique_cls]
per_cls_f1    = f1_score(y_true_leaf, y_pred_leaf, labels=unique_cls, average=None, zero_division=0)

macro_accuracy = sum(per_cls_acc) / len(per_cls_acc)
micro_accuracy = level_accs[2]
hier_accuracy  = sum(level_accs) / LEVELS

macro_f1       = float(f1_score(y_true_leaf, y_pred_leaf, average='macro', zero_division=0))
micro_f1       = float(sum(f * n for f, n in zip(per_cls_f1, n_per_cls)) / sum(n_per_cls))
hier_f1        = sum(level_macro_f1s) / LEVELS

print("\n─── Test Results ─────────────────────────────────")
print(f"  Model          : {MODEL_NAME} (BCNN)")
print(f"  Macro Accuracy : {macro_accuracy:.4f}")
print(f"  Micro Accuracy : {micro_accuracy:.4f}")
print(f"  Hier  Accuracy : {hier_accuracy:.4f}")
print(f"  Macro F1       : {macro_f1:.4f}")
print(f"  Micro F1       : {micro_f1:.4f}")
print(f"  Hier  F1       : {hier_f1:.4f}")
print("──────────────────────────────────────────────────")
