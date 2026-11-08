import os
import sys
from pathlib import Path

import torch
import pandas as pd
from dotenv import load_dotenv
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / 'runs' / 'scripts'))
load_dotenv()

from hierroute.constants import (
    HIERNET_CLASSES, HIERNET_CLASS_FOLDER_MAP, hiernet_adjacency_graph,
    SEED, MAX_CLASS_SIZE, RESOLUTION,
)
from hierroute.data_setup import ImageDataset, HierImageDataset
from hierroute.extra_functions import set_seed, RandomBackgroundNoise
from hierroute import HierRouteNet
from hiernet_eval import predict_with_confidences, apply_confidence_threshold, classify_outcome

# Runs the same confidence-threshold cascade as runs/confidence_threshold_experiment.ipynb
# (see that notebook's "Visualize confidence" / "Outcome classification" sections for the
# full explanation) as a standalone script, against a fixed list of threshold vectors, and
# saves the resulting outcome-proportion table to runs/experiment_results/.

TRAINING_RESULT_PATH = str(REPO_ROOT / 'training_result/mnr/swin_t_cnn_baseline_08-23')  # <-- point this at any checkpoint_dir
BACKBONE = 'swin_t'
EXPERT_TYPE = 'cnn'
BATCH_SIZE = 64
NUM_WORKERS = 8

# Best-effort guess at whether this checkpoint was trained with SWIN_CNN_NOISE=1, from the
# training_result/mnr/swin_t_cnn_{baseline,noise}[_date] naming convention launch_training.sh
# uses -- override manually if a path doesn't follow it.
ADD_BACKGROUND_NOISE = 'noise' in os.path.basename(TRAINING_RESULT_PATH).lower()

THRESHOLDS_TO_TEST = [
    (0.99, 0.95, 0.95),
    (0.975, 0.95, 0.9),
    (0.975, 0.95, 0.8),
    (0.975, 0.95, 0.7),
]

OUTPUT_PATH = REPO_ROOT / 'runs/experiment_results/confidence_threshold_outcomes.csv'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

# --- Rebuild the exact train/val/test split (mirrors runs/train_swin_cnn_mnr.py) -----------
DATA_DIRECTORY = os.environ['ZOOPLANKTON_DATA_DIR']
DATA_SUBDIRECTORIES = ['IssacData']
EXCLUDE_LOG_PATH = os.path.join(DATA_DIRECTORY, 'Messy_background', 'moved_images', 'log.md')

set_seed(SEED)

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

hier_dataset = HierImageDataset(
    base_dataset=dataset,
    adjacency_graph=hiernet_adjacency_graph,
    levels=3,
    leaves_only=True,
)

# Same transform pipeline train_swin_cnn_mnr.py builds -- applied to train AND test/val
# alike, since create_dataloaders(..., image_transforms=None) makes every split a Subset
# of this same hier_dataset. Kept identical here for fidelity with what training measured.
train_transforms_list = []
if ADD_BACKGROUND_NOISE:
    train_transforms_list.append(RandomBackgroundNoise(p=0.5))
train_transforms_list += [
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.Pad(padding=5, fill=0),
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
]
hier_dataset.append_image_transforms(transforms.Compose(train_transforms_list), replace=True)

train_split, val_split, test_split = hier_dataset.split_train_test_val(0.7, 0.1, 0.2)
_, _, test_loader = hier_dataset.create_dataloaders(
    batch_size=BATCH_SIZE,
    train_indices=train_split,
    val_indices=val_split,
    test_indices=test_split,
    balanced=True,
    num_workers=NUM_WORKERS,
)
print(f'Test set size: {len(test_split)}')

# --- Load the checkpoint ---------------------------------------------------------------------
model = HierRouteNet(
    hierarchy=hier_dataset.hierarchy,
    label_to_id=hier_dataset.label_to_ids,
    backbone=BACKBONE,
    expert_type=EXPERT_TYPE,
    checkpoint_dir=TRAINING_RESULT_PATH,
)
model.to(device)
model.eval()


# --- Run inference once over the test set ------------------------------------------------------
all_paths, all_confidences, all_true_leaves = [], [], []
for batch in test_loader:
    image = batch['image'].to(device)
    true_ids = batch['label_node'].argmax(dim=1).tolist()

    paths, confidences = predict_with_confidences(model, image)
    all_paths.extend(paths)
    all_confidences.extend(confidences)
    all_true_leaves.extend(true_ids)

print(f'Cached predictions for {len(all_paths)} test samples')

# --- Evaluate each threshold vector --------------------------------------------------------
OUTCOME_KEYS = ('correct_lowest', 'loosely_correct', 'loosely_incorrect', 'incorrect', 'abstained')
COLUMN_NAMES = ('fully_correct', 'loosely_correct', 'loosely_incorrect', 'incorrect', 'abstained')

rows = []
for thresholds in THRESHOLDS_TO_TEST:
    outcomes = [
        classify_outcome(apply_confidence_threshold(p, c, thresholds), p, len(c), t, model.hierarchy)
        for p, c, t in zip(all_paths, all_confidences, all_true_leaves)
    ]
    n = len(outcomes)
    row = {'threshold': ', '.join(str(t) for t in thresholds)}
    for key, col in zip(OUTCOME_KEYS, COLUMN_NAMES):
        row[col] = outcomes.count(key) / n
    rows.append(row)

table = pd.DataFrame(rows)
print()
print(table.to_string(index=False))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(OUTPUT_PATH, index=False)
print(f'\nSaved table -> {OUTPUT_PATH}')
