"""
Evaluate a trained BCNN checkpoint (from train_bcnn_mnr.py) against the full
Messy_background set, using the exact same image set/order that the hierroute
runs' *_messy_eval/eval_results.json were built from (runs/hiernet_mnr.ipynb's
evaluate_variant cell: uncapped ImageDataset, same seed, same HIERNET_CLASSES/
HIERNET_CLASS_FOLDER_MAP), so results are directly comparable across all runs.

Writes training_result/mnr/<checkpoint_dir_name>_messy_eval/eval_results.json in
the SAME shape as hierroute.Trainer.predict(save=True) produces (level_results
keyed '1'/'2'/'3', each with overall{accuracy,f1,precision,recall,n_samples} and
per_class{...}), so runs/build_comparison_tables.py can read all four uniformly.

One structural difference is corrected for here: hierroute's tree has 'Zoop-No'
as a leaf at depth 1 (it has no finer sub-classification), so its level-2/3
(mid/fine) evaluations naturally exclude Zoop-No samples. BCNN's groups/
coarse_names instead give Zoop-No a self-referential label at every level (see
train_bcnn_mnr.py's build_levels), so every sample gets a valid mid/fine
prediction too. To keep the comparison apples-to-apples, BCNN's mid/fine level
metrics here are computed over the same Zoop-No-excluded sample subset.
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HIERNET_ROOT = Path('/home/tjhinale/repo/HierNet')
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(HIERNET_ROOT))

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms

load_dotenv()

from hierroute.constants import HIERNET_CLASSES, HIERNET_CLASS_FOLDER_MAP, hiernet_adjacency_graph, SEED, RESOLUTION
from hierroute.data_setup import ImageDataset
from hierclassifier.samples_setup import HierImageDataset
from hierclassifier.bcnn_setup import BCNN_Model

CHECKPOINT_DIR = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / 'training_result' / 'mnr' / 'bcnn_densenet121_noise_08-23')
CHECKPOINT_DIR = Path(CHECKPOINT_DIR)
OUT_DIR = CHECKPOINT_DIR.parent / f'{CHECKPOINT_DIR.name}_messy_eval'
OUT_DIR.mkdir(exist_ok=True)

DATA_DIRECTORY = os.environ['ZOOPLANKTON_DATA_DIR']
MESSY_BACKGROUND_DIR = os.path.join(DATA_DIRECTORY, 'Messy_background')


def build_levels(graph, leaf_classes):
    """Same derivation as train_bcnn_mnr.py -- kept in sync manually since this
    is a standalone eval script, not an importable module."""
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

env = torch.load(CHECKPOINT_DIR / 'environment.pth', weights_only=False)
model_name = env['model_name']
assert env['dim_outputs'] == dim_outputs, (
    f"Checkpoint's dim_outputs {env['dim_outputs']} != current hiernet_adjacency_graph-derived "
    f"{dim_outputs} -- constants.py has changed since this checkpoint was trained."
)

############################# Messy_background data ###########################
# Identical construction to hiernet_mnr.ipynb's evaluate_variant cell (uncapped,
# same seed) -- this is what the swin/efficientnet *_messy_eval results were
# built from, so all four runs score against the exact same image set.

messy_dataset = ImageDataset(
    data_directory=MESSY_BACKGROUND_DIR,
    class_names=HIERNET_CLASSES,
    class_folder_map=HIERNET_CLASS_FOLDER_MAP,
    image_resolution=RESOLUTION,
    format_file='.tif',
    seed=SEED,
)
name_to_id = {name: cid for name, cid in zip(messy_dataset.class_names, messy_dataset.class_ids)}
messy_dataset.labels = [name_to_id[name] for name in messy_dataset.labels]

messy_hier_dataset = HierImageDataset(
    base_dataset=messy_dataset,
    groups=groups,
    coarse_names=coarse_names,
    image_transforms=transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor(),
    ]),
)
messy_hier_dataset.print_dataset_details()

messy_loader = DataLoader(
    messy_hier_dataset, batch_size=64,
    sampler=SequentialSampler(messy_hier_dataset),
)

############################# Model & inference ################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = BCNN_Model(
    weights_directory=str(HIERNET_ROOT / 'pre_trained_weights'),
    dim_outputs=dim_outputs,
    levels=LEVELS,
    device=device,
    model_name=model_name,
)
model.model.load_state_dict(torch.load(CHECKPOINT_DIR / 'best_model.pt', map_location=device))
model.model.eval()

labels, probs, preds, logits = model.predict(test_loader=messy_loader)

############################# Metrics (hierroute-shaped) #######################
# BCNN levels here are coarsest -> finest (dim_outputs=[2,4,13], matching
# coarse_names order). hierroute's own level_results are keyed '1' (coarsest)
# .. '3' (finest) -- map BCNN's level l to hierroute's level (l+1) so both use
# the same keys, and restrict levels 2/3 (mid/fine) to the samples that reach
# that depth in hierroute's tree (true finest label != 'Zoop-No').

finest_true = labels[LEVELS - 1].cpu().numpy()
zoop_no_id = coarse_names[0].index('Zoop-No')
reaches_depth3 = finest_true != zoop_no_id
print(f'\nSamples excluded from mid/fine metrics (Zoop-No, no finer label): '
      f'{(~reaches_depth3).sum()} / {len(reaches_depth3)}')

level_results = {}
for l in range(LEVELS):
    hierroute_level = l + 1
    y_true = labels[l].cpu().numpy()
    y_pred = preds[l].cpu().numpy()
    names = coarse_names[LEVELS - 1 - l]  # coarse_names is finest->coarsest; labels[l] is coarsest->finest

    mask = np.ones(len(y_true), dtype=bool) if hierroute_level == 1 else reaches_depth3
    yt, yp = y_true[mask], y_pred[mask]

    unique_classes = sorted(set(yt.tolist()))
    acc = float((yt == yp).mean())
    prec, rec, f1, support = precision_recall_fscore_support(
        yt, yp, labels=unique_classes, average=None, zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        yt, yp, labels=unique_classes, average='macro', zero_division=0
    )

    per_class = {}
    for i, cid in enumerate(unique_classes):
        cls_name = names[cid]
        n_cls = int(support[i])
        correct = int(((yt == cid) & (yp == cid)).sum())
        per_class[cls_name] = {
            'node_id': int(cid),
            'accuracy': round(correct / n_cls, 4) if n_cls else 0.0,
            'f1': round(float(f1[i]), 4),
            'precision': round(float(prec[i]), 4),
            'recall': round(float(rec[i]), 4),
            'n_samples': n_cls,
        }

    level_results[str(hierroute_level)] = {
        'overall': {
            'accuracy': round(acc, 4),
            'f1': round(float(f1_macro), 4),
            'precision': round(float(prec_macro), 4),
            'recall': round(float(rec_macro), 4),
            'n_samples': int(mask.sum()),
        },
        'per_class': per_class,
    }
    print(f"  level {hierroute_level}: acc={acc:.4f} f1={f1_macro:.4f} n={mask.sum()}")

with open(OUT_DIR / 'eval_results.json', 'w') as f:
    json.dump({'level_results': level_results}, f, indent=2)

print(f'\nSaved to {OUT_DIR}/eval_results.json')
