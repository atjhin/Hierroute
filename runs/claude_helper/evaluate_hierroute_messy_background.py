"""
Evaluate a trained HierRouteNet checkpoint (swin_t / swin_s / efficientnet_b0,
from train_swin_cnn_mnr.py) against the full Messy_background set, using the
exact same construction as runs/hiernet_mnr.ipynb's evaluate_variant cell
(uncapped ImageDataset, same seed, Resize+ToTensor only) -- the same
construction evaluate_bcnn_messy_background.py replicates for BCNN, so all
eval_results.json files stay directly comparable.

Usage: python runs/evaluate_hierroute_messy_background.py <checkpoint_dir> [backbone]
  backbone defaults to swin_t; pass efficientnet_b0/swin_s explicitly for those.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms

from hierroute.constants import HIERNET_CLASSES, HIERNET_CLASS_FOLDER_MAP, hiernet_adjacency_graph, SEED, RESOLUTION
from hierroute.data_setup import ImageDataset, HierImageDataset
from hierroute import HierRouteNet, Trainer

CHECKPOINT_DIR = Path(sys.argv[1])
BACKBONE = sys.argv[2] if len(sys.argv) > 2 else 'swin_t'
EXPERT_TYPE = 'cnn'
OUT_DIR = CHECKPOINT_DIR.parent / f'{CHECKPOINT_DIR.name}_messy_eval'

DATA_DIRECTORY = os.environ['ZOOPLANKTON_DATA_DIR']
MESSY_BACKGROUND_DIR = os.path.join(DATA_DIRECTORY, 'Messy_background')

messy_dataset = ImageDataset(
    data_directory=MESSY_BACKGROUND_DIR,
    class_names=HIERNET_CLASSES,
    class_folder_map=HIERNET_CLASS_FOLDER_MAP,
    image_resolution=RESOLUTION,
    format_file='.tif',
    seed=SEED,
)
messy_hier_dataset = HierImageDataset(
    base_dataset=messy_dataset,
    adjacency_graph=hiernet_adjacency_graph,
    levels=3,
    leaves_only=True,
    image_transforms=transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor(),
    ]),
)
messy_hier_dataset.print_dataset_details()

messy_loader = DataLoader(
    messy_hier_dataset, batch_size=64,
    sampler=SequentialSampler(messy_hier_dataset),
    collate_fn=messy_hier_dataset.collate_fn,
)

device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = HierRouteNet(
    hierarchy=messy_hier_dataset.hierarchy,
    label_to_id=messy_hier_dataset.label_to_ids,
    backbone=BACKBONE, expert_type=EXPERT_TYPE,
    checkpoint_dir=str(CHECKPOINT_DIR),
)

eval_trainer = Trainer(learning_rate=0, max_epochs=0, device=device, model_dir=str(OUT_DIR))
result = eval_trainer.predict(model, messy_loader, save=True)

for lvl in sorted(result['level_results'], key=int):
    r = result['level_results'][lvl]['overall']
    print(f"  level {lvl}: acc={r['accuracy']:.4f} f1={r['f1']:.4f} n={r['n_samples']}")

print(f'\nSaved to {OUT_DIR}/eval_results.json')
