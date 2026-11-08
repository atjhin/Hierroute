import os
import sys
from datetime import date
from pathlib import Path

import torch
from dotenv import load_dotenv
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
load_dotenv()

from hierroute import HierRouteNet, Trainer
from hierroute.constants import HIERNET_CLASSES, HIERNET_CLASS_FOLDER_MAP, hiernet_adjacency_graph, SEED, MAX_CLASS_SIZE, RESOLUTION
from hierroute.data_setup import ImageDataset, HierImageDataset
from hierroute.extra_functions import set_seed, RandomBackgroundNoise

DATA_DIRECTORY = os.environ['ZOOPLANKTON_DATA_DIR']
DATA_SUBDIRECTORIES = ['IssacData']
# Included in full on top of MAX_CLASS_SIZE rather than sampled down with the rest
UNCAPPED_SUBDIRECTORIES = ['Quetico_training_data']
EXCLUDE_LOG_PATH = os.path.join(DATA_DIRECTORY, 'Messy_background', 'moved_images', 'log.md')
ADD_BACKGROUND_NOISE = os.environ.get('SWIN_CNN_NOISE', '1') == '1'
BATCH_SIZE = 64
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 8))

BACKBONE = os.environ.get('MODEL_BACKBONE', 'swin_t')
EXPERT_TYPE = 'cnn'
SUPPORTED_BACKBONES = ('swin_t', 'swin_s', 'efficientnet_b0')
if BACKBONE not in SUPPORTED_BACKBONES:
    raise ValueError(f'MODEL_BACKBONE={BACKBONE!r} not supported; choose one of {SUPPORTED_BACKBONES}')
print(f'Backbone: {BACKBONE} | expert: {EXPERT_TYPE} | background noise: {ADD_BACKGROUND_NOISE}')

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
    uncapped_subdirectories=UNCAPPED_SUBDIRECTORIES,
)

hier_dataset = HierImageDataset(
    base_dataset=dataset,
    adjacency_graph=hiernet_adjacency_graph,
    levels=3,
    leaves_only=True,
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
    # transforms.RandomRotation(180, fill=210),
    # transforms.Pad(padding=5, fill=210), # modified from 0 (black), now fills wth average background color
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
]
train_transforms = transforms.Compose(train_transforms_list)
hier_dataset.append_image_transforms(train_transforms, replace=True)

train_split, val_split, test_split = hier_dataset.split_train_test_val(0.7, 0.1, 0.2)
train_loader, val_loader, test_loader = hier_dataset.create_dataloaders(
    batch_size=BATCH_SIZE,
    train_indices=train_split,
    val_indices=val_split,
    test_indices=test_split,
    balanced=True,
    num_workers=NUM_WORKERS,
)

model = HierRouteNet(
    hierarchy=hier_dataset.hierarchy,
    label_to_id=hier_dataset.label_to_ids,
    backbone=BACKBONE,
    expert_type=EXPERT_TYPE,
    freeze_backbone=False,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
run_date = date.today().strftime('%m-%d')
model_dir = f"training_result/mnr/{BACKBONE}_{EXPERT_TYPE}_" + ('noise' if ADD_BACKGROUND_NOISE else 'baseline') + f"_{run_date}"

trainer = Trainer(
    learning_rate=3e-4,
    max_epochs=40,
    device=device,
    print_every=1,
    model_dir=model_dir,
)
trainer.fit(model, train_loader, val_loader, scheduler=True, patience=10, delta=0.001)

trainer.predict(model, test_loader, save=True)
