# hierroute

Hierarchical classification of plankton images, with ready-to-use models for freshwater
zooplankton imaged on a FlowCam.

Each particle is classified top-down through a taxonomy: first zooplankton or not, then the major
group (Copepoda, Cladocera, Rotifer), then the species. Because every prediction follows a path
through the tree, it is always taxonomically consistent, and the model reports how confident it was
at each level.

```
pip install hierroute
hierroute evaluate my_crops/ --out results/
```

---

## Contents

1. [Install](#install)
2. [Quick start](#quick-start)
3. [Input layouts](#input-layouts)
4. [Outputs](#outputs)
5. [Choosing a model](#choosing-a-model)
6. [Command reference](#command-reference)
7. [Training your own model](#training-your-own-model)
8. [The taxonomy](#the-taxonomy)
9. [Limitations](#limitations)
10. [Using it from Python](#using-it-from-python)
11. [Development](#development)

---

## Install

Requires Python 3.10 or newer.

```bash
pip install hierroute
```

This installs PyTorch as well. On Linux the default PyTorch wheel includes CUDA support. If you need
a specific CUDA version, or a smaller CPU-only install, install PyTorch first by following
[pytorch.org/get-started](https://pytorch.org/get-started/locally/), then run `pip install hierroute`.

Check the install:

```bash
hierroute --version
hierroute download --list
```

The model weights (about 115 MB each) download automatically the first time a model is used.
For computers without internet access, see [Offline use](#offline-use).

---

## Quick start

**1. You have a FlowCam export with manual classes** (the CSV has a `Class` column). Crop each
particle into its own image, sorted into one folder per class:

```bash
hierroute crop path/to/flowcam_export --out crops/
```

**2. Check how well the model does on your data**, by comparing its predictions with your manual
classes:

```bash
hierroute evaluate crops/ --labelled --out results/
```

**3. Classify new, unclassified particles.** You can point it straight at the raw FlowCam export;
nothing needs to be cropped first:

```bash
hierroute evaluate path/to/flowcam_export --input-type flowcam --out results/
```

or at a folder of particle images you already have:

```bash
hierroute evaluate my_particle_images/ --out results/
```

Each command prints what it found and where it saved its results. Add `-h` to any command for all
of its options (for example `hierroute evaluate -h`).

---

## Input layouts

All images are `.tif` files.

### Raw FlowCam export (`hierroute crop`, or `hierroute evaluate --input-type flowcam`)

The mosaic images plus the export CSV(s). Folders can be nested in any way; the CSVs can sit next to
the mosaics or in a separate folder (pass it with `--csv-dir`).

```
flowcam_export/
├── 20240430_Erie_WNS0904_2mm_Rep1/
│   ├── 20240430_Erie_WNS0904_2mm_Rep1.csv          ← one row per particle
│   ├── 20240430_Erie_WNS0904_2mm_Rep1_000001.tif   ← mosaics
│   ├── 20240430_Erie_WNS0904_2mm_Rep1_000002.tif
│   └── ...
└── 20240430_Erie_WNS0904_2mm_Rep2/
    └── ...
```

The CSV needs FlowCam's `Particle ID`, `Image File`, `Image X`, `Image Y`, `Image Width` and
`Image Height` columns. Each particle is found on the mosaic named in its `Image File` column, so
CSV file names don't matter. `_bin.tif` masks and `cal_image` files are ignored.

For labelled data, the manual class comes from the `Class` column (`--label-column` to change it).
`Nauplius` becomes `Nauplius_Copepod` using the `Group` column, or `Copepod` if there is no such
column. Use `--class-map` to rename classes and `--ignore-classes` to skip some (for example
`--ignore-classes Other Duplicate`).

### Individual particle images, unlabelled (`hierroute evaluate`)

Any folder of particle `.tif` files, at any depth:

```
my_particle_images/
├── 20240430_Erie_WNS0904_2mm_Rep1_000001_1.tif
├── 20240430_Erie_WNS0904_2mm_Rep1_000001_2.tif
└── ...
```

### Individual particle images, labelled (`hierroute evaluate --labelled`)

One folder per manual class directly inside the input folder. This is what `hierroute crop` writes.

```
crops/
├── Bosmina/
│   ├── 20240430_Erie_WNS0904_2mm_Rep1_000001_1.tif
│   └── ...
├── Calanoid/
├── Detritus/
├── ...
└── manifest.csv        ← written by hierroute crop
```

Folder names are matched to the taxonomy as follows:

- **A taxonomy class name** (e.g. `Bosmina`, `Calanoid`, `Zoop-No`) is graded as that class.
- **Known aliases** (e.g. `Bosminidae` → `Bosmina`). Catch-all folders such as `Detritus`, `Debris`,
  `Bubbles` and `Small_Organism` are graded as `Zoop-No`. Coarse IDs `Copepod`, `Cladoceran`, `Rotifer`
  and `Zooplankton` are graded as that branch's `_Unknown` class. The full list is in
  [`hierroute/hierarchies/mnr_17leaf.yaml`](hierroute/hierarchies/mnr_17leaf.yaml).
- **Any other folder** (e.g. `MultiClass`, `Partial`) still gets predictions in `predictions.csv`, but is
  not graded. The command lists these folders when it runs.

To grade your own folder names, add a YAML file mapping folder → class:

```yaml
# my_folders.yaml
Bosminid: Bosmina
Cyclopoida: Cyclopoid
Junk: Zoop-No
```

```bash
hierroute evaluate crops/ --labelled --folder-map my_folders.yaml --out results/
```

---

## Outputs

### `hierroute crop`

- One `.tif` per particle: `<out>/<class>/<mosaic name>_<Particle ID>.tif`, or `<out>/<run>/...`
  with `--unlabelled`.
- `manifest.csv` with one row per particle. It records where the crop was written, its status
  (`written`, `exists`, `missing mosaic`, `zero-size box`, `ignored class`), whether the box ran past
  the mosaic edge, and every FlowCam measurement as a `csv_*` column.

Crops that already exist are skipped, so an interrupted run can simply be restarted
(`--overwrite` to redo them).

### `hierroute evaluate`

| File | When | Contents |
|---|---|---|
| `predictions.csv` | always | one row per image (see below) |
| `leaf_accuracy.png` | `--labelled` | per-class N, accuracy, precision, recall, F1 |
| `id_outcomes_table.png` | `--labelled` | per class, the share of each ID outcome (below) |
| `id_outcomes_stacked_bar.png` | `--labelled` | the same, as a stacked bar chart |
| `hierarchy_level_metrics.png` | `--labelled` | accuracy, precision, recall, F1 at each level of the tree |
| `skipped_particles.csv` | FlowCam input with problems | particles that could not be predicted, and why |

`predictions.csv` columns:

| Column | Meaning |
|---|---|
| `image_file` | crop name (`<mosaic name>_<Particle ID>.tif`) |
| `run` / `folder` | FlowCam run (raw exports) / sub-folder (unlabelled crops) |
| `actual_class`, `in_taxonomy` | manual class and whether it was graded (`--labelled`) |
| `predicted_class_1` | the model's prediction (the most specific class it reached) |
| `predicted_class_2` | the runner-up class |
| `confidence_level_1..3` | the model's confidence in its decision at each level (0–1) |
| `predicted_level_1..3` | the predicted class at each level (empty below `Zoop-No`, which has no sub-levels) |
| `correct_all` ... `incorrect_all` | the ID outcome, one `True` per graded row (`--labelled`) |
| `out_of_bounds` | the FlowCam box ran past the mosaic edge (raw exports) |
| `csv_*` | the particle's FlowCam measurements, when the CSVs or `manifest.csv` are available |

**ID outcomes** describe where a prediction first disagrees with the manual ID:

| Outcome | Meaning |
|---|---|
| Correct_All | same class as the manual ID (or within it, for a coarse manual ID like `Rotifer`) |
| Correct_Incomplete | right branch, less specific: e.g. a Cyclopoid predicted as `Copepoda_Unknown` |
| Incorrect_L3 | right major group, wrong species |
| Incorrect_L2 | right at level 1 (zooplankton or not), wrong major group |
| Incorrect_All | wrong already at level 1 |
| Abstained | only with `--thresholds`: no level was confident enough |

By default `leaf_accuracy.png` counts a `Correct_Incomplete` prediction as a hit for the true class
(add `--strict-leaf-accuracy` for exact matches only).

**Optional outputs** for `--labelled`:

| Flag | Output |
|---|---|
| `--zoop-only` | `leaf_accuracy_zoop_only.png`, leaf accuracy without the (usually very large) `Zoop-No` class |
| `--by-sample` | `id_outcomes_stacked_bar_by_sample.png`, one bar per FlowCam run |
| `--breakdown FOLDER ...` | `top_predictions_breakdown.png`, the top-3 predictions for folders such as `Unknown MultiClass` |
| `--capped [N]` | `leaf_accuracy_capped.png` and `id_outcomes_table_capped.png`, with at most N images per class (default 3000) |
| `--thresholds T1 T2 T3` | grade with confidence gating: a prediction backs off to a coarser level (or abstains) when the model's confidence at a level is below that level's threshold, e.g. `--thresholds 0.98 0.995 0.99` |
| `--exclude-runs PATTERN ...` | leave out images from FlowCam runs whose name contains a pattern |
| `--training-accuracy` | `training_result_accuracy.png`, from a self-trained model's own test split (needs `--weights`) |

---

## Choosing a model

Two models are released. Both use a Swin-T backbone with CNN experts on 64 × 64 greyscale images,
and the same 17-class taxonomy:

| Model | Flag | Use for |
|---|---|---|
| **noise** (default) | `--model noise` | messy or dirty backgrounds: trained with synthetic specks and fibres added to the background |
| baseline | `--model baseline` | clean backgrounds |

The noise model is the safer default. It is clearly more accurate on messy backgrounds and about
the same on clean ones.

| Accuracy | Model | Level 1 (Zoop yes / no) | Level 2 (group) | Level 3 (species) |
|---|---|---|---|---|
| Held-out test split (14,345 images) | noise | 0.993 | 0.987 | 0.947 |
| | baseline | 0.993 | 0.988 | 0.948 |
| Lake Erie test set (42,357 manually classified particles) | noise | 0.985 | 0.957 | 0.925 |
| | baseline | 0.984 | 0.962 | 0.931 |

Erie accuracies are from `hierarchy_level_metrics.png` (micro-averaged, with credit for `_Unknown`
siblings). Accuracy on your own data may differ, especially for lakes, seasons or FlowCam settings
not represented in training. Check with `hierroute evaluate --labelled` on a manually classified sample.

---

## Command reference

```
hierroute crop      INPUT --out DIR       FlowCam exports -> particle crops + manifest.csv
hierroute evaluate  INPUT --out DIR       predictions.csv (+ metrics with --labelled)
hierroute train     --config FILE         train a model
hierroute download  [noise|baseline|all]  download released weights
```

Frequently used `evaluate` options:

| Option | Meaning |
|---|---|
| `--model noise` / `--model baseline` | released model (default `noise`) |
| `--weights DIR` | a model you trained yourself (its run folder) |
| `--input-type flowcam` | INPUT is a raw FlowCam export instead of crops |
| `--labelled` | grade against manual classes |
| `--csv-dir DIR` | where the FlowCam CSVs are, if not inside INPUT |
| `--device cpu` | force CPU (default: a GPU if available) |
| `--batch-size N`, `--num-workers N` | speed / memory tuning (defaults 64 and 4) |

A GPU is recommended for large datasets but not required. On a GPU, around 50,000 particles take
well under a minute.

### Offline use

The weights are cached in `~/.cache/hierroute` (or `$HIERROUTE_CACHE_DIR`). On a machine with
internet, run `hierroute download`. Then copy that folder to the same place on the offline machine,
or point `HIERROUTE_CACHE_DIR` at a shared copy.

Training downloads PyTorch's ImageNet starting weights the first time, into `~/.cache/torch`
(`$TORCH_HOME`). Copy that folder the same way, or train with `--set model.pretrained_backbone=false`
to start from random weights.

---

## Training your own model

Training needs a folder with one sub-folder of `.tif` crops per class (`hierroute crop` produces this
from labelled FlowCam exports):

```bash
hierroute train --config configs/mnr_swin_t_cnn_noise.yaml --data-dir /path/to/training_images
```

`configs/` in this repository has the configurations of the released models. Copy one and edit it,
or override single settings on the command line:

```bash
hierroute train --config my_config.yaml --backbone efficientnet_b0 --expert mlp --no-noise --epochs 20
hierroute train --config my_config.yaml --set data.max_class_size=6000 --set train.patience=5
```

| Setting | Options |
|---|---|
| `hierarchy` | `mnr_17leaf` (released models), `mnr_13leaf`, `whoi_s`, `whoi_l`, or your own YAML file |
| `model.backbone` | `swin_t`, `swin_s`, `efficientnet_b0` |
| `model.expert` | `cnn`, `mlp`, `linear` |
| `augment.background_noise` | synthetic background dirt on/off (`--noise` / `--no-noise`) |
| `data.*` | data folders, image format, resolution, per-class cap, train/val/test split |
| `train.*` | learning rate, epochs, early stopping, batch size, seed, device |

All settings and their defaults are in [`hierroute/config.py`](hierroute/config.py) (`TRAIN_DEFAULTS`).

A run creates `training_result/<backbone>_<expert>_<noise|baseline>_<MM-DD>/` with:
- `best_model.pt`, the best checkpoint by validation accuracy
- `model_config.json`, `train_config.yaml`, `training_info.json`
- `eval_results.json` and `predictions.npz`, from the held-out test split

Use it with `--weights`:

```bash
hierroute evaluate crops/ --labelled --weights training_result/swin_t_cnn_noise_11-08 --out results/
```

### Your own taxonomy

Write a hierarchy YAML file and pass its path as `hierarchy`:

```yaml
name: my_lake
tree:                      # parent: [children]; every class appears as a key, leaves have []
  root: [Zoop-yes, Zoop-No]
  Zoop-yes: [Copepoda, Cladocera]
  Zoop-No: []
  Copepoda: [Cyclopoid, Calanoid]
  Cladocera: [Bosmina, Daphnia]
  Cyclopoid: []
  Calanoid: []
  Bosmina: []
  Daphnia: []
folder_map:                # optional: class -> training folders pooled into it
  Bosmina: [Bosmina, Bosminidae]
  Zoop-No: [Detritus, Bubbles, Debris]
eval_folder_map:           # optional: extra folder -> class when grading with --labelled
  Copepod: Copepoda
```

Every leaf class (one with `[]`) needs a training folder of the same name, or an entry in
`folder_map`. Leaves can sit at different depths (here `Zoop-No` is a leaf at level 1).

---

## The taxonomy

The released models predict 17 classes in 3 levels
([`mnr_17leaf.yaml`](hierroute/hierarchies/mnr_17leaf.yaml)):

```
root
├── Zoop-yes
│   ├── Copepoda
│   │   ├── Cyclopoid
│   │   ├── Calanoid
│   │   ├── Nauplius_Copepod
│   │   └── Copepoda_Unknown
│   ├── Cladocera
│   │   ├── Bosmina
│   │   ├── Daphnia
│   │   ├── Holopedium
│   │   ├── Chydorus
│   │   ├── Diaphonosoma
│   │   └── Cladocera_Unknown
│   ├── Rotifer
│   │   ├── Trichocerca
│   │   ├── Asplanchna
│   │   ├── Conochilus
│   │   ├── Kellicottia
│   │   └── Rotifer_Unknown
│   └── Zoop_Unknown
└── Zoop-No            (detritus, debris, bubbles, fibres, exoskeletons, plant matter, small organisms)
```

An `_Unknown` class means "this branch, but not resolved to species", the same as a coarse manual ID.

---

## Limitations

- Images are converted to greyscale and resized to 64 × 64 pixels, so very small particles and fine
  detail are lost.
- The model always picks one of the 17 classes. Organisms outside the taxonomy (e.g. veligers or
  larval fish) are forced into the closest class. Use `confidence_level_*` and `--thresholds` to
  flag low-confidence predictions.
- `Zoop-No` has no sub-levels, so its `predicted_level_2` and `predicted_level_3` are empty.
- Accuracy for rare classes (e.g. Conochilus, Asplanchna) rests on few training images; check them
  on your own data.

---

## Using it from Python

```python
from hierroute import EvaluateOptions, evaluate, load_model

# Same as the evaluate command
predictions = evaluate(EvaluateOptions(input="crops/", out_dir="results/", labelled=True))

# Or load a model directly
model, config = load_model("noise", device="cuda")      # or load_model(weights="training_result/my_run")
```

How the model works:

- A shared backbone (Swin-T, Swin-S or EfficientNet-B0, ImageNet-initialised) extracts features.
- Each internal node of the taxonomy has its own small "expert" classifier that chooses among that
  node's children.
- Training multiplies the experts' probabilities along each root-to-leaf path and applies a binary
  cross-entropy (or focal) loss at the leaves.
- Prediction follows the most confident child at every node, so predictions are always valid paths.

The building blocks (`HierRouteNet`, `Trainer`, `ImageDataset`, `HierImageDataset`, `Hierarchy`,
`Visualize`) are importable from `hierroute` for custom workflows.

---

## Development

```bash
git clone https://github.com/atjhin/Hierroute.git
cd Hierroute
pip install -e ".[dev]"
pytest
```

The test suite runs on CPU with synthetic data and needs no downloads except PyTorch's ImageNet
weights for a few model tests. New versions are published to PyPI automatically when a GitHub
release tagged `vX.Y.Z` is created (see `.github/workflows/publish.yml`).

## License

MIT, see [LICENSE](LICENSE). Authors: Alexander Tjhin, Emily Hassal, Dak de Kerckhove.
