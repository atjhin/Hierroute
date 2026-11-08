# Deployment plan: `pip install hierroute`

Goal: colleagues install the package with `pip install hierroute`, then use terminal commands to
crop FlowCam exports, run the released Swin-T models on `.tif` data (labelled or unlabelled), and
train their own models.

Status (2026-11-08): steps 1–9 are done and committed locally: `research` at `1d950e6`, `main` at
`fa89a1e`. Nothing has been pushed, released or published. Steps 10–12 wait for the user's approval.
See "Implementation notes" at the end.

---

## 1. Decisions

| Topic | Decision |
|---|---|
| GitHub repo | `atjhin/Hierroute` (renamed from `Zooplankton_Classification`), public |
| Distribution | PyPI, as `hierroute`. Publishing uses PyPI trusted publishing from GitHub Actions. The first real release (`0.1.0`) claims the name; there is no placeholder upload (option A) |
| Released models | `swin_t_cnn_noise_10-20` (default) and `swin_t_cnn_baseline_10-20` |
| Weight hosting | Assets on a GitHub Release (115 MB each, over GitHub's 100 MB per-file git limit). Public release approved by the user |
| License | MIT |
| Authors | Alexander Tjhin <alexander.tjhin@mail.utoronto.ca>, Emily Hassal <Emily.Hassal@ontario.ca>, Dak de Kerckhove <Dak.deKerckhove@ontario.ca> |
| BCNN | Dropped. It is not part of the package |
| Source of truth | The current local `hierroute/` state, including uncommitted edits. This is what the 10-20 models were trained with |
| Branches | `main` holds the package only, with no experiment results. `research` holds the research scripts and notebooks |

---

## 2. Branch and repo layout

### `main`

```
Hierroute/
├── pyproject.toml              # name = "hierroute", console script `hierroute`
├── LICENSE                     # MIT
├── README.md
├── .gitignore
├── hierroute/                  # the only directory shipped to PyPI
│   ├── __init__.py
│   ├── model.py hierarchy.py data_setup.py trainer.py extra_functions.py   (existing)
│   ├── flowcam.py              # FlowCam CSV parsing, mosaic matching, cropping
│   ├── inference.py            # preprocessing, predict_with_confidences, grading + metrics
│   ├── plotting.py             # table/bar rendering (from inference_common.py)
│   ├── pretrained.py           # registry of released weights: URL, sha256, model config
│   ├── config.py               # YAML loading/validation for hierarchies + training
│   ├── cli.py                  # `hierroute crop | evaluate | train | download`
│   └── hierarchies/            # mnr_17leaf.yaml, mnr_13leaf.yaml, whoi_s.yaml, whoi_l.yaml
├── configs/                    # example training configs (see section 5)
├── tests/                      # existing tests (updated) + new smoke tests with tiny fixtures
└── .github/workflows/
    ├── tests.yml               # pytest on push/PR
    └── publish.yml             # build + upload to PyPI on release tags `v*`
```

### `research` branch

This is a snapshot of the current working tree, scripts and notebooks only:
- `runs/`
- `phytoplankton/`
- `training_result/result.py`, `training_result/result.ipynb`, `training_result/plot_whoi_hierarchy.py`
- `mermaid_diagrams/`
- `hierroute/backbone_comparison.md`
- `runs/logs/` (644 KB)

### Never committed (kept local only)

- `runs/inference_results/` (232 MB)
- `training_result/*/` (weights go to the release instead)
- `.env`, `data_temp/`, `hier_zooplankton_csv_data_list.csv`, `__pycache__/`

A `git clone` fetches every branch, so nothing large goes on `research` either.

### `.gitignore`

```
.env
data_temp/
training_result/
runs/inference_results/
*.pt
*.pth
*.npz
__pycache__/
*.egg-info/
dist/
build/
```

---

## 3. Released weights

- **Where they're published.** The weights go on a GitHub Release tagged `models-v1`, separate from package releases (`v0.1.0`, ...). The publish workflow only triggers on `v*` tags. The asset URLs are:
  - `https://github.com/atjhin/Hierroute/releases/download/models-v1/swin_t_cnn_noise_10-20.pt`
  - `https://github.com/atjhin/Hierroute/releases/download/models-v1/swin_t_cnn_baseline_10-20.pt`
- **Registry.** `hierroute/pretrained.py` maps each model name to its URL, a sha256 checksum, and a frozen model config:
  - backbone `swin_t`, expert type `cnn`, resolution 64
  - the full hierarchy, in node-ID order
- **Why the hierarchy is frozen.** Node IDs come from the key order of the adjacency graph, so the checkpoints only stay valid while that order is pinned.
- **Download.** On first use the weights download to `~/.cache/hierroute/` and the checksum is verified. `hierroute download [noise|baseline|all]` pre-fetches them for offline clusters. `--weights DIR` uses a local run directory instead.
- **Loading.** `HierRouteNet` gains `pretrained: bool = True`:
  - Training keeps the ImageNet initialisation from torchvision, downloaded on demand by torchvision. The package does not store the backbone.
  - Loading a checkpoint passes `pretrained=False`, which skips that roughly 110 MB download.
- **Run directories.** Every training run writes `model_config.json` next to `best_model.pt`, in the same format as the registry entries. Self-trained models therefore load exactly like released ones, and backbone detection by folder name goes away.

---

## 4. Commands

### 4.1 `hierroute crop`: FlowCam exports → individual particle `.tif` crops

This generalises HierNet's `Zooplankton-MNR/pre_process_data.ipynb`, which stays as it is in HierNet. The crop logic lives in `hierroute/flowcam.py` and is shared with `evaluate`, so training crops and inference crops are always identical.

```bash
hierroute crop EXPORT_DIR --out CROPS_DIR [--csv-dir DIR] [--label-column Class]
               [--ignore-classes Other Duplicate ...] [--class-map map.yaml] [--workers N] [--overwrite]
```

- **Finding mosaics.** It searches recursively for mosaic `.tif`s, skipping `*_bin.tif` and `cal_image_*`.
- **Finding CSVs.** It searches recursively for CSVs, either next to the mosaics or under `--csv-dir`.
- **Matching CSVs to mosaics.** Each CSV is matched to its mosaics through its `Image File` column, which replaces the notebook's filename-suffix regex. If the column is missing, it falls back to the filename stem.
- **CSV quirks.** It handles unquoted commas in `Image File` and trailing-comma `Unnamed` columns (logic taken from `runs/quetico_process/predict_quetico_raw.py`).
- **Crop box and naming.**
  - The crop box is `(Image X, Image Y, Image X + Image Width, Image Y + Image Height)`.
  - Each crop is named `<mosaic stem>_<Particle ID>.tif`, as in the notebook.
- **Labelled vs unlabelled.**
  - With `--label-column`, crops go to `CROPS_DIR/<class>/`.
  - Without it, they go to `CROPS_DIR/<run>/`.
- **Class options.**
  - `--class-map` handles renames, plus the Nauplius rule (`Nauplius` + `group` → `Nauplius_Copepod`) as a configurable rule.
  - `--ignore-classes` skips the listed classes.
- **Manifest.** It writes `CROPS_DIR/manifest.csv`, one row per crop: file, run, particle ID, label, crop box, whether the box was out of bounds, and every CSV column as `csv_<snake_case>`.
- **Re-runs and reporting.** Crops that already exist are skipped unless `--overwrite` is passed. It reports mosaics with no CSV, CSV rows with no mosaic, and zero-size boxes.

### 4.2 `hierroute evaluate`: the one inference command, for labelled and unlabelled data

This replaces `runs/inference_scripts/`.

```bash
hierroute evaluate INPUT --out RESULTS_DIR [--model noise|baseline | --weights RUN_DIR]
                   [--input-type auto|crops|flowcam] [--labelled] [--label-column Class]
                   [--folder-map map.yaml] [--device cuda|cpu] [--batch-size N] [--num-workers N]
```

- **Input.**
  - A folder of crop `.tif`s, searched recursively, or raw FlowCam exports, which are cropped in memory without writing files.
  - `auto` picks FlowCam when it finds CSVs with an `Image File` column.
- **Preprocessing.** Greyscale → `Resize((64, 64))` → `ToTensor` → repeat to 3 channels. This is identical to `inference_common.EVAL_TRANSFORMS` and `LabeledFolderDataset`.
- **Labels.**
  - Without `--labelled`, only `predictions.csv` is written.
  - With `--labelled`, the true label comes from the class folder (crops) or from `--label-column` (FlowCam), and the metric outputs are added.
- **CSV metadata.** `predictions.csv` gets the `csv_*` columns when a `manifest.csv` or the FlowCam CSVs are available.

#### Default outputs

| File | When |
|---|---|
| `predictions.csv` | always |
| `leaf_accuracy.png` | `--labelled` |
| `id_outcomes_table.png` | `--labelled` |
| `id_outcomes_stacked_bar.png` | `--labelled` |
| `hierarchy_level_metrics.png` | `--labelled` |

The reference output is `runs/inference_results/erie/erie_test_data_swin_t_cnn_noise_09-17_exclude/`.

#### `predictions.csv` columns (same as the reference)

- `image_file`
- `actual_class`, `in_taxonomy` (labelled only)
- `predicted_class_1`, `predicted_class_2`
- `confidence_level_1..3`, `predicted_level_1..3`
- `correct_all`, `correct_incomplete`, `incorrect_l3`, `incorrect_l2`, `incorrect_all` (labelled only)
- `csv_*`

FlowCam input also gets `run` and a `skipped_particles.csv`.

#### Optional outputs (all off by default)

| Flag | Output |
|---|---|
| `--zoop-only-leaf-accuracy` | `leaf_accuracy_zoop_only.png` (looser leaf accuracy without `Zoop-No`) |
| `--by-sample` | `id_outcomes_stacked_bar_by_sample.png` |
| `--breakdown CLASS ...` | `top_predictions_breakdown.png` |
| `--training-accuracy` | `training_result_accuracy.png` (needs `--weights` run dir with `predictions.npz`) |
| `--capped [N]` | `leaf_accuracy_capped.png`, `id_outcomes_table_capped.png` (default N = 3000) |
| `--thresholds T1 T2 T3` | confidence-gated grading with abstaining, Emily style (e.g. `0.98 0.995 0.99`) |
| `--strict-leaf-accuracy` | exact-match leaf accuracy instead of the looser rule |
| `--exclude-runs PATTERN ...` | drop crops from runs whose name matches, before inference (Erie's `--exclude`) |

#### Grading defaults (Erie rules)

- **Folder → taxonomy map.** This is stored in the hierarchy YAML as `eval_folder_map` and can be overridden with `--folder-map`.
  - the training `folder_map` (e.g. `Bosminidae`/`Bosmina_1`/`Eubosmina` → `Bosmina`, `Sididae` → `Diaphonosoma`)
  - `Detritus`, `Small_Organism`, `Larval_Fish`, `Small_Nauplius`, `Small_Rotifer`, `Daphnia_Decomposing`, `Decomposing` → `Zoop-No`
  - `Rotifer` → `Rotifer_Unknown`, `Copepod` → `Copepoda_Unknown`, `Cladoceran` → `Cladocera_Unknown`, `Zooplankton` → `Zoop_Unknown`
- **Unmapped folders** get `in_taxonomy=False`. They stay in `predictions.csv` but are left out of the metrics.
- **`leaf_accuracy` uses the looser rule.** Predicting `<Branch>_Unknown` for a species under that branch counts as a hit (`loosened_leaf_metrics_table`).
- **No confidence gating.** Grading uses the model's best guess at every level, so abstaining is impossible unless `--thresholds` is given.

### 4.3 `hierroute train`: one training entry point

This replaces `runs/training_scripts/train_swin_cnn_mnr.py` and its env-var switches (`MODEL_BACKBONE`, `SWIN_CNN_NOISE`, `NUM_WORKERS`).

```bash
hierroute train --config configs/mnr_swin_t_cnn_noise.yaml [--backbone swin_s] [--expert mlp]
                [--no-noise] [--data-dir DIR] [--out DIR] [--epochs N] [--lr X] ...
```

Command-line flags override config values. The run directory gets:
- `best_model.pt`
- `model_config.json`
- `training_info.json`
- `eval_results.json`
- `predictions.npz`

### 4.4 `hierroute download`

`hierroute download [noise|baseline|all]` pre-fetches the released weights into the cache.

---

## 5. Config files

### Hierarchy YAML (`hierroute/hierarchies/*.yaml`, or a user-supplied file)

```yaml
name: mnr_17leaf
tree:                       # key order defines node IDs -- never reorder for a released model
  root: [Zoop-yes, Zoop-No]
  Zoop-yes: [Copepoda, Cladocera, Rotifer, Zoop_Unknown]
  ...
folder_map:                 # training: leaf -> on-disk folders pooled into it
  Bosmina: [Bosminidae, Bosmina_1, Eubosmina, Bosmina]
  ...
eval_folder_map:            # evaluate --labelled: on-disk folder -> node it is graded against
  Detritus: Zoop-No
  Rotifer: Rotifer_Unknown
  ...
```

The four bundled hierarchies are:
- `mnr_17leaf`: `hiernet_adjacency_graph` with `HIERNET_CLASS_FOLDER_MAP`, used by the 10-20 models
- `mnr_13leaf`: `hier_adjacency_graph`
- `whoi_s`
- `whoi_l`

`constants.py` keeps its dict versions for backward compatibility. A test asserts that each YAML matches its dict exactly, including key order.

### Training config (`configs/*.yaml`)

```yaml
hierarchy: mnr_17leaf            # bundled name or path to a YAML
model:   {backbone: swin_t, expert: cnn, loss: bce, freeze_backbone: false}
data:
  data_dir: /path/to/Hier-Zooplankton
  subdirectories: [IssacData]
  uncapped_subdirectories: [Quetico_training_data]
  exclude_log: Messy_background/moved_images/log.md   # relative to data_dir, optional
  format: .tif
  resolution: 64
  max_class_size: 12000
  split: [0.7, 0.1, 0.2]
  balanced_sampling: true
augment: {background_noise: true, noise_p: 0.8, hflip: true, vflip: true}
train:   {lr: 3.0e-4, max_epochs: 40, scheduler: cosine, patience: 10, delta: 0.001,
          batch_size: 64, num_workers: 8, seed: 666, device: auto}
out_dir: training_result/mnr     # run name: <backbone>_<expert>_<noise|baseline>_<MM-DD>
```

Bundled examples:
- `mnr_swin_t_cnn_noise.yaml` and `mnr_swin_t_cnn_baseline.yaml`, which reproduce the 10-20 runs
- a WHOI example

`runs/training_scripts/launch_*.sh` (screen sessions, GPU assignment) are specific to this machine, so they stay on `research`.

---

## 6. Packaging and publishing

- **`pyproject.toml`:**
  - setuptools, `name = "hierroute"`, `version = "0.1.0"`, `requires-python = ">=3.10"`
  - license MIT, the three authors
  - console script `hierroute = hierroute.cli:main`
  - the `hierarchies/*.yaml` files included as package data
- **Dependencies** (lower bounds only): torch, torchvision, numpy, pandas, pillow, scipy, scikit-learn, matplotlib, seaborn, pyyaml. `python-dotenv` is no longer needed.
- **`publish.yml`:**
  - triggers on GitHub Release tags `v*`
  - builds the sdist and wheel, then uploads with `pypa/gh-action-pypi-publish` under the `pypi` environment
  - trusted publishing, so no token is stored
- **First upload.** Before `0.1.0` goes to PyPI, a trial upload goes to TestPyPI.
- **The user's PyPI setup.** A pending publisher was created on 2026-11-08. A pending publisher does not reserve the name; the first real upload claims it. Its values:

| Field | Value |
|---|---|
| Project | `hierroute` |
| Owner | `atjhin` |
| Repository | `Hierroute` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

- **Fallback name.** If `hierroute` is taken on PyPI before `0.1.0`, publish as `hierroute-plankton`. The import name stays `hierroute`.

---

## 7. README outline

1. What it is (one paragraph plus the 17-leaf taxonomy tree). The current tree in the README is out of date.
2. Install: `pip install hierroute`, with a note on installing a specific CUDA or CPU build of torch first.
3. Quick start: three commands (`crop`, `evaluate`, `evaluate --labelled`).
4. Input layouts: a crop folder, labelled class folders, and raw FlowCam exports, each as a tree diagram.
5. Outputs: what each file and each `predictions.csv` column means, and how outcomes are graded.
6. Choosing a model:
   - `noise` (default) for messy or dirty backgrounds
   - `baseline` for clean backgrounds
   - accuracy figures from each model's `eval_results.json`
7. Training your own model: the config reference and how to define a new hierarchy.
8. Limitations:
   - images are processed as 64×64 greyscale
   - organisms outside the taxonomy are forced into some class
   - `Zoop-No` has no level 2 or 3
9. Offline use (`hierroute download`, `TORCH_HOME`), license, authors, citation.

---

## 8. Verification

| Check | How |
|---|---|
| Evaluate matches the old scripts | Run `runs/inference_scripts/evaluate_erie_test_data.py --exclude` and `hierroute evaluate Erie_test_data --labelled --exclude-runs _Decomp _LotsofSmalls` on the same 10-20 checkpoint. `predictions.csv` must match exactly (values within float tolerance), and the metric tables must match |
| FlowCam inference matches | Run `hierroute evaluate` on a few Quetico runs. The output must equal the rows in `/store/tjhinale/quetico_predictions/swin_t_cnn_noise_10-20/` |
| Crop matches the notebook | `hierroute crop` on a batch the notebook already processed must give the same file set with pixel-identical crops |
| Training config matches the 10-20 runs | Same sample list and the same train/val/test split (seed 666) as `train_swin_cnn_mnr.py`, plus a 1-epoch smoke run |
| Hierarchy YAML equals the dicts | Unit test, including key order |
| Released weights load | Load from the registry with `pretrained=False`; the checksum must match and there must be no ImageNet download |
| Clean install | Fresh venv, CPU only, `pip install` from the built wheel, then run all commands on the test fixtures |
| TestPyPI | `pip install -i https://test.pypi.org/simple/ hierroute` works end to end |

---

## 9. Order of work

- [x] 1. Snapshot the current working tree to the `research` branch (scripts and notebooks only; no results or weights)
- [x] 2. `main`: add `.gitignore`, commit the current `hierroute/` state, remove the research files
- [x] 3. Packaging: `pyproject.toml`, `LICENSE`, hierarchy YAMLs, `config.py`, `pretrained=` flag
- [x] 4. `flowcam.py` and `hierroute crop`
- [x] 5. `inference.py`, `plotting.py` and `hierroute evaluate`
- [x] 6. `hierroute train` and the example configs
- [x] 7. Tests and fixtures, plus `tests.yml`
- [x] 8. README
- [x] 9. Regression checks (section 8)
- [ ] 10. **User approval** → push `main` and `research`
- [ ] 11. Create the `models-v1` release with both weights; fill sha256 values into `pretrained.py`
- [ ] 12. TestPyPI trial → **user approval** → tag `v0.1.0` → PyPI publish

Nothing is pushed, released, or published without the user's explicit approval at steps 10–12.

## Out of scope for 0.1.0

- BCNN
- Uploading to Hugging Face
- Fine-tuning from a released backbone on a new hierarchy (possible later: `init_from: noise` in the training config)

---

## Implementation notes (2026-11-08)

### Verification results

| Check | Result |
|---|---|
| `crop` vs the HierNet notebook's Erie crops | All 53,517 crops in the same class folders, byte-identical (`diff -r`) |
| `evaluate --labelled` vs `runs/inference_scripts/evaluate_erie_test_data.py --exclude` (10-20 noise checkpoint) | `predictions.csv` identical in every column for all 43,907 rows. 5 of 7 PNGs are pixel-identical; the other 2 differ only in their (shortened) footnote |
| `evaluate --input-type flowcam` (raw Erie mosaics) vs crop input | Identical classes, levels and outcomes; confidences within 8e-5 (GPU batch numerics). 53k particles in 18 s on one RTX 4090 |
| `train` config vs `train_swin_cnn_mnr.py` | Identical train/val/test image lists, labels and order (50,205 / 7,173 / 14,345), identical augmentation and loader settings |
| Released weights via registry | Same outputs as the old `HierRouteNet(checkpoint_dir=...)` loader; no ImageNet download |
| Clean install | Fresh venv + CPU torch 2.15 + built wheel: `hierroute` command works from an empty folder; 130 tests pass |
| Test suite | 130 passed, 3 skipped (the skips were already there) |

### Differences from the plan

- `evaluate --input-type` defaults to `crops` instead of `auto`. A labelled test set such as
  Erie_test_data contains both class folders and FlowCam CSVs, so auto-detection would guess wrong.
  The tool suggests `--input-type flowcam` when it finds no crops but does find FlowCam CSVs.
- `--zoop-only-leaf-accuracy` became `--zoop-only`.
- The released-weights registry pins the bundled `mnr_17leaf` tree with a fingerprint, not a copy of
  it. If anyone edits or reorders that YAML file, loading the released models fails with a clear
  error, and a test catches it.
- `train` scores the held-out test split with the **best** checkpoint. The old script used the last
  epoch's weights.
- Added the `model.pretrained_backbone` training option, for offline training and fast tests.
- `publish.yml` also has a manual "Run workflow" button that uploads to TestPyPI. This needs a
  TestPyPI pending publisher with environment `testpypi`.
- Labelled crop input also maps each leaf name to itself (e.g. a `Zoop_Unknown` folder). The old
  scripts didn't; no Erie folder is affected.

### Remaining (needs the user)

1. Review `main` (`git log -p main`) and `research`.
2. Approve: push both branches, make `main` the default branch on GitHub, and create the `models-v1`
   release with `swin_t_cnn_noise_10-20.pt` and `swin_t_cnn_baseline_10-20.pt`.
3. On test.pypi.org, add the pending publisher (environment `testpypi`). Then run the publish
   workflow manually and test `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hierroute`.
4. On GitHub (repo Settings → Environments → `pypi`), add yourself as a required reviewer.
5. Approve: create GitHub release `v0.1.0`, which publishes to PyPI.
