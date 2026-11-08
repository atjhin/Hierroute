# Quetico raw FlowCam exports → per-particle predictions

Goal: run the `swin_t_cnn_noise_10-20` HierRouteNet over every particle in the raw Quetico
FlowCam exports (2020–2025) and write one CSV row per particle with the model's predictions at
the 3 hierarchy levels, plus the particle's FlowCam measurements.

Scripts: [`predict_quetico_raw.py`](predict_quetico_raw.py) (predictions) and
[`sanity_check.py`](sanity_check.py) (per-year completeness check)

## Input

`/store/tjhinale/data_temp/Secure_Files_10_20_26/` (copied from the shared drive, ~24 GB).

| Year | Layout | Runs |
|---|---|---|
| 2020QUETICO | run folders directly under the year (named `..._data`) | 24 |
| 2022QUETICO | run folders directly under the year | 146 |
| 2023QUETICO | run folders directly under the year | 259 |
| 2024Quetico | `Already classified/` · `Not already classified/` | 7 · 135 |
| 2025QUETICO | `Used in AI model/` · `Not used in AI model/{Already Classified, Not already classified, Not already classified/Used for select tests}` | 46 · 5 · 32 · 22 |

676 runs and about 2.19M particles in total. Each run folder contains:

- `<run>.csv`: the FlowCam export, one row per particle (55 columns: `Particle ID`, `Image File`,
  `Image X/Y/Width/Height`, size/shape measurements, timestamps, and so on). **It has no `Class`
  column**, because these are unlabelled exports.
- `<run>_NNNNNN.tif`: mosaics (RGB, 1920×1200) that the CSV's `Image File` points to.
- Ignored: `_bin.tif` masks, `cal_image_*.tif`, `.lst/.ctx/.edg/.cla/.txt`, and the year-level `.xlsx`
  and summary CSVs (`summary_export.csv`, `FlowCam_run_summary_extracted.csv`).

The existing CSVs are **read only and never modified**. All output goes to new files.

## Pipeline

1. **Discover runs.** Search each year folder at any depth for `<run>/<run>.csv`, so the CSV
   name must match its folder. This skips the stray summary CSVs and handles every year's layout.
   `year` is the top folder, `subset` is the path between the year and the run (blank for
   2020–2023), and `run` is the folder name.
2. **Build the particle table** from each run's CSV:
   - `image_file = <Image File stem>_<Particle ID>.tif` (same convention as the evaluation scripts)
   - every other CSV column becomes `csv_<snake_case>` (same renaming as
     `inference_common._csv_column_to_snake_case`); `Particle ID`/`Image File` are folded into `image_file`
   - trailing-comma exports (an empty `Unnamed` column) are dropped
3. **Crop.** The crop box is the same as in HierNet `Zooplankton-MNR/pre_process_data.ipynb`:
   `(Image X, Image Y, Image X + Image Width, Image Y + Image Height)` on the mosaic named in
   `Image File`. Unlike the notebook, crops are **made in memory and not saved**, because writing
   about 2.2M tiny tif files would be slow and pointless. Particles are grouped by mosaic, so each
   mosaic is opened once (in a DataLoader worker) and all of its particles are cropped from it.
4. **Preprocess.** This is identical to the evaluation scripts (`EVAL_TRANSFORMS` +
   `LabeledFolderDataset`): convert to greyscale, resize to 64×64, convert to a tensor, and repeat
   to 3 channels.
5. **Inference.** `HierRouteNet` (swin_t backbone, CNN experts) loads
   `training_result/swin_t_cnn_noise_10-20/best_model.pt`. `hiernet_eval.predict_with_confidences`
   walks greedily from the root to a leaf, and `predicted_level_d` is the node at depth d on that
   path, and `confidence_level_d` is the softmax confidence of the decision that chose it. Both
   are left blank past where the path stopped (for example `Zoop-No` has no level 2/3).
   There is no confidence gating, so this matches the `predicted_level_*`/`confidence_level_*` columns of the reference
   `predictions.csv`. Batches hold 1024 crops on one RTX 4090.
6. **Write outputs**, one year at a time to keep memory bounded.

## Output

`/store/tjhinale/quetico_predictions/swin_t_cnn_noise_10-20/`. This is on `/store` because the
outputs total a few GB and `/` is nearly full.

| File | Contents |
|---|---|
| `predictions_<year>.csv` | one row per particle: `image_file, year, subset, run, confidence_level_1/2/3, predicted_level_1/2/3, csv_*`. `subset` is the folder between the year and the run (e.g. `Used in AI model`), blank where there is none |
| `run_summary.csv` | per run: particles in its CSV, predicted, skipped, crop boxes past the mosaic edge |
| `skipped_particles.csv` | particles with no prediction and the reason (missing/unreadable mosaic, zero-size box) |

Columns dropped compared with the reference `predictions.csv`: `actual_class`, `in_taxonomy`, and
`correct_*`/`incorrect_*`, because there are no labels, and `predicted_class_1/2`. There is no
combined all-years file.

## Validation

- **Completeness (`sanity_check.py`):** for each year, it recounts from the raw folders, independently
  of the prediction script: run folders, mosaic images (`<run>_NNNNNN.tif`, excluding `_bin` and
  `cal_image`), and particle rows across all run CSVs. These are compared against
  `predictions_<year>.csv`, along with duplicate and blank-prediction checks. Results are written to
  `sanity_check.csv`. `run_summary.csv` also gives per-run particle counts.
- **Parity with the existing evaluation:** 15 runs also appear in the manually classified Quetico
  test set (`evaluate_quetico_test_data.py`). Matched on `csv_original_reference_id`, 9,595 of
  their particles have identical crop boxes and **100% identical predictions at all 3 levels**,
  which confirms the in-memory cropping and preprocessing reproduce the file-based pipeline exactly.
- **Caveat:** `Particle ID` is renumbered per export, so the same particle has a different
  `image_file` here than in the test set's crops. Always join across datasets on
  `csv_original_reference_id`, never on `image_file`.

## Notes

- `2025QUETICO/Used in AI model` runs were used to train this model, so their predictions are
  optimistic. They are included but can be filtered out with `subset`.
- `.cla` files (in 108 runs) are VisualSpreadsheet class lists (class name → particle IDs). They
  could later supply `actual_class` for the "Already classified" runs; that is not done here.
- Run with `--limit-runs N` for a quick test (writes to `..._test/`). A year whose
  `predictions_<year>.csv` already exists is skipped unless `--overwrite` is passed, so an
  interrupted run can be restarted.

```bash
cd /home/tjhinale/repo/Zooplankton_Classification
/home/tjhinale/repo/HierNet/hiernet-venv/bin/python runs/quetico_process/predict_quetico_raw.py
```

## Run results (2026-10-22)

All 676 runs and 2,193,620 particles were predicted, with none skipped. `sanity_check.py` passes
for 2020, 2022, 2023 and 2024. 2025 is flagged only because of a duplicated folder (below).

Data issues found:
- **Comma-named runs** (`2025QUETICO/Used in AI model/20250709_Argo_Vertical_2mm_rep{1,2,3},`): FlowCam
  wrote the `Image File` value unquoted, so the comma splits it. `load_run` detects rows with too many
  fields and rejoins them, and it raises an error on any malformed `Image File`.
- **Duplicate run:** `2025QUETICO/Not used in AI model/Not already classified/20251015_Pickerel_Vertical_2mm_rep2 - Copy`
  is a copy of `Used in AI model/20251015_Pickerel_Vertical_2mm_rep2` (the same 1,694 particles, matched
  by `csv_original_reference_id`). Both are kept and can be told apart by `run`/`subset`; drop one before counting.
- **Empty run:** `2023QUETICO/20230610_Burt_Horizontal_2mm_rep1` has a header-only CSV and no mosaics (0 particles).
- **Mosaics with no CSV rows:** `2024Quetico/Not already classified/20240727_Poohbah_Vertical_2mm_rep3` has
  10 mosaics on disk, but its CSV (307 particles) only lists `_000001` and `_000002`. The particles on
  mosaics 3–10 are not in the export, so they cannot be cropped.
