# Lake Erie 2024 raw FlowCam exports → per-particle predictions

Goal: run the `swin_t_cnn_baseline_09-17` HierRouteNet over every particle in the raw Lake Erie
2024 FlowCam exports and write one CSV row per particle with the model's predictions and
confidences at the 3 hierarchy levels, plus the particle's FlowCam measurements.

This uses the same pipeline as [`../quetico_process/`](../quetico_process/PLAN.md) but as separate
scripts, so the Quetico scripts are unchanged:

- [`predict_erie_raw.py`](predict_erie_raw.py): predictions
- [`sanity_check_erie.py`](sanity_check_erie.py): completeness check against the raw folders

## Input

`/store/tjhinale/data_temp/Secure_Files_10_19_26/` (16 GB, 462 runs, 2,040,740 particles; every run
is `2024MMDD_Erie_<station>_2mm_RepN`, sampled April–June 2024). There are **no year folders**: the
top level is the "used in AI model" category.

| Folder (`subset`) | Runs | Relation to the data |
|---|---|---|
| `Used in AI Model` | 42 | 39 are in the training data (`/data/Hier-Zooplankton/CSV_Data`); 3 are `_Rerun` repeats of training runs |
| `Not Used in AI Model` | 404 | none in the training data |
| `Fully classifying to test AI model - No Duplicates/Good to go` | 16 | exactly the 16 runs of the Erie test set (`/data/Hier-Zooplankton/Erie_test_data`) |

Each run folder contains `<run>.csv` (55 FlowCam columns, no `Class` column), the mosaics
`<run>_NNNNNN.tif`, and `_bin.tif`, `cal_image_*.tif`, `.lst/.ctx/.edg/.cla/.txt` files, which are
ignored. The raw files are only read, never modified.

## Differences from the Quetico pipeline

- **Run discovery and year:** run folders are found at any depth (any folder holding a `.lst` file
  or a `<folder>.csv`). `year` comes from the run name's date prefix rather than from a folder.
  `subset` is the full folder path above the run.
- **CSV fallback:** a run folder without `<run>.csv` uses its only CSV whose `Image File` values name
  that run's own mosaics. Only `Not Used in AI Model/20240409_Erie_WMS0301_2mm_Rep3` needs this; it
  has `data_export.csv`.
- **Model:** `training_result/mnr/swin_t_cnn_baseline_09-17` (a symlink to
  `/store/tjhinale/Zooplankton_Classification-training_result/mnr/`).
- **Sanity check:** runs are keyed by folder path rather than run name, so a run filed under two
  categories counts as two run folders. Duplicate `image_file`s are broken down by whether they come
  from such double-filed runs.

Cropping, preprocessing, inference and output columns are identical to Quetico:
`image_file, year, subset, run, confidence_level_1/2/3, predicted_level_1/2/3, csv_*`.

## Output

`/store/tjhinale/erie_predictions/swin_t_cnn_baseline_09-17/`: `predictions_2024.csv`,
`run_summary.csv`, `skipped_particles.csv`, `sanity_check.csv`.

```bash
cd /home/tjhinale/repo/Zooplankton_Classification
/home/tjhinale/repo/HierNet/hiernet-venv/bin/python runs/erie_process/predict_erie_raw.py
/home/tjhinale/repo/HierNet/hiernet-venv/bin/python runs/erie_process/sanity_check_erie.py
```

## Run results (2026-10-24)

2,039,429 of 2,040,740 particles predicted (462 runs, about 4 minutes). The 1,311 that were
skipped are explained below.

**Sanity check (2024)**

| Check | On disk | In predictions |
|---|---|---|
| Run folders | 462 | 462 |
| Mosaic images | 10,666 | 10,665 (8 have no CSV rows, see below) |
| Particles | 2,040,740 rows in run CSVs | 2,040,740 rows, 2,039,429 with a prediction |
| Duplicate `image_file`s | — | 10,921, all explained below |

It is flagged FAIL only because of the data issues below. Every particle that exists in a
correct export was predicted.

**Parity with the existing Erie evaluation:** the 16 `Good to go` runs are the Erie test set. Matched
on `csv_original_reference_id` against
`runs/inference_results/erie/erie_test_data_swin_t_cnn_baseline_09-17/predictions.csv`, all 53,517
test-set particles have identical crop boxes and **100% identical predictions at all 3 levels**. The
confidences agree to within 1.2e-4, which is GPU floating-point variation with batch size.

**Data issues found:**
- **Wrong CSV in a run folder:** `Not Used in AI Model/20240416_Erie_CMS0202_2mm_Rep3/20240416_Erie_CMS0202_2mm_Rep3.csv`
  is byte-identical to Rep2's CSV (same checksum). Its 1,311 rows point to Rep2 mosaics that are not
  in the Rep3 folder, so they are skipped (`skipped_particles.csv`) and also duplicate Rep2's
  `image_file`s. Rep3's own 7 mosaics therefore have no particle table. Re-exporting Rep3 from
  FlowCam would fix this.
- **Runs filed in two folders:** `20240430_Erie_WMS0504_2mm_Rep3` (4,610 particles) and
  `20240516_Erie_CNS0406_2mm_Rep1` (5,000) are in both `Not Used in AI Model` and `Good to go`, with
  identical CSVs. Both copies are kept (9,610 duplicate rows) and can be told apart by `subset`; drop
  one before counting.
- **Mosaic with no CSV rows:** `Not Used in AI Model/20240611_Erie_WMS1010_2mm_Rep1` has 39 mosaics, but
  its CSV lists particles on only 38.
- `Used in AI Model` runs were used to train this model, so their predictions are optimistic.

## Rep3 fix (2026-10-26)

The data provider confirmed that the Rep2 export had been saved in the Rep3 folder by accident
(every FlowCam export is named `data_export.csv` and a script renamed them to their folder names).
They supplied the correct export, `/store/tjhinale/data_temp/20240416_Erie_CMS0202_2mm_Rep3.csv`:
1,275 particles on Rep3's own 7 mosaics, and none of its particles appear in Rep2. They also
confirmed that the WMS1010 Rep1 mosaic with no CSV rows contains only one large bubble, so it is fine
for it to be excluded.

Only this run was predicted. The raw folder and the scripts were not changed. `predict_erie_raw.py`
was run with `--data-root` pointing to a temporary folder holding links to the fixed CSV and Rep3's
7 mosaics, under `Not Used in AI Model/20240416_Erie_CMS0202_2mm_Rep3/`, so `subset` matches.

- **Separate output:** `/store/tjhinale/erie_predictions/rep3_fix/swin_t_cnn_baseline_09-17/`
  (`predictions_2024.csv` with 1,275 rows, `run_summary.csv`, `skipped_particles.csv`,
  `sanity_check.csv`). The sanity check passes: 1 run, 7/7 mosaics, 1,275/1,275 particles predicted,
  0 duplicates.
- **Merged into the main output:** in `swin_t_cnn_baseline_09-17/predictions_2024.csv`, the 1,311 wrong
  Rep3 rows were replaced in place by the 1,275 new rows, and all other rows are byte-identical. The
  file now has 2,040,704 rows, all predicted. The only duplicates left are the 9,610 rows from the
  double-filed runs. `run_summary.csv` has the corrected Rep3 line, and `skipped_particles.csv` is now
  empty. The pre-merge files are in `swin_t_cnn_baseline_09-17/backup_before_rep3_fix/`.
- The raw `Secure_Files_10_19_26/.../20240416_Erie_CMS0202_2mm_Rep3.csv` is still the Rep2 copy, and
  the main `sanity_check.csv` is from before the fix. Rerunning `sanity_check_erie.py` on the full
  folder would flag Rep3 until the raw CSV is replaced with the fixed one.
