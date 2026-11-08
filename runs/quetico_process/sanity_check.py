"""
Check that predict_quetico_raw.py ran every particle/mosaic image under each year folder, by
recounting directly from the raw data on disk -- independently of the prediction script's own
run discovery -- and comparing against its predictions_<year>.csv.

Usage: python runs/quetico_process/sanity_check.py [predictions_dir] [data_root]

Per year (and per run, for any mismatch), compares:
  run folders      folders holding mosaics on disk         vs  distinct 'run' values predicted
  mosaic images    <run>_NNNNNN.tif files on disk           vs  distinct mosaics predicted
                   (excluding _bin.tif masks and cal_image_*.tif calibration frames)
  particles        rows across every run's <run>.csv        vs  predictions rows
  predicted        --                                       vs  rows with a predicted_level_1
  duplicates       --                                       vs  repeated image_file values
A mosaic on disk can legitimately have no predictions if its run's CSV lists no particles on it
(e.g. an empty frame); those are listed separately rather than counted as failures.

Writes <predictions_dir>/sanity_check.csv (one row per year) and prints PASS/FAIL per year.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PREDICTIONS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/store/tjhinale/quetico_predictions/swin_t_cnn_noise_10-20')
DATA_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/store/tjhinale/data_temp/Secure_Files_10_20_26')
MOSAIC_RE = re.compile(r'_\d{6}\.tif$')


def disk_counts(year_dir):
    """Mosaics per run folder, and particle rows in each run folder's own <run>.csv."""
    mosaics, run_dirs = {}, {}
    for tif in year_dir.rglob('*.tif'):
        if MOSAIC_RE.search(tif.name) and not tif.name.startswith('cal_image'):
            mosaics.setdefault(tif.parent.name, set()).add(tif.name)
            run_dirs.setdefault(tif.parent.name, set()).add(tif.parent)
    particles = {}
    for run, dirs in run_dirs.items():
        csvs = [d / f'{run}.csv' for d in dirs if (d / f'{run}.csv').exists()]
        # None = mosaics with no <run>.csv to say where the particles are
        particles[run] = sum(len(pd.read_csv(c, usecols=['Image File'])) for c in csvs) if csvs else None
    return mosaics, particles


def check_year(year_dir, predictions_path):
    mosaics_disk, particles_disk = disk_counts(year_dir)
    preds = pd.read_csv(predictions_path, usecols=['image_file', 'run', 'predicted_level_1'],
                        dtype={'run': str}, keep_default_na=False, na_values={'predicted_level_1': ['']})
    preds['mosaic'] = preds['image_file'].str.rsplit('_', n=1).str[0] + '.tif'
    mosaics_pred = preds.groupby('run')['mosaic'].agg(set).to_dict()
    particles_pred = preds.groupby('run').size().to_dict()

    run_rows = []
    for run in sorted(set(mosaics_disk) | set(mosaics_pred)):
        on_disk, predicted = mosaics_disk.get(run, set()), mosaics_pred.get(run, set())
        run_rows.append({
            'run': run,
            'mosaics_disk': len(on_disk), 'mosaics_predicted': len(predicted),
            'mosaics_without_particles': len(on_disk - predicted),
            'mosaics_predicted_not_on_disk': len(predicted - on_disk),
            'particles_csv': particles_disk.get(run), 'particles_predicted': particles_pred.get(run, 0),
        })
    runs = pd.DataFrame(run_rows)

    no_csv = runs['particles_csv'].isna()
    particle_mismatch = runs[no_csv | (runs['particles_csv'] != runs['particles_predicted'])]
    phantom_mosaics = runs[runs['mosaics_predicted_not_on_disk'] > 0]

    summary = {
        'year': year_dir.name,
        'run_folders_disk': len(mosaics_disk),
        'runs_predicted': preds['run'].nunique(),
        'mosaic_images_disk': int(runs['mosaics_disk'].sum()),
        'mosaic_images_predicted': int(runs['mosaics_predicted'].sum()),
        'mosaic_images_without_particles': int(runs['mosaics_without_particles'].sum()),
        'particles_in_csvs': int(runs['particles_csv'].fillna(0).sum()),
        'prediction_rows': len(preds),
        'rows_with_prediction': int(preds['predicted_level_1'].notna().sum()),
        'duplicate_image_files': int(preds['image_file'].duplicated().sum()),
    }
    summary['pass'] = (
        summary['run_folders_disk'] == summary['runs_predicted']
        and summary['particles_in_csvs'] == summary['prediction_rows'] == summary['rows_with_prediction']
        and summary['duplicate_image_files'] == 0
        and particle_mismatch.empty and phantom_mosaics.empty
    )
    return summary, particle_mismatch, phantom_mosaics, runs[runs['mosaics_without_particles'] > 0]


def main():
    results = []
    for year_dir in sorted(d for d in DATA_ROOT.iterdir() if d.is_dir()):
        predictions_path = PREDICTIONS_DIR / f'predictions_{year_dir.name}.csv'
        print(f'\n===== {year_dir.name} =====')
        if not predictions_path.exists():
            print(f'FAIL: {predictions_path} not found')
            results.append({'year': year_dir.name, 'pass': False})
            continue
        summary, particle_mismatch, phantom_mosaics, empty_mosaic_runs = check_year(year_dir, predictions_path)
        results.append(summary)
        for key, value in summary.items():
            if key not in ('year', 'pass'):
                print(f'  {key:34s} {value:>10,}')
        if not particle_mismatch.empty:
            print('  Runs whose predicted particle count differs from their CSV (particles_csv blank = no CSV):')
            print(particle_mismatch.to_string(index=False))
        if not phantom_mosaics.empty:
            print('  Runs with predictions for mosaics not on disk:')
            print(phantom_mosaics.to_string(index=False))
        if not empty_mosaic_runs.empty:
            print(f'  {len(empty_mosaic_runs)} run(s) have mosaics on disk that their CSV lists no particles on '
                  f'(expected for empty frames, not a failure).')
        print(f'  {"PASS" if summary["pass"] else "FAIL"}')

    table = pd.DataFrame(results)
    out_path = PREDICTIONS_DIR / 'sanity_check.csv'
    table.to_csv(out_path, index=False)
    print(f'\nSaved -> {out_path}')
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
