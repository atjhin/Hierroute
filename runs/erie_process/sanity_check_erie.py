"""
Check that predict_erie_raw.py ran every particle/mosaic image in the raw Erie folders, by
recounting directly from the raw data on disk -- independently of the prediction script's own
run discovery -- and comparing against its predictions_<year>.csv. Same checks as
runs/quetico_process/sanity_check.py; runs are grouped into years by their name's date prefix
(this dataset has no year folders) and keyed by folder path, so a run filed under two
categories counts as two run folders.

Usage: python runs/erie_process/sanity_check_erie.py [predictions_dir] [data_root]

Per year (and per run, for any mismatch), compares:
  run folders      folders holding mosaics on disk         vs  distinct subset/run predicted
  mosaic images    <run>_NNNNNN.tif files on disk           vs  distinct mosaics predicted
                   (excluding _bin.tif masks and cal_image_*.tif calibration frames)
  particles        rows across every run's particle CSV     vs  predictions rows
  predicted        --                                       vs  rows with a predicted_level_1
  duplicates       --                                       vs  repeated image_file values
A run's particle CSV is <run>.csv, or else any CSV in the folder with an 'Image File' column.
A mosaic on disk can legitimately have no predictions if its run's CSV lists no particles on it
(e.g. an empty frame); those are listed separately rather than counted as failures. Duplicate
image_files are broken down by whether they come from a run name filed in more than one folder.

Writes <predictions_dir>/sanity_check.csv (one row per year) and prints PASS/FAIL per year.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PREDICTIONS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/store/tjhinale/erie_predictions/swin_t_cnn_baseline_09-17')
DATA_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/store/tjhinale/data_temp/Secure_Files_10_19_26')
MOSAIC_RE = re.compile(r'_\d{6}\.tif$')
YEAR_RE = re.compile(r'^(\d{4})\d{4}_')


def particle_csvs(run_dir):
    own = run_dir / f'{run_dir.name}.csv'
    if own.exists():
        return [own]
    return [c for c in run_dir.glob('*.csv') if 'Image File' in pd.read_csv(c, nrows=0).columns]


def disk_counts():
    """Per run folder (keyed 'subset/run', as in the predictions): mosaic names, particle rows."""
    mosaics, run_dirs = {}, {}
    for tif in DATA_ROOT.rglob('*.tif'):
        if MOSAIC_RE.search(tif.name) and not tif.name.startswith('cal_image'):
            key = str(tif.parent.relative_to(DATA_ROOT))
            mosaics.setdefault(key, set()).add(tif.name)
            run_dirs[key] = tif.parent
    particles = {}
    for key, run_dir in run_dirs.items():
        csvs = particle_csvs(run_dir)
        # None = mosaics with no particle CSV to say where the particles are
        particles[key] = sum(len(pd.read_csv(c, usecols=['Image File'])) for c in csvs) if csvs else None
    return mosaics, particles


def check_year(year, mosaics_disk, particles_disk, predictions_path):
    preds = pd.read_csv(predictions_path, usecols=['image_file', 'subset', 'run', 'predicted_level_1'],
                        dtype={'subset': str, 'run': str}, keep_default_na=False,
                        na_values={'predicted_level_1': ['']})
    preds['key'] = (preds['subset'] + '/' + preds['run']).str.lstrip('/')
    preds['mosaic'] = preds['image_file'].str.rsplit('_', n=1).str[0] + '.tif'
    mosaics_pred = preds.groupby('key')['mosaic'].agg(set).to_dict()
    particles_pred = preds.groupby('key').size().to_dict()

    run_rows = []
    for key in sorted(set(mosaics_disk) | set(mosaics_pred)):
        on_disk, predicted = mosaics_disk.get(key, set()), mosaics_pred.get(key, set())
        run_rows.append({
            'run_folder': key,
            'mosaics_disk': len(on_disk), 'mosaics_predicted': len(predicted),
            'mosaics_without_particles': len(on_disk - predicted),
            'mosaics_predicted_not_on_disk': len(predicted - on_disk),
            'particles_csv': particles_disk.get(key), 'particles_predicted': particles_pred.get(key, 0),
        })
    runs = pd.DataFrame(run_rows)

    no_csv = runs['particles_csv'].isna()
    particle_mismatch = runs[no_csv | (runs['particles_csv'] != runs['particles_predicted'])]
    phantom_mosaics = runs[runs['mosaics_predicted_not_on_disk'] > 0]

    double_filed = preds.groupby('run')['subset'].nunique()
    double_filed = set(double_filed[double_filed > 1].index)
    duplicated = preds['image_file'].duplicated()

    summary = {
        'year': year,
        'run_folders_disk': len(runs[runs['mosaics_disk'] > 0]),
        'runs_predicted': preds['key'].nunique(),
        'mosaic_images_disk': int(runs['mosaics_disk'].sum()),
        'mosaic_images_predicted': int(runs['mosaics_predicted'].sum()),
        'mosaic_images_without_particles': int(runs['mosaics_without_particles'].sum()),
        'particles_in_csvs': int(runs['particles_csv'].fillna(0).sum()),
        'prediction_rows': len(preds),
        'rows_with_prediction': int(preds['predicted_level_1'].notna().sum()),
        'duplicate_image_files': int(duplicated.sum()),
        'duplicates_from_double_filed_runs': int((duplicated & preds['run'].isin(double_filed)).sum()),
    }
    summary['pass'] = (
        summary['run_folders_disk'] == summary['runs_predicted']
        and summary['particles_in_csvs'] == summary['prediction_rows'] == summary['rows_with_prediction']
        and summary['duplicate_image_files'] == 0
        and particle_mismatch.empty and phantom_mosaics.empty
    )
    return summary, particle_mismatch, phantom_mosaics, runs[runs['mosaics_without_particles'] > 0], \
        preds[preds['run'].isin(double_filed)].groupby(['run', 'subset']).size()


def main():
    mosaics_all, particles_all = disk_counts()
    years = sorted({YEAR_RE.match(Path(key).name).group(1) for key in mosaics_all})

    results = []
    for year in years:
        in_year = lambda key: YEAR_RE.match(Path(key).name).group(1) == year  # noqa: E731
        mosaics_disk = {k: v for k, v in mosaics_all.items() if in_year(k)}
        particles_disk = {k: v for k, v in particles_all.items() if in_year(k)}
        predictions_path = PREDICTIONS_DIR / f'predictions_{year}.csv'
        print(f'\n===== {year} =====')
        if not predictions_path.exists():
            print(f'FAIL: {predictions_path} not found')
            results.append({'year': year, 'pass': False})
            continue
        summary, particle_mismatch, phantom_mosaics, empty_mosaic_runs, double_filed = check_year(
            year, mosaics_disk, particles_disk, predictions_path)
        results.append(summary)
        for key, value in summary.items():
            if key not in ('year', 'pass'):
                print(f'  {key:34s} {value:>10,}')
        if not particle_mismatch.empty:
            print('  Run folders whose predicted particle count differs from their CSV (particles_csv blank = no CSV):')
            print(particle_mismatch.to_string(index=False))
        if not phantom_mosaics.empty:
            print('  Run folders with predictions for mosaics not on disk:')
            print(phantom_mosaics.to_string(index=False))
        if not empty_mosaic_runs.empty:
            print('  Run folders with mosaics on disk that their CSV lists no particles on '
                  '(expected for empty frames, not a failure):')
            print(empty_mosaic_runs[['run_folder', 'mosaics_disk', 'mosaics_predicted']].to_string(index=False))
        if len(double_filed):
            print('  Run names filed under more than one folder (rows per copy):')
            print(double_filed.to_string())
        print(f'  {"PASS" if summary["pass"] else "FAIL"}')

    table = pd.DataFrame(results)
    out_path = PREDICTIONS_DIR / 'sanity_check.csv'
    table.to_csv(out_path, index=False)
    print(f'\nSaved -> {out_path}')
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
