"""
Run a trained HierRouteNet checkpoint over the raw (unlabelled) Lake Erie 2024 FlowCam exports
and write one predictions row per particle -- see PLAN.md in this directory. Same pipeline as
runs/quetico_process/predict_quetico_raw.py, adapted to this dataset's layout.

Input layout (no year folders -- the top level is the "used in AI model" category):
  <data_root>/<category>[/<sub>...]/<run>/<run>.csv          one row per particle
  <data_root>/<category>[/<sub>...]/<run>/<run>_NNNNNN.tif   mosaics the CSV's 'Image File' names
A run folder without <run>.csv falls back to its only CSV whose 'Image File' values name that
run's own mosaics (e.g. 20240409_Erie_WMS0301_2mm_Rep3/data_export.csv).

Each particle is cropped in memory from its mosaic using the CSV's Image X/Y/Width/Height
(the same crop box as HierNet's Zooplankton-MNR/pre_process_data.ipynb) and passed straight
to the model -- no crop files are written. Preprocessing matches the test-data evaluation
scripts exactly: greyscale -> Resize((RESOLUTION, RESOLUTION)) -> ToTensor -> repeat to 3
channels (inference_common.EVAL_TRANSFORMS / LabeledFolderDataset).

Outputs, under <out_dir>/<checkpoint name>/:
  predictions_<year>.csv    image_file, year, subset, run, confidence_level_1/2/3,
                            predicted_level_1/2/3, csv_* metadata
  run_summary.csv           per run: particles in its CSV, predicted, skipped, out-of-bounds boxes
  skipped_particles.csv     particles with no prediction (missing mosaic / zero-size box) and why

year comes from the run name's date prefix ('20240514_Erie_...' -> '2024'); subset is the folder
path between data_root and the run (e.g. 'Not Used in AI Model'). image_file
('<Image File stem>_<Particle ID>.tif') and the csv_* columns use the same naming as
evaluate_erie_test_data.py's predictions.csv. predicted_level_d is the raw greedy cascade's node
at depth d, blank past where that particle's path stopped (e.g. Zoop-No has no level 2/3);
confidence_level_d is the local softmax confidence of the decision that chose it.

Particle IDs are renumbered per export, so the same particle can have a different image_file here
than in the manually-classified test set's crops. Join across the two on
csv_original_reference_id (FlowCam's per-particle UUID) instead.

Usage:
  python runs/erie_process/predict_erie_raw.py [--limit-runs N] [--data-root DIR]
         [--checkpoint DIR] [--out-dir DIR] [--batch-size N] [--num-workers N] [--overwrite]
  A year whose predictions_<year>.csv already exists is skipped unless --overwrite is passed.
  --limit-runs N only processes the first N runs per year (for a quick test) and writes under
  <out_dir>/<checkpoint name>_test/ instead.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'inference_scripts'))
# inference_common puts the repo root on sys.path and loads .env, so it must come before hierroute.
from inference_common import EVAL_TRANSFORMS, build_hierarchy, detect_backbone, load_model, _csv_column_to_snake_case
from hiernet_eval import predict_with_confidences
from hierroute.constants import hiernet_adjacency_graph

DEFAULT_DATA_ROOT = Path('/store/tjhinale/data_temp/Secure_Files_10_19_26')
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent.parent / 'training_result' / 'mnr' / 'swin_t_cnn_baseline_09-17'
# Outputs total a few GB -- /store, since / is nearly full on fish.
DEFAULT_OUT_DIR = Path('/store/tjhinale/erie_predictions')

# CSV_Data columns used as join keys / crop boxes. Particle ID and Image File are folded into
# image_file and not repeated as csv_* columns, matching build_csv_metadata_index.
REQUIRED_COLUMNS = ('Particle ID', 'Image File', 'Image X', 'Image Y', 'Image Width', 'Image Height')
LEVELS = (1, 2, 3)
RUN_DATE_RE = re.compile(r'^(\d{4})\d{4}_')


# ---------------------------------------------------------------------------
# Run discovery + per-run particle table
# ---------------------------------------------------------------------------
def run_csv_for(run_dir):
    """<run>/<run>.csv if present, else the folder's only CSV whose 'Image File' values name this
    run's own mosaics (e.g. a generic 'data_export.csv'), else None."""
    own = run_dir / f'{run_dir.name}.csv'
    if own.exists():
        return own
    candidates = []
    for csv_path in sorted(run_dir.glob('*.csv')):
        try:
            image_files = pd.read_csv(csv_path, usecols=['Image File'], nrows=50)['Image File'].astype(str)
        except ValueError:  # no 'Image File' column -- a summary, not a particle export
            continue
        if len(image_files) and image_files.str.startswith(run_dir.name + '_').all():
            candidates.append(csv_path)
    return candidates[0] if len(candidates) == 1 else None


def find_runs(data_root):
    """{year: [run CSV paths]} for every run folder (any folder holding a .lst run file or a
    <folder>.csv) under data_root, grouped by the run name's date prefix."""
    run_dirs = {p.parent for p in data_root.rglob('*.lst')} | {
        c.parent for c in data_root.rglob('*.csv') if c.stem == c.parent.name
    }
    runs_by_year = {}
    for run_dir in sorted(run_dirs):
        csv_path = run_csv_for(run_dir)
        if csv_path is None:
            print(f'WARNING: run folder with no usable particle CSV, skipped: {run_dir}')
            continue
        if csv_path.stem != run_dir.name:
            print(f'Note: {run_dir.relative_to(data_root)} uses {csv_path.name} as its particle CSV')
        match = RUN_DATE_RE.match(run_dir.name)
        if match is None:
            raise ValueError(f'cannot read a year from run folder name: {run_dir}')
        runs_by_year.setdefault(match.group(1), []).append(csv_path)
    return runs_by_year


def has_unquoted_commas(csv_path):
    """True if any data row has more fields than the header has (non-empty) column names."""
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        n_names = len([h for h in header if h != ''])
        return any(len(row) > n_names for row in reader)


def read_csv_with_unquoted_commas(csv_path):
    """FlowCam writes 'Image File' unquoted, so a run named with a comma splits that value across
    two fields and shifts every later column right by one (seen in Quetico 2025). Rejoin the
    surplus fields back into 'Image File' so every row lines up with the header."""
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        while header and header[-1] == '':
            header.pop()
        image_file_idx = header.index('Image File')
        rows = []
        for row in reader:
            extra = len(row) - len(header)
            if extra > 0:
                row = row[:image_file_idx] + [','.join(row[image_file_idx:image_file_idx + extra + 1])] \
                    + row[image_file_idx + extra + 1:]
            rows.append(row)
    df = pd.DataFrame(rows, columns=header)
    # Back to pandas' usual numeric parsing for every column read as text above.
    return pd.read_csv(io.StringIO(df.to_csv(index=False)))


def load_run(csv_path, data_root):
    """One run's CSV -> its particle rows, with image_file/year/subset/run and csv_* columns."""
    df = read_csv_with_unquoted_commas(csv_path) if has_unquoted_commas(csv_path) else pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]  # trailing-comma exports
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'{csv_path} is missing required column(s) {missing}')
    # Catch any other column misalignment before it silently mislabels particles.
    bad = ~df['Image File'].astype(str).str.fullmatch(r'.+_\d{6}\.tif')
    if bad.any():
        raise ValueError(f'{csv_path}: {bad.sum()} rows have a malformed Image File, e.g. {df.loc[bad, "Image File"].iloc[0]!r}')

    run_dir = csv_path.parent
    rel_parts = run_dir.relative_to(data_root).parts
    out = pd.DataFrame({
        'image_file': df['Image File'].str.removesuffix('.tif') + '_' + df['Particle ID'].astype(str) + '.tif',
        'year': RUN_DATE_RE.match(run_dir.name).group(1),
        'subset': '/'.join(rel_parts[:-1]),
        'run': run_dir.name,
    })
    metadata = df.drop(columns=['Particle ID', 'Image File']).rename(columns=_csv_column_to_snake_case)
    out = pd.concat([out, metadata], axis=1)

    boxes = df[['Image X', 'Image Y', 'Image Width', 'Image Height']].to_numpy(dtype=np.int64)
    mosaic_paths = [str(run_dir / name) for name in df['Image File']]
    return out, boxes, mosaic_paths


# ---------------------------------------------------------------------------
# Dataset: one item per mosaic -> every particle cropped from it
# ---------------------------------------------------------------------------
class MosaicCropDataset(Dataset):
    """items[i] = (mosaic_path, row_indices, boxes) for every particle on that mosaic. Opening
    each mosaic once and cropping all its particles is far cheaper than one open per particle."""

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        mosaic_path, row_indices, boxes = self.items[idx]
        try:
            mosaic = Image.open(mosaic_path).convert('L')
        except (FileNotFoundError, OSError):
            return None, row_indices, None
        width, height = mosaic.size
        crops = [self.transform(mosaic.crop((x, y, x + w, y + h))) for x, y, w, h in boxes]
        # PIL zero-pads a box past the mosaic edge (same as pre_process_data.ipynb); flag it.
        out_of_bounds = (boxes[:, 0] + boxes[:, 2] > width) | (boxes[:, 1] + boxes[:, 3] > height)
        return torch.stack(crops).repeat(1, 3, 1, 1), row_indices, out_of_bounds


def collate_mosaics(batch):
    loaded = [b for b in batch if b[0] is not None]
    missing = [b[1] for b in batch if b[0] is None]
    if not loaded:
        return None, None, None, missing
    images = torch.cat([b[0] for b in loaded])
    row_indices = np.concatenate([b[1] for b in loaded])
    out_of_bounds = np.concatenate([b[2] for b in loaded])
    return images, row_indices, out_of_bounds, missing


# ---------------------------------------------------------------------------
# One year: load every run, crop + predict, write predictions_<year>.csv
# ---------------------------------------------------------------------------
def process_year(year, run_csvs, data_root, model, id_to_name, device, args, out_dir):
    print(f'\n===== {year}: {len(run_csvs)} runs =====')

    tables, all_boxes, all_mosaics = [], [], []
    for csv_path in run_csvs:
        table, boxes, mosaics = load_run(csv_path, data_root)
        tables.append(table)
        all_boxes.append(boxes)
        all_mosaics.extend(mosaics)
    particles = pd.concat(tables, ignore_index=True)
    boxes = np.concatenate(all_boxes)
    mosaics = np.array(all_mosaics)
    n = len(particles)
    print(f'{n} particles across {len(run_csvs)} runs')

    skip_reason = np.full(n, None, dtype=object)
    empty_box = (boxes[:, 2] <= 0) | (boxes[:, 3] <= 0)
    skip_reason[empty_box] = 'zero-size box'

    items = []
    keep = np.flatnonzero(~empty_box)
    for mosaic_path, group in pd.Series(keep).groupby(mosaics[keep]):
        row_indices = group.to_numpy()
        items.append((mosaic_path, row_indices, boxes[row_indices]))

    loader = DataLoader(
        MosaicCropDataset(items, EVAL_TRANSFORMS), batch_size=args.mosaics_per_batch, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_mosaics,
    )

    predicted = {d: np.full(n, None, dtype=object) for d in LEVELS}
    confidence = {d: np.full(n, np.nan) for d in LEVELS}
    out_of_bounds = np.zeros(n, dtype=bool)
    done, start = 0, time.time()
    with torch.inference_mode():
        for images, row_indices, oob, missing in loader:
            for missing_rows in missing:
                skip_reason[missing_rows] = 'mosaic missing or unreadable'
            if images is None:
                continue
            out_of_bounds[row_indices] = oob
            for chunk_start in range(0, len(images), args.batch_size):
                chunk = images[chunk_start:chunk_start + args.batch_size].to(device, non_blocking=True)
                chunk_rows = row_indices[chunk_start:chunk_start + args.batch_size]
                paths, confs = predict_with_confidences(model, chunk)
                for row, path, conf in zip(chunk_rows, paths, confs):
                    for d in LEVELS:
                        if d < len(path):
                            predicted[d][row] = id_to_name[path[d]]
                            confidence[d][row] = conf[d - 1]
            done += len(images)
            if done // 50_000 != (done - len(images)) // 50_000:
                print(f'  {done}/{n} particles ({done / (time.time() - start):.0f}/s)')

    print(f'Predicted {done}/{n} particles in {time.time() - start:.0f}s')

    insert_at = particles.columns.get_loc('run') + 1
    for d in LEVELS:
        particles.insert(insert_at + d - 1, f'confidence_level_{d}', confidence[d])
    for d in LEVELS:
        particles.insert(insert_at + len(LEVELS) + d - 1, f'predicted_level_{d}', predicted[d])

    is_skipped = pd.notna(skip_reason)
    skipped = particles.loc[is_skipped, ['image_file', 'year', 'subset', 'run']].copy()
    skipped['reason'] = skip_reason[is_skipped]

    summary = pd.DataFrame({
        'year': particles['year'], 'subset': particles['subset'], 'run': particles['run'],
        'predicted': particles['predicted_level_1'].notna(), 'out_of_bounds_box': out_of_bounds,
    }).groupby(['year', 'subset', 'run'], sort=False).agg(
        particles=('predicted', 'size'), predicted=('predicted', 'sum'), out_of_bounds_box=('out_of_bounds_box', 'sum'),
    ).reset_index()
    summary['skipped'] = summary['particles'] - summary['predicted']

    predictions_path = out_dir / f'predictions_{year}.csv'
    particles.to_csv(predictions_path, index=False)
    print(f'Saved -> {predictions_path}')
    print(particles['predicted_level_1'].value_counts(dropna=False).to_string())
    if len(skipped):
        print(f'WARNING: {len(skipped)} particles skipped: {skipped["reason"].value_counts().to_dict()}')
    if summary['out_of_bounds_box'].sum():
        print(f'Note: {summary["out_of_bounds_box"].sum()} crop boxes extend past their mosaic (zero-padded)')
    return summary, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--limit-runs', type=int, default=0, help='first N runs per year only (test mode)')
    parser.add_argument('--batch-size', type=int, default=1024, help='crops per GPU forward pass')
    parser.add_argument('--mosaics-per-batch', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    out_dir = args.out_dir / (args.checkpoint.name + ('_test' if args.limit_runs else ''))
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_by_year = find_runs(args.data_root)
    if args.limit_runs:
        runs_by_year = {year: runs[:args.limit_runs] for year, runs in runs_by_year.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = detect_backbone(args.checkpoint)
    print(f'Using device: {device} | backbone: {backbone} | checkpoint: {args.checkpoint}')
    hierarchy, label_to_ids = build_hierarchy(hiernet_adjacency_graph)
    id_to_name = {v: k for k, v in label_to_ids.items()}
    model = load_model(hierarchy, label_to_ids, backbone, args.checkpoint, device, expert_type='cnn')

    summaries, skipped_all, done_years = [], [], set()
    for year, run_csvs in sorted(runs_by_year.items()):
        if (out_dir / f'predictions_{year}.csv').exists() and not args.overwrite:
            print(f'\n{year}: predictions_{year}.csv exists, skipping (--overwrite to redo)')
            continue
        summary, skipped = process_year(year, run_csvs, args.data_root, model, id_to_name, device, args, out_dir)
        summaries.append(summary)
        skipped_all.append(skipped)
        done_years.add(year)

    if summaries:
        # Merge with any earlier years' summaries so a resumed run still ends with a complete file.
        summary_path, skipped_path = out_dir / 'run_summary.csv', out_dir / 'skipped_particles.csv'
        for path, frames in ((summary_path, summaries), (skipped_path, skipped_all)):
            if path.exists():
                previous = pd.read_csv(path, keep_default_na=False, dtype={'year': str})
                frames.insert(0, previous[~previous['year'].isin(done_years)])
            pd.concat(frames, ignore_index=True).to_csv(path, index=False)
            print(f'Saved -> {path}')

    print(f'\nAll outputs saved under {out_dir}')


if __name__ == '__main__':
    main()
