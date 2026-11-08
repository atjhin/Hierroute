"""
Build two comparison tables from Messy_background eval_results.json files
(the *_messy_eval directories, not each run's own in-distribution
training_result/.../eval_results.json):

  1. Leaf-class accuracy table: rows = every class in HIERNET_CLASSES (the 12
     true species, each read from level 3 -- fine -- since that's their real
     classification decision; plus 'Zoop-No', which has no level-3 identity
     of its own -- it's a coarse catch-all bucket with no finer sub-
     classification -- so it's read from level 1 instead, its real decision
     ("is this even a zooplankton"). See the note in
     evaluate_bcnn_messy_background.py for why BCNN needs masking to make
     this comparable at all. Plus a 'Micro accuracy' row: pooled correct/total
     across all 12314 images (all 13 classes), not just the species subset.
  2. Per-level accuracy table: rows = hierarchy levels 1 (coarsest,
     Zoop-yes/Zoop-No) / 2 (mid, Copepoda/Cladocera/Rotifer) / 3 (fine, the
     12 species); columns = model runs (same as table 1).

All four runs are evaluated against the identical Messy_background image set
(same seed, same uncapped ImageDataset construction -- see
runs/hiernet_mnr.ipynb's evaluate_variant cell and
runs/evaluate_bcnn_messy_background.py), so cells are directly comparable.
Run counts are checked below and the script refuses to silently compare
mismatched sets.
"""

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / 'training_result' / 'mnr'
sys.path.append(str(REPO_ROOT))
from hierroute.constants import HIERNET_CLASSES

# Edit this to add/remove runs. Key = column label in the tables.
RUNS = {
    'bcnn_densenet121 (baseline)': 'bcnn_densenet121_baseline_08-25_messy_eval',
    'efficientnet_b0 (baseline)':  'efficientnet_b0_cnn_baseline_08-25_messy_eval',
    'swin_t (baseline)':           'swin_t_cnn_baseline_08-25_messy_eval',
    'bcnn_densenet121 (noise)':    'bcnn_densenet121_noise_08-25_messy_eval',
    'efficientnet_b0 (noise)':     'efficientnet_b0_cnn_noise_08-25_messy_eval',
    'swin_t (noise)':              'swin_t_cnn_noise_08-25_messy_eval',
}

LEVEL_LABELS = {
    '1': 'Level 1 (coarsest: Zoop-yes/Zoop-No)',
    '2': 'Level 2 (mid: Copepoda/Cladocera/Rotifer)',
    '3': 'Level 3 (fine: 12 species)',
}

############################# Load ##############################################

results = {}
for run_name, dirname in RUNS.items():
    path = RESULTS_DIR / dirname / 'eval_results.json'
    if not path.exists():
        raise FileNotFoundError(f"Missing eval_results.json for '{run_name}': {path}")
    results[run_name] = json.load(open(path))['level_results']

############################# Sanity: same eval set, same classes ##############

n_by_level = {
    run: {lvl: results[run][lvl]['overall']['n_samples'] for lvl in ('1', '2', '3')}
    for run in RUNS
}
mismatched_n = {lvl for lvl in ('1', '2', '3') if len({n_by_level[run][lvl] for run in RUNS}) > 1}
if mismatched_n:
    print(f'WARNING: sample counts differ across runs at level(s) {sorted(mismatched_n)} -- '
          f'these runs were NOT evaluated on the same Messy_background image set:')
    for lvl in sorted(mismatched_n):
        for run in RUNS:
            print(f'    level {lvl} | {run:<28} n={n_by_level[run][lvl]}')

species = [c for c in HIERNET_CLASSES if c != 'Zoop-No']
species_by_run = {run: set(results[run]['3']['per_class'].keys()) for run in RUNS}
missing_species = {run: set(species) - species_by_run[run] for run in RUNS}
if any(missing_species.values()):
    print(f'WARNING: some HIERNET_CLASSES species are missing from a run\'s level-3 '
          f'results: { {r: sorted(v) for r, v in missing_species.items() if v} }')
if any('Zoop-No' not in results[run]['1']['per_class'] for run in RUNS):
    print("WARNING: 'Zoop-No' missing from a run's level-1 results.")

############################# Table 1: per-leaf accuracy ########################

table1 = pd.DataFrame(index=HIERNET_CLASSES + ['Micro accuracy'], columns=list(RUNS.keys()), dtype=float)
for run in RUNS:
    per_class_fine = results[run]['3']['per_class']
    for cls in species:
        table1.loc[cls, run] = per_class_fine[cls]['accuracy']
    table1.loc['Zoop-No', run] = results[run]['1']['per_class']['Zoop-No']['accuracy']

    # Micro accuracy over the full HIERNET_CLASSES set (all 13 classes, all
    # 12314 images): pooled correct/total, combining the species-level subset
    # (level 3) with Zoop-No's own count from level 1 rather than double-
    # counting or leaving it out.
    n_species = results[run]['3']['overall']['n_samples']
    correct_species = results[run]['3']['overall']['accuracy'] * n_species
    n_zoop_no = results[run]['1']['per_class']['Zoop-No']['n_samples']
    correct_zoop_no = results[run]['1']['per_class']['Zoop-No']['accuracy'] * n_zoop_no
    table1.loc['Micro accuracy', run] = (correct_species + correct_zoop_no) / (n_species + n_zoop_no)

# Sort leaf rows by mean accuracy across runs, worst first (easiest place to spot problems)
leaf_order = table1.loc[HIERNET_CLASSES].mean(axis=1).sort_values().index.tolist()
table1 = table1.loc[leaf_order + ['Micro accuracy']]

############################# Table 2: per-level accuracy #######################

table2 = pd.DataFrame(index=[LEVEL_LABELS[l] for l in ('1', '2', '3')], columns=list(RUNS.keys()), dtype=float)
for run in RUNS:
    for lvl in ('1', '2', '3'):
        table2.loc[LEVEL_LABELS[lvl], run] = results[run][lvl]['overall']['accuracy']

############################# Output #############################################

pd.set_option('display.width', 160)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')


def to_markdown(df):
    header = '| ' + ' | '.join([''] + list(df.columns)) + ' |'
    sep = '|' + '---|' * (len(df.columns) + 1)
    rows = [
        '| ' + ' | '.join([str(idx)] + [f'{v:.4f}' for v in row]) + ' |'
        for idx, row in zip(df.index, df.values)
    ]
    return '\n'.join([header, sep] + rows)


print('\n=== Table 1: Leaf-class accuracy on Messy_background ===\n')
print(table1.to_string())

print('\n\n=== Table 2: Per-level accuracy on Messy_background ===\n')
print(table2.to_string())

out_dir = REPO_ROOT / 'runs' / 'logs'
out_dir.mkdir(exist_ok=True)
table1.to_csv(out_dir / 'messy_background_leaf_accuracy.csv')
table2.to_csv(out_dir / 'messy_background_level_accuracy.csv')
with open(out_dir / 'messy_background_leaf_accuracy.md', 'w') as f:
    f.write(to_markdown(table1))
with open(out_dir / 'messy_background_level_accuracy.md', 'w') as f:
    f.write(to_markdown(table2))

print(f'\n\nSaved CSV + Markdown to {out_dir}/messy_background_{{leaf,level}}_accuracy.{{csv,md}}')
