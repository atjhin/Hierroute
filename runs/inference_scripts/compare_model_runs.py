"""
Compare aggregate ID-outcome performance across model checkpoints / test sets, from already-
generated predictions.csv files (no re-inference needed -- predicted_class_1 + per-level
confidences are enough to reconstruct the same cascade-threshold classification
id_outcomes_table.png's Total row uses, via hierarchy.get_path_to_root + classify_id_outcome).

Usage: python runs/inference_scripts/compare_model_runs.py
  Edit the RUNS list below to add/remove checkpoints or test sets to compare.

Outputs, under runs/inference_results/:
  model_comparison.csv -- one row per (test_data, model_date, exclude): N, correct_all,
                           correct_total (correct_all + correct_incomplete), incorrect_total
                           (incorrect_l3 + incorrect_l2 + incorrect_all), abstained -- all
                           proportions of that run's taxonomy-mapped population (same
                           denominator as id_outcomes_table.png's Total row).
  model_comparison.png -- the same table, styled like the rest of this pipeline's tables.
"""
from __future__ import annotations

import pandas as pd

from inference_common import build_hierarchy, build_folder_to_leaf, render_table
from hierroute.constants import HIERNET_CLASSES, HIERNET_CLASS_FOLDER_MAP, hiernet_adjacency_graph
from hiernet_eval import apply_confidence_threshold, classify_id_outcome, ID_OUTCOME_KEYS

THRESHOLDS = (0.98, 0.995, 0.99)

EMILY_EXTRA_LEAF_FOLDERS = {'Zoop-No': ['Detritus', 'Debris']}
ERIE_EXTRA_LEAF_FOLDERS = {'Zoop-No': ['Detritus', 'Small_Organism']}
RESULTS_DIR = 'runs/inference_results'

# (test_data, model_date, exclude, predictions.csv path, extra_leaf_folders for that dataset)
RUNS = [
    ('Emily', '08-25', False,
     f'{RESULTS_DIR}/emily/emily_test_data_swin_t_cnn_noise_08-25/predictions.csv', EMILY_EXTRA_LEAF_FOLDERS),
    ('Emily', '09-13', False,
     f'{RESULTS_DIR}/emily/emily_test_data_swin_t_cnn_noise_09-13/predictions.csv', EMILY_EXTRA_LEAF_FOLDERS),
    ('Erie', '08-25', False,
     f'{RESULTS_DIR}/erie/erie_test_data_swin_t_cnn_noise_08-25/predictions.csv', ERIE_EXTRA_LEAF_FOLDERS),
    ('Erie', '08-25', True,
     f'{RESULTS_DIR}/erie/erie_test_data_swin_t_cnn_noise_08-25_exclude/predictions.csv', ERIE_EXTRA_LEAF_FOLDERS),
    ('Erie', '09-13', False,
     f'{RESULTS_DIR}/erie/erie_test_data_swin_t_cnn_noise_09-13/predictions.csv', ERIE_EXTRA_LEAF_FOLDERS),
    ('Erie', '09-13', True,
     f'{RESULTS_DIR}/erie/erie_test_data_swin_t_cnn_noise_09-13_exclude/predictions.csv', ERIE_EXTRA_LEAF_FOLDERS),
]

hierarchy, label_to_ids = build_hierarchy(hiernet_adjacency_graph)


def id_outcome_proportions(predictions_csv, extra_leaf_folders):
    """Reconstruct each taxonomy-mapped row's cascade-settled ID outcome from predicted_class_1
    (-> its fixed root-to-leaf path via the hierarchy) + confidence_level_1/2/3, then tally
    ID_OUTCOME_KEYS proportions the same way id_outcome_table_from's 'Total' row does: per true-
    leaf-class proportions (rounded to 3dp), N-weighted-averaged across classes -- not a flat
    pool -- so this reproduces id_outcomes_table.png's Total row bit for bit (rounding included)
    rather than just closely."""
    folder_to_leaf = build_folder_to_leaf(HIERNET_CLASSES, HIERNET_CLASS_FOLDER_MAP, extra_leaf_folders)
    df = pd.read_csv(predictions_csv)
    mapped = df[df['in_taxonomy']]

    outcomes_by_class = {}
    for _, row in mapped.iterrows():
        leaf_id = label_to_ids[row['predicted_class_1']]
        path = hierarchy.get_path_to_root(leaf_id)
        confs = [row[f'confidence_level_{d}'] for d in range(1, len(path))]
        predicted_node = apply_confidence_threshold(path, confs, THRESHOLDS)
        true_leaf = folder_to_leaf[row['actual_class']]
        outcome = classify_id_outcome(predicted_node, label_to_ids[true_leaf], hierarchy)
        outcomes_by_class.setdefault(true_leaf, []).append(outcome)

    n = len(mapped)
    per_class = pd.DataFrame({
        cls: {k: outs.count(k) / len(outs) for k in ID_OUTCOME_KEYS} for cls, outs in outcomes_by_class.items()
    }).T.round(3)
    weights = pd.Series({cls: len(outs) for cls, outs in outcomes_by_class.items()}) / n
    return n, per_class.multiply(weights, axis=0).sum()


rows = []
for test_data, model_date, exclude, csv_path, extra_leaf_folders in RUNS:
    n, props = id_outcome_proportions(csv_path, extra_leaf_folders)
    rows.append({
        'test_data': test_data,
        'model_date': model_date,
        'exclude': exclude,
        'N': n,
        'correct_all': props['correct_all'],
        'correct_total': props['correct_all'] + props['correct_incomplete'],
        'incorrect_total': props['incorrect_l3'] + props['incorrect_l2'] + props['incorrect_all'],
        'abstained': props['abstained'],
    })

comparison_df = pd.DataFrame(rows)
csv_path = f'{RESULTS_DIR}/model_comparison.csv'
comparison_df.round(3).to_csv(csv_path, index=False)
print(f'Saved -> {csv_path}')
print(comparison_df.round(3).to_string(index=False))

display_df = comparison_df.copy()
display_df['exclude'] = display_df['exclude'].map({True: 'Yes', False: 'No'})
display_df = display_df.set_index('test_data')[
    ['model_date', 'exclude', 'N', 'correct_all', 'correct_total', 'incorrect_total', 'abstained']
]

render_table(
    display_df,
    f'Model comparison: ID outcomes on Emily / Erie test data (thresholds={THRESHOLDS})',
    f'{RESULTS_DIR}/model_comparison.png',
    highlight_row_max=False,
    label_cols=['model_date', 'exclude'],
    footnote=('correct_total = correct_all + correct_incomplete; incorrect_total = incorrect_l3 + '
              'incorrect_l2 + incorrect_all; all proportions of that run\'s taxonomy-mapped N. '
              'exclude = _Decomp/_LotsofSmalls CSV_Data runs dropped (Erie only).'),
    count_cols={'N'},
)
