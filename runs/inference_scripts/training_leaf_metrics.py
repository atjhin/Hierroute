"""
Render training_result_accuracy.png from a training run's own held-out predictions.npz (the
same held-out split hierroute.trainer.Trainer.predict(..., save=True) already scored into
eval_results.json/predictions.npz -- NOT Erie/Emily's manually-classified test data).

evaluate_erie_test_data.py now produces this same file automatically, alongside its own results
(see TestDatasetConfig.include_training_result_accuracy / inference_common.
training_result_accuracy_table) -- this script remains for ad-hoc use on a checkpoint that has
no corresponding Erie run, or to regenerate the file without rerunning that whole evaluation.

Same "partial credit" rubric as leaf_accuracy_zoop_only.png (see inference_common.
loosened_leaf_metrics_table): a specific-species leaf backed off to its branch's own
'<Branch>_Unknown' sibling counts as a hit -- a Recall/Accuracy hit for the true species' own
row, and a Precision hit for the predicted '_Unknown' leaf's row. Unlike that table, Zoop-No is
NOT excluded here -- this is the model's own full internal test split, not Erie's
Zoop-No-dominated manually-classified crops.

Usage: python runs/inference_scripts/training_leaf_metrics.py <model_dir> [out_dir]
  <model_dir>, e.g. training_result/mnr/swin_t_cnn_noise_09-17, must contain predictions.npz
  (written by Trainer.predict(..., save=True) -- 'predictions'/'targets' are the model's own
  raw greedy leaf node id and the true leaf node id, one pair per held-out sample). Output is
  written to <out_dir>/training_result_accuracy.png; out_dir defaults to <model_dir> itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

from inference_common import build_hierarchy, render_table, training_result_accuracy_table
from hierroute.constants import hiernet_adjacency_graph

FOOTNOTE = ('A specific-species leaf backed off to its branch\'s own "<Branch>_Unknown" sibling '
            '(e.g. Cyclopoid predicted as Copepoda_Unknown) counts as a hit: a Recall/Accuracy '
            'hit for the true species\' own row, and a Precision hit for the predicted "_Unknown" '
            'leaf\'s row -- two different rows\' different metrics, not double-counted toward any '
            'single number. Ordinary species leaves\' Precision is unaffected (still exact-match '
            'only). Source: this checkpoint\'s own held-out predictions.npz, not Erie/Emily.')


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)

    model_dir = Path(argv[1])
    out_dir = Path(argv[2]) if len(argv) > 2 else model_dir

    hierarchy, label_to_ids = build_hierarchy(hiernet_adjacency_graph)
    id_to_name = {v: k for k, v in label_to_ids.items()}

    table = training_result_accuracy_table(model_dir, hierarchy, id_to_name)
    if table is None:
        print(f'{model_dir}/predictions.npz not found -- run Trainer.predict(..., save=True) for '
              f'this checkpoint first.')
        sys.exit(1)

    render_table(
        table,
        f'Leaf-class metrics on {model_dir.name}\'s held-out test split (partial credit for _Unknown)',
        out_dir / 'training_result_accuracy.png',
        summary_row='Total', highlight_row_max=False, footnote=FOOTNOTE, count_cols={'N'},
    )


if __name__ == '__main__':
    main(sys.argv)
