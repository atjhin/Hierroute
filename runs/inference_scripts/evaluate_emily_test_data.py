"""
Evaluate a trained HierRouteNet checkpoint (swin_t / swin_s / efficientnet_b0, from
runs/train_swin_cnn_mnr.py) against the "Manually Classified" crops moved into
/data/Hier-Zooplankton/Emily_test_data/ (see HierNet/Zooplankton-MNR/pre_process_data.ipynb).

Usage: python runs/inference_scripts/evaluate_emily_test_data.py <checkpoint_dir> [backbone] [t1 t2 t3] [max_per_class] [--capped]
  backbone defaults to auto-detected from the checkpoint directory name
  (efficientnet_b0_*/swin_s_*/swin_t_* prefix), falling back to swin_t.
  t1/t2/t3 (confidence-cascade thresholds, coarsest -> finest, hierroute's own t1/t2/t3
  convention) default to 0.98 0.995 0.99.
  max_per_class (per-leaf sample cap for leaf_accuracy_capped.png, see below) defaults to
  3000; to override it you must also pass t1 t2 t3.
  leaf_accuracy_capped.png/id_outcomes_table_capped.png are skipped by default -- pass --capped
  (or set include_capped=True on this file's CONFIG) to also produce them.

Ground truth: every file under Emily_test_data/<folder>/*.tif, for every folder except
'Particle' (that one is sourced only from the "To be Classified" CSVs, which carry no real
species label -- see pre_process_data.ipynb's CSV_Data_unclassified pass). A folder is graded
against the model's 13-leaf taxonomy (HIERNET_CLASSES) if it maps onto a leaf directly by
name, via HIERNET_CLASS_FOLDER_MAP (e.g. Bosminidae -> Bosmina), or via Detritus/Debris ->
Zoop-No (both are catch-all "not a zooplankton, no finer ID" buckets, same as Zoop-No's
existing folder_map entries -- confirmed with the data owner). Every other folder (Duplicate,
Unknown, MultiClass, Partial, generic Zooplankton/Cladocera/Copepod, Cladoceran, Holopedidae,
Fiber) still gets a full row in predictions.csv (with in_taxonomy=False) but is excluded from
the leaf-accuracy table and the per-class outcome chart, since there's no matching model leaf
to grade it against.

Outputs, under runs/inference_results/emily/emily_test_data_<checkpoint_dir name>/:
  predictions.csv          -- one row per image: actual class, top-2 predicted leaves,
                               per-level cascade confidence, per-level threshold pass/fail.
  leaf_accuracy.png        -- per-leaf accuracy + micro accuracy + macro precision/recall/F1,
                               styled like runs/logs/messy_background_leaf_accuracy.png
                               (single-model column). Taxonomy-mapped rows only. (Macro, not
                               micro, precision/recall/F1: in this standard closed-13-leaf
                               multiclass setting micro-P/R/F1 collapse to the same number as
                               accuracy, so macro -- unweighted per-leaf average, matching
                               eval_results.json's own precision_macro/recall_macro/f1_macro
                               convention -- is the one that carries new information.)
  id_outcomes_stacked_bar.png -- per-class manual-ID-vs-AI-ID outcome proportions under the
                               confidence cascade (hiernet_eval.plot_id_outcome_stacked_bar),
                               exactly as used in runs/confidence_threshold_experiment.ipynb.
  id_outcomes_stacked_bar_by_sample.png -- exactly id_outcomes_stacked_bar.png's rubric/
                               population/styling, but bucketed by which CSV_Data collection
                               run each crop came from (recovered from the crop filename's
                               source-CSV prefix) instead of by true leaf class.
  id_outcomes_table.png    -- the same proportions table, styled like leaf_accuracy.png.
  leaf_accuracy_capped.png -- same table as leaf_accuracy.png, but each leaf's samples are
                               capped at max_per_class (a fixed-seed random subsample, taken
                               only from leaves that exceed the cap) before recomputing N /
                               Accuracy / Precision / Recall / F1 -- so an oversized leaf like
                               Zoop-No (12k+ samples) can't dominate the picture purely on
                               volume relative to leaves with far fewer samples.
  id_outcomes_table_capped.png -- id_outcomes_table.png, same max_per_class cap.
  ambiguous_class_summary.png -- MultiClass / Partial / Unknown (no real species label, so not
                               part of any of the tables above) -- N, and the proportion of each
                               class's samples whose confidence-threshold cascade settles at
                               each hierarchy depth (or abstains).
  predictions_to_be_classified.csv -- the Particle/ "To be Classified" crops: same columns,
                               minus a meaningful actual_class, no ground truth to grade.

Shared plumbing (hierarchy/model construction, dataset classes, cascade inference, table
rendering) lives in inference_common.py / test_data_evaluation.py in this same directory --
see evaluate_erie_test_data.py for the same pipeline applied to a different taxonomy/folder
layout.
"""
from __future__ import annotations

import sys

from test_data_evaluation import TestDatasetConfig, run_evaluation

CONFIG = TestDatasetConfig(
    name='Emily',
    data_subdir='Emily_test_data',
    extra_leaf_folders={'Zoop-No': ['Detritus', 'Debris']},
    exclude_dirs=frozenset({'Classified_data', 'CSV_Data', 'CSV_Data_unclassified'}),
    ambiguous_classes=('MultiClass', 'Partial', 'Unknown'),
    unclassified_subdir='Particle',
)

if __name__ == '__main__':
    run_evaluation(CONFIG, sys.argv, doc=__doc__)
