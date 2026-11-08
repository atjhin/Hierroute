"""
Evaluate a trained HierRouteNet checkpoint (swin_t / swin_s / efficientnet_b0, from
runs/train_swin_cnn_mnr.py) against the "Manually Classified" crops in
/data/Hier-Zooplankton/Erie_test_data/ -- the same analysis as evaluate_emily_test_data.py,
applied to Erie's own folder layout and taxonomy.

Usage: python runs/inference_scripts/evaluate_erie_test_data.py <checkpoint_dir> [backbone] [t1 t2 t3] [max_per_class] [--exclude] [--capped]
  Same argument conventions as evaluate_emily_test_data.py: backbone auto-detected from the
  checkpoint directory name if omitted; t1/t2/t3 default to 0.98 0.995 0.99; max_per_class
  (leaf_accuracy_capped.png's per-leaf sample cap) defaults to 3000 and requires t1 t2 t3 to
  also be passed if overridden. --exclude/--capped can appear anywhere in the argument list.

  leaf_accuracy_capped.png/id_outcomes_table_capped.png are skipped by default -- pass --capped
  (or set include_capped=True on this file's CONFIG) to also produce them.

  --exclude drops every crop sourced from a CSV_Data run whose ORIGINAL (pre-rename) name
  contained '_Decomp' or '_LotsofSmalls' (see Erie_test_data/_original_csv_names.csv) --
  '20240514_Erie_COS026_2mm_Rep2' (ex-'..._TestAI_data_Decomp.csv') and
  '20240430_Erie_WMS0504_2mm_Rep3' (ex-'..._TestAI_data_LotsofSmalls.csv') as of this data
  drop. The drop happens before inference, so it affects every output below, not just one
  chart. When --exclude is passed, the output folder name gets an '_exclude' suffix so it
  never collides with an unfiltered run of the same checkpoint.

  Unlike Emily, this evaluation does not gate id_outcomes_stacked_bar(_by_sample).png/
  id_outcomes_table(_capped).png on the confidence-threshold cascade at all (gate_by_confidence=
  False on TestDatasetConfig): every crop is graded on the raw greedy leaf the cascade reaches
  (predicted_class_1 -- the same value leaf_accuracy.png already uses), ignoring confidence
  entirely. 'Abstained' is therefore impossible and is dropped from those outputs. t1/t2/t3
  still only affect predictions.csv's confidence_level_1/2/3 columns and (if Erie ever adds an
  unclassified_subdir) nothing else -- they no longer gate any chart or table here. For the same
  reason, ambiguous_class_summary.png (a confidence-threshold settle-depth summary) is not
  produced for Erie -- see top_predictions_breakdown.png below for what MultiClass/Partial/
  Unknown/Small_Organism actually predict as instead.

Ground truth: every file under Erie_test_data/<folder>/*.tif. A folder is graded against the
model's 13-leaf taxonomy (HIERNET_CLASSES) if it maps onto a leaf directly by name (Bosmina,
Calanoid, Cyclopoid, Daphnia, Kellicottia, Nauplius_Copepod), or via Detritus/Small_Organism/
Larval_Fish/Small_Nauplius/Small_Rotifer/Daphnia_Decomposing/Decomposing -> Zoop-No (all
catch-all "not identifiable to a real species" buckets -- none of these have an Emily-side
equivalent folder, but get the same treatment as Detritus/Debris there, per the data owner).
None of Larval_Fish/Small_Nauplius/Small_Rotifer/Daphnia_Decomposing/Decomposing exist anywhere
outside Erie_test_data, so this is grading-only: they are not part of HIERNET_CLASS_FOLDER_MAP
and the model was never trained on them. Cladoceran, Copepod, Rotifer and Zooplankton are generic/coarse manual IDs (species not
determined) that map onto their branch's own "_Unknown" leaf -- Cladoceran -> Cladocera_Unknown,
Copepod -> Copepoda_Unknown, Rotifer -> Rotifer_Unknown, Zooplankton -> Zoop_Unknown (the
catch-all leaf for "confidently that branch, species not resolved"). These are graded in
leaf_accuracy.png/id_outcomes_table.png like any other row, as an exact match against that one
"_Unknown" leaf -- not against any of the branch's specific species leaves -- since the manual ID
deliberately committed only to the coarse rank: a Rotifer crop is correct_all only if predicted
as Rotifer_Unknown; predicting e.g. Trichocerca is graded incorrect, same as any other wrong
leaf. See coarse_folders on TestDatasetConfig and run_evaluation's 'correct' column --
hierarchy.subtree_leaves of an already-leaf node returns just that node, so this exact-match rule
falls out of the same subtree-matching code path used elsewhere. Since the manual ID never named
a species, these rows' leaf_accuracy.png Precision/Recall/F1 are N/A (excluded from the macro
average) and incorrect_l3 never applies to them in id_outcomes_table.png. A prediction of
'<Branch>_Unknown' (e.g. Copepoda_Unknown) for a true species anywhere under that branch (e.g.
Cyclopoid) is graded correct_incomplete in id_outcomes_table.png -- see classify_id_outcome in
runs/scripts/hiernet_eval.py; this applies at every ancestor level, so e.g. a true Conochilus
(a Rotifer) gets correct_incomplete credit for either Rotifer_Unknown or Zoop_Unknown. Erie has
no Particle-equivalent "to be classified" folder, so there's no predictions_to_be_classified.csv
output here.

Outputs, under runs/inference_results/erie/erie_test_data_<checkpoint_dir name>[_exclude]/ -- same
set as evaluate_emily_test_data.py minus predictions_to_be_classified.csv and
ambiguous_class_summary.png, plus two more:
  predictions.csv, leaf_accuracy.png, id_outcomes_stacked_bar.png, id_outcomes_table.png, plus
  leaf_accuracy_capped.png/id_outcomes_table_capped.png when --capped is passed
  -- see evaluate_emily_test_data.py's docstring for what each contains (note gate_by_confidence
  above for how the id_outcomes_* outputs differ here).
  id_outcomes_stacked_bar_by_sample.png -- exactly id_outcomes_stacked_bar.png's rubric/
                               population/styling, but bucketed by which CSV_Data collection
                               run each crop came from (recovered from the crop filename's
                               source-CSV prefix) instead of by true leaf class.
  top_predictions_breakdown.png -- for each of breakdown_classes (Small_Organism, Unknown,
                               Small_Nauplius, Small_Rotifer, Veliger_or_Chydorus, MultiClass,
                               Partial -- catch-all/ambiguous/no-real-label folders worth knowing
                               what the model actually does with them), the top-3
                               predicted_class_1 values and each one's share of that folder's own
                               total. Works regardless of in_taxonomy status, so it covers
                               Unknown/Veliger_or_Chydorus/MultiClass/Partial (no real label at
                               all, not graded anywhere else) the same as
                               Small_Organism/Small_Nauplius/Small_Rotifer (graded as Zoop-No
                               elsewhere, but shown here broken out by actual predicted leaf).

Shared plumbing (hierarchy/model construction, dataset classes, cascade inference, table
rendering) lives in inference_common.py / test_data_evaluation.py in this same directory.
"""
from __future__ import annotations

import sys

from test_data_evaluation import TestDatasetConfig, run_evaluation

CONFIG = TestDatasetConfig(
    name='Erie',
    data_subdir='Erie_test_data',
    extra_leaf_folders={'Zoop-No': [
        'Detritus', 'Small_Organism',
        'Larval_Fish', 'Small_Nauplius', 'Small_Rotifer', 'Daphnia_Decomposing', 'Decomposing',
    ]},
    exclude_dirs=frozenset({'Classified_data', 'CSV_Data'}),
    ambiguous_classes=(),
    unclassified_subdir=None,
    exclude_sample_patterns=('_Decomp', '_LotsofSmalls'),
    coarse_folders={
        'Rotifer': 'Rotifer_Unknown',
        'Copepod': 'Copepoda_Unknown',
        'Cladoceran': 'Cladocera_Unknown',
        'Zooplankton': 'Zoop_Unknown',
    },
    breakdown_classes=('Small_Organism', 'Unknown', 'Small_Nauplius', 'Small_Rotifer', 'Veliger_or_Chydorus',
                        'MultiClass', 'Partial'),
    gate_by_confidence=False,
    include_cascade_detail=True,
    include_csv_metadata=True,
    include_loosened_leaf_metrics=True,
    loosened_leaf_metrics_exclude=frozenset({'Zoop-No'}),
    use_loosened_leaf_accuracy=True,
    include_hierarchy_level_metrics=True,
    # Every Erie_test_data crop was extracted under one of these CSV 'Class' values; only
    # 'Nauplius_Copepod' differs from its own on-disk folder name (verified against every crop
    # currently in Erie_test_data -- see build_csv_metadata_index/csv_metadata_for).
    csv_class_aliases={'Nauplius_Copepod': 'Nauplius'},
)

if __name__ == '__main__':
    run_evaluation(CONFIG, sys.argv, doc=__doc__)
