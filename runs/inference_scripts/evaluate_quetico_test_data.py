"""
Evaluate a trained HierRouteNet checkpoint against the Quetico "Manually Classified" crops in
/data/Hier-Zooplankton/quetico/{Messy,Not_Messy}/ -- the same analysis as
evaluate_erie_test_data.py, run once per background subset.

Usage: python runs/inference_scripts/evaluate_quetico_test_data.py <checkpoint_dir> [backbone] [t1 t2 t3] [max_per_class] [--capped]
  Same argument conventions as evaluate_erie_test_data.py. Each invocation evaluates both
  subsets in turn. --exclude is accepted but a no-op: no Quetico CSV_Data run is flagged.

Outputs, one folder per subset, same set as Erie:
  runs/inference_results/quetico/messy/quetico_messy_test_data_<checkpoint_dir name>/
  runs/inference_results/quetico/non-messy/quetico_not_messy_test_data_<checkpoint_dir name>/

Ground truth (on top of Erie's rules -- see evaluate_erie_test_data.py's docstring):
  Graded as Zoop-No:       Detritus, Duplicate, Bubble, Small_Nauplius, Decomposing
  Graded as Kellicottia:   Kellicottia_Smalls (pooled with Kellicottia)
  Coarse manual IDs:       Copepod -> Copepoda_Unknown, Rotifer -> Rotifer_Unknown,
                           Zooplankton -> Zoop_Unknown, Cladocera -> Cladocera_Unknown (via
                           HIERNET_CLASS_FOLDER_MAP; this folder also holds the Cladoceran crops,
                           merged into it during preprocessing)
  Ungraded:                Unknown, Conochilus_Maybe, MultiClass, Partial, Kellicottia_Multiple (MultiClass-
                           equivalent), Decomposing_Copepod/_Calanoid/_Cyclopoid and
                           Duplicate_Zooplankton (ignored). These still go through inference and
                           are listed in predictions.csv (in_taxonomy=False), but no chart or
                           metric uses them -- except breakdown_classes below.

top_predictions_breakdown.png covers Unknown, Conochilus_Maybe, Small_Nauplius, Decomposing,
Decomposing_Copepod/_Calanoid/_Cyclopoid, MultiClass, Kellicottia_Multiple (its own row, not
pooled into MultiClass) and Partial. Conochilus_Maybe and every Decomposing* folder only exist in
Messy (N/A in Not_Messy).
"""
from __future__ import annotations

import os
import sys

# inference_common puts the repo root on sys.path and loads .env, so it must come before hierroute.
from inference_common import build_folder_to_leaf
from test_data_evaluation import TestDatasetConfig, run_evaluation
from hierroute.constants import HIERNET_CLASSES, HIERNET_CLASS_FOLDER_MAP

# Folders deliberately left ungraded (in_taxonomy=False) -- only used by the sanity check below.
UNGRADED_FOLDERS = frozenset({
    'Unknown', 'Conochilus_Maybe', 'MultiClass', 'Partial', 'Kellicottia_Multiple',
    'Decomposing_Copepod', 'Decomposing_Calanoid', 'Decomposing_Cyclopoid', 'Duplicate_Zooplankton',
})


def quetico_config(name, data_subdir, results_dir):
    return TestDatasetConfig(
        name=name,
        data_subdir=data_subdir,
        results_dir=results_dir,
        extra_leaf_folders={
            'Zoop-No': ['Detritus', 'Duplicate', 'Bubble', 'Small_Nauplius', 'Decomposing'],
            'Kellicottia': ['Kellicottia_Smalls'],
        },
        exclude_dirs=frozenset({'Classified_data', 'CSV_Data'}),
        ambiguous_classes=(),
        unclassified_subdir=None,
        exclude_sample_patterns=(),
        coarse_folders={
            'Rotifer': 'Rotifer_Unknown',
            'Copepod': 'Copepoda_Unknown',
            'Zooplankton': 'Zoop_Unknown',
        },
        breakdown_classes=('Unknown', 'Conochilus_Maybe', 'Small_Nauplius', 'Decomposing',
                           'Decomposing_Copepod', 'Decomposing_Calanoid', 'Decomposing_Cyclopoid',
                           'MultiClass', 'Kellicottia_Multiple', 'Partial'),
        gate_by_confidence=False,
        include_cascade_detail=True,
        include_csv_metadata=True,
        include_loosened_leaf_metrics=True,
        loosened_leaf_metrics_exclude=frozenset({'Zoop-No'}),
        use_loosened_leaf_accuracy=True,
        include_hierarchy_level_metrics=True,
        # On-disk folder -> CSV 'Class' value(s) its crops were extracted under, where they differ.
        csv_class_aliases={'Nauplius_Copepod': 'Nauplius', 'Cladocera': ('Cladocera', 'Cladoceran')},
    )


CONFIGS = [
    quetico_config('Quetico_Messy', 'quetico/Messy', 'quetico/messy'),
    quetico_config('Quetico_Not_Messy', 'quetico/Not_Messy', 'quetico/non-messy'),
]


def warn_unaccounted_folders(config):
    """Print any on-disk label folder that is neither graded nor in UNGRADED_FOLDERS, so a new
    raw label (e.g. a 'Bubble' vs 'Bubbles' spelling) can't silently drop out of grading."""
    dataset_dir = os.path.join(os.environ['ZOOPLANKTON_DATA_DIR'], config.data_subdir)
    graded = set(build_folder_to_leaf(HIERNET_CLASSES, HIERNET_CLASS_FOLDER_MAP, config.extra_leaf_folders))
    graded |= set(config.coarse_folders)
    unaccounted = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
        and d not in config.exclude_dirs and d not in graded and d not in UNGRADED_FOLDERS
    )
    if unaccounted:
        print(f'WARNING ({config.name}): folders with no grading rule (ungraded, predictions.csv only): {unaccounted}')


if __name__ == '__main__':
    for config in CONFIGS:
        print(f'\n===== {config.name} ({config.data_subdir}) =====')
        warn_unaccounted_folders(config)
        run_evaluation(config, sys.argv, doc=__doc__)
