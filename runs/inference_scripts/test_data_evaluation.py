"""
Shared orchestration for scoring a trained HierRouteNet checkpoint against a manually-
classified test set laid out as <data_subdir>/<folder>/*.tif (folder name = raw ground-truth
label). Each dataset-specific script in this directory (evaluate_emily_test_data.py,
evaluate_erie_test_data.py, ...) just builds a TestDatasetConfig for its own taxonomy/folder
layout and calls run_evaluation(config, sys.argv) -- see those files for the CLI and for what
each output means.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from inference_common import (
    REPO_ROOT, EVAL_TRANSFORMS,
    build_hierarchy, detect_backbone, load_model, build_folder_to_leaf,
    LabeledFolderDataset, UnclassifiedFolderDataset, predict_batch,
    build_sample_stems, assign_sample, excluded_samples_from_original_names, per_group_id_outcomes,
    ambiguous_class_summary_table, leaf_metrics_table, id_outcome_table_from, top_predictions_table,
    capped_indices_by_group, render_table, plot_id_outcome_stacked_bar,
    ID_OUTCOME_KEYS, ID_OUTCOME_LABELS, ID_OUTCOME_COLORS,
    ID_OUTCOME_KEYS_NO_ABSTAIN, ID_OUTCOME_LABELS_NO_ABSTAIN, ID_OUTCOME_COLORS_NO_ABSTAIN,
    build_csv_metadata_index, csv_metadata_for, apply_confidence_threshold, classify_id_outcome,
    loosened_leaf_metrics_table, hierarchy_level_metrics_table, render_hierarchy_level_table,
    training_result_accuracy_table,
)
from hierroute.constants import HIERNET_CLASSES, HIERNET_CLASS_FOLDER_MAP, hiernet_adjacency_graph, SEED

BATCH_SIZE = 64
NUM_WORKERS = 8


@dataclass(frozen=True)
class TestDatasetConfig:
    name: str                        # e.g. 'Emily', 'Erie' -- used in output dir name / titles
    data_subdir: str                 # e.g. 'Emily_test_data', relative to ZOOPLANKTON_DATA_DIR
    extra_leaf_folders: dict = field(default_factory=dict)   # leaf -> [extra on-disk folder names]
    exclude_dirs: frozenset = frozenset({'Classified_data', 'CSV_Data'})
    ambiguous_classes: tuple = ('MultiClass', 'Partial', 'Unknown')
    unclassified_subdir: Optional[str] = None   # e.g. 'Particle', for a folder with no real label
    # False (the default) skips leaf_accuracy_capped.png/id_outcomes_table_capped.png entirely.
    # Override per-config here, or per-invocation with the --capped CLI flag (either one turns
    # it on; see run_evaluation's include_capped).
    include_capped: bool = False
    # Substrings of a CSV_Data run's ORIGINAL (pre-rename) name that mark it as flagged for
    # exclusion (e.g. Erie's '_Decomp'/'_LotsofSmalls' runs) -- only takes effect when the
    # script is invoked with --exclude; see excluded_samples_from_original_names.
    exclude_sample_patterns: tuple = ()
    # On-disk folder name -> hiernet_adjacency_graph node name, for folders whose manual ID only
    # committed to a coarser rank than a specific species. Graded via hierarchy.subtree_leaves of
    # the mapped node -- point at an *internal* node (e.g. 'Rotifer') to count correct against
    # every leaf in its subtree (any of Trichocerca/Asplanchna/Conochilus/Kellicottia/
    # Rotifer_Unknown), or at a leaf directly (e.g. 'Rotifer_Unknown', Erie's current setup) for
    # an exact match against just that one class -- subtree_leaves of an already-leaf node
    # returns just that node, so the same code path covers both. See run_evaluation's 'correct'
    # column and classify_id_outcome's subtree matching. Folders not listed here or in
    # extra_leaf_folders/HIERNET_CLASS_FOLDER_MAP stay in_taxonomy=False, same as before.
    coarse_folders: dict = field(default_factory=dict)
    # On-disk folder names (any raw actual_class value, taxonomy-mapped or not -- e.g. a
    # no-real-label folder like 'Unknown') to show a top_predictions_breakdown.png table for:
    # what predicted_class_1 that folder's crops actually landed on, top-3 by share of that
    # folder's own total. See inference_common.top_predictions_table.
    breakdown_classes: tuple = ()
    # False grades id_outcomes_stacked_bar(_by_sample)/id_outcomes_table(_capped) against the
    # raw greedy leaf (ignoring confidence entirely) instead of the confidence-threshold
    # cascade's accepted node -- 'abstained' becomes impossible, so its column is dropped from
    # those outputs. Also drops predictions.csv's pass_level_1/2/3 columns (nothing gates on
    # them anymore) and skips ambiguous_class_summary.png, since settle_depth_distribution is
    # itself a confidence-threshold cascade summary with nothing to measure once thresholds
    # aren't gating anything.
    gate_by_confidence: bool = True
    # True adds predicted_level_1/2/3 (the raw greedy cascade path's node name at each hierarchy
    # depth, blank past however deep that sample's path actually reached -- predicted_class_1
    # itself is just predicted_level_3, or predicted_level_1 for a sample whose path terminates
    # at depth 1, e.g. Zoop-No) plus 5 boolean columns (correct_all/correct_incomplete/
    # incorrect_l3/incorrect_l2/incorrect_all, from classify_id_outcome, one True per
    # in_taxonomy=True row and all blank otherwise) to predictions.csv. Graded the same way
    # id_outcome_paths is elsewhere in this function -- the raw greedy leaf if gate_by_confidence
    # is False, else the confidence-threshold cascade's accepted node (which can make every one
    # of the 5 booleans False at once, for an abstained row).
    include_cascade_detail: bool = False
    # True adds every CSV_Data column (besides Particle ID/Class/Image File, pure join-key
    # identifiers already covered by image_file/actual_class) to predictions.csv, snake_cased
    # with a csv_ prefix (e.g. csv_area_abd) -- see inference_common.build_csv_metadata_index/
    # csv_metadata_for. csv_class_aliases is required for any on-disk folder whose crops were
    # extracted under a differently-named CSV 'Class' value (e.g. Erie's 'Nauplius_Copepod'
    # folder was extracted under CSV Class 'Nauplius') -- without it, that folder's rows simply
    # won't find a metadata match and are left blank.
    # A value may also be a tuple of CSV 'Class' values, tried in order, for a folder that pools
    # crops extracted under several (e.g. quetico's 'Cladocera' folder, which also holds the
    # merged-in 'Cladoceran' crops) -- see csv_metadata_for.
    include_csv_metadata: bool = False
    csv_class_aliases: dict = field(default_factory=dict)
    # Output parent directory, relative to runs/inference_results/. None (the default) uses
    # name.lower() (e.g. 'erie'); set for datasets split into subsets with their own result
    # folders (e.g. 'quetico/messy').
    results_dir: Optional[str] = None
    # True adds leaf_accuracy_zoop_only.png: same N/Accuracy/Precision/Recall/F1 shape as
    # leaf_accuracy.png, but a specific-species leaf backed off to its branch's own
    # '<Branch>_Unknown' sibling counts as a hit too (see
    # inference_common.loosened_leaf_metrics_table), and loosened_leaf_metrics_exclude drops
    # leaf names from the table's population entirely, e.g. {'Zoop-No'} to keep an
    # overwhelmingly common, uninteresting catch-all from dominating N -- leaving only the
    # actual zooplankton ("zoop only") leaves.
    include_loosened_leaf_metrics: bool = False
    loosened_leaf_metrics_exclude: frozenset = frozenset()
    # True switches leaf_accuracy.png/leaf_accuracy_capped.png themselves from leaf_metrics_table
    # (strict exact-match) to loosened_leaf_metrics_table (no exclude_leaves -- unlike
    # leaf_accuracy_zoop_only.png, Zoop-No stays in this one).
    use_loosened_leaf_accuracy: bool = False
    # True adds hierarchy_level_metrics.png: Accuracy/Precision/Recall/F1 per hierarchy depth,
    # each level graded as its own independent flat classification (level 1 = Zoop-yes/Zoop-No,
    # level 2 = Copepoda/Cladocera/Rotifer/Zoop_Unknown, level 3 = the species-level leaves) --
    # see inference_common.hierarchy_level_metrics_table/render_hierarchy_level_table.
    include_hierarchy_level_metrics: bool = False
    # True adds training_result_accuracy.png: loosened_leaf_metrics_table computed from this
    # checkpoint's own held-out predictions.npz (see inference_common.training_result_accuracy_
    # table) -- the model's internal test split, not this dataset's manually-classified crops.
    # Silently skipped (with a printed note) if that checkpoint has no predictions.npz.
    include_training_result_accuracy: bool = False


FOOTNOTE_LEAF_METRICS = ('Precision / Recall / F1 are macro (equally weighted across leaves); '
                          'Accuracy is micro (weighted by sample count).')
FOOTNOTE_AMBIGUOUS = ('Each row sums to ~1: the proportion of that class\'s samples whose '
                       'confidence-threshold cascade settled at each hierarchy depth (or abstained, '
                       'if none cleared its threshold).')
FOOTNOTE_BREAKDOWN = ('For each folder, the top-3 predicted_class_1 values by share of that '
                       'folder\'s own total (N) -- regardless of whether the folder has a real '
                       'taxonomy label, so this also covers no-real-label folders like Unknown.')
FOOTNOTE_LOOSENED = ('A specific-species leaf backed off to its branch\'s own "<Branch>_Unknown" '
                      'sibling (e.g. Cyclopoid predicted as Copepoda_Unknown) counts as a hit: a '
                      'Recall/Accuracy hit for the true species\' own row, and a Precision hit for '
                      'the predicted "_Unknown" leaf\'s row -- two different rows\' different '
                      'metrics, not double-counted toward any single number. Ordinary species '
                      'leaves\' Precision is unaffected (still exact-match only).')
FOOTNOTE_LEVEL = ('Each hierarchy level (color-coded; see legend) is its own independent flat '
                   'classification (level 1 = Zoop-yes vs. Zoop-No, level 2 = Copepoda/Cladocera/'
                   'Rotifer/Zoop_Unknown, level 3 = the species-level leaves), ignoring every '
                   'other level. A sample is only in a level\'s population if its true leaf\'s '
                   'own depth reaches that level. A "_Unknown" sibling prediction at that depth '
                   '(e.g. true Copepoda / predicted Zoop_Unknown at level 2) counts as a hit, '
                   'pooled the same way as leaf_accuracy_zoop_only.png. Rows are sorted worst-'
                   'Accuracy-first within each level, with Total (tinted, level-colored) pinned '
                   'last; Total\'s Accuracy is sample-weighted (micro), Precision/Recall/F1 are '
                   'macro (equally weighted across that level\'s classes).')


def run_evaluation(config: TestDatasetConfig, argv, doc=None):
    exclude_flagged = '--exclude' in argv
    capped_flagged = '--capped' in argv
    argv = [a for a in argv if a not in ('--exclude', '--capped')]
    include_capped = config.include_capped or capped_flagged

    if len(argv) < 2:
        print(doc or __doc__)
        sys.exit(1)

    if exclude_flagged and not config.exclude_sample_patterns:
        print(f'--exclude passed but {config.name} has no exclude_sample_patterns configured; ignoring.')
        exclude_flagged = False

    checkpoint_dir = Path(argv[1])
    backbone = argv[2] if len(argv) > 2 else detect_backbone(checkpoint_dir)
    expert_type = 'cnn'
    thresholds = tuple(float(x) for x in argv[3:6]) if len(argv) > 5 else (0.98, 0.995, 0.99)
    max_per_class = int(argv[6]) if len(argv) > 6 else 3000

    data_directory = os.environ['ZOOPLANKTON_DATA_DIR']
    dataset_dir = os.path.join(data_directory, config.data_subdir)

    out_name = f'{config.name.lower()}_test_data_{checkpoint_dir.name}' + ('_exclude' if exclude_flagged else '')
    out_dir = REPO_ROOT / 'runs' / 'inference_results' / (config.results_dir or config.name.lower()) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    folder_to_leaf = build_folder_to_leaf(HIERNET_CLASSES, HIERNET_CLASS_FOLDER_MAP, config.extra_leaf_folders)

    hierarchy, label_to_ids = build_hierarchy(hiernet_adjacency_graph)
    id_to_name = {v: k for k, v in label_to_ids.items()}
    leaf_ids_sorted = [nid for nid in sorted(hierarchy.nodes) if hierarchy.is_leaf(nid)]

    unknown_coarse_nodes = set(config.coarse_folders.values()) - set(label_to_ids)
    assert not unknown_coarse_nodes, f'coarse_folders targets unknown hierarchy node(s): {unknown_coarse_nodes}'
    # folder_to_leaf (a real leaf, graded by exact match) extended with coarse_folders (an
    # internal node, graded by subtree membership -- see the 'correct' column below).
    folder_to_node = {**folder_to_leaf, **config.coarse_folders}

    sample_stems = build_sample_stems(os.path.join(dataset_dir, 'CSV_Data'))
    csv_metadata_index = (
        build_csv_metadata_index(os.path.join(dataset_dir, 'CSV_Data'), sample_stems)
        if config.include_csv_metadata else None
    )
    excluded_stems = excluded_samples_from_original_names(dataset_dir, config.exclude_sample_patterns) \
        if exclude_flagged else set()
    if exclude_flagged:
        print(f'Excluding samples from flagged CSV_Data runs: {sorted(excluded_stems) or "(none matched)"}')

    labeled_exclude_dirs = set(config.exclude_dirs)
    if config.unclassified_subdir:
        labeled_exclude_dirs.add(config.unclassified_subdir)

    keep_fn = (lambda filename: assign_sample(filename, sample_stems) not in excluded_stems) if excluded_stems else None
    dataset = LabeledFolderDataset(dataset_dir, EVAL_TRANSFORMS, labeled_exclude_dirs, keep_fn=keep_fn)
    print(f'{config.name}: Manually Classified crops found: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} | backbone: {backbone} | thresholds: {thresholds}')

    model = load_model(hierarchy, label_to_ids, backbone, checkpoint_dir, device, expert_type)
    leaf_ids_tensor = torch.tensor(leaf_ids_sorted, device=device)

    # -----------------------------------------------------------------------
    # Inference: one pass, caches everything predictions.csv and every chart below needs.
    # -----------------------------------------------------------------------
    rows = []
    id_outcome_paths, id_outcome_confidences, id_outcome_true_leaves, id_outcome_samples = [], [], [], []
    ambiguous_paths = {cls: [] for cls in config.ambiguous_classes}
    ambiguous_confidences = {cls: [] for cls in config.ambiguous_classes}

    with torch.no_grad():
        for images, image_paths, raw_classes in loader:
            images = images.to(device)
            batch_results = predict_batch(
                model, images, image_paths, leaf_ids_sorted, leaf_ids_tensor, id_to_name, thresholds,
            )
            for cr, raw_class in zip(batch_results, raw_classes):
                mapped_node = folder_to_node.get(raw_class)

                row = {
                    'image_file': os.path.basename(cr.image_path),
                    'actual_class': raw_class,
                    'in_taxonomy': mapped_node is not None,
                    'predicted_class_1': cr.predicted_1,
                    'predicted_class_2': cr.predicted_2,
                    'confidence_level_1': cr.conf_by_level[1],
                    'confidence_level_2': cr.conf_by_level[2],
                    'confidence_level_3': cr.conf_by_level[3],
                }
                if config.gate_by_confidence:
                    row['pass_level_1'] = cr.pass_by_level[1]
                    row['pass_level_2'] = cr.pass_by_level[2]
                    row['pass_level_3'] = cr.pass_by_level[3]

                if config.include_cascade_detail:
                    for d in (1, 2, 3):
                        row[f'predicted_level_{d}'] = id_to_name[cr.cascade_path[d]] if d < len(cr.cascade_path) else None
                    if mapped_node is not None:
                        graded_node = (
                            apply_confidence_threshold(cr.cascade_path, cr.cascade_confidences, thresholds)
                            if config.gate_by_confidence else cr.cascade_path[-1]
                        )
                        outcome = classify_id_outcome(graded_node, label_to_ids[mapped_node], hierarchy)
                        for key in ID_OUTCOME_KEYS_NO_ABSTAIN:
                            row[key] = (outcome == key)
                    else:
                        for key in ID_OUTCOME_KEYS_NO_ABSTAIN:
                            row[key] = None

                if config.include_csv_metadata:
                    metadata = csv_metadata_for(
                        os.path.basename(cr.image_path), raw_class, sample_stems, csv_metadata_index,
                        config.csv_class_aliases,
                    )
                    row.update(metadata or {})

                rows.append(row)

                if mapped_node is not None:
                    id_outcome_paths.append(cr.cascade_path)
                    id_outcome_confidences.append(cr.cascade_confidences)
                    id_outcome_true_leaves.append(label_to_ids[mapped_node])
                    id_outcome_samples.append(assign_sample(os.path.basename(cr.image_path), sample_stems))

                if raw_class in ambiguous_paths:
                    ambiguous_paths[raw_class].append(cr.cascade_path)
                    ambiguous_confidences[raw_class].append(cr.cascade_confidences)

    print(f'Ran inference on {len(rows)} images ({len(id_outcome_paths)} with a taxonomy match)')

    # -----------------------------------------------------------------------
    # 1. Predictions CSV
    # -----------------------------------------------------------------------
    predictions_df = pd.DataFrame(rows)
    predictions_path = out_dir / 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f'Saved -> {predictions_path}')

    # -----------------------------------------------------------------------
    # 1a. Top-3 predicted-class breakdown for configured folders (e.g. Unknown, Small_*)
    # -----------------------------------------------------------------------
    if config.breakdown_classes:
        breakdown_table = top_predictions_table(predictions_df, config.breakdown_classes)
        render_table(
            breakdown_table,
            f'Top predicted classes by folder on {config.data_subdir} -- {checkpoint_dir.name}',
            out_dir / 'top_predictions_breakdown.png',
            label_cols=['N', '#1 predicted', '#2 predicted', '#3 predicted'],
            footnote=FOOTNOTE_BREAKDOWN,
        )

    # -----------------------------------------------------------------------
    # 1b. Ambiguous-class settle-depth summary
    # -----------------------------------------------------------------------
    if config.ambiguous_classes and config.gate_by_confidence:
        ambiguous_table = ambiguous_class_summary_table(
            config.ambiguous_classes, ambiguous_paths, ambiguous_confidences, thresholds, hierarchy,
        )
        render_table(
            ambiguous_table,
            f'Ambiguous-class settle-depth summary (thresholds={thresholds}) -- {checkpoint_dir.name}',
            out_dir / 'ambiguous_class_summary.png',
            footnote=FOOTNOTE_AMBIGUOUS,
            count_cols={'N'},
        )

    # -----------------------------------------------------------------------
    # 2. Leaf-class metrics table, restricted to taxonomy-mapped rows
    # -----------------------------------------------------------------------
    mapped_df = predictions_df[predictions_df['in_taxonomy']].copy()
    mapped_df['true_leaf'] = mapped_df['actual_class'].map(folder_to_node)

    # A true_leaf that's an internal (coarse) node -- e.g. Erie's generic 'Rotifer' folder,
    # mapped to the 'Rotifer' node via coarse_folders -- is correct if the prediction lands on
    # any leaf in its subtree. hierarchy.subtree_leaves returns [node_id] itself for an actual
    # leaf, so this single rule covers the ordinary exact-match case too (same subtree matching
    # as classify_id_outcome's correct_all).
    valid_leaves_by_true = {
        true: {id_to_name[nid] for nid in hierarchy.subtree_leaves(label_to_ids[true])}
        for true in mapped_df['true_leaf'].unique()
    }
    mapped_df['correct'] = [
        pred in valid_leaves_by_true[true]
        for pred, true in zip(mapped_df['predicted_class_1'], mapped_df['true_leaf'])
    ]

    footnote_leaf_metrics = FOOTNOTE_LOOSENED if config.use_loosened_leaf_accuracy else FOOTNOTE_LEAF_METRICS
    if config.coarse_folders and not config.use_loosened_leaf_accuracy:
        footnote_leaf_metrics += (' A coarse manual ID -- e.g. Rotifer/Copepod/Cladoceran/'
                                   'Zooplankton -- is correct if predicted anywhere in its mapped '
                                   'coarse_folders node\'s leaf set (an exact match if mapped '
                                   'directly to a leaf, e.g. Rotifer_Unknown; any descendant '
                                   'species if mapped to an internal node); Precision/Recall/F1 '
                                   '(N/A) and the macro average skip it, since it is never itself '
                                   'a possible prediction.')

    def leaf_metrics(df):
        if config.use_loosened_leaf_accuracy:
            return loosened_leaf_metrics_table(df['predicted_class_1'], df['true_leaf'], hierarchy)
        return leaf_metrics_table(df, hierarchy=hierarchy, label_to_ids=label_to_ids)

    render_table(
        leaf_metrics(mapped_df),
        f'Leaf-class metrics on {config.data_subdir} (Manually Classified) -- {checkpoint_dir.name}',
        out_dir / 'leaf_accuracy.png',
        summary_row='Total', highlight_row_max=False, footnote=footnote_leaf_metrics, count_cols={'N'},
    )

    if include_capped:
        capped_idx = capped_indices_by_group(mapped_df['true_leaf'], max_per_class, SEED)
        capped_df = mapped_df.loc[capped_idx]

        render_table(
            leaf_metrics(capped_df),
            f'Leaf-class metrics on {config.data_subdir} (Manually Classified, capped at {max_per_class}/leaf) -- '
            f'{checkpoint_dir.name}',
            out_dir / 'leaf_accuracy_capped.png',
            summary_row='Total', highlight_row_max=False, footnote=footnote_leaf_metrics, count_cols={'N'},
        )

    if config.include_loosened_leaf_metrics:
        render_table(
            loosened_leaf_metrics_table(
                mapped_df['predicted_class_1'], mapped_df['true_leaf'], hierarchy,
                exclude_leaves=config.loosened_leaf_metrics_exclude,
            ),
            f'Leaf-class metrics on {config.data_subdir} (Manually Classified, zoop only, partial '
            f'credit for _Unknown) -- {checkpoint_dir.name}',
            out_dir / 'leaf_accuracy_zoop_only.png',
            summary_row='Total', highlight_row_max=False, footnote=FOOTNOTE_LOOSENED, count_cols={'N'},
        )

    if config.include_hierarchy_level_metrics:
        render_hierarchy_level_table(
            hierarchy_level_metrics_table(mapped_df['predicted_class_1'], mapped_df['true_leaf'], hierarchy),
            f'Per-hierarchy-level metrics on {config.data_subdir} (Manually Classified) -- {checkpoint_dir.name}',
            out_dir / 'hierarchy_level_metrics.png',
            footnote=FOOTNOTE_LEVEL, count_cols={'N'},
        )

    if config.include_training_result_accuracy:
        training_table = training_result_accuracy_table(checkpoint_dir, hierarchy, id_to_name)
        if training_table is not None:
            render_table(
                training_table,
                f'Leaf-class metrics on {checkpoint_dir.name}\'s held-out training test split '
                f'(partial credit for _Unknown)',
                out_dir / 'training_result_accuracy.png',
                summary_row='Total', highlight_row_max=False, footnote=FOOTNOTE_LOOSENED, count_cols={'N'},
            )
        else:
            print(f'No predictions.npz found under {checkpoint_dir} -- skipping training_result_accuracy.png')

    # -----------------------------------------------------------------------
    # 3. Per-class ID-outcome stacked bar + styled table
    # -----------------------------------------------------------------------
    if config.gate_by_confidence:
        outcome_keys, outcome_labels, outcome_colors = ID_OUTCOME_KEYS, ID_OUTCOME_LABELS, ID_OUTCOME_COLORS
        threshold_suffix = f' (thresholds={thresholds})'
    else:
        outcome_keys, outcome_labels, outcome_colors = (
            ID_OUTCOME_KEYS_NO_ABSTAIN, ID_OUTCOME_LABELS_NO_ABSTAIN, ID_OUTCOME_COLORS_NO_ABSTAIN,
        )
        threshold_suffix = ' (raw greedy leaf, no confidence gating)'

    id_outcome_table, id_outcome_props = id_outcome_table_from(
        model, loader, thresholds, hierarchy, id_to_name, device,
        id_outcome_paths, id_outcome_confidences, id_outcome_true_leaves,
        gate_by_confidence=config.gate_by_confidence, outcome_keys=outcome_keys,
    )

    fig = plot_id_outcome_stacked_bar(
        id_outcome_props,
        f'Per-class ID outcomes on {config.data_subdir}{threshold_suffix} -- {checkpoint_dir.name}',
        keys=outcome_keys, labels=outcome_labels, colors=outcome_colors,
    )
    fig.savefig(out_dir / 'id_outcomes_stacked_bar.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out_dir / "id_outcomes_stacked_bar.png"}')

    # Same rubric/thresholds/population as the plot above, but bucketed by which CSV_Data
    # collection run each crop came from instead of by true leaf class -- sorted() on the
    # (date-prefixed) sample names falls out chronological for free.
    id_outcome_props_by_sample = per_group_id_outcomes(
        thresholds, hierarchy, id_outcome_paths, id_outcome_confidences, id_outcome_true_leaves, id_outcome_samples,
        gate_by_confidence=config.gate_by_confidence, outcome_keys=outcome_keys,
    )
    fig = plot_id_outcome_stacked_bar(
        id_outcome_props_by_sample,
        f'Per-sample ID outcomes on {config.data_subdir}{threshold_suffix} -- {checkpoint_dir.name}',
        legend_side=True,
        keys=outcome_keys, labels=outcome_labels, colors=outcome_colors,
    )
    fig.savefig(out_dir / 'id_outcomes_stacked_bar_by_sample.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out_dir / "id_outcomes_stacked_bar_by_sample.png"}')

    render_table(
        id_outcome_table, f'Per-class ID outcomes -- {checkpoint_dir.name}', out_dir / 'id_outcomes_table.png',
        summary_row='Total', count_cols={'N'},
    )

    if include_capped:
        capped_outcome_idx = capped_indices_by_group(id_outcome_true_leaves, max_per_class, SEED)
        id_outcome_table_capped, _ = id_outcome_table_from(
            model, loader, thresholds, hierarchy, id_to_name, device,
            [id_outcome_paths[i] for i in capped_outcome_idx],
            [id_outcome_confidences[i] for i in capped_outcome_idx],
            [id_outcome_true_leaves[i] for i in capped_outcome_idx],
            gate_by_confidence=config.gate_by_confidence, outcome_keys=outcome_keys,
        )
        render_table(
            id_outcome_table_capped,
            f'Per-class ID outcomes (capped at {max_per_class}/leaf) -- {checkpoint_dir.name}',
            out_dir / 'id_outcomes_table_capped.png',
            summary_row='Total', count_cols={'N'},
        )

    # -----------------------------------------------------------------------
    # 4. Optional "to be classified" pass -- a flat folder of crops with no real label
    # -----------------------------------------------------------------------
    if config.unclassified_subdir:
        _run_unclassified_predictions(
            config, checkpoint_dir, dataset_dir, model, device,
            leaf_ids_sorted, leaf_ids_tensor, id_to_name, thresholds, out_dir,
        )

    print(f'\nAll outputs saved under {out_dir}')


def _run_unclassified_predictions(config, checkpoint_dir, dataset_dir, model, device,
                                   leaf_ids_sorted, leaf_ids_tensor, id_to_name, thresholds, out_dir):
    """No ground truth here, so this only produces the per-image predictions CSV (same columns
    as predictions.csv, minus a meaningful actual_class) -- no accuracy/precision/recall/F1
    tables, since there's nothing to grade against."""
    unclassified_dir = os.path.join(dataset_dir, config.unclassified_subdir)
    dataset = UnclassifiedFolderDataset(unclassified_dir, EVAL_TRANSFORMS)
    print(f'To be Classified crops found: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    rows = []
    with torch.no_grad():
        for images, image_paths in loader:
            images = images.to(device)
            for cr in predict_batch(model, images, image_paths, leaf_ids_sorted, leaf_ids_tensor, id_to_name, thresholds):
                row = {
                    'image_file': os.path.basename(cr.image_path),
                    'actual_class': config.unclassified_subdir,
                    'in_taxonomy': False,
                    'predicted_class_1': cr.predicted_1,
                    'predicted_class_2': cr.predicted_2,
                    'confidence_level_1': cr.conf_by_level[1],
                    'confidence_level_2': cr.conf_by_level[2],
                    'confidence_level_3': cr.conf_by_level[3],
                }
                if config.gate_by_confidence:
                    row['pass_level_1'] = cr.pass_by_level[1]
                    row['pass_level_2'] = cr.pass_by_level[2]
                    row['pass_level_3'] = cr.pass_by_level[3]
                rows.append(row)

    print(f'Ran inference on {len(rows)} images')
    predictions_df = pd.DataFrame(rows)
    predictions_path = out_dir / 'predictions_to_be_classified.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f'Saved -> {predictions_path}')
