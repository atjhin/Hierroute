"""
Dataset-agnostic building blocks for the HierRouteNet manually-classified-test-data
evaluation scripts in this directory (evaluate_emily_test_data.py, evaluate_erie_test_data.py,
and any future <name>_test_data set). Everything here is generic over which test set is being
scored -- the per-dataset scripts only supply a TestDatasetConfig (see test_data_evaluation.py)
with the folder-to-leaf overrides, ambiguous-class names, etc. specific to that dataset.
"""
from __future__ import annotations

import os
import re
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (REPO_ROOT, REPO_ROOT / 'runs' / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
load_dotenv(REPO_ROOT / '.env')

from hierroute.constants import RESOLUTION
from hierroute.hierarchy import Hierarchy
from hierroute import HierRouteNet
from hiernet_eval import (
    predict_with_confidences, settle_depth_distribution, apply_confidence_threshold, classify_id_outcome,
    per_class_id_outcomes, plot_id_outcome_stacked_bar, ID_OUTCOME_KEYS,
    ID_OUTCOME_LABELS, ID_OUTCOME_COLORS, ID_OUTCOME_KEYS_NO_ABSTAIN, ID_OUTCOME_LABELS_NO_ABSTAIN,
    ID_OUTCOME_COLORS_NO_ABSTAIN,
)

EVAL_TRANSFORMS = transforms.Compose([transforms.Resize((RESOLUTION, RESOLUTION)), transforms.ToTensor()])


# ---------------------------------------------------------------------------
# Hierarchy + model construction -- pure function of the adjacency graph, mirrors
# hierroute.data_setup.HierImageDataset._build_hierarchy exactly (same node_id assignment
# order) so a checkpoint trained via eval_loaders.py loads cleanly.
# ---------------------------------------------------------------------------
def build_hierarchy(adjacency_graph):
    label_to_ids = {name: i for i, name in enumerate(adjacency_graph.keys())}
    hierarchy = Hierarchy()
    all_children = {c for kids in adjacency_graph.values() for c in kids}
    root = next(n for n in adjacency_graph if n not in all_children)
    hierarchy.add_node(label_to_ids[root], root)

    def dfs(parent):
        for child in adjacency_graph[parent]:
            hierarchy.add_node(label_to_ids[child], child, label_to_ids[parent])
            dfs(child)

    dfs(root)
    return hierarchy, label_to_ids


def detect_backbone(checkpoint_dir):
    """efficientnet_b0_*/swin_s_*/swin_t_* prefix on the checkpoint dir name, else swin_t."""
    name_lower = Path(checkpoint_dir).name.lower()
    return next((b for b in ('efficientnet_b0', 'swin_s', 'swin_t') if name_lower.startswith(b)), 'swin_t')


def load_model(hierarchy, label_to_ids, backbone, checkpoint_dir, device, expert_type='cnn'):
    model = HierRouteNet(
        hierarchy=hierarchy,
        label_to_id=label_to_ids,
        backbone=backbone,
        expert_type=expert_type,
        checkpoint_dir=str(checkpoint_dir),
    )
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Ground-truth folder -> leaf-class mapping
# ---------------------------------------------------------------------------
def build_folder_to_leaf(hiernet_classes, class_folder_map, extra_leaf_folders=None):
    """
    folder name (on disk) -> leaf class name, for every HIERNET_CLASSES leaf: its own name,
    plus any aliases from class_folder_map (e.g. Bosminidae -> Bosmina), plus dataset-specific
    extra_leaf_folders overrides (e.g. {'Zoop-No': ['Detritus', 'Debris']} for Emily's
    Detritus/Debris catch-all, {'Zoop-No': ['Detritus', 'Small_Organism']} for Erie's).
    """
    extended_map = dict(class_folder_map)
    for leaf, extra_folders in (extra_leaf_folders or {}).items():
        extended_map[leaf] = extended_map.get(leaf, [leaf]) + list(extra_folders)

    folder_to_leaf = {}
    for leaf in hiernet_classes:
        for folder in extended_map.get(leaf, [leaf]):
            folder_to_leaf[folder] = leaf
    return folder_to_leaf


# ---------------------------------------------------------------------------
# CSV_Data sample identity -- which collection run a crop's original particle came from.
# Crop filenames retain their source CSV_Data/<stem>_data.csv's stem as a prefix (e.g.
# '<stem>_000005_1.tif'), so this recovers it without needing to reopen any CSVs.
# ---------------------------------------------------------------------------
def build_sample_stems(csv_data_dir):
    """Canonical sample identifiers (CSV_Data/<stem>_data.csv -> stem), longest first so a
    prefix match in assign_sample always picks the most specific stem."""
    stems = [f[:-len('_data.csv')] for f in os.listdir(csv_data_dir) if f.endswith('_data.csv')]
    return sorted(stems, key=len, reverse=True)


def assign_sample(image_filename, sample_stems, unmatched='Unmatched'):
    """Which CSV_Data run an image crop's filename came from (see build_sample_stems)."""
    return next((stem for stem in sample_stems if image_filename.startswith(stem + '_')), unmatched)


def parse_crop_filename(image_filename, sample_stems):
    """Split a crop filename ('<stem>_<image_num>_<particle_id>.<ext>') into the source-CSV
    stem (see build_sample_stems/assign_sample), the montage 'Image File' it was cropped from
    ('<stem>_<image_num>.tif' -- CSV_Data's own column, not to be confused with this crop's own
    filename), and its 'Particle ID' within that montage -- the exact fields a CSV_Data row is
    keyed by (jointly with 'Class'; Image File + Particle ID alone can collide across different
    classes cropped from the same montage -- see build_csv_metadata_index). Returns None if no
    stem prefix-matches."""
    stem = assign_sample(image_filename, sample_stems, unmatched=None)
    if stem is None:
        return None
    rest = os.path.splitext(image_filename[len(stem) + 1:])[0]
    image_num, particle_id = rest.rsplit('_', 1)
    return stem, f'{stem}_{image_num}.tif', int(particle_id)


def _csv_column_to_snake_case(name):
    return 'csv_' + re.sub(r'\s+', '_', name.replace('(', '').replace(')', '').strip().lower())


def build_csv_metadata_index(csv_data_dir, sample_stems):
    """Every CSV_Data column besides 'Particle ID'/'Class'/'Image File' (pure join-key
    identifiers, already covered by predictions.csv's own image_file/actual_class), renamed via
    _csv_column_to_snake_case (e.g. 'Area (ABD)' -> 'csv_area_abd') to match predictions.csv's
    snake_case style and avoid clashing with its existing columns.

    Returns {stem: {(Image File, Particle ID, Class): {csv_col: value, ...}}}. csv_metadata_for
    is the lookup half of this -- see its class_aliases param for why the Class key isn't always
    just the crop's on-disk folder name.
    """
    index = {}
    for stem in sample_stems:
        df = pd.read_csv(os.path.join(csv_data_dir, f'{stem}_data.csv'))
        metadata_cols = [c for c in df.columns if c not in ('Particle ID', 'Class', 'Image File')]
        keys = list(zip(df['Image File'], df['Particle ID'], df['Class']))
        renamed = df[metadata_cols].rename(columns=_csv_column_to_snake_case)
        index[stem] = dict(zip(keys, renamed.to_dict('records')))
    return index


def csv_metadata_for(image_filename, actual_class, sample_stems, csv_metadata_index, class_aliases=None):
    """The build_csv_metadata_index row for one crop, or None if its filename doesn't
    prefix-match any known stem or no CSV_Data row matches its (Image File, Particle ID,
    Class). A class_aliases value may be a single CSV Class or a tuple of them, tried in order
    -- for a folder pooling crops extracted under several CSV Class values (e.g. quetico's
    'Cladocera' folder, which holds both 'Cladocera' and merged-in 'Cladoceran' crops)."""
    parsed = parse_crop_filename(image_filename, sample_stems)
    if parsed is None:
        return None
    stem, image_file, particle_id = parsed
    expected = (class_aliases or {}).get(actual_class, actual_class)
    candidates = (expected,) if isinstance(expected, str) else expected
    stem_index = csv_metadata_index.get(stem, {})
    return next((stem_index[(image_file, particle_id, c)] for c in candidates
                 if (image_file, particle_id, c) in stem_index), None)


def excluded_samples_from_original_names(dataset_dir, patterns):
    """Canonical sample stems whose ORIGINAL pre-rename CSV name (before
    pre_process_data.ipynb's canonicalization pass -- see <dataset_dir>/_original_csv_names.csv)
    matched any of `patterns` (e.g. '<run>_TestAI_data_Decomp.csv' with pattern '_Decomp' ->
    exclude '<run>'). Returns an empty set if there are no patterns or no such mapping file
    (dataset has no renamed/flagged CSVs)."""
    mapping_path = os.path.join(dataset_dir, '_original_csv_names.csv')
    if not patterns or not os.path.exists(mapping_path):
        return set()
    mapping = pd.read_csv(mapping_path)
    pattern = '|'.join(re.escape(p) for p in patterns)
    flagged = mapping[mapping['original_name'].str.contains(pattern, regex=True)]
    return {name[:-len('_data.csv')] for name in flagged['renamed_to']}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class LabeledFolderDataset(Dataset):
    """Every *.tif under root_dir/<folder>/, folder name = raw ground-truth label. Skips
    exclude_dirs and anything that isn't a directory (e.g. a stray CSV at the top level).

    keep_fn, if given, is called with each bare filename ('<sample_stem>_000005_1.tif') and
    only kept when it returns True -- used to drop crops sourced from a flagged CSV_Data run
    (see build_sample_stems / assign_sample / excluded_samples_from_original_names)."""

    def __init__(self, root_dir, transform, exclude_dirs=frozenset(), keep_fn=None):
        self.transform = transform
        self.samples = []
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if folder in exclude_dirs or not os.path.isdir(folder_path):
                continue
            for filename in sorted(os.listdir(folder_path)):
                if filename.lower().endswith('.tif') and (keep_fn is None or keep_fn(filename)):
                    self.samples.append((os.path.join(folder_path, filename), folder))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, raw_class = self.samples[idx]
        image = Image.open(path).convert('L')
        image = self.transform(image).repeat(3, 1, 1)
        return image, path, raw_class


class UnclassifiedFolderDataset(Dataset):
    """Flat folder of *.tif with no ground-truth label (e.g. Emily's Particle/ 'To be
    Classified' crops)."""

    def __init__(self, root_dir, transform):
        self.transform = transform
        self.paths = sorted(
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith('.tif')
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert('L')
        image = self.transform(image).repeat(3, 1, 1)
        return image, path


# ---------------------------------------------------------------------------
# Per-image cascade inference -> shared predictions.csv row fields
# ---------------------------------------------------------------------------
class CascadeResult(NamedTuple):
    image_path: str
    cascade_path: list        # node-id path, root -> deepest node reached
    cascade_confidences: list  # per-decision local confidence
    predicted_1: str
    predicted_2: str
    conf_by_level: dict       # {1: .., 2: .., 3: ..}
    pass_by_level: dict       # {1: .., 2: .., 3: ..}


def predict_batch(model, images, image_paths, leaf_ids_sorted, leaf_ids_tensor, id_to_name, thresholds):
    """One batch of greedy-cascade inference -> one CascadeResult per image, same order as
    image_paths. Assumes a <= 3-level taxonomy, matching the rest of this module's t1/t2/t3
    convention."""
    cascade_paths, cascade_confidences = predict_with_confidences(model, images)
    leaf_probs, _ = model(images)  # (B, num_nodes) path-product probability per node
    ranked_order = torch.argsort(leaf_probs[:, leaf_ids_tensor], dim=1, descending=True)

    results = []
    for i in range(len(image_paths)):
        cascade_path, confs = cascade_paths[i], cascade_confidences[i]
        predicted_1 = id_to_name[cascade_path[-1]]

        ranked_leaf_ids = [leaf_ids_sorted[j] for j in ranked_order[i].tolist()]
        predicted_2 = id_to_name[next(nid for nid in ranked_leaf_ids if id_to_name[nid] != predicted_1)]

        conf_by_level = {d: (confs[d - 1] if d <= len(confs) else np.nan) for d in (1, 2, 3)}
        pass_by_level = {
            d: (conf_by_level[d] >= thresholds[d - 1]) if not np.isnan(conf_by_level[d]) else np.nan
            for d in (1, 2, 3)
        }

        results.append(CascadeResult(
            image_path=image_paths[i], cascade_path=cascade_path, cascade_confidences=confs,
            predicted_1=predicted_1, predicted_2=predicted_2,
            conf_by_level=conf_by_level, pass_by_level=pass_by_level,
        ))
    return results


# ---------------------------------------------------------------------------
# Ambiguous-class settle-depth summary (MultiClass / Partial / Unknown, etc.) -- these have no
# real species label, so there's nothing to grade; instead this shows, for each class, what
# DEPTH the confidence-threshold cascade actually settles at (hiernet_eval's own
# apply_confidence_threshold back-off, via settle_depth_distribution).
# ---------------------------------------------------------------------------
def ambiguous_class_summary_table(ambiguous_classes, ambiguous_paths, ambiguous_confidences, thresholds, hierarchy):
    rows = []
    for cls in ambiguous_classes:
        n = len(ambiguous_paths[cls])
        dist = settle_depth_distribution(thresholds, ambiguous_paths[cls], ambiguous_confidences[cls], hierarchy)
        rows.append({
            'N': n,
            'Abstained': dist.get('abstained', 0) / n if n else float('nan'),
            'Lowest level prediction': dist.get('level 3', 0) / n if n else float('nan'),
            'Second lowest level prediction': dist.get('level 2', 0) / n if n else float('nan'),
            'First level prediction': dist.get('level 1', 0) / n if n else float('nan'),
        })
    return pd.DataFrame(rows, index=list(ambiguous_classes))


# ---------------------------------------------------------------------------
# Top-K predicted-class breakdown for a raw on-disk folder -- what did a class's crops (e.g.
# 'Small_Nauplius', or a no-real-label folder like 'Unknown') actually get predicted as, and
# what share of that folder's own total does each of those top predictions represent. Works
# for any actual_class value in predictions_df regardless of in_taxonomy, since it only looks
# at predicted_class_1 -- there's no true label needed.
# ---------------------------------------------------------------------------
def top_predictions_table(predictions_df, classes, top_k=3):
    rows = []
    for cls in classes:
        sub = predictions_df[predictions_df['actual_class'] == cls]
        n = len(sub)
        row = {'N': n}
        counts = sub['predicted_class_1'].value_counts() if n else pd.Series(dtype=int)
        for k in range(top_k):
            if k < len(counts):
                name, cnt = counts.index[k], counts.iloc[k]
                row[f'#{k + 1} predicted'] = f'{name} ({cnt / n:.1%})'
            else:
                row[f'#{k + 1} predicted'] = 'N/A' if n == 0 else '—'
        rows.append(row)
    return pd.DataFrame(rows, index=list(classes))


# ---------------------------------------------------------------------------
# Leaf-class metrics table (N / Accuracy / Precision / Recall / F1 per leaf, worst-accuracy
# first, plus a bottom Total row: micro accuracy, macro precision/recall/F1).
# ---------------------------------------------------------------------------
def leaf_metrics_table(sub_df, hierarchy=None, label_to_ids=None):
    """
    hierarchy/label_to_ids are optional; pass both when sub_df['true_leaf'] can contain a
    coarse (non-leaf) label -- e.g. a 'Rotifer' crop with no species-level manual ID, scored
    correct against any of Rotifer's descendant leaves (see run_evaluation's 'correct' column
    and classify_id_outcome's own subtree matching). predicted_class_1 is always an actual
    leaf name, so such a row's Precision/Recall/F1 can't be computed against it as an exact
    label the usual sklearn way -- those cells are NaN (rendered N/A) and left out of the
    macro-average Total, while N/Accuracy (and the Total's N/micro-accuracy) still include it.
    """
    from sklearn.metrics import precision_recall_fscore_support

    leaf_labels = sorted(sub_df['true_leaf'].unique())
    coarse_labels = ({l for l in leaf_labels if not hierarchy.is_leaf(label_to_ids[l])}
                      if hierarchy is not None else set())
    scoring_labels = [l for l in leaf_labels if l not in coarse_labels]

    prec, rec, f1, _ = precision_recall_fscore_support(
        sub_df['true_leaf'], sub_df['predicted_class_1'],
        labels=leaf_labels, average=None, zero_division=0,
    )
    prec = pd.Series(prec, index=leaf_labels)
    rec = pd.Series(rec, index=leaf_labels)
    f1 = pd.Series(f1, index=leaf_labels)
    for lbl in coarse_labels:
        prec[lbl] = rec[lbl] = f1[lbl] = float('nan')

    acc = sub_df.groupby('true_leaf')['correct'].mean().reindex(leaf_labels)
    n = sub_df.groupby('true_leaf').size().reindex(leaf_labels)

    df = pd.DataFrame({
        'N': n.values, 'Accuracy': acc.values,
        'Precision': prec.values, 'Recall': rec.values, 'F1': f1.values,
    }, index=leaf_labels).sort_values('Accuracy')  # worst first

    micro_acc = sub_df['correct'].mean()
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        sub_df['true_leaf'], sub_df['predicted_class_1'],
        labels=scoring_labels, average='macro', zero_division=0,
    )
    df.loc['Total'] = [len(sub_df), micro_acc, prec_macro, rec_macro, f1_macro]
    return df


# ---------------------------------------------------------------------------
# Shared pooled-credit metrics: given parallel predicted/true class labels and a boolean "hit"
# mask, produce per-class N/Recall/Precision/F1 -- Recall grouped by true, Precision grouped by
# predicted, same population feeding both (so a hit born of asymmetric credit, like an
# '_Unknown' sibling call, naturally lands as a Recall win for the true class's row and a
# Precision win for the predicted class's row, without inflating any single row's own number)
# -- plus the micro-accuracy/macro-P/R/F1 tuple for a Total row.
# ---------------------------------------------------------------------------
def _pooled_class_metrics(true_values, predicted_values, hit_mask, classes):
    df = pd.DataFrame({'true': list(true_values), 'pred': list(predicted_values), 'hit': list(hit_mask)})

    n = df.groupby('true').size().reindex(classes)
    recall = df.groupby('true')['hit'].mean().reindex(classes)

    pred_n = df.groupby('pred').size()
    pred_hits = df[df['hit']].groupby('pred').size()
    precision = (pred_hits / pred_n).reindex(classes)

    f1 = 2 * precision * recall / (precision + recall)
    totals = (len(df), df['hit'].mean(), precision.mean(), recall.mean(), f1.mean())
    return n, recall, precision, f1, totals


def _is_unknown_sibling_hit(pred_name, true_name, pred_id, true_id, hierarchy):
    """True if pred_name is a '<Branch>_Unknown'-style catch-all sharing true_name's own parent
    (siblings under the same branch, pred_name just unresolved past that point) -- the same
    relationship classify_id_outcome credits as correct_incomplete at the leaf level, applied
    generically at any hierarchy depth (e.g. Zoop_Unknown vs. Copepoda are both level-2 children
    of Zoop-yes, so a true Copepoda-branch sample predicted Zoop_Unknown gets this credit too)."""
    return pred_name.endswith('_Unknown') and hierarchy.parent(pred_id) == hierarchy.parent(true_id)


# ---------------------------------------------------------------------------
# Leaf-class metrics table, but a specific-species leaf backed off to its branch's own
# '<Branch>_Unknown' sibling (classify_id_outcome's correct_incomplete) counts as a hit too --
# same shape/style as leaf_metrics_table, just a looser definition of "correct".
# ---------------------------------------------------------------------------
def loosened_leaf_metrics_table(predicted_leaves, true_leaves, hierarchy, exclude_leaves=frozenset()):
    """
    predicted_leaves/true_leaves: parallel sequences of leaf NAMES (predicted_class_1-style;
    always an actual leaf, never a coarse/internal node).

    A sample is a "hit" if classify_id_outcome(predicted, true, hierarchy) is correct_all OR
    correct_incomplete. correct_incomplete only ever happens when the predicted leaf is itself
    a '<Branch>_Unknown' leaf whose parent is a genuine ancestor of the true leaf (see
    classify_id_outcome's docstring -- an ordinary species leaf can never be the predicted side
    of a correct_incomplete outcome), so such a hit counts twice, toward two different rows'
    different metrics: a Recall/Accuracy hit for the TRUE leaf's row (it was identified to the
    right branch, just not to species), and a Precision hit for the PREDICTED '_Unknown' leaf's
    row (that prediction was, in fact, a reasonable call) -- see _pooled_class_metrics. This is
    not double-counting toward any single number -- Recall and Precision are different rows'
    different metrics -- and leaves ordinary species leaves' own Precision untouched (still
    exact-match only, identical to leaf_metrics_table's), since they can never receive this
    extra credit.

    exclude_leaves: leaf names dropped from the population entirely -- no row/N/Total credit --
    but only on the TRUE side (e.g. {'Zoop-No'} to drop Zoop-No's own huge, uninteresting
    population). A sample whose true leaf is kept but was mispredicted as an excluded leaf (e.g.
    a real Copepoda_Unknown crop predicted as Zoop-No) still counts under its true leaf's row,
    as an ordinary miss -- it just never forms an excluded-leaf row/precision bucket of its own,
    since that leaf was never one of true_leaf's own values to begin with.
    """
    name_to_id = {hierarchy.nodes[nid].name: nid for nid in hierarchy.nodes}

    df = pd.DataFrame({'true_leaf': list(true_leaves), 'predicted_leaf': list(predicted_leaves)})
    if exclude_leaves:
        df = df[~df['true_leaf'].isin(exclude_leaves)]

    hit = [
        classify_id_outcome(name_to_id[pred], name_to_id[true], hierarchy) in ('correct_all', 'correct_incomplete')
        for pred, true in zip(df['predicted_leaf'], df['true_leaf'])
    ]

    leaf_labels = sorted(df['true_leaf'].unique())
    n, recall, precision, f1, totals = _pooled_class_metrics(df['true_leaf'], df['predicted_leaf'], hit, leaf_labels)

    table = pd.DataFrame({
        'N': n.values, 'Accuracy': recall.values,
        'Precision': precision.values, 'Recall': recall.values, 'F1': f1.values,
    }, index=leaf_labels).sort_values('Accuracy')

    n_total, acc_total, prec_total, rec_total, f1_total = totals
    table.loc['Total'] = [n_total, acc_total, prec_total, rec_total, f1_total]
    return table


# ---------------------------------------------------------------------------
# loosened_leaf_metrics_table computed from a training checkpoint's own held-out
# predictions.npz (hierroute.trainer.Trainer.predict(..., save=True)) -- the model's own
# internal test split, NOT Erie/Emily's manually-classified crops.
# ---------------------------------------------------------------------------
def training_result_accuracy_table(model_dir, hierarchy, id_to_name):
    """None if model_dir has no predictions.npz. Otherwise, 'predictions'/'targets' are the
    model's raw greedy leaf node id and the true leaf node id, one pair per held-out sample --
    fed straight into loosened_leaf_metrics_table (no exclude_leaves; this is the model's own
    full test split, not Erie's Zoop-No-dominated crops)."""
    npz_path = Path(model_dir) / 'predictions.npz'
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    predicted_leaves = [id_to_name[i] for i in data['predictions'].tolist()]
    true_leaves = [id_to_name[i] for i in data['targets'].tolist()]
    return loosened_leaf_metrics_table(predicted_leaves, true_leaves, hierarchy)


# ---------------------------------------------------------------------------
# Per-hierarchy-level Accuracy/Precision/Recall/F1 -- e.g. level 1 = Zoop-yes vs Zoop-No,
# level 2 = Copepoda/Cladocera/Rotifer/Zoop_Unknown, level 3 = the species-level leaves --
# each level's own independent flat classification, ignoring every other level entirely (same
# idea as hierroute.trainer.Trainer.evaluate's level_results, computed here from predicted vs.
# true LEAF names instead of from a live model pass).
# ---------------------------------------------------------------------------
def hierarchy_level_metrics_table(predicted_leaves, true_leaves, hierarchy, max_depth=3):
    """
    Returns one combined DataFrame (index = class name, a 'Level' column, plus N/Accuracy/
    Precision/Recall/F1), rows sorted worst-accuracy-first within each level's own block, with
    a 'Total' row pinned last per block -- Total's Accuracy is the sample-weighted (micro)
    accuracy across that level's whole population; Precision/Recall/F1 are macro (equally
    weighted across that level's classes), matching leaf_metrics_table's convention.

    A sample belongs to level d's population only if its TRUE leaf's own depth reaches d (e.g.
    a true Zoop-No sample has no level-2/3 truth, so it's excluded from those levels' tables
    entirely). Its predicted-side class at depth d is the ancestor of predicted_class_1 at that
    depth, or '(none)' if the predicted leaf's own path doesn't reach that deep (e.g. predicted
    Zoop-No when checking level 2/3) -- '(none)' is never one of the scored labels, so it can
    never contribute to any real class's Precision, only cost its true class a Recall miss.

    A prediction that lands on a '_Unknown' sibling of the true class at that same depth (see
    _is_unknown_sibling_hit -- e.g. true Copepoda / predicted Zoop_Unknown at level 2, both
    children of Zoop-yes) counts as a hit too, pooled the same asymmetric way as
    loosened_leaf_metrics_table: a Recall win for the true class's row, a Precision win for the
    predicted '_Unknown' class's row.
    """
    name_to_id = {hierarchy.nodes[nid].name: nid for nid in hierarchy.nodes}
    true_ids = [name_to_id[t] for t in true_leaves]
    pred_ids = [name_to_id[p] for p in predicted_leaves]

    rows = []
    for d in range(1, max_depth + 1):
        idx = [i for i, tid in enumerate(true_ids) if hierarchy.nodes[tid].depth >= d]
        if not idx:
            continue
        true_at_d_id = [hierarchy.get_path_to_root(true_ids[i])[d] for i in idx]
        pred_at_d_id = [
            hierarchy.get_path_to_root(pred_ids[i])[d] if hierarchy.nodes[pred_ids[i]].depth >= d else None
            for i in idx
        ]
        true_at_d = [hierarchy.nodes[nid].name for nid in true_at_d_id]
        pred_at_d = [hierarchy.nodes[nid].name if nid is not None else '(none)' for nid in pred_at_d_id]

        hit = [
            (pn == tn) or (pid is not None and _is_unknown_sibling_hit(pn, tn, pid, tid, hierarchy))
            for pn, tn, pid, tid in zip(pred_at_d, true_at_d, pred_at_d_id, true_at_d_id)
        ]

        classes = sorted(set(true_at_d))
        n, recall, precision, f1, totals = _pooled_class_metrics(true_at_d, pred_at_d, hit, classes)

        level_df = pd.DataFrame({
            'N': n.values, 'Accuracy': recall.values,
            'Precision': precision.values, 'Recall': recall.values, 'F1': f1.values,
        }, index=classes).sort_values('Accuracy')

        for cls, row in level_df.iterrows():
            rows.append({'Level': d, 'Class': cls, **row.to_dict()})

        n_total, acc_total, prec_total, rec_total, f1_total = totals
        rows.append({'Level': d, 'Class': 'Total', 'N': n_total, 'Accuracy': acc_total,
                     'Precision': prec_total, 'Recall': rec_total, 'F1': f1_total})

    table = pd.DataFrame(rows).set_index('Class')
    return table[['Level', 'N', 'Accuracy', 'Precision', 'Recall', 'F1']]


# ---------------------------------------------------------------------------
# Per-class ID-outcome table (hiernet_eval.per_class_id_outcomes), styled-table-ready.
# ---------------------------------------------------------------------------
def id_outcome_table_from(model, loader, thresholds, hierarchy, id_to_name, device,
                           paths_list, confidences_list, true_leaves_list,
                           gate_by_confidence=True, outcome_keys=ID_OUTCOME_KEYS):
    """N + outcome_keys columns, plus a Total row (N-weighted/pooled average of each
    outcome column -- what fraction of every sample overall, not just per class, landed in
    that outcome bucket; still sums to ~1 like every other row).

    gate_by_confidence=False grades the raw greedy leaf instead of the confidence-threshold
    cascade's accepted node -- pass outcome_keys=ID_OUTCOME_KEYS_NO_ABSTAIN alongside it, since
    'abstained' can't occur."""
    props = per_class_id_outcomes(
        model, loader, thresholds, hierarchy, id_to_name, device,
        predictions=(paths_list, confidences_list, true_leaves_list),
        gate_by_confidence=gate_by_confidence, outcome_keys=outcome_keys,
    )
    table = pd.DataFrame(props).T[list(outcome_keys)].round(3)
    n_by_name = pd.Series(true_leaves_list).map(id_to_name).value_counts()
    table.insert(0, 'N', n_by_name.reindex(table.index).values)

    weights = table['N'] / table['N'].sum()
    totals = table[list(outcome_keys)].multiply(weights, axis=0).sum()
    table.loc['Total'] = [table['N'].sum(), *totals]
    return table, props


# ---------------------------------------------------------------------------
# Per-sample (CSV_Data run) ID-outcome breakdown -- same classify_id_outcome rubric as
# hiernet_eval.per_class_id_outcomes, just grouped by an arbitrary per-image label (here, the
# source CSV_Data run from assign_sample) instead of by true leaf class. The result is the
# same shape plot_id_outcome_stacked_bar draws, so no new plotting code is needed -- only the
# x-axis category (dict key) changes.
# ---------------------------------------------------------------------------
def per_group_id_outcomes(thresholds, hierarchy, paths_list, confidences_list, true_leaves_list, group_labels,
                           gate_by_confidence=True, outcome_keys=ID_OUTCOME_KEYS):
    outcomes_by_group = {}
    for path, conf, true_leaf, group in zip(paths_list, confidences_list, true_leaves_list, group_labels):
        predicted = apply_confidence_threshold(path, conf, thresholds) if gate_by_confidence else path[-1]
        outcome = classify_id_outcome(predicted, true_leaf, hierarchy)
        outcomes_by_group.setdefault(group, []).append(outcome)
    return {
        group: {k: outs.count(k) / len(outs) for k in outcome_keys}
        for group, outs in outcomes_by_group.items()
    }


# ---------------------------------------------------------------------------
# Fixed-seed per-class capping -- keeps an oversized leaf (e.g. Zoop-No, 12k+ samples) from
# dominating a table purely on volume relative to leaves with far fewer samples.
# ---------------------------------------------------------------------------
def capped_indices_by_group(group_values, max_per_class, seed):
    """Indices of group_values (its own pandas index if it's a Series/column, else 0..len-1),
    downsampled per distinct value to at most max_per_class via a fixed-seed random subsample;
    groups at or under the cap pass through whole."""
    series = pd.Series(group_values)
    indices = []
    for _, group in series.groupby(series):
        idxs = group.index.tolist()
        if len(idxs) > max_per_class:
            idxs = pd.Series(idxs).sample(n=max_per_class, random_state=seed).tolist()
        indices.extend(idxs)
    return indices


# ---------------------------------------------------------------------------
# Styled-table renderer (originally build_comparison_tables.py's look, via
# runs/claude_helper/render_comparison_tables.py) -- adapted to not bold a degenerate
# single-column "best".
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath

BLUE_RAMP = [
    '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec',
    '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab',
    '#184f95', '#104281', '#0d366b',
]
CMAP = LinearSegmentedColormap.from_list('seq_blue', BLUE_RAMP)

TEXT_PRIMARY = '#0b0b0b'
TEXT_ON_DARK = '#ffffff'
HEADER_BG = '#f0efec'
GRID_LINE = '#000000'
GRID_LINEWIDTH = 1.2

TITLE_FONTSIZE = 15
HEADER_FONTSIZE = 12
CELL_FONTSIZE = 12
ROW_LABEL_FONTSIZE = 12
FOOTNOTE_FONTSIZE = 9


def text_width_in(text, fontsize, weight='normal'):
    fp = FontProperties(size=fontsize, weight=weight)
    widths = [TextPath((0, 0), line, prop=fp).get_extents().width for line in text.split('\n')]
    return max(widths, default=0) / 72.0


def relative_luminance(rgb):
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_color_for(rgb):
    l_bg = relative_luminance(rgb)
    contrast_black = (l_bg + 0.05) / 0.05
    contrast_white = 1.05 / (l_bg + 0.05)
    return TEXT_ON_DARK if contrast_white > contrast_black else TEXT_PRIMARY


def wrap_header(label, max_chars=16):
    if len(label) <= max_chars:
        return label
    idx = label.rfind(' ', 0, max_chars)
    if idx == -1:
        idx = label.find(' ')
    return label if idx == -1 else label[:idx] + '\n' + label[idx + 1:]


def render_table(df, title, out_path, summary_row=None, value_fmt='{:.3f}',
                  highlight_row_max=True, footnote=None, count_cols=frozenset(), label_cols=()):
    """
    label_cols: names of df columns to render as extra plain-text label columns (white cells,
    left-aligned, own width, never heat-mapped) between the index-label column and the
    heat-mapped value columns -- e.g. splitting a combined "Emily — 08-25" index into separate
    test_data / model_date columns. Every other column is treated as a numeric value column.
    """
    label_cols = list(label_cols)
    value_cols = [c for c in df.columns if c not in label_cols]

    summary_rows = {summary_row} if isinstance(summary_row, str) else set(summary_row or [])
    n_rows, n_value_cols = len(df), len(value_cols)
    headers = [wrap_header(c) for c in value_cols]
    label_headers = [wrap_header(c) for c in label_cols]
    header_lines = max([h.count('\n') + 1 for h in headers + label_headers], default=1)

    def format_val(col, val):
        if pd.isna(val):
            return 'N/A'
        return f'{int(val):,}' if col in count_cols else value_fmt.format(val)

    CELL_PAD = 0.25

    row_values = [format_val(col, v) for col in value_cols for v in df[col]]
    max_value_w = max((text_width_in(v, CELL_FONTSIZE, 'bold') for v in row_values), default=0)
    col_w = max(1.1, max_value_w + 2 * CELL_PAD)
    for header in headers:
        col_w = max(col_w, text_width_in(header, HEADER_FONTSIZE, 'bold') + 2 * CELL_PAD)

    row_h = 0.55
    header_h = row_h * max(1.4, 0.8 * header_lines + 0.6)

    # Left-side label block: the DataFrame index, plus one column per label_cols entry, each
    # sized to its own content/header (not a single shared width).
    index_texts = [str(n) for n in df.index]
    index_w = max(text_width_in(t, ROW_LABEL_FONTSIZE, 'bold') for t in index_texts) + 2 * CELL_PAD
    label_widths = [index_w]
    for col, header in zip(label_cols, label_headers):
        texts = [str(v) for v in df[col]]
        w = max(text_width_in(t, ROW_LABEL_FONTSIZE, 'bold') for t in texts)
        w = max(w, text_width_in(header, HEADER_FONTSIZE, 'bold')) + 2 * CELL_PAD
        label_widths.append(w)
    label_offsets = [sum(label_widths[:k]) for k in range(len(label_widths))]
    label_w = sum(label_widths)

    fig_w = label_w + col_w * n_value_cols

    # Wrap the footnote to fig_w instead of drawing it as one unbroken line -- an oversized
    # single-line footnote is wider than every other artist in the figure, which makes
    # fig.tight_layout() shrink the axes to fit it, squishing the (fixed-point-size) column
    # text into now-narrower column rectangles and causing header/cell text to overlap.
    if footnote:
        char_w = text_width_in('n' * 80, FOOTNOTE_FONTSIZE) / 80 or 0.05
        footnote = '\n'.join(textwrap.wrap(footnote, width=max(20, int(fig_w / char_w))))
    footnote_lines = footnote.count('\n') + 1 if footnote else 0
    footnote_h = 0.16 * footnote_lines + 0.22 if footnote else 0.0

    fig_h = 0.9 + header_h + row_h * n_rows + footnote_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', color=TEXT_PRIMARY, loc='left', x=0, pad=14)

    header_y = 0
    ax.add_patch(Rectangle((0, header_y), label_widths[0], header_h, facecolor=HEADER_BG,
                            edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
    for k, (header, w) in enumerate(zip(label_headers, label_widths[1:]), start=1):
        x = label_offsets[k]
        ax.add_patch(Rectangle((x, header_y), w, header_h, facecolor=HEADER_BG,
                                edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
        ax.text(x + w / 2, header_y + header_h / 2, header, ha='center', va='center',
                fontsize=HEADER_FONTSIZE, fontweight='bold', color=TEXT_PRIMARY, linespacing=1.3)
    for j, header in enumerate(headers):
        x = label_w + col_w * j
        ax.add_patch(Rectangle((x, header_y), col_w, header_h, facecolor=HEADER_BG,
                                edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
        ax.text(x + col_w / 2, header_y + header_h / 2, header, ha='center', va='center',
                fontsize=HEADER_FONTSIZE, fontweight='bold', color=TEXT_PRIMARY, linespacing=1.3)

    for i, (row_name, row) in enumerate(df.iterrows()):
        y = header_h + row_h * i
        is_summary = row_name in summary_rows
        row_label_weight = 'bold' if is_summary else 'normal'

        ax.add_patch(Rectangle((0, y), label_widths[0], row_h, facecolor='white',
                                edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
        ax.text(CELL_PAD, y + row_h / 2, str(row_name), ha='left', va='center',
                fontsize=ROW_LABEL_FONTSIZE, fontweight=row_label_weight, color=TEXT_PRIMARY)

        for k, (col, w) in enumerate(zip(label_cols, label_widths[1:]), start=1):
            x = label_offsets[k]
            ax.add_patch(Rectangle((x, y), w, row_h, facecolor='white',
                                    edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
            ax.text(x + CELL_PAD, y + row_h / 2, str(row[col]), ha='left', va='center',
                    fontsize=ROW_LABEL_FONTSIZE, fontweight=row_label_weight, color=TEXT_PRIMARY)

        scored = row[value_cols].drop(labels=[c for c in count_cols if c in value_cols])
        best_col = scored.idxmax() if len(scored) else None
        for j, col in enumerate(value_cols):
            val = row[col]
            x = label_w + col_w * j
            is_count = col in count_cols
            is_na = pd.isna(val)
            bg = 'white' if (is_count or is_na) else CMAP(val)
            ax.add_patch(Rectangle((x, y), col_w, row_h, facecolor=bg,
                                    edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
            txt_color = TEXT_PRIMARY if (is_count or is_na) else text_color_for(bg)
            is_best = highlight_row_max and not is_count and not is_na and (col == best_col) and n_value_cols > 1
            weight = row_label_weight if (is_count or is_na) else ('bold' if is_best else 'normal')
            ax.text(x + col_w / 2, y + row_h / 2, format_val(col, val),
                    ha='center', va='center', fontsize=CELL_FONTSIZE, color=txt_color,
                    fontweight=weight)

    table_h = header_h + row_h * n_rows
    ax.add_patch(Rectangle((0, 0), fig_w, table_h, facecolor='none',
                            edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH * 1.6, zorder=10))

    if footnote:
        ax.text(0, table_h + footnote_h / 2, footnote, ha='left', va='center',
                fontsize=FOOTNOTE_FONTSIZE, style='italic', color='dimgray', linespacing=1.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved -> {out_path}')


# ---------------------------------------------------------------------------
# Dedicated renderer for hierarchy_level_metrics_table: color encodes hierarchy level (not
# value magnitude), since that table's rows are grouped by level rather than being one flat
# per-class list.
# ---------------------------------------------------------------------------
LEVEL_COLORS = {1: '#1f77b4', 2: '#d62728', 3: '#2ca02c'}  # blue, red, green


def _blend_with_white(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(alpha * c + (1 - alpha) for c in rgb)


def render_hierarchy_level_table(df, title, out_path, level_colors=None, value_fmt='{:.3f}',
                                  highlight_row_max=True, footnote=None, count_cols=frozenset()):
    """
    df: hierarchy_level_metrics_table's output (index = class name, columns 'Level'/N/Accuracy/
    Precision/Recall/F1, rows already sorted worst-Accuracy-first within each level's own block
    with a 'Total' row pinned last per block -- this renderer trusts that order and does not
    re-sort).

    Unlike render_table, color here encodes df['Level'] (via level_colors, default
    LEVEL_COLORS) rather than each cell's own value: every row's cells are bordered in its
    level's color, a 'Total' row's cells are additionally filled with that color at 20% opacity,
    and the value columns are otherwise plain white -- no CMAP heat-map, since color now means
    level, not magnitude. A small legend under the title spells out which color is which level.
    """
    level_colors = level_colors or LEVEL_COLORS
    value_cols = [c for c in df.columns if c != 'Level']
    n_rows, n_value_cols = len(df), len(value_cols)
    headers = [wrap_header(c) for c in value_cols]
    header_lines = max([h.count('\n') + 1 for h in headers], default=1)

    def format_val(col, val):
        if pd.isna(val):
            return 'N/A'
        return f'{int(val):,}' if col in count_cols else value_fmt.format(val)

    CELL_PAD = 0.25

    row_values = [format_val(col, v) for col in value_cols for v in df[col]]
    max_value_w = max((text_width_in(v, CELL_FONTSIZE, 'bold') for v in row_values), default=0)
    col_w = max(1.1, max_value_w + 2 * CELL_PAD)
    for header in headers:
        col_w = max(col_w, text_width_in(header, HEADER_FONTSIZE, 'bold') + 2 * CELL_PAD)

    row_h = 0.55
    header_h = row_h * max(1.4, 0.8 * header_lines + 0.6)

    index_texts = [str(n) for n in df.index]
    label_w = max(text_width_in(t, ROW_LABEL_FONTSIZE, 'bold') for t in index_texts) + 2 * CELL_PAD

    fig_w = label_w + col_w * n_value_cols

    if footnote:
        char_w = text_width_in('n' * 80, FOOTNOTE_FONTSIZE) / 80 or 0.05
        footnote = '\n'.join(textwrap.wrap(footnote, width=max(20, int(fig_w / char_w))))
    footnote_lines = footnote.count('\n') + 1 if footnote else 0
    footnote_h = 0.16 * footnote_lines + 0.22 if footnote else 0.0

    legend_h = row_h * 0.9
    fig_h = 0.9 + legend_h + header_h + row_h * n_rows + footnote_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', color=TEXT_PRIMARY, loc='left', x=0, pad=14)

    legend_x, swatch_w = 0.0, 0.28
    for level in sorted(level_colors):
        color = level_colors[level]
        ax.add_patch(Rectangle((legend_x, legend_h * 0.2), swatch_w, legend_h * 0.6,
                                facecolor=color, edgecolor='none'))
        label = f'Level {level}'
        ax.text(legend_x + swatch_w + 0.08, legend_h / 2, label, ha='left', va='center',
                fontsize=FOOTNOTE_FONTSIZE + 1, color=TEXT_PRIMARY)
        legend_x += swatch_w + 0.08 + text_width_in(label, FOOTNOTE_FONTSIZE + 1) + 0.35

    header_y = legend_h
    ax.add_patch(Rectangle((0, header_y), label_w, header_h, facecolor=HEADER_BG,
                            edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
    for j, header in enumerate(headers):
        x = label_w + col_w * j
        ax.add_patch(Rectangle((x, header_y), col_w, header_h, facecolor=HEADER_BG,
                                edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
        ax.text(x + col_w / 2, header_y + header_h / 2, header, ha='center', va='center',
                fontsize=HEADER_FONTSIZE, fontweight='bold', color=TEXT_PRIMARY, linespacing=1.3)

    for i, (row_name, row) in enumerate(df.iterrows()):
        y = header_y + header_h + row_h * i
        color = level_colors.get(int(row['Level']), GRID_LINE)
        is_total = (row_name == 'Total')
        row_label_weight = 'bold' if is_total else 'normal'
        cell_bg = _blend_with_white(color, 0.2) if is_total else 'white'
        border_w = GRID_LINEWIDTH * (1.6 if is_total else 1.1)

        ax.add_patch(Rectangle((0, y), label_w, row_h, facecolor=cell_bg, edgecolor=color, linewidth=border_w))
        ax.text(CELL_PAD, y + row_h / 2, str(row_name), ha='left', va='center',
                fontsize=ROW_LABEL_FONTSIZE, fontweight=row_label_weight, color=TEXT_PRIMARY)

        scored = row[value_cols].drop(labels=[c for c in count_cols if c in value_cols])
        best_col = scored.idxmax() if len(scored) else None
        for j, col in enumerate(value_cols):
            val = row[col]
            x = label_w + col_w * j
            is_count = col in count_cols
            is_na = pd.isna(val)
            ax.add_patch(Rectangle((x, y), col_w, row_h, facecolor=cell_bg, edgecolor=color, linewidth=border_w))
            is_best = highlight_row_max and not is_count and not is_na and (col == best_col) and n_value_cols > 1
            weight = row_label_weight if (is_count or is_na) else ('bold' if is_best else 'normal')
            ax.text(x + col_w / 2, y + row_h / 2, format_val(col, val),
                    ha='center', va='center', fontsize=CELL_FONTSIZE, color=TEXT_PRIMARY, fontweight=weight)

    table_h = header_h + row_h * n_rows
    ax.add_patch(Rectangle((0, header_y), fig_w, table_h, facecolor='none',
                            edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH * 1.6, zorder=10))

    if footnote:
        ax.text(0, header_y + table_h + footnote_h / 2, footnote, ha='left', va='center',
                fontsize=FOOTNOTE_FONTSIZE, style='italic', color='dimgray', linespacing=1.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved -> {out_path}')
