"""
grading.py — Scoring hierarchical predictions against manual labels.

Terminology:
    path        : greedy root -> leaf node-ID path the model took for one image.
    confidences : softmax confidence of each decision on that path (confidences[d] chose path[d + 1]).
    true node   : the node a labelled folder maps to. Usually a leaf, but it may be a coarser
                  node when the manual ID only committed to a coarser rank.

ID outcomes (where a prediction first disagrees with the manual ID):
    correct_all        -- predicted a leaf in the true node's subtree.
    correct_incomplete -- predicted a genuine ancestor of the true node, or a ``<Parent>_Unknown``
                          leaf whose parent is an ancestor (right branch, not resolved to species).
    incorrect_l{d}     -- agrees down to level d-1, wrong at level d (d >= 2).
    incorrect_all      -- wrong already at level 1.
    abstained          -- confidence gating rejected every level (only with thresholds).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .hierarchy import Hierarchy


# ---------------------------------------------------------------------------
# Outcome rubric
# ---------------------------------------------------------------------------
def outcome_keys(n_levels: int, gated: bool) -> tuple[str, ...]:
    """Outcome bucket keys, most to least correct."""
    keys = ["correct_all", "correct_incomplete"]
    keys += [f"incorrect_l{d}" for d in range(n_levels, 1, -1)]
    keys.append("incorrect_all")
    if gated:
        keys.append("abstained")
    return tuple(keys)


def outcome_label(key: str) -> str:
    """``'incorrect_l3'`` -> ``'Incorrect_L3'``."""
    return "_".join(part.upper() if part[0] == "l" and part[1:].isdigit() else part.capitalize()
                    for part in key.split("_"))


_OUTCOME_COLORS = {"correct_all": "#4C72B0", "correct_incomplete": "#55A868", "incorrect_all": "#C44E52",
                   "abstained": "#8172B2"}
_INCORRECT_LEVEL_COLORS = ("#E7B84B", "#DD8452", "#B07AA1", "#9C755F", "#BAB0AC")


def outcome_colors(keys: tuple[str, ...]) -> tuple[str, ...]:
    level_keys = [k for k in keys if k.startswith("incorrect_l")]
    colors = dict(_OUTCOME_COLORS)
    for i, k in enumerate(level_keys):
        colors[k] = _INCORRECT_LEVEL_COLORS[i % len(_INCORRECT_LEVEL_COLORS)]
    return tuple(colors[k] for k in keys)


def apply_confidence_threshold(path: list[int], confidences: list[float], thresholds) -> int | None:
    """
    Back off from the deepest decision toward the root until one clears its level's threshold.

    Returns the accepted node_id, or None if every level fails (abstain).
    """
    for d in reversed(range(len(confidences))):
        if confidences[d] >= thresholds[d]:
            return path[d + 1]
    return None


def classify_id_outcome(predicted_node: int | None, true_node: int, hierarchy: Hierarchy) -> str:
    """Bucket one prediction against its true node (see module docstring)."""
    if predicted_node is None:
        return "abstained"
    true_path = hierarchy.get_path_to_root(true_node)
    if predicted_node in hierarchy.subtree_leaves(true_node):
        return "correct_all"
    if predicted_node in true_path:
        return "correct_incomplete"
    # A '<Parent>_Unknown' leaf predicted for a true label under that parent counts the same as
    # predicting the parent itself. One-way: a true '_Unknown' label gets no such allowance.
    if hierarchy.nodes[predicted_node].name.endswith("_Unknown") and hierarchy.parent(predicted_node) in true_path:
        return "correct_incomplete"
    pred_path = hierarchy.get_path_to_root(predicted_node)
    for level in range(1, len(true_path)):
        pred_at_level = pred_path[level] if level < len(pred_path) else None
        if pred_at_level != true_path[level]:
            return "incorrect_all" if level == 1 else f"incorrect_l{level}"
    return "incorrect_all"


def graded_node(path: list[int], confidences: list[float], thresholds=None) -> int | None:
    """The node a prediction is graded on: the raw greedy leaf, or the gated node if thresholds are given."""
    return apply_confidence_threshold(path, confidences, thresholds) if thresholds is not None else path[-1]


def id_outcome_proportions(paths, confidences, true_nodes, groups, hierarchy, keys, thresholds=None) -> dict:
    """``{group: {outcome key: proportion}}`` -- group is usually the true class name, or a run name."""
    by_group: dict = {}
    for path, conf, true_node, group in zip(paths, confidences, true_nodes, groups):
        outcome = classify_id_outcome(graded_node(path, conf, thresholds), true_node, hierarchy)
        by_group.setdefault(group, []).append(outcome)
    return {group: {k: outs.count(k) / len(outs) for k in keys} for group, outs in by_group.items()}


def id_outcome_table(paths, confidences, true_nodes, hierarchy, keys, thresholds=None) -> tuple[pd.DataFrame, dict]:
    """
    Per true class: N plus the proportion in each outcome bucket, and an N-weighted Total row.

    Returns:
        (table, proportions) -- proportions is the ``{class: {key: share}}`` dict the stacked bar draws.
    """
    names = [hierarchy.nodes[n].name for n in true_nodes]
    props = id_outcome_proportions(paths, confidences, true_nodes, names, hierarchy, keys, thresholds)
    table = pd.DataFrame(props).T[list(keys)].round(3)
    n_by_name = pd.Series(names).value_counts()
    table.insert(0, "N", n_by_name.reindex(table.index).values)
    weights = table["N"] / table["N"].sum()
    totals = table[list(keys)].multiply(weights, axis=0).sum()
    table.loc["Total"] = [table["N"].sum(), *totals]
    return table, props


# ---------------------------------------------------------------------------
# Leaf-class metrics
# ---------------------------------------------------------------------------
def leaf_metrics_table(true_names, predicted_names, correct, hierarchy: Hierarchy) -> pd.DataFrame:
    """
    Strict per-class N / Accuracy / Precision / Recall / F1 (worst accuracy first) plus a Total row
    (micro accuracy, macro precision / recall / F1).

    ``correct`` marks a prediction inside the true node's subtree. A true node that is not a leaf
    (a coarse manual ID) gets N/A precision / recall / F1, left out of the macro averages.
    """
    df = pd.DataFrame({"true": list(true_names), "pred": list(predicted_names), "correct": list(correct)})
    name_to_id = hierarchy.name_to_id()
    labels = sorted(df["true"].unique())
    coarse = {l for l in labels if not hierarchy.is_leaf(name_to_id[l])}
    scoring = [l for l in labels if l not in coarse]

    prec, rec, f1, _ = precision_recall_fscore_support(df["true"], df["pred"], labels=labels,
                                                       average=None, zero_division=0)
    prec, rec, f1 = (pd.Series(v, index=labels) for v in (prec, rec, f1))
    for label in coarse:
        prec[label] = rec[label] = f1[label] = float("nan")

    acc = df.groupby("true")["correct"].mean().reindex(labels)
    n = df.groupby("true").size().reindex(labels)
    table = pd.DataFrame({"N": n.values, "Accuracy": acc.values, "Precision": prec.values,
                          "Recall": rec.values, "F1": f1.values}, index=labels).sort_values("Accuracy")

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        df["true"], df["pred"], labels=scoring, average="macro", zero_division=0)
    table.loc["Total"] = [len(df), df["correct"].mean(), prec_macro, rec_macro, f1_macro]
    return table


def _pooled_class_metrics(true_values, predicted_values, hit_mask, classes):
    """Per-class N / recall (grouped by true) / precision (grouped by predicted) / F1 from a hit mask."""
    df = pd.DataFrame({"true": list(true_values), "pred": list(predicted_values), "hit": list(hit_mask)})
    n = df.groupby("true").size().reindex(classes)
    recall = df.groupby("true")["hit"].mean().reindex(classes)
    pred_n = df.groupby("pred").size()
    pred_hits = df[df["hit"]].groupby("pred").size()
    precision = (pred_hits / pred_n).reindex(classes)
    f1 = 2 * precision * recall / (precision + recall)
    totals = (len(df), df["hit"].mean(), precision.mean(), recall.mean(), f1.mean())
    return n, recall, precision, f1, totals


def loosened_leaf_metrics_table(true_names, predicted_names, hierarchy: Hierarchy,
                                exclude=frozenset()) -> pd.DataFrame:
    """
    Per-class metrics where a prediction graded ``correct_all`` or ``correct_incomplete`` counts as a hit.

    A species predicted as its branch's ``_Unknown`` leaf is a recall hit for the true species' row
    and a precision hit for the ``_Unknown`` row. ``exclude`` drops true classes from the
    population (e.g. ``{'Zoop-No'}``).
    """
    name_to_id = hierarchy.name_to_id()
    df = pd.DataFrame({"true": list(true_names), "pred": list(predicted_names)})
    if exclude:
        df = df[~df["true"].isin(list(exclude))]
    hit = [classify_id_outcome(name_to_id[p], name_to_id[t], hierarchy) in ("correct_all", "correct_incomplete")
           for p, t in zip(df["pred"], df["true"])]
    labels = sorted(df["true"].unique())
    n, recall, precision, f1, totals = _pooled_class_metrics(df["true"], df["pred"], hit, labels)
    table = pd.DataFrame({"N": n.values, "Accuracy": recall.values, "Precision": precision.values,
                          "Recall": recall.values, "F1": f1.values}, index=labels).sort_values("Accuracy")
    table.loc["Total"] = list(totals)
    return table


def _is_unknown_sibling_hit(pred_name, pred_id, true_id, hierarchy: Hierarchy) -> bool:
    return pred_name.endswith("_Unknown") and hierarchy.parent(pred_id) == hierarchy.parent(true_id)


def hierarchy_level_metrics_table(true_names, predicted_names, hierarchy: Hierarchy) -> pd.DataFrame:
    """
    N / Accuracy / Precision / Recall / F1 per class at every hierarchy level, each level graded as
    its own flat classification, with a Total row per level.

    A sample is in level d's population only if its true node reaches depth d. Predicting a
    ``_Unknown`` sibling of the true class at that depth counts as a hit (pooled as in
    :func:`loosened_leaf_metrics_table`). A prediction that stops above depth d is ``(none)``.
    """
    name_to_id = hierarchy.name_to_id()
    true_ids = [name_to_id[t] for t in true_names]
    pred_ids = [name_to_id[p] for p in predicted_names]

    rows = []
    for d in range(1, hierarchy.max_depth + 1):
        idx = [i for i, tid in enumerate(true_ids) if hierarchy.nodes[tid].depth >= d]
        if not idx:
            continue
        true_at_d_id = [hierarchy.get_path_to_root(true_ids[i])[d] for i in idx]
        pred_at_d_id = [hierarchy.get_path_to_root(pred_ids[i])[d] if hierarchy.nodes[pred_ids[i]].depth >= d else None
                        for i in idx]
        true_at_d = [hierarchy.nodes[n].name for n in true_at_d_id]
        pred_at_d = [hierarchy.nodes[n].name if n is not None else "(none)" for n in pred_at_d_id]
        hit = [(pn == tn) or (pid is not None and _is_unknown_sibling_hit(pn, pid, tid, hierarchy))
               for pn, tn, pid, tid in zip(pred_at_d, true_at_d, pred_at_d_id, true_at_d_id)]

        classes = sorted(set(true_at_d))
        n, recall, precision, f1, totals = _pooled_class_metrics(true_at_d, pred_at_d, hit, classes)
        level_df = pd.DataFrame({"N": n.values, "Accuracy": recall.values, "Precision": precision.values,
                                 "Recall": recall.values, "F1": f1.values}, index=classes).sort_values("Accuracy")
        for cls, row in level_df.iterrows():
            rows.append({"Level": d, "Class": cls, **row.to_dict()})
        n_total, acc_total, prec_total, rec_total, f1_total = totals
        rows.append({"Level": d, "Class": "Total", "N": n_total, "Accuracy": acc_total,
                     "Precision": prec_total, "Recall": rec_total, "F1": f1_total})

    table = pd.DataFrame(rows).set_index("Class")
    return table[["Level", "N", "Accuracy", "Precision", "Recall", "F1"]]


def top_predictions_table(predictions: pd.DataFrame, classes, top_k: int = 3) -> pd.DataFrame:
    """For each labelled folder in ``classes``: N and its top-k predicted leaves with their share."""
    rows = []
    for cls in classes:
        sub = predictions[predictions["actual_class"] == cls]
        n = len(sub)
        row = {"N": n}
        counts = sub["predicted_class_1"].value_counts() if n else pd.Series(dtype=int)
        for k in range(top_k):
            if k < len(counts):
                row[f"#{k + 1} predicted"] = f"{counts.index[k]} ({counts.iloc[k] / n:.1%})"
            else:
                row[f"#{k + 1} predicted"] = "N/A" if n == 0 else "—"
        rows.append(row)
    return pd.DataFrame(rows, index=list(classes))


def capped_indices_by_group(group_values, max_per_class: int, seed: int) -> list:
    """Indices downsampled to at most ``max_per_class`` per distinct value (fixed-seed sample)."""
    series = pd.Series(group_values)
    indices = []
    for _, group in series.groupby(series):
        idxs = group.index.tolist()
        if len(idxs) > max_per_class:
            idxs = pd.Series(idxs).sample(n=max_per_class, random_state=seed).tolist()
        indices.extend(idxs)
    return indices


def training_result_accuracy_table(run_dir, hierarchy: Hierarchy) -> pd.DataFrame | None:
    """Loosened leaf metrics on a training run's own held-out test split (``predictions.npz``), or None."""
    from pathlib import Path
    npz_path = Path(run_dir) / "predictions.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    id_to_name = {nid: node.name for nid, node in hierarchy.nodes.items()}
    return loosened_leaf_metrics_table([id_to_name[i] for i in data["targets"].tolist()],
                                       [id_to_name[i] for i in data["predictions"].tolist()], hierarchy)
