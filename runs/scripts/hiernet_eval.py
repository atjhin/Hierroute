"""
Shared evaluation + confidence-threshold-cascade logic for HierRouteNet, used by
runs/hiernet_mnr.ipynb, runs/confidence_threshold_experiment.ipynb and
runs/experiment_scripts/confidence_threshold_table.py -- so all three stay
consistent and the messy-background / held-out-test-split analyses run through
one code path.

A HierRouteNet routes root-to-leaf by greedily taking the highest-confidence child
at each internal node. These helpers let you additionally gate that routing on a
per-level confidence threshold (backing off to a coarser ancestor when the model
isn't confident enough), score a checkpoint on any loader, build the per-level
confidence ROC curves used to pick thresholds, and grid-search whole threshold
vectors.

    import sys; sys.path.append('scripts')            # from runs/
    from hiernet_eval import (
        # cascade primitives
        predict_with_confidences, apply_confidence_threshold, classify_outcome, classify_id_outcome,
        # inference + per-sample cache
        collect_predictions,
        # per-level confidence distributions
        split_level_confidences, plot_level_confidence_hist,
        # per-level ROC
        level_roc, plot_level_roc, roc_summary_frame,
        # whole-set outcome tally for one threshold vector
        evaluate_thresholds, settle_depth_distribution,
        # threshold grid search
        threshold_grid_search, pareto_frontier, plot_threshold_grid, print_grid_search_report,
        # checkpoint eval + per-class breakdown
        evaluate_checkpoint, per_class_outcomes, plot_outcome_stacked_bar,
        per_class_id_outcomes, plot_id_outcome_stacked_bar,
    )
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from hierroute import HierRouteNet, Trainer


# ---------------------------------------------------------------------------
# Confidence-threshold cascade primitives
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_with_confidences(model, x):
    """
    Same greedy root-to-leaf walk as HierRouteNet.predict(), but also returns each step's
    local (conditional) softmax confidence -- the same `weights` HierRouteNet.forward() uses
    for its path-product, not the cumulative path probability.

    Returns:
        paths       : list[list[int]]   -- paths[i] = [root, ..., deepest node reached]
        confidences : list[list[float]] -- confidences[i][d] is the confidence behind the
                      decision that produced paths[i][d + 1]. A sample whose greedy path
                      terminates early (e.g. at a leaf that sits above the deepest hierarchy
                      level) simply has a shorter confidences list.
    """
    feat = model.shared(x)
    if model.backbone_type.startswith("swin"):
        feat = feat.permute(0, 3, 1, 2)
    if model.expert_type != "cnn":
        feat = torch.flatten(model.pool(feat), 1)

    B = feat.shape[0]
    current_nodes = [model.hierarchy.root] * B
    paths = [[model.hierarchy.root] for _ in range(B)]
    confidences = [[] for _ in range(B)]

    while not all(model.hierarchy.is_leaf(n) for n in current_nodes):
        node_to_indices = {}
        for i, node_id in enumerate(current_nodes):
            if not model.hierarchy.is_leaf(node_id):
                node_to_indices.setdefault(node_id, []).append(i)

        for node_id, indices in node_to_indices.items():
            group_feat = feat[indices]
            weights, _ = model.local_classifiers[str(node_id)](group_feat)  # soft mode -> (weights, logits)
            conf, child_idx = weights.max(dim=-1)
            children = model.hierarchy.nodes[node_id].children
            for sample_idx, cidx, c in zip(indices, child_idx.tolist(), conf.tolist()):
                next_node = children[cidx]
                current_nodes[sample_idx] = next_node
                paths[sample_idx].append(next_node)
                confidences[sample_idx].append(c)

    return paths, confidences


def apply_confidence_threshold(path, confidences, thresholds):
    """
    path/confidences: one sample's output from predict_with_confidences.
    thresholds: list of per-depth thresholds; thresholds[d] gates confidences[d]
                (the decision that produced path[d + 1]).

    Starts from the deepest decision made and backs off toward root; returns the accepted
    node_id, or None if every available depth -- down to and including level 1 -- fails its
    threshold (unclassified/abstain).
    """
    for d in reversed(range(len(confidences))):
        if confidences[d] >= thresholds[d]:
            return path[d + 1]
    return None


def classify_outcome(predicted_node, path, max_depth, true_leaf_id, hierarchy):
    """
    Classifies a cascade-accepted prediction against the true leaf:

        correct_lowest    -- predicted the true leaf itself (only possible with no backoff --
                              a leaf can't sit partway along another leaf's path).
        loosely_correct   -- backed off to a genuine ancestor of the true leaf: right branch,
                              just coarser.
        loosely_incorrect -- backed off, but to a node that isn't an ancestor either: hedged,
                              and still wrong.
        incorrect         -- committed to the deepest decision (no backoff) and it was the
                              wrong leaf entirely.
        abstained         -- predicted_node is None (every level failed its threshold).
    """
    if predicted_node is None:
        return "abstained"
    if predicted_node == true_leaf_id:
        return "correct_lowest"
    if predicted_node in hierarchy.get_path_to_root(true_leaf_id):
        return "loosely_correct"
    predicted_depth = path.index(predicted_node)
    if predicted_depth < max_depth:
        return "loosely_incorrect"
    return "incorrect"


OUTCOME_KEYS = ("correct_lowest", "loosely_correct", "loosely_incorrect", "incorrect", "abstained")
OUTCOME_LABELS = ("correct", "loosely correct", "loosely incorrect", "incorrect", "abstained")
OUTCOME_COLORS = ("#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2")
_LEVEL_PALETTE = ("#4C72B0", "#55A868", "#C44E52", "#8172B2", "#DD8452")


def classify_id_outcome(predicted_node, true_leaf_id, hierarchy):
    """
    Bucket a cascade-accepted prediction against the true (manual) label by *where it
    first disagrees* -- the manual-ID-vs-AI-ID rubric HierNet reports with:

        correct_all        -- predicted a leaf in the true label's own subtree (AI ID ==
                              manual ID, for an exact-leaf manual ID; any of its descendant
                              species, for a manual ID only identified to a coarser genus/
                              family node -- see true_leaf_id below).
        correct_incomplete -- backed off to a genuine ancestor of the true label (right
                              branch, just coarser than the manual ID), OR predicted a
                              '<Parent>_Unknown' sibling leaf (e.g. Copepoda_Unknown) whose
                              parent is a genuine ancestor of the true label -- same idea,
                              since an '_Unknown' leaf means "this branch, unresolved to
                              species" even though it isn't literally an ancestor node. This
                              credit is one-way: a true label that is itself an '_Unknown'
                              leaf does not get it back for a specific-species prediction.
        incorrect_l{d}     -- agrees with the manual ID down to level d-1, wrong at level
                              d (d >= 2); the deepest such bucket is the leaf level.
        incorrect_all      -- wrong already at level 1 (e.g. Zoop-yes vs Zoop-No): no
                              level of the prediction agrees with the manual ID.
        abstained          -- predicted_node is None (every level failed its threshold).

    true_leaf_id is usually an actual leaf, but may also be an internal node when the manual
    ID only committed to a coarser rank (e.g. a 'Rotifer' crop with no species call): subtree
    matching (via hierarchy.subtree_leaves, which returns [node_id] itself for a real leaf)
    means this works unchanged for both, and incorrect_l{d} buckets deeper than true_leaf_id's
    own depth simply never occur since there's no finer ground truth to disagree with.

    Unlike classify_outcome (which splits the wrong predictions by whether the cascade
    committed or hedged), this splits them by the level the error entered at. Assumes a
    <= 3-level taxonomy, matching the rest of this module's t1/t2/t3 vocabulary.
    """
    if predicted_node is None:
        return "abstained"
    true_path = hierarchy.get_path_to_root(true_leaf_id)   # [root, level-1, ..., true_leaf]
    if predicted_node in hierarchy.subtree_leaves(true_leaf_id):
        return "correct_all"
    if predicted_node in true_path:
        return "correct_incomplete"
    # An '_Unknown' leaf (e.g. Copepoda_Unknown -- a sibling of Cyclopoid/Calanoid/... under
    # Copepoda, not their ancestor; see hierroute.constants.hiernet_adjacency_graph) predicted
    # for a true label that descends from that same parent is graded the same as predicting
    # the parent node directly would be: right branch, just not resolved to species. One-way
    # only -- the true label itself being an '_Unknown' bucket does NOT get this same
    # allowance when a specific species is predicted for it; that stays under the ordinary
    # rules below.
    if hierarchy.nodes[predicted_node].name.endswith('_Unknown') and hierarchy.parent(predicted_node) in true_path:
        return "correct_incomplete"
    pred_path = hierarchy.get_path_to_root(predicted_node)  # [root, level-1, ...]
    for level in range(1, len(true_path)):
        pred_at_level = pred_path[level] if level < len(pred_path) else None
        if pred_at_level != true_path[level]:
            return "incorrect_all" if level == 1 else f"incorrect_l{level}"
    return "incorrect_all"  # unreachable: agreeing at every level means predicted_node == true_leaf_id


ID_OUTCOME_KEYS = ("correct_all", "correct_incomplete", "incorrect_l3", "incorrect_l2", "incorrect_all", "abstained")
ID_OUTCOME_LABELS = ("Correct_All", "Correct_Incomplete", "Incorrect_L3", "Incorrect_L2", "Incorrect_All", "Abstained")
ID_OUTCOME_COLORS = ("#4C72B0", "#55A868", "#E7B84B", "#DD8452", "#C44E52", "#8172B2")

# Same rubric minus the last bucket -- for evaluations that skip apply_confidence_threshold
# entirely (predicted_node is always the raw greedy leaf, so 'abstained' can never occur;
# see per_class_id_outcomes'/per_group_id_outcomes' gate_by_confidence param).
ID_OUTCOME_KEYS_NO_ABSTAIN = ID_OUTCOME_KEYS[:-1]
ID_OUTCOME_LABELS_NO_ABSTAIN = ID_OUTCOME_LABELS[:-1]
ID_OUTCOME_COLORS_NO_ABSTAIN = ID_OUTCOME_COLORS[:-1]


# ---------------------------------------------------------------------------
# Inference: one greedy-routing pass over a loader
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Run greedy routing once over `loader`, caching per sample:

        all_paths        : list[list[int]]     greedy root-to-leaf node paths
        all_confidences  : list[list[float]]   per-decision local softmax confidences
        all_true_leaves  : list[int]           true leaf node_id (from batch['label_node'])

    Everything downstream (thresholds, ROC, per-class outcomes) reuses this cache --
    the model never re-runs.
    """
    model.eval()
    all_paths, all_confidences, all_true_leaves = [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        true_ids = batch["label_node"].argmax(dim=1).tolist()
        paths, confidences = predict_with_confidences(model, image)
        all_paths.extend(paths)
        all_confidences.extend(confidences)
        all_true_leaves.extend(true_ids)
    return all_paths, all_confidences, all_true_leaves


def _n_levels(all_confidences):
    return max((len(c) for c in all_confidences), default=0)


# ---------------------------------------------------------------------------
# Per-level confidence distributions (leaf-graded) + histogram
# ---------------------------------------------------------------------------
def split_level_confidences(all_paths, all_confidences, all_true_leaves):
    """
    Split each greedy decision's local confidence by whether the sample's *final leaf*
    prediction is correct (`path[-1] == true_leaf`) -- a coarse, leaf-level grading.
    (Contrast level_roc, which grades each decision at its own level.)

    Returns (level_confidences, wrong_level_confidences, correct_level_confidences, n_wrong)
    where the three are dicts keyed depth (1-indexed) -> list[float].
    """
    n_levels = _n_levels(all_confidences)
    level = {d: [] for d in range(1, n_levels + 1)}
    wrong = {d: [] for d in range(1, n_levels + 1)}
    correct = {d: [] for d in range(1, n_levels + 1)}
    n_wrong = 0
    for path, confs, true_leaf in zip(all_paths, all_confidences, all_true_leaves):
        is_wrong = path[-1] != true_leaf
        n_wrong += is_wrong
        for depth, conf in enumerate(confs, start=1):
            level[depth].append(conf)
            (wrong if is_wrong else correct)[depth].append(conf)
    return level, wrong, correct, n_wrong


def plot_level_confidence_hist(data_by_level, title, bins=30):
    """1xN histogram of per-level confidence (data_by_level keys = levels), mean marked."""
    import matplotlib.pyplot as plt

    levels = sorted(data_by_level)
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, depth in zip(axes, levels):
        color = _LEVEL_PALETTE[(depth - 1) % len(_LEVEL_PALETTE)]
        vals = data_by_level[depth]
        if vals:
            x_min = min(vals)
            ax.hist(vals, bins=bins, range=(x_min, 1), color=color, alpha=0.85)
            ax.axvline(np.mean(vals), color="black", linestyle="--", linewidth=1,
                       label=f"mean={np.mean(vals):.2f}")
            ax.legend(fontsize=8)
            ax.set_xlim(x_min, 1)
        else:
            ax.text(0.5, 0.5, f"no samples reach\na level-{depth} decision",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title(f"Level {depth} (n={len(vals)})")
        ax.set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-level confidence ROC (graded at each level, not at the leaf)
# ---------------------------------------------------------------------------
def level_roc(all_paths, all_confidences, all_true_leaves, hierarchy):
    """
    For every hierarchy level, treat the greedy decision's local confidence as a score for
    "did the model route into the right branch AT THIS LEVEL" -- positive iff path[d] lies on
    the true leaf's own root-to-leaf path. Sweeping the accept-if-conf >= t threshold traces
    an ROC curve where TPR = correct level-d decisions kept, FPR = wrong ones kept.

    Returns {depth: None | {fpr, tpr, thr, auc, youden_thr, youden_tpr, youden_fpr,
                            n_correct, n_wrong}}, keyed 1..deepest-decision.
    """
    from sklearn.metrics import roc_curve, auc

    n_levels = _n_levels(all_confidences)
    correct = {d: [] for d in range(1, n_levels + 1)}
    wrong = {d: [] for d in range(1, n_levels + 1)}
    for path, confs, true_leaf in zip(all_paths, all_confidences, all_true_leaves):
        true_path = hierarchy.get_path_to_root(true_leaf)
        for depth, conf in enumerate(confs, start=1):
            (correct if path[depth] in true_path else wrong)[depth].append(conf)

    roc = {}
    for depth in range(1, n_levels + 1):
        pos = np.asarray(correct[depth], dtype=float)
        neg = np.asarray(wrong[depth], dtype=float)
        if len(pos) == 0 or len(neg) == 0:
            roc[depth] = None
            continue
        y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
        y_score = np.concatenate([pos, neg])
        fpr, tpr, thr = roc_curve(y_true, y_score)
        j = tpr - fpr
        best = int(np.argmax(j))
        roc[depth] = {
            "fpr": fpr,
            "tpr": tpr,
            "thr": thr,
            "auc": float(auc(fpr, tpr)),
            "youden_thr": float(np.clip(thr[best], 0.0, 1.0)),
            "youden_tpr": float(tpr[best]),
            "youden_fpr": float(fpr[best]),
            "n_correct": int(len(pos)),
            "n_wrong": int(len(neg)),
        }
    return roc


def roc_summary_frame(roc_by_level):
    """Youden-J-optimal per-level thresholds as a tidy DataFrame (candidate cascade vector)."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "level": depth,
                "auc": round(r["auc"], 4),
                "youden_threshold": round(r["youden_thr"], 4),
                "tpr_at_threshold": round(r["youden_tpr"], 4),
                "fpr_at_threshold": round(r["youden_fpr"], 4),
                "n_correct": r["n_correct"],
                "n_wrong": r["n_wrong"],
            }
            for depth, r in sorted(roc_by_level.items())
            if r is not None
        ]
    )


def plot_level_roc(roc_by_level, title=None):
    """1xN panel of per-level confidence ROC curves with the Youden-J operating point marked."""
    import matplotlib.pyplot as plt

    levels = sorted(roc_by_level)
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4.5), squeeze=False)
    axes = axes[0]
    for ax, depth in zip(axes, levels):
        color = _LEVEL_PALETTE[(depth - 1) % len(_LEVEL_PALETTE)]
        r = roc_by_level[depth]
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
        if r is None:
            ax.text(
                0.5, 0.5, f"no correct/wrong split\nat a level-{depth} decision",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray",
            )
            ax.set_title(f"Level {depth}")
        else:
            ax.plot(r["fpr"], r["tpr"], color=color, linewidth=2, label=f"AUC = {r['auc']:.3f}")
            ax.scatter(
                [r["youden_fpr"]], [r["youden_tpr"]], color="black", zorder=5,
                label=(f"Youden thr = {r['youden_thr']:.3f}\n"
                       f"TPR={r['youden_tpr']:.2f}, FPR={r['youden_fpr']:.2f}"),
            )
            ax.annotate(
                f"{r['youden_thr']:.3f}", (r["youden_fpr"], r["youden_tpr"]),
                textcoords="offset points", xytext=(8, -12), fontsize=8,
            )
            ax.set_title(f"Level {depth}  (n={r['n_correct']} correct / {r['n_wrong']} wrong)")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("FPR  (wrong decisions kept)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("TPR  (correct decisions kept)")
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Whole-set outcome tally / settle-depth distribution for one threshold vector
# ---------------------------------------------------------------------------
def evaluate_thresholds(thresholds, all_paths, all_confidences, all_true_leaves, hierarchy):
    """
    Apply the cascade with `thresholds` to every cached sample and tally outcomes.

    Returns {'thresholds': tuple, 'n': int,
             <key>: "<count> (<pct>)" for each OUTCOME_KEYS entry,
             '_counts': {key: raw int}}.
    """
    outcomes = [
        classify_outcome(apply_confidence_threshold(p, c, thresholds), p, len(c), t, hierarchy)
        for p, c, t in zip(all_paths, all_confidences, all_true_leaves)
    ]
    n = len(outcomes)
    counts = {k: outcomes.count(k) for k in OUTCOME_KEYS}
    return {
        "thresholds": tuple(thresholds),
        "n": n,
        **{k: f"{v} ({v / n:.1%})" for k, v in counts.items()},
        "_counts": counts,
    }


def settle_depth_distribution(thresholds, all_paths, all_confidences, hierarchy):
    """
    How many samples the cascade with `thresholds` ultimately settles at each hierarchy
    depth (vs. 'abstained' when it backs off past level 1). Returns a sorted pandas Series.
    """
    import pandas as pd

    depths = []
    for p, c in zip(all_paths, all_confidences):
        node = apply_confidence_threshold(p, c, thresholds)
        depths.append("abstained" if node is None else f"level {hierarchy.nodes[node].depth}")
    return pd.Series(depths).value_counts().sort_index()


# ---------------------------------------------------------------------------
# Threshold grid search  (vectorized; every rate over all n samples)
# ---------------------------------------------------------------------------
def build_outcome_table(all_paths, all_confidences, all_true_leaves, hierarchy, n_levels=None):
    """
    Threshold-independent per-sample arrays that make the grid search a pure-NumPy sweep:

      conf_pad[i, d]    : confidence of sample i's depth-(d+1) decision, -inf past its own
                          decision count so a padded level never clears a threshold.
      outcome_cat[i, d] : outcome if the cascade settles at level d+1 --
                          0 correct_lowest | 1 loosely_correct | 2 loosely_incorrect | 3 incorrect
                          (exact mirror of classify_outcome).
    """
    if n_levels is None:
        n_levels = _n_levels(all_confidences)
    n = len(all_paths)
    conf_pad = np.full((n, n_levels), -np.inf)
    outcome_cat = np.zeros((n, n_levels), dtype=np.int8)
    for i, (path, confs, true_leaf) in enumerate(zip(all_paths, all_confidences, all_true_leaves)):
        true_path = set(hierarchy.get_path_to_root(true_leaf))
        L = len(confs)
        for s in range(1, L + 1):
            conf_pad[i, s - 1] = confs[s - 1]
            node = path[s]
            if node == true_leaf:
                outcome_cat[i, s - 1] = 0
            elif node in true_path:
                outcome_cat[i, s - 1] = 1
            elif s < L:
                outcome_cat[i, s - 1] = 2
            else:
                outcome_cat[i, s - 1] = 3
    return conf_pad, outcome_cat


def _settle_levels(conf_pad, thresholds):
    """Per-sample deepest level index (0-based) whose confidence clears its threshold; -1 = abstain."""
    clears = conf_pad >= np.asarray(thresholds, dtype=float)
    last = conf_pad.shape[1] - 1 - clears[:, ::-1].argmax(axis=1)
    return np.where(clears.any(axis=1), last, -1)


def score_thresholds(conf_pad, outcome_cat, thresholds):
    """Vectorized equivalent of evaluate_thresholds(...)['_counts'] for one threshold vector."""
    settle = _settle_levels(conf_pad, thresholds)
    covered = settle >= 0
    sel = outcome_cat[np.arange(len(outcome_cat)), np.where(covered, settle, 0)]
    return {
        "correct_lowest": int(((sel == 0) & covered).sum()),
        "loosely_correct": int(((sel == 1) & covered).sum()),
        "loosely_incorrect": int(((sel == 2) & covered).sum()),
        "incorrect": int(((sel == 3) & covered).sum()),
        "abstained": int((~covered).sum()),
    }


def threshold_grid_search(all_paths, all_confidences, all_true_leaves, hierarchy,
                          t1_grid, t2_grid, t3_grid, verify=True):
    """
    Full Cartesian sweep over (t1_grid x t2_grid x t3_grid). One row per combo; every rate
    is over ALL n samples (abstentions included in the denominator), so the four sum to 1:

      acc         = (correct_lowest + loosely_correct) / n   -- landed on the right branch
      strict_acc  = correct_lowest / n                        -- landed on the exact leaf
      err         = (loosely_incorrect + incorrect) / n       -- committed and was wrong
      abstain_pct = abstained / n                             -- made no prediction

    `covered` (= n - abstained) is kept for reference. `verify` cross-checks the vectorized
    scorer against evaluate_thresholds on one probe.
    """
    import pandas as pd

    conf_pad, outcome_cat = build_outcome_table(all_paths, all_confidences, all_true_leaves,
                                                hierarchy, n_levels=3)
    n = len(all_paths)

    if verify:
        probe = [t1_grid[len(t1_grid) // 2], t2_grid[len(t2_grid) // 2], t3_grid[len(t3_grid) // 2]]
        got = score_thresholds(conf_pad, outcome_cat, probe)
        ref = evaluate_thresholds(probe, all_paths, all_confidences, all_true_leaves, hierarchy)["_counts"]
        assert got == ref, f"vectorized score_thresholds disagrees with evaluate_thresholds: {got} != {ref}"

    rows = []
    for t1, t2, t3 in itertools.product(t1_grid, t2_grid, t3_grid):
        c = score_thresholds(conf_pad, outcome_cat, (t1, t2, t3))
        rows.append({
            "t1": t1, "t2": t2, "t3": t3,
            "acc": (c["correct_lowest"] + c["loosely_correct"]) / n,
            "strict_acc": c["correct_lowest"] / n,
            "err": (c["loosely_incorrect"] + c["incorrect"]) / n,
            "abstain_pct": c["abstained"] / n,
            "covered": n - c["abstained"],
        })
    return pd.DataFrame(rows)


def pareto_frontier(grid_df, x="abstain_pct", y="err"):
    """
    Non-dominated staircase minimizing BOTH x and y -- by default the risk-coverage
    frontier: fewest abstentions AND fewest flat-wrong predictions (both / n). As the
    thresholds tighten you abstain more and `err` falls; this traces the lowest `err`
    reachable at each abstention level. (`acc` alone is maximized by barely abstaining,
    so it is not a useful frontier axis here.)
    """
    s = grid_df.sort_values([x, y], ascending=[True, True])
    return s[s[y] < s[y].cummin().shift(1, fill_value=np.inf)].copy()


_GRID_FMT = {
    "acc": "{:.4f}".format,
    "strict_acc": "{:.4f}".format,
    "err": "{:.4f}".format,
    "abstain_pct": "{:.2%}".format,
}


def print_grid_search_report(grid_df, all_paths, all_confidences, all_true_leaves, hierarchy, n_best=10):
    """
    Print the two bracketing operating points -- highest overall accuracy and lowest
    overall error -- the lowest-error combo's settle-depth breakdown, the top-`n_best`
    lowest-error combos, and the (abstain % -> err) Pareto frontier. Returns the
    lowest-error row.
    """
    n = len(all_paths)
    hi_acc = grid_df.loc[grid_df["acc"].idxmax()]
    lo_err = grid_df.loc[grid_df["err"].idxmin()]
    settle = settle_depth_distribution([lo_err.t1, lo_err.t2, lo_err.t3], all_paths, all_confidences, hierarchy)

    print("=" * 70)
    print("HIGHEST OVERALL ACCURACY   (acc = (correct_lowest + loosely_correct) / n)")
    print("=" * 70)
    print(f"  thresholds [t1, t2, t3]  = [{hi_acc.t1}, {hi_acc.t2}, {hi_acc.t3}]")
    print(f"  acc = {hi_acc.acc:.4f}   strict_acc = {hi_acc.strict_acc:.4f}   "
          f"err = {hi_acc.err:.4f}   abstain % = {hi_acc.abstain_pct:.2%}")

    print("\n" + "=" * 70)
    print("LOWEST OVERALL ERROR   (err = (loosely_incorrect + incorrect) / n)")
    print("=" * 70)
    print(f"  thresholds [t1, t2, t3]  = [{lo_err.t1}, {lo_err.t2}, {lo_err.t3}]")
    print(f"  err        = {lo_err.err:.4f}   ({int(round(lo_err.err * n))} / {n} flat-wrong)")
    print(f"  acc        = {lo_err.acc:.4f}   (strict {lo_err.strict_acc:.4f})")
    print(f"  abstain %  = {lo_err.abstain_pct:.2%}   ({int(round(lo_err.abstain_pct * n))} / {n})")
    print("  settles at:  " + "   ".join(f"{k}={v}" for k, v in settle.items()))

    print(f"\nTop {n_best} combinations by lowest error:")
    print(grid_df.sort_values("err").head(n_best).to_string(index=False, formatters=_GRID_FMT))

    print("\nPareto frontier  (abstain % -> lowest reachable error):")
    cols = ["t1", "t2", "t3", "abstain_pct", "err", "acc", "strict_acc"]
    print(pareto_frontier(grid_df)[cols].to_string(index=False, formatters=_GRID_FMT))
    return lo_err


def plot_threshold_grid(grid_df, title=None):
    """
    1x2 view of a threshold_grid_search DataFrame (all rates / n):
      (A) error % vs abstain %, coloured by t3, with the (abstain, err) Pareto frontier and
          the lowest-error point starred.
      (B) error surface over (t2, t3) sliced at the lowest-error combo's t1, accuracy contours.

    Returns (fig, frontier, best_combo) with best_combo = the lowest-error row.
    """
    import matplotlib.pyplot as plt

    frontier = pareto_frontier(grid_df)
    best = grid_df.loc[grid_df["err"].idxmin()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    sc = ax.scatter(grid_df["abstain_pct"] * 100, grid_df["err"] * 100,
                    c=grid_df["t3"], cmap="viridis", s=16, alpha=0.6, linewidths=0)
    ax.plot(frontier["abstain_pct"] * 100, frontier["err"] * 100,
            color="black", lw=1.5, marker="o", ms=4, label="Pareto frontier")
    ax.scatter([best.abstain_pct * 100], [best.err * 100], marker="*", s=280,
               color="crimson", edgecolor="black", zorder=6, label=f"min err = {best.err:.3f}")
    fig.colorbar(sc, ax=ax, label="level-3 threshold  t3")
    ax.set_xlabel("abstain %   (abstained / n)")
    ax.set_ylabel("error %   ((loosely_incorrect + incorrect) / n)")
    ax.set_title(f"Error vs abstention  --  {len(grid_df):,} threshold combinations")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sub = grid_df[grid_df["t1"] == best.t1]
    err_g = sub.pivot(index="t2", columns="t3", values="err")
    acc_g = sub.pivot(index="t2", columns="t3", values="acc")
    pm = ax.pcolormesh(err_g.columns.values, err_g.index.values, err_g.values * 100,
                       cmap="viridis", shading="nearest")
    a_lo, a_hi = acc_g.values.min() * 100, acc_g.values.max() * 100
    levels = [lv for lv in (80, 85, 88, 90, 92, 94, 95, 96, 97, 98, 99) if a_lo < lv < a_hi]
    if levels:
        cs = ax.contour(acc_g.columns.values, acc_g.index.values, acc_g.values * 100,
                        levels=levels, colors="white", linewidths=0.9)
        ax.clabel(cs, fmt="%.0f%% acc", fontsize=7)
    ax.scatter([best.t3], [best.t2], marker="*", s=280, color="crimson", edgecolor="black", zorder=6)
    fig.colorbar(pm, ax=ax, label="error %")
    ax.set_xlabel("level-3 threshold  t3")
    ax.set_ylabel("level-2 threshold  t2")
    ax.set_title(f"Error over (t2, t3)  at  t1 = {best.t1}"
                 + ("\nwhite = accuracy % contours" if levels else "\n(accuracy ~flat across this slice)"))

    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig, frontier, best


# ---------------------------------------------------------------------------
# Checkpoint evaluation on an arbitrary loader
# ---------------------------------------------------------------------------
def evaluate_checkpoint(
    checkpoint_dir,
    hier_dataset,
    loader,
    device,
    tag,
    *,
    backbone="swin_t",
    expert_type="cnn",
    save=True,
):
    """
    Load `checkpoint_dir` fresh against `hier_dataset`'s hierarchy and run Trainer.predict
    over `loader`. Predictions/metrics are written to `{checkpoint_dir}_{tag}_eval` (kept
    separate from the checkpoint's own training-split eval_results.json).

    Returns (model, level_results). `tag` is the only thing that distinguishes a
    Messy_background run ('messy') from a held-out-test-split run ('test').
    """
    model = HierRouteNet(
        hierarchy=hier_dataset.hierarchy,
        label_to_id=hier_dataset.label_to_ids,
        backbone=backbone,
        expert_type=expert_type,
        checkpoint_dir=str(checkpoint_dir),
    )
    out_dir = f"{checkpoint_dir}_{tag}_eval"
    eval_trainer = Trainer(learning_rate=0, max_epochs=0, device=device, model_dir=out_dir)
    result = eval_trainer.predict(model, loader, save=save)
    return model, result["level_results"]


# ---------------------------------------------------------------------------
# Per-class cascade outcome breakdown
# ---------------------------------------------------------------------------
def per_class_outcomes(model, loader, thresholds, hierarchy, id_to_name, device, predictions=None):
    """
    Per true-class proportions of each cascade outcome (OUTCOME_KEYS) under `thresholds`.

    predictions: optional pre-computed (all_paths, all_confidences, all_true_leaves) tuple
    from collect_predictions -- pass it to avoid a second inference pass.
    """
    if predictions is None:
        predictions = collect_predictions(model, loader, device)
    all_paths, all_confidences, all_true_leaves = predictions

    outcomes_by_class = {}
    for path, conf, true_leaf in zip(all_paths, all_confidences, all_true_leaves):
        predicted = apply_confidence_threshold(path, conf, thresholds)
        outcome = classify_outcome(predicted, path, len(conf), true_leaf, hierarchy)
        outcomes_by_class.setdefault(true_leaf, []).append(outcome)

    return {
        id_to_name[leaf_id]: {k: outs.count(k) / len(outs) for k in OUTCOME_KEYS}
        for leaf_id, outs in outcomes_by_class.items()
    }


def plot_outcome_stacked_bar(proportions_by_class, title,
                             keys=OUTCOME_KEYS, labels=OUTCOME_LABELS, colors=OUTCOME_COLORS,
                             legend_side=False):
    """
    Stacked bar of per-class outcome proportions (one bar per true class), stacked bottom
    -> top in `keys` order. Defaults to the classify_outcome rubric; pass the ID_OUTCOME_*
    tuples for the classify_id_outcome rubric.

    legend_side=True puts the legend in a single column to the right of the plot instead of
    below it -- use this when x labels are long enough (e.g. full CSV_Data run names) that the
    default rotated-label-plus-below-legend layout would overlap.
    """
    import matplotlib.pyplot as plt

    classes = sorted(proportions_by_class.keys())
    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.9), 5))
    bottom = np.zeros(len(classes))
    for key, label, color in zip(keys, labels, colors):
        vals = np.array([proportions_by_class[c].get(key, 0.0) for c in classes])
        ax.bar(classes, vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    if legend_side:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=len(labels))
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-class ID-outcome breakdown  (manual-ID-vs-AI-ID rubric, cascade-gated)
# ---------------------------------------------------------------------------
def per_class_id_outcomes(model, loader, thresholds, hierarchy, id_to_name, device, predictions=None,
                           gate_by_confidence=True, outcome_keys=ID_OUTCOME_KEYS):
    """
    Per true-class proportions of each classify_id_outcome bucket (outcome_keys) -- one
    distribution per true (manual) class, the shape plot_outcome_stacked_bar(..., keys=
    outcome_keys, ...) draws.

    gate_by_confidence=True (default) grades the confidence-threshold cascade's accepted node
    (apply_confidence_threshold(path, conf, thresholds), possibly None/'abstained'). Pass False
    to instead grade the raw greedy leaf (path[-1], ignoring confidence entirely) -- 'abstained'
    becomes impossible, so pass outcome_keys=ID_OUTCOME_KEYS_NO_ABSTAIN alongside it.

    predictions: optional pre-computed (all_paths, all_confidences, all_true_leaves) tuple
    from collect_predictions -- pass it to avoid a second inference pass.
    """
    if predictions is None:
        predictions = collect_predictions(model, loader, device)
    all_paths, all_confidences, all_true_leaves = predictions

    outcomes_by_class = {}
    for path, conf, true_leaf in zip(all_paths, all_confidences, all_true_leaves):
        predicted = apply_confidence_threshold(path, conf, thresholds) if gate_by_confidence else path[-1]
        outcome = classify_id_outcome(predicted, true_leaf, hierarchy)
        outcomes_by_class.setdefault(true_leaf, []).append(outcome)

    return {
        id_to_name[leaf_id]: {k: outs.count(k) / len(outs) for k in outcome_keys}
        for leaf_id, outs in outcomes_by_class.items()
    }


def plot_id_outcome_stacked_bar(proportions_by_class, title, legend_side=False,
                                 keys=ID_OUTCOME_KEYS, labels=ID_OUTCOME_LABELS, colors=ID_OUTCOME_COLORS):
    """plot_outcome_stacked_bar with the classify_id_outcome rubric (ID_OUTCOME_* tuples by
    default; pass the *_NO_ABSTAIN variants for a gate_by_confidence=False evaluation)."""
    return plot_outcome_stacked_bar(proportions_by_class, title,
                                    keys=keys, labels=labels, colors=colors, legend_side=legend_side)
