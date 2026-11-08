"""
evaluate.py — Predict on particle images and, for labelled data, grade the predictions.

Inputs:
    crops   : a folder of individual particle .tif crops. Labelled crops sit in one folder per
              class directly under the input folder (the layout ``hierroute crop`` writes).
    flowcam : raw FlowCam exports (mosaics + CSVs); particles are cropped in memory.

Outputs, under the output folder:
    predictions.csv              always
    leaf_accuracy.png            labelled only
    id_outcomes_table.png        labelled only
    id_outcomes_stacked_bar.png  labelled only
    hierarchy_level_metrics.png  labelled only
    plus optional extras (see EvaluateOptions).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import grading, plotting
from .flowcam import (ClassRules, MosaicCropDataset, collate_mosaics, csv_column_to_snake_case,
                      find_flowcam_csvs, load_particles, read_flowcam_csv, run_name_from_crop, strip_dup_suffix)
from .inference import CropDataset, Predictor, eval_transform, find_crops, level_columns
from .pretrained import PRETRAINED, load_model

FOOTNOTE_STRICT = ("Precision / Recall / F1 are macro (equally weighted across leaves); "
                   "Accuracy is micro (weighted by sample count).")
FOOTNOTE_COARSE = (" A coarse manual ID (mapped to an internal node) is correct if predicted anywhere in "
                   "that node's subtree; its Precision/Recall/F1 are N/A and left out of the macro average.")
FOOTNOTE_LOOSENED = ('A specific-species leaf backed off to its branch\'s own "<Branch>_Unknown" '
                     'sibling (e.g. Cyclopoid predicted as Copepoda_Unknown) counts as a hit: a '
                     'Recall/Accuracy hit for the true species\' own row, and a Precision hit for '
                     'the predicted "_Unknown" leaf\'s row -- two different rows\' different '
                     'metrics, not double-counted toward any single number. Ordinary species '
                     'leaves\' Precision is unaffected (still exact-match only).')
FOOTNOTE_LEVEL = ("Each hierarchy level (color-coded; see legend) is its own independent flat "
                  "classification, ignoring every other level. A sample is only in a level's "
                  "population if its true class reaches that level. A \"_Unknown\" sibling prediction "
                  "at that depth (e.g. true Copepoda / predicted Zoop_Unknown at level 2) counts as a "
                  "hit. Rows are sorted worst-Accuracy-first within each level, with Total pinned last; "
                  "Total's Accuracy is sample-weighted (micro), Precision/Recall/F1 are macro.")
FOOTNOTE_BREAKDOWN = ("For each folder, the top-3 predicted_class_1 values by share of that folder's "
                      "own total (N), whether or not the folder maps onto the taxonomy.")


@dataclass
class EvaluateOptions:
    """
    Settings for :func:`evaluate`. Only ``input`` and ``out_dir`` are required.

    Optional outputs (all off by default): ``zoop_only`` (leaf_accuracy_zoop_only.png),
    ``by_sample`` (id_outcomes_stacked_bar_by_sample.png), ``breakdown``
    (top_predictions_breakdown.png), ``training_accuracy`` (training_result_accuracy.png) and
    ``capped`` (leaf_accuracy_capped.png / id_outcomes_table_capped.png).
    """
    input: Path
    out_dir: Path
    model: str | None = None              # released model name (default "noise")
    weights: Path | None = None           # run directory / .pt of a self-trained model
    input_type: str = "crops"             # crops | flowcam
    labelled: bool = False
    class_rules: ClassRules = field(default_factory=ClassRules)
    csv_dir: Path | None = None           # FlowCam CSVs, if not under input
    folder_map: dict[str, str] | None = None   # extra on-disk folder -> node for grading
    thresholds: tuple[float, ...] | None = None
    strict_leaf_accuracy: bool = False
    exclude_runs: tuple[str, ...] = ()
    zoop_only: bool = False
    zoop_only_exclude: tuple[str, ...] = ("Zoop-No",)
    by_sample: bool = False
    breakdown: tuple[str, ...] = ()
    training_accuracy: bool = False
    capped: int | None = None
    seed: int = 666
    device: str = "auto"
    batch_size: int = 64
    num_workers: int = 4
    mosaics_per_batch: int = 4


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _model_display_name(opts: EvaluateOptions) -> str:
    if opts.weights is not None:
        weights = Path(opts.weights)
        return (weights if weights.is_dir() else weights.parent).name
    return Path(PRETRAINED[opts.model or "noise"]["file"]).stem


# ---------------------------------------------------------------------------
# Sample table
# ---------------------------------------------------------------------------
def _crop_samples(opts: EvaluateOptions) -> pd.DataFrame:
    paths, labels = find_crops(opts.input, opts.labelled)
    if not paths:
        hint = (" Found FlowCam CSVs there -- use --input-type flowcam for raw exports."
                if find_flowcam_csvs(opts.input) else "")
        layout = "one folder per class containing .tif crops" if opts.labelled else ".tif crops"
        raise FileNotFoundError(f"No {layout} found under {opts.input}.{hint}")
    samples = pd.DataFrame({"path": paths, "image_file": [Path(p).name for p in paths]})
    samples["run"] = samples["image_file"].map(run_name_from_crop)
    if opts.labelled:
        samples["actual_class"] = labels
    else:
        samples["folder"] = [str(Path(p).parent.relative_to(opts.input)) for p in paths]
    return samples


def _flowcam_samples(opts: EvaluateOptions) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = load_particles(opts.input, opts.csv_dir, opts.class_rules if opts.labelled else None)
    print(table.summary())
    particles = table.particles
    skipped = particles.loc[particles["status"] != "ok", ["image_file", "run", "csv_file", "status"]]
    samples = particles[particles["status"] == "ok"].copy()
    if opts.labelled:
        samples = samples.rename(columns={"label": "actual_class"})
    return samples.reset_index(drop=True), skipped


def _csv_metadata_for_crops(samples: pd.DataFrame, opts: EvaluateOptions) -> pd.DataFrame | None:
    """
    FlowCam measurements for each crop: from ``manifest.csv`` (written by ``hierroute crop``) in the
    input folder if present, else from FlowCam CSVs found under ``--csv-dir`` or the input folder.
    Returns a frame aligned with ``samples`` (csv_* columns), or None if no source is found.
    """
    manifest_path = Path(opts.input) / "manifest.csv"
    if manifest_path.exists() and opts.csv_dir is None:
        manifest = pd.read_csv(manifest_path, low_memory=False)
        manifest = manifest[manifest["crop_path"].notna()]
        keys = list(zip(manifest["crop_path"].map(lambda p: Path(p).parent.name), manifest["image_file"]))
        csv_cols = [c for c in manifest.columns if c.startswith("csv_") and c != "csv_file"]
        index = dict(zip(keys, manifest[csv_cols].to_dict("records")))
        sample_keys = zip(samples["path"].map(lambda p: Path(p).parent.name), samples["image_file"])
        return pd.DataFrame([index.get(k, {}) for k in sample_keys], index=samples.index)

    csv_paths = find_flowcam_csvs(opts.csv_dir if opts.csv_dir is not None else opts.input)
    if not csv_paths:
        return None
    rules = opts.class_rules
    index: dict = {}
    for csv_path in csv_paths:
        df = read_flowcam_csv(csv_path)
        has_label = rules.label_column in df.columns
        labels = rules.labels(df) if has_label else pd.Series([None] * len(df))
        drop = [c for c in ("Particle ID", "Image File", rules.label_column) if c in df.columns]
        records = df.drop(columns=drop).rename(columns=csv_column_to_snake_case).to_dict("records")
        for image_file, pid, label, record in zip(df["Image File"], df["Particle ID"], labels, records):
            index.setdefault((image_file, int(pid)), []).append((None if pd.isna(label) else label, record))

    rows = []
    actual = samples["actual_class"] if "actual_class" in samples else [None] * len(samples)
    for image_file, actual_class in zip(samples["image_file"], actual):
        stem, _, pid = strip_dup_suffix(Path(image_file).stem).rpartition("_")
        candidates = index.get((stem + ".tif", int(pid)), []) if pid.isdigit() else []
        match = next((rec for label, rec in candidates if actual_class is not None and label == actual_class), None)
        if match is None and len(candidates) == 1:
            match = candidates[0][1]
        rows.append(match or {})
    return pd.DataFrame(rows, index=samples.index)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _run_inference(samples: pd.DataFrame, opts: EvaluateOptions, predictor: Predictor, resolution: int):
    """Predictions aligned with samples' index; out_of_bounds / unreadable flags for FlowCam input."""
    transform = eval_transform(resolution)
    predictions = [None] * len(samples)
    out_of_bounds = np.zeros(len(samples), dtype=bool)
    unreadable = np.zeros(len(samples), dtype=bool)

    if opts.input_type == "flowcam":
        loader = DataLoader(MosaicCropDataset(samples, transform), batch_size=opts.mosaics_per_batch,
                            shuffle=False, num_workers=opts.num_workers, collate_fn=collate_mosaics)
        for images, row_indices, oob, missing in loader:
            for rows in missing:
                unreadable[rows] = True
            if images is None:
                continue
            out_of_bounds[row_indices] = oob
            for start in range(0, len(images), opts.batch_size):
                chunk_rows = row_indices[start:start + opts.batch_size]
                for row, pred in zip(chunk_rows, predictor(images[start:start + opts.batch_size])):
                    predictions[row] = pred
    else:
        loader = DataLoader(CropDataset(samples["path"].tolist(), transform), batch_size=opts.batch_size,
                            shuffle=False, num_workers=opts.num_workers)
        done = 0
        for images, idx in loader:
            for row, pred in zip(idx.tolist(), predictor(images)):
                predictions[row] = pred
            done += len(idx)
            if done // 20_000 != (done - len(idx)) // 20_000:
                print(f"  {done}/{len(samples)} images")
    return predictions, out_of_bounds, unreadable


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def evaluate(opts: EvaluateOptions) -> pd.DataFrame:
    """Run inference (and grading, if labelled) and write outputs to ``opts.out_dir``. Returns predictions."""
    opts.input = Path(opts.input)
    out_dir = Path(opts.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(opts.device)

    model, config = load_model(opts.model, opts.weights, device)
    hierarchy = model.hierarchy
    n_levels = hierarchy.max_depth
    if opts.thresholds is not None and len(opts.thresholds) != n_levels:
        raise ValueError(f"--thresholds needs {n_levels} values (one per hierarchy level), got {len(opts.thresholds)}")
    name = _model_display_name(opts)
    print(f"Model: {name} ({config.backbone} + {config.expert_type} experts, hierarchy {config.hierarchy.name}) "
          f"| device: {device}")

    skipped = None
    if opts.input_type == "flowcam":
        samples, skipped = _flowcam_samples(opts)
        metadata = None
    elif opts.input_type == "crops":
        samples = _crop_samples(opts)
        metadata = _csv_metadata_for_crops(samples, opts)
    else:
        raise ValueError(f"input_type must be 'crops' or 'flowcam', got {opts.input_type!r}")

    if opts.exclude_runs:
        drop = samples["run"].map(lambda r: any(p in r for p in opts.exclude_runs))
        print(f"Excluding {int(drop.sum())} images from runs matching {list(opts.exclude_runs)}")
        samples = samples[~drop].reset_index(drop=True)
        if metadata is not None:
            metadata = metadata[~drop.values].reset_index(drop=True)
    if samples.empty:
        raise ValueError("No images left to evaluate.")
    print(f"Images: {len(samples)}")

    folder_to_node = config.hierarchy.folder_to_node(opts.folder_map) if opts.labelled else {}
    name_to_id = hierarchy.name_to_id()
    id_to_name = {nid: node.name for nid, node in hierarchy.nodes.items()}

    predictions, out_of_bounds, unreadable = _run_inference(samples, opts, Predictor(model, device),
                                                            config.resolution)

    # ---- predictions.csv ----
    keys_no_abstain = grading.outcome_keys(n_levels, gated=False)
    rows, graded = [], []
    for i, pred in enumerate(predictions):
        sample = samples.iloc[i]
        row = {"image_file": sample["image_file"]}
        if opts.input_type == "flowcam":
            row["run"] = sample["run"]
        elif not opts.labelled:
            row["folder"] = sample["folder"]
        true_node = None
        if opts.labelled:
            true_node = folder_to_node.get(sample["actual_class"])
            row["actual_class"] = sample["actual_class"]
            row["in_taxonomy"] = true_node is not None
        if pred is None:
            rows.append(row)
            continue
        row["predicted_class_1"] = pred.predicted_1
        row["predicted_class_2"] = pred.predicted_2
        row.update(level_columns(pred, id_to_name, n_levels, opts.thresholds))
        if opts.labelled:
            if true_node is not None:
                outcome = grading.classify_id_outcome(
                    grading.graded_node(pred.path, pred.confidences, opts.thresholds), name_to_id[true_node], hierarchy)
                row.update({k: outcome == k for k in keys_no_abstain})
                graded.append((i, pred, name_to_id[true_node]))
            else:
                row.update({k: None for k in keys_no_abstain})
        if opts.input_type == "flowcam":
            row["out_of_bounds"] = bool(out_of_bounds[i])
        rows.append(row)

    predictions_df = pd.DataFrame(rows)
    if opts.input_type == "flowcam":
        csv_cols = [c for c in samples.columns if c.startswith("csv_") and c != "csv_file"]
        predictions_df = pd.concat([predictions_df, samples[csv_cols].reset_index(drop=True)], axis=1)
        if unreadable.any():
            extra = samples.loc[unreadable, ["image_file", "run", "csv_file"]].assign(status="unreadable mosaic")
            skipped = pd.concat([skipped, extra], ignore_index=True)
    elif metadata is not None:
        predictions_df = pd.concat([predictions_df, metadata.reset_index(drop=True)], axis=1)

    predictions_path = out_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved -> {predictions_path}")
    if skipped is not None and len(skipped):
        skipped.to_csv(out_dir / "skipped_particles.csv", index=False)
        print(f"WARNING: {len(skipped)} particles not predicted: {skipped['status'].value_counts().to_dict()} "
              f"-> {out_dir / 'skipped_particles.csv'}")

    if opts.labelled:
        _write_labelled_outputs(predictions_df, samples, graded, opts, hierarchy, folder_to_node, name, out_dir)
    else:
        print(predictions_df["predicted_class_1"].value_counts().to_string())

    print(f"\nAll outputs saved under {out_dir}")
    return predictions_df


def _write_labelled_outputs(predictions_df, samples, graded, opts, hierarchy, folder_to_node, name, out_dir):
    source = opts.input.name
    mapped = predictions_df[predictions_df["in_taxonomy"] & predictions_df["predicted_class_1"].notna()].copy()
    unmapped = predictions_df.loc[~predictions_df["in_taxonomy"], "actual_class"].value_counts()
    print(f"Graded {len(mapped)} of {len(predictions_df)} images against the taxonomy")
    if len(unmapped):
        print("Folders not in the taxonomy (kept in predictions.csv, not graded): "
              + ", ".join(f"{k} ({v})" for k, v in unmapped.items()))
    if mapped.empty:
        print("Nothing to grade: no labelled folder maps onto the taxonomy (see --folder-map).")
        return
    mapped["true_node"] = mapped["actual_class"].map(folder_to_node)

    # ---- leaf_accuracy ----
    def leaf_table(df):
        if opts.strict_leaf_accuracy:
            valid = {t: {hierarchy.nodes[n].name for n in hierarchy.subtree_leaves(hierarchy.name_to_id()[t])}
                     for t in df["true_node"].unique()}
            correct = [p in valid[t] for p, t in zip(df["predicted_class_1"], df["true_node"])]
            return grading.leaf_metrics_table(df["true_node"], df["predicted_class_1"], correct, hierarchy)
        return grading.loosened_leaf_metrics_table(df["true_node"], df["predicted_class_1"], hierarchy)

    footnote = (FOOTNOTE_STRICT + FOOTNOTE_COARSE) if opts.strict_leaf_accuracy else FOOTNOTE_LOOSENED
    plotting.render_table(leaf_table(mapped), f"Leaf-class metrics on {source} (Manually Classified) -- {name}",
                          out_dir / "leaf_accuracy.png", summary_row="Total", highlight_row_max=False,
                          footnote=footnote, count_cols={"N"})
    if opts.capped:
        capped_df = mapped.loc[grading.capped_indices_by_group(mapped["true_node"], opts.capped, opts.seed)]
        plotting.render_table(leaf_table(capped_df),
                              f"Leaf-class metrics on {source} (Manually Classified, capped at {opts.capped}/leaf) -- {name}",
                              out_dir / "leaf_accuracy_capped.png", summary_row="Total", highlight_row_max=False,
                              footnote=footnote, count_cols={"N"})
    if opts.zoop_only:
        plotting.render_table(
            grading.loosened_leaf_metrics_table(mapped["true_node"], mapped["predicted_class_1"], hierarchy,
                                                exclude=frozenset(opts.zoop_only_exclude)),
            f"Leaf-class metrics on {source} (Manually Classified, zoop only, partial credit for _Unknown) -- {name}",
            out_dir / "leaf_accuracy_zoop_only.png", summary_row="Total", highlight_row_max=False,
            footnote=FOOTNOTE_LOOSENED, count_cols={"N"})

    # ---- hierarchy_level_metrics ----
    plotting.render_hierarchy_level_table(
        grading.hierarchy_level_metrics_table(mapped["true_node"], mapped["predicted_class_1"], hierarchy),
        f"Per-hierarchy-level metrics on {source} (Manually Classified) -- {name}",
        out_dir / "hierarchy_level_metrics.png", footnote=FOOTNOTE_LEVEL, count_cols={"N"})

    if opts.training_accuracy:
        table = grading.training_result_accuracy_table(opts.weights, hierarchy) if opts.weights else None
        if table is None:
            print("No predictions.npz next to the weights -- skipping training_result_accuracy.png "
                  "(needs --weights pointing at a training run directory)")
        else:
            plotting.render_table(table, f"Leaf-class metrics on {name}'s held-out training test split "
                                         f"(partial credit for _Unknown)",
                                  out_dir / "training_result_accuracy.png", summary_row="Total",
                                  highlight_row_max=False, footnote=FOOTNOTE_LOOSENED, count_cols={"N"})

    if opts.breakdown:
        plotting.render_table(grading.top_predictions_table(predictions_df, opts.breakdown),
                              f"Top predicted classes by folder on {source} -- {name}",
                              out_dir / "top_predictions_breakdown.png",
                              label_cols=["N", "#1 predicted", "#2 predicted", "#3 predicted"],
                              footnote=FOOTNOTE_BREAKDOWN)

    # ---- id_outcomes ----
    gated = opts.thresholds is not None
    keys = grading.outcome_keys(hierarchy.max_depth, gated)
    labels = tuple(grading.outcome_label(k) for k in keys)
    colors = grading.outcome_colors(keys)
    suffix = f" (thresholds={tuple(opts.thresholds)})" if gated else " (raw greedy leaf, no confidence gating)"
    paths = [p.path for _, p, _ in graded]
    confs = [p.confidences for _, p, _ in graded]
    trues = [t for _, _, t in graded]

    table, props = grading.id_outcome_table(paths, confs, trues, hierarchy, keys, opts.thresholds)
    fig = plotting.plot_outcome_stacked_bar(props, f"Per-class ID outcomes on {source}{suffix} -- {name}",
                                            keys=keys, labels=labels, colors=colors)
    fig.savefig(out_dir / "id_outcomes_stacked_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_dir / 'id_outcomes_stacked_bar.png'}")
    plotting.render_table(table, f"Per-class ID outcomes -- {name}", out_dir / "id_outcomes_table.png",
                          summary_row="Total", count_cols={"N"})

    if opts.by_sample:
        runs = [samples.iloc[i]["run"] for i, _, _ in graded]
        by_run = grading.id_outcome_proportions(paths, confs, trues, runs, hierarchy, keys, opts.thresholds)
        fig = plotting.plot_outcome_stacked_bar(by_run, f"Per-sample ID outcomes on {source}{suffix} -- {name}",
                                                keys=keys, labels=labels, colors=colors, legend_side=True)
        fig.savefig(out_dir / "id_outcomes_stacked_bar_by_sample.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {out_dir / 'id_outcomes_stacked_bar_by_sample.png'}")

    if opts.capped:
        idx = grading.capped_indices_by_group(trues, opts.capped, opts.seed)
        capped_table, _ = grading.id_outcome_table([paths[i] for i in idx], [confs[i] for i in idx],
                                                   [trues[i] for i in idx], hierarchy, keys, opts.thresholds)
        plotting.render_table(capped_table, f"Per-class ID outcomes (capped at {opts.capped}/leaf) -- {name}",
                              out_dir / "id_outcomes_table_capped.png", summary_row="Total", count_cols={"N"})
