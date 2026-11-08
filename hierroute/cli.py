"""
cli.py — The ``hierroute`` command.

    hierroute crop      FlowCam exports -> individual particle .tif crops + manifest.csv
    hierroute evaluate  crops or FlowCam exports -> predictions.csv (+ metrics with --labelled)
    hierroute train     training config YAML -> run directory usable by evaluate --weights
    hierroute download  fetch released model weights ahead of time

Run ``hierroute <command> -h`` for each command's options.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


def _add_class_rule_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("FlowCam class labels")
    group.add_argument("--label-column", default="Class", help="CSV column with the manual class (default: Class)")
    group.add_argument("--append-group-for", nargs="*", default=["Nauplius"], metavar="CLASS",
                       help="classes renamed to <class>_<group> from the CSV's Group column "
                            "(default: Nauplius -> Nauplius_Copepod); pass with no values to disable")
    group.add_argument("--default-group", default="Copepod",
                       help="group used when the CSV has no Group column (default: Copepod)")
    group.add_argument("--class-map", type=Path, metavar="YAML",
                       help="YAML mapping {CSV class: new class name}, applied after the group rule")
    group.add_argument("--ignore-classes", nargs="*", default=[], metavar="CLASS",
                       help="classes to skip entirely (after renaming), e.g. Other Duplicate")


def _class_rules(args):
    import yaml
    from .flowcam import ClassRules
    rename = {}
    if args.class_map:
        rename = {str(k): str(v) for k, v in (yaml.safe_load(args.class_map.read_text()) or {}).items()}
    return ClassRules(label_column=args.label_column, append_group_for=list(args.append_group_for),
                      default_group=args.default_group, rename=rename, ignore=set(args.ignore_classes))


# ---------------------------------------------------------------------------
# crop
# ---------------------------------------------------------------------------
def cmd_crop(args) -> int:
    from .flowcam import load_particles, write_crops
    rules = _class_rules(args) if not args.unlabelled else None
    try:
        table = load_particles(args.input, args.csv_dir, rules)
    except ValueError as err:
        if rules is not None and f"'{rules.label_column}' not found" in str(err):
            print(f"{err}\nThese exports have no manual classes: re-run with --unlabelled "
                  f"(or set --label-column).", file=sys.stderr)
            return 2
        raise
    print(table.summary())
    by = "run" if rules is None else "label"
    manifest = write_crops(table, args.out, by=by, overwrite=args.overwrite, workers=args.workers)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    counts = manifest["crop_status"].value_counts().to_dict()
    print(f"Crops: {counts}")
    if manifest["out_of_bounds"].any():
        print(f"Note: {int(manifest['out_of_bounds'].sum())} crop boxes extend past their mosaic (zero-padded)")
    if rules is not None:
        print("Crops per class:\n" + manifest.loc[manifest["crop_path"].notna(), "label"].value_counts().to_string())
    print(f"Saved -> {manifest_path}")
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def cmd_evaluate(args) -> int:
    from .config import load_folder_map
    from .evaluate import EvaluateOptions, evaluate
    if args.model and args.weights:
        print("Use either --model or --weights, not both.", file=sys.stderr)
        return 2
    opts = EvaluateOptions(
        input=args.input,
        out_dir=args.out,
        model=args.model,
        weights=args.weights,
        input_type=args.input_type,
        labelled=args.labelled,
        class_rules=_class_rules(args),
        csv_dir=args.csv_dir,
        folder_map=load_folder_map(args.folder_map) if args.folder_map else None,
        thresholds=tuple(args.thresholds) if args.thresholds else None,
        strict_leaf_accuracy=args.strict_leaf_accuracy,
        exclude_runs=tuple(args.exclude_runs),
        zoop_only=args.zoop_only,
        by_sample=args.by_sample,
        breakdown=tuple(args.breakdown),
        training_accuracy=args.training_accuracy,
        capped=args.capped,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    evaluate(opts)
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def cmd_train(args) -> int:
    from .config import load_train_config, set_dotted
    from .train import train
    config = load_train_config(args.config)
    shortcuts = {
        "hierarchy": ("hierarchy",), "backbone": ("model", "backbone"), "expert": ("model", "expert"),
        "data_dir": ("data", "data_dir"), "out_dir": ("out_dir",), "run_name": ("run_name",),
        "epochs": ("train", "max_epochs"), "lr": ("train", "lr"), "batch_size": ("train", "batch_size"),
        "num_workers": ("train", "num_workers"), "device": ("train", "device"), "seed": ("train", "seed"),
    }
    for arg, keys in shortcuts.items():
        value = getattr(args, arg)
        if value is not None:
            target = config
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = str(value) if isinstance(value, Path) else value
    if args.noise is not None:
        config["augment"]["background_noise"] = args.noise
    for item in args.set or []:
        if "=" not in item:
            print(f"--set expects section.key=value, got {item!r}", file=sys.stderr)
            return 2
        key, value = item.split("=", 1)
        set_dotted(config, key, value)
    train(config)
    return 0


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
def cmd_download(args) -> int:
    from .pretrained import PRETRAINED, cache_dir, download
    if args.list:
        for name, entry in PRETRAINED.items():
            print(f"{name:10s} {entry['file']:32s} {entry['description']}")
        print(f"\nCache folder: {cache_dir()} (override with HIERROUTE_CACHE_DIR)")
        return 0
    names = list(PRETRAINED) if "all" in args.models else args.models
    for name in names:
        path = download(name, force=args.force)
        print(f"{name}: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    parser = argparse.ArgumentParser(prog="hierroute", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"hierroute {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # crop
    p = sub.add_parser("crop", help="crop particles out of FlowCam exports",
                       description="Crop every particle in FlowCam exports (mosaic .tif + CSV) into its own "
                                   ".tif, one folder per class (labelled) or per run (--unlabelled), and write "
                                   "manifest.csv with each crop's FlowCam measurements.")
    p.add_argument("input", type=Path, help="folder with the mosaic .tif files (searched recursively)")
    p.add_argument("--out", type=Path, required=True, help="output folder for the crops")
    p.add_argument("--csv-dir", type=Path, help="folder with the export CSVs, if not under INPUT")
    p.add_argument("--unlabelled", action="store_true",
                   help="ignore class labels; write crops into one folder per run")
    p.add_argument("--overwrite", action="store_true", help="re-write crops that already exist")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="parallel processes")
    _add_class_rule_args(p)
    p.set_defaults(func=cmd_crop)

    # evaluate
    p = sub.add_parser("evaluate", help="predict on particle images (and grade them if labelled)",
                       description="Run a model on particle crops or raw FlowCam exports and write "
                                   "predictions.csv. With --labelled, also grade against the manual labels "
                                   "and write leaf_accuracy.png, id_outcomes_table.png, "
                                   "id_outcomes_stacked_bar.png and hierarchy_level_metrics.png.")
    p.add_argument("input", type=Path, help="folder of crops, or of FlowCam exports with --input-type flowcam")
    p.add_argument("--out", type=Path, required=True, help="output folder")
    which = p.add_argument_group("model")
    which.add_argument("--model", choices=["noise", "baseline"],
                       help="released model (default: noise -- trained with synthetic background dirt)")
    which.add_argument("--weights", type=Path, help="run directory (or .pt) from hierroute train")
    data = p.add_argument_group("input")
    data.add_argument("--input-type", choices=["crops", "flowcam"], default="crops",
                      help="crops: individual particle .tif files (default); flowcam: mosaics + CSVs")
    data.add_argument("--labelled", action="store_true",
                      help="crops are in one folder per manual class (or, for flowcam, the CSV has a class column); "
                           "adds the metric outputs")
    data.add_argument("--csv-dir", type=Path, help="FlowCam CSVs for crop measurements / flowcam input, if not under INPUT")
    data.add_argument("--exclude-runs", nargs="*", default=[], metavar="PATTERN",
                      help="drop images whose FlowCam run name contains any PATTERN")
    _add_class_rule_args(p)
    grade = p.add_argument_group("grading (with --labelled)")
    grade.add_argument("--folder-map", type=Path, metavar="YAML",
                       help="YAML {folder name: taxonomy node} added to the hierarchy's own folder mapping")
    grade.add_argument("--thresholds", type=float, nargs="+", metavar="T",
                       help="per-level confidence thresholds (e.g. 0.98 0.995 0.99): ID outcomes back off to a "
                            "coarser level, or abstain, when a decision is below its threshold")
    grade.add_argument("--strict-leaf-accuracy", action="store_true",
                       help="leaf_accuracy counts exact matches only (no credit for a branch's _Unknown leaf)")
    extra = p.add_argument_group("optional outputs (with --labelled)")
    extra.add_argument("--zoop-only", action="store_true", help="leaf_accuracy_zoop_only.png (leaf accuracy without Zoop-No)")
    extra.add_argument("--by-sample", action="store_true", help="id_outcomes_stacked_bar_by_sample.png (one bar per run)")
    extra.add_argument("--breakdown", nargs="*", default=[], metavar="FOLDER",
                       help="top_predictions_breakdown.png: top-3 predictions for these folders")
    extra.add_argument("--training-accuracy", action="store_true",
                       help="training_result_accuracy.png from the --weights run's held-out test split")
    extra.add_argument("--capped", type=int, nargs="?", const=3000, metavar="N",
                       help="also write leaf_accuracy_capped.png / id_outcomes_table_capped.png with at most N "
                            "images per class (default N: 3000)")
    run = p.add_argument_group("runtime")
    run.add_argument("--device", default="auto", help="auto (default), cpu, cuda or cuda:N")
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--num-workers", type=int, default=4)
    p.set_defaults(func=cmd_evaluate)

    # train
    p = sub.add_parser("train", help="train a model from a config file",
                       description="Train a HierRouteNet. Settings come from the built-in defaults, then "
                                   "--config, then the flags below.")
    p.add_argument("--config", type=Path, help="training config YAML (see configs/ in the repository)")
    p.add_argument("--hierarchy", help="bundled hierarchy name (mnr_17leaf, mnr_13leaf, whoi_s, whoi_l) or YAML path")
    p.add_argument("--backbone", choices=["swin_t", "swin_s", "efficientnet_b0"])
    p.add_argument("--expert", choices=["cnn", "mlp", "linear"])
    p.add_argument("--data-dir", type=Path, help="folder with one subfolder per class")
    p.add_argument("--out-dir", type=Path, help="where run directories are created")
    p.add_argument("--run-name", help="run directory name (default: <backbone>_<expert>_<noise|baseline>_<MM-DD>)")
    noise = p.add_mutually_exclusive_group()
    noise.add_argument("--noise", dest="noise", action="store_true", default=None,
                       help="train with synthetic background dirt augmentation")
    noise.add_argument("--no-noise", dest="noise", action="store_false", help="train without it")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--device", help="auto, cpu, cuda or cuda:N")
    p.add_argument("--seed", type=int)
    p.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE",
                   help="override any config value, e.g. --set data.max_class_size=6000 (repeatable)")
    p.set_defaults(func=cmd_train)

    # download
    p = sub.add_parser("download", help="download released model weights",
                       description="Download released weights into the cache folder (for offline machines).")
    p.add_argument("models", nargs="*", default=["all"], help="noise, baseline or all (default: all)")
    p.add_argument("--list", action="store_true", help="list released models and the cache folder")
    p.add_argument("--force", action="store_true", help="download again even if cached")
    p.set_defaults(func=cmd_download)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
