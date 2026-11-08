"""End-to-end runs of the `hierroute` command on tiny synthetic data (CPU, no downloads)."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from hierroute.cli import main


def _evaluate(*args):
    return main(["evaluate", *map(str, args), "--device", "cpu", "--num-workers", "0", "--batch-size", "4"])


@pytest.fixture
def crops(flowcam_export, tmp_path):
    out = tmp_path / "crops"
    assert main(["crop", str(flowcam_export), "--out", str(out), "--workers", "1"]) == 0
    return out


def test_crop_writes_manifest(crops):
    manifest = pd.read_csv(crops / "manifest.csv")
    assert len(manifest) == 8
    assert {"image_file", "label", "crop_path", "crop_status", "csv_area_abd"} <= set(manifest.columns)


def test_crop_unlabelled(flowcam_export, tmp_path):
    out = tmp_path / "by_run"
    assert main(["crop", str(flowcam_export), "--out", str(out), "--unlabelled", "--workers", "1"]) == 0
    assert len(list((out / "runA").glob("*.tif"))) == 8


def test_evaluate_labelled_crops(crops, toy_run_dir, tmp_path):
    out = tmp_path / "eval"
    assert _evaluate(crops, "--out", out, "--weights", toy_run_dir, "--labelled") == 0
    for name in ("predictions.csv", "leaf_accuracy.png", "id_outcomes_table.png",
                 "id_outcomes_stacked_bar.png", "hierarchy_level_metrics.png"):
        assert (out / name).exists(), name
    assert not (out / "leaf_accuracy_zoop_only.png").exists()

    pred = pd.read_csv(out / "predictions.csv")
    assert len(pred) == 8
    assert list(pred.columns[:5]) == ["image_file", "actual_class", "in_taxonomy", "predicted_class_1",
                                      "predicted_class_2"]
    in_tax = dict(zip(pred["actual_class"], pred["in_taxonomy"]))
    assert in_tax["Debris"] and in_tax["Critter"] and in_tax["Cat"]          # folder_map / eval_folder_map / leaf
    assert not in_tax["Other"] and not in_tax["Nauplius_Copepod"]
    graded = pred[pred["in_taxonomy"]]
    outcomes = ["correct_all", "correct_incomplete", "incorrect_l2", "incorrect_all"]
    assert (graded[outcomes].sum(axis=1) == 1).all()
    assert pred["csv_area_abd"].notna().all()                                # from manifest.csv


def test_flowcam_input_matches_crops(flowcam_export, crops, toy_run_dir, tmp_path):
    _evaluate(crops, "--out", tmp_path / "a", "--weights", toy_run_dir, "--labelled")
    _evaluate(flowcam_export, "--out", tmp_path / "b", "--weights", toy_run_dir, "--labelled",
              "--input-type", "flowcam")
    a = pd.read_csv(tmp_path / "a" / "predictions.csv").set_index("image_file").sort_index()
    b = pd.read_csv(tmp_path / "b" / "predictions.csv").set_index("image_file").sort_index()
    assert (a["predicted_class_1"] == b["predicted_class_1"]).all()
    assert np.allclose(a["confidence_level_1"], b["confidence_level_1"], atol=1e-5)
    assert b["out_of_bounds"].sum() == 1


def test_evaluate_unlabelled(crops, toy_run_dir, tmp_path):
    out = tmp_path / "eval"
    assert _evaluate(crops, "--out", out, "--weights", toy_run_dir) == 0
    pred = pd.read_csv(out / "predictions.csv")
    assert "actual_class" not in pred.columns and "folder" in pred.columns
    assert not (out / "leaf_accuracy.png").exists()


def test_evaluate_thresholds_and_optional_outputs(crops, toy_run_dir, tmp_path):
    out = tmp_path / "eval"
    assert _evaluate(crops, "--out", out, "--weights", toy_run_dir, "--labelled", "--thresholds", "0.99", "0.99",
                     "--zoop-only", "--by-sample", "--breakdown", "Other", "--capped", "2",
                     "--strict-leaf-accuracy") == 0
    for name in ("leaf_accuracy_zoop_only.png", "id_outcomes_stacked_bar_by_sample.png",
                 "top_predictions_breakdown.png", "leaf_accuracy_capped.png", "id_outcomes_table_capped.png"):
        assert (out / name).exists(), name
    assert "pass_level_1" in pd.read_csv(out / "predictions.csv").columns


def test_exclude_runs(crops, toy_run_dir, tmp_path):
    with pytest.raises(ValueError, match="No images left"):
        _evaluate(crops, "--out", tmp_path / "eval", "--weights", toy_run_dir, "--labelled", "--exclude-runs", "runA")


def test_wrong_threshold_count(crops, toy_run_dir, tmp_path):
    with pytest.raises(ValueError, match="needs 2 values"):
        _evaluate(crops, "--out", tmp_path / "eval", "--weights", toy_run_dir, "--thresholds", "0.9")


def test_no_crops_hints_flowcam(flowcam_export, toy_run_dir, tmp_path):
    (flowcam_export / "runA" / "runA_000001_bin.tif").unlink()
    for mosaic in (flowcam_export / "runA").glob("*.tif"):
        mosaic.unlink()
    with pytest.raises(FileNotFoundError, match="input-type flowcam"):
        _evaluate(flowcam_export, "--out", tmp_path / "eval", "--weights", toy_run_dir)


def test_train_then_evaluate(tmp_path, toy_hierarchy_file):
    data = tmp_path / "data"
    rng = np.random.default_rng(0)
    for cls in ("Cat", "Dog", "Animal_Unknown", "Debris"):
        (data / cls).mkdir(parents=True)
        for i in range(15):
            Image.fromarray(rng.integers(0, 255, (40, 40), dtype=np.uint8)).save(data / cls / f"{cls}_{i}.tif")
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump({
        "hierarchy": str(toy_hierarchy_file),
        "model": {"backbone": "efficientnet_b0", "expert": "linear", "pretrained_backbone": False},
        "data": {"data_dir": str(data), "resolution": 32},
        "train": {"max_epochs": 1, "batch_size": 6, "num_workers": 0, "device": "cpu"},
        "out_dir": str(tmp_path / "runs"),
    }))
    assert main(["train", "--config", str(config), "--run-name", "toy", "--no-noise"]) == 0

    run_dir = tmp_path / "runs" / "toy"
    for name in ("best_model.pt", "model_config.json", "train_config.yaml", "training_info.json",
                 "eval_results.json", "predictions.npz"):
        assert (run_dir / name).exists(), name
    model_config = json.loads((run_dir / "model_config.json").read_text())
    assert model_config["backbone"] == "efficientnet_b0" and model_config["hierarchy"]["name"] == "toy"
    assert yaml.safe_load((run_dir / "train_config.yaml").read_text())["augment"]["background_noise"] is False

    out = tmp_path / "eval"
    assert _evaluate(data, "--out", out, "--weights", run_dir, "--labelled", "--training-accuracy") == 0
    assert (out / "training_result_accuracy.png").exists()


def test_download_list(capsys):
    assert main(["download", "--list"]) == 0
    assert "noise" in capsys.readouterr().out
