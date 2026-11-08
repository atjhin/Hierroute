import numpy as np
import pandas as pd
import pytest
from PIL import Image

from hierroute.flowcam import (ClassRules, csv_column_to_snake_case, find_mosaics, load_particles,
                               read_flowcam_csv, run_name_from_crop, write_crops)


def test_naming_helpers():
    assert csv_column_to_snake_case("Area (ABD)") == "csv_area_abd"
    assert run_name_from_crop("2024_Erie_Rep1_000012_34.tif") == "2024_Erie_Rep1"
    assert run_name_from_crop("odd_name.tif") == "odd_name"


def test_find_mosaics_skips_masks(flowcam_export):
    assert sorted(find_mosaics(flowcam_export)) == ["runA_000001.tif", "runA_000002.tif"]


def test_unquoted_comma_in_image_file(tmp_path):
    path = tmp_path / "run,1.csv"
    path.write_text("Particle ID,Image File,Image X,Image Y,Image Width,Image Height\n"
                    "1,run,1_000001.tif,0,0,5,5\n")
    df = read_flowcam_csv(path)
    assert df.loc[0, "Image File"] == "run,1_000001.tif"
    assert df.loc[0, "Image Width"] == 5


def test_class_rules():
    df = pd.DataFrame({"Class": ["Nauplius", "Cat", "Nauplius", "Other"], "group": ["Rotifer", None, None, None]})
    labels = ClassRules(rename={"Cat": "Feline"}).labels(df)
    assert labels.tolist() == ["Nauplius_Rotifer", "Feline", "Nauplius_Copepod", "Other"]
    no_group = pd.DataFrame({"Class": ["Nauplius"]})
    assert ClassRules().labels(no_group).tolist() == ["Nauplius_Copepod"]
    assert ClassRules(append_group_for=[]).labels(no_group).tolist() == ["Nauplius"]


def test_load_particles_labelled(flowcam_export):
    table = load_particles(flowcam_export, rules=ClassRules(ignore={"Other"}))
    p = table.particles
    assert len(p) == 8
    assert p["status"].value_counts().to_dict() == {"ok": 7, "ignored class": 1}
    assert "csv_area_abd" in p.columns
    assert p.loc[p["particle_id"] == 1, "image_file"].item() == "runA_000001_1.tif"
    assert p.loc[p["particle_id"] == 4, "label"].item() == "Nauplius_Copepod"
    assert set(p["run"]) == {"runA"}


def test_separate_csv_dir(flowcam_export, tmp_path):
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    (flowcam_export / "runA" / "runA.csv").rename(csv_dir / "runA_data.csv")
    table = load_particles(flowcam_export, csv_dir=csv_dir, rules=ClassRules())
    assert (table.particles["status"] == "ok").all()


def test_missing_mosaic(flowcam_export):
    (flowcam_export / "runA" / "runA_000002.tif").unlink()
    table = load_particles(flowcam_export, rules=ClassRules())
    assert (table.particles["status"] == "missing mosaic").sum() == 3


def test_write_crops_by_label_matches_manual_crop(flowcam_export, tmp_path):
    out = tmp_path / "crops"
    manifest = write_crops(load_particles(flowcam_export, rules=ClassRules()), out, by="label", workers=1)
    assert (manifest["crop_status"] == "written").all()
    assert sorted(p.name for p in out.iterdir()) == ["Cat", "Critter", "Debris", "Dog", "Nauplius_Copepod", "Other"]

    mosaic = Image.open(flowcam_export / "runA" / "runA_000001.tif")
    expected = np.asarray(mosaic.crop((5, 5, 35, 45)))
    actual = np.asarray(Image.open(out / "Cat" / "runA_000001_1.tif"))
    assert np.array_equal(expected, actual)

    again = write_crops(load_particles(flowcam_export, rules=ClassRules()), out, by="label", workers=1)
    assert (again["crop_status"] == "exists").all()


def test_write_crops_by_run_flags_out_of_bounds(flowcam_export, tmp_path):
    manifest = write_crops(load_particles(flowcam_export), tmp_path / "crops", by="run", workers=2)
    assert (tmp_path / "crops" / "runA").is_dir()
    assert manifest["out_of_bounds"].sum() == 1   # the Dog at (180, 120, 40, 40) on a 200x150 mosaic


def test_missing_label_column(flowcam_export):
    with pytest.raises(ValueError, match="Label column"):
        load_particles(flowcam_export, rules=ClassRules(label_column="Species"))
