import sys
import os
import tempfile
import shutil

import pytest
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Ensure hierroute is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hierroute.hierarchy import Hierarchy
from hierroute.constants import hier_adjacency_graph


# ------------------------------------------------------------------ #
#  Hierarchy fixtures                                                  #
# ------------------------------------------------------------------ #

@pytest.fixture
def mnr_adjacency_graph():
    return hier_adjacency_graph


@pytest.fixture
def hierarchy_and_labels(mnr_adjacency_graph):
    """Build a Hierarchy + label_to_id from the MNR adjacency graph."""
    label_to_id = {name: i for i, name in enumerate(mnr_adjacency_graph.keys())}

    hierarchy = Hierarchy()
    all_children = set()
    for _, kids in mnr_adjacency_graph.items():
        all_children.update(kids)
    root = [n for n in mnr_adjacency_graph if n not in all_children][0]

    hierarchy.add_node(label_to_id[root], root)

    def dfs(parent):
        for child in mnr_adjacency_graph[parent]:
            hierarchy.add_node(label_to_id[child], child, label_to_id[parent])
            dfs(child)

    dfs(root)
    return hierarchy, label_to_id


# ------------------------------------------------------------------ #
#  Small hierarchy for quick tests                                     #
# ------------------------------------------------------------------ #

@pytest.fixture
def small_adjacency_graph():
    """A minimal 3-level tree: root → A, B; A → A1, A2; B → (leaf)."""
    return {
        "root": ["A", "B"],
        "A": ["A1", "A2"],
        "B": [],
        "A1": [],
        "A2": [],
    }


@pytest.fixture
def small_hierarchy(small_adjacency_graph):
    label_to_id = {name: i for i, name in enumerate(small_adjacency_graph.keys())}
    hierarchy = Hierarchy()

    all_children = set()
    for _, kids in small_adjacency_graph.items():
        all_children.update(kids)
    root = [n for n in small_adjacency_graph if n not in all_children][0]

    hierarchy.add_node(label_to_id[root], root)

    def dfs(parent):
        for child in small_adjacency_graph[parent]:
            hierarchy.add_node(label_to_id[child], child, label_to_id[parent])
            dfs(child)

    dfs(root)
    return hierarchy, label_to_id


# ------------------------------------------------------------------ #
#  Tensor fixtures                                                     #
# ------------------------------------------------------------------ #

@pytest.fixture
def dummy_batch():
    return torch.randn(4, 3, 64, 64)


# ------------------------------------------------------------------ #
#  Temporary image dataset on disk                                     #
# ------------------------------------------------------------------ #

@pytest.fixture
def tmp_image_dir():
    """Create a temp directory with synthetic 64×64 .tif images in 3 class folders."""
    tmpdir = tempfile.mkdtemp()
    classes = ["ClassA", "ClassB", "ClassC"]
    n_images_per_class = 20

    for cls in classes:
        cls_dir = os.path.join(tmpdir, cls)
        os.makedirs(cls_dir)
        for i in range(n_images_per_class):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L"
            )
            img.save(os.path.join(cls_dir, f"img_{i:03d}.tif"))

    yield tmpdir, classes, n_images_per_class
    shutil.rmtree(tmpdir)


@pytest.fixture
def base_image_dataset(tmp_image_dir):
    """ImageDataset built from synthetic images."""
    from hierroute.data_setup import ImageDataset

    tmpdir, classes, _ = tmp_image_dir
    ds = ImageDataset(
        data_directory=tmpdir,
        class_names=classes,
        max_class_size=20,
        image_resolution=64,
        image_transforms=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]),
        format_file=".tif",
    )
    return ds


# ------------------------------------------------------------------ #
#  Synthetic FlowCam export + toy hierarchy (crop / evaluate / train) #
# ------------------------------------------------------------------ #

TOY_HIERARCHY = {
    "name": "toy",
    "tree": {"root": ["Animal", "Junk"], "Animal": ["Cat", "Dog", "Animal_Unknown"], "Junk": [],
             "Cat": [], "Dog": [], "Animal_Unknown": []},
    "folder_map": {"Junk": ["Debris"]},
    "eval_folder_map": {"Critter": "Animal_Unknown"},
}

# (class, x, y, w, h) per particle on each of two mosaics
TOY_PARTICLES = {
    "runA_000001.tif": [("Cat", 5, 5, 30, 40), ("Dog", 50, 10, 25, 25), ("Debris", 100, 60, 20, 30),
                        ("Nauplius", 10, 80, 30, 30), ("Other", 150, 100, 20, 20)],
    "runA_000002.tif": [("Critter", 0, 0, 40, 40), ("Cat", 60, 60, 35, 35), ("Dog", 180, 120, 40, 40)],
}


@pytest.fixture
def toy_hierarchy_file(tmp_path):
    import yaml
    path = tmp_path / "toy_hierarchy.yaml"
    path.write_text(yaml.safe_dump(TOY_HIERARCHY, sort_keys=False))
    return path


@pytest.fixture
def flowcam_export(tmp_path):
    """exports/runA/ with two RGB mosaics and runA.csv (FlowCam column names)."""
    run_dir = tmp_path / "exports" / "runA"
    run_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, pid = [], 1
    for mosaic_name, particles in TOY_PARTICLES.items():
        Image.fromarray(rng.integers(0, 255, (150, 200, 3), dtype=np.uint8), mode="RGB").save(run_dir / mosaic_name)
        for cls, x, y, w, h in particles:
            rows.append({"Particle ID": pid, "Class": cls, "Area (ABD)": 10.0 * pid, "Image File": mosaic_name,
                         "Image X": x, "Image Y": y, "Image Width": w, "Image Height": h})
            pid += 1
    import pandas as pd
    pd.DataFrame(rows).to_csv(run_dir / "runA.csv", index=False)
    Image.fromarray(np.zeros((150, 200), dtype=np.uint8)).save(run_dir / "runA_000001_bin.tif")
    return tmp_path / "exports"


@pytest.fixture
def toy_run_dir(tmp_path, toy_hierarchy_file):
    """An untrained toy model saved like a `hierroute train` run directory."""
    from hierroute.config import load_hierarchy
    from hierroute.pretrained import ModelConfig, build_model

    config = ModelConfig(name="toy_run", backbone="efficientnet_b0", expert_type="linear", resolution=32,
                         hierarchy=load_hierarchy(toy_hierarchy_file))
    torch.manual_seed(0)
    model = build_model(config)
    run_dir = tmp_path / "toy_run"
    run_dir.mkdir()
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    config.save(run_dir)
    return run_dir
