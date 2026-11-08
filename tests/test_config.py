import pytest
import yaml

from hierroute import constants
from hierroute.config import (BUNDLED_HIERARCHIES, HierarchySpec, TRAIN_DEFAULTS, deep_merge,
                              load_hierarchy, load_train_config, set_dotted)
from hierroute.pretrained import PRETRAINED, tree_fingerprint


# ================================================================== #
#  Bundled hierarchy YAML files                                        #
# ================================================================== #

class TestBundledHierarchies:
    @pytest.mark.parametrize("name, graph", [
        ("mnr_17leaf", constants.hiernet_adjacency_graph),
        ("mnr_13leaf", constants.hier_adjacency_graph),
        ("whoi_s", constants.whoi_adjacency_graph_s),
        ("whoi_l", constants.whoi_adjacency_graph_l),
    ])
    def test_tree_matches_constants_including_order(self, name, graph):
        spec = load_hierarchy(name)
        assert list(spec.tree.items()) == list(graph.items())

    def test_mnr_17leaf_folder_map_matches_constants(self):
        assert load_hierarchy("mnr_17leaf").folder_map == constants.HIERNET_CLASS_FOLDER_MAP

    def test_mnr_13leaf_classes_match_constants(self):
        spec = load_hierarchy("mnr_13leaf")
        assert spec.training_classes == constants.ZOOPLANKTON_CLASSES
        assert spec.folder_map == constants.MNR_CLASS_FOLDER_MAP

    def test_released_weights_fingerprints_match_bundled_hierarchy(self):
        for name, entry in PRETRAINED.items():
            spec = load_hierarchy(entry["hierarchy"])
            assert tree_fingerprint(spec.tree) == entry["hierarchy_fingerprint"], name

    @pytest.mark.parametrize("name", BUNDLED_HIERARCHIES)
    def test_build(self, name):
        spec = load_hierarchy(name)
        hierarchy, label_to_id = spec.build()
        assert len(hierarchy) == len(spec.tree)
        assert sorted(label_to_id) == sorted(spec.tree)

    def test_unknown_name_raises(self):
        with pytest.raises(FileNotFoundError):
            load_hierarchy("not_a_hierarchy")


# ================================================================== #
#  HierarchySpec                                                       #
# ================================================================== #

@pytest.fixture
def spec():
    return HierarchySpec(
        name="toy",
        tree={"root": ["Animal", "Junk"], "Animal": ["Cat", "Dog", "Animal_Unknown"], "Junk": [],
              "Cat": [], "Dog": [], "Animal_Unknown": []},
        folder_map={"Junk": ["Debris", "Bubbles"], "Animal_Unknown": ["Animal"]},
        eval_folder_map={"Critter": "Animal_Unknown"},
    )


class TestHierarchySpec:
    def test_leaves_and_training_classes(self, spec):
        assert spec.leaves == ["Junk", "Cat", "Dog", "Animal_Unknown"]
        assert spec.training_classes == spec.leaves

    def test_folder_to_node_priority(self, spec):
        mapping = spec.folder_to_node({"Bubbles": "Cat"})
        assert mapping["Cat"] == "Cat"                 # leaf's own name
        assert mapping["Debris"] == "Junk"             # folder_map
        assert mapping["Animal"] == "Animal_Unknown"   # folder_map
        assert mapping["Critter"] == "Animal_Unknown"  # eval_folder_map
        assert mapping["Bubbles"] == "Cat"             # extra overrides folder_map

    def test_folder_to_node_rejects_unknown_nodes(self, spec):
        with pytest.raises(ValueError):
            spec.folder_to_node({"X": "NotANode"})

    def test_round_trip(self, spec):
        assert HierarchySpec.from_dict(spec.to_dict()) == spec

    def test_yaml_file(self, spec, tmp_path):
        path = tmp_path / "toy.yaml"
        path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False))
        assert load_hierarchy(path) == spec

    @pytest.mark.parametrize("tree", [
        {"root": ["A"]},                                   # child without its own entry
        {"root": ["A", "B"], "A": ["C"], "B": ["C"], "C": []},  # two parents
        {"r1": ["A"], "r2": [], "A": []},                  # two roots
    ])
    def test_invalid_trees(self, tree):
        with pytest.raises(ValueError):
            HierarchySpec(name="bad", tree=tree)

    def test_folder_map_must_reference_nodes(self):
        with pytest.raises(ValueError):
            HierarchySpec(name="bad", tree={"root": ["A"], "A": []}, folder_map={"B": ["x"]})


# ================================================================== #
#  Training config                                                     #
# ================================================================== #

class TestTrainConfig:
    def test_defaults(self):
        config = load_train_config()
        assert config == TRAIN_DEFAULTS
        assert config is not TRAIN_DEFAULTS

    def test_file_merges_over_defaults(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("model: {backbone: swin_s}\ndata: {max_class_size: 10}\n")
        config = load_train_config(path)
        assert config["model"]["backbone"] == "swin_s"
        assert config["model"]["expert"] == TRAIN_DEFAULTS["model"]["expert"]
        assert config["data"]["max_class_size"] == 10

    def test_unknown_keys_raise(self):
        with pytest.raises(KeyError):
            deep_merge(TRAIN_DEFAULTS, {"modle": {}})
        with pytest.raises(KeyError):
            deep_merge(TRAIN_DEFAULTS, {"model": {"backbon": "x"}})

    def test_set_dotted_parses_yaml_values(self):
        config = load_train_config()
        set_dotted(config, "data.split", "[0.8, 0.1, 0.1]")
        set_dotted(config, "train.max_epochs", "3")
        set_dotted(config, "data.exclude_log", "null")
        assert config["data"]["split"] == [0.8, 0.1, 0.1]
        assert config["train"]["max_epochs"] == 3
        assert config["data"]["exclude_log"] is None
        with pytest.raises(KeyError):
            set_dotted(config, "train.nope", "1")

    @pytest.mark.parametrize("name", ["mnr_swin_t_cnn_noise", "mnr_swin_t_cnn_baseline", "whoi_swin_t_mlp"])
    def test_example_configs_load(self, name):
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "configs" / f"{name}.yaml"
        config = load_train_config(path)
        load_hierarchy(config["hierarchy"])
