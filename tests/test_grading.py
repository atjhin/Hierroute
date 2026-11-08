import math

import pandas as pd
import pytest

from hierroute.grading import (apply_confidence_threshold, classify_id_outcome, hierarchy_level_metrics_table,
                               id_outcome_table, leaf_metrics_table, loosened_leaf_metrics_table, outcome_colors,
                               outcome_keys, outcome_label)
from hierroute.hierarchy import Hierarchy

TREE = {
    "root": ["Zoop", "Junk"],
    "Zoop": ["Copepoda", "Rotifer", "Zoop_Unknown"],
    "Junk": [],
    "Copepoda": ["Cyclopoid", "Calanoid", "Copepoda_Unknown"],
    "Rotifer": [],
    "Zoop_Unknown": [],
    "Cyclopoid": [],
    "Calanoid": [],
    "Copepoda_Unknown": [],
}


@pytest.fixture
def h():
    hierarchy, ids = Hierarchy.from_adjacency(TREE)
    return hierarchy, ids


class TestOutcomeKeys:
    def test_three_levels(self):
        assert outcome_keys(3, gated=False) == ("correct_all", "correct_incomplete", "incorrect_l3",
                                                "incorrect_l2", "incorrect_all")
        assert outcome_keys(3, gated=True)[-1] == "abstained"

    def test_labels_and_colors(self):
        keys = outcome_keys(3, gated=True)
        assert [outcome_label(k) for k in keys] == ["Correct_All", "Correct_Incomplete", "Incorrect_L3",
                                                    "Incorrect_L2", "Incorrect_All", "Abstained"]
        assert outcome_colors(keys) == ("#4C72B0", "#55A868", "#E7B84B", "#DD8452", "#C44E52", "#8172B2")

    def test_deeper_hierarchy(self):
        assert "incorrect_l5" in outcome_keys(5, gated=False)


class TestClassifyIdOutcome:
    def test_exact(self, h):
        hierarchy, ids = h
        assert classify_id_outcome(ids["Cyclopoid"], ids["Cyclopoid"], hierarchy) == "correct_all"

    def test_coarse_truth_subtree(self, h):
        hierarchy, ids = h
        assert classify_id_outcome(ids["Calanoid"], ids["Copepoda"], hierarchy) == "correct_all"

    def test_ancestor_is_incomplete(self, h):
        hierarchy, ids = h
        assert classify_id_outcome(ids["Copepoda"], ids["Cyclopoid"], hierarchy) == "correct_incomplete"

    def test_unknown_sibling_is_incomplete(self, h):
        hierarchy, ids = h
        assert classify_id_outcome(ids["Copepoda_Unknown"], ids["Cyclopoid"], hierarchy) == "correct_incomplete"
        assert classify_id_outcome(ids["Zoop_Unknown"], ids["Cyclopoid"], hierarchy) == "correct_incomplete"

    def test_unknown_credit_is_one_way(self, h):
        hierarchy, ids = h
        assert classify_id_outcome(ids["Cyclopoid"], ids["Copepoda_Unknown"], hierarchy) == "incorrect_l3"

    def test_wrong_levels(self, h):
        hierarchy, ids = h
        assert classify_id_outcome(ids["Calanoid"], ids["Cyclopoid"], hierarchy) == "incorrect_l3"
        assert classify_id_outcome(ids["Rotifer"], ids["Cyclopoid"], hierarchy) == "incorrect_l2"
        assert classify_id_outcome(ids["Junk"], ids["Cyclopoid"], hierarchy) == "incorrect_all"

    def test_abstained(self, h):
        hierarchy, ids = h
        assert classify_id_outcome(None, ids["Cyclopoid"], hierarchy) == "abstained"


def test_apply_confidence_threshold(h):
    _, ids = h
    path = [ids["root"], ids["Zoop"], ids["Copepoda"], ids["Cyclopoid"]]
    assert apply_confidence_threshold(path, [0.99, 0.99, 0.99], (0.9, 0.9, 0.9)) == ids["Cyclopoid"]
    assert apply_confidence_threshold(path, [0.99, 0.99, 0.5], (0.9, 0.9, 0.9)) == ids["Copepoda"]
    assert apply_confidence_threshold(path, [0.5, 0.5, 0.5], (0.9, 0.9, 0.9)) is None


class TestTables:
    def test_strict_leaf_metrics(self, h):
        hierarchy, _ = h
        true = ["Cyclopoid", "Cyclopoid", "Calanoid", "Junk"]
        pred = ["Cyclopoid", "Calanoid", "Calanoid", "Junk"]
        table = leaf_metrics_table(true, pred, [t == p for t, p in zip(true, pred)], hierarchy)
        assert table.loc["Cyclopoid", "Accuracy"] == 0.5
        assert table.loc["Total", "N"] == 4
        assert table.loc["Total", "Accuracy"] == 0.75

    def test_loosened_credits_unknown_sibling(self, h):
        hierarchy, _ = h
        table = loosened_leaf_metrics_table(["Cyclopoid", "Cyclopoid"], ["Copepoda_Unknown", "Calanoid"], hierarchy)
        assert table.loc["Cyclopoid", "Accuracy"] == 0.5

    def test_loosened_exclude(self, h):
        hierarchy, _ = h
        table = loosened_leaf_metrics_table(["Junk", "Cyclopoid"], ["Junk", "Cyclopoid"], hierarchy,
                                            exclude={"Junk"})
        assert "Junk" not in table.index

    def test_hierarchy_levels(self, h):
        hierarchy, _ = h
        table = hierarchy_level_metrics_table(["Cyclopoid", "Junk"], ["Calanoid", "Junk"], hierarchy)
        assert set(table["Level"]) == {1, 2, 3}
        level1 = table[table["Level"] == 1]
        assert level1.loc["Total", "Accuracy"] == 1.0
        level3 = table[table["Level"] == 3]
        assert level3.loc["Cyclopoid", "Accuracy"] == 0.0

    def test_id_outcome_table_rows_sum_to_one(self, h):
        hierarchy, ids = h
        paths = [[ids["root"], ids["Zoop"], ids["Copepoda"], ids["Cyclopoid"]],
                 [ids["root"], ids["Junk"]]]
        confs = [[1.0, 1.0, 1.0], [1.0]]
        keys = outcome_keys(3, gated=False)
        table, props = id_outcome_table(paths, confs, [ids["Cyclopoid"], ids["Cyclopoid"]], hierarchy, keys)
        assert table.loc["Cyclopoid", "N"] == 2
        assert math.isclose(table.loc["Cyclopoid", list(keys)].sum(), 1.0)
        assert props["Cyclopoid"]["incorrect_all"] == 0.5
