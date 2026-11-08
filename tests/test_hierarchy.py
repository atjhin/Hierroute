import pytest
from hierroute.hierarchy import Node, Hierarchy


# ================================================================== #
#  Node                                                                #
# ================================================================== #

class TestNode:
    def test_node_creation(self):
        node = Node(0, "root")
        assert node.node_id == 0
        assert node.name == "root"
        assert node.parent is None
        assert node.children == []
        assert node.depth is None

    def test_node_with_parent(self):
        node = Node(1, "child", parent=0)
        assert node.parent == 0

    def test_add_child(self):
        node = Node(0, "root")
        node.add_child(1)
        node.add_child(2)
        assert node.children == [1, 2]

    def test_add_child_no_duplicates(self):
        node = Node(0, "root")
        node.add_child(1)
        node.add_child(1)
        assert node.children == [1]


# ================================================================== #
#  Hierarchy — small tree                                              #
# ================================================================== #

class TestHierarchy:
    def test_add_nodes_and_root(self, small_hierarchy):
        h, labels = small_hierarchy
        assert h.root == labels["root"]

    def test_len(self, small_hierarchy):
        h, _ = small_hierarchy
        assert len(h) == 5  # root, A, B, A1, A2

    def test_children(self, small_hierarchy):
        h, labels = small_hierarchy
        root_children = h.children(labels["root"])
        assert labels["A"] in root_children
        assert labels["B"] in root_children

    def test_parent(self, small_hierarchy):
        h, labels = small_hierarchy
        assert h.parent(labels["A"]) == labels["root"]
        assert h.parent(labels["root"]) is None

    def test_is_leaf(self, small_hierarchy):
        h, labels = small_hierarchy
        assert not h.is_leaf(labels["root"])
        assert not h.is_leaf(labels["A"])
        assert h.is_leaf(labels["B"])
        assert h.is_leaf(labels["A1"])
        assert h.is_leaf(labels["A2"])

    def test_depth(self, small_hierarchy):
        h, labels = small_hierarchy
        assert h.depth(labels["root"]) == 0
        assert h.depth(labels["A"]) == 1
        assert h.depth(labels["B"]) == 1
        assert h.depth(labels["A1"]) == 2
        assert h.depth(labels["A2"]) == 2

    def test_get_path_to_root(self, small_hierarchy):
        h, labels = small_hierarchy
        path = h.get_path_to_root(labels["A1"])
        assert path[0] == labels["root"]
        assert path[-1] == labels["A1"]
        assert len(path) == 3

    def test_get_path_to_root_for_root(self, small_hierarchy):
        h, labels = small_hierarchy
        path = h.get_path_to_root(labels["root"])
        assert path == [labels["root"]]

    def test_descendants(self, small_hierarchy):
        h, labels = small_hierarchy
        desc = h.descendants(labels["root"])
        assert len(desc) == 4  # A, B, A1, A2
        assert labels["root"] not in desc

    def test_subtree_leaves(self, small_hierarchy):
        h, labels = small_hierarchy
        leaves = h.subtree_leaves(labels["A"])
        assert set(leaves) == {labels["A1"], labels["A2"]}

    def test_subtree_leaves_of_leaf(self, small_hierarchy):
        h, labels = small_hierarchy
        leaves = h.subtree_leaves(labels["B"])
        assert leaves == [labels["B"]]

    def test_get_leaf_index(self, small_hierarchy):
        h, labels = small_hierarchy
        leaf_idx = h.get_leaf_index()
        assert len(leaf_idx) == len(h)
        for node_id, val in enumerate(leaf_idx):
            if h.is_leaf(node_id):
                assert val == 1
            else:
                assert val == 0

    def test_invalid_node_path(self, small_hierarchy):
        h, _ = small_hierarchy
        with pytest.raises(ValueError):
            h.get_path_to_root(999)


# ================================================================== #
#  Hierarchy — full MNR graph                                          #
# ================================================================== #

class TestMNRHierarchy:
    def test_node_count(self, hierarchy_and_labels):
        h, _ = hierarchy_and_labels
        assert len(h) == 19

    def test_leaf_count(self, hierarchy_and_labels):
        h, _ = hierarchy_and_labels
        leaf_index = h.get_leaf_index()
        assert sum(leaf_index) == 13

    def test_max_depth(self, hierarchy_and_labels):
        h, _ = hierarchy_and_labels
        max_d = max(h.depth(nid) for nid in h.nodes)
        assert max_d == 3

    def test_all_paths_start_at_root(self, hierarchy_and_labels):
        h, _ = hierarchy_and_labels
        for nid in h.nodes:
            path = h.get_path_to_root(nid)
            assert path[0] == h.root


class TestFromAdjacency:
    def test_matches_manual_build(self, hierarchy_and_labels, mnr_adjacency_graph):
        expected, expected_ids = hierarchy_and_labels
        built, ids = Hierarchy.from_adjacency(mnr_adjacency_graph)
        assert ids == expected_ids
        assert {k: (n.name, n.parent, n.children) for k, n in built.nodes.items()} == \
            {k: (n.name, n.parent, n.children) for k, n in expected.nodes.items()}

    def test_max_depth_and_name_to_id(self, small_adjacency_graph):
        h, ids = Hierarchy.from_adjacency(small_adjacency_graph)
        assert h.max_depth == 2
        assert h.name_to_id() == ids
