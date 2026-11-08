from __future__ import annotations


class Node:
    """
    A single node in a taxonomy tree.

    Attributes:
        node_id  (int):        Unique integer identifier for this node.
        name     (str):        Human-readable class name (e.g. "Copepoda").
        parent   (int | None): node_id of the parent node; None for the root.
        children (list[int]):  Ordered list of child node_ids.
        depth    (int | None): Distance from the root (root = 0); set by
                               Hierarchy.add_node after insertion.
    """

    def __init__(self, node_id: int, name: str, parent: int | None = None) -> None:
        self.node_id:  int             = node_id
        self.name:     str             = name
        self.parent:   int | None      = parent
        self.children: list[int]       = []
        self.depth:    int | None      = None

    def add_child(self, child_id: int) -> None:
        """
        Append a child node ID to this node's children list (no duplicates).

        Args:
            child_id (int): node_id of the child to add.
        """
        if child_id not in self.children:
            self.children.append(child_id)

    def __repr__(self) -> str:
        return (
            f"Node(id={self.node_id}, name={self.name}, "
            f"parent={self.parent}, children={self.children})"
        )


class Hierarchy:
    """
    A rooted tree that represents a taxonomic classification hierarchy.

    Nodes are added top-down via add_node(); parent nodes must be inserted
    before their children. The tree is stored as a flat dict of Node objects
    keyed by integer node_id.

    Attributes:
        nodes (dict[int, Node]): Mapping from node_id to Node object.
        root  (int | None):      node_id of the root node (the node whose
                                 parent is None); set automatically on insert.

    Example::

        hier = Hierarchy()
        hier.add_node(0, "root")
        hier.add_node(1, "Zoop-yes", parent=0)
        hier.add_node(2, "Copepoda", parent=1)
    """

    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.root:  int | None      = None

    @classmethod
    def from_adjacency(cls, adjacency_graph: dict[str, list[str]]) -> tuple["Hierarchy", dict[str, int]]:
        """
        Build a Hierarchy from a parent -> children adjacency graph.

        Node IDs follow the graph's key order, so a trained checkpoint only loads
        against a graph with the same key order it was trained with.

        Args:
            adjacency_graph (dict[str, list[str]]): Every node as a key, mapped to its
                ordered children (empty list for leaves).

        Returns:
            tuple: ``(hierarchy, label_to_id)`` where ``label_to_id`` maps node name to node_id.
        """
        label_to_id = {name: i for i, name in enumerate(adjacency_graph.keys())}
        hierarchy = cls()
        all_children = {c for kids in adjacency_graph.values() for c in kids}
        root = next(n for n in adjacency_graph if n not in all_children)
        hierarchy.add_node(label_to_id[root], root)

        def dfs(parent: str) -> None:
            for child in adjacency_graph[parent]:
                hierarchy.add_node(label_to_id[child], child, label_to_id[parent])
                dfs(child)

        dfs(root)
        return hierarchy, label_to_id

    @property
    def max_depth(self) -> int:
        """Number of non-root levels (depth of the deepest node)."""
        return max(node.depth for node in self.nodes.values())

    def name_to_id(self) -> dict[str, int]:
        """Mapping from node name to node_id."""
        return {node.name: node_id for node_id, node in self.nodes.items()}

    # ------------------------------------------------------------------ #
    # Built-ins                                                            #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Return the total number of nodes in the hierarchy."""
        return len(self.nodes)

    # ------------------------------------------------------------------ #
    # Mutation                                                             #
    # ------------------------------------------------------------------ #

    def add_node(self, node_id: int, name: str, parent: int | None = None) -> None:
        """
        Insert a new node into the hierarchy.

        The parent node must already exist unless this is the root (parent=None).
        Depth is computed automatically from the parent chain.

        Args:
            node_id (int):        Unique identifier for the new node.
            name    (str):        Human-readable class label.
            parent  (int | None): node_id of the parent; None makes this the root.
        """
        self.nodes[node_id] = Node(node_id, name, parent)
        self.nodes[node_id].depth = self.depth(node_id)
        if parent is not None:
            self.nodes[parent].add_child(node_id)
        else:
            self.root = node_id

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    def children(self, node_id: int) -> list[int]:
        """
        Return the direct children of a node.

        Args:
            node_id (int): Target node.

        Returns:
            list[int]: Ordered list of child node_ids (empty for leaves).
        """
        return self.nodes[node_id].children

    def parent(self, node_id: int) -> int | None:
        """
        Return the parent node_id of a node.

        Args:
            node_id (int): Target node.

        Returns:
            int | None: Parent node_id, or None if node_id is the root.
        """
        return self.nodes[node_id].parent

    def is_leaf(self, node_id: int) -> bool:
        """
        Check whether a node is a leaf (has no children).

        Args:
            node_id (int): Target node.

        Returns:
            bool: True if the node has no children, False otherwise.
        """
        return len(self.nodes[node_id].children) == 0

    def depth(self, node_id: int) -> int:
        """
        Compute the depth of a node (number of edges from root).

        Root has depth 0; its children have depth 1, and so on.

        Args:
            node_id (int): Target node.

        Returns:
            int: Depth of the node.
        """
        d:   int = 0
        cur: int = node_id
        while self.nodes[cur].parent is not None:
            d  += 1
            cur = self.nodes[cur].parent
        return d

    # ------------------------------------------------------------------ #
    # Path & traversal                                                     #
    # ------------------------------------------------------------------ #

    def get_path_to_root(self, leaf_id: int) -> list[int]:
        """
        Return the path from the root down to the given node.

        Args:
            leaf_id (int): node_id of the target node (need not be a leaf).

        Returns:
            list[int]: Node IDs ordered from root to leaf_id,
                       e.g. [root_id, ..., leaf_id].

        Raises:
            ValueError: If leaf_id is not present in the hierarchy.
        """
        if leaf_id not in self.nodes:
            raise ValueError(f"Node {leaf_id} not found in hierarchy.")

        path:       list[int] = []
        current_id: int | None = leaf_id

        while current_id is not None:
            path.append(current_id)
            current_id = self.nodes[current_id].parent

        return list(reversed(path))

    def descendants(self, node_id: int) -> list[int]:
        """
        Return all descendant node IDs via depth-first search (excludes node_id itself).

        Args:
            node_id (int): Root of the subtree to traverse.

        Returns:
            list[int]: All descendant node IDs in DFS order.
        """
        result: list[int] = []

        def dfs(n: int) -> None:
            for child in self.nodes[n].children:
                result.append(child)
                dfs(child)

        dfs(node_id)
        return result

    def subtree_leaves(self, node_id: int) -> list[int]:
        """
        Return all leaf node IDs in the subtree rooted at node_id.

        Args:
            node_id (int): Root of the subtree.

        Returns:
            list[int]: Leaf node IDs in DFS order. If node_id is itself a
                       leaf, returns [node_id].
        """
        leaves: list[int] = []

        def dfs(n: int) -> None:
            if self.is_leaf(n):
                leaves.append(n)
            for child in self.nodes[n].children:
                dfs(child)

        dfs(node_id)
        return leaves

    def get_leaf_index(self) -> list[int]:
        """
        Build a binary mask indicating which nodes are leaves.

        Nodes are ordered by ascending node_id.

        Returns:
            list[int]: A list of length len(self.nodes) where position i
                       is 1 if node i is a leaf, 0 if it is an internal node.

        Example:
            If nodes 0-4 exist and nodes 3, 4 are leaves:
            >>> hier.get_leaf_index()
            [0, 0, 0, 1, 1]
        """
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.node_id)
        return [1 if self.is_leaf(n.node_id) else 0 for n in sorted_nodes]
