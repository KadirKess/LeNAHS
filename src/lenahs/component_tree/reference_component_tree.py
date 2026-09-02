"""Readable component-tree construction with an immersion-style union-find sweep.

The sweep follows Carlinet and Geraud, "A Comparative Review of Component Tree
Computation Algorithms," IEEE TIP, 2014. Equal-level RAG nodes are processed
together so representative choice follows the NAHS pixel-count and anchor rule
instead of incidental processing order.
"""

import numpy as np

from lenahs.component_tree.canonical import choose_representative
from lenahs.component_tree.types import ComponentTree, ComponentTreeKind, FlatZoneGraph


def _find_root(disjoint_set_parents: np.ndarray, node: int) -> int:
    root = node
    while disjoint_set_parents[root] != root:
        root = int(disjoint_set_parents[root])

    while disjoint_set_parents[node] != node:
        parent = int(disjoint_set_parents[node])
        disjoint_set_parents[node] = root
        node = parent

    return root


def _join_components(
    disjoint_set_parents: np.ndarray,
    ranks: np.ndarray,
    left: int,
    right: int,
) -> int:
    left_root = _find_root(disjoint_set_parents, left)
    right_root = _find_root(disjoint_set_parents, right)
    if left_root == right_root:
        return left_root

    if ranks[left_root] < ranks[right_root]:
        left_root, right_root = right_root, left_root

    disjoint_set_parents[right_root] = left_root
    if ranks[left_root] == ranks[right_root]:
        ranks[left_root] += 1

    return left_root


def _build_adjacency(graph: FlatZoneGraph) -> list[list[int]]:
    adjacency = [[] for _ in range(graph.levels.size)]
    for left, right in graph.edge_index:
        left = int(left)
        right = int(right)
        adjacency[left].append(right)
        adjacency[right].append(left)
    return adjacency


def build_reference_component_tree(
    graph: FlatZoneGraph,
    kind: ComponentTreeKind,
) -> ComponentTree:
    """Build a max-tree or min-tree directly from a flat-zone RAG."""

    kind = ComponentTreeKind(kind)
    node_count = graph.levels.size
    tree_parents = np.full(node_count, -1, dtype=np.int64)
    disjoint_set_parents = np.arange(node_count, dtype=np.int64)
    ranks = np.zeros(node_count, dtype=np.int64)
    component_representatives = np.full(node_count, -1, dtype=np.int64)
    active = np.zeros(node_count, dtype=bool)
    adjacency = _build_adjacency(graph)

    levels = np.unique(graph.levels)
    if kind == ComponentTreeKind.MAX_TREE:
        levels = levels[::-1]

    for level in levels:
        current_nodes = np.flatnonzero(graph.levels == level)
        previous_roots = set()
        for node in current_nodes:
            for neighbor in adjacency[int(node)]:
                if active[neighbor]:
                    previous_roots.add(_find_root(disjoint_set_parents, neighbor))
        previous_representatives = {
            root: int(component_representatives[root]) for root in previous_roots
        }

        for node in current_nodes:
            node = int(node)
            active[node] = True
            component_representatives[node] = node

        for node in current_nodes:
            node = int(node)
            for neighbor in adjacency[node]:
                if active[neighbor]:
                    _join_components(
                        disjoint_set_parents,
                        ranks,
                        node,
                        neighbor,
                    )

        current_nodes_by_root: dict[int, list[int]] = {}
        for node in current_nodes:
            node = int(node)
            root = _find_root(disjoint_set_parents, node)
            current_nodes_by_root.setdefault(root, []).append(node)

        absorbed_representatives: dict[int, list[int]] = {}
        for old_root, representative in previous_representatives.items():
            new_root = _find_root(disjoint_set_parents, old_root)
            if new_root in current_nodes_by_root:
                absorbed_representatives.setdefault(new_root, []).append(representative)

        for root, nodes in current_nodes_by_root.items():
            candidates = np.asarray(nodes, dtype=np.int64)
            representative = choose_representative(
                candidates,
                graph.pixel_counts,
                graph.anchors,
            )

            tree_parents[candidates] = representative
            for child in absorbed_representatives.get(root, []):
                tree_parents[child] = representative
            component_representatives[root] = representative

    final_roots = {
        _find_root(disjoint_set_parents, int(node)) for node in np.flatnonzero(active)
    }
    for root in final_roots:
        representative = int(component_representatives[root])
        tree_parents[representative] = representative

    return ComponentTree(kind=kind, parents=tree_parents)
