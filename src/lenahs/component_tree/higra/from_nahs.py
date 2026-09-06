"""Reconstruction of explicit Higra components from NAHS parents."""

from dataclasses import dataclass

import higra as hg
import numpy as np
from numpy.typing import NDArray

from lenahs.component_tree.types import ComponentTree, ComponentTreeKind, FlatZoneGraph


@dataclass
class HigraComponentTree:
    tree: hg.Tree
    altitudes: NDArray[np.uint8]


def _find_component_representatives(
    parents: NDArray[np.int64],
    levels: NDArray[np.uint8],
) -> NDArray[np.int64]:
    component_representatives = np.empty(parents.size, dtype=np.int64)

    for node in range(parents.size):
        representative = node
        parent = int(parents[representative])
        while parent != representative and levels[parent] == levels[representative]:
            representative = parent
            parent = int(parents[representative])
        component_representatives[node] = representative

    return component_representatives


def _component_depths(
    representatives: NDArray[np.int64],
    component_parents: dict[int, int],
) -> dict[int, int]:
    depths: dict[int, int] = {}

    for representative in representatives:
        representative = int(representative)
        path = []
        current = representative

        while current not in depths and component_parents[current] != current:
            path.append(current)
            current = component_parents[current]

        depth = depths.get(current, 0)
        depths.setdefault(current, depth)
        for component in reversed(path):
            depth += 1
            depths[component] = depth

    return depths


def _build_leaf_graph(graph: FlatZoneGraph) -> hg.UndirectedGraph:
    leaf_graph = hg.UndirectedGraph(graph.levels.size)
    if graph.edge_index.size:
        leaf_graph.add_edges(graph.edge_index[:, 0], graph.edge_index[:, 1])
    return leaf_graph


def nahs_to_higra(
    graph: FlatZoneGraph,
    component_tree: ComponentTree,
) -> HigraComponentTree:
    """Reconstruct a Higra tree whose leaves use the NAHS RAG node IDs."""

    ComponentTreeKind(component_tree.kind)
    parents = np.asarray(component_tree.parents, dtype=np.int64)
    levels = np.asarray(graph.levels, dtype=np.uint8)

    if parents.shape != levels.shape:
        raise ValueError("The NAHS tree must contain one parent per RAG node.")

    # Equal-level parent links recover the flat zones introduced together
    component_for_node = _find_component_representatives(parents, levels)
    representatives = np.unique(component_for_node)

    # A group's representative carries its link to the enclosing component
    component_parents = {
        int(representative): int(component_for_node[parents[representative]])
        for representative in representatives
    }
    depths = _component_depths(representatives, component_parents)

    # Deeper components come first so every internal parent has a greater ID
    ordered_representatives = sorted(
        (int(representative) for representative in representatives),
        key=lambda representative: (
            -depths[representative],
            int(graph.anchors[representative]),
        ),
    )

    leaf_count = parents.size
    component_ids = {
        representative: leaf_count + index
        for index, representative in enumerate(ordered_representatives)
    }
    higra_parents = np.empty(leaf_count + representatives.size, dtype=np.int64)

    for leaf in range(leaf_count):
        representative = int(component_for_node[leaf])
        higra_parents[leaf] = component_ids[representative]

    for representative in ordered_representatives:
        component = component_ids[representative]
        parent_representative = component_parents[representative]
        higra_parents[component] = component_ids[parent_representative]

    altitudes = np.empty(higra_parents.size, dtype=np.uint8)
    altitudes[:leaf_count] = levels
    for representative in ordered_representatives:
        altitudes[component_ids[representative]] = levels[representative]

    higra_tree = hg.Tree(higra_parents)
    hg.CptHierarchy.link(higra_tree, _build_leaf_graph(graph))
    return HigraComponentTree(tree=higra_tree, altitudes=altitudes)
