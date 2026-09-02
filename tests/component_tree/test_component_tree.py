from itertools import product

import numpy as np
import pytest

from lenahs.component_tree import (
    ComponentTreeKind,
    FlatZoneGraph,
    build_reference_component_tree,
    build_reference_rag,
)

from .cases import CASES


def _components(nodes: set[int], edges: set[tuple[int, int]]) -> set[frozenset[int]]:
    remaining = set(nodes)
    components = set()

    while remaining:
        start = remaining.pop()
        component = {start}
        pending = [start]

        while pending:
            node = pending.pop()
            for left, right in edges:
                if left == node and right in remaining:
                    remaining.remove(right)
                    component.add(right)
                    pending.append(right)
                elif right == node and left in remaining:
                    remaining.remove(left)
                    component.add(left)
                    pending.append(left)

        components.add(frozenset(component))

    return components


def _assert_tree_matches_threshold_components(graph, tree):
    node_count = graph.levels.size
    assert tree.parents.shape == (node_count,)
    assert np.all((0 <= tree.parents) & (tree.parents < node_count))

    roots = np.flatnonzero(tree.parents == np.arange(node_count))
    assert roots.size == 1
    root = int(roots[0])

    for start in range(node_count):
        seen = set()
        node = start
        while node != root:
            assert node not in seen
            seen.add(node)
            node = int(tree.parents[node])

    rag_edges = {tuple(map(int, edge)) for edge in graph.edge_index}
    tree_edges = {
        tuple(sorted((node, int(parent))))
        for node, parent in enumerate(tree.parents)
        if node != parent
    }

    for level in np.unique(graph.levels):
        if tree.kind == ComponentTreeKind.MAX_TREE:
            active = set(np.flatnonzero(graph.levels >= level).tolist())
        else:
            active = set(np.flatnonzero(graph.levels <= level).tolist())

        expected = _components(active, rag_edges)
        actual = _components(active, tree_edges)
        assert actual == expected

        for component in expected:
            candidates = [node for node in component if graph.levels[node] == level]
            if not candidates:
                continue

            representative = min(
                candidates,
                key=lambda node: (
                    -int(graph.pixel_counts[node]),
                    int(graph.anchors[node]),
                ),
            )
            for node in candidates:
                if node != representative:
                    assert tree.parents[node] == representative

            parent = int(tree.parents[representative])
            if representative != root:
                if tree.kind == ComponentTreeKind.MAX_TREE:
                    assert graph.levels[parent] < level
                else:
                    assert graph.levels[parent] > level


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize(
    ("kind", "expected_key"),
    [
        (ComponentTreeKind.MAX_TREE, "max_parents"),
        (ComponentTreeKind.MIN_TREE, "min_parents"),
    ],
)
def test_named_parent_arrays(case_name, kind, expected_key):
    case = CASES[case_name]
    graph = build_reference_rag(case["image"])

    tree = build_reference_component_tree(graph, kind)

    assert tree.kind == kind
    np.testing.assert_array_equal(tree.parents, case[expected_key])
    _assert_tree_matches_threshold_components(graph, tree)


def _renumber_graph(graph, new_to_old):
    old_to_new = np.empty_like(new_to_old)
    old_to_new[new_to_old] = np.arange(new_to_old.size)
    return FlatZoneGraph(
        label_map=old_to_new[graph.label_map],
        edge_index=old_to_new[graph.edge_index],
        levels=graph.levels[new_to_old],
        pixel_counts=graph.pixel_counts[new_to_old],
        anchors=graph.anchors[new_to_old],
    )


def _parents_by_anchor(graph, tree):
    return {
        int(graph.anchors[node]): int(graph.anchors[parent])
        for node, parent in enumerate(tree.parents)
    }


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_rag_node_renumbering_preserves_tree_by_anchor(case_name, kind):
    graph = build_reference_rag(CASES[case_name]["image"])
    original = build_reference_component_tree(graph, kind)
    new_to_old = np.arange(graph.levels.size - 1, -1, -1)
    renumbered_graph = _renumber_graph(graph, new_to_old)

    renumbered = build_reference_component_tree(renumbered_graph, kind)

    assert _parents_by_anchor(renumbered_graph, renumbered) == _parents_by_anchor(
        graph, original
    )


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_increasing_level_remapping_preserves_topology(case_name, kind):
    graph = build_reference_rag(CASES[case_name]["image"])
    remapped_levels = np.asarray(
        graph.levels.astype(np.uint16) * 2 + 1,
        dtype=np.uint8,
    )
    remapped_graph = FlatZoneGraph(
        graph.label_map,
        graph.edge_index,
        remapped_levels,
        graph.pixel_counts,
        graph.anchors,
    )

    original = build_reference_component_tree(graph, kind)
    remapped = build_reference_component_tree(remapped_graph, kind)

    np.testing.assert_array_equal(remapped.parents, original.parents)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize(
    ("original_kind", "remapped_kind"),
    [
        (ComponentTreeKind.MAX_TREE, ComponentTreeKind.MIN_TREE),
        (ComponentTreeKind.MIN_TREE, ComponentTreeKind.MAX_TREE),
    ],
)
def test_decreasing_level_remapping_exchanges_max_and_min(
    case_name, original_kind, remapped_kind
):
    graph = build_reference_rag(CASES[case_name]["image"])
    remapped_graph = FlatZoneGraph(
        graph.label_map,
        graph.edge_index,
        np.asarray(255 - graph.levels, dtype=np.uint8),
        graph.pixel_counts,
        graph.anchors,
    )

    original = build_reference_component_tree(graph, original_kind)
    remapped = build_reference_component_tree(remapped_graph, remapped_kind)

    np.testing.assert_array_equal(remapped.parents, original.parents)


def test_all_two_by_three_images_over_three_levels():
    for values in product(range(3), repeat=6):
        image = np.asarray(values, dtype=np.uint8).reshape(2, 3)
        graph = build_reference_rag(image)

        for kind in ComponentTreeKind:
            tree = build_reference_component_tree(graph, kind)
            _assert_tree_matches_threshold_components(graph, tree)
