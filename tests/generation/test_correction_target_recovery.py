from itertools import product

import numpy as np
import pytest

from lenahs.component_tree.reference import (
    build_reference_component_tree,
    build_reference_rag,
)
from lenahs.component_tree.types import ComponentTree, ComponentTreeKind, FlatZoneGraph
from lenahs.generation import derive_correction_targets
from lenahs.generation.samples import _build_production_tree
from lenahs.generation.targets import CorrectionAction


def _checked_graph(image, kind):
    graph, tree = _build_production_tree(image, kind)
    reference = build_reference_rag(image)
    reference_tree = build_reference_component_tree(reference, kind)
    for field in ("label_map", "edge_index", "levels", "pixel_counts", "anchors"):
        np.testing.assert_array_equal(getattr(graph, field), getattr(reference, field))
    np.testing.assert_array_equal(tree.parents, reference_tree.parents)
    return graph, tree


def _threshold_depth(image, kind):
    components = set()
    height, width = image.shape
    for level in np.unique(image):
        active = (
            image >= level if kind == ComponentTreeKind.MAX_TREE else image <= level
        )
        remaining = set(zip(*np.nonzero(active)))
        while remaining:
            start = remaining.pop()
            component = {start}
            pending = [start]
            while pending:
                row, column = pending.pop()
                for neighbor in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                ):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        pending.append(neighbor)
            components.add(frozenset(row * width + column for row, column in component))

    # Distinct threshold components count strict depth, without NAHS parent links.
    depths = np.full(height * width, -1, dtype=np.int64)
    for component in components:
        for pixel in component:
            depths[pixel] += 1
    return depths.reshape(image.shape)


def _inspect_targets(graph, targets):
    parents = targets.corrective_parents
    actions = targets.actions
    node_count = graph.levels.size
    assert parents.shape == actions.shape == (node_count,)
    assert np.all((parents >= 0) & (parents < node_count))
    assert set(actions) <= set(CorrectionAction)
    roots = [node for node in range(node_count) if parents[node] == node]
    assert len(roots) == 1
    assert actions[roots[0]] == CorrectionAction.KEEP_ZERO

    terminals = np.empty(node_count, dtype=np.int64)
    node_depths = np.zeros(node_count, dtype=np.int64)
    for node in range(node_count):
        current = node
        visited = set()
        while parents[current] != current:
            assert current not in visited
            visited.add(current)
            node_depths[node] += actions[current] == CorrectionAction.KEEP_ONE
            current = int(parents[current])
        assert current == roots[0]

        current = node
        while actions[current] == CorrectionAction.MERGE:
            current = int(parents[current])
        terminals[node] = current

    return terminals, node_depths[graph.label_map]


def _assert_exact_recovery(clean, clean_graph, clean_tree, observed_graph, targets):
    terminals, pixel_depth = _inspect_targets(observed_graph, targets)
    clean_groups = {
        frozenset(np.flatnonzero(clean_graph.label_map.ravel() == zone)): zone
        for zone in range(clean_graph.levels.size)
    }
    group_labels = terminals[observed_graph.label_map]
    group_roles = {}
    for representative in np.unique(terminals):
        members = frozenset(np.flatnonzero(group_labels.ravel() == representative))
        assert members in clean_groups
        group_roles[int(representative)] = clean_groups[members]
        observed_members = np.flatnonzero(terminals == representative)
        assert observed_graph.anchors[representative] == min(
            observed_graph.anchors[observed_members]
        )
    assert set(group_roles.values()) == set(range(clean_graph.levels.size))

    for representative, clean_zone in group_roles.items():
        parent_group = terminals[targets.corrective_parents[representative]]
        assert group_roles[int(parent_group)] == clean_tree.parents[clean_zone]
        clean_parent = clean_tree.parents[clean_zone]
        increment = clean_graph.levels[clean_zone] != clean_graph.levels[clean_parent]
        assert (
            targets.actions[representative] == CorrectionAction.KEEP_ONE
        ) == increment

    contracted_edges = set()
    for left, right in observed_graph.edge_index:
        first = group_roles[int(terminals[left])]
        second = group_roles[int(terminals[right])]
        if first != second:
            contracted_edges.add(tuple(sorted((first, second))))
    assert contracted_edges == {tuple(sorted(edge)) for edge in clean_graph.edge_index}
    np.testing.assert_array_equal(pixel_depth, _threshold_depth(clean, clean_tree.kind))


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_exhaustive_binary_clean_ternary_observed_recovery(kind):
    clean_cases = []
    for values in product((10, 20), repeat=4):
        image = np.array(values, dtype=np.uint8).reshape(2, 2)
        graph, tree = _checked_graph(image, kind)
        clean_cases.append((image, graph, tree))
    observed_cases = []
    for values in product((0, 10, 20), repeat=4):
        image = np.array(values, dtype=np.uint8).reshape(2, 2)
        graph, _ = _checked_graph(image, kind)
        observed_cases.append(graph)

    refining_count = 0
    mixed_count = 0
    for clean, clean_graph, clean_tree in clean_cases:
        for observed_graph in observed_cases:
            targets, diagnostics = derive_correction_targets(
                clean_graph, clean_tree, observed_graph
            )
            pure = all(
                len(set(clean_graph.label_map[observed_graph.label_map == zone])) == 1
                for zone in range(observed_graph.levels.size)
            )
            if pure:
                _assert_exact_recovery(
                    clean, clean_graph, clean_tree, observed_graph, targets
                )
                assert not diagnostics.missing_clean_zones.any()
                assert not diagnostics.clean_root_missing
                assert diagnostics.artificial_root_attachment_count == 0
                refining_count += 1
            else:
                _inspect_targets(observed_graph, targets)
                mixed_count += 1

    assert refining_count == 630
    assert mixed_count == 666


def _permute_graph(graph, permutation):
    original_ids = np.argsort(permutation)
    return FlatZoneGraph(
        permutation[graph.label_map],
        permutation[graph.edge_index],
        graph.levels[original_ids],
        graph.pixel_counts[original_ids],
        graph.anchors[original_ids],
    )


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_random_refining_recovery_with_permuted_ids(kind):
    rng = np.random.default_rng(20261006)
    for _ in range(32):
        clean = rng.choice(np.array([10, 40, 80], dtype=np.uint8), size=(6, 7))
        # Disjoint intensity ranges prevent observed zones from crossing clean roles.
        observed = clean + rng.integers(0, 6, size=clean.shape, dtype=np.uint8)
        clean_graph, clean_tree = _checked_graph(clean, kind)
        observed_graph, _ = _checked_graph(observed, kind)
        clean_permutation = rng.permutation(clean_graph.levels.size)
        observed_permutation = rng.permutation(observed_graph.levels.size)
        clean_graph = _permute_graph(clean_graph, clean_permutation)
        observed_graph = _permute_graph(observed_graph, observed_permutation)
        clean_tree = ComponentTree(
            kind,
            clean_permutation[clean_tree.parents[np.argsort(clean_permutation)]],
        )
        targets, _ = derive_correction_targets(clean_graph, clean_tree, observed_graph)
        _assert_exact_recovery(clean, clean_graph, clean_tree, observed_graph, targets)


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
@pytest.mark.parametrize(
    "clean, observed, parents, actions, depth",
    [
        (
            [[10, 10], [10, 20]],
            [[0, 0], [20, 10]],
            [0, 0, 0],
            [1, 0, 2],
            [[0, 0], [0, 1]],
        ),
        (
            [[10, 10, 10], [10, 20, 30]],
            [[0, 0, 0], [10, 20, 10]],
            [0, 0, 0, 2],
            [1, 0, 2, 2],
            [[0, 0, 0], [0, 1, 2]],
        ),
        (
            [[10, 20, 30]],
            [[11, 21, 31]],
            [0, 0, 1],
            [1, 2, 2],
            [[0, 1, 2]],
        ),
        ([[10]], [[200]], [0], [1], [[0]]),
        (
            [[10, 10, 20, 10, 10]],
            [[11, 15, 15, 15, 12]],
            [0, 0, 0],
            [1, 0, 1],
            [[0, 0, 0, 0, 0]],
        ),
        (
            [[10, 10, 10], [20, 20, 20]],
            [[11, 15, 12], [21, 15, 22]],
            [0, 0, 0, 0, 3],
            [1, 0, 0, 2, 0],
            [[0, 0, 0], [1, 0, 1]],
        ),
    ],
    ids=[
        "quotient-cycle",
        "several-displaced-children",
        "uniform-shift",
        "changed-single-pixel",
        "mixed-bridge",
        "disconnected-output-group",
    ],
)
def test_named_two_dimensional_recovery(kind, clean, observed, parents, actions, depth):
    clean = np.array(clean, dtype=np.uint8)
    observed = np.array(observed, dtype=np.uint8)
    if kind == ComponentTreeKind.MIN_TREE:
        clean = 255 - clean
        observed = 255 - observed
    clean_graph, clean_tree = _checked_graph(clean, kind)
    observed_graph, _ = _checked_graph(observed, kind)
    targets, _ = derive_correction_targets(clean_graph, clean_tree, observed_graph)
    np.testing.assert_array_equal(targets.corrective_parents, parents)
    np.testing.assert_array_equal(targets.actions, actions)
    _, pixel_depth = _inspect_targets(observed_graph, targets)
    np.testing.assert_array_equal(pixel_depth, depth)
