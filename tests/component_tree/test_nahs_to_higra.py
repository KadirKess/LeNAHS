import higra as hg
import numpy as np
import pytest

from lenahs.component_tree import ComponentTree, ComponentTreeKind
from lenahs.component_tree.higra import (
    HigraRagContext,
    build_higra_rag,
    higra_to_nahs,
    nahs_to_higra,
)

from .cases import CASES


def _build_higra_tree(graph, context, kind):
    higra_levels = graph.levels[context.higra_to_nahs]
    if kind == ComponentTreeKind.MAX_TREE:
        return hg.component_tree_max_tree(context.rag, higra_levels)
    return hg.component_tree_min_tree(context.rag, higra_levels)


def _component_structure(higra_tree, altitudes, leaf_to_nahs):
    parents = np.asarray(higra_tree.parents(), dtype=np.int64)
    leaf_count = leaf_to_nahs.size
    introduced_leaves = {}

    for leaf in range(leaf_count):
        component = int(parents[leaf])
        introduced_leaves.setdefault(component, set()).add(int(leaf_to_nahs[leaf]))

    structure = {}
    for component in range(leaf_count, parents.size):
        members = frozenset(introduced_leaves[component])
        parent = int(parents[component])
        parent_members = (
            None if parent == component else frozenset(introduced_leaves[parent])
        )
        structure[members] = (int(altitudes[component]), parent_members)

    return structure


def _case_component_tree(case, kind):
    key = "max_parents" if kind == ComponentTreeKind.MAX_TREE else "min_parents"
    return ComponentTree(kind=kind, parents=case[key])


@pytest.mark.parametrize(
    ("case_name", "kind", "expected"),
    [
        (
            "nested_peak",
            ComponentTreeKind.MAX_TREE,
            {
                frozenset({2}): (2, frozenset({1})),
                frozenset({1}): (1, frozenset({0})),
                frozenset({0}): (0, None),
            },
        ),
        (
            "nested_peak",
            ComponentTreeKind.MIN_TREE,
            {
                frozenset({0}): (0, frozenset({1})),
                frozenset({1}): (1, frozenset({2})),
                frozenset({2}): (2, None),
            },
        ),
        (
            "saddle_or_merge",
            ComponentTreeKind.MAX_TREE,
            {
                frozenset({0}): (3, frozenset({1})),
                frozenset({2}): (3, frozenset({1})),
                frozenset({1}): (1, frozenset({3})),
                frozenset({3}): (0, None),
            },
        ),
        (
            "saddle_or_merge",
            ComponentTreeKind.MIN_TREE,
            {
                frozenset({3}): (0, frozenset({1})),
                frozenset({1}): (1, frozenset({0, 2})),
                frozenset({0, 2}): (3, None),
            },
        ),
    ],
)
def test_named_components_match_membership_levels_and_parents(
    case_name,
    kind,
    expected,
):
    case = CASES[case_name]
    graph, _ = build_higra_rag(case["image"])
    component_tree = _case_component_tree(case, kind)

    reconstructed = nahs_to_higra(graph, component_tree)
    leaf_to_nahs = np.arange(graph.levels.size, dtype=np.int64)

    assert (
        _component_structure(
            reconstructed.tree,
            reconstructed.altitudes,
            leaf_to_nahs,
        )
        == expected
    )


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_reconstructed_tree_has_higra_hierarchy_ordering(case_name, kind):
    case = CASES[case_name]
    graph, _ = build_higra_rag(case["image"])

    reconstructed = nahs_to_higra(
        graph,
        _case_component_tree(case, kind),
    )

    higra_parents = np.asarray(reconstructed.tree.parents(), dtype=np.int64)
    root = higra_parents.size - 1
    non_root = np.arange(root)
    leaf_graph = hg.CptHierarchy.get_leaf_graph(reconstructed.tree)
    sources, targets = leaf_graph.edge_list()
    actual_edges = set(zip(sources.tolist(), targets.tolist(), strict=True))

    assert hg.CptHierarchy.validate(reconstructed.tree)
    assert np.all(higra_parents[non_root] > non_root)
    assert higra_parents[root] == root
    assert leaf_graph.num_vertices() == graph.levels.size
    assert actual_edges == set(map(tuple, graph.edge_index.tolist()))
    np.testing.assert_array_equal(
        reconstructed.altitudes[: graph.levels.size],
        graph.levels,
    )

    internal = np.arange(graph.levels.size, root)
    if kind == ComponentTreeKind.MAX_TREE:
        assert np.all(
            reconstructed.altitudes[higra_parents[internal]]
            <= reconstructed.altitudes[internal]
        )
    else:
        assert np.all(
            reconstructed.altitudes[higra_parents[internal]]
            >= reconstructed.altitudes[internal]
        )


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_nahs_to_higra_to_nahs_reproduces_exact_parents(case_name, kind):
    case = CASES[case_name]
    graph, _ = build_higra_rag(case["image"])
    original = _case_component_tree(case, kind)

    reconstructed = nahs_to_higra(graph, original)
    context = HigraRagContext(
        rag=hg.CptHierarchy.get_leaf_graph(reconstructed.tree),
        higra_to_nahs=np.arange(graph.levels.size, dtype=np.int64),
    )
    round_trip = higra_to_nahs(
        graph,
        reconstructed.tree,
        context,
        kind,
    )

    np.testing.assert_array_equal(round_trip.parents, original.parents)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_higra_to_nahs_to_higra_preserves_component_structure(case_name, kind):
    graph, context = build_higra_rag(CASES[case_name]["image"])
    original_tree, original_altitudes = _build_higra_tree(graph, context, kind)
    component_tree = higra_to_nahs(
        graph,
        original_tree,
        context,
        kind,
    )

    reconstructed = nahs_to_higra(graph, component_tree)
    reconstructed_leaf_map = np.arange(graph.levels.size, dtype=np.int64)

    assert _component_structure(
        reconstructed.tree,
        reconstructed.altitudes,
        reconstructed_leaf_map,
    ) == _component_structure(
        original_tree,
        original_altitudes,
        context.higra_to_nahs,
    )
