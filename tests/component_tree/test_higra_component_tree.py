import itertools

import higra as hg
import numpy as np
import pytest

from lenahs.component_tree import (
    ComponentTreeKind,
    build_reference_component_tree,
    build_reference_rag,
)
from lenahs.component_tree.higra_component_tree import higra_to_component_tree
from lenahs.component_tree.higra_component_tree_vectorized import (
    higra_to_component_tree_vectorized,
)
from lenahs.component_tree.higra_rag import HigraRagContext, build_higra_rag

from .cases import CASES


CONVERTERS = [
    pytest.param(higra_to_component_tree, id="readable"),
    pytest.param(higra_to_component_tree_vectorized, id="vectorized"),
]


def _build_higra_tree(graph, context, kind):
    higra_levels = graph.levels[context.higra_to_nahs]
    if kind == ComponentTreeKind.MAX_TREE:
        return hg.component_tree_max_tree(context.rag, higra_levels)[0]
    return hg.component_tree_min_tree(context.rag, higra_levels)[0]


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize(
    ("kind", "expected_key"),
    [
        (ComponentTreeKind.MAX_TREE, "max_parents"),
        (ComponentTreeKind.MIN_TREE, "min_parents"),
    ],
)
@pytest.mark.parametrize("converter", CONVERTERS)
def test_named_parent_arrays(case_name, kind, expected_key, converter):
    case = CASES[case_name]
    graph, context = build_higra_rag(case["image"])
    higra_tree = _build_higra_tree(graph, context, kind)

    tree = converter(graph, higra_tree, context, kind)

    assert tree.kind == kind
    np.testing.assert_array_equal(tree.parents, case[expected_key])


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("kind", list(ComponentTreeKind))
@pytest.mark.parametrize("converter", CONVERTERS)
def test_named_conversions_match_reference(case_name, kind, converter):
    image = CASES[case_name]["image"]
    graph, context = build_higra_rag(image)
    higra_tree = _build_higra_tree(graph, context, kind)

    converted = converter(graph, higra_tree, context, kind)
    reference_graph = build_reference_rag(image)
    reference = build_reference_component_tree(reference_graph, kind)

    np.testing.assert_array_equal(converted.parents, reference.parents)


@pytest.mark.parametrize(
    ("case_name", "expected_root"),
    [("equal_size_tie", 0), ("size_beats_anchor", 2)],
)
@pytest.mark.parametrize("converter", CONVERTERS)
def test_canonical_representative_selects_size_then_anchor(
    case_name,
    expected_root,
    converter,
):
    graph, context = build_higra_rag(CASES[case_name]["image"])
    higra_tree = _build_higra_tree(graph, context, ComponentTreeKind.MAX_TREE)

    tree = converter(
        graph,
        higra_tree,
        context,
        ComponentTreeKind.MAX_TREE,
    )

    assert tree.parents[expected_root] == expected_root
    assert np.all(tree.parents == expected_root)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("kind", list(ComponentTreeKind))
@pytest.mark.parametrize("converter", CONVERTERS)
def test_higra_leaf_renumbering_preserves_parents(case_name, kind, converter):
    graph, context = build_higra_rag(CASES[case_name]["image"])
    higra_tree = _build_higra_tree(graph, context, kind)
    original = converter(graph, higra_tree, context, kind)

    leaf_count = graph.levels.size
    new_to_old = np.arange(leaf_count - 1, -1, -1)
    old_to_new = np.empty(leaf_count, dtype=np.int64)
    old_to_new[new_to_old] = np.arange(leaf_count)

    original_parents = np.asarray(higra_tree.parents(), dtype=np.int64)
    permuted_parents = original_parents.copy()
    permuted_parents[old_to_new[np.arange(leaf_count)]] = original_parents[:leaf_count]
    permuted_tree = hg.Tree(permuted_parents)
    permuted_context = HigraRagContext(
        rag=context.rag,
        higra_to_nahs=context.higra_to_nahs[new_to_old],
    )

    permuted = converter(graph, permuted_tree, permuted_context, kind)

    np.testing.assert_array_equal(permuted.parents, original.parents)


@pytest.mark.parametrize("converter", CONVERTERS)
def test_higra_component_renumbering_preserves_parents(converter):
    graph, context = build_higra_rag(CASES["saddle_or_merge"]["image"])
    kind = ComponentTreeKind.MAX_TREE
    higra_tree = _build_higra_tree(graph, context, kind)
    original = converter(graph, higra_tree, context, kind)

    original_parents = np.asarray(higra_tree.parents(), dtype=np.int64)
    old_to_new = np.arange(original_parents.size, dtype=np.int64)
    old_to_new[4:6] = [5, 4]
    permuted_parents = np.empty_like(original_parents)
    permuted_parents[old_to_new] = old_to_new[original_parents]
    permuted_tree = hg.Tree(permuted_parents)

    permuted = converter(graph, permuted_tree, context, kind)

    np.testing.assert_array_equal(permuted.parents, original.parents)


@pytest.mark.parametrize("converter", CONVERTERS)
def test_all_two_by_three_images_over_three_levels_match_reference(converter):
    for values in itertools.product(range(3), repeat=6):
        image = np.asarray(values, dtype=np.uint8).reshape(2, 3)
        graph, context = build_higra_rag(image)
        reference_graph = build_reference_rag(image)

        for kind in ComponentTreeKind:
            higra_tree = _build_higra_tree(graph, context, kind)
            converted = converter(graph, higra_tree, context, kind)
            reference = build_reference_component_tree(reference_graph, kind)

            np.testing.assert_array_equal(converted.parents, reference.parents)
