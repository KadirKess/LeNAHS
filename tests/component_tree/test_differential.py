from itertools import product

import higra as hg
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lenahs.component_tree import (
    ComponentTreeKind,
    build_reference_component_tree,
    build_reference_rag,
)
from lenahs.component_tree.higra import (
    HigraRagContext,
    build_higra_rag,
    higra_to_nahs_vectorized,
    nahs_to_higra,
)


@st.composite
def _small_uint8_images(draw):
    height = draw(st.integers(min_value=1, max_value=4))
    width = draw(st.integers(min_value=1, max_value=4))
    values = draw(
        st.lists(
            st.integers(min_value=0, max_value=255),
            min_size=height * width,
            max_size=height * width,
        )
    )
    return np.asarray(values, dtype=np.uint8).reshape(height, width)


def _build_higra_tree(graph, context, kind):
    higra_levels = graph.levels[context.higra_to_nahs]
    if kind == ComponentTreeKind.MAX_TREE:
        return hg.component_tree_max_tree(context.rag, higra_levels)
    return hg.component_tree_min_tree(context.rag, higra_levels)


def _build_converted_tree(image, kind):
    graph, context = build_higra_rag(image)
    higra_tree, altitudes = _build_higra_tree(graph, context, kind)
    component_tree = higra_to_nahs_vectorized(
        graph,
        higra_tree,
        context,
        kind,
    )
    return graph, context, higra_tree, altitudes, component_tree


def _component_structure_by_pixels(
    graph,
    higra_tree,
    altitudes,
    leaf_to_nahs,
    pixel_ids=None,
):
    parents = np.asarray(higra_tree.parents(), dtype=np.int64)
    leaf_count = leaf_to_nahs.size
    if pixel_ids is None:
        pixel_ids = np.arange(graph.label_map.size, dtype=np.int64).reshape(
            graph.label_map.shape
        )

    members = [set() for _ in range(parents.size)]
    for leaf, nahs_node in enumerate(leaf_to_nahs):
        zone_pixels = pixel_ids[graph.label_map == int(nahs_node)]
        members[leaf].update(int(pixel) for pixel in zone_pixels)

    for node, parent in enumerate(parents):
        parent = int(parent)
        if node != parent:
            members[parent].update(members[node])

    structure = {}
    for component in range(leaf_count, parents.size):
        parent = int(parents[component])
        parent_members = None if parent == component else frozenset(members[parent])
        structure[frozenset(members[component])] = (
            int(altitudes[component]),
            parent_members,
        )

    return structure


def _assert_graphs_equal(actual, expected):
    np.testing.assert_array_equal(actual.label_map, expected.label_map)
    np.testing.assert_array_equal(actual.edge_index, expected.edge_index)
    np.testing.assert_array_equal(actual.levels, expected.levels)
    np.testing.assert_array_equal(actual.pixel_counts, expected.pixel_counts)
    np.testing.assert_array_equal(actual.anchors, expected.anchors)


def _assert_complete_round_trip(image):
    for kind in ComponentTreeKind:
        graph, context, original, altitudes, component_tree = _build_converted_tree(
            image,
            kind,
        )
        reconstructed = nahs_to_higra(graph, component_tree)

        assert hg.CptHierarchy.validate(reconstructed.tree)
        assert _component_structure_by_pixels(
            graph,
            reconstructed.tree,
            reconstructed.altitudes,
            np.arange(graph.levels.size, dtype=np.int64),
        ) == _component_structure_by_pixels(
            graph,
            original,
            altitudes,
            context.higra_to_nahs,
        )

        reconstructed_context = HigraRagContext(
            rag=hg.CptHierarchy.get_leaf_graph(reconstructed.tree),
            higra_to_nahs=np.arange(graph.levels.size, dtype=np.int64),
        )
        round_trip = higra_to_nahs_vectorized(
            graph,
            reconstructed.tree,
            reconstructed_context,
            kind,
        )
        np.testing.assert_array_equal(round_trip.parents, component_tree.parents)


@settings(max_examples=50, derandomize=True, deadline=None)
@given(image=_small_uint8_images())
def test_random_small_images_match_the_reference_through_the_complete_path(image):
    reference_graph = build_reference_rag(image)
    graph, context = build_higra_rag(image)
    _assert_graphs_equal(graph, reference_graph)

    for kind in ComponentTreeKind:
        higra_tree, altitudes = _build_higra_tree(graph, context, kind)
        component_tree = higra_to_nahs_vectorized(
            graph,
            higra_tree,
            context,
            kind,
        )
        reference_tree = build_reference_component_tree(reference_graph, kind)
        np.testing.assert_array_equal(component_tree.parents, reference_tree.parents)

        reconstructed = nahs_to_higra(graph, component_tree)
        assert _component_structure_by_pixels(
            graph,
            reconstructed.tree,
            reconstructed.altitudes,
            np.arange(graph.levels.size, dtype=np.int64),
        ) == _component_structure_by_pixels(
            graph,
            higra_tree,
            altitudes,
            context.higra_to_nahs,
        )


@pytest.mark.parametrize(
    ("shape", "levels"),
    [
        pytest.param((2, 3), range(3), id="two_by_three_three_levels"),
        pytest.param((3, 3), range(2), id="three_by_three_binary"),
    ],
)
def test_exhaustive_small_images_preserve_both_conversion_round_trips(
    shape,
    levels,
):
    for values in product(levels, repeat=int(np.prod(shape))):
        image = np.asarray(values, dtype=np.uint8).reshape(shape)
        _assert_complete_round_trip(image)


@settings(max_examples=40, derandomize=True, deadline=None)
@given(image=_small_uint8_images())
def test_monotone_level_remapping_preserves_the_production_hierarchy(image):
    level_ranks = np.searchsorted(np.unique(image), image)
    increasing_image = np.asarray(level_ranks * 7 + 3, dtype=np.uint8)
    decreasing_image = np.asarray(252 - level_ranks * 7, dtype=np.uint8)

    original = {
        kind: _build_converted_tree(image, kind)[-1] for kind in ComponentTreeKind
    }
    increasing = {
        kind: _build_converted_tree(increasing_image, kind)[-1]
        for kind in ComponentTreeKind
    }
    decreasing = {
        kind: _build_converted_tree(decreasing_image, kind)[-1]
        for kind in ComponentTreeKind
    }

    for kind in ComponentTreeKind:
        np.testing.assert_array_equal(
            increasing[kind].parents,
            original[kind].parents,
        )

    np.testing.assert_array_equal(
        decreasing[ComponentTreeKind.MIN_TREE].parents,
        original[ComponentTreeKind.MAX_TREE].parents,
    )
    np.testing.assert_array_equal(
        decreasing[ComponentTreeKind.MAX_TREE].parents,
        original[ComponentTreeKind.MIN_TREE].parents,
    )


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(np.fliplr, id="horizontal_reflection"),
        pytest.param(np.flipud, id="vertical_reflection"),
        pytest.param(np.rot90, id="quarter_turn"),
    ],
)
@settings(max_examples=20, derandomize=True, deadline=None)
@given(image=_small_uint8_images())
def test_spatial_transformations_preserve_component_structure(transform, image):
    original_pixel_ids = np.arange(image.size, dtype=np.int64).reshape(image.shape)

    transformed_image = transform(image)
    transformed_pixel_ids = transform(original_pixel_ids)

    for kind in ComponentTreeKind:
        graph, _, _, _, component_tree = _build_converted_tree(image, kind)
        reconstructed = nahs_to_higra(graph, component_tree)

        transformed_graph, _, _, _, transformed_tree = _build_converted_tree(
            transformed_image,
            kind,
        )
        transformed_reconstructed = nahs_to_higra(
            transformed_graph,
            transformed_tree,
        )

        assert _component_structure_by_pixels(
            graph,
            reconstructed.tree,
            reconstructed.altitudes,
            np.arange(graph.levels.size, dtype=np.int64),
            original_pixel_ids,
        ) == _component_structure_by_pixels(
            transformed_graph,
            transformed_reconstructed.tree,
            transformed_reconstructed.altitudes,
            np.arange(transformed_graph.levels.size, dtype=np.int64),
            transformed_pixel_ids,
        )


def test_spatial_transform_uses_the_new_row_major_anchor_for_a_tie():
    image = np.asarray(
        [
            [3, 1, 3],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    original_pixel_ids = np.arange(image.size, dtype=np.int64).reshape(image.shape)

    graph, _, _, _, component_tree = _build_converted_tree(
        image,
        ComponentTreeKind.MIN_TREE,
    )
    transformed_graph, _, _, _, transformed_tree = _build_converted_tree(
        np.fliplr(image),
        ComponentTreeKind.MIN_TREE,
    )

    original_root = int(
        np.flatnonzero(
            component_tree.parents == np.arange(graph.levels.size, dtype=np.int64)
        )[0]
    )
    transformed_root = int(
        np.flatnonzero(
            transformed_tree.parents
            == np.arange(transformed_graph.levels.size, dtype=np.int64)
        )[0]
    )
    original_region = frozenset(
        int(pixel) for pixel in original_pixel_ids[graph.label_map == original_root]
    )
    transformed_region = frozenset(
        int(pixel)
        for pixel in np.fliplr(original_pixel_ids)[
            transformed_graph.label_map == transformed_root
        ]
    )

    assert original_region == frozenset({0})
    assert transformed_region == frozenset({2})


def test_repeated_complete_generation_is_deterministic():
    image = np.asarray(
        [
            [9, 2, 7, 7],
            [9, 4, 4, 1],
            [6, 6, 3, 1],
        ],
        dtype=np.uint8,
    )

    for kind in ComponentTreeKind:
        first = _build_converted_tree(image, kind)
        second = _build_converted_tree(image, kind)
        _assert_graphs_equal(first[0], second[0])
        np.testing.assert_array_equal(first[2].parents(), second[2].parents())
        np.testing.assert_array_equal(first[3], second[3])
        np.testing.assert_array_equal(first[4].parents, second[4].parents)
