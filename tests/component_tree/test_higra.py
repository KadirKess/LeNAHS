import itertools

import numpy as np
import pytest

from lenahs.component_tree import build_reference_rag
from lenahs.component_tree.higra import build_higra_rag, _remap_higra_regions

from .cases import CASES


def _assert_same_graph(left, right):
    np.testing.assert_array_equal(left.label_map, right.label_map)
    np.testing.assert_array_equal(left.edge_index, right.edge_index)
    np.testing.assert_array_equal(left.levels, right.levels)
    np.testing.assert_array_equal(left.pixel_counts, right.pixel_counts)
    np.testing.assert_array_equal(left.anchors, right.anchors)


@pytest.mark.parametrize("case_name", CASES)
def test_named_cases_match_reference_rag(case_name):
    image = CASES[case_name]["image"]

    graph, _ = build_higra_rag(image)

    _assert_same_graph(graph, build_reference_rag(image))


def test_repeated_calls_return_identical_nahs_arrays():
    image = CASES["saddle_or_merge"]["image"]

    first, _ = build_higra_rag(image)
    second, _ = build_higra_rag(image)

    _assert_same_graph(first, second)


def test_context_maps_higra_regions_to_nahs_nodes():
    image = CASES["saddle_or_merge"]["image"]

    graph, context = build_higra_rag(image)

    np.testing.assert_array_equal(
        context.higra_to_nahs[context.rag.vertex_map].reshape(image.shape),
        graph.label_map,
    )


def test_source_region_permutation_does_not_change_mapped_graph():
    image = CASES["saddle_or_merge"]["image"]
    source, context = build_higra_rag(image)
    pixel_levels = image.reshape(-1)
    pixel_to_higra = np.asarray(
        context.rag.vertex_map,
        dtype=np.int64,
    )
    sources, targets = context.rag.edge_list()

    permutation = np.array([2, 0, 3, 1], dtype=np.int64)
    permuted, _ = _remap_higra_regions(
        image.shape,
        pixel_levels,
        permutation[pixel_to_higra],
        (permutation[sources], permutation[targets]),
    )

    _assert_same_graph(permuted, source)


@pytest.mark.parametrize(
    "image",
    [
        np.array([[4, 4, 9, 2]], dtype=np.uint8),
        np.array([[4], [4], [9], [2]], dtype=np.uint8),
        np.full((2, 3), 12, dtype=np.uint8),
    ],
)
def test_degenerate_shapes_match_reference_rag(image):
    graph, _ = build_higra_rag(image)

    _assert_same_graph(graph, build_reference_rag(image))


def test_non_contiguous_image_matches_reference_rag():
    source = np.array(
        [[1, 8, 1, 8, 2], [1, 8, 3, 8, 2], [4, 8, 3, 8, 2]],
        dtype=np.uint8,
    )
    image = source[:, ::2]
    assert not image.flags.c_contiguous

    graph, _ = build_higra_rag(image)

    _assert_same_graph(graph, build_reference_rag(image))


@pytest.mark.parametrize(
    ("image", "error", "message"),
    [
        ([], TypeError, "NumPy array"),
        (np.empty((0, 1), dtype=np.uint8), ValueError, "empty"),
        (np.empty((1, 0), dtype=np.uint8), ValueError, "empty"),
        (np.zeros((2, 2, 1), dtype=np.uint8), ValueError, "2D"),
        (np.zeros((2, 2), dtype=np.float32), ValueError, "np.uint8"),
        (np.zeros((2, 2), dtype=np.int64), ValueError, "np.uint8"),
    ],
)
def test_invalid_image_matches_reference_contract(image, error, message):
    with pytest.raises(error, match=message):
        build_higra_rag(image)


def test_all_two_by_three_images_over_three_levels_match_reference_rag():
    for values in itertools.product(range(3), repeat=6):
        image = np.asarray(values, dtype=np.uint8).reshape(2, 3)

        graph, _ = build_higra_rag(image)

        _assert_same_graph(graph, build_reference_rag(image))
