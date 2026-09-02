import numpy as np
import pytest

from lenahs.component_tree import build_reference_rag

from .cases import CASES


@pytest.mark.parametrize("case_name", CASES)
def test_named_case_rag_facts(case_name):
    case = CASES[case_name]

    graph = build_reference_rag(case["image"])

    np.testing.assert_array_equal(graph.label_map, case["label_map"])
    assert {tuple(edge) for edge in graph.edge_index.tolist()} == case["edges"]
    assert graph.edge_index.shape == (len(case["edges"]), 2)
    np.testing.assert_array_equal(graph.levels, case["levels"])
    np.testing.assert_array_equal(graph.pixel_counts, case["pixel_counts"])
    np.testing.assert_array_equal(graph.anchors, case["anchors"])
    assert np.issubdtype(graph.label_map.dtype, np.integer)
    assert np.issubdtype(graph.edge_index.dtype, np.integer)
    assert graph.levels.dtype == np.uint8
    assert np.issubdtype(graph.pixel_counts.dtype, np.integer)
    assert np.issubdtype(graph.anchors.dtype, np.integer)


def test_one_column_image():
    image = np.array([[4], [4], [9], [2]], dtype=np.uint8)

    graph = build_reference_rag(image)

    np.testing.assert_array_equal(graph.label_map[:, 0], [0, 0, 1, 2])
    assert {tuple(edge) for edge in graph.edge_index.tolist()} == {(0, 1), (1, 2)}
    np.testing.assert_array_equal(graph.levels, [4, 9, 2])
    np.testing.assert_array_equal(graph.pixel_counts, [2, 1, 1])
    np.testing.assert_array_equal(graph.anchors, [0, 2, 3])


def test_diagonal_contact_does_not_join_zones_or_create_edges():
    image = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    graph = build_reference_rag(image)

    np.testing.assert_array_equal(graph.label_map, [[0, 1], [2, 3]])
    assert {tuple(edge) for edge in graph.edge_index.tolist()} == {
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    }


def test_non_contiguous_view_matches_its_contiguous_copy():
    source = np.array(
        [[1, 8, 1, 8, 2], [1, 8, 3, 8, 2], [4, 8, 3, 8, 2]],
        dtype=np.uint8,
    )
    view = source[:, ::2]
    assert not view.flags.c_contiguous

    from_view = build_reference_rag(view)
    from_copy = build_reference_rag(np.ascontiguousarray(view))

    np.testing.assert_array_equal(from_view.label_map, from_copy.label_map)
    np.testing.assert_array_equal(from_view.edge_index, from_copy.edge_index)
    np.testing.assert_array_equal(from_view.levels, from_copy.levels)
    np.testing.assert_array_equal(from_view.pixel_counts, from_copy.pixel_counts)
    np.testing.assert_array_equal(from_view.anchors, from_copy.anchors)


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
def test_invalid_image_fails_clearly(image, error, message):
    with pytest.raises(error, match=message):
        build_reference_rag(image)


@pytest.mark.parametrize("case_name", CASES)
def test_every_label_is_used_and_describes_one_connected_level(case_name):
    image = CASES[case_name]["image"]
    graph = build_reference_rag(image)
    labels = np.unique(graph.label_map)

    np.testing.assert_array_equal(labels, np.arange(len(graph.levels)))

    height, width = image.shape
    for zone in labels:
        pixels = set(zip(*np.where(graph.label_map == zone)))
        pending = [next(iter(pixels))]
        reached = {pending[0]}

        while pending:
            row, column = pending.pop()
            for row_offset, column_offset in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ):
                neighbor = (row + row_offset, column + column_offset)
                if neighbor in pixels and neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)

        assert reached == pixels
        zone_values = image[graph.label_map == zone]
        assert np.all(zone_values == graph.levels[zone])
        assert len(pixels) == graph.pixel_counts[zone]
        assert min(row * width + column for row, column in pixels) == graph.anchors[zone]


@pytest.mark.parametrize("case_name", CASES)
def test_edges_are_exactly_the_cross_zone_pixel_boundaries(case_name):
    label_map = build_reference_rag(CASES[case_name]["image"]).label_map
    height, width = label_map.shape
    boundary_edges = set()

    for row in range(height):
        for column in range(width):
            zone = int(label_map[row, column])
            for row_offset, column_offset in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ):
                neighbor_row = row + row_offset
                neighbor_column = column + column_offset
                if not (
                    0 <= neighbor_row < height and 0 <= neighbor_column < width
                ):
                    continue
                neighbor_zone = int(label_map[neighbor_row, neighbor_column])
                if zone != neighbor_zone:
                    boundary_edges.add(tuple(sorted((zone, neighbor_zone))))

    graph = build_reference_rag(CASES[case_name]["image"])
    stored_edges = {tuple(edge) for edge in graph.edge_index.tolist()}

    assert stored_edges == boundary_edges
    assert all(left < right for left, right in stored_edges)
    assert graph.edge_index.tolist() == [list(edge) for edge in sorted(stored_edges)]
