import numpy as np
import pytest

from lenahs.component_tree import FlatZoneGraph
from lenahs.generation import NoiseDirection, derive_corruption_targets


def _graph(image: np.ndarray, label_map: np.ndarray) -> FlatZoneGraph:
    label_map = np.asarray(label_map, dtype=np.int64)
    node_count = int(label_map.max()) + 1
    flat_labels = label_map.ravel()
    anchors = np.full(node_count, image.size, dtype=np.int64)
    np.minimum.at(anchors, flat_labels, np.arange(image.size, dtype=np.int64))
    return FlatZoneGraph(
        label_map=label_map,
        edge_index=np.empty((0, 2), dtype=np.int64),
        levels=image.ravel()[anchors],
        pixel_counts=np.bincount(flat_labels, minlength=node_count),
        anchors=anchors,
    )


def test_clean_identity_has_unchanged_targets() -> None:
    image = np.array([[10, 10, 30], [10, 20, 30]], dtype=np.uint8)
    label_map = np.array([[0, 0, 2], [0, 1, 2]], dtype=np.int64)

    graph = _graph(image, label_map)
    targets = derive_corruption_targets(image, graph, graph)

    np.testing.assert_array_equal(targets.noise_direction, [0, 0, 0])
    np.testing.assert_array_equal(targets.overlap_purity, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(targets.overlap_clean_zone_count, [1, 1, 1])


@pytest.mark.parametrize(
    "clean_values, clean_labels, observed, direction, purity, overlap_count",
    [
        ([40, 40], [0, 0], 40, NoiseDirection.UNCHANGED, 1.0, 1),
        ([40, 40], [0, 0], 255, NoiseDirection.BRIGHTENED, 1.0, 1),
        ([200, 200], [0, 0], 0, NoiseDirection.DARKENED, 1.0, 1),
        ([20, 80], [0, 1], 60, NoiseDirection.BRIGHTENED, 0.5, 2),
        ([20, 80], [0, 1], 40, NoiseDirection.DARKENED, 0.5, 2),
        ([20, 80], [0, 1], 50, NoiseDirection.UNCHANGED, 0.5, 2),
        ([20, 20, 80], [0, 0, 1], 35, NoiseDirection.DARKENED, 2 / 3, 2),
        ([20, 20, 80], [0, 0, 1], 45, NoiseDirection.BRIGHTENED, 2 / 3, 2),
        ([0, 1], [0, 1], 0, NoiseDirection.DARKENED, 0.5, 2),
        ([0, 1], [0, 1], 1, NoiseDirection.BRIGHTENED, 0.5, 2),
        ([254, 255], [0, 1], 254, NoiseDirection.DARKENED, 0.5, 2),
        ([254, 255], [0, 1], 255, NoiseDirection.BRIGHTENED, 0.5, 2),
        ([255, 255], [0, 0], 255, NoiseDirection.UNCHANGED, 1.0, 1),
    ],
)
def test_direction_compares_observed_level_with_unrounded_pixel_mean(
    clean_values, clean_labels, observed, direction, purity, overlap_count
) -> None:
    clean_image = np.array([clean_values], dtype=np.uint8)
    corrupted_image = np.full_like(clean_image, observed)

    clean_graph = _graph(clean_image, np.array([clean_labels], dtype=np.int64))
    corrupted_graph = _graph(
        corrupted_image,
        np.zeros_like(clean_image, dtype=np.int64),
    )
    targets = derive_corruption_targets(clean_image, clean_graph, corrupted_graph)

    np.testing.assert_array_equal(targets.noise_direction, [direction])
    np.testing.assert_allclose(targets.overlap_purity, [purity])
    np.testing.assert_array_equal(targets.overlap_clean_zone_count, [overlap_count])


def test_mixed_equality_is_unchanged_among_changed_and_unchanged_zones() -> None:
    clean_image = np.array([[10, 10, 30, 30], [10, 10, 30, 30]], dtype=np.uint8)
    corrupted_image = np.array([[15, 20, 20, 30], [15, 20, 20, 30]], dtype=np.uint8)
    clean_labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int64)
    corrupted_labels = np.array([[0, 1, 1, 2], [0, 1, 1, 2]], dtype=np.int64)

    targets = derive_corruption_targets(
        clean_image,
        _graph(clean_image, clean_labels),
        _graph(corrupted_image, corrupted_labels),
    )

    np.testing.assert_array_equal(targets.noise_direction, [1, 0, 0])
    np.testing.assert_array_equal(targets.overlap_purity, [1.0, 0.5, 1.0])
    np.testing.assert_array_equal(targets.overlap_clean_zone_count, [1, 2, 1])


@pytest.mark.parametrize("permute_clean", [False, True])
@pytest.mark.parametrize("permute_observed", [False, True])
def test_targets_follow_observed_ids_independently_of_clean_ids(
    permute_clean, permute_observed
) -> None:
    clean_image = np.array([[20, 20, 80, 80, 80, 100]], dtype=np.uint8)
    corrupted_image = np.array([[50, 50, 50, 40, 80, 80]], dtype=np.uint8)
    clean_labels = np.array([[0, 0, 1, 1, 1, 2]])
    corrupted_labels = np.array([[0, 0, 0, 1, 2, 2]])
    clean_permutation = np.array([2, 0, 1]) if permute_clean else np.arange(3)
    observed_permutation = np.array([1, 2, 0]) if permute_observed else np.arange(3)

    permuted_clean_labels = clean_permutation[clean_labels]
    permuted_corrupted_labels = observed_permutation[corrupted_labels]
    targets = derive_corruption_targets(
        clean_image,
        _graph(clean_image, permuted_clean_labels),
        _graph(corrupted_image, permuted_corrupted_labels),
    )

    # The three observed zones have clean means of 40, 80, and 90.
    np.testing.assert_array_equal(
        targets.noise_direction[observed_permutation], [1, 2, 2]
    )
    np.testing.assert_allclose(
        targets.overlap_purity[observed_permutation], [2 / 3, 1.0, 0.5]
    )
    np.testing.assert_array_equal(
        targets.overlap_clean_zone_count[observed_permutation], [2, 1, 2]
    )
