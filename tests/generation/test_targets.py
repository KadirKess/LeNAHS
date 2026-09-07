import numpy as np
import pytest

from lenahs.generation import NoiseDirection, derive_corruption_targets


def test_clean_identity_has_unchanged_targets() -> None:
    image = np.array([[10, 10, 30], [10, 20, 30]], dtype=np.uint8)
    label_map = np.array([[0, 0, 2], [0, 1, 2]], dtype=np.int64)

    targets = derive_corruption_targets(image, image, label_map, label_map)

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

    targets = derive_corruption_targets(
        clean_image,
        corrupted_image,
        np.array([clean_labels], dtype=np.int64),
        np.zeros_like(clean_image, dtype=np.int64),
    )

    np.testing.assert_array_equal(targets.noise_direction, [direction])
    np.testing.assert_allclose(targets.overlap_purity, [purity])
    np.testing.assert_array_equal(targets.overlap_clean_zone_count, [overlap_count])


def test_mixed_equality_is_unchanged_among_changed_and_unchanged_zones() -> None:
    clean_image = np.array([[10, 10, 30, 30], [10, 10, 30, 30]], dtype=np.uint8)
    corrupted_image = np.array([[15, 20, 20, 30], [15, 20, 20, 30]], dtype=np.uint8)
    clean_labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int64)
    corrupted_labels = np.array([[0, 1, 1, 2], [0, 1, 1, 2]], dtype=np.int64)

    targets = derive_corruption_targets(
        clean_image, corrupted_image, clean_labels, corrupted_labels
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

    targets = derive_corruption_targets(
        clean_image,
        corrupted_image,
        clean_permutation[clean_labels],
        observed_permutation[corrupted_labels],
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


def test_target_derivation_rejects_non_flat_observed_zone() -> None:
    clean_image = np.array([[10, 10]], dtype=np.uint8)
    corrupted_image = np.array([[10, 11]], dtype=np.uint8)
    label_map = np.array([[0, 0]], dtype=np.int64)

    with pytest.raises(ValueError, match="does not describe flat zones"):
        derive_corruption_targets(clean_image, corrupted_image, label_map, label_map)


@pytest.mark.parametrize(
    "invalid_map, error",
    [([[-1, 0]], ValueError), ([[0, 2]], ValueError), ([[0.0, 0.0]], TypeError)],
)
@pytest.mark.parametrize("invalid_side", ["clean", "observed"])
def test_target_derivation_rejects_invalid_label_ids(
    invalid_map, error, invalid_side
) -> None:
    image = np.array([[10, 10]], dtype=np.uint8)
    valid_map = np.array([[0, 0]], dtype=np.int64)
    clean_labels = np.array(invalid_map) if invalid_side == "clean" else valid_map
    observed_labels = np.array(invalid_map) if invalid_side == "observed" else valid_map

    with pytest.raises(error):
        derive_corruption_targets(image, image, clean_labels, observed_labels)
