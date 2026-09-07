from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class NoiseDirection(IntEnum):
    """Class indices for comparison with the mean clean intensity."""

    UNCHANGED = 0
    BRIGHTENED = 1
    DARKENED = 2


@dataclass
class CorruptionTargets:
    noise_direction: NDArray[np.int64]
    overlap_purity: NDArray[np.float64]
    overlap_clean_zone_count: NDArray[np.integer]


def _validate_image(name: str, image: NDArray[np.uint8]) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if image.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if image.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8.")
    if image.size == 0:
        raise ValueError(f"{name} must not be empty.")


def _validate_label_map(name: str, label_map: NDArray[np.integer]) -> int:
    if not isinstance(label_map, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if label_map.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if not np.issubdtype(label_map.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype.")
    if label_map.size == 0 or np.min(label_map) < 0:
        raise ValueError(f"{name} must contain nonnegative labels.")

    labels = np.unique(label_map)
    expected = np.arange(labels.size, dtype=labels.dtype)
    if not np.array_equal(labels, expected):
        raise ValueError(f"{name} labels must be contiguous and start at zero.")
    return int(labels.size)


def _validate_flat_zones(
    name: str,
    image: NDArray[np.uint8],
    label_map: NDArray[np.integer],
    zone_count: int,
) -> None:
    for zone in range(zone_count):
        levels = image[label_map == zone]
        if np.any(levels != levels[0]):
            raise ValueError(f"{name} does not describe flat zones in its image.")


def derive_corruption_targets(
    clean_image: NDArray[np.uint8],
    corrupted_image: NDArray[np.uint8],
    clean_label_map: NDArray[np.integer],
    corrupted_label_map: NDArray[np.integer],
) -> CorruptionTargets:
    """Compare observed levels with mean clean levels, including mixed zones."""

    _validate_image("clean_image", clean_image)
    _validate_image("corrupted_image", corrupted_image)
    if clean_image.shape != corrupted_image.shape:
        raise ValueError("clean_image and corrupted_image must have the same shape.")
    if clean_label_map.shape != clean_image.shape:
        raise ValueError("clean_label_map must have the same shape as clean_image.")
    if corrupted_label_map.shape != corrupted_image.shape:
        raise ValueError(
            "corrupted_label_map must have the same shape as corrupted_image."
        )

    clean_zone_count = _validate_label_map("clean_label_map", clean_label_map)
    corrupted_zone_count = _validate_label_map(
        "corrupted_label_map", corrupted_label_map
    )
    _validate_flat_zones(
        "clean_label_map", clean_image, clean_label_map, clean_zone_count
    )
    _validate_flat_zones(
        "corrupted_label_map",
        corrupted_image,
        corrupted_label_map,
        corrupted_zone_count,
    )

    noise_direction = np.empty(corrupted_zone_count, dtype=np.int64)
    overlap_purity = np.empty(corrupted_zone_count, dtype=np.float64)
    overlap_clean_zone_count = np.empty(corrupted_zone_count, dtype=np.int64)

    for observed_zone in range(corrupted_zone_count):
        pixels = corrupted_label_map == observed_zone
        clean_zones, counts = np.unique(clean_label_map[pixels], return_counts=True)
        clean_values = clean_image[pixels]
        observed_level = corrupted_image[pixels][0]

        # Comparing sums preserves fractional means and exact equality.
        # Python integers avoid overflow in both the sum and the product.
        clean_sum = sum(map(int, clean_values))
        observed_sum = int(observed_level) * clean_values.size
        if observed_sum > clean_sum:
            noise_direction[observed_zone] = NoiseDirection.BRIGHTENED
        elif observed_sum < clean_sum:
            noise_direction[observed_zone] = NoiseDirection.DARKENED
        else:
            noise_direction[observed_zone] = NoiseDirection.UNCHANGED

        overlap_purity[observed_zone] = float(np.max(counts) / np.sum(counts))
        overlap_clean_zone_count[observed_zone] = clean_zones.size

    return CorruptionTargets(
        noise_direction=noise_direction,
        overlap_purity=overlap_purity,
        overlap_clean_zone_count=overlap_clean_zone_count,
    )
