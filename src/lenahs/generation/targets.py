from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from lenahs.component_tree.types import FlatZoneGraph


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


def derive_corruption_targets(
    clean_image: NDArray[np.uint8],
    clean_graph: FlatZoneGraph,
    corrupted_graph: FlatZoneGraph,
) -> CorruptionTargets:
    """Compare observed levels with mean clean levels, including mixed zones."""

    clean_labels = clean_graph.label_map.ravel()
    corrupted_labels = corrupted_graph.label_map.ravel()
    corrupted_zone_count = corrupted_graph.levels.size

    zone_pixel_counts = np.bincount(
        corrupted_labels,
        minlength=corrupted_zone_count,
    ).astype(np.int64, copy=False)
    clean_sums = np.zeros(corrupted_zone_count, dtype=np.int64)
    np.add.at(
        clean_sums,
        corrupted_labels,
        clean_image.ravel().astype(np.int64, copy=False),
    )
    observed_sums = (
        corrupted_graph.levels.astype(np.int64, copy=False) * zone_pixel_counts
    )

    noise_direction = np.full(
        corrupted_zone_count,
        NoiseDirection.UNCHANGED,
        dtype=np.int64,
    )
    noise_direction[observed_sums > clean_sums] = NoiseDirection.BRIGHTENED
    noise_direction[observed_sums < clean_sums] = NoiseDirection.DARKENED

    zone_pairs, pair_counts = np.unique(
        np.column_stack((corrupted_labels, clean_labels)),
        axis=0,
        return_counts=True,
    )
    pair_corrupted_zones = zone_pairs[:, 0]
    overlap_clean_zone_count = np.bincount(
        pair_corrupted_zones,
        minlength=corrupted_zone_count,
    ).astype(np.int64, copy=False)
    largest_overlap = np.zeros(corrupted_zone_count, dtype=np.int64)
    np.maximum.at(largest_overlap, pair_corrupted_zones, pair_counts)
    overlap_purity = largest_overlap / zone_pixel_counts

    return CorruptionTargets(
        noise_direction=noise_direction,
        overlap_purity=overlap_purity,
        overlap_clean_zone_count=overlap_clean_zone_count,
    )
