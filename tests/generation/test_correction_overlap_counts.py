import numpy as np
import pytest

from lenahs.component_tree.reference import build_reference_rag
from lenahs.generation.targets import _count_zone_overlaps


@pytest.mark.parametrize("permute_ids", [False, True])
def test_sparse_counts_match_independent_pixel_pairs(permute_ids):
    rng = np.random.default_rng(20261006)
    image_pairs = [
        ([[10]], [[200]]),
        ([[10, 10], [10, 20]], [[0, 0], [20, 10]]),
        ([[10, 10, 20, 30]], [[7, 7, 7, 8]]),
    ]
    for _ in range(32):
        image_pairs.append(
            (
                rng.integers(0, 5, size=(6, 7), dtype=np.uint8),
                rng.integers(0, 5, size=(6, 7), dtype=np.uint8),
            )
        )

    for clean, observed in image_pairs:
        clean_graph = build_reference_rag(np.array(clean, dtype=np.uint8))
        observed_graph = build_reference_rag(np.array(observed, dtype=np.uint8))
        clean_labels = clean_graph.label_map
        observed_labels = observed_graph.label_map
        if permute_ids:
            clean_labels = rng.permutation(clean_graph.levels.size)[clean_labels]
            observed_labels = rng.permutation(observed_graph.levels.size)[
                observed_labels
            ]

        # Coordinate indexing supplies an oracle independent of flattened iteration.
        expected = {}
        for row, column in np.ndindex(clean_labels.shape):
            pair = (int(observed_labels[row, column]), int(clean_labels[row, column]))
            expected[pair] = expected.get(pair, 0) + 1

        for clean_view, observed_view in (
            (clean_labels, observed_labels),
            (clean_labels[:, ::-1], observed_labels[:, ::-1]),
            (clean_labels.T, observed_labels.T),
        ):
            counts = _count_zone_overlaps(
                clean_view, observed_view, observed_graph.levels.size
            )
            actual = {
                (observed_zone, clean_zone): count
                for observed_zone, overlaps in enumerate(counts)
                for clean_zone, count in overlaps.items()
            }
            assert actual == expected
            assert sum(actual.values()) == clean_labels.size
            assert all(count > 0 for count in actual.values())
