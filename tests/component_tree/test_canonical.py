from itertools import permutations

import numpy as np
import pytest

from lenahs.component_tree import choose_representative

from .cases import CASES


def test_larger_zone_wins_despite_later_anchor():
    nodes = np.array([0, 1], dtype=np.int64)
    pixel_counts = np.array([2, 5], dtype=np.int64)
    anchors = np.array([3, 20], dtype=np.int64)

    representative = choose_representative(nodes, pixel_counts, anchors)

    assert representative == 1


def test_smaller_anchor_wins_when_counts_tie():
    nodes = np.array([0, 1], dtype=np.int64)
    pixel_counts = np.array([4, 4], dtype=np.int64)
    anchors = np.array([9, 2], dtype=np.int64)

    representative = choose_representative(nodes, pixel_counts, anchors)

    assert representative == 1


def test_candidate_permutation_does_not_change_selected_anchor():
    pixel_counts = np.array([3, 7, 7], dtype=np.int64)
    anchors = np.array([1, 8, 4], dtype=np.int64)

    selected_anchors = {
        int(
            anchors[
                choose_representative(
                    np.array(order, dtype=np.int64), pixel_counts, anchors
                )
            ]
        )
        for order in permutations(range(3))
    }

    assert selected_anchors == {4}


def test_node_renumbering_does_not_change_selected_anchor():
    nodes = np.array([0, 1, 2], dtype=np.int64)
    pixel_counts = np.array([3, 7, 7], dtype=np.int64)
    anchors = np.array([1, 8, 4], dtype=np.int64)
    representative = choose_representative(nodes, pixel_counts, anchors)

    renumbering = np.array([2, 0, 1], dtype=np.int64)
    renumbered_counts = pixel_counts[renumbering]
    renumbered_anchors = anchors[renumbering]
    renumbered_nodes = np.arange(nodes.size, dtype=np.int64)
    renumbered_representative = choose_representative(
        renumbered_nodes, renumbered_counts, renumbered_anchors
    )

    assert anchors[representative] == renumbered_anchors[renumbered_representative]
    assert anchors[representative] == 4


def test_duplicate_candidate_anchors_fail():
    nodes = np.array([0, 2], dtype=np.int64)
    pixel_counts = np.array([1, 4, 2], dtype=np.int64)
    anchors = np.array([5, 8, 5], dtype=np.int64)

    with pytest.raises(ValueError, match="anchors must be unique"):
        choose_representative(nodes, pixel_counts, anchors)


def test_empty_candidate_set_fails():
    nodes = np.array([], dtype=np.int64)
    pixel_counts = np.array([1], dtype=np.int64)
    anchors = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError, match="at least one candidate"):
        choose_representative(nodes, pixel_counts, anchors)


@pytest.mark.parametrize(
    ("case_name", "expected_representative"),
    [
        ("equal_size_tie", 0),
        ("size_beats_anchor", 2),
    ],
)
def test_named_canonical_selection_cases(case_name, expected_representative):
    case = CASES[case_name]
    candidates = np.array([0, 2], dtype=np.int64)

    representative = choose_representative(
        candidates, case["pixel_counts"], case["anchors"]
    )

    assert representative == expected_representative
