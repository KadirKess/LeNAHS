import inspect

import numpy as np
import pytest

from lenahs.component_tree import ComponentTreeKind
from lenahs.correction.decoder import (
    CorrectionDecodeError,
    DecodeFailureReason,
    decode_correction,
)
from lenahs.generation import CorrectionAction, build_generation_sample

MERGE = CorrectionAction.MERGE
KEEP_ZERO = CorrectionAction.KEEP_ZERO
KEEP_ONE = CorrectionAction.KEEP_ONE


def _ints(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _row(node_count: int) -> np.ndarray:
    # One pixel per observed node, so pixel depth reads off node depth directly
    return np.arange(node_count, dtype=np.int64).reshape(1, node_count)


def _assert_decoded(decoded, **expected) -> None:
    for name, values in expected.items():
        actual = getattr(decoded, name)
        assert actual.dtype == np.int64, name
        np.testing.assert_array_equal(actual, _ints(values), err_msg=name)


def _assert_fails(reason, parents, actions, label_map) -> None:
    with pytest.raises(CorrectionDecodeError) as caught:
        decode_correction(parents, actions, label_map)
    assert caught.value.reason == reason


# The five-node contraction worked through in the decoder guide
WORKED_PARENTS = _ints([0, 0, 1, 2, 3])
WORKED_ACTIONS = _ints([KEEP_ZERO, KEEP_ONE, MERGE, MERGE, KEEP_ZERO])


def test_single_node_is_one_root_group_at_depth_zero():
    label_map = np.zeros((3, 4), dtype=np.int64)

    decoded = decode_correction(_ints([0]), _ints([KEEP_ZERO]), label_map)

    _assert_decoded(
        decoded,
        representatives=[0],
        observed_to_output=[0],
        parents=[0],
        increments=[0],
        depths=[0],
        pixel_depth=np.zeros((3, 4)),
    )


def test_equal_level_link_keeps_separate_node_at_parent_depth():
    # Max-tree of [10, 20, 10]. Node 2 is a separate zone at the root level.
    decoded = decode_correction(
        _ints([0, 0, 0]), _ints([KEEP_ZERO, KEEP_ONE, KEEP_ZERO]), _row(3)
    )

    _assert_decoded(
        decoded,
        representatives=[0, 1, 2],
        observed_to_output=[0, 1, 2],
        parents=[0, 0, 0],
        increments=[0, 1, 0],
        depths=[0, 1, 0],
        pixel_depth=[[0, 1, 0]],
    )


def test_worked_contraction_matches_every_hand_calculated_array():
    decoded = decode_correction(WORKED_PARENTS, WORKED_ACTIONS, _row(5))

    _assert_decoded(
        decoded,
        representatives=[0, 1, 4],
        observed_to_output=[0, 1, 1, 1, 2],
        parents=[0, 0, 1],
        increments=[0, 1, 0],
        depths=[0, 1, 1],
        pixel_depth=[[0, 1, 1, 1, 1]],
    )


def test_nonzero_root_maps_parents_through_output_identifiers():
    # Copying observed identifiers or assuming root zero both fail here
    decoded = decode_correction(
        _ints([2, 0, 2]), _ints([MERGE, KEEP_ONE, KEEP_ZERO]), _row(3)
    )

    _assert_decoded(
        decoded,
        representatives=[1, 2],
        observed_to_output=[1, 0, 1],
        parents=[1, 1],
        increments=[1, 0],
        depths=[1, 0],
        pixel_depth=[[0, 1, 0]],
    )


def test_several_fragments_join_a_nonadjacent_representative():
    decoded = decode_correction(
        _ints([0, 0, 0]), _ints([KEEP_ZERO, MERGE, MERGE]), _row(3)
    )

    _assert_decoded(
        decoded,
        representatives=[0],
        observed_to_output=[0, 0, 0],
        parents=[0],
        increments=[0],
        depths=[0],
        pixel_depth=[[0, 0, 0]],
    )


# Recursion would exceed Python's default limit, and rescanning each chain
# suffix would make these quadratic in the chain length
LONG_CHAIN = 20_000


def _chain_parents(node_count: int) -> np.ndarray:
    parents = np.arange(-1, node_count - 1, dtype=np.int64)
    parents[0] = 0
    return parents


def test_long_merge_chain_contracts_to_the_root():
    actions = np.full(LONG_CHAIN, MERGE, dtype=np.int64)
    actions[0] = KEEP_ZERO

    decoded = decode_correction(_chain_parents(LONG_CHAIN), actions, _row(LONG_CHAIN))

    _assert_decoded(
        decoded,
        representatives=[0],
        observed_to_output=np.zeros(LONG_CHAIN),
        depths=[0],
        pixel_depth=np.zeros((1, LONG_CHAIN)),
    )


def test_long_retained_chain_reaches_depth_n_minus_one():
    actions = np.full(LONG_CHAIN, KEEP_ONE, dtype=np.int64)
    actions[0] = KEEP_ZERO

    decoded = decode_correction(_chain_parents(LONG_CHAIN), actions, _row(LONG_CHAIN))

    _assert_decoded(
        decoded,
        depths=np.arange(LONG_CHAIN),
        pixel_depth=np.arange(LONG_CHAIN).reshape(1, LONG_CHAIN),
    )


def test_narrower_integer_inputs_decode_to_int64_results():
    decoded = decode_correction(
        WORKED_PARENTS.astype(np.int32),
        WORKED_ACTIONS.astype(np.uint8),
        _row(5).astype(np.int32),
    )

    _assert_decoded(decoded, depths=[0, 1, 1], pixel_depth=[[0, 1, 1, 1, 1]])


def test_non_contiguous_label_map_views_use_their_own_indexing():
    # Node depths for the worked example are [0, 1, 1, 1, 1]
    stored = np.array([[0, 1], [2, 3], [4, 4]], dtype=np.int64)
    transposed = stored.T
    reversed_row = np.array([[0, 1, 2, 3, 4, 4]], dtype=np.int64)[:, ::-1]
    assert not transposed.flags.c_contiguous
    assert not reversed_row.flags.c_contiguous

    np.testing.assert_array_equal(
        decode_correction(WORKED_PARENTS, WORKED_ACTIONS, transposed).pixel_depth,
        [[0, 1, 1], [1, 1, 1]],
    )
    np.testing.assert_array_equal(
        decode_correction(WORKED_PARENTS, WORKED_ACTIONS, reversed_row).pixel_depth,
        [[1, 1, 1, 1, 1, 0]],
    )


def test_disconnected_output_group_is_a_valid_decode():
    # Nodes 1 and 3 are not adjacent in the row but form one output group
    decoded = decode_correction(
        _ints([0, 0, 0, 1]), _ints([KEEP_ZERO, KEEP_ONE, KEEP_ZERO, MERGE]), _row(4)
    )

    _assert_decoded(
        decoded,
        observed_to_output=[0, 1, 2, 1],
        pixel_depth=[[0, 1, 0, 1]],
    )


def _pixel_partition(observed_to_output, label_map) -> set[frozenset]:
    groups = {}
    for pixel, group in enumerate(observed_to_output[label_map].ravel()):
        groups.setdefault(int(group), set()).add(pixel)
    return {frozenset(pixels) for pixels in groups.values()}


def test_renumbering_observed_nodes_preserves_pixel_groups_and_depth():
    label_map = np.array([[0, 1, 1, 2], [3, 3, 4, 4]], dtype=np.int64)
    # new_id[old] is the new identifier of old observed node old
    new_id = _ints([3, 0, 4, 2, 1])
    renumbered_parents = np.empty(5, dtype=np.int64)
    renumbered_actions = np.empty(5, dtype=np.int64)
    renumbered_parents[new_id] = new_id[WORKED_PARENTS]
    renumbered_actions[new_id] = WORKED_ACTIONS
    renumbered_labels = new_id[label_map]

    original = decode_correction(WORKED_PARENTS, WORKED_ACTIONS, label_map)
    renumbered = decode_correction(
        renumbered_parents, renumbered_actions, renumbered_labels
    )

    np.testing.assert_array_equal(renumbered.pixel_depth, original.pixel_depth)
    assert _pixel_partition(
        renumbered.observed_to_output, renumbered_labels
    ) == _pixel_partition(original.observed_to_output, label_map)


def test_decoder_reads_only_pointers_actions_and_observed_labels():
    # Intensities, clean records and noise settings cannot reach the decoder
    parameters = inspect.signature(decode_correction).parameters
    assert list(parameters) == ["corrective_parents", "actions", "observed_label_map"]


@pytest.mark.parametrize(
    ("kind", "image", "expected_depth"),
    [
        (ComponentTreeKind.MAX_TREE, [[10, 20, 10]], [[0, 1, 0]]),
        # The min-tree root is the brightest level, and darker zones lie deeper
        (ComponentTreeKind.MIN_TREE, [[20, 10, 20]], [[0, 1, 0]]),
        (ComponentTreeKind.MIN_TREE, [[10, 20, 10]], [[1, 0, 1]]),
        (ComponentTreeKind.MAX_TREE, [[10, 20], [30, 20]], [[0, 1], [2, 1]]),
    ],
)
def test_perfect_clean_targets_decode_to_hand_calculated_depth(
    kind, image, expected_depth
):
    # The clean observation is its own clean reference. Its depth counts strict
    # level changes along the tree, one per change whatever its size.
    sample = build_generation_sample(np.asarray(image, dtype=np.uint8), kind)
    targets = sample.correction_targets

    decoded = decode_correction(
        targets.corrective_parents,
        targets.actions,
        sample.observed.graph.label_map,
    )

    np.testing.assert_array_equal(decoded.pixel_depth, expected_depth)


VALID_ROW = _row(3)
VALID_PARENTS = _ints([0, 0, 1])
VALID_ACTIONS = _ints([KEEP_ZERO, KEEP_ONE, MERGE])


def test_valid_reference_inputs_for_the_failure_cases_decode():
    decode_correction(VALID_PARENTS, VALID_ACTIONS, VALID_ROW)


@pytest.mark.parametrize(
    ("parents", "actions"),
    [
        (VALID_PARENTS.reshape(1, 3), VALID_ACTIONS),
        (VALID_PARENTS, VALID_ACTIONS.reshape(3, 1)),
        (VALID_PARENTS[:2], VALID_ACTIONS),
        (VALID_PARENTS, VALID_ACTIONS[:2]),
        (_ints([0, 0, 1, 2]), _ints([KEEP_ZERO, KEEP_ONE, MERGE, MERGE])),
    ],
    ids=["parents_2d", "actions_2d", "short_parents", "short_actions", "both_long"],
)
def test_rejects_malformed_shapes(parents, actions):
    _assert_fails(DecodeFailureReason.MALFORMED_SHAPE, parents, actions, VALID_ROW)


@pytest.mark.parametrize(
    ("parents", "actions"),
    [
        (VALID_PARENTS.astype(np.float64), VALID_ACTIONS),
        (VALID_PARENTS, VALID_ACTIONS.astype(np.float32)),
        (np.array([True, False, False]), VALID_ACTIONS),
        (VALID_PARENTS, np.array([True, True, False])),
    ],
    ids=["float_parents", "float_actions", "bool_parents", "bool_actions"],
)
def test_rejects_non_integer_predictions(parents, actions):
    _assert_fails(DecodeFailureReason.INVALID_DTYPE, parents, actions, VALID_ROW)


@pytest.mark.parametrize("bad_pointer", [-1, 3, 1_000])
def test_rejects_out_of_range_pointers(bad_pointer):
    parents = VALID_PARENTS.copy()
    parents[2] = bad_pointer
    _assert_fails(
        DecodeFailureReason.OUT_OF_RANGE_POINTER, parents, VALID_ACTIONS, VALID_ROW
    )


@pytest.mark.parametrize("bad_action", [-1, 3])
def test_rejects_unknown_actions(bad_action):
    actions = VALID_ACTIONS.copy()
    actions[2] = bad_action
    _assert_fails(DecodeFailureReason.UNKNOWN_ACTION, VALID_PARENTS, actions, VALID_ROW)


@pytest.mark.parametrize(
    "parents",
    [_ints([1, 2, 0]), _ints([0, 1, 0]), _ints([0, 1, 2])],
    ids=["cycle_without_root", "two_roots", "three_roots"],
)
def test_rejects_anything_but_one_self_parent(parents):
    actions = _ints([KEEP_ZERO, KEEP_ZERO, KEEP_ZERO])
    _assert_fails(DecodeFailureReason.ROOT_COUNT, parents, actions, VALID_ROW)


@pytest.mark.parametrize("root_action", [MERGE, KEEP_ONE])
def test_rejects_root_that_is_not_keep_zero(root_action):
    actions = VALID_ACTIONS.copy()
    actions[0] = root_action
    _assert_fails(
        DecodeFailureReason.INVALID_ROOT_ACTION, VALID_PARENTS, actions, VALID_ROW
    )


@pytest.mark.parametrize(
    "parents",
    [_ints([0, 2, 1]), _ints([0, 2, 3, 1]), _ints([0, 2, 3, 1, 1])],
    ids=["two_cycle", "three_cycle", "cycle_with_tail"],
)
def test_rejects_cycle_beside_a_valid_root(parents):
    node_count = len(parents)
    actions = np.full(node_count, KEEP_ONE, dtype=np.int64)
    actions[0] = KEEP_ZERO
    _assert_fails(
        DecodeFailureReason.NONTRIVIAL_CYCLE, parents, actions, _row(node_count)
    )
