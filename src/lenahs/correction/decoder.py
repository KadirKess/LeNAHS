"""Hard decoding of corrective pointers and actions into a hierarchical structure."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from lenahs.generation.targets import CorrectionAction


class DecodeFailureReason(StrEnum):
    MALFORMED_SHAPE = "malformed_shape"
    INVALID_DTYPE = "invalid_dtype"
    OUT_OF_RANGE_POINTER = "out_of_range_pointer"
    UNKNOWN_ACTION = "unknown_action"
    ROOT_COUNT = "root_count"
    NONTRIVIAL_CYCLE = "nontrivial_cycle"
    INVALID_ROOT_ACTION = "invalid_root_action"


class CorrectionDecodeError(ValueError):
    def __init__(self, reason: DecodeFailureReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class DecodedCorrection:
    # Output nodes are groups of observed nodes with their own identifiers,
    # not clean identifiers. Each group is defined by one retained observed node.
    representatives: NDArray[np.int64]
    observed_to_output: NDArray[np.int64]
    parents: NDArray[np.int64]
    increments: NDArray[np.int64]
    depths: NDArray[np.int64]
    pixel_depth: NDArray[np.int64]


def decode_correction(
    corrective_parents: NDArray[np.integer],
    actions: NDArray[np.integer],
    observed_label_map: NDArray[np.integer],
) -> DecodedCorrection:
    """Decode hard predictions or oracle targets for one observed image."""

    # The label map is trusted RAG data, so it fixes the node count that the
    # predictions are checked against
    observed_node_count = int(observed_label_map.max()) + 1

    root = _validate_predictions(corrective_parents, actions, observed_node_count)
    representative_for_node = _resolve_merge_representatives(
        corrective_parents, actions
    )
    representatives, observed_to_output, parents, increments = _contract_tree(
        corrective_parents, actions, representative_for_node
    )
    depths = _accumulate_depths(parents, increments, observed_to_output[root])

    # Every pixel of an observed zone gets the depth of the group it joined
    pixel_depth = depths[observed_to_output[observed_label_map]]

    return DecodedCorrection(
        representatives=representatives,
        observed_to_output=observed_to_output,
        parents=parents,
        increments=increments,
        depths=depths,
        pixel_depth=pixel_depth,
    )


def _validate_predictions(
    corrective_parents: NDArray[np.integer],
    actions: NDArray[np.integer],
    observed_node_count: int,
) -> int:
    n = observed_node_count
    if corrective_parents.ndim != 1 or actions.ndim != 1:
        raise CorrectionDecodeError(
            DecodeFailureReason.MALFORMED_SHAPE,
            message="Parents and actions must be 1D arrays.",
        )
    if corrective_parents.size != n or actions.size != n:
        raise CorrectionDecodeError(
            DecodeFailureReason.MALFORMED_SHAPE,
            message=f"Parents and actions must have one entry per observed node ({n}).",
        )
    if n == 0:
        raise CorrectionDecodeError(
            DecodeFailureReason.MALFORMED_SHAPE,
            message="Observed node count must be greater than zero.",
        )

    # NumPy does not count bool as an integer dtype, so masks fail here too
    if not np.issubdtype(corrective_parents.dtype, np.integer) or not np.issubdtype(
        actions.dtype, np.integer
    ):
        raise CorrectionDecodeError(
            DecodeFailureReason.INVALID_DTYPE,
            message="Parents and actions must be integer arrays.",
        )

    out_of_range = np.flatnonzero((corrective_parents < 0) | (corrective_parents >= n))
    if out_of_range.size > 0:
        node = out_of_range[0]
        raise CorrectionDecodeError(
            DecodeFailureReason.OUT_OF_RANGE_POINTER,
            message=f"Parent {corrective_parents[node]} of node {node} is outside 0 to {n - 1}.",
        )

    unknown = np.flatnonzero(~np.isin(actions, list(CorrectionAction)))
    if unknown.size > 0:
        node = unknown[0]
        raise CorrectionDecodeError(
            DecodeFailureReason.UNKNOWN_ACTION,
            message=f"Action {actions[node]} of node {node} is not MERGE (0), KEEP_ZERO (1) or KEEP_ONE (2).",
        )

    roots = np.flatnonzero(corrective_parents == np.arange(n))
    if roots.size != 1:
        raise CorrectionDecodeError(
            DecodeFailureReason.ROOT_COUNT,
            message=f"Expected exactly one self-parent root, found {roots.size}.",
        )
    root = int(roots[0])
    if actions[root] != CorrectionAction.KEEP_ZERO:
        raise CorrectionDecodeError(
            DecodeFailureReason.INVALID_ROOT_ACTION,
            message="Root node must have action KEEP_ZERO (1).",
        )

    # State 0 is unvisited, 1 is on the current chain and 2 is known to reach the
    # root. Reaching a 1 again means the chain looped back on itself.
    state = np.zeros(n, dtype=np.int8)
    state[root] = 2
    for start in range(n):
        if state[start] == 2:
            continue

        path = []
        node = start
        while state[node] == 0:
            path.append(node)
            state[node] = 1
            node = corrective_parents[node]

        if state[node] == 1:
            raise CorrectionDecodeError(
                DecodeFailureReason.NONTRIVIAL_CYCLE,
                message=f"Parent chain from node {start} loops back to node {node}.",
            )

        state[path] = 2

    return root


def _resolve_merge_representatives(
    corrective_parents: NDArray[np.integer],
    actions: NDArray[np.integer],
) -> NDArray[np.int64]:
    n = corrective_parents.size

    # Retained nodes are their own representative, merge nodes start unresolved
    representative_for_node = np.where(
        actions != CorrectionAction.MERGE, np.arange(n, dtype=np.int64), -1
    )

    for start in range(n):
        if representative_for_node[start] != -1:
            continue

        # In theory, the model should make noise nodes merge directly to a
        # representative, but we don't have any guarantee of that.
        path = []
        node = start
        while representative_for_node[node] == -1:
            path.append(node)
            node = corrective_parents[node]

        representative_for_node[path] = representative_for_node[node]

    return representative_for_node


def _contract_tree(
    corrective_parents: NDArray[np.integer],
    actions: NDArray[np.integer],
    representative_for_node: NDArray[np.int64],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    # Output identifiers follow increasing retained observed identifier
    representatives = np.flatnonzero(actions != CorrectionAction.MERGE)
    output_node_count = representatives.size

    # Output id of each retained node, left at -1 for merge nodes
    output_id_for_representative = np.full(corrective_parents.size, -1, dtype=np.int64)
    output_id_for_representative[representatives] = np.arange(output_node_count)

    observed_to_output = output_id_for_representative[representative_for_node]

    # A retained node can point at a merge node, so its pointer goes through
    # observed_to_output. Copying the observed pointer would mix identifier spaces.
    parents = observed_to_output[corrective_parents[representatives]]
    increments = (actions[representatives] == CorrectionAction.KEEP_ONE).astype(
        np.int64
    )

    return representatives, observed_to_output, parents, increments


def _accumulate_depths(
    parents: NDArray[np.int64],
    increments: NDArray[np.int64],
    root: int,
) -> NDArray[np.int64]:
    output_node_count = parents.size
    children = [[] for _ in range(output_node_count)]

    # The root's self-link is not a child edge
    for node in range(output_node_count):
        parent = parents[node]
        if parent != node:
            children[parent].append(node)

    # Depth comes from increments, so a KEEP_ZERO child sits at its parent's
    # depth. A parent's depth is set before its children are pushed.
    depths = np.zeros(output_node_count, dtype=np.int64)
    pending = [root]
    while pending:
        node = pending.pop()
        for child in children[node]:
            depths[child] = depths[node] + increments[child]
            pending.append(child)

    return depths
