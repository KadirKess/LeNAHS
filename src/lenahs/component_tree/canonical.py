import numpy as np
from numpy.typing import NDArray


def choose_representative(
    nodes: NDArray[np.integer],
    pixel_counts: NDArray[np.integer],
    anchors: NDArray[np.integer],
) -> int:
    """Choose the flat zone that represents a component."""

    if nodes.size == 0:
        raise ValueError("nodes must contain at least one candidate.")

    candidate_counts = pixel_counts[nodes]
    candidate_anchors = anchors[nodes]
    if np.unique(candidate_anchors).size != candidate_anchors.size:
        raise ValueError("Candidate anchors must be unique.")

    best_index = 0
    for index in range(1, nodes.size):
        larger = candidate_counts[index] > candidate_counts[best_index]
        same_size = candidate_counts[index] == candidate_counts[best_index]
        earlier = candidate_anchors[index] < candidate_anchors[best_index]
        if larger or (same_size and earlier):
            best_index = index

    return int(nodes[best_index])
