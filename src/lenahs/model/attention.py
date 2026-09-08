"""Exact graph-local parent attention calculations."""

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class ParentAttentionOutput:
    scores: tuple[Tensor, ...]
    probabilities: tuple[Tensor, ...]
    parent_message: Tensor


def readable_graph_attention(
    query: Tensor,
    key: Tensor,
    hidden_state: Tensor,
    graph_ptr: Tensor,
) -> ParentAttentionOutput:
    """Compute all-node parent attention separately for each packed graph."""

    scale = sqrt(query.shape[1])
    scores = []
    probabilities = []
    parent_messages = []

    # Score candidates and form expected parent states within each image
    for graph_index in range(graph_ptr.numel() - 1):
        start = int(graph_ptr[graph_index])
        stop = int(graph_ptr[graph_index + 1])
        graph_scores = query[start:stop] @ key[start:stop].T / scale
        graph_probabilities = F.softmax(graph_scores, dim=1)

        scores.append(graph_scores)
        probabilities.append(graph_probabilities)
        parent_messages.append(graph_probabilities @ hidden_state[start:stop])

    return ParentAttentionOutput(
        scores=tuple(scores),
        probabilities=tuple(probabilities),
        parent_message=torch.cat(parent_messages, dim=0),
    )
