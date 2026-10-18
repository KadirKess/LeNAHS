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
    query: Tensor
    key: Tensor
    log_normalizers: Tensor


def readable_graph_attention(
    query: Tensor,
    key: Tensor,
    hidden_state: Tensor,
    graph_ptr: Tensor,
) -> ParentAttentionOutput:
    """Compute all-node parent attention separately for each packed graph."""

    # Scaling the queries once is cheaper than scaling every n by n score block
    scaled_query = query / sqrt(query.shape[1])
    scores = []
    probabilities = []
    parent_messages = []
    log_normalizers = []

    # Score candidates and form expected parent states within each image. Reading
    # the bounds once avoids one device synchronization per graph.
    bounds = graph_ptr.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        graph_scores = scaled_query[start:stop] @ key[start:stop].T
        graph_probabilities = F.softmax(graph_scores, dim=1)

        scores.append(graph_scores)
        probabilities.append(graph_probabilities)
        parent_messages.append(graph_probabilities @ hidden_state[start:stop])
        log_normalizers.append(torch.logsumexp(graph_scores, dim=1))

    return ParentAttentionOutput(
        scores=tuple(scores),
        probabilities=tuple(probabilities),
        parent_message=torch.cat(parent_messages, dim=0),
        query=query,
        key=key,
        log_normalizers=torch.cat(log_normalizers, dim=0),
    )
