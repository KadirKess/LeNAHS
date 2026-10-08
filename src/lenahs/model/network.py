"""Recurrent RGNN orchestration and per-step outputs."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ObservedStep:
    hidden_state: Tensor
    parent_scores: tuple[Tensor, ...]
    parent_probabilities: tuple[Tensor, ...]
    parent_message: Tensor


class RGNN(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        recurrent_processor: nn.Module,
        parent_head: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.recurrent_processor = recurrent_processor
        self.parent_head = parent_head

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        graph_ptr: Tensor,
        steps: int,
    ) -> tuple[ObservedStep, ...]:
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer.")
        if steps <= 0:
            raise ValueError("steps must be positive.")

        static_embedding = self.encoder(node_features)

        return self._run_steps(static_embedding, edge_index, graph_ptr, steps)

    def _run_steps(
        self,
        static_embedding: Tensor,
        edge_index: Tensor,
        graph_ptr: Tensor,
        steps: int,
    ) -> tuple[ObservedStep, ...]:
        # Start without a hidden state or an initial parent guess
        hidden_state = torch.zeros_like(static_embedding)
        parent_message = torch.zeros_like(static_embedding)
        outputs = []

        for _ in range(steps):
            hidden_state = self.recurrent_processor(
                hidden_state,
                static_embedding,
                edge_index,
                parent_message,
            )
            parent_output = self.parent_head(hidden_state, graph_ptr)

            parent_message = parent_output.parent_message
            outputs.append(
                ObservedStep(
                    hidden_state=hidden_state,
                    parent_scores=parent_output.scores,
                    parent_probabilities=parent_output.probabilities,
                    parent_message=parent_message,
                )
            )

        return tuple(outputs)
