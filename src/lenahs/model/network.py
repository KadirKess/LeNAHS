"""Recurrent RGNN orchestration and per-step outputs."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lenahs.model.direction import DirectionGradientMode, DirectionHeadInput


@dataclass(frozen=True)
class RGNNStep:
    hidden_state: Tensor
    parent_scores: tuple[Tensor, ...]
    parent_probabilities: tuple[Tensor, ...]
    parent_message: Tensor
    direction_logits: Tensor | None


class RGNN(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        recurrent_processor: nn.Module,
        parent_head: nn.Module,
        direction_head: nn.Module | None,
        *,
        direction_gradients: DirectionGradientMode = DirectionGradientMode.SHARED_MODEL,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.recurrent_processor = recurrent_processor
        self.parent_head = parent_head
        self.direction_head = direction_head
        self.direction_gradients = DirectionGradientMode(direction_gradients)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        graph_ptr: Tensor,
        steps: int,
    ) -> tuple[RGNNStep, ...]:
        if steps <= 0:
            raise ValueError("steps must be positive.")

        static_embedding = self.encoder(node_features)

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
            direction_logits = None
            if self.direction_head is not None:
                direction_features = DirectionHeadInput(
                    hidden_state=hidden_state,
                    parent_message=parent_message,
                )
                if self.direction_gradients == DirectionGradientMode.HEAD_ONLY:
                    direction_features = DirectionHeadInput(
                        hidden_state=hidden_state.detach(),
                        parent_message=parent_message.detach(),
                    )
                direction_logits = self.direction_head(direction_features)
            outputs.append(
                RGNNStep(
                    hidden_state=hidden_state,
                    parent_scores=parent_output.scores,
                    parent_probabilities=parent_output.probabilities,
                    parent_message=parent_message,
                    direction_logits=direction_logits,
                )
            )

        return tuple(outputs)
