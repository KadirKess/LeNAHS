"""Noise-direction heads and their shared input contract."""

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor, nn


class DirectionHeadKind(StrEnum):
    NONE = "none"
    PARENT_CONTEXT = "parent_context"
    NODE_ONLY = "node_only"


class DirectionGradientMode(StrEnum):
    SHARED_MODEL = "shared_model"
    HEAD_ONLY = "head_only"


@dataclass(frozen=True)
class DirectionHeadInput:
    hidden_state: Tensor
    parent_message: Tensor


class ParentContextDirectionHead(nn.Module):
    def __init__(self, hidden_width: int) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(2 * hidden_width, hidden_width)
        self.output_layer = nn.Linear(hidden_width, 3)

    def forward(self, features: DirectionHeadInput) -> Tensor:
        head_input = torch.cat(
            (features.hidden_state, features.parent_message),
            dim=1,
        )
        hidden = torch.relu(self.hidden_layer(head_input))
        return self.output_layer(hidden)


class NodeOnlyDirectionHead(nn.Module):
    def __init__(self, hidden_width: int) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(hidden_width, hidden_width)
        self.output_layer = nn.Linear(hidden_width, 3)

    def forward(self, features: DirectionHeadInput) -> Tensor:
        hidden = torch.relu(self.hidden_layer(features.hidden_state))
        return self.output_layer(hidden)
