"""Predict merge and depth actions from corrective destination context."""

import torch
from torch import Tensor, nn


class MergeDepthHead(nn.Module):
    def __init__(self, hidden_width: int) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(2 * hidden_width, hidden_width)
        self.output_layer = nn.Linear(hidden_width, 3)

    def forward(
        self,
        hidden_state: Tensor,
        expected_destination_state: Tensor,
    ) -> Tensor:
        head_input = torch.cat((hidden_state, expected_destination_state), dim=1)
        hidden = torch.relu(self.hidden_layer(head_input))
        return self.output_layer(hidden)
