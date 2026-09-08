"""Node encoding and recurrent RAG updates."""

import torch
from torch import Tensor, nn
from torch_geometric.nn.aggr import MaxAggregation, MinAggregation


class NodeEncoder(nn.Module):
    def __init__(self, input_width: int, hidden_width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_width, hidden_width)

    def forward(self, node_features: Tensor) -> Tensor:
        return torch.relu(self.linear(node_features))


class RecurrentRagProcessor(nn.Module):
    def __init__(self, hidden_width: int) -> None:
        super().__init__()
        self.hidden_width = hidden_width
        self.maximum = MaxAggregation()
        self.minimum = MinAggregation()
        self.gru = nn.GRUCell(4 * hidden_width, hidden_width)

    def forward(
        self,
        previous_hidden_state: Tensor,
        static_embedding: Tensor,
        edge_index: Tensor,
        previous_parent_message: Tensor,
    ) -> Tensor:
        source, destination = edge_index
        source_hidden = previous_hidden_state[source]
        node_count = previous_hidden_state.shape[0]

        # Aggregate previous hidden states over each local RAG neighbourhood
        local_maximum = self.maximum(
            source_hidden,
            index=destination,
            dim_size=node_count,
        )
        local_minimum = self.minimum(
            source_hidden,
            index=destination,
            dim_size=node_count,
        )

        recurrent_input = torch.cat(
            (
                previous_parent_message,
                local_maximum,
                local_minimum,
                static_embedding,
            ),
            dim=1,
        )
        return self.gru(recurrent_input, previous_hidden_state)
