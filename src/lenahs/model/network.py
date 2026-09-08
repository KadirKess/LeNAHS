from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.nn.aggr import MaxAggregation, MinAggregation

from lenahs.model.adapter import STATIC_FEATURE_NAMES
from lenahs.model.attention import ParentAttentionOutput, readable_graph_attention


@dataclass(frozen=True)
class BaselineModelConfig:
    input_width: int = len(STATIC_FEATURE_NAMES)
    hidden_width: int = 32


@dataclass(frozen=True)
class RecurrentStepOutput:
    hidden_state: Tensor
    parent_scores: tuple[Tensor, ...]
    parent_probabilities: tuple[Tensor, ...]
    parent_message: Tensor


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


class AllNodeParentHead(nn.Module):
    def __init__(self, hidden_width: int) -> None:
        super().__init__()
        self.query_projection = nn.Linear(hidden_width, hidden_width)
        self.key_projection = nn.Linear(hidden_width, hidden_width)

    def forward(
        self,
        hidden_state: Tensor,
        graph_ptr: Tensor,
    ) -> ParentAttentionOutput:
        query = self.query_projection(hidden_state)
        key = self.key_projection(hidden_state)
        return readable_graph_attention(query, key, hidden_state, graph_ptr)


class RecurrentParentModel(nn.Module):
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
    ) -> tuple[RecurrentStepOutput, ...]:
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
            outputs.append(
                RecurrentStepOutput(
                    hidden_state=hidden_state,
                    parent_scores=parent_output.scores,
                    parent_probabilities=parent_output.probabilities,
                    parent_message=parent_message,
                )
            )

        return tuple(outputs)


def build_baseline_model(
    config: BaselineModelConfig = BaselineModelConfig(),
) -> RecurrentParentModel:
    if config.input_width <= 0:
        raise ValueError("input_width must be positive.")
    if config.hidden_width <= 0:
        raise ValueError("hidden_width must be positive.")

    return RecurrentParentModel(
        encoder=NodeEncoder(config.input_width, config.hidden_width),
        recurrent_processor=RecurrentRagProcessor(config.hidden_width),
        parent_head=AllNodeParentHead(config.hidden_width),
    )
