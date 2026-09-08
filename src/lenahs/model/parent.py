"""Graph-local parent prediction heads."""

from torch import Tensor, nn

from lenahs.model.attention import ParentAttentionOutput, readable_graph_attention


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
