from math import sqrt

import torch
from torch.nn import functional as F

from lenahs.model import readable_graph_attention


def test_readable_attention_computes_independent_graph_blocks() -> None:
    query = torch.tensor(
        [[1.0, 2.0], [0.0, 1.0], [2.0, -1.0], [3.0, 4.0]],
    )
    key = torch.tensor(
        [[2.0, 0.0], [1.0, 1.0], [-1.0, 2.0], [5.0, -2.0]],
    )
    hidden_state = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [7.0, 8.0]],
    )
    graph_ptr = torch.tensor([0, 3, 4])

    output = readable_graph_attention(query, key, hidden_state, graph_ptr)

    expected_first_scores = query[:3] @ key[:3].T / sqrt(2)
    expected_first_probabilities = F.softmax(expected_first_scores, dim=1)
    assert [value.shape for value in output.scores] == [(3, 3), (1, 1)]
    torch.testing.assert_close(output.scores[0], expected_first_scores)
    torch.testing.assert_close(output.scores[1], torch.tensor([[7 / sqrt(2)]]))
    torch.testing.assert_close(output.probabilities[0], expected_first_probabilities)
    torch.testing.assert_close(output.probabilities[1], torch.ones(1, 1))
    torch.testing.assert_close(
        output.parent_message,
        torch.cat(
            (
                expected_first_probabilities @ hidden_state[:3],
                hidden_state[3:],
            )
        ),
    )


def test_one_node_attention_returns_its_current_hidden_state() -> None:
    hidden_state = torch.tensor([[2.0, -3.0, 4.0]])

    output = readable_graph_attention(
        query=torch.tensor([[1.0, 2.0, 3.0]]),
        key=torch.tensor([[4.0, 5.0, 6.0]]),
        hidden_state=hidden_state,
        graph_ptr=torch.tensor([0, 1]),
    )

    assert output.scores[0].shape == (1, 1)
    torch.testing.assert_close(output.probabilities[0], torch.ones(1, 1))
    torch.testing.assert_close(output.parent_message, hidden_state)
