from math import sqrt

import pytest
import torch
from torch import nn

from lenahs.model import (
    RGNN,
    AllNodeParentHead,
    NodeEncoder,
    ParentAttentionOutput,
    RecurrentRagProcessor,
    RGNNConfig,
    build_rgnn,
)


def _path_edges() -> torch.Tensor:
    return torch.tensor(
        [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]],
        dtype=torch.int64,
    )


class _CaptureGru(nn.Module):
    def __init__(self, hidden_width: int) -> None:
        super().__init__()
        self.hidden_width = hidden_width
        self.recurrent_input = None

    def forward(self, recurrent_input, hidden_state):
        self.recurrent_input = recurrent_input
        return recurrent_input[:, : self.hidden_width]


@pytest.mark.parametrize(
    ("edge_index", "expected_maximum", "expected_minimum"),
    [
        (
            _path_edges(),
            [[3.0, 4.0], [5.0, 4.0], [5.0, 4.0]],
            [[1.0, 2.0], [1.0, 0.0], [3.0, 0.0]],
        ),
        (
            torch.tensor(
                [
                    [0, 1, 0, 2, 0, 3, 0, 1, 2, 3],
                    [1, 0, 2, 0, 3, 0, 0, 1, 2, 3],
                ]
            ),
            [[7.0, 8.0], [3.0, 4.0], [5.0, 2.0], [7.0, 8.0]],
            [[1.0, 0.0], [1.0, 2.0], [1.0, 0.0], [1.0, 2.0]],
        ),
        (torch.tensor([[0], [0]]), [[1.0, 2.0]], [[1.0, 2.0]]),
    ],
    ids=["path", "star", "one_node"],
)
def test_local_aggregations_match_direct_neighborhood_extrema(
    edge_index,
    expected_maximum,
    expected_minimum,
) -> None:
    node_count = int(edge_index.max()) + 1
    hidden_state = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 0.0], [7.0, 8.0]][:node_count]
    )
    processor = RecurrentRagProcessor(hidden_width=2)
    capture = _CaptureGru(hidden_width=2)
    processor.gru = capture

    processor(
        previous_hidden_state=hidden_state,
        static_embedding=torch.zeros_like(hidden_state),
        edge_index=edge_index,
        previous_parent_message=torch.zeros_like(hidden_state),
    )

    assert capture.recurrent_input is not None
    torch.testing.assert_close(
        capture.recurrent_input[:, 2:4],
        torch.tensor(expected_maximum),
    )
    torch.testing.assert_close(
        capture.recurrent_input[:, 4:6],
        torch.tensor(expected_minimum),
    )


def test_one_step_agrees_with_independent_fixed_weight_calculation() -> None:
    model = RGNN(
        encoder=NodeEncoder(input_width=3, hidden_width=1),
        recurrent_processor=RecurrentRagProcessor(hidden_width=1),
        parent_head=AllNodeParentHead(hidden_width=1),
    )
    with torch.no_grad():
        model.encoder.linear.weight.copy_(torch.tensor([[1.0, 2.0, -1.0]]))
        model.encoder.linear.bias.fill_(0.1)
        processor = model.recurrent_processor
        processor.gru.weight_ih.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.2],
                    [0.0, 0.0, 0.0, -0.3],
                    [0.0, 0.0, 0.0, 0.5],
                ]
            )
        )
        processor.gru.weight_hh.zero_()
        processor.gru.bias_ih.copy_(torch.tensor([0.1, -0.2, 0.05]))
        processor.gru.bias_hh.copy_(torch.tensor([0.2, 0.3, -0.1]))
        model.parent_head.query_projection.weight.fill_(1.5)
        model.parent_head.query_projection.bias.fill_(0.2)
        model.parent_head.key_projection.weight.fill_(-0.5)
        model.parent_head.key_projection.bias.fill_(0.4)

    node_features = torch.tensor([[0.2, 0.3, 0.1], [0.7, 0.1, 0.4]])
    output = model(
        node_features,
        torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]]),
        torch.tensor([0, 2]),
        steps=1,
    )[0]

    static_embedding = torch.relu(
        node_features @ torch.tensor([[1.0], [2.0], [-1.0]]) + 0.1
    )
    reset = torch.sigmoid(0.2 * static_embedding + 0.1 + 0.2)
    update = torch.sigmoid(-0.3 * static_embedding - 0.2 + 0.3)
    candidate = torch.tanh(0.5 * static_embedding + 0.05 - 0.1 * reset)
    expected_hidden = (1.0 - update) * candidate
    query = 1.5 * expected_hidden + 0.2
    key = -0.5 * expected_hidden + 0.4
    expected_scores = query @ key.T / sqrt(1)
    expected_probabilities = torch.softmax(expected_scores, dim=1)

    torch.testing.assert_close(output.hidden_state, expected_hidden)
    torch.testing.assert_close(output.parent_scores[0], expected_scores)
    torch.testing.assert_close(output.parent_probabilities[0], expected_probabilities)
    torch.testing.assert_close(
        output.parent_message,
        expected_probabilities @ expected_hidden,
    )


class _CountingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, node_features):
        self.calls += 1
        return node_features.clone()


class _RecordingProcessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parent_messages = []
        self.static_embeddings = []

    def forward(
        self,
        previous_hidden_state,
        static_embedding,
        edge_index,
        previous_parent_message,
    ):
        self.parent_messages.append(previous_parent_message.clone())
        self.static_embeddings.append(static_embedding)
        return previous_hidden_state + static_embedding + previous_parent_message


class _IdentityParentHead(nn.Module):
    def forward(self, hidden_state, graph_ptr):
        probabilities = []
        for graph_index in range(graph_ptr.numel() - 1):
            size = int(graph_ptr[graph_index + 1] - graph_ptr[graph_index])
            probabilities.append(torch.eye(size, device=hidden_state.device))
        return ParentAttentionOutput(
            scores=tuple(probabilities),
            probabilities=tuple(probabilities),
            parent_message=hidden_state,
        )


def test_second_step_uses_new_parent_message_and_one_static_embedding() -> None:
    encoder = _CountingEncoder()
    processor = _RecordingProcessor()
    model = RGNN(
        encoder,
        processor,
        _IdentityParentHead(),
    )
    node_features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    outputs = model(
        node_features,
        torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([0, 2]),
        steps=2,
    )

    assert encoder.calls == 1
    assert processor.static_embeddings[0] is processor.static_embeddings[1]
    torch.testing.assert_close(
        processor.parent_messages[0], torch.zeros_like(node_features)
    )
    torch.testing.assert_close(processor.parent_messages[1], outputs[0].hidden_state)
    torch.testing.assert_close(outputs[1].hidden_state, 3 * node_features)


def test_edge_reordering_does_not_change_recurrent_outputs() -> None:
    torch.manual_seed(11)
    model = build_rgnn(RGNNConfig(hidden_width=4))
    node_features = torch.rand(3, 3)
    edge_index = _path_edges()

    original = model(node_features, edge_index, torch.tensor([0, 3]), steps=2)
    reordered = model(
        node_features,
        edge_index[:, torch.tensor([6, 2, 4, 0, 5, 3, 1])],
        torch.tensor([0, 3]),
        steps=2,
    )

    for original_step, reordered_step in zip(original, reordered, strict=True):
        torch.testing.assert_close(
            original_step.hidden_state, reordered_step.hidden_state
        )
        torch.testing.assert_close(
            original_step.parent_scores[0], reordered_step.parent_scores[0]
        )
        torch.testing.assert_close(
            original_step.parent_message, reordered_step.parent_message
        )


def test_heterogeneous_graph_shapes_and_probability_normalization() -> None:
    torch.manual_seed(12)
    model = build_rgnn(RGNNConfig(hidden_width=5))

    output = model(
        torch.rand(4, 3),
        torch.tensor([[0, 1, 1, 2, 0, 1, 2, 3], [1, 0, 2, 1, 0, 1, 2, 3]]),
        torch.tensor([0, 3, 4]),
        steps=1,
    )[0]

    assert output.hidden_state.shape == (4, 5)
    assert output.parent_message.shape == (4, 5)
    assert [scores.shape for scores in output.parent_scores] == [(3, 3), (1, 1)]
    for probabilities in output.parent_probabilities:
        torch.testing.assert_close(
            probabilities.sum(dim=1),
            torch.ones(probabilities.shape[0]),
        )
    torch.testing.assert_close(output.parent_probabilities[1], torch.ones(1, 1))


def test_gradients_reach_every_shared_module_through_recurrence() -> None:
    torch.manual_seed(13)
    model = build_rgnn(RGNNConfig(hidden_width=4))
    node_features = torch.rand(3, 3, requires_grad=True)

    outputs = model(
        node_features,
        _path_edges(),
        torch.tensor([0, 3]),
        steps=3,
    )
    loss = outputs[-1].parent_scores[0].square().sum()
    loss.backward()

    assert node_features.grad is not None
    assert torch.count_nonzero(node_features.grad)
    shared_modules = (model.encoder, model.recurrent_processor, model.parent_head)
    for module in shared_modules:
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad)


def test_model_requires_a_positive_recurrent_horizon() -> None:
    model = build_rgnn()

    with pytest.raises(ValueError, match="steps must be positive"):
        model(
            torch.rand(1, 3),
            torch.tensor([[0], [0]]),
            torch.tensor([0, 1]),
            steps=0,
        )


@pytest.mark.parametrize("steps", [-1, 0, True, 1.5, "2"])
def test_invalid_horizon_fails_before_encoding(steps) -> None:
    encoder = _CountingEncoder()
    model = RGNN(encoder, _RecordingProcessor(), _IdentityParentHead())
    error = ValueError if type(steps) is int else TypeError
    with pytest.raises(error, match="steps must be"):
        model(torch.ones(1, 2), torch.tensor([[0], [0]]), torch.tensor([0, 1]), steps)
    assert encoder.calls == 0


def test_three_steps_and_gradients_match_independent_gru_equations() -> None:
    model = build_rgnn(RGNNConfig(hidden_width=2)).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.linspace(
                    -0.2, 0.3, parameter.numel(), dtype=torch.float64
                ).reshape(parameter.shape)
            )
    features = torch.tensor(
        [[0.1, 0.4, 0.0], [0.8, 0.2, 0.4], [0.3, 0.4, 0.7], [0.5, 1.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edges = torch.tensor([[0, 1, 1, 2, 0, 1, 2, 3], [1, 0, 2, 1, 0, 1, 2, 3]])
    actual = model(features, edges, torch.tensor([0, 3, 4]), steps=3)
    encoder = model.encoder.linear
    embedding = torch.relu(features @ encoder.weight.T + encoder.bias)
    hidden = torch.zeros_like(embedding)
    message = torch.zeros_like(embedding)
    expected = []
    gru = model.recurrent_processor.gru
    for _ in range(3):
        maxima = []
        minima = []
        for node in range(4):
            neighbors = [
                int(source)
                for source, destination in edges.T
                if int(destination) == node
            ]
            values = torch.stack([hidden[neighbor] for neighbor in neighbors])
            maxima.append(values.max(dim=0).values)
            minima.append(values.min(dim=0).values)
        inputs = torch.cat(
            (message, torch.stack(maxima), torch.stack(minima), embedding), dim=1
        )
        input_reset, input_update, input_candidate = (
            inputs @ gru.weight_ih.T + gru.bias_ih
        ).chunk(3, dim=1)
        hidden_reset, hidden_update, hidden_candidate = (
            hidden @ gru.weight_hh.T + gru.bias_hh
        ).chunk(3, dim=1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_candidate + reset * hidden_candidate)
        hidden = (1 - update) * candidate + update * hidden
        query_layer = model.parent_head.query_projection
        key_layer = model.parent_head.key_projection
        query = hidden @ query_layer.weight.T + query_layer.bias
        key = hidden @ key_layer.weight.T + key_layer.bias
        scores = tuple(
            query[start:stop] @ key[start:stop].T / sqrt(2)
            for start, stop in [(0, 3), (3, 4)]
        )
        probabilities = tuple(torch.softmax(score, dim=1) for score in scores)
        message = torch.cat(
            [
                probability @ hidden[start:stop]
                for probability, (start, stop) in zip(
                    probabilities, [(0, 3), (3, 4)], strict=True
                )
            ]
        )
        expected.append((hidden, scores, probabilities, message))

    for actual_step, (hidden, scores, probabilities, message) in zip(
        actual, expected, strict=True
    ):
        torch.testing.assert_close(actual_step.hidden_state, hidden)
        torch.testing.assert_close(actual_step.parent_message, message)
        for actual_scores, expected_scores in zip(
            actual_step.parent_scores, scores, strict=True
        ):
            torch.testing.assert_close(actual_scores, expected_scores)
        for actual_probabilities, expected_probabilities in zip(
            actual_step.parent_probabilities, probabilities, strict=True
        ):
            torch.testing.assert_close(actual_probabilities, expected_probabilities)

    actual_loss = (
        sum(score.square().sum() for score in actual[-1].parent_scores)
        + actual[-1].parent_message.square().sum()
    )
    expected_loss = (
        sum(score.square().sum() for score in expected[-1][1])
        + expected[-1][3].square().sum()
    )
    inputs = (features, *model.parameters())
    actual_gradients = torch.autograd.grad(actual_loss, inputs, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected_loss, inputs)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, expected_gradient)
