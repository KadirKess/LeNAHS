import json
from math import sqrt

import pytest
import torch
from torch import nn

from lenahs.model import (
    AllNodeParentHead,
    CorrectionModel,
    CorrectionModelConfig,
    MergeDepthHead,
    ObservedStep,
    RGNNConfig,
    build_correction_model,
    build_rgnn,
)


def _path_edges():
    return torch.tensor([[0, 1, 1, 2, 0, 1, 2, 3], [1, 0, 2, 1, 0, 1, 2, 3]])


class _FixedObserved(nn.Module):
    def __init__(self, hidden_state, parent_message):
        super().__init__()
        self.hidden_state = hidden_state
        self.encoder = lambda features: self.hidden_state
        self.parent_message = parent_message

    def _run_steps(self, embedding, edge_index, graph_ptr, steps):
        blocks = tuple(
            self.hidden_state.new_zeros(int(stop - start), int(stop - start))
            for start, stop in zip(graph_ptr[:-1], graph_ptr[1:], strict=True)
        )
        return (
            ObservedStep(
                self.hidden_state,
                blocks,
                blocks,
                self.parent_message,
                parent_query=self.hidden_state,
                parent_key=self.hidden_state,
                parent_log_normalizers=self.hidden_state.new_zeros(
                    self.hidden_state.shape[0]
                ),
            ),
        )


def _fixed_readout():
    hidden = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.0, 4.0]], dtype=torch.float64)
    parent_head = AllNodeParentHead(2).double()
    action_head = MergeDepthHead(2).double()
    with torch.no_grad():
        parent_head.query_projection.weight.copy_(
            torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
        )
        parent_head.query_projection.bias.copy_(torch.tensor([0.1, 0.2]))
        parent_head.key_projection.weight.copy_(torch.tensor([[0.5, -1.0], [2.0, 1.0]]))
        parent_head.key_projection.bias.copy_(torch.tensor([0.3, -0.2]))
        action_head.hidden_layer.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.5, 0.0], [0.0, -1.0, 0.0, 2.0]])
        )
        action_head.hidden_layer.bias.copy_(torch.tensor([0.25, 0.5]))
        action_head.output_layer.weight.copy_(
            torch.tensor([[1.0, 2.0], [-1.0, 1.0], [0.5, -0.5]])
        )
        action_head.output_layer.bias.copy_(torch.tensor([0.0, 0.1, 0.2]))
    return CorrectionModel(
        _FixedObserved(hidden, torch.zeros_like(hidden)), parent_head, action_head
    )


def test_corrective_readout_matches_hand_calculated_unequal_graphs():
    model = _fixed_readout()
    output = model(
        torch.zeros(3, 3),
        torch.tensor([[0, 1, 2], [0, 1, 2]]),
        torch.tensor([0, 2, 3]),
        observed_steps=1,
    )
    assert len(output.observed_steps) == 1
    assert len(output.correction_steps) == 1
    correction = output.correction_steps[0]
    hidden = model.observed_model.hidden_state
    query = torch.tensor([[5.1, 0.2], [1.1, -3.3], [8.1, 2.2]], dtype=torch.float64)
    key = torch.tensor([[-1.2, 3.8], [2.8, 4.8], [-3.7, 3.8]], dtype=torch.float64)
    scores = (query[:2] @ key[:2].T / sqrt(2), query[2:] @ key[2:].T / sqrt(2))
    probabilities = tuple(torch.softmax(score, dim=1) for score in scores)
    destination = torch.cat((probabilities[0] @ hidden[:2], hidden[2:]))
    activation = torch.relu(
        torch.stack(
            (
                hidden[:, 0] + 0.5 * destination[:, 0] + 0.25,
                -hidden[:, 1] + 2 * destination[:, 1] + 0.5,
            ),
            dim=1,
        )
    )
    actions = torch.stack(
        (
            activation[:, 0] + 2 * activation[:, 1],
            -activation[:, 0] + activation[:, 1] + 0.1,
            0.5 * activation[:, 0] - 0.5 * activation[:, 1] + 0.2,
        ),
        dim=1,
    )
    assert correction.hidden_state is output.observed_steps[-1].hidden_state
    for actual, expected in zip(correction.corrective_scores, scores, strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(
        correction.corrective_probabilities, probabilities, strict=True
    ):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(correction.expected_destination_state, destination)
    torch.testing.assert_close(correction.action_logits, actions)


def test_actions_read_corrective_context_and_ignore_observed_parent_message():
    model = _fixed_readout()
    features = torch.zeros(3, 3)
    edges = torch.tensor([[0, 1, 2], [0, 1, 2]])
    ptr = torch.tensor([0, 2, 3])
    original = model(features, edges, ptr, observed_steps=1).correction_steps[0]
    model.observed_model.parent_message = torch.full((3, 2), 1000.0)
    changed_observed = model(features, edges, ptr, observed_steps=1).correction_steps[0]
    torch.testing.assert_close(
        changed_observed.action_logits, original.action_logits, rtol=0, atol=0
    )
    changed_destination = model.action_head(
        original.hidden_state, torch.zeros_like(original.expected_destination_state)
    )
    assert not torch.allclose(changed_destination, original.action_logits)


def test_separate_parent_parameters_and_no_corrective_processor():
    model = build_correction_model(CorrectionModelConfig(hidden_width=3))
    observed = model.observed_model.parent_head
    corrective = model.corrective_parent_head
    assert type(observed) is type(corrective) is AllNodeParentHead
    assert {id(p) for p in observed.parameters()}.isdisjoint(
        id(p) for p in corrective.parameters()
    )
    assert sum(isinstance(module, nn.GRUCell) for module in model.modules()) == 1
    assert sum(p.numel() for p in corrective.parameters()) == 2 * 3 * 3 + 2 * 3
    assert (
        sum(p.numel() for p in model.action_head.parameters()) == 2 * 3 * 3 + 4 * 3 + 3
    )


def test_adding_readout_preserves_every_observed_step_exactly():
    torch.manual_seed(23)
    baseline = build_rgnn(RGNNConfig(hidden_width=4))
    model = build_correction_model(CorrectionModelConfig(hidden_width=4))
    model.observed_model.load_state_dict(baseline.state_dict(), strict=True)
    features = torch.tensor(
        [[0.1, 0.4, 0.0], [0.8, 0.2, 0.4], [0.3, 0.4, 0.7], [0.5, 1.0, 0.0]]
    )
    ptr = torch.tensor([0, 3, 4])
    expected = baseline(features, _path_edges(), ptr, steps=3)
    actual = model(features, _path_edges(), ptr, observed_steps=3).observed_steps
    for actual_step, expected_step in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_step.hidden_state, expected_step.hidden_state, rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual_step.parent_message, expected_step.parent_message, rtol=0, atol=0
        )
        for field in ("parent_scores", "parent_probabilities"):
            for actual_block, expected_block in zip(
                getattr(actual_step, field), getattr(expected_step, field), strict=True
            ):
                torch.testing.assert_close(actual_block, expected_block, rtol=0, atol=0)


def test_self_and_non_neighbor_destinations_remain_candidates():
    model = _fixed_readout()
    hidden = model.observed_model.hidden_state
    model.observed_model.hidden_state = torch.cat((hidden, hidden[:1]), dim=0)
    model.observed_model.parent_message = torch.zeros(4, 2)
    output = model(
        torch.zeros(4, 3), _path_edges(), torch.tensor([0, 3, 4]), observed_steps=1
    ).correction_steps[0]
    probability = output.corrective_probabilities[0]
    assert probability.shape == (3, 3)
    assert torch.all(probability.diag() > 0)
    assert probability[0, 2] > 0
    assert probability[2, 0] > 0


@pytest.mark.parametrize("test_loss", ["action", "destination"])
def test_correction_gradients_reach_the_shared_observed_model(test_loss):
    torch.manual_seed(29)
    model = build_correction_model(CorrectionModelConfig(hidden_width=4)).double()
    with torch.no_grad():
        model.observed_model.encoder.linear.weight.abs_()
        model.observed_model.encoder.linear.bias.fill_(0.5)
        model.action_head.hidden_layer.bias.fill_(2.0)
    features = torch.tensor(
        [[0.1, 0.4, 0.0], [0.8, 0.2, 0.4], [0.3, 0.4, 0.7], [0.5, 1.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    final = model(
        features, _path_edges(), torch.tensor([0, 3, 4]), observed_steps=3
    ).correction_steps[0]
    loss = (
        final.action_logits.square().sum()
        if test_loss == "action"
        else final.corrective_scores[0].square().sum()
    )
    loss.backward()
    for module in (
        model.observed_model.encoder,
        model.observed_model.recurrent_processor,
        model.observed_model.parent_head,
        model.corrective_parent_head,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        )
        assert any(torch.count_nonzero(gradient) for gradient in gradients)
    assert torch.isfinite(features.grad).all()
    assert torch.count_nonzero(features.grad)
    if test_loss == "action":
        assert torch.count_nonzero(
            model.corrective_parent_head.query_projection.weight.grad
        )
        assert torch.count_nonzero(
            model.corrective_parent_head.key_projection.weight.grad
        )
        for parameter in model.action_head.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(model.action_head.output_layer.weight.grad)


def test_complete_correction_specification_and_state_dict_round_trip():
    config = CorrectionModelConfig(hidden_width=3)
    settings = json.loads(json.dumps(config.to_dict()))
    assert settings["correction_recurrent"] is False
    reconstructed = CorrectionModelConfig.from_dict(settings)
    assert reconstructed == config
    model = build_correction_model(config)
    restored = build_correction_model(reconstructed)
    restored.load_state_dict(model.state_dict(), strict=True)
    features = torch.rand(4, 3)
    ptr = torch.tensor([0, 3, 4])
    expected = model(features, _path_edges(), ptr, observed_steps=2).correction_steps[0]
    actual = restored(features, _path_edges(), ptr, observed_steps=2).correction_steps[
        0
    ]
    torch.testing.assert_close(actual.action_logits, expected.action_logits)
    torch.testing.assert_close(
        actual.expected_destination_state, expected.expected_destination_state
    )
    for actual_block, expected_block in zip(
        actual.corrective_scores, expected.corrective_scores, strict=True
    ):
        torch.testing.assert_close(actual_block, expected_block)


@pytest.mark.parametrize("field", ["corrective_parent_head", "action_head"])
def test_correction_config_rejects_unimplemented_choices(field):
    settings = CorrectionModelConfig().to_dict()
    settings[field] = "unknown"
    with pytest.raises(ValueError):
        CorrectionModelConfig.from_dict(settings)


def test_one_pass_model_encodes_once_and_keeps_graphs_independent():
    model = build_correction_model(CorrectionModelConfig(hidden_width=3))
    features = torch.rand(4, 3)
    ptr = torch.tensor([0, 3, 4])
    calls = []
    handle = model.observed_model.encoder.register_forward_hook(
        lambda *args: calls.append(1)
    )
    original = model(features, _path_edges(), ptr, observed_steps=2).correction_steps[0]
    assert calls == [1]
    handle.remove()
    changed_features = features.clone()
    changed_features[3] = 100
    changed = model(
        changed_features, _path_edges(), ptr, observed_steps=2
    ).correction_steps[0]
    torch.testing.assert_close(
        changed.action_logits[:3], original.action_logits[:3], rtol=0, atol=0
    )
    torch.testing.assert_close(
        changed.corrective_scores[0], original.corrective_scores[0], rtol=0, atol=0
    )
