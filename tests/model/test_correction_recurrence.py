from math import sqrt

import pytest
import torch

from lenahs.model import (
    CorrectionModelConfig,
    build_correction_model,
)


def _inputs():
    features = torch.tensor(
        [[0.1, 0.4, 0.0], [0.8, 0.2, 0.4], [0.3, 0.4, 0.7], [0.5, 1.0, 0.0]],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 1, 2, 0, 1, 2, 3], [1, 0, 2, 1, 0, 1, 2, 3]])
    return features, edges, torch.tensor([0, 3, 4])


def _direct_readout(model, hidden):
    head = model.corrective_parent_head
    query = hidden @ head.query_projection.weight.T + head.query_projection.bias
    key = hidden @ head.key_projection.weight.T + head.key_projection.bias
    scores = tuple(
        query[a:b] @ key[a:b].T / sqrt(hidden.shape[1]) for a, b in [(0, 3), (3, 4)]
    )
    probabilities = tuple(torch.softmax(score, dim=1) for score in scores)
    message = torch.cat(
        [
            probability @ hidden[a:b]
            for probability, (a, b) in zip(probabilities, [(0, 3), (3, 4)], strict=True)
        ]
    )
    action = model.action_head
    logits = (
        torch.relu(
            torch.cat((hidden, message), dim=1) @ action.hidden_layer.weight.T
            + action.hidden_layer.bias
        )
        @ action.output_layer.weight.T
        + action.output_layer.bias
    )
    return scores, probabilities, message, logits


def _direct_update(model, hidden, embedding, message, edges):
    maxima, minima = [], []
    for node in range(hidden.shape[0]):
        neighbors = [
            int(source) for source, destination in edges.T if int(destination) == node
        ]
        values = torch.stack([hidden[neighbor] for neighbor in neighbors])
        maxima.append(values.max(dim=0).values)
        minima.append(values.min(dim=0).values)
    inputs = torch.cat(
        (message, torch.stack(maxima), torch.stack(minima), embedding), dim=1
    )
    gru = model.corrective_processor.gru
    ir, iz, inn = (inputs @ gru.weight_ih.T + gru.bias_ih).chunk(3, dim=1)
    hr, hz, hn = (hidden @ gru.weight_hh.T + gru.bias_hh).chunk(3, dim=1)
    reset = torch.sigmoid(ir + hr)
    update = torch.sigmoid(iz + hz)
    candidate = torch.tanh(inn + reset * hn)
    return (1 - update) * candidate + update * hidden


@pytest.mark.parametrize("observed_steps", [0, 2])
@pytest.mark.parametrize("correction_steps", [1, 2])
def test_corrective_updates_match_independent_equations(
    observed_steps, correction_steps
):
    model = build_correction_model(CorrectionModelConfig(2, True)).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.linspace(
                    -0.2, 0.3, parameter.numel(), dtype=torch.float64
                ).reshape(parameter.shape)
            )
    features, edges, ptr = _inputs()
    output = model(features, edges, ptr, observed_steps, correction_steps)
    encoder = model.observed_model.encoder.linear
    embedding = torch.relu(features @ encoder.weight.T + encoder.bias)
    hidden = output.observed_steps[-1].hidden_state if observed_steps else embedding
    assert len(output.observed_steps) == observed_steps
    assert len(output.correction_steps) == correction_steps + 1
    for index, actual in enumerate(output.correction_steps):
        scores, probabilities, message, logits = _direct_readout(model, hidden)
        torch.testing.assert_close(actual.hidden_state, hidden)
        torch.testing.assert_close(actual.expected_destination_state, message)
        torch.testing.assert_close(actual.action_logits, logits)
        for field, expected in [
            (actual.corrective_scores, scores),
            (actual.corrective_probabilities, probabilities),
        ]:
            for a, e in zip(field, expected, strict=True):
                torch.testing.assert_close(a, e)
        if index < correction_steps:
            hidden = _direct_update(model, hidden, embedding, message, edges)


@pytest.mark.parametrize(
    "recurrent,observed_steps,correction_steps",
    [(False, 0, 0), (False, 2, 0), (True, 0, 2), (True, 2, 2)],
)
def test_all_modes_encode_once_and_preserve_inputs(
    recurrent, observed_steps, correction_steps
):
    model = build_correction_model(CorrectionModelConfig(3, recurrent)).double()
    features, edges, ptr = _inputs()
    originals = [value.clone() for value in (features, edges, ptr)]
    calls = []
    handle = model.observed_model.encoder.register_forward_hook(
        lambda *args: calls.append(1)
    )
    output = model(features, edges, ptr, observed_steps, correction_steps)
    handle.remove()
    assert calls == [1]
    assert len(output.observed_steps) == observed_steps
    assert len(output.correction_steps) == correction_steps + 1
    for value, original in zip((features, edges, ptr), originals, strict=True):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_zero_observed_steps_bypass_observed_modules_and_use_features():
    model = build_correction_model(CorrectionModelConfig(2)).double()
    with torch.no_grad():
        model.observed_model.encoder.linear.weight.fill_(0.5)
        model.observed_model.encoder.linear.bias.fill_(0.1)
        model.action_head.hidden_layer.weight.fill_(0.2)
        model.action_head.hidden_layer.bias.fill_(0.5)
        model.action_head.output_layer.weight.fill_(0.3)
    features, edges, ptr = _inputs()

    def forbidden(*args):
        pytest.fail(
            "Zero observed steps must bypass observed processing and prediction."
        )

    handles = [
        module.register_forward_pre_hook(forbidden)
        for module in (
            model.observed_model.recurrent_processor,
            model.observed_model.parent_head,
        )
    ]
    output = model(features, edges, ptr, 0)
    changed = model(features + 1, edges, ptr, 0)
    for handle in handles:
        handle.remove()
    assert output.observed_steps == ()
    expected = torch.relu(features @ torch.full((3, 2), 0.5, dtype=torch.float64) + 0.1)
    torch.testing.assert_close(output.correction_steps[0].hidden_state, expected)
    assert not torch.allclose(
        output.correction_steps[0].action_logits,
        changed.correction_steps[0].action_logits,
    )
    output.correction_steps[0].action_logits.square().sum().backward()
    assert model.observed_model.encoder.linear.weight.grad is not None
    for module in (
        model.observed_model.recurrent_processor,
        model.observed_model.parent_head,
    ):
        assert all(parameter.grad is None for parameter in module.parameters())


def test_corrective_modules_are_reused_and_disjoint_from_observed_processor():
    model = build_correction_model(CorrectionModelConfig(3, True)).double()
    observed = {id(p) for p in model.observed_model.recurrent_processor.parameters()}
    corrective = {id(p) for p in model.corrective_processor.parameters()}
    assert observed.isdisjoint(corrective)
    shapes = {name: parameter.shape for name, parameter in model.named_parameters()}
    counts = [0, 0, 0]
    handles = []
    for index, module in enumerate(
        (model.corrective_processor, model.corrective_parent_head, model.action_head)
    ):

        def count(*args, index=index):
            counts[index] += 1

        handles.append(module.register_forward_hook(count))
    features, edges, ptr = _inputs()
    model(features, edges, ptr, 2, 3)
    for handle in handles:
        handle.remove()
    assert counts == [3, 4, 4]
    model(features, edges, ptr, 2, 1)
    assert shapes == {
        name: parameter.shape for name, parameter in model.named_parameters()
    }


def test_later_predictions_depend_on_earlier_soft_messages_but_not_actions():
    torch.manual_seed(47)
    model = build_correction_model(CorrectionModelConfig(3, True)).double()
    features, edges, ptr = _inputs()
    output = model(features, edges, ptr, 2, 2)
    first = output.correction_steps[0]
    final = output.correction_steps[-1]
    loss = final.corrective_scores[0].square().sum()
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("action_head.")
    ]
    gradients = torch.autograd.grad(
        loss,
        (
            first.expected_destination_state,
            first.corrective_probabilities[0],
            first.action_logits,
            *parameters,
        ),
        allow_unused=True,
    )
    for gradient in gradients[:2]:
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient)
    assert gradients[2] is None
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients[3:]
    )
    assert any(torch.count_nonzero(gradient) for gradient in gradients[3:])
    with torch.no_grad():
        for parameter in model.action_head.parameters():
            parameter.add_(10)
    changed = model(features, edges, ptr, 2, 2)
    for original, new in zip(
        output.correction_steps, changed.correction_steps, strict=True
    ):
        torch.testing.assert_close(
            original.hidden_state, new.hidden_state, rtol=0, atol=0
        )
        for a, b in zip(original.corrective_scores, new.corrective_scores, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_disabled_mode_matches_the_previous_one_pass_execution():
    model = build_correction_model(CorrectionModelConfig(3)).double()
    features, edges, ptr = _inputs()
    observed = model.observed_model(features, edges, ptr, steps=2)
    expected = model._read_correction(observed[-1].hidden_state, ptr)
    actual = model(features, edges, ptr, 2, 0).correction_steps[0]
    assert model.corrective_processor is None
    for field in ("hidden_state", "expected_destination_state", "action_logits"):
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field), rtol=0, atol=0
        )
    for field in ("corrective_scores", "corrective_probabilities"):
        for a, e in zip(getattr(actual, field), getattr(expected, field), strict=True):
            torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("observed_steps", [0, 2])
def test_recurrent_correction_remains_graph_local(observed_steps):
    torch.manual_seed(59)
    model = build_correction_model(CorrectionModelConfig(3, True)).double()
    features, edges, ptr = _inputs()
    original = model(features, edges, ptr, observed_steps, 2)
    changed_features = features.clone()
    changed_features[3] += 10
    changed = model(changed_features, edges, ptr, observed_steps, 2)
    for before, after in zip(
        original.correction_steps, changed.correction_steps, strict=True
    ):
        torch.testing.assert_close(
            before.hidden_state[:3], after.hidden_state[:3], rtol=0, atol=0
        )
        torch.testing.assert_close(
            before.corrective_scores[0], after.corrective_scores[0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            before.action_logits[:3], after.action_logits[:3], rtol=0, atol=0
        )
