import pytest
import torch

from lenahs.generation import NoiseDirection
from lenahs.model import (
    RGNN,
    AllNodeParentHead,
    DirectionGradientMode,
    DirectionHeadInput,
    DirectionHeadKind,
    ModelData,
    NodeEncoder,
    NodeOnlyDirectionHead,
    ParentContextDirectionHead,
    RecurrentRagProcessor,
    RGNNConfig,
    build_rgnn,
)


def _path_edges() -> torch.Tensor:
    return torch.tensor(
        [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]],
        dtype=torch.int64,
    )


def _noise_rgnn(*, gradients=DirectionGradientMode.SHARED_MODEL) -> RGNN:
    return build_rgnn(
        RGNNConfig(
            hidden_width=4,
            direction_head=DirectionHeadKind.PARENT_CONTEXT,
            direction_gradients=gradients,
        )
    )


def test_direction_heads_use_their_declared_recurrent_features() -> None:
    parent_context = ParentContextDirectionHead(hidden_width=2)
    node_only = NodeOnlyDirectionHead(hidden_width=2)
    with torch.no_grad():
        parent_context.hidden_layer.weight.zero_()
        parent_context.hidden_layer.weight[:, 2:].copy_(torch.eye(2))
        parent_context.hidden_layer.bias.zero_()
        parent_context.output_layer.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
        parent_context.output_layer.bias.zero_()

    hidden_state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    zero_parent = torch.zeros_like(hidden_state)
    changed_parent = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    zero_features = DirectionHeadInput(hidden_state, zero_parent)
    changed_features = DirectionHeadInput(hidden_state, changed_parent)

    assert not torch.equal(
        parent_context(zero_features), parent_context(changed_features)
    )
    torch.testing.assert_close(node_only(zero_features), node_only(changed_features))


def test_direction_head_shapes_and_parameter_counts_are_exact() -> None:
    hidden_width = 5
    node_count = 4
    features = DirectionHeadInput(
        hidden_state=torch.rand(node_count, hidden_width),
        parent_message=torch.rand(node_count, hidden_width),
    )
    parent_context = ParentContextDirectionHead(hidden_width)
    node_only = NodeOnlyDirectionHead(hidden_width)

    assert parent_context(features).shape == (node_count, 3)
    assert node_only(features).shape == (node_count, 3)

    parent_context_count = sum(
        parameter.numel() for parameter in parent_context.parameters()
    )
    node_only_count = sum(parameter.numel() for parameter in node_only.parameters())
    assert parent_context_count == 2 * hidden_width**2 + 4 * hidden_width + 3
    assert node_only_count == hidden_width**2 + 4 * hidden_width + 3
    assert parent_context_count - node_only_count == hidden_width**2


def test_direction_logits_use_noise_direction_class_indices_directly() -> None:
    head = ParentContextDirectionHead(hidden_width=2)
    with torch.no_grad():
        head.hidden_layer.weight.zero_()
        head.hidden_layer.bias.zero_()
        head.output_layer.weight.zero_()
        head.output_layer.bias.copy_(torch.tensor([0.5, 1.5, 2.5]))

    logits = head(
        DirectionHeadInput(
            hidden_state=torch.zeros(2, 2),
            parent_message=torch.zeros(2, 2),
        )
    )

    torch.testing.assert_close(
        logits[:, NoiseDirection.UNCHANGED], torch.full((2,), 0.5)
    )
    torch.testing.assert_close(
        logits[:, NoiseDirection.BRIGHTENED], torch.full((2,), 1.5)
    )
    torch.testing.assert_close(
        logits[:, NoiseDirection.DARKENED], torch.full((2,), 2.5)
    )


def test_direction_logits_are_returned_at_every_recurrent_step() -> None:
    outputs = _noise_rgnn()(
        torch.rand(3, 3),
        _path_edges(),
        torch.tensor([0, 3]),
        steps=3,
    )

    assert len(outputs) == 3
    assert all(output.direction_logits.shape == (3, 3) for output in outputs)


def test_clean_rgnn_does_not_construct_or_run_a_direction_head() -> None:
    model = build_rgnn(RGNNConfig(hidden_width=4))

    outputs = model(
        torch.rand(3, 3),
        _path_edges(),
        torch.tensor([0, 3]),
        steps=3,
    )

    assert model.direction_head is None
    assert all(output.direction_logits is None for output in outputs)
    assert all("direction_head" not in name for name, _ in model.named_parameters())


def test_direction_loss_reaches_head_and_shared_rgnn() -> None:
    torch.manual_seed(14)
    model = _noise_rgnn()
    node_features = torch.rand(3, 3, requires_grad=True)

    outputs = model(
        node_features,
        _path_edges(),
        torch.tensor([0, 3]),
        steps=3,
    )
    outputs[-1].direction_logits.square().sum().backward()

    assert node_features.grad is not None
    assert torch.count_nonzero(node_features.grad)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad)


@pytest.mark.parametrize(
    "direction_head",
    [ParentContextDirectionHead(hidden_width=4), NodeOnlyDirectionHead(hidden_width=4)],
    ids=["parent_context", "node_only"],
)
def test_head_only_gradients_do_not_reach_the_shared_rgnn(direction_head) -> None:
    torch.manual_seed(17)
    with torch.no_grad():
        direction_head.hidden_layer.weight.fill_(0.1)
        direction_head.hidden_layer.bias.fill_(1.0)
        direction_head.output_layer.weight.fill_(0.1)
        direction_head.output_layer.bias.fill_(0.1)

    model = RGNN(
        encoder=NodeEncoder(input_width=3, hidden_width=4),
        recurrent_processor=RecurrentRagProcessor(hidden_width=4),
        parent_head=AllNodeParentHead(hidden_width=4),
        direction_head=direction_head,
        direction_gradients=DirectionGradientMode.HEAD_ONLY,
    )
    node_features = torch.rand(3, 3, requires_grad=True)

    output = model(
        node_features,
        _path_edges(),
        torch.tensor([0, 3]),
        steps=2,
    )[-1]
    output.direction_logits.square().sum().backward()

    assert node_features.grad is None
    for module in (model.encoder, model.recurrent_processor, model.parent_head):
        assert all(parameter.grad is None for parameter in module.parameters())
    for parameter in model.direction_head.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad)


def test_replacing_direction_head_cannot_change_parent_recurrence() -> None:
    torch.manual_seed(15)
    model = _noise_rgnn()
    node_features = torch.rand(3, 3)

    baseline_outputs = model(
        node_features,
        _path_edges(),
        torch.tensor([0, 3]),
        steps=3,
    )
    model.direction_head = NodeOnlyDirectionHead(hidden_width=4)
    node_only_outputs = model(
        node_features,
        _path_edges(),
        torch.tensor([0, 3]),
        steps=3,
    )

    for baseline, node_only in zip(baseline_outputs, node_only_outputs, strict=True):
        torch.testing.assert_close(baseline.hidden_state, node_only.hidden_state)
        torch.testing.assert_close(baseline.parent_message, node_only.parent_message)
        for baseline_scores, node_only_scores in zip(
            baseline.parent_scores,
            node_only.parent_scores,
            strict=True,
        ):
            torch.testing.assert_close(baseline_scores, node_only_scores)
        for baseline_probabilities, node_only_probabilities in zip(
            baseline.parent_probabilities,
            node_only.parent_probabilities,
            strict=True,
        ):
            torch.testing.assert_close(
                baseline_probabilities,
                node_only_probabilities,
            )


def test_oracle_direction_targets_do_not_enter_the_rgnn() -> None:
    torch.manual_seed(16)
    model = _noise_rgnn()
    node_features = torch.rand(3, 3)
    edge_index = _path_edges()
    unchanged = ModelData(
        x=node_features,
        edge_index=edge_index,
        noise_direction=torch.zeros(3, dtype=torch.int64),
        num_nodes=3,
    )
    mixed = ModelData(
        x=node_features,
        edge_index=edge_index,
        noise_direction=torch.tensor(
            [
                NoiseDirection.BRIGHTENED,
                NoiseDirection.DARKENED,
                NoiseDirection.UNCHANGED,
            ]
        ),
        num_nodes=3,
    )

    unchanged_output = model(
        unchanged.x,
        unchanged.edge_index,
        torch.tensor([0, 3]),
        steps=1,
    )[0]
    mixed_output = model(
        mixed.x,
        mixed.edge_index,
        torch.tensor([0, 3]),
        steps=1,
    )[0]

    torch.testing.assert_close(
        unchanged_output.direction_logits,
        mixed_output.direction_logits,
    )
