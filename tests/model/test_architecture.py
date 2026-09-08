import pytest

from lenahs.model import (
    DirectionGradientMode,
    DirectionHeadKind,
    NodeOnlyDirectionHead,
    ParentContextDirectionHead,
    RGNNConfig,
    build_rgnn,
)


@pytest.mark.parametrize(
    ("head_kind", "expected_type"),
    [
        (DirectionHeadKind.NONE, type(None)),
        (DirectionHeadKind.PARENT_CONTEXT, ParentContextDirectionHead),
        (DirectionHeadKind.NODE_ONLY, NodeOnlyDirectionHead),
    ],
)
def test_architecture_selects_one_direction_head(head_kind, expected_type) -> None:
    model = build_rgnn(RGNNConfig(direction_head=head_kind))

    assert isinstance(model.direction_head, expected_type)


def test_architecture_selects_direction_gradient_flow() -> None:
    model = build_rgnn(
        RGNNConfig(
            direction_head=DirectionHeadKind.PARENT_CONTEXT,
            direction_gradients=DirectionGradientMode.HEAD_ONLY,
        )
    )

    assert model.direction_gradients == DirectionGradientMode.HEAD_ONLY


def test_architecture_rejects_direction_gradients_without_a_head() -> None:
    with pytest.raises(ValueError, match="without a direction head"):
        build_rgnn(
            RGNNConfig(
                direction_head=DirectionHeadKind.NONE,
                direction_gradients=DirectionGradientMode.HEAD_ONLY,
            )
        )


@pytest.mark.parametrize(
    "config",
    [RGNNConfig(input_width=0), RGNNConfig(hidden_width=0)],
)
def test_architecture_rejects_nonpositive_widths(config) -> None:
    with pytest.raises(ValueError, match="width must be positive"):
        build_rgnn(config)
