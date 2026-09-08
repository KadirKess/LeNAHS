"""Serializable RGNN choices and model construction."""

from dataclasses import dataclass

from lenahs.model.adapter import STATIC_FEATURE_NAMES
from lenahs.model.direction import (
    DirectionGradientMode,
    DirectionHeadKind,
    NodeOnlyDirectionHead,
    ParentContextDirectionHead,
)
from lenahs.model.network import RGNN
from lenahs.model.parent import AllNodeParentHead
from lenahs.model.recurrent import NodeEncoder, RecurrentRagProcessor


@dataclass(frozen=True)
class RGNNConfig:
    input_width: int = len(STATIC_FEATURE_NAMES)
    hidden_width: int = 32
    direction_head: DirectionHeadKind = DirectionHeadKind.NONE
    direction_gradients: DirectionGradientMode = DirectionGradientMode.SHARED_MODEL


def build_rgnn(config: RGNNConfig = RGNNConfig()) -> RGNN:
    if config.input_width <= 0:
        raise ValueError("input_width must be positive.")
    if config.hidden_width <= 0:
        raise ValueError("hidden_width must be positive.")

    direction_head_kind = DirectionHeadKind(config.direction_head)
    direction_gradients = DirectionGradientMode(config.direction_gradients)

    if direction_head_kind == DirectionHeadKind.NONE:
        if direction_gradients != DirectionGradientMode.SHARED_MODEL:
            raise ValueError(
                "A model without a direction head cannot set its gradients."
            )
        direction_head = None
    elif direction_head_kind == DirectionHeadKind.PARENT_CONTEXT:
        direction_head = ParentContextDirectionHead(config.hidden_width)
    elif direction_head_kind == DirectionHeadKind.NODE_ONLY:
        direction_head = NodeOnlyDirectionHead(config.hidden_width)
    else:
        raise AssertionError(f"Unhandled direction head: {direction_head_kind}")

    return RGNN(
        encoder=NodeEncoder(config.input_width, config.hidden_width),
        recurrent_processor=RecurrentRagProcessor(config.hidden_width),
        parent_head=AllNodeParentHead(config.hidden_width),
        direction_head=direction_head,
        direction_gradients=direction_gradients,
    )
