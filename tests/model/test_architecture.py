import json

import pytest
import torch

from lenahs.model import RGNNConfig, build_rgnn


def test_saved_specification_reconstructs_the_observed_model() -> None:
    config = RGNNConfig(hidden_width=5)
    encoded = json.dumps(config.to_dict())
    decoded = RGNNConfig.from_dict(json.loads(encoded))
    assert decoded == config
    assert json.loads(encoded) == {
        "hidden_width": 5,
        "static_features": ["normalized_level", "normalized_area", "normalized_anchor"],
        "encoder": "linear_relu",
        "recurrent_processor": "rag_max_min_gru",
        "parent_head": "all_node_dot_product",
        "attention": "dense_per_graph",
    }
    original = build_rgnn(config)
    reconstructed = build_rgnn(decoded)
    reconstructed.load_state_dict(original.state_dict(), strict=True)
    features = torch.tensor([[0.2, 0.5, 0.0], [0.8, 0.5, 1.0]])
    edges = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]])
    ptr = torch.tensor([0, 2])
    for actual, expected in zip(
        reconstructed(features, edges, ptr, steps=2),
        original(features, edges, ptr, steps=2),
        strict=True,
    ):
        torch.testing.assert_close(actual.hidden_state, expected.hidden_state)
        torch.testing.assert_close(actual.parent_scores[0], expected.parent_scores[0])
        torch.testing.assert_close(
            actual.parent_probabilities[0], expected.parent_probabilities[0]
        )
        torch.testing.assert_close(actual.parent_message, expected.parent_message)


@pytest.mark.parametrize("hidden_width", [1, 3, 32])
def test_retained_parameter_names_shapes_and_count(hidden_width) -> None:
    model = build_rgnn(RGNNConfig(hidden_width=hidden_width))
    h = hidden_width
    expected = {
        "encoder.linear.weight": (h, 3),
        "encoder.linear.bias": (h,),
        "recurrent_processor.gru.weight_ih": (3 * h, 4 * h),
        "recurrent_processor.gru.weight_hh": (3 * h, h),
        "recurrent_processor.gru.bias_ih": (3 * h,),
        "recurrent_processor.gru.bias_hh": (3 * h,),
        "parent_head.query_projection.weight": (h, h),
        "parent_head.query_projection.bias": (h,),
        "parent_head.key_projection.weight": (h, h),
        "parent_head.key_projection.bias": (h,),
    }
    assert {
        name: tuple(tensor.shape) for name, tensor in model.state_dict().items()
    } == expected
    assert sum(p.numel() for p in model.parameters()) == 17 * h * h + 12 * h


@pytest.mark.parametrize("hidden_width", [0, -1, True, 1.5, "4"])
def test_invalid_hidden_width_fails_at_construction_and_loading(hidden_width):
    config = RGNNConfig(hidden_width=hidden_width)
    settings = RGNNConfig().to_dict()
    settings["hidden_width"] = hidden_width
    error = ValueError if type(hidden_width) is int else TypeError
    with pytest.raises(error, match="hidden_width"):
        build_rgnn(config)
    with pytest.raises(error, match="hidden_width"):
        RGNNConfig.from_dict(settings)
    with pytest.raises(error, match="hidden_width"):
        config.to_dict()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (
            "static_features",
            ["normalized_anchor", "normalized_area", "normalized_level"],
        ),
        ("encoder", "unknown"),
        ("recurrent_processor", "unknown"),
        ("parent_head", "unknown"),
        ("attention", "unknown"),
    ],
)
def test_saved_specification_rejects_unsupported_features_or_modules(key, value):
    settings = RGNNConfig().to_dict()
    settings[key] = value
    with pytest.raises(ValueError, match=f"Unsupported RGNN setting {key}"):
        RGNNConfig.from_dict(settings)


@pytest.mark.parametrize("missing", [False, True])
def test_saved_specification_rejects_extra_or_missing_fields(missing):
    settings = RGNNConfig().to_dict()
    if missing:
        del settings["hidden_width"]
    else:
        settings["unexpected"] = 1
    with pytest.raises(ValueError, match="exactly the specification fields"):
        RGNNConfig.from_dict(settings)
