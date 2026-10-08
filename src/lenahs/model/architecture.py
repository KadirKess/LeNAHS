"""Serializable observed RGNN specification and construction."""

from collections.abc import Mapping
from dataclasses import dataclass

from lenahs.model.action import MergeDepthHead
from lenahs.model.adapter import STATIC_FEATURE_NAMES
from lenahs.model.correction import CorrectionModel, _validate_correction_horizons
from lenahs.model.network import RGNN
from lenahs.model.parent import AllNodeParentHead
from lenahs.model.recurrent import NodeEncoder, RecurrentRagProcessor


@dataclass(frozen=True)
class RGNNConfig:
    hidden_width: int = 32

    def to_dict(self) -> dict[str, object]:
        _validate_hidden_width(self.hidden_width)
        return {
            "hidden_width": self.hidden_width,
            "static_features": list(STATIC_FEATURE_NAMES),
            "encoder": "linear_relu",
            "recurrent_processor": "rag_max_min_gru",
            "parent_head": "all_node_dot_product",
            "attention": "dense_per_graph",
        }

    @classmethod
    def from_dict(cls, settings: Mapping[str, object]) -> "RGNNConfig":
        expected = cls().to_dict()
        if set(settings) != set(expected):
            raise ValueError(
                "RGNN settings must contain exactly the specification fields."
            )
        for name, value in expected.items():
            if name != "hidden_width" and settings[name] != value:
                raise ValueError(
                    f"Unsupported RGNN setting {name}: {settings[name]!r}."
                )
        hidden_width = settings["hidden_width"]
        _validate_hidden_width(hidden_width)
        return cls(hidden_width=hidden_width)


def _validate_hidden_width(hidden_width: int) -> None:
    if isinstance(hidden_width, bool) or not isinstance(hidden_width, int):
        raise TypeError("hidden_width must be an integer.")
    if hidden_width <= 0:
        raise ValueError("hidden_width must be positive.")


def build_rgnn(config: RGNNConfig = RGNNConfig()) -> RGNN:
    _validate_hidden_width(config.hidden_width)
    return RGNN(
        encoder=NodeEncoder(len(STATIC_FEATURE_NAMES), config.hidden_width),
        recurrent_processor=RecurrentRagProcessor(config.hidden_width),
        parent_head=AllNodeParentHead(config.hidden_width),
    )


@dataclass(frozen=True)
class CorrectionModelConfig:
    hidden_width: int = 32
    correction_recurrent: bool = False

    def to_dict(self) -> dict[str, object]:
        _validate_recurrence_flag(self.correction_recurrent)
        return {
            "observed_model": RGNNConfig(self.hidden_width).to_dict(),
            "corrective_parent_head": "all_node_dot_product",
            "action_head": "parent_context_merge_depth",
            "correction_recurrent": self.correction_recurrent,
        }

    @classmethod
    def from_dict(cls, settings: Mapping[str, object]) -> "CorrectionModelConfig":
        expected = cls().to_dict()
        if set(settings) != set(expected):
            raise ValueError(
                "Correction settings must contain exactly the specification fields."
            )
        for name in ("corrective_parent_head", "action_head"):
            if settings[name] != expected[name]:
                raise ValueError(
                    f"Unsupported correction setting {name}: {settings[name]!r}."
                )
        _validate_recurrence_flag(settings["correction_recurrent"])
        observed_settings = settings["observed_model"]
        if not isinstance(observed_settings, Mapping):
            raise TypeError("observed_model must contain an RGNN specification.")
        observed = RGNNConfig.from_dict(observed_settings)
        return cls(
            hidden_width=observed.hidden_width,
            correction_recurrent=settings["correction_recurrent"],
        )


def build_correction_model(
    config: CorrectionModelConfig = CorrectionModelConfig(),
) -> CorrectionModel:
    _validate_recurrence_flag(config.correction_recurrent)
    observed_model = build_rgnn(RGNNConfig(config.hidden_width))
    return CorrectionModel(
        observed_model=observed_model,
        corrective_parent_head=AllNodeParentHead(config.hidden_width),
        action_head=MergeDepthHead(config.hidden_width),
        corrective_processor=RecurrentRagProcessor(config.hidden_width)
        if config.correction_recurrent
        else None,
    )


def _validate_recurrence_flag(value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError("correction_recurrent must be a Boolean.")


@dataclass(frozen=True)
class CorrectionRunConfig:
    model: CorrectionModelConfig
    observed_steps: int
    correction_steps: int = 0

    def to_dict(self) -> dict[str, object]:
        model_settings = self.model.to_dict()
        _validate_correction_horizons(
            self.observed_steps, self.correction_steps, self.model.correction_recurrent
        )
        return {
            "model": model_settings,
            "observed_steps": self.observed_steps,
            "correction_steps": self.correction_steps,
        }

    @classmethod
    def from_dict(cls, settings: Mapping[str, object]) -> "CorrectionRunConfig":
        if set(settings) != {"model", "observed_steps", "correction_steps"}:
            raise ValueError("Run settings must contain model and both step counts.")
        if not isinstance(settings["model"], Mapping):
            raise TypeError("model must contain a correction specification.")
        model = CorrectionModelConfig.from_dict(settings["model"])
        _validate_correction_horizons(
            settings["observed_steps"],
            settings["correction_steps"],
            model.correction_recurrent,
        )
        return cls(model, settings["observed_steps"], settings["correction_steps"])
