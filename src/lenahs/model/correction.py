"""Observed processing followed by representative-pointer correction."""

from dataclasses import dataclass

from torch import Tensor, nn

from lenahs.model.action import MergeDepthHead
from lenahs.model.network import RGNN, ObservedStep
from lenahs.model.parent import AllNodeParentHead
from lenahs.model.recurrent import RecurrentRagProcessor


@dataclass(frozen=True)
class CorrectionStep:
    hidden_state: Tensor
    corrective_scores: tuple[Tensor, ...]
    corrective_probabilities: tuple[Tensor, ...]
    expected_destination_state: Tensor
    action_logits: Tensor


@dataclass(frozen=True)
class CorrectionModelOutput:
    observed_steps: tuple[ObservedStep, ...]
    correction_steps: tuple[CorrectionStep, ...]


class CorrectionModel(nn.Module):
    def __init__(
        self,
        observed_model: RGNN,
        corrective_parent_head: AllNodeParentHead,
        action_head: MergeDepthHead,
        corrective_processor: RecurrentRagProcessor | None = None,
    ) -> None:
        super().__init__()
        self.observed_model = observed_model
        self.corrective_parent_head = corrective_parent_head
        self.action_head = action_head
        self.corrective_processor = corrective_processor

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        graph_ptr: Tensor,
        observed_steps: int,
        correction_steps: int = 0,
    ) -> CorrectionModelOutput:
        _validate_correction_horizons(
            observed_steps, correction_steps, self.corrective_processor is not None
        )

        # We have two paths depending on the number of observed steps. If there
        # are none, we skip the RGNN and just use the static embedding.
        embedding = self.observed_model.encoder(node_features)
        if observed_steps > 0:
            observed = self.observed_model._run_steps(
                embedding, edge_index, graph_ptr, observed_steps
            )
            hidden_state = observed[-1].hidden_state
        else:
            observed = ()
            hidden_state = embedding

        # Read the corrective message before the first update. Every later update
        # uses corrective feedback and the same original feature embedding and RAG.
        correction = self._read_correction(hidden_state, graph_ptr)
        corrections = [correction]
        for _ in range(correction_steps):
            hidden_state = self.corrective_processor(
                hidden_state,
                embedding,
                edge_index,
                correction.expected_destination_state,
            )
            correction = self._read_correction(hidden_state, graph_ptr)
            corrections.append(correction)
        return CorrectionModelOutput(
            observed_steps=observed,
            correction_steps=tuple(corrections),
        )

    def _read_correction(
        self,
        hidden_state: Tensor,
        graph_ptr: Tensor,
    ) -> CorrectionStep:
        destinations = self.corrective_parent_head(hidden_state, graph_ptr)

        # Corrective context comes from this head, not the observed parent message.
        # Keeping the soft destination state lets action gradients reach the scorer.
        action_logits = self.action_head(hidden_state, destinations.parent_message)
        return CorrectionStep(
            hidden_state=hidden_state,
            corrective_scores=destinations.scores,
            corrective_probabilities=destinations.probabilities,
            expected_destination_state=destinations.parent_message,
            action_logits=action_logits,
        )


def _validate_correction_horizons(
    observed_steps: int,
    correction_steps: int,
    correction_recurrent: bool,
) -> None:
    for name, value in (
        ("observed_steps", observed_steps),
        ("correction_steps", correction_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative.")
    if correction_recurrent and correction_steps == 0:
        raise ValueError(
            "Enabled corrective recurrence requires positive correction_steps."
        )
    if not correction_recurrent and correction_steps != 0:
        raise ValueError(
            "Disabled corrective recurrence requires zero correction_steps."
        )
