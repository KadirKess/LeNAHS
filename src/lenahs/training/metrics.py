"""Argmax accuracies and trivial baselines per image."""

import torch
from torch import Tensor
from torch_geometric.utils import scatter

from lenahs.generation.targets import CorrectionAction
from lenahs.model.adapter import ModelData
from lenahs.model.correction import CorrectionModelOutput

# Accuracies compare argmax predictions with targets. Baselines compare targets
# with a trivial rule and need no model.
IMAGE_QUANTITIES = (
    "observed_parent_accuracy",
    "corrective_parent_accuracy",
    "action_accuracy",
    "copy_observed_baseline",
    "always_merge_baseline",
)


@torch.no_grad()
def graph_local_argmax(
    query: Tensor,
    key: Tensor,
    graph_ptr: Tensor,
) -> tuple[Tensor, Tensor]:
    """Local argmax parents and a mask of nodes whose maximum is tied.

    The argmax takes the first maximal candidate. The score scale is skipped
    because it does not change the argmax.
    """
    predictions = []
    tied = []
    bounds = graph_ptr.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        scores = query[start:stop] @ key[start:stop].T
        best = scores.max(dim=1, keepdim=True).values
        predictions.append(scores.argmax(dim=1))
        tied.append((scores == best).sum(dim=1) > 1)
    return torch.cat(predictions), torch.cat(tied)


def image_accuracy(correct: Tensor, batch: Tensor, graph_count: int) -> Tensor:
    return scatter(correct.float(), batch, dim=0, reduce="mean", dim_size=graph_count)


def batch_image_metrics(
    output: CorrectionModelOutput, data: ModelData
) -> dict[str, Tensor]:
    """One value per image for each quantity, plus node counts and tie counts."""
    observed = output.observed_steps[-1]
    corrective = output.correction_steps[-1]
    graph_count = data.num_graphs

    observed_predictions, observed_tied = graph_local_argmax(
        observed.parent_query, observed.parent_key, data.ptr
    )
    corrective_predictions, corrective_tied = graph_local_argmax(
        corrective.corrective_query, corrective.corrective_key, data.ptr
    )
    action_predictions = corrective.action_logits.argmax(dim=1)

    def per_image(correct: Tensor) -> Tensor:
        return image_accuracy(correct, data.batch, graph_count)

    def per_image_count(mask: Tensor) -> Tensor:
        return scatter(
            mask.long(), data.batch, dim=0, reduce="sum", dim_size=graph_count
        )

    return {
        "observed_parent_accuracy": per_image(
            observed_predictions == data.observed_parents
        ),
        "corrective_parent_accuracy": per_image(
            corrective_predictions == data.corrective_parents
        ),
        "action_accuracy": per_image(action_predictions == data.correction_actions),
        "copy_observed_baseline": per_image(
            data.observed_parents == data.corrective_parents
        ),
        "always_merge_baseline": per_image(
            data.correction_actions == CorrectionAction.MERGE
        ),
        "observed_ties": per_image_count(observed_tied),
        "corrective_ties": per_image_count(corrective_tied),
        "node_count": data.ptr[1:] - data.ptr[:-1],
        "is_clean": data.is_clean,
    }


def summarize_image_metrics(per_image: dict[str, list[Tensor]]) -> dict[str, float]:
    """Image mean and node-pooled value of each quantity, overall, clean and noisy."""
    values = {name: torch.cat(batches) for name, batches in per_image.items()}
    node_count = values["node_count"].double()
    is_clean = values["is_clean"]

    summary = {}
    groups = (
        ("all", torch.ones_like(is_clean)),
        ("clean", is_clean),
        ("noisy", ~is_clean),
    )
    for group, mask in groups:
        if not mask.any():
            continue
        weights = node_count[mask]
        for name in IMAGE_QUANTITIES:
            value = values[name][mask].double()
            summary[f"{group}/{name}/image_mean"] = value.mean().item()
            # A per-image accuracy times its node count is its number of correct nodes
            summary[f"{group}/{name}/pooled"] = (
                (value * weights).sum() / weights.sum()
            ).item()
        summary[f"{group}/images"] = int(mask.sum())
        summary[f"{group}/nodes"] = int(weights.sum())
        summary[f"{group}/observed_ties"] = int(values["observed_ties"][mask].sum())
        summary[f"{group}/corrective_ties"] = int(values["corrective_ties"][mask].sum())
    return summary
