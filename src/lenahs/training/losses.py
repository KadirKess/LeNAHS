"""Graph-local task losses with image-weighted reductions."""

from dataclasses import dataclass
from math import sqrt

from torch import Tensor
from torch.nn import functional as F
from torch_geometric.utils import scatter

from lenahs.model.adapter import ModelData
from lenahs.model.correction import CorrectionModelOutput


@dataclass(frozen=True)
class CorrectionLosses:
    observed_parent: Tensor
    corrective_parent: Tensor
    action: Tensor

    def total(self) -> Tensor:
        return self.observed_parent + self.corrective_parent + self.action

    def as_floats(self) -> dict[str, float]:
        return {
            "observed_parent": self.observed_parent.item(),
            "corrective_parent": self.corrective_parent.item(),
            "action": self.action.item(),
        }


def parent_cross_entropy(
    query: Tensor,
    key: Tensor,
    log_normalizer: Tensor,
    local_targets: Tensor,
    graph_ptr: Tensor,
    batch: Tensor,
) -> Tensor:
    """Per-node cross-entropy as log Z minus the target score.

    This equals ordinary cross-entropy over each graph's score row without
    reading a stored score matrix, so a streaming attention backend can supply
    log Z instead.
    """
    # Targets are local to their graph, so shift them to rows of the packed batch
    global_targets = local_targets + graph_ptr[batch]
    target_scores = (query * key[global_targets]).sum(dim=1) / sqrt(query.shape[1])
    return log_normalizer - target_scores


def image_mean(node_values: Tensor, batch: Tensor, graph_count: int) -> Tensor:
    # Every image weighs the same in the batch, whatever its node count
    per_image = scatter(node_values, batch, dim=0, reduce="mean", dim_size=graph_count)
    return per_image.mean()


def correction_losses(
    output: CorrectionModelOutput, data: ModelData
) -> CorrectionLosses:
    """Observed loss at the last observed step, corrective losses at the last index."""
    observed = output.observed_steps[-1]
    corrective = output.correction_steps[-1]

    observed_nodes = parent_cross_entropy(
        observed.parent_query,
        observed.parent_key,
        observed.parent_log_normalizers,
        data.observed_parents,
        data.ptr,
        data.batch,
    )
    corrective_nodes = parent_cross_entropy(
        corrective.corrective_query,
        corrective.corrective_key,
        corrective.corrective_log_normalizers,
        data.corrective_parents,
        data.ptr,
        data.batch,
    )
    action_nodes = F.cross_entropy(
        corrective.action_logits, data.correction_actions, reduction="none"
    )

    return CorrectionLosses(
        observed_parent=image_mean(observed_nodes, data.batch, data.num_graphs),
        corrective_parent=image_mean(corrective_nodes, data.batch, data.num_graphs),
        action=image_mean(action_nodes, data.batch, data.num_graphs),
    )
