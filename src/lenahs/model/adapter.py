"""Translate generated samples into model-facing PyG graphs."""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops, to_undirected

from lenahs.generation.samples import GenerationSample

STATIC_FEATURE_NAMES = (
    "normalized_level",
    "normalized_area",
    "normalized_anchor",
)


class ModelData(Data):
    """PyG data for one observed flat-zone graph and its targets."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in {"observed_parents", "corrective_parents"}:
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def adapt_generation_sample(
    sample: GenerationSample,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> ModelData:
    """Convert the observed tree image and its targets to PyG tensors."""

    if not dtype.is_floating_point:
        raise TypeError("Model node features require a floating-point dtype.")

    observed = sample.observed
    node_count = observed.graph.levels.size
    image_pixel_count = observed.graph.label_map.size
    target_device = torch.device(device) if device is not None else torch.device("cpu")

    node_features = np.column_stack(
        (
            observed.graph.levels.astype(np.float64) / 255.0,
            observed.graph.pixel_counts.astype(np.float64) / image_pixel_count,
            observed.graph.anchors.astype(np.float64) / max(image_pixel_count - 1, 1),
        )
    )
    edge_index = torch.as_tensor(
        observed.graph.edge_index.T.copy(),
        dtype=torch.int64,
        device=target_device,
    )
    edge_index = to_undirected(edge_index, num_nodes=node_count)
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=node_count)

    return ModelData(
        x=torch.as_tensor(node_features, dtype=dtype, device=target_device),
        edge_index=edge_index,
        observed_parents=torch.as_tensor(
            observed.component_tree.parents.astype(np.int64, copy=False),
            dtype=torch.int64,
            device=target_device,
        ),
        corrective_parents=torch.as_tensor(
            sample.correction_targets.corrective_parents,
            dtype=torch.int64,
            device=target_device,
        ),
        correction_actions=torch.as_tensor(
            sample.correction_targets.actions,
            dtype=torch.int64,
            device=target_device,
        ),
        num_nodes=node_count,
    )
