"""One pass of training or evaluation over a loader."""

from collections import defaultdict

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from lenahs.model.adapter import ModelData
from lenahs.model.architecture import CorrectionRunConfig
from lenahs.model.correction import CorrectionModelOutput
from lenahs.training.losses import correction_losses
from lenahs.training.metrics import batch_image_metrics, summarize_image_metrics

# Colab redraws subprocess output slowly, so refresh the bars at most every few seconds
PROGRESS_SECONDS = 5


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _forward(
    model: nn.Module, data: ModelData, run: CorrectionRunConfig
) -> CorrectionModelOutput:
    return model(
        data.x, data.edge_index, data.ptr, run.observed_steps, run.correction_steps
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    run: CorrectionRunConfig,
    gradient_clip_norm: float,
    device: torch.device,
) -> dict[str, float]:
    """Mean training losses over the epoch's images."""
    model.train()
    totals = defaultdict(float)
    graph_count = 0
    progress = tqdm(loader, desc="train", leave=False, mininterval=PROGRESS_SECONDS)
    for data in progress:
        data = data.to(device)
        losses = correction_losses(_forward(model, data, run), data)

        optimizer.zero_grad()
        losses.total().backward()
        clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        # Weight each batch by its graphs, since the last batch can be smaller
        for name, value in losses.as_floats().items():
            totals[name] += value * data.num_graphs
        graph_count += data.num_graphs

        progress.set_postfix(
            loss=f"{sum(totals.values()) / graph_count:.4f}", refresh=False
        )

    return {name: total / graph_count for name, total in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    run: CorrectionRunConfig,
    device: torch.device,
) -> dict[str, float]:
    """Split losses, accuracies, baselines and tie counts."""
    model.eval()
    per_image = defaultdict(list)
    loss_totals = defaultdict(float)
    graph_count = 0

    for data in tqdm(
        loader, desc="evaluate", leave=False, mininterval=PROGRESS_SECONDS
    ):
        data = data.to(device)
        output = _forward(model, data, run)
        for name, value in correction_losses(output, data).as_floats().items():
            loss_totals[name] += value * data.num_graphs
        graph_count += data.num_graphs

        # Keep per-image values on the CPU so a whole split never sits on the GPU
        for name, value in batch_image_metrics(output, data).items():
            per_image[name].append(value.cpu())

    summary = summarize_image_metrics(per_image)
    for name, total in loss_totals.items():
        summary[f"loss/{name}"] = total / graph_count
    return summary
