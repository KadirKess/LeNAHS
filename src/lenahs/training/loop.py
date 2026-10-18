"""Training epochs, evaluation, checkpoints and run records."""

import json
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import inf
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from lenahs.model.adapter import ModelData
from lenahs.model.architecture import CorrectionRunConfig, build_correction_model
from lenahs.model.correction import CorrectionModelOutput
from lenahs.training.data import ObservationDataset, make_loader
from lenahs.training.losses import correction_losses
from lenahs.training.metrics import batch_image_metrics, summarize_image_metrics

# The checkpoint uses the same image weighting as the loss
SELECTION_METRIC = "all/corrective_parent_accuracy/image_mean"
PROGRESS_INTERVAL = 200


@dataclass(frozen=True)
class PreliminaryRunSettings:
    dataset_root: str
    include_noisy: bool
    run: CorrectionRunConfig
    max_epochs: int
    plateau_patience: int
    early_stopping_patience: int
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    plateau_factor: float = 0.1
    gradient_clip_norm: float = 1.0
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 0

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "run": self.run.to_dict()}


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _forward(
    model: nn.Module, data: ModelData, settings: PreliminaryRunSettings
) -> CorrectionModelOutput:
    return model(
        data.x,
        data.edge_index,
        data.ptr,
        settings.run.observed_steps,
        settings.run.correction_steps,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    settings: PreliminaryRunSettings,
    device: torch.device,
) -> dict[str, float]:
    """Mean training losses over the epoch's images."""
    model.train()
    totals = defaultdict(float)
    graph_count = 0
    started = time.perf_counter()

    for batch_index, data in enumerate(loader, start=1):
        data = data.to(device)
        losses = correction_losses(_forward(model, data, settings), data)

        optimizer.zero_grad()
        losses.total().backward()
        clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
        optimizer.step()

        # Weight each batch by its graphs, since the last batch can be smaller
        for name, value in losses.as_floats().items():
            totals[name] += value * data.num_graphs
        graph_count += data.num_graphs

        if batch_index % PROGRESS_INTERVAL == 0:
            running = sum(totals.values()) / graph_count
            seconds = time.perf_counter() - started
            print(
                f"  batch {batch_index}/{len(loader)}  loss {running:.4f}  {seconds:.0f}s",
                flush=True,
            )

    return {name: total / graph_count for name, total in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    settings: PreliminaryRunSettings,
    device: torch.device,
) -> dict[str, float]:
    """Split losses, accuracies, baselines and tie counts."""
    model.eval()
    per_image = defaultdict(list)
    loss_totals = defaultdict(float)
    graph_count = 0

    for data in loader:
        data = data.to(device)
        output = _forward(model, data, settings)
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


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    # Write then rename, so a disconnect never leaves a truncated checkpoint
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device)


def _write_json(path: Path, content: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2))
    temporary.replace(path)


def _git_revision() -> str | None:
    # A "-dirty" suffix records that uncommitted changes were part of the run
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fit(
    settings: PreliminaryRunSettings,
    run_directory: str | Path,
    resume: bool = False,
) -> dict[str, float]:
    """Train, keep the best validation checkpoint and evaluate test once."""
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    last_path = run_directory / "last.pt"
    best_path = run_directory / "best.pt"

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    device = select_device()

    def loader(split: str, shuffle: bool) -> DataLoader:
        dataset = ObservationDataset(
            settings.dataset_root, split, settings.include_noisy
        )
        return make_loader(
            dataset, settings.batch_size, shuffle, settings.seed, settings.num_workers
        )

    train_loader = loader("train", shuffle=True)
    validation_loader = loader("val", shuffle=False)

    model = build_correction_model(settings.run.model).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=settings.plateau_factor,
        patience=settings.plateau_patience,
    )

    first_epoch, best, stale, history = 0, -inf, 0, []
    if resume and last_path.exists():
        # A resumed run continues the schedule, but its shuffling order differs
        # from an uninterrupted run
        state = load_checkpoint(last_path, device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        first_epoch = state["epoch"] + 1
        best, stale, history = state["best"], state["stale"], state["history"]
    else:
        manifest = Path(settings.dataset_root) / "manifest.json"
        _write_json(
            run_directory / "settings.json",
            {
                "settings": settings.to_dict(),
                "git": _git_revision(),
                "torch": torch.__version__,
                "device": str(device),
                "dataset_manifest": json.loads(manifest.read_text()),
            },
        )

    for epoch in range(first_epoch, settings.max_epochs):
        if stale >= settings.early_stopping_patience:
            break

        started = time.perf_counter()
        train = train_epoch(model, train_loader, optimizer, settings, device)
        validation = evaluate(model, validation_loader, settings, device)
        score = validation[SELECTION_METRIC]
        scheduler.step(score)

        improved = score > best
        best, stale = (score, 0) if improved else (best, stale + 1)
        history.append(
            {
                "epoch": epoch,
                "seconds": time.perf_counter() - started,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train,
                "validation": validation,
            }
        )

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best": best,
            "stale": stale,
            "history": history,
        }
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(best_path, state)
        _write_json(run_directory / "history.json", history)

        print(
            f"epoch {epoch}  train loss {sum(train.values()):.4f}  "
            f"val corrective {score:.4f}  best {best:.4f}  "
            f"lr {optimizer.param_groups[0]['lr']:.1e}  "
            f"{history[-1]['seconds']:.0f}s",
            flush=True,
        )

    if not best_path.exists():
        raise RuntimeError("No epoch produced a finite validation score.")
    model.load_state_dict(load_checkpoint(best_path, device)["model"])
    test = evaluate(model, loader("test", shuffle=False), settings, device)
    _write_json(run_directory / "test.json", test)
    return test
