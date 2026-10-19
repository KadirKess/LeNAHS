"""A training run: its settings, epochs, checkpoint selection and test result."""

import json
import random
import time
from dataclasses import asdict, dataclass
from math import inf
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from lenahs.model.architecture import CorrectionRunConfig, build_correction_model
from lenahs.training.data import ObservationDataset, make_loader
from lenahs.training.epochs import evaluate, select_device, train_epoch
from lenahs.training.records import (
    git_revision,
    load_checkpoint,
    save_checkpoint,
    write_json,
)

# The checkpoint uses the same image weighting as the loss
SELECTION_METRIC = "all/joint_correction_accuracy/image_mean"


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


LOSS_NAMES = ("observed_parent", "corrective_parent", "action")
ACCURACY_NAMES = (
    "observed_parent_accuracy",
    "corrective_parent_accuracy",
    "action_accuracy",
    "joint_correction_accuracy",
)


def epoch_report(history: list[dict], stale: int, patience: int) -> str:
    """Console summary of the latest epoch, one row per loss or accuracy source.

    Train losses are means over the epoch while the weights change, so they are
    not directly comparable with validation losses measured after it.
    """
    entry = history[-1]
    train, validation = entry["train"], entry["validation"]
    # The best checkpoint is the last epoch that improved, stale epochs ago.
    # Every epoch so far is stale when no score has been finite.
    best = history[-1 - stale] if stale < len(history) else None

    def row(label: str, values: list[float]) -> str:
        return f"  {label:<24}" + "".join(f"{value:>12.4f}" for value in values)

    lines = [
        f"epoch {entry['epoch']}  lr {entry['learning_rate']:.1e}  "
        f"{entry['seconds']:.0f}s",
        f"  {'loss':<24}"
        + "".join(f"{name:>12}" for name in ("observed", "corrective", "action"))
        + f"{'total':>12}",
        row(
            "train (epoch mean)", [train[n] for n in LOSS_NAMES] + [sum(train.values())]
        ),
        row(
            "val",
            [validation[f"loss/{n}"] for n in LOSS_NAMES]
            + [sum(validation[f"loss/{n}"] for n in LOSS_NAMES)],
        ),
        f"  {'accuracy, image mean':<24}"
        + "".join(
            f"{name:>12}" for name in ("observed", "corrective", "action", "joint")
        ),
    ]
    # A clean-only run has no noisy group
    for group in ("clean", "noisy"):
        if f"{group}/images" in validation:
            lines.append(
                row(
                    f"val {group}",
                    [validation[f"{group}/{n}/image_mean"] for n in ACCURACY_NAMES],
                )
            )
    best_text = (
        "none yet"
        if best is None
        else f"{best['validation'][SELECTION_METRIC]:.4f} at epoch {best['epoch']}"
    )
    lines.append(
        f"  selection (val joint, image mean) {validation[SELECTION_METRIC]:.4f}  "
        f"best {best_text}  stale {stale}/{patience}"
    )
    return "\n".join(lines)


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
        write_json(
            run_directory / "settings.json",
            {
                "settings": settings.to_dict(),
                "selection_metric": SELECTION_METRIC,
                "git": git_revision(),
                "torch": torch.__version__,
                "device": str(device),
                "dataset_manifest": json.loads(manifest.read_text()),
            },
        )

    for epoch in range(first_epoch, settings.max_epochs):
        if stale >= settings.early_stopping_patience:
            break

        started = time.perf_counter()
        train = train_epoch(
            model,
            train_loader,
            optimizer,
            settings.run,
            settings.gradient_clip_norm,
            device,
        )
        validation = evaluate(model, validation_loader, settings.run, device)
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
        write_json(run_directory / "history.json", history)

        # Early stopping can end the run before this estimate
        mean_seconds = sum(entry["seconds"] for entry in history) / len(history)
        remaining_hours = mean_seconds * (settings.max_epochs - epoch - 1) / 3600
        print(
            epoch_report(history, stale, settings.early_stopping_patience)
            + f"\n  at most {remaining_hours:.1f}h left\n",
            flush=True,
        )

    if not best_path.exists():
        raise RuntimeError("No epoch produced a finite validation score.")
    model.load_state_dict(load_checkpoint(best_path, device)["model"])
    test = evaluate(model, loader("test", shuffle=False), settings.run, device)
    write_json(run_directory / "test.json", test)
    return test
