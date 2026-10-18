import json

import pytest
import torch
from torch_geometric.data import Batch

from lenahs.component_tree import ComponentTreeKind
from lenahs.dataset.build import build_sample_family, generate_dataset
from lenahs.generation import SaltAndPepperNoise, SquareImageConfig
from lenahs.model import (
    CorrectionModelConfig,
    CorrectionRunConfig,
    adapt_generation_sample,
    build_correction_model,
)
from lenahs.training.loop import PreliminaryRunSettings, fit
from lenahs.training.losses import correction_losses

SMALL_IMAGES = SquareImageConfig(
    height=16, width=16, min_squares=2, max_squares=4, min_side=2, max_side=8
)
NOISE = SaltAndPepperNoise(probability=0.05)
SMALL_RUN = CorrectionRunConfig(
    CorrectionModelConfig(hidden_width=16, correction_recurrent=True),
    observed_steps=2,
    correction_steps=1,
)


def test_all_three_losses_fall_on_a_few_graphs() -> None:
    # A learning check for target indexing and gradient flow, not learnability
    family = build_sample_family(
        1, 0, SMALL_IMAGES, ComponentTreeKind.MAX_TREE, NOISE, 2
    )
    samples = [family.clean, *(o.sample for o in family.noisy)]
    batch = Batch.from_data_list([adapt_generation_sample(s) for s in samples])
    torch.manual_seed(0)
    model = build_correction_model(SMALL_RUN.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    def losses():
        output = model(batch.x, batch.edge_index, batch.ptr, 2, 1)
        return correction_losses(output, batch)

    initial = losses().as_floats()
    for _ in range(150):
        optimizer.zero_grad()
        losses().total().backward()
        optimizer.step()
    final = losses().as_floats()

    for name in initial:
        assert final[name] < 0.5 * initial[name], name


@pytest.fixture
def dataset_root(tmp_path):
    root = tmp_path / "data"
    generate_dataset(
        root,
        2,
        {"train": 3, "val": 1, "test": 1},
        SMALL_IMAGES,
        ComponentTreeKind.MAX_TREE,
        NOISE,
        2,
    )
    return root


def _settings(root, max_epochs):
    return PreliminaryRunSettings(
        dataset_root=str(root),
        include_noisy=True,
        run=SMALL_RUN,
        max_epochs=max_epochs,
        plateau_patience=3,
        early_stopping_patience=10,
        batch_size=4,
    )


def test_fit_writes_run_records_and_resumes(dataset_root, tmp_path) -> None:
    run = tmp_path / "run"

    fit(_settings(dataset_root, 2), run)
    test = fit(_settings(dataset_root, 3), run, resume=True)

    history = json.loads((run / "history.json").read_text())
    assert [entry["epoch"] for entry in history] == [0, 1, 2]
    settings = json.loads((run / "settings.json").read_text())
    assert settings["settings"]["run"]["observed_steps"] == 2
    assert settings["dataset_manifest"]["split_sample_counts"]["train"] == 3
    assert json.loads((run / "test.json").read_text()) == test
    assert test["all/images"] == 3
    assert {"best.pt", "last.pt"} <= {path.name for path in run.iterdir()}
