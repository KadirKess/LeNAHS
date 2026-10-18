from math import exp, log

import pytest
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from lenahs.component_tree import ComponentTreeKind
from lenahs.dataset.build import build_sample_family
from lenahs.generation import SaltAndPepperNoise, SquareImageConfig
from lenahs.model import (
    CorrectionModelConfig,
    adapt_generation_sample,
    build_correction_model,
)
from lenahs.training.losses import correction_losses, image_mean, parent_cross_entropy

SMALL_IMAGES = SquareImageConfig(
    height=16, width=16, min_squares=2, max_squares=4, min_side=2, max_side=8
)


def test_parent_cross_entropy_matches_hand_calculation() -> None:
    # One node whose candidate scores are [2, 0, 1] with width one, target 2
    query = torch.tensor([[1.0], [0.0], [0.0]])
    key = torch.tensor([[2.0], [0.0], [1.0]])
    log_normalizer = torch.tensor([log(exp(2) + exp(0) + exp(1)), 0.0, 0.0])

    losses = parent_cross_entropy(
        query,
        key,
        log_normalizer,
        local_targets=torch.tensor([2, 0, 0]),
        graph_ptr=torch.tensor([0, 3]),
        batch=torch.zeros(3, dtype=torch.long),
    )

    assert losses[0].item() == pytest.approx(2.4076 - 1.0, abs=1e-4)


def test_image_mean_weights_images_equally() -> None:
    values = torch.tensor([0.2, 0.4, 1.0, 0.0, 0.5])
    batch = torch.tensor([0, 0, 1, 1, 1])

    assert image_mean(values, batch, 2).item() == pytest.approx(0.4)
    assert values.mean().item() == pytest.approx(0.42)


@pytest.fixture(scope="module")
def graphs():
    families = [
        build_sample_family(
            3,
            index,
            SMALL_IMAGES,
            ComponentTreeKind.MAX_TREE,
            SaltAndPepperNoise(0.05),
            2,
        )
        for index in range(2)
    ]
    samples = [s for f in families for s in (f.clean, *(o.sample for o in f.noisy))]
    return [adapt_generation_sample(sample, dtype=torch.float64) for sample in samples]


def _forward(model, graphs, order):
    batch = Batch.from_data_list([graphs[i] for i in order])
    return model(batch.x, batch.edge_index, batch.ptr, 2, 1), batch


def test_losses_match_dense_cross_entropy_per_image(graphs) -> None:
    torch.manual_seed(0)
    model = build_correction_model(CorrectionModelConfig(8, True)).double()
    output, batch = _forward(model, graphs, range(len(graphs)))
    observed, corrective = output.observed_steps[-1], output.correction_steps[-1]
    assert int(batch.corrective_parents[batch.ptr[1] :].max()) > 0

    expected = {"observed_parent": [], "corrective_parent": [], "action": []}
    bounds = batch.ptr.tolist()
    for index, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
        expected["observed_parent"].append(
            F.cross_entropy(
                observed.parent_scores[index], batch.observed_parents[start:stop]
            )
        )
        expected["corrective_parent"].append(
            F.cross_entropy(
                corrective.corrective_scores[index],
                batch.corrective_parents[start:stop],
            )
        )
        expected["action"].append(
            F.cross_entropy(
                corrective.action_logits[start:stop],
                batch.correction_actions[start:stop],
            )
        )

    losses = correction_losses(output, batch)
    for name, values in expected.items():
        reference = torch.stack(values).mean()
        actual = getattr(losses, name)
        assert actual.item() == pytest.approx(reference.item(), rel=1e-12)
        actual_gradients = torch.autograd.grad(
            actual, model.parameters(), retain_graph=True, allow_unused=True
        )
        reference_gradients = torch.autograd.grad(
            reference, model.parameters(), retain_graph=True, allow_unused=True
        )
        for a, b in zip(actual_gradients, reference_gradients, strict=True):
            if a is not None:
                torch.testing.assert_close(a, b)


def test_losses_do_not_depend_on_graph_order(graphs) -> None:
    torch.manual_seed(0)
    model = build_correction_model(CorrectionModelConfig(8, True)).double()

    forward = correction_losses(*_forward(model, graphs, [0, 1, 2, 3, 4, 5])).total()
    shuffled = correction_losses(*_forward(model, graphs, [5, 3, 1, 0, 2, 4])).total()

    assert forward.item() == pytest.approx(shuffled.item(), rel=1e-12)
