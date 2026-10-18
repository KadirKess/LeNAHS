from types import SimpleNamespace

import pytest
import torch

from lenahs.generation import CorrectionAction
from lenahs.training.metrics import (
    batch_image_metrics,
    graph_local_argmax,
    summarize_image_metrics,
)


def test_argmax_returns_local_ids_and_first_index_on_ties() -> None:
    # Graph A has two nodes. Graph B has three, and its first node ties
    query = torch.tensor([[1.0], [1.0], [1.0], [1.0], [-1.0]])
    key = torch.tensor([[0.0], [2.0], [3.0], [3.0], [1.0]])

    predictions, tied = graph_local_argmax(query, key, torch.tensor([0, 2, 5]))

    assert predictions.tolist() == [1, 1, 0, 0, 2]
    assert tied.tolist() == [False, False, True, True, False]


def _metric_batch(values, node_counts, clean):
    names = (
        "observed_parent_accuracy",
        "corrective_parent_accuracy",
        "action_accuracy",
        "copy_observed_baseline",
        "always_merge_baseline",
    )
    batch = {name: torch.tensor(values) for name in names}
    batch["observed_ties"] = torch.zeros(len(values), dtype=torch.long)
    batch["corrective_ties"] = torch.zeros(len(values), dtype=torch.long)
    batch["node_count"] = torch.tensor(node_counts)
    batch["is_clean"] = torch.tensor(clean)
    return batch


def test_summary_reports_image_mean_and_pooled_over_unequal_batches() -> None:
    # Graph A has 2 nodes and half correct. Graph B, alone in a smaller final
    # batch, has 3 nodes all correct.
    batches = [_metric_batch([0.5], [2], [True]), _metric_batch([1.0], [3], [False])]
    per_image = {name: [b[name] for b in batches] for name in batches[0]}

    summary = summarize_image_metrics(per_image)

    assert summary["all/corrective_parent_accuracy/image_mean"] == pytest.approx(0.75)
    assert summary["all/corrective_parent_accuracy/pooled"] == pytest.approx(0.8)
    assert summary["clean/corrective_parent_accuracy/image_mean"] == pytest.approx(0.5)
    assert summary["noisy/corrective_parent_accuracy/pooled"] == pytest.approx(1.0)
    assert (summary["all/images"], summary["all/nodes"]) == (2, 5)


def test_clean_only_summary_has_no_noisy_group() -> None:
    per_image = {
        name: [value] for name, value in _metric_batch([1.0], [4], [True]).items()
    }

    summary = summarize_image_metrics(per_image)

    assert "clean/action_accuracy/image_mean" in summary
    assert not any(name.startswith("noisy/") for name in summary)


def test_baselines_on_a_clean_observation() -> None:
    # A clean observation keeps every node, so corrective parents equal observed ones
    parents = torch.tensor([0, 0, 1])
    actions = torch.full((3,), int(CorrectionAction.KEEP_ONE))
    actions[0] = int(CorrectionAction.KEEP_ZERO)
    hidden = torch.eye(3)
    step = SimpleNamespace(
        parent_query=hidden,
        parent_key=hidden,
        corrective_query=hidden,
        corrective_key=hidden,
        action_logits=torch.zeros(3, 3),
    )
    output = SimpleNamespace(observed_steps=(step,), correction_steps=(step,))
    data = SimpleNamespace(
        num_graphs=1,
        ptr=torch.tensor([0, 3]),
        batch=torch.zeros(3, dtype=torch.long),
        observed_parents=parents,
        corrective_parents=parents,
        correction_actions=actions,
        is_clean=torch.tensor([True]),
    )

    metrics = batch_image_metrics(output, data)

    assert metrics["copy_observed_baseline"].tolist() == [1.0]
    assert metrics["always_merge_baseline"].tolist() == [0.0]
    assert metrics["node_count"].tolist() == [3]
