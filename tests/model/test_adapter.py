import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from lenahs.component_tree import (
    ComponentTree,
    ComponentTreeKind,
    FlatZoneGraph,
    build_reference_component_tree,
    build_reference_rag,
)
from lenahs.generation import (
    GenerationSample,
    TreeImage,
    build_generation_sample,
    derive_correction_targets,
)
from lenahs.model import (
    ModelData,
    adapt_generation_sample,
)


def _reference_sample(image: np.ndarray, kind=ComponentTreeKind.MAX_TREE):
    def build_tree(values, requested_kind):
        graph = build_reference_rag(values)
        return graph, build_reference_component_tree(graph, requested_kind)

    return build_generation_sample(image, kind, tree_builder=build_tree)


def _tree_image(
    image,
    label_map,
    edges,
    levels,
    pixel_counts,
    anchors,
    parents,
):
    return TreeImage(
        image=np.asarray(image, dtype=np.uint8),
        graph=FlatZoneGraph(
            label_map=np.asarray(label_map, dtype=np.int64),
            edge_index=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
            levels=np.asarray(levels, dtype=np.uint8),
            pixel_counts=np.asarray(pixel_counts, dtype=np.int64),
            anchors=np.asarray(anchors, dtype=np.int64),
        ),
        component_tree=ComponentTree(
            kind=ComponentTreeKind.MAX_TREE,
            parents=np.asarray(parents, dtype=np.int64),
        ),
    )


def _sample(clean, observed=None):
    if observed is None:
        observed = clean
    targets, diagnostics = derive_correction_targets(
        clean.graph, clean.component_tree, observed.graph
    )
    return GenerationSample(
        clean=clean,
        observed=observed,
        correction_targets=targets,
        correction_diagnostics=diagnostics,
    )


def test_exact_features_use_observed_non_square_image_area() -> None:
    tree_image = _tree_image(
        [[0, 0, 128], [255, 255, 255]],
        [[0, 0, 1], [2, 2, 2]],
        [(0, 1), (1, 2)],
        [0, 128, 255],
        [2, 1, 3],
        [0, 2, 3],
        [0, 0, 1],
    )

    adapted = adapt_generation_sample(_sample(tree_image))

    assert adapted.x.dtype == torch.float32
    torch.testing.assert_close(
        adapted.x,
        torch.tensor(
            [[0.0, 2 / 6, 0.0], [128 / 255, 1 / 6, 2 / 5], [1.0, 3 / 6, 3 / 5]],
            dtype=torch.float32,
        ),
    )


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_one_pixel_features_and_targets_are_finite_for_both_tree_kinds(kind) -> None:
    adapted = adapt_generation_sample(
        _reference_sample(np.array([[7]], dtype=np.uint8), kind)
    )

    torch.testing.assert_close(
        adapted.x,
        torch.tensor([[7 / 255, 1.0, 0.0]]),
    )
    assert torch.isfinite(adapted.x).all()
    torch.testing.assert_close(adapted.edge_index, torch.tensor([[0], [0]]))
    torch.testing.assert_close(adapted.observed_parents, torch.tensor([0]))
    torch.testing.assert_close(
        adapted.correction_actions,
        torch.tensor([1]),
    )


def test_corrupted_sample_selects_only_observed_graph_and_parent_targets() -> None:
    clean = _tree_image(
        [[10, 20]], [[0, 1]], [(0, 1)], [10, 20], [1, 1], [0, 1], [0, 0]
    )
    corrupted = _tree_image([[30, 30]], [[0, 0]], [], [30], [2], [0], [0])

    adapted = adapt_generation_sample(_sample(clean, corrupted))

    assert adapted.x.shape == (1, 3)
    torch.testing.assert_close(adapted.x[:, 0], torch.tensor([30 / 255]))
    torch.testing.assert_close(adapted.observed_parents, torch.tensor([0]))
    torch.testing.assert_close(adapted.correction_actions, torch.tensor([1]))
    assert not hasattr(adapted, "image")
    assert not hasattr(adapted, "label_map")
    assert not hasattr(adapted, "overlap_purity")
    assert not hasattr(adapted, "noise_config")


def test_anchor_feature_exposes_an_asymmetry_missing_from_level_and_area() -> None:
    sample = _reference_sample(np.array([[2, 3, 2]], dtype=np.uint8))
    adapted = adapt_generation_sample(sample)
    swap = torch.tensor([2, 1, 0])

    torch.testing.assert_close(
        adapted.x[:, :2],
        adapted.x[swap, :2],
    )
    assert not torch.equal(
        adapted.x,
        adapted.x[swap],
    )

    swapped_edges = swap[adapted.edge_index]
    swapped_edge_pairs = sorted(map(tuple, swapped_edges.T.tolist()))
    assert swapped_edge_pairs == sorted(map(tuple, adapted.edge_index.T.tolist()))

    relabeled_parents = swap[adapted.observed_parents[swap]]
    assert not torch.equal(adapted.observed_parents, relabeled_parents)
    torch.testing.assert_close(adapted.observed_parents, torch.tensor([0, 0, 0]))


def test_edges_are_bidirectional_self_looped_deduplicated_and_sorted() -> None:
    tree_image = _tree_image(
        [[1, 2, 3]],
        [[0, 1, 2]],
        [(2, 1), (0, 1), (1, 0), (2, 1)],
        [1, 2, 3],
        [1, 1, 1],
        [0, 1, 2],
        [0, 0, 1],
    )

    adapted = adapt_generation_sample(_sample(tree_image))

    torch.testing.assert_close(
        adapted.edge_index,
        torch.tensor(
            [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]],
            dtype=torch.int64,
        ),
    )


def test_pyg_loader_offsets_edges_but_not_parent_targets() -> None:
    three_nodes = adapt_generation_sample(
        _reference_sample(np.array([[2, 3, 2]], dtype=np.uint8))
    )
    one_node = adapt_generation_sample(
        _reference_sample(np.array([[7]], dtype=np.uint8))
    )

    packed = next(
        iter(DataLoader([three_nodes, one_node], batch_size=2, shuffle=False))
    )

    torch.testing.assert_close(packed.ptr, torch.tensor([0, 3, 4]))
    torch.testing.assert_close(packed.batch, torch.tensor([0, 0, 0, 1]))
    torch.testing.assert_close(packed.observed_parents, torch.tensor([0, 0, 0, 0]))
    assert tuple(packed.edge_index[:, -1].tolist()) == (3, 3)
    assert not torch.any((packed.edge_index[0] < 3) & (packed.edge_index[1] >= 3))
    assert not torch.any((packed.edge_index[0] >= 3) & (packed.edge_index[1] < 3))

    restored = packed.to_data_list()
    for actual, expected in zip(restored, [three_nodes, one_node], strict=True):
        torch.testing.assert_close(actual.x, expected.x)
        torch.testing.assert_close(actual.edge_index, expected.edge_index)
        torch.testing.assert_close(actual.observed_parents, expected.observed_parents)
        torch.testing.assert_close(
            actual.corrective_parents, expected.corrective_parents
        )
        torch.testing.assert_close(
            actual.correction_actions, expected.correction_actions
        )


def test_pyg_loader_preserves_nonzero_local_parents_in_later_graphs() -> None:
    first = adapt_generation_sample(
        _reference_sample(np.array([[7]], dtype=np.uint8), ComponentTreeKind.MIN_TREE)
    )
    second = adapt_generation_sample(
        _reference_sample(
            np.array([[10, 20]], dtype=np.uint8), ComponentTreeKind.MIN_TREE
        )
    )

    packed = next(iter(DataLoader([first, second], batch_size=2, shuffle=False)))

    torch.testing.assert_close(packed.ptr, torch.tensor([0, 1, 3]))
    torch.testing.assert_close(packed.observed_parents, torch.tensor([0, 1, 1]))
    torch.testing.assert_close(packed.corrective_parents, torch.tensor([0, 1, 1]))
    assert torch.any(packed.edge_index == 2)


def test_adapter_respects_requested_floating_dtype_and_device() -> None:
    sample = _reference_sample(np.array([[10, 20]], dtype=np.uint8))

    adapted = adapt_generation_sample(sample, dtype=torch.float64, device="cpu")

    assert adapted.x.dtype == torch.float64
    assert adapted.x.device == torch.device("cpu")
    assert adapted.edge_index.dtype == torch.int64
    assert adapted.observed_parents.dtype == torch.int64


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_adapter_rejects_non_floating_feature_dtype(dtype) -> None:
    sample = _reference_sample(np.array([[10]], dtype=np.uint8))

    with pytest.raises(TypeError, match="floating-point dtype"):
        adapt_generation_sample(sample, dtype=dtype)


def test_model_data_exposes_only_model_inputs_and_targets() -> None:
    adapted = adapt_generation_sample(
        _reference_sample(np.array([[10, 20]], dtype=np.uint8))
    )

    assert isinstance(adapted, ModelData)
    assert set(adapted.keys()) == {
        "x",
        "edge_index",
        "observed_parents",
        "corrective_parents",
        "correction_actions",
        "num_nodes",
    }


@pytest.mark.parametrize(
    ("kind", "observed_image", "observed_parents", "corrective_parents", "actions"),
    [
        (ComponentTreeKind.MAX_TREE, [[11, 12, 20]], [0, 0, 1], [0, 0, 0], [1, 0, 2]),
        (ComponentTreeKind.MIN_TREE, [[11, 12, 20]], [1, 2, 2], [2, 0, 2], [2, 0, 1]),
        (ComponentTreeKind.MAX_TREE, [[25, 25, 35]], [0, 0], [0, 0], [1, 2]),
        (ComponentTreeKind.MIN_TREE, [[25, 25, 35]], [1, 1], [1, 1], [2, 1]),
    ],
)
def test_refining_and_mixed_observations_have_three_exact_targets(
    kind, observed_image, observed_parents, corrective_parents, actions
):
    refining = len(observed_parents) == 3
    clean_image = [[10, 10, 20]] if refining else [[10, 20, 30]]
    clean = _reference_sample(np.asarray(clean_image, dtype=np.uint8), kind).clean
    observed = _reference_sample(np.asarray(observed_image, dtype=np.uint8), kind).clean
    adapted = adapt_generation_sample(_sample(clean, observed))
    torch.testing.assert_close(adapted.observed_parents, torch.tensor(observed_parents))
    torch.testing.assert_close(
        adapted.corrective_parents, torch.tensor(corrective_parents)
    )
    torch.testing.assert_close(adapted.correction_actions, torch.tensor(actions))
    expected_levels = [11, 12, 20] if refining else [25, 35]
    torch.testing.assert_close(adapted.x[:, 0], torch.tensor(expected_levels) / 255)


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_adapter_preserves_targets_under_independent_node_relabeling(kind):
    clean = _reference_sample(np.array([[10, 10, 20]], dtype=np.uint8), kind).clean
    observed = _reference_sample(np.array([[11, 12, 20]], dtype=np.uint8), kind).clean
    original = adapt_generation_sample(_sample(clean, observed))

    def permute(tree_image, permutation):
        old_order = np.argsort(permutation)
        graph = tree_image.graph
        return TreeImage(
            image=tree_image.image,
            graph=FlatZoneGraph(
                label_map=permutation[graph.label_map],
                edge_index=permutation[graph.edge_index],
                levels=graph.levels[old_order],
                pixel_counts=graph.pixel_counts[old_order],
                anchors=graph.anchors[old_order],
            ),
            component_tree=ComponentTree(
                kind=kind,
                parents=permutation[tree_image.component_tree.parents[old_order]],
            ),
        )

    clean_permutation = np.array([1, 0])
    observed_permutation = np.array([2, 0, 1])
    reordered = adapt_generation_sample(
        _sample(
            permute(clean, clean_permutation), permute(observed, observed_permutation)
        )
    )
    order = torch.as_tensor(observed_permutation)
    torch.testing.assert_close(reordered.x[order], original.x)
    torch.testing.assert_close(
        reordered.correction_actions[order], original.correction_actions
    )
    for key in ("observed_parents", "corrective_parents"):
        original_parents = getattr(original, key)
        reordered_parents = getattr(reordered, key)
        torch.testing.assert_close(reordered_parents[order], order[original_parents])
    remapped_edges = order[original.edge_index]
    assert sorted(map(tuple, reordered.edge_index.T.tolist())) == sorted(
        map(tuple, remapped_edges.T.tolist())
    )
