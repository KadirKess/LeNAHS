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
    CorruptionTargets,
    GenerationSample,
    NoiseDirection,
    TreeImage,
    build_generation_sample,
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


def _sample(clean, corrupted=None, directions=None):
    observed = corrupted if corrupted is not None else clean
    node_count = observed.graph.levels.size
    if directions is None:
        directions = np.zeros(node_count, dtype=np.int64)
    return GenerationSample(
        clean=clean,
        corrupted=corrupted,
        corruption_targets=CorruptionTargets(
            noise_direction=np.asarray(directions, dtype=np.int64),
            overlap_purity=np.ones(node_count, dtype=np.float64),
            overlap_clean_zone_count=np.ones(node_count, dtype=np.int64),
        ),
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
    torch.testing.assert_close(adapted.parents, torch.tensor([0]))
    torch.testing.assert_close(
        adapted.noise_direction,
        torch.tensor([NoiseDirection.UNCHANGED]),
    )


def test_corrupted_sample_selects_only_observed_graph_and_parent_targets() -> None:
    clean = _tree_image(
        [[10, 20]], [[0, 1]], [(0, 1)], [10, 20], [1, 1], [0, 1], [0, 0]
    )
    corrupted = _tree_image([[30, 30]], [[0, 0]], [], [30], [2], [0], [0])

    adapted = adapt_generation_sample(
        _sample(clean, corrupted, [NoiseDirection.BRIGHTENED])
    )

    assert adapted.x.shape == (1, 3)
    torch.testing.assert_close(adapted.x[:, 0], torch.tensor([30 / 255]))
    torch.testing.assert_close(adapted.parents, torch.tensor([0]))
    torch.testing.assert_close(adapted.noise_direction, torch.tensor([1]))
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

    relabeled_parents = swap[adapted.parents[swap]]
    assert not torch.equal(adapted.parents, relabeled_parents)
    torch.testing.assert_close(adapted.parents, torch.tensor([0, 0, 0]))


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
    torch.testing.assert_close(packed.parents, torch.tensor([0, 0, 0, 0]))
    assert tuple(packed.edge_index[:, -1].tolist()) == (3, 3)
    assert not torch.any((packed.edge_index[0] < 3) & (packed.edge_index[1] >= 3))
    assert not torch.any((packed.edge_index[0] >= 3) & (packed.edge_index[1] < 3))

    restored = packed.to_data_list()
    for actual, expected in zip(restored, [three_nodes, one_node], strict=True):
        torch.testing.assert_close(actual.x, expected.x)
        torch.testing.assert_close(actual.edge_index, expected.edge_index)
        torch.testing.assert_close(actual.parents, expected.parents)
        torch.testing.assert_close(actual.noise_direction, expected.noise_direction)


def test_pyg_loader_preserves_nonzero_local_parents_in_later_graphs() -> None:
    first = adapt_generation_sample(
        _sample(_tree_image([[7]], [[0]], [], [7], [1], [0], [0]))
    )
    second = adapt_generation_sample(
        _sample(
            _tree_image(
                [[10, 20]],
                [[0, 1]],
                [(0, 1)],
                [10, 20],
                [1, 1],
                [0, 1],
                [1, 1],
            )
        )
    )

    packed = next(iter(DataLoader([first, second], batch_size=2, shuffle=False)))

    torch.testing.assert_close(packed.ptr, torch.tensor([0, 1, 3]))
    torch.testing.assert_close(packed.parents, torch.tensor([0, 1, 1]))
    assert torch.any(packed.edge_index == 2)


def test_adapter_respects_requested_floating_dtype_and_device() -> None:
    sample = _reference_sample(np.array([[10, 20]], dtype=np.uint8))

    adapted = adapt_generation_sample(sample, dtype=torch.float64, device="cpu")

    assert adapted.x.dtype == torch.float64
    assert adapted.x.device == torch.device("cpu")
    assert adapted.edge_index.dtype == torch.int64
    assert adapted.parents.dtype == torch.int64


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
        "parents",
        "noise_direction",
        "num_nodes",
    }
