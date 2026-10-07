import tracemalloc

import numpy as np
import pytest

from lenahs.component_tree.reference import (
    build_reference_component_tree,
    build_reference_rag,
)
from lenahs.component_tree.types import ComponentTree, ComponentTreeKind, FlatZoneGraph
from lenahs.generation import derive_correction_targets


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
@pytest.mark.parametrize(
    "clean, observed, parents, actions",
    [
        ([10], [10], [0], [1]),
        ([10, 20, 10], [10, 20, 10], [0, 0, 0], [1, 2, 1]),
        ([30, 20, 10], [30, 20, 10], [1, 2, 2], [2, 2, 1]),
        ([20, 20, 20], [10, 30, 10], [0, 0, 0], [1, 0, 0]),
        ([10, 10, 20], [1, 2, 3], [0, 0, 0], [1, 0, 2]),
        ([10, 20, 30], [30, 20, 10], [0, 0, 1], [1, 2, 2]),
        ([10, 20, 30], [7, 7, 8], [0, 0], [1, 2]),
        ([20, 10, 30], [7, 7, 8], [0, 0], [1, 1]),
        ([10, 20, 20], [7, 7, 7], [0], [1]),
    ],
    ids=[
        "one-pixel",
        "equal-level-canonical-link",
        "root-at-last-zone",
        "fragments-with-repeated-observed-level",
        "different-clean-and-observed-ids",
        "branch-reversal",
        "missing-ancestor",
        "missing-root-with-unequal-levels",
        "only-nonroot-role-survives",
    ],
)
def test_named_correction_targets(kind, clean, observed, parents, actions):
    clean_image = np.array([clean], dtype=np.uint8)
    observed_image = np.array([observed], dtype=np.uint8)
    if kind == ComponentTreeKind.MIN_TREE:
        clean_image = 255 - clean_image
        observed_image = 255 - observed_image

    clean_graph = build_reference_rag(clean_image)
    clean_tree = build_reference_component_tree(clean_graph, kind)
    observed_graph = build_reference_rag(observed_image)

    targets, _ = derive_correction_targets(clean_graph, clean_tree, observed_graph)

    np.testing.assert_array_equal(targets.corrective_parents, parents)
    np.testing.assert_array_equal(targets.actions, actions)
    assert targets.corrective_parents.dtype == np.int64
    assert targets.actions.dtype == np.int64


def _permute_graph(graph, permutation):
    original_ids = np.argsort(permutation)
    return FlatZoneGraph(
        label_map=permutation[graph.label_map],
        edge_index=permutation[graph.edge_index],
        levels=graph.levels[original_ids],
        pixel_counts=graph.pixel_counts[original_ids],
        anchors=graph.anchors[original_ids],
    )


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
@pytest.mark.parametrize("permute_clean", [False, True])
@pytest.mark.parametrize("permute_observed", [False, True])
def test_overlap_and_root_ties_follow_anchors(kind, permute_clean, permute_observed):
    clean_image = np.array([[20, 10, 20]], dtype=np.uint8)
    observed_image = np.array([[7, 7, 8]], dtype=np.uint8)
    if kind == ComponentTreeKind.MIN_TREE:
        clean_image = 255 - clean_image
        observed_image = 255 - observed_image

    clean_graph = build_reference_rag(clean_image)
    clean_tree = build_reference_component_tree(clean_graph, kind)
    observed_graph = build_reference_rag(observed_image)

    clean_permutation = np.array([2, 1, 0]) if permute_clean else np.arange(3)
    observed_permutation = np.array([1, 0]) if permute_observed else np.arange(2)
    clean_graph = _permute_graph(clean_graph, clean_permutation)
    observed_graph = _permute_graph(observed_graph, observed_permutation)
    clean_tree = ComponentTree(
        kind=kind,
        parents=clean_permutation[clean_tree.parents[np.argsort(clean_permutation)]],
    )

    targets, diagnostics = derive_correction_targets(
        clean_graph, clean_tree, observed_graph
    )

    # The overlap tie loses the clean root. The root tie keeps the first anchor.
    np.testing.assert_array_equal(
        targets.corrective_parents[observed_permutation],
        [observed_permutation[0], observed_permutation[0]],
    )
    np.testing.assert_array_equal(targets.actions[observed_permutation], [1, 1])
    np.testing.assert_array_equal(
        diagnostics.overlap_clean_zone_count[observed_permutation], [2, 1]
    )
    np.testing.assert_allclose(
        diagnostics.overlap_purity[observed_permutation], [0.5, 1.0]
    )
    np.testing.assert_array_equal(
        diagnostics.missing_clean_zones[clean_permutation], [False, True, False]
    )
    assert diagnostics.clean_root_missing is True
    assert diagnostics.artificial_root_attachment_count == 1


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
@pytest.mark.parametrize(
    "clean, observed, overlap_count, purity, missing, root_missing, attachment_count",
    [
        (
            [10, 20, 10],
            [10, 20, 10],
            [1, 1, 1],
            [1, 1, 1],
            [False, False, False],
            False,
            0,
        ),
        ([20, 20, 20], [10, 30, 10], [1, 1, 1], [1, 1, 1], [False], False, 0),
        (
            [10, 10, 20, 30],
            [7, 7, 7, 8],
            [2, 1],
            [2 / 3, 1],
            [False, True, False],
            False,
            0,
        ),
        (
            [20, 10, 30],
            [7, 7, 8],
            [2, 1],
            [0.5, 1],
            [False, True, False],
            True,
            1,
        ),
        ([10, 20, 20], [7, 7, 7], [2], [2 / 3], [True, False], True, 0),
        (
            [30, 20, 10],
            [7, 8, 8],
            [1, 2],
            [1, 0.5],
            [False, False, True],
            True,
            0,
        ),
        (
            [20, 10, 30, 10, 40],
            [7, 7, 8, 8, 9],
            [2, 2, 1],
            [0.5, 0.5, 1],
            [False, True, False, True, False],
            True,
            2,
        ),
    ],
    ids=[
        "clean-identity",
        "pure-fragments",
        "unequal-mixed-overlap",
        "missing-root-with-two-forest-roots",
        "missing-root-with-one-observed-zone",
        "missing-last-clean-root-with-one-forest-root",
        "three-forest-roots",
    ],
)
def test_named_correction_diagnostics(
    kind,
    clean,
    observed,
    overlap_count,
    purity,
    missing,
    root_missing,
    attachment_count,
):
    clean_image = np.array([clean], dtype=np.uint8)
    observed_image = np.array([observed], dtype=np.uint8)
    if kind == ComponentTreeKind.MIN_TREE:
        clean_image = 255 - clean_image
        observed_image = 255 - observed_image

    clean_graph = build_reference_rag(clean_image)
    clean_tree = build_reference_component_tree(clean_graph, kind)
    observed_graph = build_reference_rag(observed_image)

    _, diagnostics = derive_correction_targets(clean_graph, clean_tree, observed_graph)

    np.testing.assert_array_equal(diagnostics.overlap_clean_zone_count, overlap_count)
    np.testing.assert_allclose(diagnostics.overlap_purity, purity)
    np.testing.assert_array_equal(diagnostics.missing_clean_zones, missing)
    assert diagnostics.clean_root_missing is root_missing
    assert diagnostics.artificial_root_attachment_count == attachment_count
    assert diagnostics.overlap_clean_zone_count.dtype == np.int64
    assert diagnostics.overlap_purity.dtype == np.float64
    assert diagnostics.missing_clean_zones.dtype == np.bool_


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_long_missing_ancestor_chain(kind):
    clean_image = np.arange(256, dtype=np.uint8).reshape(1, 256)
    observed_image = np.zeros((1, 256), dtype=np.uint8)
    observed_image[0, 254:] = [1, 2]
    if kind == ComponentTreeKind.MIN_TREE:
        clean_image = 255 - clean_image
        observed_image = 255 - observed_image

    clean_graph = build_reference_rag(clean_image)
    clean_tree = build_reference_component_tree(clean_graph, kind)
    observed_graph = build_reference_rag(observed_image)

    targets, _ = derive_correction_targets(clean_graph, clean_tree, observed_graph)

    # The first zone selects role zero. Roles 254 and 255 keep one increment each.
    np.testing.assert_array_equal(targets.corrective_parents, [0, 0, 1])
    np.testing.assert_array_equal(targets.actions, [1, 2, 2])


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_checkerboard_identity_at_full_image_scale(kind):
    rows, columns = np.indices((256, 256))
    image = ((rows + columns) % 2).astype(np.uint8)
    graph = build_reference_rag(image)
    tree = build_reference_component_tree(graph, kind)

    assert graph.levels.size == 65_536
    tracemalloc.start()
    try:
        targets, _ = derive_correction_targets(graph, tree, graph)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Sparse construction fits comfortably here. A Cartesian matrix needs gigabytes.
    assert peak_bytes < 64 * 1024**2

    np.testing.assert_array_equal(targets.corrective_parents, tree.parents)
    expected_actions = np.where(graph.levels == graph.levels[tree.parents], 1, 2)
    np.testing.assert_array_equal(targets.actions, expected_actions)
