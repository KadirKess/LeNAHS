from numbers import Integral

import numpy as np

from lenahs.dataset.records import SampleFamily
from lenahs.generation.samples import GenerationSample, TreeImage
from lenahs.generation.targets import CorrectionAction


def validate_sample_family(family: SampleFamily) -> None:
    validate_nonnegative_integer("sample_index", family.sample_index)
    _validate_tree_image(family.clean.clean)
    if not _same_tree_image(family.clean.clean, family.clean.observed):
        raise ValueError("The clean family observation must equal its clean reference.")
    _validate_generation_sample(family.clean)

    realization_indices = []
    for observation in family.noisy:
        validate_nonnegative_integer("realization_index", observation.realization_index)
        realization_indices.append(observation.realization_index)
        if not _same_tree_image(family.clean.clean, observation.sample.clean):
            raise ValueError(
                "Each noisy observation must reference the family clean image."
            )
        _validate_generation_sample(observation.sample)

    if len(set(realization_indices)) != len(realization_indices):
        raise ValueError("Realization indices must be unique within a sample family.")


def validate_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    if value > np.iinfo(np.int64).max:
        raise ValueError(f"{name} must fit in a signed 64-bit integer.")


def _validate_generation_sample(sample: GenerationSample) -> None:
    if sample.observed is not sample.clean:
        _validate_tree_image(sample.observed)
    if sample.observed.image.shape != sample.clean.image.shape:
        raise ValueError("Clean and observed images must have the same shape.")
    if sample.observed.component_tree.kind != sample.clean.component_tree.kind:
        raise ValueError("Clean and observed component-tree kinds must match.")
    observed_nodes = sample.observed.graph.levels.size
    clean_nodes = sample.clean.graph.levels.size
    targets = sample.correction_targets
    diagnostics = sample.correction_diagnostics

    for name, values, dtype, count in [
        ("corrective_parents", targets.corrective_parents, np.int64, observed_nodes),
        ("actions", targets.actions, np.int64, observed_nodes),
        ("overlap_purity", diagnostics.overlap_purity, np.float64, observed_nodes),
        (
            "overlap_clean_zone_count",
            diagnostics.overlap_clean_zone_count,
            np.int64,
            observed_nodes,
        ),
        ("missing_clean_zones", diagnostics.missing_clean_zones, np.bool_, clean_nodes),
    ]:
        if values.shape != (count,):
            raise ValueError(f"{name} must have shape ({count},).")
        if values.dtype != dtype:
            raise TypeError(f"{name} must have dtype {np.dtype(dtype)}.")

    parents = targets.corrective_parents
    if np.any((parents < 0) | (parents >= observed_nodes)):
        raise ValueError("corrective_parents must use local observed node identifiers.")
    if np.any((targets.actions < 0) | (targets.actions > 2)):
        raise ValueError("actions must be MERGE, KEEP_ZERO or KEEP_ONE.")
    roots = np.flatnonzero(parents == np.arange(observed_nodes))
    if roots.size != 1 or targets.actions[roots[0]] != CorrectionAction.KEEP_ZERO:
        raise ValueError(
            "Correction targets must have one self-parented KEEP_ZERO root."
        )
    _validate_rooted_pointers(parents, int(roots[0]))

    purity = diagnostics.overlap_purity
    if np.any(~np.isfinite(purity) | (purity <= 0) | (purity > 1)):
        raise ValueError("overlap_purity must be finite and in (0, 1].")
    overlap_counts = diagnostics.overlap_clean_zone_count
    if np.any((overlap_counts < 1) | (overlap_counts > clean_nodes)):
        raise ValueError(
            "overlap_clean_zone_count must lie between 1 and clean node count."
        )
    if not isinstance(diagnostics.clean_root_missing, bool):
        raise TypeError("clean_root_missing must be a Boolean.")
    attachments = diagnostics.artificial_root_attachment_count
    validate_nonnegative_integer("artificial_root_attachment_count", attachments)
    if attachments >= observed_nodes:
        raise ValueError(
            "artificial_root_attachment_count must be less than node count."
        )


def _validate_rooted_pointers(parents: np.ndarray, root: int) -> None:
    # Once a path reaches the root, later paths may stop at any of its nodes.
    # The active-path mark catches cycles without repeatedly walking long chains.
    state = np.zeros(parents.size, dtype=np.uint8)
    state[root] = 2
    for start in range(parents.size):
        node = start
        path = []
        while state[node] == 0:
            state[node] = 1
            path.append(node)
            node = int(parents[node])
        if state[node] == 1:
            raise ValueError("Correction pointers contain a cycle outside the root.")
        for node in path:
            state[node] = 2


def _validate_tree_image(tree_image: TreeImage) -> None:
    if tree_image.image.ndim != 2 or tree_image.image.dtype != np.uint8:
        raise TypeError("Tree images must be two-dimensional uint8 arrays.")

    graph = tree_image.graph
    node_count = graph.levels.size
    if node_count == 0 or tree_image.image.size == 0:
        raise ValueError("Tree images and graphs must be nonempty.")
    if graph.label_map.shape != tree_image.image.shape:
        raise ValueError("A graph label map must have the same shape as its image.")
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[1] != 2:
        raise ValueError("edge_index must have shape (edge_count, 2).")
    for name, values in [
        ("levels", graph.levels),
        ("pixel_counts", graph.pixel_counts),
        ("anchors", graph.anchors),
        ("parents", tree_image.component_tree.parents),
    ]:
        if values.shape != (node_count,):
            raise ValueError(f"{name} must have one value per graph node.")

    if not np.issubdtype(graph.label_map.dtype, np.integer):
        raise TypeError("label_map must have an integer dtype.")
    if not np.issubdtype(graph.edge_index.dtype, np.integer):
        raise TypeError("edge_index must have an integer dtype.")
    if graph.levels.dtype != np.uint8:
        raise TypeError("levels must have dtype uint8.")
    if not np.issubdtype(graph.pixel_counts.dtype, np.integer):
        raise TypeError("pixel_counts must have an integer dtype.")
    if not np.issubdtype(graph.anchors.dtype, np.integer):
        raise TypeError("anchors must have an integer dtype.")
    if not np.issubdtype(tree_image.component_tree.parents.dtype, np.integer):
        raise TypeError("parents must have an integer dtype.")


def _same_tree_image(first: TreeImage, second: TreeImage) -> bool:
    if first is second:
        return True
    return (
        first.component_tree.kind == second.component_tree.kind
        and np.array_equal(first.image, second.image)
        and np.array_equal(first.graph.label_map, second.graph.label_map)
        and np.array_equal(first.graph.edge_index, second.graph.edge_index)
        and np.array_equal(first.graph.levels, second.graph.levels)
        and np.array_equal(first.graph.pixel_counts, second.graph.pixel_counts)
        and np.array_equal(first.graph.anchors, second.graph.anchors)
        and np.array_equal(first.component_tree.parents, second.component_tree.parents)
    )
