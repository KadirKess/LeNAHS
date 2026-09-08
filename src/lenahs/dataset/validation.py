from numbers import Integral

import numpy as np

from lenahs.dataset.records import SampleFamily
from lenahs.generation.samples import GenerationSample, TreeImage


def validate_sample_family(family: SampleFamily) -> None:
    validate_nonnegative_integer("sample_index", family.sample_index)
    if family.clean.corrupted is not None:
        raise ValueError("The clean family sample must not contain a corruption.")
    _validate_generation_sample(family.clean)

    realization_indices = []
    for observation in family.noisy:
        validate_nonnegative_integer("realization_index", observation.realization_index)
        realization_indices.append(observation.realization_index)
        if observation.sample.corrupted is None:
            raise ValueError("Each noisy observation must contain a corrupted image.")
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
    _validate_tree_image(sample.clean)
    target_nodes = sample.clean.graph.levels.size

    if sample.corrupted is not None:
        _validate_tree_image(sample.corrupted)
        if sample.corrupted.image.shape != sample.clean.image.shape:
            raise ValueError("Clean and corrupted images must have the same shape.")
        if sample.corrupted.component_tree.kind != sample.clean.component_tree.kind:
            raise ValueError("Clean and corrupted component-tree kinds must match.")
        target_nodes = sample.corrupted.graph.levels.size

    targets = sample.corruption_targets
    for name, values in [
        ("noise_direction", targets.noise_direction),
        ("overlap_purity", targets.overlap_purity),
        ("overlap_clean_zone_count", targets.overlap_clean_zone_count),
    ]:
        if values.shape != (target_nodes,):
            raise ValueError(f"{name} must have one value per target graph node.")

    if targets.noise_direction.dtype != np.int64:
        raise TypeError("noise_direction must have dtype int64.")
    if targets.overlap_purity.dtype != np.float64:
        raise TypeError("overlap_purity must have dtype float64.")
    if not np.issubdtype(targets.overlap_clean_zone_count.dtype, np.integer):
        raise TypeError("overlap_clean_zone_count must have an integer dtype.")


def _validate_tree_image(tree_image: TreeImage) -> None:
    if tree_image.image.ndim != 2 or tree_image.image.dtype != np.uint8:
        raise TypeError("Tree images must be two-dimensional uint8 arrays.")

    graph = tree_image.graph
    node_count = graph.levels.size
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
