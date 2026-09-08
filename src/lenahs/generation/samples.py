from collections.abc import Callable
from dataclasses import dataclass

import higra as hg
import numpy as np
from numpy.typing import NDArray

from lenahs.component_tree.higra import build_higra_rag, higra_to_nahs_vectorized
from lenahs.component_tree.types import ComponentTree, ComponentTreeKind, FlatZoneGraph
from lenahs.generation.noise import NoiseConfig, corrupt_image
from lenahs.generation.targets import CorruptionTargets, derive_corruption_targets


@dataclass
class TreeImage:
    image: NDArray[np.uint8]
    graph: FlatZoneGraph
    component_tree: ComponentTree


@dataclass
class GenerationSample:
    clean: TreeImage
    corrupted: TreeImage | None
    corruption_targets: CorruptionTargets


type TreeBuilder = Callable[
    [NDArray[np.uint8], ComponentTreeKind],
    tuple[FlatZoneGraph, ComponentTree],
]


def _build_production_tree(
    image: NDArray[np.uint8],
    kind: ComponentTreeKind,
) -> tuple[FlatZoneGraph, ComponentTree]:
    graph, context = build_higra_rag(image)
    higra_levels = graph.levels[context.higra_to_nahs]

    if kind == ComponentTreeKind.MAX_TREE:
        higra_tree = hg.component_tree_max_tree(context.rag, higra_levels)[0]
    else:
        higra_tree = hg.component_tree_min_tree(context.rag, higra_levels)[0]

    component_tree = higra_to_nahs_vectorized(graph, higra_tree, context, kind)
    return graph, component_tree


def build_generation_sample(
    clean_image: NDArray[np.uint8],
    kind: ComponentTreeKind,
    noise_config: NoiseConfig | None = None,
    noise_rng: np.random.Generator | None = None,
    *,
    tree_builder: TreeBuilder = _build_production_tree,
) -> GenerationSample:
    """Build clean and optional corrupted component-tree representations."""

    kind = ComponentTreeKind(kind)
    if noise_config is None and noise_rng is not None:
        raise ValueError("noise_rng requires a noise_config.")
    if noise_config is not None and noise_rng is None:
        raise ValueError("noise_config requires a noise_rng.")

    corrupted_image = None
    if noise_config is not None:
        corrupted_image = corrupt_image(clean_image, noise_config, noise_rng)

    clean_graph, clean_component_tree = tree_builder(clean_image, kind)
    clean = TreeImage(
        image=clean_image,
        graph=clean_graph,
        component_tree=clean_component_tree,
    )

    if corrupted_image is None:
        corruption_targets = derive_corruption_targets(
            clean_image,
            clean_image,
            clean_graph.label_map,
            clean_graph.label_map,
        )
        return GenerationSample(
            clean=clean,
            corrupted=None,
            corruption_targets=corruption_targets,
        )

    corrupted_graph, corrupted_component_tree = tree_builder(corrupted_image, kind)
    corrupted = TreeImage(
        image=corrupted_image,
        graph=corrupted_graph,
        component_tree=corrupted_component_tree,
    )
    corruption_targets = derive_corruption_targets(
        clean_image,
        corrupted_image,
        clean_graph.label_map,
        corrupted_graph.label_map,
    )
    return GenerationSample(
        clean=clean,
        corrupted=corrupted,
        corruption_targets=corruption_targets,
    )
