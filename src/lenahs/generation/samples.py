from collections.abc import Callable
from dataclasses import dataclass

import higra as hg
import numpy as np
from numpy.typing import NDArray

from lenahs.component_tree.higra import build_higra_rag, higra_to_nahs_vectorized
from lenahs.component_tree.types import ComponentTree, ComponentTreeKind, FlatZoneGraph
from lenahs.generation.noise import NoiseConfig, corrupt_image
from lenahs.generation.targets import (
    CorrectionDiagnostics,
    CorrectionTargets,
    derive_correction_targets,
)


@dataclass
class TreeImage:
    image: NDArray[np.uint8]
    graph: FlatZoneGraph
    component_tree: ComponentTree


@dataclass
class GenerationSample:
    clean: TreeImage
    observed: TreeImage
    correction_targets: CorrectionTargets
    correction_diagnostics: CorrectionDiagnostics


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
    """Build the clean reference, observed input and correction supervision."""

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
        observed = clean
    else:
        observed_graph, observed_tree = tree_builder(corrupted_image, kind)
        observed = TreeImage(
            image=corrupted_image,
            graph=observed_graph,
            component_tree=observed_tree,
        )

    targets, diagnostics = derive_correction_targets(
        clean.graph, clean.component_tree, observed.graph
    )
    return GenerationSample(
        clean=clean,
        observed=observed,
        correction_targets=targets,
        correction_diagnostics=diagnostics,
    )
