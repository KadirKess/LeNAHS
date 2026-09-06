"""Readable conversion from Higra components to NAHS parents."""

import higra as hg
import numpy as np

from lenahs.component_tree.canonical import choose_representative
from lenahs.component_tree.higra.rag import HigraRagContext
from lenahs.component_tree.types import ComponentTree, ComponentTreeKind, FlatZoneGraph


def higra_to_nahs(
    graph: FlatZoneGraph,
    higra_tree: hg.Tree,
    context: HigraRagContext,
    kind: ComponentTreeKind,
) -> ComponentTree:
    """Convert a canonical Higra component tree to NAHS parent pointers."""

    kind = ComponentTreeKind(kind)
    leaf_count = graph.levels.size
    if context.higra_to_nahs.shape != (leaf_count,):
        raise ValueError("The Higra leaf map must contain one entry per RAG node.")

    higra_parents = np.asarray(higra_tree.parents(), dtype=np.int64)
    if higra_parents.size <= leaf_count:
        raise ValueError("The Higra tree must contain an internal component node.")

    # Direct leaf children are the flat zones introduced by each component
    introduced_leaves: dict[int, list[int]] = {}
    for higra_leaf in range(leaf_count):
        component = int(higra_parents[higra_leaf])
        nahs_node = int(context.higra_to_nahs[higra_leaf])
        introduced_leaves.setdefault(component, []).append(nahs_node)

    # Choose every component representative before connecting nested components
    representatives: dict[int, int] = {}
    for component in range(leaf_count, higra_parents.size):
        candidates = np.asarray(introduced_leaves.get(component, []), dtype=np.int64)
        if candidates.size == 0:
            raise ValueError(
                "Every canonical Higra component must introduce a RAG leaf."
            )
        representatives[component] = choose_representative(
            candidates,
            graph.pixel_counts,
            graph.anchors,
        )

    # Join leaves introduced by one component through their representative
    # The representative then links that group to its enclosing component
    parents = np.empty(leaf_count, dtype=np.int64)
    for component, candidate_list in introduced_leaves.items():
        candidates = np.asarray(candidate_list, dtype=np.int64)
        representative = representatives[component]
        parents[candidates] = representative

        enclosing_component = int(higra_parents[component])
        if enclosing_component == component:
            parents[representative] = representative
        else:
            parents[representative] = representatives[enclosing_component]

    return ComponentTree(kind=kind, parents=parents)
