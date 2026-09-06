"""Higra flat-zone RAG construction and identifier remapping."""

from dataclasses import dataclass

import higra as hg
import numpy as np
from numpy.typing import NDArray

from lenahs.component_tree.types import FlatZoneGraph


@dataclass
class HigraRagContext:
    rag: hg.UndirectedGraph
    higra_to_nahs: NDArray[np.int64]


def _check_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")
    if image.ndim != 2:
        raise ValueError("Input image must be 2D.")
    if image.size == 0:
        raise ValueError("Input image must not be empty.")
    if image.dtype != np.uint8:
        raise ValueError("Input image must be of type np.uint8.")


def _remap_higra_regions(
    image_shape: tuple[int, int],
    pixel_levels: NDArray[np.uint8],
    pixel_to_higra: NDArray[np.integer],
    higra_edges: tuple[NDArray[np.integer], NDArray[np.integer]],
) -> tuple[FlatZoneGraph, NDArray[np.int64]]:
    region_count = int(np.max(pixel_to_higra)) + 1
    pixel_indices = np.arange(pixel_to_higra.size, dtype=np.int64)

    # Scatter pixel indices into their regions and retain the smallest as each anchor
    higra_anchors = np.full(region_count, pixel_to_higra.size, dtype=np.int64)
    np.minimum.at(higra_anchors, pixel_to_higra, pixel_indices)

    # Count each region's pixels and read its constant level at the anchor
    higra_pixel_counts = np.bincount(
        pixel_to_higra,
        minlength=region_count,
    ).astype(np.int64, copy=False)
    higra_levels = pixel_levels[higra_anchors]

    # Sort regions by anchor for NAHS IDs, then invert that permutation
    nahs_to_higra = np.argsort(higra_anchors)
    higra_to_nahs = np.empty(region_count, dtype=np.int64)
    higra_to_nahs[nahs_to_higra] = np.arange(region_count, dtype=np.int64)

    # Translate every pixel label from Higra IDs to deterministic NAHS IDs
    label_map = higra_to_nahs[pixel_to_higra].reshape(image_shape)

    # Translate and normalize the endpoints of each undirected RAG edge
    sources, targets = higra_edges
    sources = higra_to_nahs[np.asarray(sources, dtype=np.int64)]
    targets = higra_to_nahs[np.asarray(targets, dtype=np.int64)]
    edge_index = np.column_stack(
        (np.minimum(sources, targets), np.maximum(sources, targets))
    )

    # Sort edges by source and then target, preserving shape (0, 2) when empty
    if edge_index.size:
        edge_order = np.lexsort((edge_index[:, 1], edge_index[:, 0]))
        edge_index = edge_index[edge_order]
    else:
        edge_index = np.empty((0, 2), dtype=np.int64)

    # Reorder every region attribute from Higra indexing into NAHS indexing
    graph = FlatZoneGraph(
        label_map=label_map.astype(np.int64, copy=False),
        edge_index=edge_index.astype(np.int64, copy=False),
        levels=higra_levels[nahs_to_higra].astype(np.uint8, copy=False),
        pixel_counts=higra_pixel_counts[nahs_to_higra],
        anchors=higra_anchors[nahs_to_higra],
    )
    return graph, higra_to_nahs


def build_higra_rag(image: np.ndarray) -> tuple[FlatZoneGraph, HigraRagContext]:
    """Build a deterministic flat-zone RAG through Higra."""

    _check_image(image)

    # Build the four-connected pixel graph and align image values with its vertices
    pixel_graph = hg.get_4_adjacency_graph(image.shape)
    pixel_levels = hg.linearize_vertex_weights(image, pixel_graph)

    # Collapse connected equal-level pixels and recover their Higra region labels
    rag = hg.make_region_adjacency_graph_from_labelisation(
        pixel_graph,
        pixel_levels,
    )
    pixel_to_higra = hg.CptRegionAdjacencyGraph.get_vertex_map(rag)

    graph, higra_to_nahs = _remap_higra_regions(
        image.shape,
        pixel_levels,
        pixel_to_higra,
        rag.edge_list(),
    )

    # Return NAHS arrays while retaining the Higra RAG and its ID translation
    return graph, HigraRagContext(rag=rag, higra_to_nahs=higra_to_nahs)
