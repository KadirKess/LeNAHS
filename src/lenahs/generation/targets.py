from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from lenahs.component_tree.types import ComponentTree, FlatZoneGraph


class CorrectionAction(IntEnum):
    MERGE = 0
    KEEP_ZERO = 1
    KEEP_ONE = 2


@dataclass
class CorrectionTargets:
    corrective_parents: NDArray[np.int64]
    actions: NDArray[np.int64]


@dataclass
class CorrectionDiagnostics:
    """Clean-reference diagnostics for analysis, never model or decoder inputs."""

    # Per observed zone, the number of distinct clean zones beneath its pixels.
    overlap_clean_zone_count: NDArray[np.int64]
    # Per observed zone, the fraction of pixels belonging to its assigned clean role.
    overlap_purity: NDArray[np.float64]
    # Per clean zone, whether no observed zone selected it as a clean role.
    missing_clean_zones: NDArray[np.bool_]
    # Whether the original clean root has no observed representative.
    clean_root_missing: bool
    # Former forest roots attached to the selected root, excluding its self-link.
    artificial_root_attachment_count: int


def derive_correction_targets(
    clean_graph: FlatZoneGraph,
    clean_tree: ComponentTree,
    observed_graph: FlatZoneGraph,
) -> tuple[CorrectionTargets, CorrectionDiagnostics]:
    """Derive one corrective pointer and action per observed zone."""
    overlap_counts = _count_zone_overlaps(
        clean_graph.label_map,
        observed_graph.label_map,
        observed_graph.levels.size,
    )
    clean_role = _assign_clean_roles(overlap_counts, clean_graph.anchors)
    representatives = _select_observed_representatives(
        clean_role, observed_graph.anchors, clean_graph.levels.size
    )
    corrective_parents, actions, forest_roots = _build_correction_forest(
        clean_graph, clean_tree, clean_role, representatives
    )
    selected_root = _select_correction_root(
        forest_roots, clean_role, clean_graph, clean_tree, observed_graph.anchors
    )

    # Former forest roots stay separate and add no depth beneath the selected root.
    for forest_root in forest_roots:
        corrective_parents[forest_root] = selected_root
        actions[forest_root] = CorrectionAction.KEEP_ZERO

    diagnostics = _derive_correction_diagnostics(
        overlap_counts,
        clean_role,
        representatives,
        clean_tree.parents,
        observed_graph.pixel_counts,
        len(forest_roots),
    )
    return CorrectionTargets(corrective_parents, actions), diagnostics


def _count_zone_overlaps(
    clean_label_map: NDArray[np.integer],
    observed_label_map: NDArray[np.integer],
    observed_zone_count: int,
) -> list[dict[int, int]]:
    overlap_counts = [{} for _ in range(observed_zone_count)]

    # Read corresponding pixels once and store only zone pairs that occur.
    for clean_label, observed_label in zip(
        clean_label_map.ravel(), observed_label_map.ravel()
    ):
        clean_zone = int(clean_label)
        observed_zone = int(observed_label)
        counts = overlap_counts[observed_zone]
        counts[clean_zone] = counts.get(clean_zone, 0) + 1

    return overlap_counts


def _assign_clean_roles(
    overlap_counts: list[dict[int, int]],
    clean_anchors: NDArray[np.integer],
) -> NDArray[np.int64]:
    clean_role = np.full(len(overlap_counts), -1, dtype=np.int64)

    for observed_zone, overlaps in enumerate(overlap_counts):
        max_clean_zone = max(overlaps, key=overlaps.get)
        max_count = overlaps[max_clean_zone]
        tied_clean_zones = [
            clean_zone for clean_zone, count in overlaps.items() if count == max_count
        ]

        # Clean anchors decide equal overlaps independently of zone numbering.
        clean_role[observed_zone] = min(
            tied_clean_zones, key=lambda zone: clean_anchors[zone]
        )

    return clean_role


def _select_observed_representatives(
    clean_role: NDArray[np.int64],
    observed_anchors: NDArray[np.integer],
    clean_zone_count: int,
) -> NDArray[np.int64]:
    representatives = np.full(clean_zone_count, -1, dtype=np.int64)

    # Visit each observed zone once instead of searching again for every clean role.
    for observed_zone, clean_zone in enumerate(clean_role):
        current = representatives[clean_zone]
        if current == -1 or observed_anchors[observed_zone] < observed_anchors[current]:
            representatives[clean_zone] = observed_zone

    return representatives


def _find_represented_ancestors(
    clean_parents: NDArray[np.integer],
    representatives: NDArray[np.int64],
) -> NDArray[np.int64]:
    children = [[] for _ in range(clean_parents.size)]
    for clean_zone, clean_parent in enumerate(clean_parents):
        if clean_parent == clean_zone:
            root = clean_zone
        else:
            children[clean_parent].append(clean_zone)

    nearest_ancestor = np.full(clean_parents.size, -1, dtype=np.int64)
    pending = [root]

    # Parents are visited first so children can reuse the nearest represented role.
    # Missing roles pass that ancestor onward. The root has no proper ancestor.
    while pending:
        clean_zone = pending.pop()
        if representatives[clean_zone] != -1:
            ancestor_for_children = clean_zone
        else:
            ancestor_for_children = nearest_ancestor[clean_zone]

        for child in children[clean_zone]:
            nearest_ancestor[child] = ancestor_for_children
            pending.append(child)

    return nearest_ancestor


def _build_correction_forest(
    clean_graph: FlatZoneGraph,
    clean_tree: ComponentTree,
    clean_role: NDArray[np.int64],
    representatives: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], list[int]]:
    corrective_parents = np.full(clean_role.size, -1, dtype=np.int64)
    actions = np.full(clean_role.size, -1, dtype=np.int64)

    # Only representatives carry hierarchy links. Other assigned zones merge into them.
    for observed_zone, clean_zone in enumerate(clean_role):
        destination = representatives[clean_zone]
        if destination != observed_zone:
            corrective_parents[observed_zone] = destination
            actions[observed_zone] = CorrectionAction.MERGE

    nearest_ancestor = _find_represented_ancestors(clean_tree.parents, representatives)
    forest_roots = []

    for clean_zone, observed_zone in enumerate(representatives):
        if observed_zone == -1:
            continue

        clean_ancestor = nearest_ancestor[clean_zone]
        if clean_ancestor == -1:
            forest_roots.append(int(observed_zone))
            continue

        corrective_parents[observed_zone] = representatives[clean_ancestor]

        # Skipped roles add no extra increments. Compare the actual link endpoints.
        if clean_graph.levels[clean_zone] == clean_graph.levels[clean_ancestor]:
            actions[observed_zone] = CorrectionAction.KEEP_ZERO
        else:
            actions[observed_zone] = CorrectionAction.KEEP_ONE

    return corrective_parents, actions, forest_roots


def _select_correction_root(
    forest_roots: list[int],
    clean_role: NDArray[np.int64],
    clean_graph: FlatZoneGraph,
    clean_tree: ComponentTree,
    observed_anchors: NDArray[np.integer],
) -> int:
    selected_root = forest_roots[0]

    # Max-trees choose the lowest clean level. Min-trees choose the highest.
    for candidate in forest_roots:
        candidate_level = clean_graph.levels[clean_role[candidate]]
        selected_level = clean_graph.levels[clean_role[selected_root]]

        if clean_tree.kind == "max_tree":
            better_level = candidate_level < selected_level
        else:
            better_level = candidate_level > selected_level

        if better_level:
            selected_root = candidate
        elif candidate_level == selected_level:
            if observed_anchors[candidate] < observed_anchors[selected_root]:
                selected_root = candidate

    return selected_root


def _derive_correction_diagnostics(
    overlap_counts: list[dict[int, int]],
    clean_role: NDArray[np.int64],
    representatives: NDArray[np.int64],
    clean_parents: NDArray[np.integer],
    observed_pixel_counts: NDArray[np.integer],
    forest_root_count: int,
) -> CorrectionDiagnostics:
    overlap_clean_zone_count = np.empty(clean_role.size, dtype=np.int64)
    overlap_purity = np.empty(clean_role.size, dtype=np.float64)

    # Reuse the counted overlaps instead of reading the paired images again.
    for observed_zone, overlaps in enumerate(overlap_counts):
        overlap_clean_zone_count[observed_zone] = len(overlaps)
        assigned_pixel_count = overlaps[clean_role[observed_zone]]
        overlap_purity[observed_zone] = (
            assigned_pixel_count / observed_pixel_counts[observed_zone]
        )

    missing_clean_zones = representatives == -1
    for clean_zone, clean_parent in enumerate(clean_parents):
        if clean_parent == clean_zone:
            clean_root_missing = bool(missing_clean_zones[clean_zone])
            break

    return CorrectionDiagnostics(
        overlap_clean_zone_count=overlap_clean_zone_count,
        overlap_purity=overlap_purity,
        missing_clean_zones=missing_clean_zones,
        clean_root_missing=clean_root_missing,
        artificial_root_attachment_count=forest_root_count - 1,
    )


class NoiseDirection(IntEnum):
    """Class indices for comparison with the mean clean intensity."""

    UNCHANGED = 0
    BRIGHTENED = 1
    DARKENED = 2


@dataclass
class CorruptionTargets:
    noise_direction: NDArray[np.int64]
    overlap_purity: NDArray[np.float64]
    overlap_clean_zone_count: NDArray[np.integer]


def derive_corruption_targets(
    clean_image: NDArray[np.uint8],
    clean_graph: FlatZoneGraph,
    corrupted_graph: FlatZoneGraph,
) -> CorruptionTargets:
    """Compare observed levels with mean clean levels, including mixed zones."""

    clean_labels = clean_graph.label_map.ravel()
    corrupted_labels = corrupted_graph.label_map.ravel()
    corrupted_zone_count = corrupted_graph.levels.size

    zone_pixel_counts = np.bincount(
        corrupted_labels,
        minlength=corrupted_zone_count,
    ).astype(np.int64, copy=False)
    clean_sums = np.zeros(corrupted_zone_count, dtype=np.int64)
    np.add.at(
        clean_sums,
        corrupted_labels,
        clean_image.ravel().astype(np.int64, copy=False),
    )
    observed_sums = (
        corrupted_graph.levels.astype(np.int64, copy=False) * zone_pixel_counts
    )

    noise_direction = np.full(
        corrupted_zone_count,
        NoiseDirection.UNCHANGED,
        dtype=np.int64,
    )
    noise_direction[observed_sums > clean_sums] = NoiseDirection.BRIGHTENED
    noise_direction[observed_sums < clean_sums] = NoiseDirection.DARKENED

    zone_pairs, pair_counts = np.unique(
        np.column_stack((corrupted_labels, clean_labels)),
        axis=0,
        return_counts=True,
    )
    pair_corrupted_zones = zone_pairs[:, 0]
    overlap_clean_zone_count = np.bincount(
        pair_corrupted_zones,
        minlength=corrupted_zone_count,
    ).astype(np.int64, copy=False)
    largest_overlap = np.zeros(corrupted_zone_count, dtype=np.int64)
    np.maximum.at(largest_overlap, pair_corrupted_zones, pair_counts)
    overlap_purity = largest_overlap / zone_pixel_counts

    return CorruptionTargets(
        noise_direction=noise_direction,
        overlap_purity=overlap_purity,
        overlap_clean_zone_count=overlap_clean_zone_count,
    )
