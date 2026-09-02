import numpy as np

from lenahs.component_tree.types import FlatZoneGraph


def _check_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")
    if image.ndim != 2:
        raise ValueError("Input image must be 2D.")
    if image.size == 0:
        raise ValueError("Input image must not be empty.")
    if image.dtype != np.uint8:
        raise ValueError("Input image must be of type np.uint8.")


def _build_flat_zones(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_map = np.full(image.shape, -1, dtype=np.int64)
    levels = []
    pixel_counts = []
    anchors = []

    height, width = image.shape
    for row in range(height):
        for column in range(width):
            if label_map[row, column] != -1:
                continue

            # Start a new flat zone
            zone = len(levels)
            level = image[row, column]
            stack = [(row, column)]
            label_map[row, column] = zone
            pixel_count = 0

            while stack:
                current_row, current_column = stack.pop()
                pixel_count += 1

                # Visit 4-connected neighbors
                for row_offset, column_offset in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    neighbor_row = current_row + row_offset
                    neighbor_column = current_column + column_offset
                    inside_image = (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_column < width
                    )
                    if not inside_image:
                        continue
                    if label_map[neighbor_row, neighbor_column] != -1:
                        continue
                    if image[neighbor_row, neighbor_column] != level:
                        continue

                    label_map[neighbor_row, neighbor_column] = zone
                    stack.append((neighbor_row, neighbor_column))

            levels.append(level)
            pixel_counts.append(pixel_count)
            anchors.append(row * width + column)

    return (
        label_map,
        np.asarray(levels, dtype=np.uint8),
        np.asarray(pixel_counts, dtype=np.int64),
        np.asarray(anchors, dtype=np.int64),
    )


def _build_rag_edges(label_map: np.ndarray) -> np.ndarray:
    edges = set()
    height, width = label_map.shape

    for row in range(height):
        for column in range(width):
            zone = int(label_map[row, column])

            # Check right and down so each boundary is visited once
            if column + 1 < width:
                right_zone = int(label_map[row, column + 1])
                if zone != right_zone:
                    edges.add(tuple(sorted((zone, right_zone))))

            if row + 1 < height:
                lower_zone = int(label_map[row + 1, column])
                if zone != lower_zone:
                    edges.add(tuple(sorted((zone, lower_zone))))

    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)


def build_reference_rag(image: np.ndarray) -> FlatZoneGraph:
    _check_image(image)
    label_map, levels, pixel_counts, anchors = _build_flat_zones(image)
    edge_index = _build_rag_edges(label_map)

    return FlatZoneGraph(
        label_map=label_map,
        edge_index=edge_index,
        levels=levels,
        pixel_counts=pixel_counts,
        anchors=anchors,
    )
