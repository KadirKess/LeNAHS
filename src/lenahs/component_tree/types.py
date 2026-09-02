from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class ComponentTreeKind(StrEnum):
    MAX_TREE = "max_tree"
    MIN_TREE = "min_tree"


@dataclass
class FlatZoneGraph:
    label_map: NDArray[np.integer]
    edge_index: NDArray[np.integer]
    levels: NDArray[np.uint8]
    pixel_counts: NDArray[np.integer]
    anchors: NDArray[np.integer]


@dataclass
class ComponentTree:
    kind: ComponentTreeKind
    parents: NDArray[np.integer]
