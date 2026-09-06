"""Data structures and algorithms for flat-zone component trees."""

from lenahs.component_tree.canonical import choose_representative
from lenahs.component_tree.reference.component_tree import (
    build_reference_component_tree,
)
from lenahs.component_tree.reference.rag import build_reference_rag
from lenahs.component_tree.types import ComponentTree, ComponentTreeKind, FlatZoneGraph

__all__ = [
    "ComponentTree",
    "ComponentTreeKind",
    "FlatZoneGraph",
    "build_reference_component_tree",
    "build_reference_rag",
    "choose_representative",
]
