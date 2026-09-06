import builtins
import importlib
import sys

import numpy as np

from lenahs.component_tree import ComponentTree, ComponentTreeKind, FlatZoneGraph


def test_flat_zone_graph_exposes_supplied_fields_without_conversion():
    label_map = np.array([[0, 1]], dtype=np.int32)
    edge_index = np.array([[0, 1]], dtype=np.int64)
    levels = np.array([10, 20], dtype=np.uint8)
    pixel_counts = np.array([1, 1], dtype=np.int64)
    anchors = np.array([0, 1], dtype=np.int64)

    graph = FlatZoneGraph(
        label_map=label_map,
        edge_index=edge_index,
        levels=levels,
        pixel_counts=pixel_counts,
        anchors=anchors,
    )

    assert graph.label_map is label_map
    assert graph.edge_index is edge_index
    assert graph.levels is levels
    assert graph.pixel_counts is pixel_counts
    assert graph.anchors is anchors


def test_component_tree_contains_one_kind_and_one_parent_array():
    parents = np.array([0, 0], dtype=np.int64)

    tree = ComponentTree(kind=ComponentTreeKind.MAX_TREE, parents=parents)

    assert tree.kind is ComponentTreeKind.MAX_TREE
    assert tree.parents is parents
    assert set(vars(tree)) == {"kind", "parents"}


def test_component_tree_constructor_does_not_convert_legacy_root():
    parents = np.array([-1, 0], dtype=np.int64)

    tree = ComponentTree(kind=ComponentTreeKind.MAX_TREE, parents=parents)

    assert tree.parents is parents
    assert tree.parents.tolist() == [-1, 0]


def test_public_imports_do_not_require_deferred_frameworks(monkeypatch):
    forbidden = {"higra", "torch", "torch_geometric", "zarr", "lightning"}
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            raise AssertionError(
                f"public component-tree import attempted to load {name}"
            )
        return original_import(name, *args, **kwargs)

    for module_name in list(sys.modules):
        if module_name == "lenahs" or module_name.startswith("lenahs."):
            monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("lenahs")

    assert module.FlatZoneGraph.__module__ == "lenahs.component_tree.types"
    assert module.ComponentTree.__module__ == "lenahs.component_tree.types"
