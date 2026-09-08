import json

import pytest

from lenahs.component_tree import ComponentTreeKind
from lenahs.dataset import (
    DatasetManifest,
    load_dataset_manifest,
    save_dataset_manifest,
)


def test_manifest_round_trip_preserves_dataset_definition(tmp_path) -> None:
    manifest = DatasetManifest(
        base_seed=812,
        component_tree_kind=ComponentTreeKind.MIN_TREE,
        image_source="random_squares",
        image_config={"height": 256, "width": 256, "levels": [0, 255]},
        split_sample_counts={"train": 8_000, "validation": 1_000, "test": 1_000},
    )
    path = tmp_path / "dataset" / "manifest.json"

    save_dataset_manifest(path, manifest)

    assert load_dataset_manifest(path) == manifest


def test_manifest_refuses_to_replace_an_existing_definition(tmp_path) -> None:
    manifest = DatasetManifest(
        base_seed=1,
        component_tree_kind=ComponentTreeKind.MAX_TREE,
        image_source="external",
        image_config={},
        split_sample_counts={"train": 1},
    )
    path = tmp_path / "manifest.json"
    save_dataset_manifest(path, manifest)

    with pytest.raises(FileExistsError, match="already exists"):
        save_dataset_manifest(path, manifest)


def test_manifest_rejects_an_unknown_schema_version(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 7}))

    with pytest.raises(ValueError, match="Unsupported schema version 7"):
        load_dataset_manifest(path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("base_seed", -1, ValueError),
        ("base_seed", True, TypeError),
        ("connectivity", 8, ValueError),
        ("image_source", "", ValueError),
        ("split_sample_counts", {"train": -1}, ValueError),
        ("image_config", {"shape": {256, 256}}, TypeError),
    ],
)
def test_manifest_rejects_invalid_dataset_metadata(
    tmp_path, field, value, error
) -> None:
    arguments = {
        "base_seed": 1,
        "component_tree_kind": ComponentTreeKind.MAX_TREE,
        "image_source": "random_squares",
        "image_config": {"height": 256, "width": 256},
        "split_sample_counts": {"train": 1},
        "connectivity": 4,
    }
    arguments[field] = value
    manifest = DatasetManifest(**arguments)

    with pytest.raises(error):
        save_dataset_manifest(tmp_path / "manifest.json", manifest)
