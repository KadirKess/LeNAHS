import numpy as np
import pytest

from lenahs.component_tree import (
    ComponentTreeKind,
    build_reference_component_tree,
    build_reference_rag,
)
from lenahs.dataset import DatasetManifest, NpzDataset, SampleFamily
from lenahs.generation import build_generation_sample


def _build_reference_tree(image, kind):
    graph = build_reference_rag(image)
    return graph, build_reference_component_tree(graph, kind)


def _clean_family(sample_index: int, kind: ComponentTreeKind) -> SampleFamily:
    image = np.array([[10, 10, 30], [10, 20, 30]], dtype=np.uint8)
    sample = build_generation_sample(
        image,
        kind,
        tree_builder=_build_reference_tree,
    )
    return SampleFamily(sample_index=sample_index, clean=sample, noisy=())


def _manifest(kind: ComponentTreeKind) -> DatasetManifest:
    return DatasetManifest(
        base_seed=17,
        component_tree_kind=kind,
        image_source="random_squares",
        image_config={"height": 256, "width": 256},
        split_sample_counts={"train": 8, "validation": 1, "test": 1},
    )


def test_dataset_creates_manifest_and_split_directories(tmp_path) -> None:
    dataset = NpzDataset.create(
        tmp_path / "dataset", _manifest(ComponentTreeKind.MAX_TREE)
    )

    assert (dataset.path / "manifest.json").is_file()
    assert (dataset.path / "train").is_dir()
    assert (dataset.path / "validation").is_dir()
    assert (dataset.path / "test").is_dir()


def test_dataset_saves_and_loads_a_family_by_split_and_index(tmp_path) -> None:
    dataset = NpzDataset.create(
        tmp_path / "dataset", _manifest(ComponentTreeKind.MAX_TREE)
    )
    family = _clean_family(23, ComponentTreeKind.MAX_TREE)

    path = dataset.save("train", family)
    reopened = NpzDataset.open(dataset.path)
    loaded = reopened.load("train", 23)

    assert path == dataset.path / "train" / "sample_00000023.npz"
    assert loaded.sample_index == 23
    np.testing.assert_array_equal(loaded.clean.clean.image, family.clean.clean.image)


def test_dataset_rejects_an_unknown_split(tmp_path) -> None:
    dataset = NpzDataset.create(
        tmp_path / "dataset", _manifest(ComponentTreeKind.MAX_TREE)
    )

    with pytest.raises(ValueError, match="Unknown dataset split"):
        dataset.save("development", _clean_family(0, ComponentTreeKind.MAX_TREE))


def test_dataset_rejects_the_other_component_tree_kind(tmp_path) -> None:
    dataset = NpzDataset.create(
        tmp_path / "dataset", _manifest(ComponentTreeKind.MAX_TREE)
    )

    with pytest.raises(ValueError, match="dataset uses max_tree"):
        dataset.save("train", _clean_family(0, ComponentTreeKind.MIN_TREE))
