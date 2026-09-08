from pathlib import Path

from lenahs.dataset.manifest import load_dataset_manifest, save_dataset_manifest
from lenahs.dataset.npz import load_sample_family, save_sample_family
from lenahs.dataset.records import DatasetManifest, SampleFamily
from lenahs.dataset.validation import validate_nonnegative_integer


class NpzDataset:
    def __init__(self, path: str | Path, manifest: DatasetManifest) -> None:
        self.path = Path(path)
        self.manifest = manifest

    @classmethod
    def create(cls, path: str | Path, manifest: DatasetManifest) -> "NpzDataset":
        dataset_path = Path(path)
        save_dataset_manifest(dataset_path / "manifest.json", manifest)
        for split in manifest.split_sample_counts:
            (dataset_path / split).mkdir(exist_ok=True)
        return cls(dataset_path, manifest)

    @classmethod
    def open(cls, path: str | Path) -> "NpzDataset":
        dataset_path = Path(path)
        manifest = load_dataset_manifest(dataset_path / "manifest.json")
        return cls(dataset_path, manifest)

    def save(self, split: str, family: SampleFamily) -> Path:
        self._validate_split(split)
        self._validate_tree_kind(family)
        path = self.family_path(split, family.sample_index)
        save_sample_family(path, family)
        return path

    def load(self, split: str, sample_index: int) -> SampleFamily:
        self._validate_split(split)
        family = load_sample_family(self.family_path(split, sample_index))
        self._validate_tree_kind(family)
        return family

    def family_path(self, split: str, sample_index: int) -> Path:
        self._validate_split(split)
        validate_nonnegative_integer("sample_index", sample_index)
        return self.path / split / f"sample_{sample_index:08d}.npz"

    def _validate_split(self, split: str) -> None:
        if split not in self.manifest.split_sample_counts:
            raise ValueError(f"Unknown dataset split: {split!r}.")

    def _validate_tree_kind(self, family: SampleFamily) -> None:
        kind = family.clean.clean.component_tree.kind
        if kind != self.manifest.component_tree_kind:
            raise ValueError(
                f"Sample family uses {kind.value}, but the dataset uses "
                f"{self.manifest.component_tree_kind.value}."
            )
