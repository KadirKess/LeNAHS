import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from lenahs.component_tree.types import ComponentTreeKind
from lenahs.dataset.records import (
    CORRECTION_TARGET_POLICY,
    SCHEMA_VERSION,
    DatasetManifest,
)
from lenahs.dataset.validation import validate_nonnegative_integer


def save_dataset_manifest(path: str | Path, manifest: DatasetManifest) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Dataset manifest already exists: {destination}")
    _validate_manifest(manifest)

    content = {
        "schema_version": SCHEMA_VERSION,
        "correction_target_policy": manifest.correction_target_policy,
        "base_seed": manifest.base_seed,
        "component_tree_kind": manifest.component_tree_kind.value,
        "connectivity": manifest.connectivity,
        "image_source": manifest.image_source,
        "image_config": manifest.image_config,
        "split_sample_counts": manifest.split_sample_counts,
    }
    encoded = json.dumps(content, indent=2, sort_keys=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomically(destination, encoded)


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    source = Path(path)
    content = json.loads(source.read_text())
    version = content.get("schema_version")
    if version == 1:
        raise ValueError(
            "Schema 1 datasets require regeneration into a new schema-2 root "
            "or a separately reviewed offline conversion."
        )
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version {version!r} in {source}; "
            f"expected {SCHEMA_VERSION}."
        )

    manifest = DatasetManifest(
        correction_target_policy=content["correction_target_policy"],
        base_seed=content["base_seed"],
        component_tree_kind=ComponentTreeKind(content["component_tree_kind"]),
        connectivity=content["connectivity"],
        image_source=content["image_source"],
        image_config=content["image_config"],
        split_sample_counts=content["split_sample_counts"],
    )
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: DatasetManifest) -> None:
    if manifest.correction_target_policy != CORRECTION_TARGET_POLICY:
        raise ValueError(
            f"Unknown correction target policy: {manifest.correction_target_policy!r}."
        )
    validate_nonnegative_integer("base_seed", manifest.base_seed)
    if not isinstance(manifest.component_tree_kind, ComponentTreeKind):
        raise TypeError("component_tree_kind must be a ComponentTreeKind.")
    if manifest.connectivity != 4:
        raise ValueError("The initial dataset format requires four-connectivity.")
    if not manifest.image_source:
        raise ValueError("image_source must not be empty.")
    for split, count in manifest.split_sample_counts.items():
        if (
            not isinstance(split, str)
            or not split
            or split in {".", ".."}
            or "/" in split
            or "\\" in split
        ):
            raise ValueError("Split names must be nonempty directory names.")
        validate_nonnegative_integer(f"split_sample_counts[{split!r}]", count)

    try:
        encoded_config = json.dumps(manifest.image_config)
    except TypeError as error:
        raise TypeError("image_config must contain JSON-compatible values.") from error
    if json.loads(encoded_config) != manifest.image_config:
        raise ValueError("image_config values must survive an exact JSON round trip.")


def _write_text_atomically(destination: Path, content: str) -> None:
    with NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as file:
        temporary_path = Path(file.name)

    try:
        temporary_path.write_text(content)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
