import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from lenahs.component_tree.types import ComponentTree, ComponentTreeKind, FlatZoneGraph
from lenahs.dataset.records import (
    SCHEMA_VERSION,
    NoisyObservation,
    SampleFamily,
)
from lenahs.dataset.validation import validate_sample_family
from lenahs.generation.noise import (
    AdditiveGaussianNoise,
    NoiseConfig,
    PoissonNoise,
    SaltAndPepperNoise,
    SpeckleNoise,
)
from lenahs.generation.samples import GenerationSample, TreeImage
from lenahs.generation.targets import CorrectionDiagnostics, CorrectionTargets

_NOISE_CONFIG_TYPES = {
    "salt_and_pepper": SaltAndPepperNoise,
    "additive_gaussian": AdditiveGaussianNoise,
    "speckle": SpeckleNoise,
    "poisson": PoissonNoise,
}
_NOISE_CONFIG_NAMES = {
    config_type: name for name, config_type in _NOISE_CONFIG_TYPES.items()
}


def save_sample_family(path: str | Path, family: SampleFamily) -> None:
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("Sample family paths must end in .npz.")
    if destination.exists():
        raise FileExistsError(f"Sample family already exists: {destination}")
    validate_sample_family(family)
    for observation in family.noisy:
        _encode_noise_config(observation.noise_config)

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "sample_index": np.asarray(family.sample_index, dtype=np.int64),
        "realization_indices": np.asarray(
            [observation.realization_index for observation in family.noisy],
            dtype=np.int64,
        ),
        "noise_configs": np.asarray(
            [
                _encode_noise_config(observation.noise_config)
                for observation in family.noisy
            ],
            dtype=np.str_,
        ),
    }
    _store_tree_image(arrays, "clean", family.clean.clean)
    _store_targets(arrays, "clean.targets", family.clean.correction_targets)
    _store_diagnostics(arrays, "clean.diagnostics", family.clean.correction_diagnostics)

    for position, observation in enumerate(family.noisy):
        prefix = f"noisy.{position}"
        _store_tree_image(arrays, prefix, observation.sample.observed)
        _store_targets(
            arrays,
            f"{prefix}.targets",
            observation.sample.correction_targets,
        )

        _store_diagnostics(
            arrays,
            f"{prefix}.diagnostics",
            observation.sample.correction_diagnostics,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as file:
        temporary_path = Path(file.name)

    try:
        with temporary_path.open("wb") as file:
            np.savez_compressed(file, **arrays)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_sample_family(path: str | Path) -> SampleFamily:
    source = Path(path)
    with np.load(source, allow_pickle=False) as arrays:
        version = int(arrays["schema_version"])
        if version == 1:
            raise ValueError(
                "Schema 1 families require regeneration into a new schema-2 root "
                "or a separately reviewed offline conversion."
            )
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version {version!r} in {source}; "
                f"expected {SCHEMA_VERSION}."
            )

        sample_index = int(arrays["sample_index"])
        realization_indices = arrays["realization_indices"]
        noise_configs = arrays["noise_configs"]
        if realization_indices.shape != noise_configs.shape:
            raise ValueError("Realization indices and noise configurations must align.")

        clean_image = _load_tree_image(arrays, "clean")
        clean_targets = _load_targets(arrays, "clean.targets")
        clean_sample = GenerationSample(
            clean=clean_image,
            observed=clean_image,
            correction_targets=clean_targets,
            correction_diagnostics=_load_diagnostics(arrays, "clean.diagnostics"),
        )

        observations = []
        for position, (realization_index, encoded_config) in enumerate(
            zip(realization_indices, noise_configs, strict=True)
        ):
            prefix = f"noisy.{position}"
            observed = _load_tree_image(arrays, prefix)
            targets = _load_targets(arrays, f"{prefix}.targets")
            observations.append(
                NoisyObservation(
                    realization_index=int(realization_index),
                    noise_config=_decode_noise_config(str(encoded_config)),
                    sample=GenerationSample(
                        clean=clean_image,
                        observed=observed,
                        correction_targets=targets,
                        correction_diagnostics=_load_diagnostics(
                            arrays, f"{prefix}.diagnostics"
                        ),
                    ),
                )
            )

    family = SampleFamily(
        sample_index=sample_index,
        clean=clean_sample,
        noisy=tuple(observations),
    )
    validate_sample_family(family)
    return family


def _store_tree_image(
    arrays: dict[str, np.ndarray],
    prefix: str,
    tree_image: TreeImage,
) -> None:
    arrays[f"{prefix}.image"] = tree_image.image
    arrays[f"{prefix}.graph.label_map"] = tree_image.graph.label_map
    arrays[f"{prefix}.graph.edge_index"] = tree_image.graph.edge_index
    arrays[f"{prefix}.graph.levels"] = tree_image.graph.levels
    arrays[f"{prefix}.graph.pixel_counts"] = tree_image.graph.pixel_counts
    arrays[f"{prefix}.graph.anchors"] = tree_image.graph.anchors
    arrays[f"{prefix}.component_tree.kind"] = np.asarray(
        tree_image.component_tree.kind.value
    )
    arrays[f"{prefix}.component_tree.parents"] = tree_image.component_tree.parents


def _load_tree_image(arrays: Any, prefix: str) -> TreeImage:
    graph = FlatZoneGraph(
        label_map=arrays[f"{prefix}.graph.label_map"].copy(),
        edge_index=arrays[f"{prefix}.graph.edge_index"].copy(),
        levels=arrays[f"{prefix}.graph.levels"].copy(),
        pixel_counts=arrays[f"{prefix}.graph.pixel_counts"].copy(),
        anchors=arrays[f"{prefix}.graph.anchors"].copy(),
    )
    component_tree = ComponentTree(
        kind=ComponentTreeKind(str(arrays[f"{prefix}.component_tree.kind"])),
        parents=arrays[f"{prefix}.component_tree.parents"].copy(),
    )
    return TreeImage(
        image=arrays[f"{prefix}.image"].copy(),
        graph=graph,
        component_tree=component_tree,
    )


def _store_targets(
    arrays: dict[str, np.ndarray],
    prefix: str,
    targets: CorrectionTargets,
) -> None:
    arrays[f"{prefix}.corrective_parents"] = targets.corrective_parents
    arrays[f"{prefix}.actions"] = targets.actions


def _load_targets(arrays: Any, prefix: str) -> CorrectionTargets:
    return CorrectionTargets(
        corrective_parents=arrays[f"{prefix}.corrective_parents"].copy(),
        actions=arrays[f"{prefix}.actions"].copy(),
    )


def _store_diagnostics(
    arrays: dict[str, np.ndarray],
    prefix: str,
    diagnostics: CorrectionDiagnostics,
) -> None:
    arrays[f"{prefix}.overlap_clean_zone_count"] = diagnostics.overlap_clean_zone_count
    arrays[f"{prefix}.overlap_purity"] = diagnostics.overlap_purity
    arrays[f"{prefix}.missing_clean_zones"] = diagnostics.missing_clean_zones
    arrays[f"{prefix}.clean_root_missing"] = np.asarray(
        diagnostics.clean_root_missing, dtype=np.bool_
    )
    arrays[f"{prefix}.artificial_root_attachment_count"] = np.asarray(
        diagnostics.artificial_root_attachment_count, dtype=np.int64
    )


def _load_diagnostics(arrays: Any, prefix: str) -> CorrectionDiagnostics:
    return CorrectionDiagnostics(
        overlap_clean_zone_count=arrays[f"{prefix}.overlap_clean_zone_count"].copy(),
        overlap_purity=arrays[f"{prefix}.overlap_purity"].copy(),
        missing_clean_zones=arrays[f"{prefix}.missing_clean_zones"].copy(),
        clean_root_missing=_load_scalar(
            arrays, f"{prefix}.clean_root_missing", np.bool_
        ),
        artificial_root_attachment_count=_load_scalar(
            arrays, f"{prefix}.artificial_root_attachment_count", np.int64
        ),
    )


def _load_scalar(arrays: Any, key: str, dtype: type) -> bool | int:
    value = arrays[key]
    if value.shape != () or value.dtype != dtype:
        raise ValueError(f"{key} must be a scalar with dtype {np.dtype(dtype)}.")
    return value.item()


def _encode_noise_config(config: NoiseConfig) -> str:
    config_name = _NOISE_CONFIG_NAMES.get(type(config))
    if config_name is None:
        raise TypeError(f"Unsupported noise configuration: {type(config).__name__}.")
    return json.dumps(
        {"kind": config_name, "parameters": asdict(config)},
        sort_keys=True,
        separators=(",", ":"),
    )


def _decode_noise_config(value: str) -> NoiseConfig:
    content = json.loads(value)
    try:
        config_type = _NOISE_CONFIG_TYPES[content["kind"]]
    except KeyError as error:
        raise ValueError(
            f"Unknown stored noise kind: {content.get('kind')!r}."
        ) from error
    return config_type(**content["parameters"])
