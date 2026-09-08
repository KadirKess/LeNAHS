from dataclasses import dataclass
from typing import Any

from lenahs.component_tree.types import ComponentTreeKind
from lenahs.generation.noise import NoiseConfig
from lenahs.generation.samples import GenerationSample

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetManifest:
    base_seed: int
    component_tree_kind: ComponentTreeKind
    image_source: str
    image_config: dict[str, Any]
    split_sample_counts: dict[str, int]
    connectivity: int = 4


@dataclass(frozen=True)
class NoisyObservation:
    realization_index: int
    noise_config: NoiseConfig
    sample: GenerationSample


@dataclass(frozen=True)
class SampleFamily:
    sample_index: int
    clean: GenerationSample
    noisy: tuple[NoisyObservation, ...]
