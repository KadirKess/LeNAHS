from dataclasses import dataclass
from typing import Any

from lenahs.component_tree.types import ComponentTreeKind
from lenahs.generation.noise import NoiseConfig
from lenahs.generation.samples import GenerationSample

SCHEMA_VERSION = 2
CORRECTION_TARGET_POLICY = "pixel_count_representative_v1"


@dataclass(frozen=True)
class DatasetManifest:
    base_seed: int
    component_tree_kind: ComponentTreeKind
    image_source: str
    image_config: dict[str, Any]
    split_sample_counts: dict[str, int]
    connectivity: int = 4
    correction_target_policy: str = CORRECTION_TARGET_POLICY


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
