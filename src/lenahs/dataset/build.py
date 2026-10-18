"""Generate clean families and their noisy observations into a dataset root."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path

from lenahs.component_tree.types import ComponentTreeKind
from lenahs.dataset.records import DatasetManifest, NoisyObservation, SampleFamily
from lenahs.dataset.store import NpzDataset
from lenahs.generation.images import SquareImageConfig, generate_square_image
from lenahs.generation.noise import (
    AdditiveGaussianNoise,
    NoiseConfig,
    PoissonNoise,
    SaltAndPepperNoise,
    SpeckleNoise,
)
from lenahs.generation.samples import build_generation_sample
from lenahs.generation.seeds import GenerationStage, derive_rng

# Each corruption family draws from its own stream so that adding one family
# never changes the realizations of another.
NOISE_STAGES = {
    SaltAndPepperNoise: GenerationStage.SALT_AND_PEPPER_NOISE,
    AdditiveGaussianNoise: GenerationStage.ADDITIVE_GAUSSIAN_NOISE,
    SpeckleNoise: GenerationStage.SPECKLE_NOISE,
    PoissonNoise: GenerationStage.POISSON_NOISE,
}


def build_sample_family(
    base_seed: int,
    sample_index: int,
    image_config: SquareImageConfig,
    kind: ComponentTreeKind,
    noise_config: NoiseConfig,
    realization_count: int,
) -> SampleFamily:
    """Build one family from derived streams, independent of generation order."""
    image_rng = derive_rng(base_seed, sample_index, GenerationStage.SQUARE_IMAGE)
    clean_image = generate_square_image(image_config, image_rng)
    clean_sample = build_generation_sample(clean_image, kind)

    noise_stage = NOISE_STAGES[type(noise_config)]
    noisy = []
    for realization_index in range(realization_count):
        noise_rng = derive_rng(
            base_seed, sample_index, noise_stage, realization_index=realization_index
        )
        sample = build_generation_sample(clean_image, kind, noise_config, noise_rng)
        noisy.append(
            NoisyObservation(
                sample=sample,
                realization_index=realization_index,
                noise_config=noise_config,
            )
        )

    return SampleFamily(sample_index, clean_sample, tuple(noisy))


def generate_dataset(
    path: str | Path,
    base_seed: int,
    split_family_counts: dict[str, int],
    image_config: SquareImageConfig,
    kind: ComponentTreeKind,
    noise_config: NoiseConfig,
    realization_count: int,
    workers: int = 1,
) -> NpzDataset:
    manifest = DatasetManifest(
        base_seed=base_seed,
        component_tree_kind=kind,
        image_source="square_images",
        image_config=asdict(image_config),
        split_sample_counts=split_family_counts,
    )
    dataset = NpzDataset.create(path, manifest)

    # Splits take consecutive index ranges so no clean image appears in two splits
    jobs = []
    next_index = 0
    for split, family_count in split_family_counts.items():
        stop = next_index + family_count
        jobs += [(split, index) for index in range(next_index, stop)]
        next_index = stop

    # Families have their own streams and files, so workers can write them in any order
    write = partial(
        _write_family,
        dataset,
        base_seed,
        image_config,
        kind,
        noise_config,
        realization_count,
    )
    if workers == 1:
        for job in jobs:
            write(job)
    else:
        with ProcessPoolExecutor(workers) as pool:
            for _ in pool.map(write, jobs, chunksize=32):
                pass
    return dataset


def _write_family(
    dataset: NpzDataset,
    base_seed: int,
    image_config: SquareImageConfig,
    kind: ComponentTreeKind,
    noise_config: NoiseConfig,
    realization_count: int,
    job: tuple[str, int],
) -> None:
    split, sample_index = job
    family = build_sample_family(
        base_seed, sample_index, image_config, kind, noise_config, realization_count
    )
    dataset.save(split, family)
