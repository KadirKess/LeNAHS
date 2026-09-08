from enum import Enum, unique
from numbers import Integral

import numpy as np


@unique
class GenerationStage(Enum):
    """Stable identifiers for independently reproducible random streams."""

    SQUARE_IMAGE = 0
    SALT_AND_PEPPER_NOISE = 1
    ADDITIVE_GAUSSIAN_NOISE = 2
    SPECKLE_NOISE = 3
    POISSON_NOISE = 4


def _validate_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def derive_rng(
    base_seed: int,
    sample_index: int,
    stage: GenerationStage,
    *,
    realization_index: int = 0,
) -> np.random.Generator:
    """Derive one sample-stage stream independently of generation order."""

    _validate_nonnegative_integer("base_seed", base_seed)
    _validate_nonnegative_integer("sample_index", sample_index)
    _validate_nonnegative_integer("realization_index", realization_index)
    if not isinstance(stage, GenerationStage):
        raise TypeError("stage must be a GenerationStage.")

    identity = [int(base_seed), int(sample_index), stage.value]
    if realization_index != 0:
        identity.append(int(realization_index))

    seed_sequence = np.random.SeedSequence(identity)
    return np.random.Generator(np.random.PCG64(seed_sequence))
