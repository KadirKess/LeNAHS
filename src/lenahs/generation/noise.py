from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SaltAndPepperNoise:
    probability: float = 0.01
    salt_fraction: float = 0.5


@dataclass(frozen=True)
class AdditiveGaussianNoise:
    standard_deviation: float = 25.0
    mean: float = 0.0


@dataclass(frozen=True)
class SpeckleNoise:
    standard_deviation: float = 0.1


@dataclass(frozen=True)
class PoissonNoise:
    peak: float = 30.0


type NoiseConfig = (
    SaltAndPepperNoise | AdditiveGaussianNoise | SpeckleNoise | PoissonNoise
)


def _validate_image(clean_image: NDArray[np.uint8]) -> None:
    if not isinstance(clean_image, np.ndarray):
        raise TypeError("clean_image must be a NumPy array.")
    if clean_image.ndim != 2:
        raise ValueError("clean_image must be two-dimensional.")
    if clean_image.dtype != np.uint8:
        raise TypeError("clean_image must have dtype uint8.")
    if clean_image.size == 0:
        raise ValueError("clean_image must not be empty.")


def _validate_probability(name: str, value: float) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _to_uint8(values: NDArray[np.floating]) -> NDArray[np.uint8]:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _apply_salt_and_pepper(
    clean_image: NDArray[np.uint8],
    config: SaltAndPepperNoise,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    _validate_probability("probability", config.probability)
    _validate_probability("salt_fraction", config.salt_fraction)

    corrupted_image = clean_image.copy()
    if config.probability == 0.0:
        return corrupted_image

    draws = rng.random(clean_image.shape)
    pepper_limit = config.probability * (1.0 - config.salt_fraction)
    corrupted_image[draws < pepper_limit] = 0
    corrupted_image[(draws >= pepper_limit) & (draws < config.probability)] = 255
    return corrupted_image


def _apply_additive_gaussian(
    clean_image: NDArray[np.uint8],
    config: AdditiveGaussianNoise,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    if not isfinite(config.mean):
        raise ValueError("mean must be finite.")
    if not isfinite(config.standard_deviation) or config.standard_deviation < 0.0:
        raise ValueError("standard_deviation must be finite and nonnegative.")
    if config.mean == 0.0 and config.standard_deviation == 0.0:
        return clean_image.copy()

    perturbation = rng.normal(
        config.mean,
        config.standard_deviation,
        clean_image.shape,
    )
    return _to_uint8(clean_image.astype(np.float64) + perturbation)


def _apply_speckle(
    clean_image: NDArray[np.uint8],
    config: SpeckleNoise,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    if not isfinite(config.standard_deviation) or config.standard_deviation < 0.0:
        raise ValueError("standard_deviation must be finite and nonnegative.")
    if config.standard_deviation == 0.0:
        return clean_image.copy()

    clean_values = clean_image.astype(np.float64)
    multiplier = rng.normal(0.0, config.standard_deviation, clean_image.shape)
    return _to_uint8(clean_values + clean_values * multiplier)


def _apply_poisson(
    clean_image: NDArray[np.uint8],
    config: PoissonNoise,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    if not isfinite(config.peak) or config.peak <= 0.0:
        raise ValueError("peak must be finite and positive.")

    expected_counts = clean_image.astype(np.float64) * (config.peak / 255.0)
    sampled_counts = rng.poisson(expected_counts)
    return _to_uint8(sampled_counts.astype(np.float64) * (255.0 / config.peak))


def corrupt_image(
    clean_image: NDArray[np.uint8],
    config: NoiseConfig,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    """Apply one configured corruption without modifying the clean image."""

    _validate_image(clean_image)

    if isinstance(config, SaltAndPepperNoise):
        return _apply_salt_and_pepper(clean_image, config, rng)
    if isinstance(config, AdditiveGaussianNoise):
        return _apply_additive_gaussian(clean_image, config, rng)
    if isinstance(config, SpeckleNoise):
        return _apply_speckle(clean_image, config, rng)
    if isinstance(config, PoissonNoise):
        return _apply_poisson(clean_image, config, rng)

    raise TypeError(f"Unsupported noise configuration: {type(config).__name__}.")
