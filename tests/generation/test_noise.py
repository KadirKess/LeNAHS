from math import sqrt

import numpy as np
import pytest

from lenahs.generation.noise import (
    AdditiveGaussianNoise,
    PoissonNoise,
    SaltAndPepperNoise,
    SpeckleNoise,
    corrupt_image,
)


def test_salt_and_pepper_extreme_fractions() -> None:
    image = np.full((4, 5), 80, dtype=np.uint8)

    pepper = corrupt_image(
        image,
        SaltAndPepperNoise(probability=1.0, salt_fraction=0.0),
        np.random.default_rng(1),
    )
    salt = corrupt_image(
        image,
        SaltAndPepperNoise(probability=1.0, salt_fraction=1.0),
        np.random.default_rng(1),
    )

    np.testing.assert_array_equal(pepper, np.zeros_like(image))
    np.testing.assert_array_equal(salt, np.full_like(image, 255))
    np.testing.assert_array_equal(image, np.full_like(image, 80))


def test_salt_and_pepper_event_frequencies_match_configuration() -> None:
    image = np.full((400, 500), 127, dtype=np.uint8)
    probability = 0.2
    salt_fraction = 0.25

    corrupted = corrupt_image(
        image,
        SaltAndPepperNoise(
            probability=probability,
            salt_fraction=salt_fraction,
        ),
        np.random.default_rng(19),
    )

    pixel_count = image.size
    expected_probabilities = {
        0: probability * (1.0 - salt_fraction),
        127: 1.0 - probability,
        255: probability * salt_fraction,
    }
    for value, expected_probability in expected_probabilities.items():
        observed_count = np.count_nonzero(corrupted == value)
        expected_count = pixel_count * expected_probability
        tolerance = 6.0 * sqrt(
            pixel_count * expected_probability * (1.0 - expected_probability)
        )
        assert abs(observed_count - expected_count) <= tolerance


@pytest.mark.parametrize(
    "config",
    [
        SaltAndPepperNoise(probability=0.0),
        AdditiveGaussianNoise(standard_deviation=0.0),
        SpeckleNoise(standard_deviation=0.0),
    ],
)
def test_zero_noise_is_an_exact_copy(config: object) -> None:
    image = np.array([[0, 80], [160, 255]], dtype=np.uint8)

    corrupted = corrupt_image(image, config, np.random.default_rng(3))

    np.testing.assert_array_equal(corrupted, image)
    assert corrupted is not image


@pytest.mark.parametrize(
    "config",
    [
        AdditiveGaussianNoise(standard_deviation=8.0),
        SpeckleNoise(standard_deviation=0.2),
        PoissonNoise(peak=20.0),
    ],
)
def test_sampled_noise_is_reproducible_and_uint8(config: object) -> None:
    image = np.array([[0, 32, 128], [180, 220, 255]], dtype=np.uint8)

    first = corrupt_image(image, config, np.random.default_rng(7))
    second = corrupt_image(image, config, np.random.default_rng(7))

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.uint8
    assert first.shape == image.shape
    np.testing.assert_array_equal(image, [[0, 32, 128], [180, 220, 255]])


def test_additive_gaussian_sample_moments_match_configuration() -> None:
    image = np.full((400, 500), 128, dtype=np.uint8)
    mean = 3.0
    standard_deviation = 12.0

    corrupted = corrupt_image(
        image,
        AdditiveGaussianNoise(mean=mean, standard_deviation=standard_deviation),
        np.random.default_rng(23),
    )
    perturbation = corrupted.astype(np.float64) - image

    sample_count = image.size
    mean_tolerance = 6.0 * standard_deviation / sqrt(sample_count)
    standard_deviation_tolerance = (
        6.0 * standard_deviation / sqrt(2.0 * (sample_count - 1))
    )
    assert abs(perturbation.mean() - mean) <= mean_tolerance
    assert (
        abs(perturbation.std(ddof=1) - standard_deviation)
        <= standard_deviation_tolerance
    )


def test_speckle_sample_moments_scale_with_intensity() -> None:
    row_length = 200_000
    image = np.vstack(
        [
            np.zeros(row_length, dtype=np.uint8),
            np.full(row_length, 40, dtype=np.uint8),
            np.full(row_length, 120, dtype=np.uint8),
        ]
    )
    standard_deviation = 0.1

    corrupted = corrupt_image(
        image,
        SpeckleNoise(standard_deviation=standard_deviation),
        np.random.default_rng(29),
    )
    perturbation = corrupted.astype(np.float64) - image

    np.testing.assert_array_equal(corrupted[0], image[0])
    for row, intensity in [(1, 40), (2, 120)]:
        expected_standard_deviation = intensity * standard_deviation
        mean_tolerance = 6.0 * expected_standard_deviation / sqrt(row_length)
        standard_deviation_tolerance = (
            6.0 * expected_standard_deviation / sqrt(2.0 * (row_length - 1))
        )
        assert abs(perturbation[row].mean()) <= mean_tolerance
        assert (
            abs(perturbation[row].std(ddof=1) - expected_standard_deviation)
            <= standard_deviation_tolerance
        )


def test_poisson_sample_moments_match_peak_parameterization() -> None:
    intensity = 100
    peak = 50.0
    image = np.full((400, 500), intensity, dtype=np.uint8)

    corrupted = corrupt_image(
        image,
        PoissonNoise(peak=peak),
        np.random.default_rng(31),
    )

    sample_count = image.size
    output_scale = 255.0 / peak
    expected_count = intensity / output_scale
    expected_variance = intensity * output_scale
    mean_tolerance = 6.0 * sqrt(expected_variance / sample_count)
    variance_tolerance = 6.0 * sqrt(
        output_scale**4 * (expected_count + 2.0 * expected_count**2) / sample_count
    )
    assert abs(corrupted.mean() - intensity) <= mean_tolerance
    assert abs(corrupted.var(ddof=1) - expected_variance) <= variance_tolerance


def test_gaussian_rounding_and_clipping_follow_uint8_contract() -> None:
    image = np.array([[0, 1, 254, 255]], dtype=np.uint8)

    half_step = corrupt_image(
        image,
        AdditiveGaussianNoise(mean=0.5, standard_deviation=0.0),
        np.random.default_rng(0),
    )
    clipped_low = corrupt_image(
        image,
        AdditiveGaussianNoise(mean=-1000.0, standard_deviation=0.0),
        np.random.default_rng(0),
    )
    clipped_high = corrupt_image(
        image,
        AdditiveGaussianNoise(mean=1000.0, standard_deviation=0.0),
        np.random.default_rng(0),
    )

    np.testing.assert_array_equal(half_step, [[0, 2, 254, 255]])
    np.testing.assert_array_equal(clipped_low, np.zeros_like(image))
    np.testing.assert_array_equal(clipped_high, np.full_like(image, 255))


@pytest.mark.parametrize(
    "config",
    [
        SaltAndPepperNoise(probability=-0.1),
        SaltAndPepperNoise(probability=1.1),
        SaltAndPepperNoise(probability=float("nan")),
        SaltAndPepperNoise(salt_fraction=1.1),
        SaltAndPepperNoise(salt_fraction=float("inf")),
        AdditiveGaussianNoise(standard_deviation=-1.0),
        AdditiveGaussianNoise(standard_deviation=float("inf")),
        AdditiveGaussianNoise(mean=float("nan")),
        SpeckleNoise(standard_deviation=-1.0),
        SpeckleNoise(standard_deviation=float("nan")),
        PoissonNoise(peak=0.0),
        PoissonNoise(peak=float("inf")),
    ],
)
def test_invalid_noise_configuration_fails(config: object) -> None:
    image = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError):
        corrupt_image(image, config, np.random.default_rng(0))


def test_noise_requires_two_dimensional_uint8_image() -> None:
    with pytest.raises(TypeError):
        corrupt_image(
            np.zeros((2, 2), dtype=np.float32),
            SaltAndPepperNoise(),
            np.random.default_rng(0),
        )

    with pytest.raises(ValueError):
        corrupt_image(
            np.zeros((2, 2, 1), dtype=np.uint8),
            SaltAndPepperNoise(),
            np.random.default_rng(0),
        )
