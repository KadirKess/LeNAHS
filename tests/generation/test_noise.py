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


@pytest.mark.parametrize(
    "config",
    [
        SaltAndPepperNoise(probability=-0.1),
        SaltAndPepperNoise(salt_fraction=1.1),
        AdditiveGaussianNoise(standard_deviation=-1.0),
        SpeckleNoise(standard_deviation=-1.0),
        PoissonNoise(peak=0.0),
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
