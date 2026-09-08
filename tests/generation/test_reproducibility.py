import numpy as np
import pytest

from lenahs.generation import (
    AdditiveGaussianNoise,
    GenerationStage,
    PoissonNoise,
    SaltAndPepperNoise,
    SpeckleNoise,
    SquareImageConfig,
    corrupt_image,
    derive_rng,
    generate_square_image,
)

SQUARE_CONFIG = SquareImageConfig(
    height=18,
    width=20,
    min_squares=4,
    max_squares=8,
    min_side=2,
    max_side=7,
)


def _generate_pair(base_seed: int, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
    clean_image = generate_square_image(
        SQUARE_CONFIG,
        derive_rng(base_seed, sample_index, GenerationStage.SQUARE_IMAGE),
    )
    corrupted_image = corrupt_image(
        clean_image,
        AdditiveGaussianNoise(standard_deviation=12.0),
        derive_rng(
            base_seed,
            sample_index,
            GenerationStage.ADDITIVE_GAUSSIAN_NOISE,
        ),
    )
    return clean_image, corrupted_image


def test_stage_identifiers_are_stable() -> None:
    assert {stage.name: stage.value for stage in GenerationStage} == {
        "SQUARE_IMAGE": 0,
        "SALT_AND_PEPPER_NOISE": 1,
        "ADDITIVE_GAUSSIAN_NOISE": 2,
        "SPECKLE_NOISE": 3,
        "POISSON_NOISE": 4,
    }


def test_seed_derivation_has_stable_reference_streams() -> None:
    expected = {
        GenerationStage.SQUARE_IMAGE: [14794472610689844499, 15822706442690279510],
        GenerationStage.SALT_AND_PEPPER_NOISE: [
            4176691949612360892,
            14363986460750849246,
        ],
        GenerationStage.ADDITIVE_GAUSSIAN_NOISE: [
            11157848788723678081,
            9415907038508155347,
        ],
        GenerationStage.SPECKLE_NOISE: [
            7530612779834857058,
            1149816173006977436,
        ],
        GenerationStage.POISSON_NOISE: [
            15417801977815605657,
            3675370332953186241,
        ],
    }

    actual = {
        stage: derive_rng(123, 45, stage).bit_generator.random_raw(2).tolist()
        for stage in expected
    }

    assert actual == expected


@pytest.mark.parametrize(
    "stage, noise_config",
    [
        (
            GenerationStage.SALT_AND_PEPPER_NOISE,
            SaltAndPepperNoise(probability=0.2),
        ),
        (
            GenerationStage.ADDITIVE_GAUSSIAN_NOISE,
            AdditiveGaussianNoise(standard_deviation=12.0),
        ),
        (GenerationStage.SPECKLE_NOISE, SpeckleNoise(standard_deviation=0.2)),
        (GenerationStage.POISSON_NOISE, PoissonNoise(peak=18.0)),
    ],
)
def test_same_sample_identity_reproduces_exact_arrays(stage, noise_config) -> None:
    first_clean = generate_square_image(
        SQUARE_CONFIG,
        derive_rng(91, 7, GenerationStage.SQUARE_IMAGE),
    )
    second_clean = generate_square_image(
        SQUARE_CONFIG,
        derive_rng(91, 7, GenerationStage.SQUARE_IMAGE),
    )
    first_corrupted = corrupt_image(
        first_clean,
        noise_config,
        derive_rng(91, 7, stage),
    )
    second_corrupted = corrupt_image(
        second_clean,
        noise_config,
        derive_rng(91, 7, stage),
    )

    np.testing.assert_array_equal(first_clean, second_clean)
    np.testing.assert_array_equal(first_corrupted, second_corrupted)


def test_generation_order_does_not_change_samples() -> None:
    forward = {index: _generate_pair(48, index) for index in range(6)}
    reverse = {index: _generate_pair(48, index) for index in reversed(range(6))}

    for index in forward:
        np.testing.assert_array_equal(forward[index][0], reverse[index][0])
        np.testing.assert_array_equal(forward[index][1], reverse[index][1])


def test_simulated_worker_partition_does_not_change_samples() -> None:
    sequential = {index: _generate_pair(23, index) for index in range(8)}
    worker_order = [0, 2, 4, 6, 1, 3, 5, 7]
    partitioned = {index: _generate_pair(23, index) for index in worker_order}

    for index in sequential:
        np.testing.assert_array_equal(sequential[index][0], partitioned[index][0])
        np.testing.assert_array_equal(sequential[index][1], partitioned[index][1])


def test_sample_indices_have_distinct_streams() -> None:
    draws = {
        derive_rng(5, index, GenerationStage.SQUARE_IMAGE).bytes(32)
        for index in range(16)
    }

    assert len(draws) == 16


def test_named_stages_have_distinct_streams() -> None:
    draws = {derive_rng(5, 11, stage).bytes(32) for stage in GenerationStage}

    assert len(draws) == len(GenerationStage)


def test_consuming_one_stage_does_not_advance_another() -> None:
    expected = derive_rng(17, 3, GenerationStage.POISSON_NOISE).bytes(32)

    image_rng = derive_rng(17, 3, GenerationStage.SQUARE_IMAGE)
    image_rng.bytes(4096)
    actual = derive_rng(17, 3, GenerationStage.POISSON_NOISE).bytes(32)

    assert actual == expected


def test_global_random_state_does_not_influence_derived_stream() -> None:
    previous_state = np.random.get_state()
    try:
        np.random.seed(1)
        first = derive_rng(72, 4, GenerationStage.SPECKLE_NOISE).bytes(32)
        np.random.seed(999)
        second = derive_rng(72, 4, GenerationStage.SPECKLE_NOISE).bytes(32)
    finally:
        np.random.set_state(previous_state)

    assert first == second


@pytest.mark.parametrize("name", ["base_seed", "sample_index"])
@pytest.mark.parametrize(
    "value, error",
    [(-1, ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_invalid_sample_identity_fails(
    name: str,
    value: object,
    error: type[Exception],
) -> None:
    arguments = {
        "base_seed": 1,
        "sample_index": 2,
        "stage": GenerationStage.SQUARE_IMAGE,
    }
    arguments[name] = value

    with pytest.raises(error):
        derive_rng(**arguments)


def test_stage_must_be_named() -> None:
    with pytest.raises(TypeError, match="GenerationStage"):
        derive_rng(1, 2, 0)
