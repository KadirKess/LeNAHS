import numpy as np
import pytest

from lenahs.generation.images import SquareImageConfig, generate_square_image


def test_fixed_background_without_squares() -> None:
    config = SquareImageConfig(
        height=3,
        width=5,
        min_squares=0,
        max_squares=0,
        min_side=1,
        max_side=1,
        min_background_level=17,
        max_background_level=17,
    )

    image = generate_square_image(config, np.random.default_rng(4))

    np.testing.assert_array_equal(image, np.full((3, 5), 17, dtype=np.uint8))


def test_square_generation_is_reproducible() -> None:
    config = SquareImageConfig(height=24, width=20, max_side=12)

    first = generate_square_image(config, np.random.default_rng(12))
    second = generate_square_image(config, np.random.default_rng(12))

    np.testing.assert_array_equal(first, second)
    assert first.shape == (24, 20)
    assert first.dtype == np.uint8


@pytest.mark.parametrize(
    "config",
    [
        SquareImageConfig(height=0),
        SquareImageConfig(min_squares=-1),
        SquareImageConfig(min_squares=5, max_squares=4),
        SquareImageConfig(min_side=0),
        SquareImageConfig(height=8, width=8, max_side=9),
        SquareImageConfig(min_background_level=-1),
        SquareImageConfig(min_square_level=100, max_square_level=99),
    ],
)
def test_invalid_square_configuration_fails(config: SquareImageConfig) -> None:
    with pytest.raises(ValueError):
        generate_square_image(config, np.random.default_rng(0))
