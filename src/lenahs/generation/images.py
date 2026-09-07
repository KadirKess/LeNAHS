from dataclasses import dataclass
from math import exp, log

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SquareImageConfig:
    height: int = 256
    width: int = 256
    min_squares: int = 64
    max_squares: int = 192
    min_side: int = 4
    max_side: int = 64
    min_background_level: int = 0
    max_background_level: int = 255
    min_square_level: int = 0
    max_square_level: int = 255


def _validate_level_interval(name: str, minimum: int, maximum: int) -> None:
    if not 0 <= minimum <= maximum <= 255:
        raise ValueError(f"{name} levels must satisfy 0 <= minimum <= maximum <= 255.")


def _validate_config(config: SquareImageConfig) -> None:
    if config.height <= 0 or config.width <= 0:
        raise ValueError("Image height and width must be positive.")
    if config.min_squares < 0 or config.max_squares < config.min_squares:
        raise ValueError("Square counts must satisfy 0 <= min_squares <= max_squares.")
    if config.min_side <= 0 or config.max_side < config.min_side:
        raise ValueError("Side lengths must satisfy 0 < min_side <= max_side.")
    if config.max_side > min(config.height, config.width):
        raise ValueError("max_side must fit within both image dimensions.")

    _validate_level_interval(
        "Background", config.min_background_level, config.max_background_level
    )
    _validate_level_interval("Square", config.min_square_level, config.max_square_level)


def _sample_side(config: SquareImageConfig, rng: np.random.Generator) -> int:
    if config.min_side == config.max_side:
        return config.min_side

    value = exp(
        rng.uniform(
            log(config.min_side),
            log(config.max_side + 1),
        )
    )
    return min(int(value), config.max_side)


def generate_square_image(
    config: SquareImageConfig,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    """Paint random squares over a random grayscale background."""

    _validate_config(config)

    background_level = rng.integers(
        config.min_background_level,
        config.max_background_level + 1,
    )
    clean_image = np.full(
        (config.height, config.width),
        background_level,
        dtype=np.uint8,
    )

    square_count = int(rng.integers(config.min_squares, config.max_squares + 1))
    for _ in range(square_count):
        side = _sample_side(config, rng)
        row = int(rng.integers(0, config.height - side + 1))
        column = int(rng.integers(0, config.width - side + 1))
        level = rng.integers(config.min_square_level, config.max_square_level + 1)
        clean_image[row : row + side, column : column + side] = level

    return clean_image
