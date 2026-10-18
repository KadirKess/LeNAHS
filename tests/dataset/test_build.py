import numpy as np
import pytest

from lenahs.component_tree import ComponentTreeKind
from lenahs.dataset import NpzDataset
from lenahs.dataset.build import build_sample_family, generate_dataset
from lenahs.generation import SaltAndPepperNoise, SquareImageConfig

SMALL_IMAGES = SquareImageConfig(
    height=16, width=16, min_squares=2, max_squares=4, min_side=2, max_side=8
)
NOISE = SaltAndPepperNoise(probability=0.05)


def _family(sample_index: int, base_seed: int = 3):
    return build_sample_family(
        base_seed, sample_index, SMALL_IMAGES, ComponentTreeKind.MAX_TREE, NOISE, 2
    )


def _images(family):
    return [family.clean.observed.image] + [
        o.sample.observed.image for o in family.noisy
    ]


def test_family_is_reproducible_from_its_seeds() -> None:
    first, second = _family(5), _family(5)

    for a, b in zip(_images(first), _images(second), strict=True):
        np.testing.assert_array_equal(a, b)


def test_family_holds_clean_and_distinct_realizations() -> None:
    family = _family(5)

    assert [o.realization_index for o in family.noisy] == [0, 1]
    assert all(o.noise_config == NOISE for o in family.noisy)
    for observation in family.noisy:
        np.testing.assert_array_equal(
            observation.sample.clean.image, family.clean.clean.image
        )
    assert not np.array_equal(*[o.sample.observed.image for o in family.noisy])


@pytest.mark.parametrize("workers", [1, 2])
def test_splits_take_disjoint_index_ranges(tmp_path, workers) -> None:
    counts = {"train": 3, "val": 2, "test": 1}
    generate_dataset(
        tmp_path / "data",
        3,
        counts,
        SMALL_IMAGES,
        ComponentTreeKind.MAX_TREE,
        NOISE,
        2,
        workers,
    )

    names = {
        split: sorted(p.name for p in (tmp_path / "data" / split).glob("*.npz"))
        for split in counts
    }
    assert names == {
        "train": [f"sample_{i:08d}.npz" for i in range(3)],
        "val": [f"sample_{i:08d}.npz" for i in range(3, 5)],
        "test": ["sample_00000005.npz"],
    }
    loaded = NpzDataset.open(tmp_path / "data").load("val", 3)
    np.testing.assert_array_equal(
        loaded.clean.observed.image, _family(3).clean.observed.image
    )
