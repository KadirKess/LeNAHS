import numpy as np
import pytest

from lenahs.component_tree import (
    ComponentTreeKind,
    build_reference_component_tree,
    build_reference_rag,
)
from lenahs.dataset import (
    NoisyObservation,
    SampleFamily,
    load_sample_family,
    save_sample_family,
)
from lenahs.generation import (
    AdditiveGaussianNoise,
    GenerationSample,
    GenerationStage,
    PoissonNoise,
    SaltAndPepperNoise,
    SpeckleNoise,
    build_generation_sample,
    derive_rng,
)


def _build_reference_tree(image, kind):
    graph = build_reference_rag(image)
    return graph, build_reference_component_tree(graph, kind)


def _build_family(image: np.ndarray) -> SampleFamily:
    clean = build_generation_sample(
        image,
        ComponentTreeKind.MAX_TREE,
        tree_builder=_build_reference_tree,
    )
    configurations = [
        (
            SaltAndPepperNoise(probability=0.4, salt_fraction=0.25),
            GenerationStage.SALT_AND_PEPPER_NOISE,
        ),
        (
            AdditiveGaussianNoise(standard_deviation=12.0, mean=-1.0),
            GenerationStage.ADDITIVE_GAUSSIAN_NOISE,
        ),
        (SpeckleNoise(standard_deviation=0.2), GenerationStage.SPECKLE_NOISE),
        (PoissonNoise(peak=18.0), GenerationStage.POISSON_NOISE),
    ]
    noisy = []
    for realization_index, (noise_config, stage) in enumerate(configurations):
        sample = build_generation_sample(
            image,
            ComponentTreeKind.MAX_TREE,
            noise_config,
            derive_rng(
                41,
                7,
                stage,
                realization_index=realization_index,
            ),
            tree_builder=_build_reference_tree,
        )
        noisy.append(
            NoisyObservation(
                realization_index=realization_index,
                noise_config=noise_config,
                sample=sample,
            )
        )

    return SampleFamily(sample_index=7, clean=clean, noisy=tuple(noisy))


def _assert_tree_image_equal(actual, expected) -> None:
    assert type(actual) is type(expected)
    assert actual.component_tree.kind == expected.component_tree.kind
    for actual_array, expected_array in [
        (actual.image, expected.image),
        (actual.graph.label_map, expected.graph.label_map),
        (actual.graph.edge_index, expected.graph.edge_index),
        (actual.graph.levels, expected.graph.levels),
        (actual.graph.pixel_counts, expected.graph.pixel_counts),
        (actual.graph.anchors, expected.graph.anchors),
        (actual.component_tree.parents, expected.component_tree.parents),
    ]:
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        np.testing.assert_array_equal(actual_array, expected_array)


def _assert_sample_equal(actual: GenerationSample, expected: GenerationSample) -> None:
    assert type(actual) is GenerationSample
    _assert_tree_image_equal(actual.clean, expected.clean)
    if expected.corrupted is None:
        assert actual.corrupted is None
    else:
        assert actual.corrupted is not None
        _assert_tree_image_equal(actual.corrupted, expected.corrupted)

    for actual_array, expected_array in [
        (
            actual.corruption_targets.noise_direction,
            expected.corruption_targets.noise_direction,
        ),
        (
            actual.corruption_targets.overlap_purity,
            expected.corruption_targets.overlap_purity,
        ),
        (
            actual.corruption_targets.overlap_clean_zone_count,
            expected.corruption_targets.overlap_clean_zone_count,
        ),
    ]:
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        np.testing.assert_array_equal(actual_array, expected_array)


def test_family_round_trip_reconstructs_existing_dataclasses(tmp_path) -> None:
    family = _build_family(np.array([[10, 10, 30], [10, 20, 30]], dtype=np.uint8))
    path = tmp_path / "train" / "sample_000007.npz"

    save_sample_family(path, family)
    loaded = load_sample_family(path)

    assert type(loaded) is SampleFamily
    assert loaded.sample_index == family.sample_index
    _assert_sample_equal(loaded.clean, family.clean)
    assert len(loaded.noisy) == len(family.noisy)
    for actual, expected in zip(loaded.noisy, family.noisy, strict=True):
        assert type(actual) is NoisyObservation
        assert actual.realization_index == expected.realization_index
        assert actual.noise_config == expected.noise_config
        _assert_sample_equal(actual.sample, expected.sample)
        assert actual.sample.clean is loaded.clean.clean


def test_family_file_stores_clean_arrays_only_once(tmp_path) -> None:
    family = _build_family(np.array([[10, 20], [30, 40]], dtype=np.uint8))
    path = tmp_path / "sample_000007.npz"
    save_sample_family(path, family)

    with np.load(path, allow_pickle=False) as arrays:
        clean_images = [name for name in arrays.files if name == "clean.image"]
        noisy_clean_fields = [
            name
            for name in arrays.files
            if name.startswith("noisy.") and ".clean." in name
        ]

    assert clean_images == ["clean.image"]
    assert noisy_clean_fields == []


def test_separate_family_files_can_have_different_image_shapes(tmp_path) -> None:
    first = _build_family(np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8))
    second = _build_family(np.array([[10, 20], [30, 40], [50, 60]], dtype=np.uint8))
    first_path = tmp_path / "sample_000007.npz"
    second_path = tmp_path / "sample_000008.npz"
    second = SampleFamily(sample_index=8, clean=second.clean, noisy=second.noisy)

    save_sample_family(first_path, first)
    save_sample_family(second_path, second)

    assert load_sample_family(first_path).clean.clean.image.shape == (2, 3)
    assert load_sample_family(second_path).clean.clean.image.shape == (3, 2)


def test_clean_family_without_noisy_observations_round_trips(tmp_path) -> None:
    family = _build_family(np.array([[10, 20]], dtype=np.uint8))
    family = SampleFamily(
        sample_index=family.sample_index, clean=family.clean, noisy=()
    )
    path = tmp_path / "sample_000007.npz"

    save_sample_family(path, family)
    loaded = load_sample_family(path)

    assert loaded.noisy == ()
    _assert_sample_equal(loaded.clean, family.clean)


def test_family_writer_refuses_to_replace_an_existing_sample(tmp_path) -> None:
    family = _build_family(np.array([[10, 20]], dtype=np.uint8))
    path = tmp_path / "sample_000007.npz"
    save_sample_family(path, family)

    with pytest.raises(FileExistsError, match="already exists"):
        save_sample_family(path, family)


def test_family_writer_requires_an_npz_path(tmp_path) -> None:
    family = _build_family(np.array([[10, 20]], dtype=np.uint8))

    with pytest.raises(ValueError, match="must end in .npz"):
        save_sample_family(tmp_path / "sample_000007.data", family)


def test_family_rejects_duplicate_realization_indices(tmp_path) -> None:
    family = _build_family(np.array([[10, 20]], dtype=np.uint8))
    duplicate = NoisyObservation(
        realization_index=family.noisy[0].realization_index,
        noise_config=family.noisy[1].noise_config,
        sample=family.noisy[1].sample,
    )
    invalid = SampleFamily(
        sample_index=family.sample_index,
        clean=family.clean,
        noisy=(family.noisy[0], duplicate),
    )

    with pytest.raises(ValueError, match="must be unique"):
        save_sample_family(tmp_path / "sample_000007.npz", invalid)


def test_family_rejects_a_different_clean_reference(tmp_path) -> None:
    family = _build_family(np.array([[10, 20]], dtype=np.uint8))
    other_family = _build_family(np.array([[10, 30]], dtype=np.uint8))
    invalid = SampleFamily(
        sample_index=family.sample_index,
        clean=family.clean,
        noisy=(other_family.noisy[0],),
    )

    with pytest.raises(ValueError, match="family clean image"):
        save_sample_family(tmp_path / "sample_000007.npz", invalid)


def test_family_reader_rejects_an_unknown_schema_version(tmp_path) -> None:
    path = tmp_path / "sample_000007.npz"
    np.savez_compressed(path, schema_version=np.asarray(9, dtype=np.int64))

    with pytest.raises(ValueError, match="Unsupported schema version 9"):
        load_sample_family(path)
