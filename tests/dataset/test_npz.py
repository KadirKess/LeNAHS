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


def _build_family(image: np.ndarray, kind=ComponentTreeKind.MAX_TREE) -> SampleFamily:
    clean = build_generation_sample(
        image,
        kind,
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
            kind,
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
    _assert_tree_image_equal(actual.observed, expected.observed)
    for name in ("corrective_parents", "actions"):
        actual_array = getattr(actual.correction_targets, name)
        expected_array = getattr(expected.correction_targets, name)
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        np.testing.assert_array_equal(actual_array, expected_array)
    for name in ("overlap_purity", "overlap_clean_zone_count", "missing_clean_zones"):
        actual_array = getattr(actual.correction_diagnostics, name)
        expected_array = getattr(expected.correction_diagnostics, name)
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        np.testing.assert_array_equal(actual_array, expected_array)
    for name in ("clean_root_missing", "artificial_root_attachment_count"):
        actual_value = getattr(actual.correction_diagnostics, name)
        expected_value = getattr(expected.correction_diagnostics, name)
        assert type(actual_value) is type(expected_value)
        assert actual_value == expected_value


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_family_round_trip_reconstructs_existing_dataclasses(tmp_path, kind) -> None:
    family = _build_family(np.array([[10, 10, 30], [10, 20, 30]], dtype=np.uint8), kind)
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

        assert int(arrays["schema_version"]) == 2
        assert not any("noise_direction" in name for name in arrays.files)
        assert all(arrays[name].dtype != object for name in arrays.files)
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


def test_loading_and_adaptation_do_not_rederive_targets(tmp_path, monkeypatch) -> None:
    import lenahs.generation.samples as sample_module
    import lenahs.generation.targets as target_module
    from lenahs.model import adapt_generation_sample

    family = _build_family(np.array([[10, 10, 30], [10, 20, 30]], dtype=np.uint8))
    path = tmp_path / "family.npz"
    save_sample_family(path, family)

    def forbidden(*args, **kwargs):
        pytest.fail("Stored correction supervision must not be rederived.")

    monkeypatch.setattr(sample_module, "derive_correction_targets", forbidden)
    monkeypatch.setattr(target_module, "derive_correction_targets", forbidden)
    loaded = load_sample_family(path)
    for observation in loaded.noisy:
        adapted = adapt_generation_sample(observation.sample)
        np.testing.assert_array_equal(
            adapted.corrective_parents.numpy(),
            observation.sample.correction_targets.corrective_parents,
        )


@pytest.mark.parametrize(
    ("key", "value", "error", "message"),
    [
        ("clean.targets.corrective_parents", np.array([0, 0]), ValueError, "shape"),
        ("clean.targets.corrective_parents", np.array([0, 0, 3]), ValueError, "local"),
        ("clean.targets.corrective_parents", np.array([0, 0, -1]), ValueError, "local"),
        (
            "clean.targets.corrective_parents",
            np.array([0, 0, 0], dtype=np.int32),
            TypeError,
            "int64",
        ),
        ("clean.targets.corrective_parents", np.array([0, 2, 1]), ValueError, "cycle"),
        (
            "clean.targets.corrective_parents",
            np.array([0, 1, 1]),
            ValueError,
            "one self-parented",
        ),
        ("clean.targets.actions", np.array([0, 2, 2]), ValueError, "KEEP_ZERO root"),
        ("clean.targets.actions", np.array([1, 3, 2]), ValueError, "actions"),
        ("clean.targets.actions", np.array([1, -1, 2]), ValueError, "actions"),
        (
            "clean.targets.actions",
            np.array([1, 2, 2], dtype=np.int32),
            TypeError,
            "int64",
        ),
        (
            "clean.diagnostics.overlap_purity",
            np.array([1.0, np.nan, 1.0]),
            ValueError,
            "finite",
        ),
        (
            "clean.diagnostics.overlap_purity",
            np.array([1.0, 1.1, 1.0]),
            ValueError,
            "finite",
        ),
        (
            "clean.diagnostics.overlap_purity",
            np.array([1.0, 0.0, 1.0]),
            ValueError,
            "finite",
        ),
        (
            "clean.diagnostics.overlap_clean_zone_count",
            np.array([1, 4, 1]),
            ValueError,
            "clean node count",
        ),
        (
            "clean.diagnostics.missing_clean_zones",
            np.array([False, False]),
            ValueError,
            "shape",
        ),
        (
            "clean.diagnostics.missing_clean_zones",
            np.array([0, 0, 0]),
            TypeError,
            "bool",
        ),
        (
            "clean.diagnostics.clean_root_missing",
            np.asarray(0),
            ValueError,
            "scalar with dtype bool",
        ),
        (
            "clean.diagnostics.clean_root_missing",
            np.array([False]),
            ValueError,
            "scalar",
        ),
        (
            "clean.diagnostics.artificial_root_attachment_count",
            np.asarray(-1),
            ValueError,
            "nonnegative",
        ),
        (
            "clean.diagnostics.artificial_root_attachment_count",
            np.asarray(3),
            ValueError,
            "less than node count",
        ),
        (
            "noisy.0.component_tree.kind",
            np.asarray("min_tree"),
            ValueError,
            "kinds must match",
        ),
        ("noisy.0.image", np.array([[1]], dtype=np.uint8), ValueError, "same shape"),
        ("realization_indices", np.array([0, 0, 2, 3]), ValueError, "unique"),
        (
            "clean.targets.actions",
            np.array([1, 2, 2], dtype=object),
            ValueError,
            "Object arrays",
        ),
    ],
)
def test_reader_rejects_malformed_schema_two_fields(
    tmp_path, key, value, error, message
):
    family = _build_family(np.array([[10, 20, 30]], dtype=np.uint8))
    path = tmp_path / "family.npz"
    save_sample_family(path, family)
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays[key] = value
    np.savez_compressed(path, **arrays)
    with pytest.raises(error, match=message):
        load_sample_family(path)


def test_schema_one_family_is_left_untouched(tmp_path) -> None:
    path = tmp_path / "family.npz"
    np.savez_compressed(path, schema_version=np.asarray(1, dtype=np.int64))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Schema 1.*regeneration.*new schema-2 root"):
        load_sample_family(path)
    assert path.read_bytes() == original


def test_failed_family_write_removes_only_its_temporary_file(tmp_path, monkeypatch):
    import lenahs.dataset.npz as codec

    family = _build_family(np.array([[10, 20]], dtype=np.uint8))
    path = tmp_path / "family.npz"

    def fail_write(file, **arrays):
        file.write(b"partial data")
        raise OSError("simulated write failure")

    monkeypatch.setattr(codec.np, "savez_compressed", fail_write)
    with pytest.raises(OSError, match="simulated write failure"):
        save_sample_family(path, family)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("kind", "clean_values", "observed_values", "parents", "actions", "missing"),
    [
        (
            ComponentTreeKind.MAX_TREE,
            [[10, 10, 20]],
            [[11, 12, 20]],
            [0, 0, 0],
            [1, 0, 2],
            [False, False],
        ),
        (
            ComponentTreeKind.MIN_TREE,
            [[10, 10, 20]],
            [[11, 12, 20]],
            [2, 0, 2],
            [2, 0, 1],
            [False, False],
        ),
        (
            ComponentTreeKind.MAX_TREE,
            [[10, 20, 30]],
            [[25, 25, 35]],
            [0, 0],
            [1, 2],
            [False, True, False],
        ),
        (
            ComponentTreeKind.MIN_TREE,
            [[10, 20, 30]],
            [[25, 25, 35]],
            [1, 1],
            [2, 1],
            [False, True, False],
        ),
    ],
)
def test_schema_two_keeps_clean_and_observed_diagnostic_scopes(
    tmp_path, kind, clean_values, observed_values, parents, actions, missing
):
    from lenahs.generation import derive_correction_targets

    clean = build_generation_sample(np.asarray(clean_values, dtype=np.uint8), kind)
    observed = build_generation_sample(
        np.asarray(observed_values, dtype=np.uint8), kind
    ).clean
    targets, diagnostics = derive_correction_targets(
        clean.clean.graph, clean.clean.component_tree, observed.graph
    )
    sample = GenerationSample(clean.clean, observed, targets, diagnostics)
    family = SampleFamily(
        sample_index=0,
        clean=clean,
        noisy=(
            NoisyObservation(0, AdditiveGaussianNoise(standard_deviation=5), sample),
        ),
    )
    path = tmp_path / "family.npz"
    save_sample_family(path, family)
    loaded = load_sample_family(path).noisy[0].sample
    np.testing.assert_array_equal(loaded.correction_targets.corrective_parents, parents)
    np.testing.assert_array_equal(loaded.correction_targets.actions, actions)
    np.testing.assert_array_equal(
        loaded.correction_diagnostics.missing_clean_zones, missing
    )
    observed_count = len(parents)
    assert loaded.correction_diagnostics.overlap_purity.shape == (observed_count,)
    assert loaded.correction_diagnostics.overlap_clean_zone_count.shape == (
        observed_count,
    )
    _assert_sample_equal(loaded, sample)
