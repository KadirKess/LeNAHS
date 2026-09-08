from collections.abc import Callable

import numpy as np
import pytest

import lenahs.generation.samples as sample_module
from lenahs.component_tree import (
    ComponentTreeKind,
    build_reference_component_tree,
    build_reference_rag,
)
from lenahs.generation import (
    NoiseDirection,
    SaltAndPepperNoise,
    build_generation_sample,
)


def _build_reference_tree(image, kind):
    graph = build_reference_rag(image)
    return graph, build_reference_component_tree(graph, kind)


def test_clean_sample_contains_requested_tree_and_unchanged_targets() -> None:
    clean_image = np.array([[10, 10, 30], [10, 20, 30]], dtype=np.uint8)

    sample = build_generation_sample(
        clean_image,
        ComponentTreeKind.MAX_TREE,
        tree_builder=_build_reference_tree,
    )

    assert sample.clean.image is clean_image
    assert sample.clean.component_tree.kind == ComponentTreeKind.MAX_TREE
    assert sample.corrupted is None
    np.testing.assert_array_equal(
        sample.corruption_targets.noise_direction,
        NoiseDirection.UNCHANGED,
    )
    np.testing.assert_array_equal(sample.corruption_targets.overlap_purity, 1.0)
    np.testing.assert_array_equal(
        sample.corruption_targets.overlap_clean_zone_count,
        1,
    )


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_corrupted_sample_keeps_clean_and_observed_representations(kind) -> None:
    clean_image = np.array([[10, 10, 30], [10, 20, 30]], dtype=np.uint8)

    sample = build_generation_sample(
        clean_image,
        kind,
        SaltAndPepperNoise(probability=1.0, salt_fraction=1.0),
        np.random.default_rng(3),
        tree_builder=_build_reference_tree,
    )

    assert sample.clean.image is clean_image
    assert sample.corrupted is not None
    assert sample.corrupted.image is not clean_image
    np.testing.assert_array_equal(sample.corrupted.image, 255)
    np.testing.assert_array_equal(clean_image, [[10, 10, 30], [10, 20, 30]])
    assert sample.clean.component_tree.kind == kind
    assert sample.corrupted.component_tree.kind == kind
    np.testing.assert_array_equal(
        sample.corruption_targets.noise_direction,
        [NoiseDirection.BRIGHTENED],
    )
    np.testing.assert_allclose(sample.corruption_targets.overlap_purity, [0.5])
    np.testing.assert_array_equal(
        sample.corruption_targets.overlap_clean_zone_count,
        [3],
    )


def test_direction_targets_do_not_depend_on_tree_kind() -> None:
    clean_image = np.array([[10, 10, 30], [10, 20, 30]], dtype=np.uint8)
    config = SaltAndPepperNoise(probability=1.0, salt_fraction=1.0)

    max_sample = build_generation_sample(
        clean_image,
        ComponentTreeKind.MAX_TREE,
        config,
        np.random.default_rng(4),
        tree_builder=_build_reference_tree,
    )
    min_sample = build_generation_sample(
        clean_image,
        ComponentTreeKind.MIN_TREE,
        config,
        np.random.default_rng(4),
        tree_builder=_build_reference_tree,
    )

    np.testing.assert_array_equal(
        max_sample.corruption_targets.noise_direction,
        min_sample.corruption_targets.noise_direction,
    )
    np.testing.assert_array_equal(
        max_sample.corruption_targets.overlap_purity,
        min_sample.corruption_targets.overlap_purity,
    )
    np.testing.assert_array_equal(
        max_sample.corruption_targets.overlap_clean_zone_count,
        min_sample.corruption_targets.overlap_clean_zone_count,
    )


def test_tree_builder_receives_each_image_with_only_the_requested_kind() -> None:
    clean_image = np.array([[10, 20]], dtype=np.uint8)
    calls: list[tuple[np.ndarray, ComponentTreeKind]] = []

    def recording_builder(image, kind):
        calls.append((image, kind))
        return _build_reference_tree(image, kind)

    sample = build_generation_sample(
        clean_image,
        ComponentTreeKind.MIN_TREE,
        SaltAndPepperNoise(probability=1.0, salt_fraction=1.0),
        np.random.default_rng(5),
        tree_builder=recording_builder,
    )

    assert sample.corrupted is not None
    assert calls == [
        (sample.clean.image, ComponentTreeKind.MIN_TREE),
        (sample.corrupted.image, ComponentTreeKind.MIN_TREE),
    ]


@pytest.mark.parametrize(
    ("noise_config", "noise_rng", "message"),
    [
        (None, np.random.default_rng(0), "noise_rng requires a noise_config"),
        (SaltAndPepperNoise(), None, "noise_config requires a noise_rng"),
    ],
)
def test_noise_configuration_and_generator_must_be_supplied_together(
    noise_config,
    noise_rng,
    message,
) -> None:
    clean_image = np.array([[10]], dtype=np.uint8)

    with pytest.raises(ValueError, match=message):
        build_generation_sample(
            clean_image,
            ComponentTreeKind.MAX_TREE,
            noise_config,
            noise_rng,
            tree_builder=_build_reference_tree,
        )


def test_invalid_noise_fails_before_component_tree_construction() -> None:
    clean_image = np.array([[10]], dtype=np.uint8)
    tree_builder: Callable = pytest.fail

    with pytest.raises(ValueError, match="probability must be between 0 and 1"):
        build_generation_sample(
            clean_image,
            ComponentTreeKind.MAX_TREE,
            SaltAndPepperNoise(probability=1.1),
            np.random.default_rng(0),
            tree_builder=tree_builder,
        )


@pytest.mark.parametrize(
    ("kind", "selected_name", "unselected_name"),
    [
        (
            ComponentTreeKind.MAX_TREE,
            "component_tree_max_tree",
            "component_tree_min_tree",
        ),
        (
            ComponentTreeKind.MIN_TREE,
            "component_tree_min_tree",
            "component_tree_max_tree",
        ),
    ],
)
def test_default_builder_dispatches_requested_higra_tree_and_vectorized_converter(
    monkeypatch,
    kind,
    selected_name,
    unselected_name,
) -> None:
    clean_image = np.array([[0, 2, 0], [0, 1, 0]], dtype=np.uint8)
    selected = getattr(sample_module.hg, selected_name)
    calls = {"selected": 0, "unselected": 0, "converter": 0}

    def record_selected(*args, **kwargs):
        calls["selected"] += 1
        return selected(*args, **kwargs)

    def record_unselected(*args, **kwargs):
        calls["unselected"] += 1
        raise AssertionError("The unrequested Higra tree builder was called.")

    converter = sample_module.higra_to_nahs_vectorized

    def record_converter(*args, **kwargs):
        calls["converter"] += 1
        return converter(*args, **kwargs)

    monkeypatch.setattr(sample_module.hg, selected_name, record_selected)
    monkeypatch.setattr(sample_module.hg, unselected_name, record_unselected)
    monkeypatch.setattr(
        sample_module,
        "higra_to_nahs_vectorized",
        record_converter,
    )

    build_generation_sample(clean_image, kind)

    assert calls == {"selected": 1, "unselected": 0, "converter": 1}


@pytest.mark.parametrize("kind", list(ComponentTreeKind))
def test_default_builder_matches_reference_component_tree(kind) -> None:
    clean_image = np.array(
        [[0, 2, 0], [0, 1, 0], [0, 0, 0]],
        dtype=np.uint8,
    )

    sample = build_generation_sample(clean_image, kind)
    reference_graph, reference_tree = _build_reference_tree(clean_image, kind)

    np.testing.assert_array_equal(
        sample.clean.graph.label_map, reference_graph.label_map
    )
    np.testing.assert_array_equal(
        sample.clean.graph.edge_index, reference_graph.edge_index
    )
    np.testing.assert_array_equal(sample.clean.graph.levels, reference_graph.levels)
    np.testing.assert_array_equal(
        sample.clean.component_tree.parents,
        reference_tree.parents,
    )
