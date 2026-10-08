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
    SaltAndPepperNoise,
    build_generation_sample,
)


def _build_reference_tree(image, kind):
    graph = build_reference_rag(image)
    return graph, build_reference_component_tree(graph, kind)


@pytest.mark.parametrize(
    ("kind", "parents", "actions"),
    [
        (ComponentTreeKind.MAX_TREE, [0, 2, 0], [1, 2, 2]),
        (ComponentTreeKind.MIN_TREE, [2, 1, 1], [2, 1, 2]),
    ],
)
def test_clean_observation_has_real_correction_targets(kind, parents, actions) -> None:
    image = np.array([[10, 10, 30], [10, 20, 30]], dtype=np.uint8)
    sample = build_generation_sample(image, kind, tree_builder=_build_reference_tree)

    assert sample.clean.image is image
    assert sample.observed is sample.clean
    assert sample.clean.component_tree.kind == kind
    np.testing.assert_array_equal(sample.correction_targets.corrective_parents, parents)
    np.testing.assert_array_equal(sample.correction_targets.actions, actions)
    np.testing.assert_array_equal(sample.correction_diagnostics.overlap_purity, 1.0)
    np.testing.assert_array_equal(
        sample.correction_diagnostics.overlap_clean_zone_count, 1
    )
    np.testing.assert_array_equal(
        sample.correction_diagnostics.missing_clean_zones, False
    )
    assert sample.correction_diagnostics.clean_root_missing is False
    assert sample.correction_diagnostics.artificial_root_attachment_count == 0


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
    assert sample.observed is not None
    assert sample.observed.image is not clean_image
    np.testing.assert_array_equal(sample.observed.image, 255)
    np.testing.assert_array_equal(clean_image, [[10, 10, 30], [10, 20, 30]])
    assert sample.clean.component_tree.kind == kind
    assert sample.observed.component_tree.kind == kind
    np.testing.assert_array_equal(
        sample.correction_targets.corrective_parents,
        [0],
    )
    np.testing.assert_allclose(sample.correction_diagnostics.overlap_purity, [0.5])
    np.testing.assert_array_equal(
        sample.correction_diagnostics.overlap_clean_zone_count,
        [3],
    )


def test_missing_clean_root_diagnostic_depends_on_requested_tree_kind() -> None:
    image = np.array([[10, 10, 30], [10, 20, 30]], dtype=np.uint8)
    samples = [
        build_generation_sample(
            image,
            kind,
            SaltAndPepperNoise(probability=1.0, salt_fraction=1.0),
            np.random.default_rng(4),
            tree_builder=_build_reference_tree,
        )
        for kind in ComponentTreeKind
    ]
    # The only observed zone selects level 10 by pixel count. It is the max-tree
    # root, whereas the min-tree root at level 30 has no representative.
    assert samples[0].correction_diagnostics.clean_root_missing is False
    assert samples[1].correction_diagnostics.clean_root_missing is True
    for sample in samples:
        np.testing.assert_array_equal(sample.correction_targets.corrective_parents, [0])
        np.testing.assert_array_equal(sample.correction_targets.actions, [1])
        np.testing.assert_array_equal(
            sample.correction_diagnostics.missing_clean_zones, [False, True, True]
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

    assert sample.observed is not None
    assert calls == [
        (sample.clean.image, ComponentTreeKind.MIN_TREE),
        (sample.observed.image, ComponentTreeKind.MIN_TREE),
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


@pytest.mark.parametrize("noisy", [False, True])
def test_sample_derives_correction_supervision_once(monkeypatch, noisy) -> None:
    image = np.array([[10, 10, 20]], dtype=np.uint8)
    derive = sample_module.derive_correction_targets
    calls = []

    def record_targets(clean_graph, clean_tree, observed_graph):
        calls.append((clean_graph, clean_tree, observed_graph))
        return derive(clean_graph, clean_tree, observed_graph)

    monkeypatch.setattr(sample_module, "derive_correction_targets", record_targets)
    sample = build_generation_sample(
        image,
        ComponentTreeKind.MAX_TREE,
        SaltAndPepperNoise(probability=0.5) if noisy else None,
        np.random.default_rng(10) if noisy else None,
    )
    assert len(calls) == 1
    assert calls[0][0] is sample.clean.graph
    assert calls[0][1] is sample.clean.component_tree
    assert calls[0][2] is sample.observed.graph
