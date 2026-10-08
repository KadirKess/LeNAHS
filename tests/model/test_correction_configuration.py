import json

import pytest
import torch

from lenahs.model import (
    CorrectionModelConfig,
    CorrectionRunConfig,
    build_correction_model,
)


def _inputs():
    features = torch.tensor(
        [[0.1, 0.4, 0.0], [0.8, 0.2, 0.4], [0.3, 0.4, 0.7], [0.5, 1.0, 0.0]],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 1, 2, 0, 1, 2, 3], [1, 0, 2, 1, 0, 1, 2, 3]])
    return features, edges, torch.tensor([0, 3, 4])


@pytest.mark.parametrize(
    "recurrent,observed_steps,correction_steps",
    [(False, 0, 0), (False, 4, 0), (True, 0, 3), (True, 4, 3)],
)
def test_full_run_settings_round_trip(recurrent, observed_steps, correction_steps):
    run = CorrectionRunConfig(
        CorrectionModelConfig(3, recurrent), observed_steps, correction_steps
    )
    restored = CorrectionRunConfig.from_dict(json.loads(json.dumps(run.to_dict())))
    assert restored == run
    model = build_correction_model(run.model).double()
    rebuilt = build_correction_model(restored.model).double()
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    features, edges, ptr = _inputs()
    expected = model(features, edges, ptr, run.observed_steps, run.correction_steps)
    actual = rebuilt(
        features, edges, ptr, restored.observed_steps, restored.correction_steps
    )
    for a, b in zip(expected.correction_steps, actual.correction_steps, strict=True):
        torch.testing.assert_close(a.hidden_state, b.hidden_state, rtol=0, atol=0)
        torch.testing.assert_close(a.action_logits, b.action_logits, rtol=0, atol=0)


@pytest.mark.parametrize(
    "recurrent,observed_steps,correction_steps,error",
    [
        (False, -1, 0, ValueError),
        (False, 1.5, 0, TypeError),
        (False, True, 0, TypeError),
        (False, 1, -1, ValueError),
        (False, 1, True, TypeError),
        (False, 1, 0.5, TypeError),
        (False, 1, 1, ValueError),
        (True, 1, 0, ValueError),
    ],
)
def test_invalid_horizons_fail_before_encoding_and_in_saved_settings(
    recurrent, observed_steps, correction_steps, error
):
    model = build_correction_model(CorrectionModelConfig(3, recurrent)).double()

    def forbidden(*args):
        pytest.fail("Invalid horizons must fail before encoding.")

    handle = model.observed_model.encoder.register_forward_pre_hook(forbidden)
    features, edges, ptr = _inputs()
    with pytest.raises(error):
        model(features, edges, ptr, observed_steps, correction_steps)
    handle.remove()
    run = CorrectionRunConfig(
        CorrectionModelConfig(3, recurrent), observed_steps, correction_steps
    )
    with pytest.raises(error):
        run.to_dict()
    settings = {
        "model": run.model.to_dict(),
        "observed_steps": observed_steps,
        "correction_steps": correction_steps,
    }
    with pytest.raises(error):
        CorrectionRunConfig.from_dict(settings)


@pytest.mark.parametrize("flag", [0, 1, "true", None])
def test_recurrence_flag_requires_a_boolean(flag):
    config = CorrectionModelConfig(3, flag)
    with pytest.raises(TypeError, match="Boolean"):
        build_correction_model(config)
    with pytest.raises(TypeError, match="Boolean"):
        config.to_dict()
    settings = CorrectionModelConfig().to_dict()
    settings["correction_recurrent"] = flag
    with pytest.raises(TypeError, match="Boolean"):
        CorrectionModelConfig.from_dict(settings)
