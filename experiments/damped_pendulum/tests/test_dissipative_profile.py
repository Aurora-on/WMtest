"""Profile planning/failure accounting only; these tests take no optimizer steps."""
import copy
import json
from pathlib import Path

import pytest
import torch

from scripts import profile_dissipative as profile


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return profile.read_config(ROOT / "configs/dissipative_passive_family.json")


def test_curriculum_count_and_validation_cost(config):
    assert profile.curriculum_counts(config) == {4: 1000, 8: 3000, 16: 6000, 32: 10000}
    timings = {str(h): {"mean_seconds": i, "min_seconds": i / 2, "max_seconds": i * 2}
               for i, h in enumerate(profile.HORIZONS, 1)}
    result = profile.estimate_runtime(config, timings, 10.)
    assert result["validation_calls"] == 80
    assert result["estimated_total_seconds"] == 65800.
    assert result["observed_step_min_based_seconds"] == 33300.
    assert result["observed_step_max_based_seconds"] == 130800.


def test_partial_budget_counts_final_validation(config):
    config = copy.deepcopy(config)
    config["training"]["steps"] = 1123
    assert profile.curriculum_counts(config) == {4: 1000, 8: 123}
    result = profile.make_plan("unused.h5", config, "cpu", 2, 4)
    assert result["formal_validation_calls"] == 5
    assert result["total_profile_optimizer_updates"] == 24


def test_no_execute_does_not_read_data_or_construct_optimizer(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Planning must not construct training objects or read full data")
    monkeypatch.setattr(profile, "preflight", forbidden)
    monkeypatch.setattr(profile, "DissipativeWorldModel", forbidden)
    monkeypatch.setattr(profile.torch.optim, "AdamW", forbidden)
    output = tmp_path / "must_not_exist.json"
    result = profile.main(["--data", str(tmp_path / "does_not_exist.h5"),
                           "--config", str(ROOT / "configs/dissipative_passive_family.json"),
                           "--output", str(output), "--device", "cpu", "--threads", "2"])
    assert result["status"] == "plan_only" and result["threads"] == 2
    assert result["validation_windows"] == 288 and result["validation_horizons"] == [8, 32]
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["optimizer_updates_enabled"] is False


@pytest.mark.parametrize("flag", ["--warmup", "--steps", "--threads"])
def test_cli_rejects_zero_budget(flag):
    with pytest.raises(SystemExit):
        profile.main(["--data", "unused.h5", "--config", str(ROOT / "configs/dissipative_passive_family.json"),
                      "--output", "unused.json", flag, "0"])


def test_preflight_failure_is_recorded_without_training(config, tmp_path, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("fixture data mismatch")
    monkeypatch.setattr(profile, "preflight", reject)
    output = tmp_path / "failure.json"
    with pytest.raises(ValueError, match="fixture data mismatch"):
        profile.execute_profile("unused.h5", output, config, "cpu")
    result = json.loads(output.read_text())
    assert result["status"] == "failed" and result["phase"] == "preflight"
    assert result["optimizer_updates_completed"] == 0
    assert not result["formal_training_started"]
    with pytest.raises(FileExistsError):
        profile.execute_profile("unused.h5", output, config, "cpu")


@pytest.mark.parametrize('explicit_modules', [False, True])
def test_profiler_uses_trainer_parameter_groups_and_learning_rates_without_optimizer(config, monkeypatch,
                                                                                   explicit_modules):
    """Intercept AdamW construction; inspect real parameter identities only."""
    from dpbench.train_dissipative import build_optimizer_parameter_groups, configure_trainable_modules

    config = copy.deepcopy(config)
    config['model'].update(encoder_width=8, sigreg_projections=4)
    config['model'].setdefault('core', {})['hidden_dim'] = 4
    if explicit_modules:
        config['training']['trainable_modules'] = ['encoder', 'initializer', 'potential']
    torch.manual_seed(81)
    model = profile.DissipativeWorldModel(config['model'])
    before = {key: value.clone() for key, value in model.state_dict().items()}
    report = configure_trainable_modules(model, config['training'])
    expected = build_optimizer_parameter_groups(model, config['training'])
    calls = []
    sentinel = object()

    def inspect_only(groups, **kwargs):
        calls.append((groups, kwargs))
        return sentinel

    monkeypatch.setattr(profile.torch.optim, 'AdamW', inspect_only)
    assert profile._optimizer(model, config) is sentinel
    assert len(calls) == 1
    groups, kwargs = calls[0]
    assert kwargs == {'weight_decay': config['training']['weight_decay']}
    for actual, reference in zip(groups, expected, strict=True):
        assert {k: v for k, v in actual.items() if k != 'params'} == {
            k: v for k, v in reference.items() if k != 'params'}
        assert [id(p) for p in actual['params']] == [id(p) for p in reference['params']]
    grouped = {id(p) for group in groups for p in group['params']}
    assert grouped == {id(p) for p in model.parameters() if p.requires_grad}
    if explicit_modules:
        assert report['trainable_modules'] == ['encoder', 'initializer', 'potential']
        assert all(id(p) not in grouped and not p.requires_grad for p in model.core.input_net.parameters())
        assert all(id(p) not in grouped and not p.requires_grad for p in model.core.damping_net.parameters())
    else:
        assert report['mode'] == 'legacy_all'
        assert grouped == {id(p) for p in model.parameters()}
    assert all(torch.equal(before[k], value) for k, value in model.state_dict().items())
