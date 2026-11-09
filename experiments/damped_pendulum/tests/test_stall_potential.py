"""Force/energy/gradient regressions for the optional unsquashed potential.

All checks use synthetic states and fixed parameters; no optimizer or training.
"""
import pytest
import torch
from torch import nn

from dpbench.dissipative_core import DissipativeCore
from dpbench.efficient_dissipative_core import EfficientDissipativeCore


DTYPE = torch.float64


def make_core(kind, **overrides):
    options = dict(q_dim=2, hidden_dim=4, context_dim=2, substeps=3,
                   residual_atol=1e-12, residual_rtol=1e-12)
    options.update(overrides)
    torch.manual_seed(5105)
    return kind(**options)


def inputs():
    return dict(q=torch.tensor([[.3, -.2], [-.4, .7]], dtype=DTYPE),
                p=torch.tensor([[.4, -.1], [-.3, .2]], dtype=DTYPE),
                context=torch.tensor([[.2, -.3], [.1, .4]], dtype=DTYPE),
                action=torch.zeros(2, 1, dtype=DTYPE),
                dt=torch.tensor([[.1], [.13]], dtype=DTYPE))


@pytest.mark.parametrize("kind", [DissipativeCore, EfficientDissipativeCore])
@pytest.mark.parametrize("mode", ["", "identity", None, False, []])
def test_invalid_potential_mode_is_rejected(kind, mode):
    with pytest.raises(ValueError, match="potential_mode"):
        make_core(kind, potential_mode=mode)


@pytest.mark.parametrize("kind", [DissipativeCore, EfficientDissipativeCore])
def test_legacy_default_and_state_dict_keys_remain_compatible(kind):
    legacy_default = make_core(kind)
    explicit_legacy = make_core(kind, potential_mode="softplus")
    raw = make_core(kind, potential_mode="raw")
    assert legacy_default.potential_mode == "softplus"
    assert isinstance(legacy_default.potential_net[1], nn.Softplus)
    assert isinstance(raw.potential_net[1], nn.Identity)
    assert legacy_default.state_dict().keys() == raw.state_dict().keys()
    explicit_legacy.load_state_dict(legacy_default.state_dict(), strict=True)
    raw.load_state_dict(legacy_default.state_dict(), strict=True)
    x = inputs()
    with torch.no_grad():
        expected = torch.nn.functional.softplus(legacy_default.potential_net[0](x["q"])).squeeze(-1)
        torch.testing.assert_close(legacy_default.potential(x["q"]), expected, rtol=0, atol=0)
        first = legacy_default(**x)
        second = explicit_legacy(**x)
    for actual, reference in zip(first[:2], second[:2]):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for key in first[2]:
        torch.testing.assert_close(first[2][key], second[2][key], rtol=0, atol=0)


def test_negative_raw_energy_preserves_analytic_force_and_known_lower_bound():
    raw = make_core(EfficientDissipativeCore, potential_mode="raw")
    net = raw.potential_net[0]
    # V = 2*tanh(tanh(q0)) - 20: bounded below by -22, with a finite force
    # near q0=0 despite the arbitrarily negative choice of energy origin.
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net[0].weight[0, 0] = 1
        net[2].weight[0, 0] = 1
        net[4].weight[0, 0] = 2
        net[4].bias.fill_(-20)
    legacy = make_core(EfficientDissipativeCore)
    legacy.load_state_dict(raw.state_dict(), strict=True)
    q = inputs()["q"].requires_grad_()
    gradient = raw.potential_gradient(q)
    expected = torch.zeros_like(q)
    expected[:, 0] = 2 * (1 - q[:, 0].tanh().tanh().square()) * (1 - q[:, 0].tanh().square())
    torch.testing.assert_close(gradient, expected, rtol=1e-13, atol=1e-14)
    assert (raw.potential(q) < 0).all() and (gradient[:, 0] > 1).all()
    assert (legacy.potential_gradient(q).norm(dim=-1) < 1e-7).all()
    grid = torch.stack((torch.linspace(-100, 100, 101, dtype=DTYPE),
                        torch.linspace(50, -50, 101, dtype=DTYPE)), -1)
    assert (raw.potential(grid) >= -22).all()


@pytest.mark.parametrize("kind", [DissipativeCore, EfficientDissipativeCore])
def test_negative_raw_potential_has_valid_ledger_and_energy_shift_invariance(kind):
    core = make_core(kind, potential_mode="raw")
    with torch.no_grad():
        core.potential_net[0][-1].bias.fill_(-20)
    shifted = make_core(kind, potential_mode="raw")
    shifted.load_state_dict(core.state_dict())
    with torch.no_grad():
        shifted.potential_net[0][-1].bias.add_(25)
    x = inputs()
    with torch.no_grad():
        qn, pn, info = core(**x)
        sq, sp, shifted_info = shifted(**x)
        assert (core.energy(x["q"], x["p"]) < 0).all()
        balance = core.energy(qn, pn) - core.energy(x["q"], x["p"]) - info["work"] + info["dissipation"]
    assert (info["dissipation"] >= 0).all()
    assert info["solver_residual"].max() < 1e-10
    assert balance.abs().max() < 1e-10
    torch.testing.assert_close(balance, info["residual_work"] + info["chain_defect"], rtol=0, atol=1e-11)
    torch.testing.assert_close(qn, sq, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(pn, sp, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(info["dissipation"], shifted_info["dissipation"], rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("target_name", ["q", "p", "context", "dt", "damping_weight", "potential_weight"])
def test_raw_implicit_gradient_matches_complete_step_cpu_finite_difference(target_name):
    core = make_core(EfficientDissipativeCore, potential_mode="raw")
    with torch.no_grad():
        core.potential_net[0][-1].bias.fill_(-20)
    x = inputs()
    for key in ("q", "p", "context", "dt"):
        x[key].requires_grad_(True)
    targets = {**x, "damping_weight": core.damping_net[-1].weight,
               "potential_weight": core.potential_net[0][-1].weight}
    target = targets[target_name]
    direction = torch.linspace(.2, .8, target.numel(), dtype=DTYPE).reshape(target.shape)
    direction /= direction.norm()

    def objective():
        q, p, info = core(**x)
        return (.2 * q.square() + .3 * p.square()).sum(), info["solver_split_depth"]

    value, branch = objective()
    grad = torch.autograd.grad(value, target)[0]
    assert torch.isfinite(grad).all() and grad.norm() > 1e-9
    expected = (grad * direction).sum()
    original = target.detach().clone()
    values = []
    epsilon = 2e-5
    try:
        with torch.no_grad():
            for sign in (1, -1):
                target.copy_(original + sign * epsilon * direction)
                changed, current_branch = objective()
                torch.testing.assert_close(current_branch, branch, rtol=0, atol=0)
                values.append(changed)
    finally:
        with torch.no_grad():
            target.copy_(original)
    numerical = (values[0] - values[1]) / (2 * epsilon)
    torch.testing.assert_close(expected, numerical, rtol=2e-5, atol=2e-8)
