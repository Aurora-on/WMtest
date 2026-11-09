"""Numerical contracts for predicted-coefficient DG and implicit root gradients.

No optimizer updates. References are analytic Cayley evolution, finite
differences of complete accepted steps, and an independent DOP853 ODE solve.
The frozen-coefficient method is intentionally not asserted equal to the older
velocity-dependent midpoint constitutive equation.
"""
from pathlib import Path
import os

import numpy as np
import pytest
import torch
from torch import nn

from dpbench.dissipative_core import SolverError
from dpbench.efficient_dissipative_core import EfficientDissipativeCore


DTYPE = torch.float64
DEVICE = os.environ.get("EFFICIENT_TEST_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "studies/passive_failure_20261104/replay_47690/failure_capture.pt"


@pytest.fixture(autouse=True)
def numerical_device():
    if DEVICE not in {"cpu", "cuda"}:
        pytest.fail("EFFICIENT_TEST_DEVICE must be cpu or cuda")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        pytest.fail("CUDA numerical checks requested without CUDA")
    with torch.device(DEVICE):
        yield


class PolynomialPotential(nn.Module):
    def __init__(self, quadratic=1., quartic=0.):
        super().__init__()
        self.quadratic = nn.Parameter(torch.tensor(quadratic, dtype=DTYPE))
        self.quartic = nn.Parameter(torch.tensor(quartic, dtype=DTYPE))

    def forward(self, q):
        return (.5*self.quadratic*q.square() + .25*self.quartic*q.pow(4)).sum(-1, keepdim=True)


class PolynomialCore(EfficientDissipativeCore):
    """Analytic test-only potential, via the documented gradient overrides."""
    def potential_gradient(self, q):
        return self.potential_net.quadratic*q + self.potential_net.quartic*q.pow(3)

    def discrete_gradient(self, q, q_next):
        k, b = self.potential_net.quadratic, self.potential_net.quartic
        return .5*k*(q+q_next) + .25*b*(q.pow(3)+q.square()*q_next+q*q_next.square()+q_next.pow(3))


class NonlinearDampingFactor(nn.Module):
    def forward(self, features):
        q, v = features[:, :1], features[:, 1:2]
        return (.25 + .15*q.square() + .1*v.square()).sqrt()


def polynomial_core(quartic=0., damping=.3, nonlinear_damping=False, **kwargs):
    options = dict(q_dim=1, damping_mode="position_velocity" if nonlinear_damping else "constant",
                   substeps=1, solver="picard", integrator="frozen_dg",
                   picard_iterations=6, newton_iterations=12, max_bisections=4)
    options.update(kwargs)
    core = PolynomialCore(**options)
    core.potential_net = PolynomialPotential(1., quartic)
    if nonlinear_damping:
        core.damping_net = NonlinearDampingFactor()
    else:
        with torch.no_grad():
            core.constant_factor.fill_(damping**.5)
    return core.to(DEVICE)


def captured(indices=(20,), **kwargs):
    assert CAPTURE.is_file(), "Real training failure capture is required"
    data = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    options = dict(data["core_config"], integrator="frozen_dg", solver="picard",
                   picard_iterations=6, newton_iterations=12, substeps=1, max_bisections=4)
    options.update(kwargs)
    core = EfficientDissipativeCore(**options).to(DEVICE)
    core.load_state_dict(data["core_state_dict"], strict=True)
    inputs = {key: None if value is None else value[list(indices)].to(DEVICE).clone()
              for key, value in data["input"].items()}
    return core, inputs


def call(core, x):
    return core(x["q"], x["p"], x["action"], x["h"], context=x.get("context"))


def objective(core, x):
    qn, pn, info = call(core, x)
    return (.2*qn.square()+.3*pn.square()).sum(), info


def assert_ledger(core, q, p, qn, pn, info):
    assert torch.isfinite(torch.cat((qn, pn), -1)).all()
    assert all(torch.isfinite(value).all() for value in info.values())
    assert torch.all(info["dissipation"] >= -1e-12)
    assert torch.all(info["solver_residual_ratio"] <= 1+1e-10)
    assert torch.all(info["solver_split_depth"] <= core.max_bisections)
    observed = core.energy(qn, pn)-core.energy(q, p)-info["work"]+info["dissipation"]
    scale = 1+core.energy(qn, pn).abs()+core.energy(q, p).abs()+info["dissipation"].abs()
    assert torch.all(abs(observed-info["energy_residual"]) <= 1e-10*scale)
    assert torch.all(abs(observed-info["residual_work"]-info["chain_defect"]) <= 1e-10*scale)


def test_real_failure_batch_passes_new_equation_and_original_acceptance_tolerances():
    core, x = captured(indices=tuple(range(32)))
    assert core.residual_atol == 1e-8 and core.residual_rtol == 1e-7 and core.chain_atol == 1e-10
    with torch.no_grad():
        qn, pn, info = call(core, x)
    assert_ledger(core, x["q"], x["p"], qn, pn, info)


@pytest.mark.parametrize("target_name", ["q", "p", "context", "h", "damping_weight", "potential_weight"])
def test_captured_implicit_gradient_matches_complete_step_finite_difference(target_name):
    # Tight forward acceptance separates finite-root error from the derivative
    # comparison. Finite differences recompute the entire midpoint predictor.
    core, x = captured(residual_atol=1e-11, residual_rtol=1e-11)
    for key in ("q", "p", "context", "h"):
        x[key].requires_grad_(True)
    targets = {**{key: x[key] for key in ("q", "p", "context", "h")},
               "damping_weight": core.damping_net[-1].weight,
               "potential_weight": core.potential_net[0][0].weight}
    target = targets[target_name]
    direction = torch.linspace(.2, .8, target.numel(), dtype=DTYPE, device=target.device).reshape(target.shape)
    direction = direction / direction.norm()
    value, info = objective(core, x)
    gradient = torch.autograd.grad(value, target)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 1e-10
    expected = (gradient*direction).sum()
    original = target.detach().clone()
    epsilon = 1e-6 if target_name == "h" else 2e-5
    values = []
    try:
        with torch.no_grad():
            for sign in (1, -1):
                target.copy_(original + sign*epsilon*direction)
                perturbed, diagnostics = objective(core, x)
                # Root-gradient comparisons must stay in the same subdivision.
                torch.testing.assert_close(diagnostics["solver_split_depth"], info["solver_split_depth"], rtol=0, atol=0)
                values.append(perturbed)
    finally:
        with torch.no_grad():
            target.copy_(original)
    observed = (values[0]-values[1])/(2*epsilon)
    torch.testing.assert_close(expected, observed, rtol=4e-4, atol=5e-6)


@pytest.mark.parametrize("damping", [0., .3, 20.])
def test_harmonic_step_matches_cayley_and_energy(damping):
    core = polynomial_core(damping=damping, residual_atol=1e-12, residual_rtol=1e-12)
    q = torch.tensor([[.8], [-1.1]], dtype=DTYPE)
    p = torch.tensor([[-.4], [.7]], dtype=DTYPE)
    h = torch.tensor([[.1], [.17]], dtype=DTYPE)
    velocity = (p-h*q/2)/(1+h*damping/2+h.square()/4)
    with torch.no_grad():
        qn, pn, info = core(q, p, torch.zeros_like(q), h)
    torch.testing.assert_close(qn, q+h*velocity, rtol=1e-10, atol=2e-12)
    torch.testing.assert_close(pn, 2*velocity-p, rtol=1e-10, atol=2e-12)
    assert info["solver_split_depth"].max() == 0
    torch.testing.assert_close(info["dissipation"], (h*damping*velocity.square()).squeeze(-1), rtol=1e-10, atol=2e-12)
    assert_ledger(core, q, p, qn, pn, info)


def test_harmonic_implicit_state_time_and_parameter_gradients_match_closed_form():
    core = polynomial_core(damping=.3, residual_atol=1e-12, residual_rtol=1e-12)
    q = torch.tensor([[.8], [-.9]], dtype=DTYPE, requires_grad=True)
    p = torch.tensor([[-.4], [.2]], dtype=DTYPE, requires_grad=True)
    h = torch.tensor([[.1], [.15]], dtype=DTYPE, requires_grad=True)
    k, factor = core.potential_net.quadratic, core.constant_factor
    qn, pn, _ = core(q, p, torch.zeros_like(q), h)
    actual = torch.autograd.grad((.2*qn.square()+.3*pn.square()).sum(), (q, p, h, k, factor))
    damping = factor.square().sum()
    v = (p-h*k*q/2)/(1+h*damping/2+h.square()*k/4)
    reference = (.2*(q+h*v).square()+.3*(2*v-p).square()).sum()
    expected = torch.autograd.grad(reference, (q, p, h, k, factor))
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=2e-8, atol=2e-10)


def test_nonlinear_position_velocity_damping_has_second_order_time_refinement():
    from scipy.integrate import solve_ivp
    core = polynomial_core(quartic=.2, nonlinear_damping=True,
                           residual_atol=1e-12, residual_rtol=1e-12)
    initial = np.array([.8, -.4])
    def rhs(_t, state):
        q, p = state
        return [p, -q-.2*q**3-(.25+.15*q*q+.1*p*p)*p]
    reference = solve_ivp(rhs, (0., .4), initial, method="DOP853", rtol=1e-12, atol=1e-14)
    assert reference.success
    errors = []
    with torch.no_grad():
        for dt in (.08, .04, .02, .01):
            q = torch.tensor([[initial[0]]], dtype=DTYPE)
            p = torch.tensor([[initial[1]]], dtype=DTYPE)
            for _ in range(round(.4/dt)):
                qn, pn, info = core(q, p, torch.zeros_like(q), dt)
                assert_ledger(core, q, p, qn, pn, info)
                q, p = qn, pn
            actual = torch.cat((q, p), -1).cpu().numpy().reshape(-1)
            errors.append(np.linalg.norm(actual-reference.y[:, -1]))
    orders = np.log2(np.array(errors[:-1])/np.array(errors[1:]))
    assert np.all(np.diff(errors) < 0), errors
    assert np.all(orders[-2:] > 1.8), dict(errors=errors, orders=orders.tolist())


def test_batch_composition_and_permutation_do_not_change_states_or_input_gradients():
    core, x = captured(indices=(0, 5, 20, 31), residual_atol=1e-11, residual_rtol=1e-11)
    x["q"].requires_grad_(True); x["p"].requires_grad_(True)
    qn, pn, _ = call(core, x)
    bulk_grad = torch.autograd.grad((qn+.37*pn).sum(), (x["q"], x["p"]))
    singles, gradients = [], []
    for index in range(4):
        local = {key: None if value is None else value[index:index+1].detach().clone() for key, value in x.items()}
        local["q"].requires_grad_(True); local["p"].requires_grad_(True)
        sq, sp, _ = call(core, local)
        singles.append((sq, sp))
        gradients.append(torch.autograd.grad((sq+.37*sp).sum(), (local["q"], local["p"])))
    for i, batch in enumerate((qn, pn)):
        torch.testing.assert_close(batch, torch.cat([z[i] for z in singles]), rtol=2e-9, atol=2e-10)
        torch.testing.assert_close(bulk_grad[i], torch.cat([z[i] for z in gradients]), rtol=2e-8, atol=2e-9)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        changed = {key: None if value is None else value[permutation] for key, value in x.items()}
        pq, pp, _ = call(core, changed)
    torch.testing.assert_close(pq, qn[permutation], rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(pp, pn[permutation], rtol=2e-9, atol=2e-10)


def test_nonfinite_input_is_rejected_before_an_accepted_step():
    core = polynomial_core()
    q = torch.tensor([[float("nan")]], dtype=DTYPE)
    with pytest.raises((ValueError, SolverError)):
        core(q, torch.zeros_like(q), torch.zeros_like(q), .1)


def test_equilibrium_and_small_intervals_have_finite_nonzero_gradients():
    core = polynomial_core(damping=.3)
    q = torch.tensor([[0.], [.4]], dtype=DTYPE, requires_grad=True)
    p = torch.tensor([[0.], [1e-9]], dtype=DTYPE, requires_grad=True)
    qn, pn, info = core(q, p, torch.zeros_like(q), 1e-8)
    assert qn[0].item() == pn[0].item() == 0.
    gradients = torch.autograd.grad((qn+pn).sum(), (q, p))
    assert all(torch.isfinite(g).all() and g.norm() > .1 for g in gradients)
    assert_ledger(core, q, p, qn, pn, info)


def test_actual_fd_newton_fallback_matches_harmonic_solution_and_gradient():
    core = polynomial_core(picard_iterations=1, newton_iterations=1, max_bisections=0,
                           residual_atol=1e-10, residual_rtol=1e-10)
    q = torch.tensor([[.8]], dtype=DTYPE, requires_grad=True)
    p = torch.tensor([[-.4]], dtype=DTYPE, requires_grad=True)
    h = torch.tensor([[.7]], dtype=DTYPE, requires_grad=True)
    qn, pn, info = core(q, p, torch.zeros_like(q), h)
    assert info["solver_newton_iterations"].min() >= 1
    assert info["solver_split_depth"].max() == 0
    assert info["solver_replay_iterations"].max() == 0
    got = torch.autograd.grad((qn.square()+pn.square()).sum(), (q, p, h))
    v = (p-h*q/2)/(1+.3*h/2+h.square()/4)
    want = torch.autograd.grad(((q+h*v).square()+(2*v-p).square()).sum(), (q, p, h))
    for actual, expected in zip(got, want):
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
    assert_ledger(core, q, p, qn, pn, info)


def test_real_bisection_leaf_gradients_match_explicit_substeps_and_parameter_fd():
    core = polynomial_core(quartic=1., picard_iterations=1, newton_iterations=1,
                           residual_atol=1e-10, residual_rtol=1e-10)
    q = torch.tensor([[1.2]], dtype=DTYPE, requires_grad=True)
    p = torch.tensor([[-.5]], dtype=DTYPE, requires_grad=True)
    h = torch.tensor([[.4]], dtype=DTYPE, requires_grad=True)
    qn, pn, info = core(q, p, torch.zeros_like(q), h)
    assert info["solver_split_depth"].item() == 2
    assert info["solver_leaf_steps"].item() == 4
    assert info["solver_failed_attempts"].item() == 3
    assert_ledger(core, q, p, qn, pn, info)
    objective_value = (qn.square()+pn.square()).sum()
    gradients = torch.autograd.grad(objective_value, (q, p, h, core.potential_net.quartic))
    reference = polynomial_core(quartic=1., substeps=4, picard_iterations=1,
                                newton_iterations=1, max_bisections=0,
                                residual_atol=1e-10, residual_rtol=1e-10)
    reference.load_state_dict(core.state_dict())
    rq, rp, rh = [x.detach().clone().requires_grad_(True) for x in (q, p, h)]
    rqn, rpn, ri = reference(rq, rp, torch.zeros_like(rq), rh)
    reference_gradients = torch.autograd.grad((rqn.square()+rpn.square()).sum(),
                                             (rq, rp, rh, reference.potential_net.quartic))
    torch.testing.assert_close(qn, rqn, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(pn, rpn, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(info["dissipation"], ri["dissipation"], rtol=2e-10, atol=2e-11)
    for actual, expected in zip(gradients, reference_gradients):
        torch.testing.assert_close(actual, expected, rtol=3e-8, atol=2e-9)
    values = []
    epsilon = 1e-5
    with torch.no_grad():
        try:
            for value in (1+epsilon, 1-epsilon):
                core.potential_net.quartic.fill_(value)
                aq, ap, ai = core(q, p, torch.zeros_like(q), h)
                assert ai["solver_split_depth"].item() == 2
                assert ai["solver_leaf_steps"].item() == 4
                values.append((aq.square()+ap.square()).sum())
        finally:
            core.potential_net.quartic.fill_(1.)
    torch.testing.assert_close(gradients[-1], (values[0]-values[1])/(2*epsilon), rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("negative_curvature", [-4., -4.+1e-10])
def test_valid_forward_equilibrium_with_singular_or_ill_conditioned_root_rejects_backward(negative_curvature):
    # F(v)=diag(1+k_j/4)v at h=1,q=p=0. F(0)=0 and its energy
    # ledger is valid, but the first root direction is singular/ill-conditioned.
    core = polynomial_core(q_dim=2, damping=0., max_bisections=0)
    core.potential_net.quadratic = nn.Parameter(torch.tensor([negative_curvature, 0.], dtype=DTYPE))
    q = torch.zeros(1, 2, dtype=DTYPE, requires_grad=True)
    p = torch.zeros_like(q, requires_grad=True)
    qn, pn, info = core(q, p, torch.zeros(1, 1, dtype=DTYPE), 1.)
    assert torch.equal(qn, q) and torch.equal(pn, p)
    assert info["solver_residual"].item() == 0.
    with pytest.raises(SolverError, match="Implicit backward rejected"):
        (qn+pn).sum().backward()
    assert q.grad is None and p.grad is None


def test_fd_newton_jacobian_matches_independent_state_autograd_on_capture():
    core, x = captured(indices=(0, 20, 31))
    with torch.no_grad():
        damping, drive, initial = core._predict_coefficients(x["q"], x["p"], x["action"], x["h"], x["context"])
        finite_difference = core._fd_jacobian(initial, x["q"], x["p"], x["h"], damping, drive)
    velocity = initial.detach().requires_grad_(True)
    residual, _ = core._frozen_residual(velocity, x["q"], x["p"], x["h"], damping, drive)
    exact = torch.stack([torch.autograd.grad(residual[:, j].sum(), velocity, retain_graph=True)[0]
                         for j in range(core.q_dim)], -2)
    torch.testing.assert_close(finite_difference, exact, rtol=2e-5, atol=2e-7)


def test_small_forward_residual_with_condition_one_but_large_root_error_rejects_backward():
    # In one dimension every nonzero Jacobian has condition number one. Near
    # cancellation J~1e-8 nevertheless amplifies F~1e-9 to a correction~0.1.
    # This is a real quartic potential, not a mocked condition estimator.
    core = polynomial_core(quartic=1., damping=.3, max_bisections=0)
    with torch.no_grad():
        core.potential_net.quadratic.fill_(-4.6+4e-8)
    q = torch.zeros(1, 1, dtype=DTYPE, requires_grad=True)
    p = torch.full_like(q, 1e-9, requires_grad=True)
    qn, pn, info = core(q, p, torch.zeros_like(q), 1.)
    assert_ledger(core, q, p, qn, pn, info)
    assert info["solver_residual"].item() < core.residual_atol
    with torch.no_grad():
        v = qn  # q0=0 and h=1.
        coefficient = 1+.3/2+core.potential_net.quadratic/4
        residual = coefficient*v + v.pow(3)/8-p
        jacobian = coefficient + 3*v.square()/8
        torch.testing.assert_close(abs(jacobian)*abs(1/jacobian), torch.ones_like(jacobian))
        assert abs(residual/jacobian).item() > .01
    with pytest.raises(SolverError, match="root correction"):
        (qn+pn).sum().backward()
    assert q.grad is None and p.grad is None
