"""Algebra, finite-algorithm gradients and failure handling; no optimization."""
import pytest
import torch
from torch import nn

from dpbench.dissipative_core import DissipativeCore, SolverError


class HarmonicPotential(nn.Module):
    def __init__(self, stiffness):
        super().__init__()
        self.stiffness = nn.Parameter(torch.tensor(stiffness, dtype=torch.float64))

    def forward(self, q):
        return (0.5 * self.stiffness * q.square().sum(-1)).unsqueeze(-1)


class UnitPort(nn.Module):
    def forward(self, q):
        return q * 0 + 1


def harmonic(damping=0.3, **kwargs):
    core = DissipativeCore(q_dim=1, damping_mode="constant", **kwargs)
    core.potential_net = HarmonicPotential(1.7)
    core.input_net = UnitPort()
    with torch.no_grad():
        core.constant_factor.fill_(damping ** 0.5)
    return core


def test_harmonic_closed_step_and_energy():
    core = harmonic(substeps=1, residual_atol=1e-12, residual_rtol=1e-12)
    q = torch.tensor([[0.8], [-1.1]], dtype=torch.float64)
    p = torch.tensor([[-0.4], [0.7]], dtype=torch.float64)
    u = torch.tensor([[0.3], [-0.2]], dtype=torch.float64)
    h = torch.tensor([[0.1], [0.17]], dtype=torch.float64)
    v = (p-h*1.7*q/2+h*u/2)/(1+h*.3/2+h*h*1.7/4)
    q_next, p_next, diagnostics = core(q, p, u, h)
    torch.testing.assert_close(q_next, q+h*v, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(p_next, 2*v-p, rtol=1e-11, atol=1e-12)
    assert diagnostics["energy_residual"].abs().max() < 1e-11
    assert torch.all(diagnostics["dissipation"] >= 0)


def test_velocity_position_damping_is_psd_and_depends_on_both():
    torch.manual_seed(17)
    core = DissipativeCore(q_dim=2)
    q = torch.tensor([[0.4, -0.6]], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([[0.9, 0.2]], dtype=torch.float64, requires_grad=True)
    matrix = core.damping(q, v)
    assert torch.linalg.eigvalsh(matrix).min() > 0
    dq, dv = torch.autograd.grad(matrix.sum(), (q, v))
    assert dq.abs().sum() > 1e-8 and dv.abs().sum() > 1e-8
    force = (core.damping(q, v) @ v.unsqueeze(-1)).squeeze(-1)
    assert (force * v).sum() >= 0


def test_finite_newton_energy_residual_accounting():
    torch.manual_seed(12)
    core = DissipativeCore(newton_iterations=1, substeps=2, strict=False,
                           residual_atol=0, residual_rtol=0)
    q = torch.tensor([[1.0, -1.1], [0.5, .3]], dtype=torch.float64)
    p = torch.tensor([[1.2, -.9], [-.7, 1.6]], dtype=torch.float64)
    u = torch.tensor([[0.8], [-0.8]], dtype=torch.float64)
    _, _, info = core(q, p, u, .4)
    assert info["solver_residual"].max() > 1e-9
    torch.testing.assert_close(info["energy_residual"],
                               info["residual_work"] + info["chain_defect"],
                               rtol=1e-7, atol=3e-14)


def test_state_and_damping_gradients_match_finite_algorithm():
    torch.manual_seed(21)
    core = DissipativeCore(q_dim=1, hidden_dim=8, substeps=1, newton_iterations=3,
                           strict=False, residual_atol=0, residual_rtol=0)
    q = torch.tensor([[.7]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([[.4]], dtype=torch.float64, requires_grad=True)
    u = torch.tensor([[.3]], dtype=torch.float64)

    def objective(q, p):
        qn, pn, _ = core(q, p, u, .2)
        return (qn.square() + .3 * pn.square()).sum()

    loss = objective(q, p)
    parameter = core.damping_net[-1].bias
    gq, gp, gd = torch.autograd.grad(loss, (q, p, parameter))
    epsilon = 1e-5
    with torch.no_grad():
        fdq = (objective(q+epsilon, p)-objective(q-epsilon, p))/(2*epsilon)
        fdp = (objective(q, p+epsilon)-objective(q, p-epsilon))/(2*epsilon)
        saved = parameter.detach().clone()
        parameter.copy_(saved+epsilon)
        plus = objective(q, p)
        parameter.copy_(saved-epsilon)
        minus = objective(q, p)
        parameter.copy_(saved)
    torch.testing.assert_close(gq.squeeze(), fdq, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(gp.squeeze(), fdp, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(gd.squeeze(), (plus-minus)/(2*epsilon), rtol=2e-5, atol=1e-7)
    assert gd.abs().sum() > 1e-7


def test_solver_failure_is_not_silent():
    torch.manual_seed(12)
    core = DissipativeCore(newton_iterations=1, substeps=1,
                           residual_atol=1e-14, residual_rtol=0)
    q = torch.tensor([[1., -1.1]], dtype=torch.float64)
    p = torch.tensor([[1.2, -.9]], dtype=torch.float64)
    with pytest.raises(SolverError, match="failed tolerance"):
        core(q, p, torch.ones(1, 1, dtype=torch.float64), 1.)


def test_no_grad_evaluation_and_batch_independence():
    torch.manual_seed(17)
    core = DissipativeCore(substeps=2)
    q = torch.tensor([[.4, -.7], [1., .2]], dtype=torch.float64)
    p = torch.tensor([[.7, .1], [-.8, .3]], dtype=torch.float64)
    u = torch.tensor([[.3], [-.4]], dtype=torch.float64)
    with torch.no_grad():
        qn, pn, info = core(q, p, u, .1)
        singles = [core(q[i:i+1], p[i:i+1], u[i:i+1], .1) for i in range(2)]
        assert not torch.is_grad_enabled()
    assert not qn.requires_grad and not pn.requires_grad
    torch.testing.assert_close(qn, torch.cat([x[0] for x in singles]), atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(pn, torch.cat([x[1] for x in singles]), atol=1e-12, rtol=1e-11)
    assert info["energy_residual"].abs().max() < 1e-7


def test_tiny_displacement_has_finite_gradient_and_audited_defect():
    core = harmonic(damping=0., substeps=1)
    q = torch.tensor([[.4]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([[1e-9]], dtype=torch.float64, requires_grad=True)
    qn, pn, info = core(q, p, torch.zeros(1, 1, dtype=torch.float64), 1e-8)
    g = torch.autograd.grad(qn.sum()+pn.sum(), (q, p))
    assert all(torch.isfinite(x).all() for x in g)
    assert info["chain_defect_max"].max() < 1e-14


@pytest.mark.parametrize("mode", ["none", "constant", "position", "position_velocity"])
def test_modes_and_serializable_config(mode):
    import json
    core = DissipativeCore(damping_mode=mode)
    clone = DissipativeCore(**json.loads(json.dumps(core.config)))
    clone.load_state_dict(core.state_dict())
    q, v = torch.ones(3, 2, dtype=torch.float64), torch.full((3, 2), .4, dtype=torch.float64)
    torch.testing.assert_close(core.damping(q, v), clone.damping(q, v))
    if mode == "none":
        assert torch.count_nonzero(core.damping(q, v)) == 0
    if mode in {"constant", "position"}:
        torch.testing.assert_close(core.damping(q, v), core.damping(q, -v))


def test_float32_is_supported_with_explicit_tolerances():
    core = harmonic(substeps=1, residual_atol=1e-5, residual_rtol=1e-5,
                    chain_atol=1e-5).float()
    q, p = torch.tensor([[.8]]), torch.tensor([[.3]])
    qn, pn, info = core(q, p, torch.zeros(1, 1), .1)
    assert qn.dtype == pn.dtype == torch.float32
    assert info["energy_residual"].abs().max() < 1e-5


def test_rk4_matches_independent_integrated_state_work_dissipation():
    import numpy as np
    from scipy.integrate import solve_ivp

    q, p, u, duration = .8, -.4, .6, .7
    stiffness, damping = 1.7, .3

    def reference_rhs(_t, state):
        return [state[1], -stiffness*state[0]-damping*state[1]+u,
                u*state[1], damping*state[1]**2]

    reference = solve_ivp(reference_rhs, (0, duration), [q, p, 0, 0], method="DOP853",
                          rtol=1e-12, atol=1e-13).y[:, -1]
    core = harmonic(integrator="rk4", substeps=32)
    with torch.no_grad():
        qn, pn, info = core(torch.tensor([[q]], dtype=torch.float64),
                            torch.tensor([[p]], dtype=torch.float64),
                            torch.tensor([[u]], dtype=torch.float64), duration)
    actual = [qn.item(), pn.item(), info["work"].item(), info["dissipation"].item()]
    np.testing.assert_allclose(actual, reference, atol=1e-8, rtol=1e-7)
    assert info["dissipation"].min() >= 0
    assert info["solver_iterations"].sum() == info["solver_residual"].sum() == 0
    assert info["chain_defect"].sum() == info["residual_work"].sum() == 0
    torch.testing.assert_close(info["energy_residual"], info["numerical_balance_defect"],
                               atol=2e-14, rtol=1e-7)


def test_rk4_step_refinement_converges_without_claiming_exact_balance():
    import numpy as np
    from scipy.linalg import expm

    duration = .8
    q, p, u = .8, -.4, .6
    reference = expm(np.array([[0., 1., 0.], [-1.7, -.3, u], [0., 0., 0.]]) * duration)
    reference = reference @ np.array([q, p, 1.])
    errors, defects = [], []
    for substeps in [1, 2, 4]:
        # Zero implicit tolerances must not reject RK4 truncation error.
        core = harmonic(integrator="rk4", substeps=substeps, residual_atol=0,
                        residual_rtol=0, chain_atol=0, strict=True)
        with torch.no_grad():
            qn, pn, info = core(torch.tensor([[q]], dtype=torch.float64),
                                torch.tensor([[p]], dtype=torch.float64),
                                torch.tensor([[u]], dtype=torch.float64), duration)
        errors.append(np.linalg.norm(np.array([qn.item(), pn.item()])-reference[:2]))
        defects.append(abs(info["energy_residual"].item()))
    assert errors[0]/errors[1] > 12 and errors[1]/errors[2] > 12
    assert defects[0] > 1e-5 and defects[2] < defects[0]/20


def test_rk4_complete_force_parameter_gradient_and_shared_parameters():
    torch.manual_seed(41)
    dg = DissipativeCore(q_dim=1, hidden_dim=8, substeps=2)
    rk = DissipativeCore(**dict(dg.config, integrator="rk4"))
    rk.load_state_dict(dg.state_dict())
    q = torch.tensor([[.8]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([[-.4]], dtype=torch.float64, requires_grad=True)
    u = torch.tensor([[.6]], dtype=torch.float64)
    torch.testing.assert_close(dg.energy(q,p), rk.energy(q,p))
    torch.testing.assert_close(dg.damping(q,p), rk.damping(q,p))
    torch.testing.assert_close(dg.input_matrix(q), rk.input_matrix(q))
    qn, pn, _ = rk(q,p,u,.2)
    parameter = rk.damping_net[-1].bias
    gradient = torch.autograd.grad(qn.square().sum()+pn.square().sum(), parameter)[0]
    epsilon = 1e-5
    with torch.no_grad():
        saved = parameter.clone()
        parameter.copy_(saved+epsilon)
        qp, pp, _ = rk(q,p,u,.2)
        parameter.copy_(saved-epsilon)
        qm, pm, _ = rk(q,p,u,.2)
        parameter.copy_(saved)
        finite_difference = (qp.square().sum()+pp.square().sum()
                             -qm.square().sum()-pm.square().sum())/(2*epsilon)
    assert gradient.abs().sum() > 1e-7
    torch.testing.assert_close(gradient.squeeze(), finite_difference, rtol=1e-5, atol=1e-8)
def test_nan_residual_is_not_mistaken_for_convergence(monkeypatch):
    core = DissipativeCore(q_dim=1, hidden_dim=4)
    q = torch.zeros(1, 1, dtype=torch.float64)

    def invalid_residual(velocity, *args):
        zero = torch.zeros_like(velocity)
        return torch.full_like(velocity, float("nan")), zero, zero, zero

    monkeypatch.setattr(core, "_residual", invalid_residual)
    with pytest.raises(SolverError, match="Non-finite Newton residual"):
        core(q, q, q, .1)


def test_finite_state_with_overflowing_energy_is_rejected():
    core = DissipativeCore(q_dim=1, hidden_dim=4)
    q = torch.zeros(1, 1, dtype=torch.float64)
    with pytest.raises(SolverError, match="Non-finite initial energy"):
        core(q, torch.full_like(q, 1e308), q, .1)


@pytest.mark.parametrize("integrator", ["discrete_gradient", "rk4"])
def test_context_gradients_follow_the_executed_solver(integrator):
    torch.manual_seed(173)
    core = DissipativeCore(q_dim=1, hidden_dim=8, context_dim=2,
                           integrator=integrator, substeps=2)
    q = torch.tensor([[.7]], dtype=torch.float64)
    p = torch.tensor([[.4]], dtype=torch.float64)
    u = torch.zeros_like(q)
    context = torch.tensor([[.3, -.5]], dtype=torch.float64, requires_grad=True)

    def objective(c):
        qn, pn, _ = core(q, p, u, .2, context=c)
        return (qn.square() + .3 * pn.square()).sum()

    grad = torch.autograd.grad(objective(context), context)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 1e-8
    epsilon = 1e-5
    with torch.no_grad():
        differences = []
        for i in range(2):
            offset = torch.zeros_like(context)
            offset[0, i] = epsilon
            differences.append((objective(context + offset) - objective(context - offset)) / (2 * epsilon))
    torch.testing.assert_close(grad[0], torch.stack(differences), atol=1e-8, rtol=2e-4)
    matrix = core.damping(q, p, context)
    assert not torch.equal(matrix, core.damping(q, p, -context))
    assert torch.linalg.eigvalsh(matrix).min() >= 0


def test_constant_context_harmonic_trajectory_and_energy_ledger():
    core = DissipativeCore(q_dim=1, context_dim=2, damping_mode="constant",
                           substeps=1, residual_atol=1e-12, residual_rtol=1e-12)
    core.potential_net = HarmonicPotential(1.7)
    core.input_net = UnitPort()
    core.damping_net = nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        core.damping_net.weight.copy_(torch.tensor([[.5, .2]], dtype=torch.float64))
    context = torch.tensor([[.3, -.5], [1.2, .4]], dtype=torch.float64)
    frozen = context.clone()
    damping = (.5 * context[:, :1] + .2 * context[:, 1:]).square()
    q = torch.full((2, 1), .8, dtype=torch.float64)
    p = torch.full((2, 1), -.4, dtype=torch.float64)
    initial_energy = core.energy(q, p).detach()
    dissipated = torch.zeros(2, dtype=torch.float64)
    h = .1
    for _ in range(3):
        expected_v = (p - h * 1.7 * q / 2) / (1 + h * damping / 2 + h * h * 1.7 / 4)
        qn, pn, info = core(q, p, torch.zeros_like(q), h, context=context)
        torch.testing.assert_close(qn, q + h * expected_v, atol=1e-12, rtol=1e-11)
        torch.testing.assert_close(pn, 2 * expected_v - p, atol=1e-12, rtol=1e-11)
        dissipated += info["dissipation"]
        q, p = qn, pn
    torch.testing.assert_close(core.energy(q, p).detach() - initial_energy + dissipated,
                               torch.zeros_like(dissipated), atol=1e-11, rtol=0)
    assert torch.equal(context, frozen)


@pytest.mark.parametrize("mode", ["none", "constant", "position", "position_velocity"])
def test_context_modes_validate_inputs_and_restore_config(mode):
    core = DissipativeCore(q_dim=1, hidden_dim=8, context_dim=2, damping_mode=mode)
    clone = DissipativeCore(**core.config)
    clone.load_state_dict(core.state_dict())
    q = torch.ones(2, 1, dtype=torch.float64)
    context = torch.tensor([[.2, .3], [-.4, .5]], dtype=torch.float64)
    torch.testing.assert_close(core.damping(q, q, context), clone.damping(q, q, context))
    for invalid in (None, torch.zeros(2, 1), torch.full((2, 2), float("nan"))):
        with pytest.raises(ValueError, match="context"):
            core(q, q, torch.zeros_like(q), .1, context=invalid)
    if mode == "constant":
        torch.testing.assert_close(core.damping(q, q, context), core.damping(2 * q, -q, context))
    if mode == "none":
        assert torch.count_nonzero(core.damping(q, q, context)) == 0


def test_context_disabled_rejects_silently_ignored_context():
    core = DissipativeCore(q_dim=1)
    q = torch.zeros(1, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="context_dim=0"):
        core(q, q, q, .1, context=torch.ones(1, 2))

