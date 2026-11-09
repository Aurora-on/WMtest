"""Independent numerical acceptance tests for safeguarded dissipative steps.

These checks use actual nonlinear equations and the captured failing training
sample. They do not optimize parameters, relax solver acceptance, or claim global
root uniqueness. Finite differences are compared within a stable executed branch.
"""
from pathlib import Path
import os

import pytest
import torch
from torch import nn

from dpbench.dissipative_core import DissipativeCore, SolverError

DTYPE = torch.float64
DEVICE = os.environ.get("ROBUST_TEST_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "studies/passive_failure_20261104/replay_47690/failure_capture.pt"
SIGNATURE = ("solver_backtracks", "solver_leaf_steps", "solver_split_depth", "solver_replay_iterations")


@pytest.fixture(autouse=True)
def numerical_device():
    """Run identical equations on the requested device, without leaking defaults."""
    if DEVICE not in {"cpu", "cuda"}:
        pytest.fail("ROBUST_TEST_DEVICE must be cpu or cuda")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        pytest.fail("CUDA numerical acceptance requested but CUDA is unavailable")
    with torch.device(DEVICE):
        yield


class PolynomialPotential(nn.Module):
    def __init__(self, quadratic=1., quartic=0.):
        super().__init__()
        self.quadratic = nn.Parameter(torch.tensor(quadratic, dtype=DTYPE))
        self.quartic = nn.Parameter(torch.tensor(quartic, dtype=DTYPE))

    def forward(self, q):
        return (.5*self.quadratic*q.square() + .25*self.quartic*q.pow(4)).sum(-1, keepdim=True)


class DoubleWell(nn.Module):
    def forward(self, q):
        return (q.square()-1).square().sum(-1, keepdim=True)


class UnitPort(nn.Module):
    def forward(self, q):
        return q*0+1


class NonfinitePotential(nn.Module):
    def forward(self, q):
        return (q*float("nan")).sum(-1, keepdim=True)


def configured_core(quartic=0., damping=.3, **kwargs):
    options = dict(q_dim=1, damping_mode="constant", substeps=1,
                   solver="backtracking", newton_iterations=12, max_bisections=4)
    options.update(kwargs)
    core = DissipativeCore(**options)
    core.potential_net = PolynomialPotential(1., quartic)
    core.input_net = UnitPort()
    with torch.no_grad():
        core.constant_factor.fill_(damping**.5)
    return core


def captured(solver="backtracking", budget=12, indices=(20,), max_bisections=0):
    assert CAPTURE.is_file(), "Real training-failure capture is required for this regression"
    data = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    options = dict(data["core_config"], solver=solver, newton_iterations=budget,
                   substeps=1, max_bisections=max_bisections)
    core = DissipativeCore(**options)
    core.load_state_dict(data["core_state_dict"])
    inputs = {k: v[list(indices)].to(DEVICE).clone() if v is not None else None for k,v in data["input"].items()}
    return core, inputs


def signature(info):
    return tuple(tuple(info[k].reshape(-1).tolist()) for k in SIGNATURE)


def objective(core, q, p, action, dt, context=None):
    qn,pn,info=core(q,p,action,dt,context=context)
    return (.2*qn.square()+.3*pn.square()).sum(),info


def ledger(core,q,p,qn,pn,info):
    assert torch.isfinite(torch.cat((qn,pn),-1)).all()
    assert all(torch.isfinite(v).all() for v in info.values())
    assert (info["dissipation"] >= -1e-13).all()
    actual = core.energy(qn,pn)-core.energy(q,p)-info["work"]+info["dissipation"]
    torch.testing.assert_close(actual,info["energy_residual"],rtol=1e-7,atol=2e-12)
    torch.testing.assert_close(actual,info["residual_work"]+info["chain_defect"],rtol=2e-6,atol=2e-12)


def test_captured_failure_is_fixed_without_relaxing_tolerances():
    legacy,x=captured("full",6)
    with pytest.raises(SolverError):
        legacy(x["q"],x["p"],x["action"],x["h"],context=x["context"])
    robust,x=captured()
    assert robust.residual_atol==legacy.residual_atol
    assert robust.residual_rtol==legacy.residual_rtol
    assert robust.chain_atol==legacy.chain_atol
    with torch.no_grad():
        qn,pn,info=robust(x["q"],x["p"],x["action"],x["h"],context=x["context"])
    assert info["solver_backtracks"].item() >= 4
    assert info["solver_split_depth"].item()==0
    assert info["solver_replay_iterations"].item() <= 6
    tolerance=robust.residual_atol+robust.residual_rtol*torch.maximum(x["p"].abs().amax(-1),pn.abs().amax(-1))
    assert (info["solver_residual"] <= tolerance).all()
    ledger(robust,x["q"],x["p"],qn,pn,info)


def test_actual_backtracking_state_context_and_parameter_directional_gradients():
    core,x=captured()
    q=x["q"].requires_grad_(True);p=x["p"].requires_grad_(True);c=x["context"].requires_grad_(True)
    parameter=core.damping_net[-1].bias
    targets=(q,p,c,parameter)
    directions=tuple(torch.linspace(.2,.8,t.numel(),dtype=DTYPE).reshape(t.shape) for t in targets)
    loss,info=objective(core,q,p,x["action"],x["h"],c)
    assert info["solver_backtracks"].item()>0
    gradients=torch.autograd.grad(loss,targets)
    eps=2e-6
    for index,(target,direction,gradient) in enumerate(zip(targets,directions,gradients)):
        assert torch.isfinite(gradient).all() and gradient.norm()>1e-8
        outputs=[]
        saved=target.detach().clone()
        with torch.no_grad():
            for sign in (1,-1):
                if index==3:
                    target.copy_(saved+sign*eps*direction)
                    value,di=objective(core,q,p,x["action"],x["h"],c)
                else:
                    states=[q,p,c]
                    states[index]=states[index]+sign*eps*direction
                    value,di=objective(core,states[0],states[1],x["action"],x["h"],states[2])
                outputs.append(value)
                assert signature(di)==signature(info), "FD crossed a solver branch"
            if index==3:target.copy_(saved)
        finite_difference=(outputs[0]-outputs[1])/(2*eps)
        torch.testing.assert_close((gradient*direction).sum(),finite_difference,rtol=3e-4,atol=3e-6)


def test_nonlinear_bisection_preserves_gradient_and_accepted_leaf_energy_accounting():
    core=configured_core(quartic=1.,newton_iterations=2,max_bisections=4)
    q=torch.tensor([[1.2]],dtype=DTYPE,requires_grad=True)
    p=torch.tensor([[-.5]],dtype=DTYPE,requires_grad=True)
    u=torch.tensor([[.3]],dtype=DTYPE)
    qn,pn,info=core(q,p,u,.8)
    assert info["solver_split_depth"].item()>0
    assert info["solver_failed_attempts"].item()>0
    assert info["solver_leaf_steps"].item()>1
    ledger(core,q,p,qn,pn,info)
    targets=(q,p,core.potential_net.quartic)
    gradients=torch.autograd.grad((.2*qn.square()+.3*pn.square()).sum(),targets)
    eps=2e-6
    for i,(target,grad) in enumerate(zip(targets,gradients)):
        saved=target.detach().clone();values=[]
        with torch.no_grad():
            for sign in (1,-1):
                if i==2:
                    target.copy_(saved+sign*eps)
                    val,di=objective(core,q,p,u,.8)
                else:
                    val,di=objective(core,q+sign*eps if i==0 else q,p+sign*eps if i==1 else p,u,.8)
                assert signature(di)==signature(info), "FD crossed a subdivision branch"
                values.append(val)
            if i==2:target.copy_(saved)
        assert torch.isfinite(grad).all()
        torch.testing.assert_close(grad.squeeze(),(values[0]-values[1])/(2*eps),rtol=3e-4,atol=2e-6)


def test_adaptive_batch_order_and_composition_do_not_change_states_or_gradients():
    core=configured_core(quartic=1.,newton_iterations=2,max_bisections=4)
    q=torch.tensor([[1.2],[.1],[-.8]],dtype=DTYPE,requires_grad=True)
    p=torch.tensor([[-.5],[.05],[.3]],dtype=DTYPE,requires_grad=True)
    u=torch.tensor([[.3],[0.],[-.1]],dtype=DTYPE)
    qn,pn,info=core(q,p,u,.8)
    grads=torch.autograd.grad((qn.square()+pn.square()).sum(),(q,p))
    assert len(torch.unique(info["solver_split_depth"])) > 1
    singles=[];single_grads=[]
    for i in range(3):
        qi=q[i:i+1].detach().requires_grad_(True);pi=p[i:i+1].detach().requires_grad_(True)
        result=core(qi,pi,u[i:i+1],.8);singles.append(result)
        single_grads.append(torch.autograd.grad((result[0].square()+result[1].square()).sum(),(qi,pi)))
    for index,actual in enumerate((qn,pn)):
        torch.testing.assert_close(actual,torch.cat([r[index] for r in singles]),atol=2e-12,rtol=1e-10)
        torch.testing.assert_close(grads[index],torch.cat([r[index] for r in single_grads]),atol=2e-11,rtol=1e-9)
    for name in SIGNATURE:
        torch.testing.assert_close(info[name],torch.cat([r[2][name] for r in singles]),atol=0,rtol=0)
    order=torch.tensor([2,0,1])
    with torch.no_grad():
        reordered=core(q[order],p[order],u[order],.8)
    for index,actual in enumerate((qn,pn)):
        torch.testing.assert_close(actual[order],reordered[index],atol=2e-12,rtol=1e-10)


def test_exhausted_bisection_budget_is_a_visible_failure():
    core=configured_core(quartic=1.,newton_iterations=1,max_bisections=0)
    q=torch.tensor([[1.2]],dtype=DTYPE);p=torch.tensor([[-.5]],dtype=DTYPE)
    with pytest.raises(SolverError):core(q,p,torch.zeros_like(q),.8)


def test_singular_initial_jacobian_is_recovered_only_by_valid_substeps():
    core=configured_core(damping=0.,max_bisections=0)
    core.potential_net=DoubleWell()
    q=torch.zeros(1,1,dtype=DTYPE);u=torch.ones_like(q)
    # V''(0)=-4 and h=1 make J=1+h^2 V''/4 exactly zero, while u=1 gives F!=0.
    with pytest.raises(SolverError):core(q,q,u,1.)
    refined=configured_core(damping=0.,max_bisections=4)
    refined.potential_net=DoubleWell()
    with torch.no_grad():qn,pn,info=refined(q,q,u,1.)
    assert info["solver_split_depth"].item() > 0
    ledger(refined,q,q,qn,pn,info)


def test_nonfinite_equations_cannot_be_accepted_by_backtracking_or_bisection():
    core=configured_core(max_bisections=2)
    core.potential_net=NonfinitePotential()
    q=torch.ones(1,1,dtype=DTYPE)
    with pytest.raises(SolverError):core(q,q,torch.zeros_like(q),.1)


@pytest.mark.parametrize("damping",[0.,.3])
def test_harmonic_forced_step_matches_closed_solution_and_energy(damping):
    core=configured_core(damping=damping,residual_atol=1e-12,residual_rtol=1e-12)
    q=torch.tensor([[.8],[-1.1]],dtype=DTYPE);p=torch.tensor([[-.4],[.7]],dtype=DTYPE)
    u=torch.tensor([[.3],[-.2]],dtype=DTYPE);h=torch.tensor([[.1],[.17]],dtype=DTYPE)
    v=(p-h*q/2+h*u/2)/(1+h*damping/2+h*h/4)
    with torch.no_grad():qn,pn,info=core(q,p,u,h)
    torch.testing.assert_close(qn,q+h*v,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(pn,2*v-p,rtol=1e-11,atol=1e-12)
    assert info["solver_split_depth"].max()==0
    ledger(core,q,p,qn,pn,info)


def test_equilibrium_tiny_displacement_and_conservative_rollout():
    core=configured_core(damping=0.)
    q=torch.tensor([[0.],[.4]],dtype=DTYPE,requires_grad=True)
    p=torch.tensor([[0.],[1e-9]],dtype=DTYPE,requires_grad=True)
    qn,pn,info=core(q,p,torch.zeros_like(q),1e-8)
    assert qn[0].item()==pn[0].item()==0.
    grads=torch.autograd.grad(qn.sum()+pn.sum(),(q,p))
    assert all(torch.isfinite(g).all() for g in grads)
    ledger(core,q,p,qn,pn,info)
    q=torch.tensor([[.8]],dtype=DTYPE);p=torch.tensor([[-.4]],dtype=DTYPE)
    initial=core.energy(q,p).detach()
    with torch.no_grad():
        for _ in range(200):q,p,_=core(q,p,torch.zeros_like(q),.1)
    torch.testing.assert_close(core.energy(q,p),initial,atol=1e-10,rtol=1e-10)


class ExponentialPotential(nn.Module):
    def forward(self,q):
        return torch.exp(-q).sum(-1,keepdim=True)


class BellDampingFactor(nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude=nn.Parameter(torch.tensor(54.492820630990536,dtype=DTYPE))

    def forward(self,features):
        return self.amplitude.sqrt()*torch.exp(-.5*features[:,1:2].square())


def test_nonfinite_rejected_trials_do_not_poison_accepted_path_gradients():
    # At q=0,p=1,h=.1 this smooth positive potential / PSD damping combination
    # has initial J ~1e-12. The first Newton candidate goes to v~-9.5e11,
    # overflowing exp(-q_next). Half steps are well behaved, so planning must
    # discard those invalid graphs and differentiate only the accepted leaves.
    core=DissipativeCore(q_dim=1,substeps=1,solver="backtracking",
                         newton_iterations=12,max_bisections=3)
    core.potential_net=ExponentialPotential();core.damping_net=BellDampingFactor()
    q=torch.zeros(1,1,dtype=DTYPE,requires_grad=True)
    p=torch.ones(1,1,dtype=DTYPE,requires_grad=True)
    u=torch.zeros_like(q)
    qn,pn,info=core(q,p,u,.1)
    assert info["solver_failed_attempts"].item()>0
    assert info["solver_split_depth"].item()>0
    grads=torch.autograd.grad(qn.square().sum()+pn.square().sum(),
                              (q,p,core.damping_net.amplitude))
    assert all(torch.isfinite(g).all() for g in grads)
    ledger(core,q,p,qn,pn,info)
    reference=DissipativeCore(q_dim=1,substeps=2,solver="backtracking",
                              newton_iterations=12,max_bisections=0)
    reference.potential_net=ExponentialPotential();reference.damping_net=BellDampingFactor()
    reference.load_state_dict(core.state_dict())
    with torch.no_grad():rq,rp,ri=reference(q,p,u,.1)
    torch.testing.assert_close(qn,rq,rtol=1e-10,atol=1e-12)
    torch.testing.assert_close(pn,rp,rtol=1e-10,atol=1e-12)
    torch.testing.assert_close(info["dissipation"],ri["dissipation"],rtol=1e-10,atol=1e-12)


def test_backtracking_action_and_time_interval_gradients_match_executed_map():
    core,x=captured()
    u=x["action"].requires_grad_(True);h=x["h"].requires_grad_(True)
    loss,info=objective(core,x["q"],x["p"],u,h,x["context"])
    gradients=torch.autograd.grad(loss,(u,h))
    eps=1e-6
    for index,grad in enumerate(gradients):
        with torch.no_grad():
            plus,pi=objective(core,x["q"],x["p"],u+eps if index==0 else u,
                               h+eps if index==1 else h,x["context"])
            minus,mi=objective(core,x["q"],x["p"],u-eps if index==0 else u,
                                h-eps if index==1 else h,x["context"])
        assert signature(pi)==signature(mi)==signature(info)
        assert torch.isfinite(grad).all()
        torch.testing.assert_close(grad.squeeze(),(plus-minus)/(2*eps),rtol=3e-4,atol=3e-6)


def test_full_captured_training_batch_matches_isolated_difficult_member():
    batch,x=captured(indices=tuple(range(32)))
    with torch.no_grad():
        qn,pn,info=batch(x["q"],x["p"],x["action"],x["h"],context=x["context"])
        sq,sp,si=batch(x["q"][20:21],x["p"][20:21],x["action"][20:21],
                      x["h"][20:21],context=x["context"][20:21])
    torch.testing.assert_close(qn[20:21],sq,rtol=1e-10,atol=1e-12)
    torch.testing.assert_close(pn[20:21],sp,rtol=1e-10,atol=1e-12)
    assert info["solver_residual_ratio"].max() <= 1.
    for name in SIGNATURE:
        torch.testing.assert_close(info[name][20:21],si[name],atol=0,rtol=0)
    ledger(batch,x["q"],x["p"],qn,pn,info)


@pytest.mark.parametrize('solver', ['full', 'backtracking'])
def test_finite_energies_do_not_allow_overflowed_chain_tolerance(solver):
    class HugeOffset(nn.Module):
        def forward(self, q):
            return 1e308 + (q.square()*0).sum(-1, keepdim=True)
    core = configured_core(damping=0., solver=solver, max_bisections=0)
    core.potential_net = HugeOffset()
    q=torch.zeros(1,1,dtype=DTYPE);p=torch.zeros_like(q)
    assert torch.isfinite(core.energy(q,p)).all()
    with pytest.raises(SolverError):
        core(q,p,torch.zeros_like(q),.1)


@pytest.mark.parametrize('solver', ['full', 'backtracking'])
def test_finite_leaf_ledgers_cannot_overflow_macro_step(solver):
    class ZeroPotential(nn.Module):
        def forward(self, q):
            return (q*0).square().sum(-1,keepdim=True)
    class HugePort(nn.Module):
        def forward(self, q):
            return q*0+q.new_tensor([1e308,0.])
    core=DissipativeCore(q_dim=2,damping_mode='constant',substeps=2,
                        solver=solver,newton_iterations=12,max_bisections=0)
    core.potential_net=ZeroPotential();core.input_net=HugePort()
    with torch.no_grad():
        core.constant_factor.zero_();core.constant_factor[0]=1e154
    q=torch.zeros(1,2,dtype=DTYPE);p=q.new_tensor([[1.,0.]]);u=q.new_ones(1,1)
    with pytest.raises(SolverError,match='macro-step'):
        core(q,p,u,2.)
