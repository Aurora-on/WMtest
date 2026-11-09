"""Numerical checks for the independent, frozen-substep diagnostic harness."""
import torch

from dpbench.dissipative_core import DissipativeCore
from scripts.analyze_passive_solver_failure import diagnostics, solve


def test_instrumented_recurrence_matches_original_step():
    torch.manual_seed(912)
    core=DissipativeCore(context_dim=3)
    for parameter in core.parameters():parameter.requires_grad_(False)
    q=torch.randn(5,2,dtype=torch.float64)
    p=torch.randn_like(q)
    action=torch.zeros(5,1,dtype=torch.float64)
    h=torch.full((5,1),.05,dtype=torch.float64)
    context=torch.randn(5,3,dtype=torch.float64)
    with torch.enable_grad():
        expected_q,expected_p,expected=core._step(q,p,action,h,False,context)
        actual_q,actual_p,report=solve(core,q,p,action,h,context,6)
    torch.testing.assert_close(actual_q,expected_q,rtol=0,atol=0)
    torch.testing.assert_close(actual_p,expected_p,rtol=0,atol=0)
    torch.testing.assert_close(torch.tensor(report["updates"],dtype=torch.float64),expected["solver_iterations"],rtol=0,atol=0)
    assert all(report["final"]["accepted"])


def test_distinguishes_insufficient_iterations_and_force_nonmonotonicity():
    torch.manual_seed(913)
    core=DissipativeCore()
    for parameter in core.parameters():parameter.requires_grad_(False)
    q=torch.zeros(1,2,dtype=torch.float64);p=torch.full_like(q,100)
    action=torch.zeros(1,1,dtype=torch.float64);h=torch.ones(1,1,dtype=torch.float64)
    core.damping=lambda q,v,context=None:torch.diag_embed(v.square())
    _,_,short=solve(core,q,p,action,h,None,6)
    _,_,long=solve(core,q,p,action,h,None,24)
    assert not all(short["final"]["accepted"])
    assert all(long["final"]["accepted"])
    core.damping=lambda q,v,context=None:torch.diag_embed(torch.exp(-v.square()))
    velocity=torch.full_like(q,2).requires_grad_(True)
    _,_,_,info=diagnostics(core,q,p,action,h,None,velocity)
    assert min(info["damping_eigenvalues"][0])>0
    assert max(info["damping_force_velocity_jacobian_sym_eigenvalues"][0])<0
