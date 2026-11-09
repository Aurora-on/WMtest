#!/usr/bin/env python3
"""Read-only analysis of one captured implicit substep, without parameter updates.

The baseline reproduces the existing masked finite Newton recurrence. Extra
iterations/line search solve the same residual; subdividing h changes the map.
This is local solver evidence, not a retraining or rollout performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from dpbench.dissipative_core import DissipativeCore


def values(tensor):
    if tensor is None:return None
    return tensor.detach().cpu().tolist()


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def jacobian(y,x):
    return torch.stack([torch.autograd.grad(y[:,j].sum(),x,retain_graph=True)[0] for j in range(y.shape[1])],-2)


def diagnostics(core,q,p,action,h,context,velocity):
    """Every diagnostic is evaluated at the current Newton iterate."""
    residual,gradient,force,drive=core._residual(velocity,q,p,action,h,context)
    jac=jacobian(residual,velocity)
    q1=q+h*velocity;p1=2*velocity-p;middle=(q+q1)*.5
    D=core.damping(middle,velocity,context)
    # Hold q_mid fixed to isolate d[D(q_mid,v,c)v]/dv, not the total q(v) derivative.
    vv=velocity.detach().clone().requires_grad_(True)
    fixed_force=(core.damping(middle.detach(),vv,context)@vv.unsqueeze(-1)).squeeze(-1)
    velocity_jac=jacobian(fixed_force,vv)
    eig=torch.linalg.eigvalsh(.5*(velocity_jac+velocity_jac.transpose(-1,-2)))
    singular=torch.linalg.svdvals(jac)
    threshold=10*math.sqrt(torch.finfo(q.dtype).eps)*(1+q.norm(dim=-1))
    delta=q1-q;norm=delta.norm(dim=-1)
    numerator=core.potential(q1)-core.potential(q)-(gradient*delta).sum(-1)
    # The numerator above uses the final DG (its value is the chain defect).
    m=middle.detach().clone().requires_grad_(True)
    midpoint_gradient=torch.autograd.grad(core.potential(m).sum(),m)[0]
    divided_difference_numerator=core.potential(q1)-core.potential(q)-(midpoint_gradient*delta).sum(-1)
    tolerance=.5*(core.residual_atol+core.residual_rtol*torch.maximum(p.abs().amax(-1),p1.abs().amax(-1)))
    item={"residual_linf":values(residual.abs().amax(-1)),"residual_l2":values(residual.norm(dim=-1)),
          "newton_tolerance":values(tolerance),"residual_to_tolerance":values(residual.abs().amax(-1)/tolerance),
          "velocity":values(velocity),"q_next":values(q1),"p_next":values(p1),
          "jacobian_condition_2":values(singular[:,0]/singular[:,-1]),
          "jacobian_min_singular":values(singular[:,-1]),"jacobian_max_singular":values(singular[:,0]),
          "q_norm":values(q.norm(dim=-1)),"p_norm":values(p.norm(dim=-1)),
          "q_next_norm":values(q1.norm(dim=-1)),"p_next_norm":values(p1.norm(dim=-1)),
          "damping_eigenvalues":values(torch.linalg.eigvalsh(D)),
          "damping_force_velocity_jacobian_sym_eigenvalues":values(eig),
          "displacement_norm":values(norm),"small_branch_threshold_norm":values(threshold),
          "small_displacement_branch":values(norm<=threshold),
          "displacement_threshold_ratio":values(norm/threshold),
          "dg_divided_difference_numerator":values(divided_difference_numerator),
          "dg_chain_defect":values(numerator)}
    return residual,jac,tolerance,item


def final_ledger(core,q,p,action,h,context,velocity):
    _,gradient,force,drive=core._residual(velocity,q,p,action,h,context)
    q1=q+h*velocity;p1=2*velocity-p;dq=q1-q;dp=p1-p
    rq=dq-h*velocity;rp=dp-h*(-gradient-force+drive)
    residual=torch.cat([rq,rp],-1).abs().amax(-1)
    energy0=core.energy(q,p);energy1=core.energy(q1,p1)
    defect=energy1-energy0-(gradient*dq).sum(-1)-(velocity*dp).sum(-1)
    tolerance=core.residual_atol+core.residual_rtol*torch.maximum(p.abs().amax(-1),p1.abs().amax(-1))
    chainlimit=core.chain_atol*(1+energy0.abs()+energy1.abs())
    finite=torch.isfinite(torch.cat([q1,p1],-1)).all(-1)&torch.isfinite(residual)&torch.isfinite(defect)
    accepted=finite&(residual<=tolerance)&(defect.abs()<=chainlimit)
    return q1.detach(),p1.detach(),{"accepted":values(accepted),"dynamics_residual":values(residual),
        "dynamics_tolerance":values(tolerance),"dynamics_residual_to_tolerance":values(residual/tolerance),
        "chain_defect":values(defect),"chain_limit":values(chainlimit),"q_next":values(q1),"p_next":values(p1)}


def solve(core,q,p,action,h,context,budget,armijo=False):
    velocity=p.detach().clone().requires_grad_(True)
    traces=[];updates=torch.zeros(q.shape[0],device=q.device,dtype=torch.int64)
    error=None
    with torch.enable_grad():
        for iteration in range(budget+1):
            try:
                residual,jac,tol,item=diagnostics(core,q,p,action,h,context,velocity)
                item["iteration"]=iteration;traces.append(item)
                active=residual.detach().abs().amax(-1)>tol.detach()
                if not active.any() or iteration==budget:break
                update=torch.linalg.solve(jac,residual.unsqueeze(-1)).squeeze(-1)
                alpha=torch.ones(q.shape[0],device=q.device,dtype=q.dtype)
                if armijo:
                    old_phi=.5*residual.detach().square().sum(-1)
                    unresolved=active.clone()
                    for backtrack in range(21):
                        trial=velocity-alpha[:,None]*update
                        candidate=core._residual(trial,q,p,action,h,context)[0].detach()
                        phi=.5*candidate.square().sum(-1)
                        accepted=torch.isfinite(phi)&(phi<=(1-2e-4*alpha)*old_phi)
                        unresolved=active&~accepted
                        if not unresolved.any():break
                        alpha=torch.where(unresolved,alpha*.5,alpha)
                    item["armijo_alpha"]=values(alpha)
                    if unresolved.any():
                        error="Armijo failed to find sufficient decrease within 20 halvings";break
                item["update_norm"]=values(update.norm(dim=-1))
                velocity=(velocity-torch.where(active[:,None],alpha[:,None]*update,torch.zeros_like(update))).detach().requires_grad_(True)
                updates+=active.to(updates.dtype)
            except (RuntimeError,ValueError) as exc:
                error=f"{type(exc).__name__}: {exc}";break
        try:q1,p1,ledger=final_ledger(core,q,p,action,h,context,velocity)
        except (RuntimeError,ValueError) as exc:
            q1,p1=q.detach(),p.detach();ledger={"accepted":[False]*len(q)}
            error=error or f"Final ledger: {type(exc).__name__}: {exc}"
    return q1,p1,{"method":"armijo_newton" if armijo else "full_newton","budget":budget,
        "updates":values(updates),"error":error,"trace":traces,"final":ledger}


def subdivided(core,q,p,action,h,context,parts):
    reports=[];q1,p1=q,p
    for i in range(parts):
        q1,p1,rep=solve(core,q1,p1,action,h/parts,context,core.newton_iterations)
        reports.append(rep)
        if not all(rep["final"]["accepted"]):break
    return {"method":f"{parts}_substeps_full_newton","changes_discrete_map":True,
        "substeps_requested":parts,"substeps_completed":sum(all(r["final"]["accepted"]) for r in reports),
        "accepted":len(reports)==parts and all(all(r["final"]["accepted"]) for r in reports),
        "q_next":values(q1),"p_next":values(p1),"substeps":reports}


def analyze(path,device="cpu",sample_indices=None):
    start=time.perf_counter();capture=torch.load(path,map_location="cpu",weights_only=True)
    core=DissipativeCore(**capture["core_config"]).to(device)
    core.load_state_dict(capture["core_state_dict"])
    for param in core.parameters():param.requires_grad_(False)
    tensors={k:(v.detach().to(device).clone() if v is not None else None) for k,v in capture["input"].items()}
    q,p,action,h,context=[tensors[k] for k in ("q","p","action","h","context")]
    if h.ndim==0:h=h.expand(len(q),1)
    elif h.ndim==1:h=h[:,None]
    tensors["h"]=h
    with torch.enable_grad():
        _,_,original=solve(core,q,p,action,h,context,core.newton_iterations)
    failed=[i for i,ok in enumerate(original["final"]["accepted"]) if not ok]
    chosen=failed if sample_indices is None else sample_indices
    reports=[]
    for index in chosen:
        args=(core,q[index:index+1],p[index:index+1],action[index:index+1],h[index:index+1],None if context is None else context[index:index+1])
        variants={}
        for budget in (6,12,24):
            _,_,variants[f"full_newton_{budget}"]=solve(*args,budget)
        _,_,variants["armijo_newton_24"]=solve(*args,24,True)
        variants["two_half_substeps"]=subdivided(*args,2)
        variants["four_quarter_substeps"]=subdivided(*args,4)
        compact=[]
        for name,result in variants.items():
            if "final" in result:
                final=result["final"]
                compact.append({"method":name,"accepted":all(final["accepted"]),"updates":result["updates"],"residual":final.get("dynamics_residual"),"tolerance":final.get("dynamics_tolerance"),"error":result["error"],"changes_discrete_map":False})
            else:compact.append({"method":name,"accepted":result["accepted"],"substeps_completed":result["substeps_completed"],"changes_discrete_map":True})
        reports.append({"sample_index":index,"input":{k:values(v[index:index+1]) if v is not None else None for k,v in tensors.items()},"comparison":compact,"variants":variants})
    return clean({"capture_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"device":str(device),"torch":str(torch.__version__),"core_config":core.config,
        "capture_metadata":{k:capture.get(k) for k in ("training_step","phase","substep_call_in_loss","resume_step","error_type","error","track_gradient")},
        "batch_size":len(q),"reproduced_failed_sample_indices":failed,"same_failure_reproduced":bool(failed),
        "original_batched_newton":original,"individual_analyses":reports,"elapsed_seconds":time.perf_counter()-start,
        "interpretation_limits":["No optimizer updates or parameter changes; frozen local-substep analysis only.","CPU/CUDA and batch-size changes can alter floating-point rounding; compare the original batched replay first.","D positive semidefinite does not imply velocity-force Jacobian monotonicity or global root uniqueness.","A negative symmetric force-Jacobian eigenvalue is local nonmonotonicity evidence, not proof it caused this failure.","Extra Newton iterations/Armijo use the same implicit residual; reducing h changes the numerical map.","All acceptances retain the original dynamics and chain-defect tolerances."]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu");parser.add_argument("--sample-index",type=int,action="append")
    args=parser.parse_args();torch.set_num_threads(1)
    if args.output.exists():raise FileExistsError("Refusing to overwrite analysis evidence")
    result=analyze(args.capture,args.device,args.sample_index);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"failed_indices":result["reproduced_failed_sample_indices"],"comparisons":[r["comparison"] for r in result["individual_analyses"]],"elapsed_seconds":result["elapsed_seconds"]}))


if __name__=="__main__":main()
