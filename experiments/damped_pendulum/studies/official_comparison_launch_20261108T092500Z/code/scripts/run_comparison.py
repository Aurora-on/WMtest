#!/usr/bin/env python3
"""A20k -> single20k / multi20k for each official-core baseline. No launch by default.

--route scratch explicitly enables the optional 40k per-condition from-scratch
track; never silently substitute it for staged transfer. This local launcher
does not submit to Slurm. It uses only GPUs already allocated to its process.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dpbench.comparison_protocol import make_config, source_manifest, digest, json_digest, preflight
from dpbench.damping_stage_data import validate_pair
from dpbench.train_dissipative import atomic_json


def plan(args):
    jobs=[]
    for method in args.methods:
        for seed in args.seeds:
            stages=("a","single","multi") if args.route=="staged" else ("single","multi")
            configs={}
            for stage in stages:
                cfg=make_config(method,stage,seed,steps=20000 if args.route=="staged" else 40000)
                cfg["training_route"]=args.route;configs[stage]=cfg
            jobs.append(dict(method=method,seed=seed,configs=configs))
    return dict(schema_version=1,route=args.route,jobs=jobs,
                inputs={name:str(getattr(args,"data_"+name).resolve()) if getattr(args,"data_"+name) else None for name in ("a","single","multi")},
                formal_training_started=False,
                expected_optimizer_updates_per_method_seed=60000 if args.route=="staged" else 80000,
                single_path_budget=40000,
                B_training="all model parameters, fresh optimizer; never reuse our frozen A encoder weights",
                readout="each final model gets its own K0 train288/select96 calibrated readout on corresponding conditions",
                test_policy="validation only; held-out test and unseen damping remain sealed",
                adaptations="official cores/losses; 64px CNN frontends and causal21-frame adapters are explicitly declared",
                gpus=args.gpus,device=args.device,skip_evaluation=args.skip_evaluation)


def environments(device,gpus):
    env=dict(os.environ);env.update(OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",PYTHONPATH=str(ROOT))
    if device=="cpu":return [dict(env,CUDA_VISIBLE_DEVICES="")]
    import torch
    ids=[int(s) for s in gpus.split(",")]
    if not ids or len(ids)>2 or len(set(ids))!=len(ids) or min(ids)<0 or max(ids)>=torch.cuda.device_count():
        raise ValueError("Select one or two distinct indices within the allocated CUDA_VISIBLE_DEVICES")
    physical=env.get("CUDA_VISIBLE_DEVICES","").split(",") if env.get("CUDA_VISIBLE_DEVICES") else [str(i) for i in range(torch.cuda.device_count())]
    return [dict(env,CUDA_VISIBLE_DEVICES=physical[i]) for i in ids]


def verify_evaluation(directory, checkpoint_sha256, dataset_sha256, protocol_sha256, step):
    directory=Path(directory)
    done=json.loads((directory/"complete.json").read_text())
    required={"readout.npz","readout_provenance.json","summary.json","val/summary.json","val/trajectories.npz"}
    artifacts=done.get("artifacts",{})
    if done.get("status")!="completed" or not required<=set(artifacts):
        raise ValueError("Incomplete evaluation manifest")
    expected={"checkpoint_sha256":checkpoint_sha256,"dataset_sha256":dataset_sha256,"protocol_sha256":protocol_sha256}
    for name,sha in artifacts.items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path)!=sha:
            raise ValueError("Evaluation artifact hash mismatch")
    prov=json.loads((directory/"readout_provenance.json").read_text())
    summary=json.loads((directory/"summary.json").read_text())
    for item in (done,prov,summary):
        if any(item.get(k)!=v for k,v in expected.items()):raise ValueError("Evaluation identity mismatch")
    evaluation=prov.get("evaluation_protocol",{})
    if evaluation.get("horizons")!=[8,16,32] or evaluation.get("split")!="val" or prov.get("checkpoint_step")!=step:
        raise ValueError("Evaluation step/horizons/split differ")
    eval_hash=json_digest(evaluation)
    if any(item.get("evaluation_protocol_sha256")!=eval_hash for item in (done,prov,summary)):
        raise ValueError("Evaluation protocol hash mismatch")
    if summary.get("evaluation_protocol")!=evaluation:
        raise ValueError("Evaluation summary protocol differs")
    if summary.get("status")!="completed" or summary.get("checkpoint_step")!=step:
        raise ValueError("Evaluation summary is incomplete or belongs to another step")
    evaluations=summary.get("evaluation")
    result=evaluations.get("val") if isinstance(evaluations,dict) else None
    if (not isinstance(result,dict) or result.get("split")!="val"
            or result.get("horizon_steps")!=32 or not isinstance(result.get("rollout"),dict)
            or set(result["rollout"])!={"8","16","32"}):
        raise ValueError("Evaluation results do not cover the declared validation horizons")
    if json.loads((directory/"val/summary.json").read_text())!=result:
        raise ValueError("Validation summary disagrees with the published evaluation")
    return done


def worker(task_path):
    import torch
    from dpbench.train_comparison import execute_run,validate_saved
    from dpbench.evaluate_comparison import evaluate_checkpoint
    task=json.loads(Path(task_path).read_text()); config=task["config"]; out=Path(task["output"])
    out.mkdir(parents=True,exist_ok=True)
    with (out/".worker.lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            hashes=source_manifest()
            if hashes!=task["sources"] or digest(task["data"])!=task["data_sha256"]:
                raise ValueError("Task sources/data changed")
            if task.get("parent") and digest(task["parent"])!=task.get("parent_sha256"):
                raise ValueError("Parent checkpoint changed after task creation")
            report=preflight(task["data"],config)
            checkpoints=sorted((out/"segments").glob("*/last.pt"))
            candidate=checkpoints[-1] if checkpoints else None
            saved=None
            if candidate:
                saved=validate_saved(torch.load(candidate,map_location="cpu",weights_only=True),config,report,hashes)
            if saved is None or saved["step"]!=config["training"]["steps"] or saved["validation_status"]!="completed":
                segments=out/"segments";segments.mkdir(exist_ok=True)
                numbers=[int(p.name) for p in segments.iterdir() if p.is_dir() and p.name.isdigit()]
                segment=segments/f"{max(numbers,default=0)+1:04d}"
                candidate=execute_run(task["data"],segment,config,task["device"],resume=candidate,
                                      init_from=task.get("parent") if candidate is None else None,
                                      expected_parent_data_sha256=task.get("parent_data_sha256"))
            saved=torch.load(candidate,map_location="cpu",weights_only=True)
            if saved["step"]!=config["training"]["steps"] or saved["validation_status"]!="completed":
                raise ValueError("Exact target checkpoint not validated")
            checkpoint_info={"path":str(candidate.resolve()),"sha256":digest(candidate),"step":saved["step"],"best_not_selected":True}
            atomic_json(out/"checkpoint_manifest.json",checkpoint_info)
            if not task["skip_evaluation"]:
                published=out/"evaluation"
                if (published/"complete.json").exists():
                    verify_evaluation(published,checkpoint_info["sha256"],task["data_sha256"],json_digest(config),saved["step"])
                else:
                    if published.exists():raise ValueError("Incomplete published evaluation requires inspection")
                    attempt=out/"evaluation_attempts"/uuid.uuid4().hex;attempt.parent.mkdir(exist_ok=True)
                    evaluate_checkpoint(candidate,task["data"],attempt,device=task["device"],horizons=(8,16,32))
                    verify_evaluation(attempt,checkpoint_info["sha256"],task["data_sha256"],json_digest(config),saved["step"])
                    os.replace(attempt,published)
            atomic_json(out/"worker_status.json",dict(status="trained_not_evaluated" if task["skip_evaluation"] else "completed",checkpoint=checkpoint_info))
        except BaseException as exc:
            atomic_json(out/"worker_status.json",dict(status="failed",error_type=type(exc).__name__,error=str(exc)));raise


def launch(tasks,envs):
    processes=[]
    try:
        for task,env in zip(tasks,envs):
            output=Path(json.loads(Path(task).read_text())["output"]);output.mkdir(parents=True,exist_ok=True)
            handle=(output/"worker.log").open("a")
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--worker",str(task),"--execute"],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            processes.append((process,handle,task))
        failures=[]
        for proc,handle,task in processes:
            if proc.wait()!=0:failures.append(str(task))
        if failures:raise RuntimeError(f"Workers failed; inspect worker_status.json/log: {failures}")
    except BaseException:
        for proc,_,_ in processes:
            if proc.poll() is None:proc.terminate()
        for proc,_,_ in processes:
            if proc.poll() is None:
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
        raise
    finally:
        for _,handle,_ in processes:handle.close()


def execute(args,spec):
    if not args.output or any(path is None for path in spec["inputs"].values()):
        raise ValueError("Execution requires output and all three paired data paths (A also audits scratch pairing)")
    paired={name:validate_pair(spec["inputs"]["a"],spec["inputs"][name],multi=name=="multi") for name in ("single","multi")}
    envs=environments(args.device,args.gpus); root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/".pipeline.lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=dict(plan=spec,sources=source_manifest(),data_sha256={k:digest(p) for k,p in spec["inputs"].items()},paired=paired)
        marker=root/"pipeline.json"
        if marker.exists():
            if json.loads(marker.read_text())!=manifest: raise ValueError("Existing pipeline protocol/data/source identity differs")
        elif any(p.name!=".pipeline.lock" for p in root.iterdir()):
            raise FileExistsError("Choose a new output or resume an identical pipeline")
        else:atomic_json(marker,manifest)
        try:
            for job in spec["jobs"]:
                base=root/f"{job['method']}_seed{job['seed']}";base.mkdir(exist_ok=True)
                def task(stage,parent=None):
                    cfg=job["configs"][stage]; data=spec["inputs"][stage]
                    preflight(data,cfg)
                    record=dict(config=cfg,data=data,data_sha256=manifest["data_sha256"][stage],sources=manifest["sources"],
                                output=str(base/stage),device=args.device,parent=parent,
                                parent_sha256=digest(parent) if parent else None,
                                parent_data_sha256=manifest["data_sha256"]["a"] if parent else None,
                                skip_evaluation=args.skip_evaluation)
                    path=base/(stage+"_task.json");atomic_json(path,record);return path
                parent=None
                if args.route=="staged":
                    launch([task("a")],[envs[0]])
                    parent=json.loads((base/"a/checkpoint_manifest.json").read_text())["path"]
                branches=[task(name,parent) for name in ("single","multi")]
                if len(envs)==2:launch(branches,envs)
                else:
                    for branch in branches:launch([branch],envs)
            atomic_json(root/"status.json",dict(status="trained_not_evaluated" if args.skip_evaluation else "completed",jobs=len(spec["jobs"]),route=args.route))
        except BaseException as exc:
            atomic_json(root/"status.json",dict(status="failed",error_type=type(exc).__name__,error=str(exc)));raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods",nargs="+",choices=["lehami","lewm","hjepa"],default=["lehami","lewm","hjepa"])
    p.add_argument("--seeds",nargs="+",type=int,default=[0]);p.add_argument("--route",choices=["staged","scratch"],default="staged")
    for name in ("a","single","multi"):p.add_argument("--data-"+name,type=Path)
    p.add_argument("--output",type=Path);p.add_argument("--execute",action="store_true")
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu");p.add_argument("--gpus",default="0,1")
    p.add_argument("--skip-evaluation",action="store_true",help="Explicitly produce trained_not_evaluated, not completed results")
    p.add_argument("--write-plan",type=Path);p.add_argument("--worker",type=Path,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.worker:
        if not args.execute:p.error("Internal worker requires --execute")
        worker(args.worker);return
    if len(set(args.methods))!=len(args.methods) or len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0:
        p.error("Methods/seeds must be unique and seeds nonnegative")
    spec=plan(args)
    if args.write_plan:
        if args.write_plan.exists():raise FileExistsError(args.write_plan)
        args.write_plan.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.write_plan,spec)
    if not args.execute:print(json.dumps(spec,indent=2,ensure_ascii=False));return
    execute(args,spec)

if __name__=="__main__":main()
