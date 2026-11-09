"""Resumable official-core baseline trainer; preflight unless --execute is set."""
from __future__ import annotations
import argparse
import contextlib
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import time
import uuid
import numpy as np
import torch
from .comparison_models import build_comparison_model
from .comparison_protocol import preflight, json_digest, digest, source_manifest, maximum_horizon
from .dissipative_data import DissipativeWindowDataset, StepBatchSampler, fixed_validation_indices
from .train_dissipative import atomic_json, atomic_torch_save, capture_rng, restore_rng, runtime_signature
from .models import synchronize


@contextlib.contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".train.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _batch(dataset, indices, horizon, device):
    return {k:v.to(device) for k,v in dataset.batch(indices,horizon).items()
            if k in ("pixels", "action", "dt_frame")}


def _horizon(t, step):
    return next(s["horizon"] for s in t["curriculum"] if step <= s["until_step"])


def _lr(t, step):
    if t["warmup_steps"] and step <= t["warmup_steps"]:
        return step/t["warmup_steps"]
    progress = (step-t["warmup_steps"])/max(1,t["steps"]-t["warmup_steps"])
    return t["minimum_lr_ratio"]+(1-t["minimum_lr_ratio"])*.5*(1+math.cos(math.pi*min(1,max(0,progress))))


def scalar_losses(losses):
    values={}
    for key,value in losses.items():
        if isinstance(value,torch.Tensor) and value.numel()==1:
            values[key]=float(value.detach())
    if "loss" not in values or "prediction_loss" not in values or not all(math.isfinite(v) for v in values.values()):
        raise FloatingPointError("Adapter must provide finite scalar loss and prediction_loss")
    return values


def validation(model, dataset, indices, config, device):
    rng = capture_rng(); training = model.training
    # Preserve all persistent buffers as well as RNG across validation.
    buffers={name:b.detach().clone() for name,b in model.named_buffers()}
    result={}
    try:
        _seed(config["training"]["val_seed"]); model.eval()
        for horizon in config["training"]["val_horizons"]:
            total={}; count=0; n=config["training"]["batch_size"]
            with torch.no_grad():
                for start in range(0,len(indices),n):
                    ids=indices[start:start+n]
                    values=scalar_losses(model.loss(_batch(dataset,ids,horizon,device),horizon,stage="val"))
                    for key,value in values.items(): total[key]=total.get(key,0.)+value*len(ids)
                    count+=len(ids)
            result[str(horizon)]={key:value/count for key,value in total.items()}
        return dict(horizons=result, score=float(np.mean([r["prediction_loss"] for r in result.values()])),
                    score_scope="model-specific latent validation; not a cross-method physical ranking")
    finally:
        with torch.no_grad():
            for name,b in model.named_buffers(): b.copy_(buffers[name])
        restore_rng(rng); model.train(training)


def validate_saved(saved, config, report, sources, runtime=None):
    if (saved.get("kind")!="comparison" or saved.get("checkpoint_version")!=1
            or saved.get("protocol_sha256")!=json_digest(config) or saved.get("train_config")!=config
            or saved.get("dataset_sha256")!=report["dataset_sha256"] or saved.get("source_sha256")!=sources):
        raise ValueError("Checkpoint protocol/data/source identity mismatch")
    step=saved.get("step")
    if type(step) is not int or not 0<=step<=config["training"]["steps"] or saved.get("sampler_next_step")!=step:
        raise ValueError("Invalid checkpoint/sampler progress")
    if runtime is not None and saved.get("runtime")!=runtime:
        raise ValueError("Exact resume requires the same runtime; weights-only transfer is a separate operation")
    if any(not torch.isfinite(t).all() for t in saved["state_dict"].values()):
        raise ValueError("Nonfinite checkpoint state")
    return saved


def execute_run(data_path, output, config, device="cpu", resume=None, init_from=None,
                expected_parent_step=20000, stop_after_step=None, expected_parent_data_sha256=None):
    """Resume all training state, or transfer only a verified final A model to B."""
    if resume and init_from: raise ValueError("resume and init_from are mutually exclusive")
    config=copy.deepcopy(config); report=preflight(data_path,config); sources=source_manifest()
    if config["data"]["stage"]!="a" and config["training_route"]=="staged" and not (resume or init_from):
        raise ValueError("Staged B requires an exact A initialization; scratch is a separate explicit route")
    if config["training_route"]=="scratch" and init_from:
        raise ValueError("Scratch route cannot inherit pretrained weights")
    dev=torch.device(device)
    if dev.type not in {"cpu","cuda"} or (dev.type=="cuda" and not torch.cuda.is_available()):
        raise ValueError("Select an available CPU or CUDA device")
    t=config["training"]; limit=t["steps"] if stop_after_step is None else stop_after_step
    if type(limit) is not int or not 1<=limit<=t["steps"]: raise ValueError("Invalid stop_after_step")
    torch.set_num_threads(t["threads"]); runtime=runtime_signature(dev); output=Path(output)
    with run_lock(output):
        existing=(output/"run.json").exists()
        if existing and not resume:
            raise FileExistsError("Nonempty run requires an explicit resume checkpoint")
        if not existing and any(p.name!=".train.lock" for p in output.iterdir()):
            raise FileExistsError("Refusing to mix unrecognized run artifacts")
        _seed(t["seed"])
        model=build_comparison_model(config["model"],device=dev)
        resolved=copy.deepcopy(model.config)
        optimizer=torch.optim.AdamW(model.parameters(),lr=t["lr"],weight_decay=t["weight_decay"])
        train=DissipativeWindowDataset(data_path,"train",21,maximum_horizon(config),81,2)
        val=DissipativeWindowDataset(data_path,"val",21,maximum_horizon(config),81,2)
        sampler=StepBatchSampler(train,t["batch_size"],t["seed"])
        indices=fixed_validation_indices(val,t["val_windows"],t["val_seed"])
        start_step=0; parent=None; best=None; best_step=None; elapsed_prior=0.; last_validation=None
        try:
            if init_from:
                if config["data"]["stage"]=="a": raise ValueError("A weights transfer is only valid for B branches")
                saved=torch.load(init_from,map_location="cpu",weights_only=True)
                parent_config=saved.get("train_config")
                if not isinstance(parent_config,dict): raise ValueError("A checkpoint is missing its training protocol")
                validate_saved(saved,parent_config,{"dataset_sha256":saved.get("dataset_sha256")},sources)
                if expected_parent_data_sha256 is not None and saved.get("dataset_sha256")!=expected_parent_data_sha256:
                    raise ValueError("A checkpoint does not match the independently verified parent dataset")
                # Same architecture and verified A endpoint; do not reuse its optimizer/RNG.
                if (saved.get("kind")!="comparison" or saved.get("model_config")!=resolved
                        or saved.get("step")!=expected_parent_step or saved.get("validation_status")!="completed"
                        or saved.get("train_config",{}).get("data",{}).get("stage")!="a"
                        or parent_config["training"]["steps"]!=expected_parent_step
                        or not isinstance(saved.get("validation"),dict)
                        or not math.isfinite(saved["validation"].get("score",float("nan")))
                        or saved.get("source_sha256")!=sources
                        or saved.get("parent_config_sha256")!=report["parent_config_sha256"]):
                    raise ValueError("Invalid A checkpoint for weights-only B initialization")
                model.load_state_dict(saved["state_dict"],strict=True)
                parent=dict(checkpoint=str(Path(init_from).resolve()),sha256=digest(init_from),
                            step=saved["step"],optimizer_reset=True,all_parameters_trainable=True,
                            independently_verified_data_sha256=expected_parent_data_sha256)
            if resume:
                saved=validate_saved(torch.load(resume,map_location="cpu",weights_only=True),config,report,sources,runtime)
                if saved["model_config"]!=resolved or saved["validation_indices"]!=indices:
                    raise ValueError("Resolved model or validation anchors changed")
                start_step=saved["step"]
                if start_step>limit: raise ValueError("Requested stop precedes checkpoint")
                model.load_state_dict(saved["state_dict"]); optimizer.load_state_dict(saved["optimizer"])
                parent=saved.get("parent_transfer"); best=saved.get("best_score"); best_step=saved.get("best_step")
                elapsed_prior=saved.get("elapsed_seconds",0.); last_validation=saved.get("validation")
                restore_rng(saved["rng"])
                if existing:
                    old=json.loads((output/"run.json").read_text())
                    if old["protocol_sha256"]!=json_digest(config) or old["dataset_sha256"]!=report["dataset_sha256"]:
                        raise ValueError("Output run identity differs")
                    history_path=output/"history.jsonl"
                    if history_path.exists():
                        rows=[json.loads(s) for s in history_path.read_text().splitlines() if s.strip()]
                        committed=[r for r in rows if r["step"]<=start_step]
                        uncommitted=[r for r in rows if r["step"]>start_step]
                        if uncommitted:
                            (output/f"history_uncommitted_{uuid.uuid4().hex}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in uncommitted))
                            history_path.write_text("".join(json.dumps(r)+"\n" for r in committed))
            metadata=dict(kind="comparison",model_config=resolved,train_config=config,
                          protocol_sha256=json_digest(config),dataset_sha256=report["dataset_sha256"],
                          parent_config_sha256=report["parent_config_sha256"],source_sha256=sources,runtime=runtime,
                          provenance=model.provenance(),parent_transfer=parent,preflight=report,
                          precision="float32; method-specific explicit solver computation documented by adapter",
                          evaluation_selection="exact milestone, not best.pt",fixture_only=config["fixture_only"])
            if not existing: atomic_json(output/"run.json",metadata)
            tick=time.perf_counter(); model.train()
            def save(step, value):
                checkpoint=dict(checkpoint_version=1,**metadata,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                                step=step,sampler_next_step=step,rng=capture_rng(),validation=value,
                                validation_status="completed" if value is not None else "pending",
                                validation_indices=indices,best_score=best,best_step=best_step,
                                elapsed_seconds=elapsed_prior+time.perf_counter()-tick)
                atomic_torch_save(output/"last.pt",checkpoint)
                if best_step==step and value is not None: atomic_torch_save(output/"best.pt",checkpoint)
                return checkpoint
            if start_step==limit and last_validation is not None:
                save(start_step,last_validation)
            with (output/"history.jsonl").open("a") as log:
                for step in range(start_step+1,limit+1):
                    synchronize(dev); began=time.perf_counter(); k=_horizon(t,step)
                    for group in optimizer.param_groups: group["lr"]=t["lr"]*_lr(t,step)
                    optimizer.zero_grad(set_to_none=True)
                    losses=model.loss(_batch(train,sampler.indices(step-1),k,dev),k,stage="train")
                    values=scalar_losses(losses); losses["loss"].backward()
                    gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),t["gradient_clip"],error_if_nonfinite=True)
                    optimizer.step(); synchronize(dev)
                    row=dict(step=step,horizon=k,losses=values,gradient_norm_before_clip=float(gradient),
                             learning_rate=optimizer.param_groups[0]["lr"],train_seconds=time.perf_counter()-began)
                    last_validation=None
                    if step%t["val_every"]==0 or step==limit:
                        before=time.perf_counter(); last_validation=validation(model,val,indices,config,dev)
                        row.update(validation=last_validation,validation_seconds=time.perf_counter()-before)
                        if best is None or last_validation["score"]<best: best=last_validation["score"];best_step=step
                    log.write(json.dumps(row,allow_nan=False)+"\n"); log.flush()
                    if last_validation is not None or step%t["checkpoint_every"]==0: save(step,last_validation)
                    atomic_json(output/"status.json",dict(status="running",completed_steps=step,target_steps=t["steps"],fixture_only=config["fixture_only"]))
            if start_step==limit and last_validation is None:
                last_validation=validation(model,val,indices,config,dev); save(limit,last_validation)
            final=dict(status="completed" if limit==t["steps"] else "paused_at_checkpoint",completed_steps=limit,
                       target_steps=t["steps"],new_optimizer_updates=limit-start_step,
                       elapsed_seconds=elapsed_prior+time.perf_counter()-tick,fixture_only=config["fixture_only"])
            atomic_json(output/"status.json",final)
            return output/"last.pt"
        except BaseException as exc:
            atomic_json(output/"status.json",dict(status="failed",error_type=type(exc).__name__,error=str(exc),
                                                  recovery_checkpoint=str(output/"last.pt") if (output/"last.pt").exists() else None))
            raise
        finally:
            train.close(); val.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path);parser.add_argument("--execute",action="store_true")
    parser.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    g=parser.add_mutually_exclusive_group();g.add_argument("--resume",type=Path);g.add_argument("--init-from",type=Path)
    parser.add_argument("--expected-parent-step",type=int,default=20000);parser.add_argument("--stop-after-step",type=int)
    parser.add_argument("--parent-data",type=Path,help="Bind weights-only A transfer to independently supplied data bytes")
    args=parser.parse_args(argv);config=json.loads(args.config.read_text())
    if not args.execute: print(json.dumps(preflight(args.data,config),indent=2));return
    if args.output is None: parser.error("--execute requires --output")
    result=execute_run(args.data,args.output,config,args.device,args.resume,args.init_from,args.expected_parent_step,args.stop_after_step,
                       digest(args.parent_data) if args.parent_data else None)
    print(json.dumps({"checkpoint":str(result)}))

if __name__=="__main__": main()
