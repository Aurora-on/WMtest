"""Compare matched CUDA and MPS pilot workloads and extrapolate bounded budgets.

No network access or job submission. Inputs must contain completed, matching
train/evaluate runs. The workload is matched, not the numerical hardware backend.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def positive(value: Any, label: str, allow_zero: bool = False) -> float:
    v = float(value)
    require(math.isfinite(v) and (v >= 0 if allow_zero else v > 0), f"Invalid {label}: {value}")
    return v


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    at = (len(ordered) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def load_run(path: Path, warmup: int) -> dict:
    run = read_json(path / "run.json")
    summary = read_json(path / "summary.json")
    evaluation = read_json(path / "evaluation/metrics.json")
    history = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines() if line.strip()]
    n = int(run["args"]["steps"])
    require([row["step"] for row in history] == list(range(1, n + 1)), f"Incomplete/duplicate history: {path}")
    require(summary["steps"] == n and summary["batch_size"] == run["args"]["batch_size"], f"Summary/args mismatch: {path}")
    require(n > warmup >= 0, f"No post-warmup samples: {path}")
    require(not run["benchmark_only"] and not run["args"]["benchmark"], f"Need validation/checkpoint run: {path}")
    wall = [positive(row["wall_step_seconds"], "step wall time") for row in history]
    compute = [positive(row["compute_step_seconds"], "compute time") for row in history]
    require(all(math.isfinite(float(row["loss"])) for row in history), f"Non-finite loss: {path}")
    observed = wall[warmup:]
    mean = statistics.mean(observed)
    cycle_wall = positive(summary["wall_seconds"], "training cycle time")
    other = cycle_wall - sum(wall)
    require(other >= -1e-3, f"Training cycle shorter than summed steps: {path}")
    validation_steps = [row["step"] for row in history if "validation_loss" in row]
    interval = int(run["args"]["val_every"])
    expected_validation = sorted(set(range(interval, n + 1, interval)) | {n})
    require(validation_steps == expected_validation, f"Unexpected validation schedule: {path}")
    require(all(math.isfinite(float(row["validation_loss"])) for row in history if "validation_loss" in row),
            f"Invalid validation loss: {path}")
    half = max(1, len(observed) // 2)
    timing = {
        "warmup_steps_excluded": warmup,
        "measured_steps": len(observed),
        "step_seconds_mean": mean,
        "step_seconds_median": statistics.median(observed),
        "step_seconds_p90": percentile(observed, .9),
        "compute_seconds_mean": statistics.mean(compute[warmup:]),
        "clips_per_second": run["args"]["batch_size"] / mean,
        "first_half_step_seconds_mean": statistics.mean(observed[:half]),
        "second_half_step_seconds_mean": statistics.mean(observed[half:]),
        "training_internal_wall_seconds": cycle_wall,
        "all_step_wall_seconds_sum": sum(wall),
        "validation_save_log_other_seconds": max(0., other),
        "observed_validation_events": len(validation_steps),
        "warmup_excess_seconds": max(0., sum(wall[:warmup]) - warmup * mean),
        "evaluation_internal_wall_seconds": positive(evaluation["total_seconds"], "evaluation internal time"),
        "rollout_internal_seconds": positive(evaluation["rollout_seconds"], "rollout time", allow_zero=True),
    }
    timing["second_over_first_half_ratio"] = timing["second_half_step_seconds_mean"] / timing["first_half_step_seconds_mean"]
    return dict(path=str(path.resolve()), run=run, summary=summary, evaluation=evaluation, timing=timing)


def eval_signature(evaluation: dict) -> dict:
    probe_settings = {}
    for name, info in evaluation["probe_info"].items():
        # Best ridge alpha/MLP step and fitted values may differ across backends;
        # workload budgets and architecture must not.
        probe_settings[name] = {k: info[k] for k in ("optimization_steps", "hidden_layers", "label_budget") if k in info}
    coverage = sorted((row["probe"], row["mode"], float(row["horizon_s"]), int(row["episodes"]))
                      for row in evaluation["aggregate"])
    actual_coverage = sorted((int(row["episode"]), row["probe"], row["mode"], float(row["horizon_s"]))
                            for row in evaluation["episodes"])
    return dict(seed=evaluation["seed"], train_labels=evaluation["calibration_train_labels"],
                val_labels=evaluation["calibration_val_labels"], probe_settings=probe_settings,
                coverage=coverage, actual_coverage=actual_coverage)


def verify_pair(local: dict, server: dict, model: str, hardware: dict) -> list[str]:
    l, s = local["run"], server["run"]
    require(l["args"]["model"] == s["args"]["model"] == model, f"Wrong model in {model} pair")
    require(l["device"].startswith("mps") and s["device"].startswith("cuda"), "Expected local MPS/server CUDA pair")
    require(l["precision"] == s["precision"] == "float32", "Only matched FP32 workload supported")
    fields = ("dataset_sha256", "upstream", "config", "parameters", "trainable_parameters", "train_windows",
              "validation_windows_used", "scheduler", "gradient_clip_norm", "checkpoint_rule")
    checked = []
    for key in fields:
        require(key in l and key in s and l[key] == s[key], f"Mismatch in {model}: {key}")
        checked.append(key)
    require(s["dataset_sha256"] == hardware["dataset_sha256"], f"Hardware/data hash mismatch: {model}")
    require(float(l["dataset_attributes"]["damping"]) == float(s["dataset_attributes"]["damping"]) == .2,
            "Expected fixed b=0.2 profile data")
    ignored_args = {"data", "output", "device", "warmup"}
    la = {k: v for k, v in l["args"].items() if k not in ignored_args}
    sa = {k: v for k, v in s["args"].items() if k not in ignored_args}
    require(la == sa, f"Training CLI budgets/options differ in {model}: {la} vs {sa}")
    checked.append("training_args_except_paths_device_original_warmup")
    require(eval_signature(local["evaluation"]) == eval_signature(server["evaluation"]), f"Evaluation budget/episodes mismatch: {model}")
    checked.append("readout_and_evaluation_budget_and_test_episode_ids")
    for side in (local, server):
        require(side["evaluation"]["model"] == model, "Evaluation model mismatch")
    require(int(hardware["steps"]) == s["args"]["steps"], "Hardware metadata steps mismatch")
    require(int(hardware["batch_size"]) == s["args"]["batch_size"], "Hardware metadata batch mismatch")
    return checked


def process_table(server_path: Path) -> tuple[dict, dict]:
    complete = read_json(server_path / "COMPLETE.json")
    records = read_json(server_path / "process_times.json")
    require(complete.get("success") is True, "Server profile not complete")
    require(complete.get("records") == records, "COMPLETE/process_times records differ")
    index = {}
    for row in records:
        key = (row["model"], row["phase"])
        require(key not in index, f"Duplicate process time: {key}")
        require(row["returncode"] == 0, f"Failed server process: {key}")
        index[key] = positive(row["process_wall_seconds"], "process wall time")
    require(set(index) == {(m, p) for m in ("lehami", "lewm") for p in ("train", "evaluate")}, "Missing/unexpected process times")
    return index, complete


def auxiliary_measurements(server_path: Path, local_path: Path, expected_job: str, workload_config: dict) -> dict:
    """Optional already-downloaded evidence; never query Slurm/accounting here."""
    out = {}
    completion = server_path / "slurm_completion.txt"
    if completion.exists():
        text = completion.read_text()
        def field(name: str) -> str:
            hit = re.search(r"\b" + re.escape(name) + r"=(\S+)", text)
            require(hit is not None, f"Missing Slurm completion field: {name}")
            return hit.group(1)
        require(field("JobId") == str(expected_job), "Slurm completion job ID mismatch")
        require(field("JobState") == "COMPLETED" and field("ExitCode") == "0:0", "Slurm job did not finish successfully")
        runtime = field("RunTime")
        hit = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", runtime)
        require(hit is not None, f"Unrecognized Slurm RunTime: {runtime}")
        day, hour, minute, second = hit.groups()
        seconds = int(day or 0)*86400 + int(hour)*3600 + int(minute)*60 + int(second)
        out["slurm"] = dict(job_id=field("JobId"), state=field("JobState"), exit_code=field("ExitCode"),
                            runtime=runtime, runtime_seconds=seconds,
                            source=str(completion), evidence_method="Downloaded scontrol show job output; no sacct query")
    manifest_file = server_path / "simulation_manifest.json"
    if manifest_file.exists():
        manifest = read_json(manifest_file)
        require(manifest["config"] == workload_config, "Simulation config differs from training dataset recipe")
        datasets = manifest["datasets"]
        require({float(d["damping"]) for d in datasets} == {0., .2}, "Unexpected simulation damping matrix")
        internal = sum(positive(d["generation_total_seconds"], "generation total") for d in datasets)
        simulation = dict(server_internal_generation_seconds=internal,
                          server_ode_simulation_seconds=sum(positive(d["simulation_seconds"], "ODE simulation") for d in datasets),
                          total_frames=sum(int(d["frames"]) for d in datasets),
                          server_total_bytes=sum(int(d["bytes"]) for d in datasets),
                          server_regenerated_dataset_hashes={d["file"]: d["sha256"] for d in datasets},
                          configuration_matches_workload=True,
                          scope="Internal generation includes ODE, rendering, independent numerical audit, HDF5 writing and hashing. Regeneration is a separate timing task; model timing uses transferred matching data.")
        log = server_path / f"simulation_{expected_job}.log"
        if log.exists():
            times = re.findall(r"^real\s+([0-9]+(?:\.[0-9]+)?)\s*$", log.read_text(), flags=re.MULTILINE)
            require(len(times) == 1, "Expected one external simulation real time")
            outer = positive(times[0], "simulation process time")
            require(outer >= internal-1e-3, "Simulation external time shorter than internal")
            simulation["server_process_seconds"] = outer
        local_costs = local_path.resolve().parents[1] / "reports/data_costs.json"
        if local_costs.exists():
            local = read_json(local_costs)
            require(int(local["total_frames"]) == simulation["total_frames"], "Local/server simulation frame count mismatch")
            simulation["local_internal_generation_seconds"] = positive(local["generation_seconds"], "local generation")
            simulation["server_divided_by_local_internal_time"] = internal / simulation["local_internal_generation_seconds"]
            simulation["local_data_costs_source"] = str(local_costs)
            simulation["regenerated_hdf5_hashes_equal_local"] = (
                simulation["server_regenerated_dataset_hashes"] == {d["file"]: d["sha256"] for d in local["datasets"]})
        out["simulation"] = simulation
    return out


def estimate_single(timing: dict, n: int, val_every: int, external_train: float = 0., external_eval: float | None = None) -> dict:
    require(n >= timing["warmup_steps_excluded"], "Budget smaller than measured warmup")
    events = math.ceil(n / val_every)
    steps = n * timing["step_seconds_mean"]
    overhead = timing["validation_save_log_other_seconds"] * events / timing["observed_validation_events"]
    train_internal = steps + timing["warmup_excess_seconds"] + overhead
    evaluation = timing["evaluation_internal_wall_seconds"] if external_eval is None else external_eval
    return dict(optimizer_steps=n, validation_events=events, steady_step_work_seconds=steps,
                warmup_excess_seconds=timing["warmup_excess_seconds"],
                empirical_validation_save_log_seconds=overhead,
                training_internal_seconds=train_internal, process_startup_extra_seconds=external_train,
                evaluation_seconds=evaluation, total_seconds=train_internal + external_train + evaluation)


def compare(server_path: Path, local_path: Path, warmup: int = 20, margin: float = .25) -> dict:
    hardware = read_json(server_path / "hardware.json")
    proc, complete = process_table(server_path)
    require(warmup == 20, "Matched report requires the agreed step>20 statistic")
    require(math.isfinite(margin) and margin >= 0, "Margin must be nonnegative")
    models, raw = {}, {}
    for model in ("lehami", "lewm"):
        local = load_run(local_path / f"{model}_b0.2_seed0", warmup)
        server = load_run(server_path / model, warmup)
        checks = verify_pair(local, server, model, hardware)
        lt, st = local["timing"], server["timing"]
        for phase, internal in (("train", st["training_internal_wall_seconds"]), ("evaluate", st["evaluation_internal_wall_seconds"])):
            require(proc[(model, phase)] >= internal - 1e-3, f"External time shorter than internal: {model}/{phase}")
        memory = {key: server["summary"][key] for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "process_peak_rss_bytes")}
        for key, value in memory.items():
            positive(value, key, allow_zero=True)
        models[model] = dict(
            verification={"matched": True, "checked": checks},
            local=dict(timing=lt, device=local["run"]["device"], torch=local["run"]["torch_version"],
                       python=local["run"]["python"], machine=local["run"]["machine"],
                       checkpoint_step=local["evaluation"]["checkpoint_step"]),
            server=dict(timing=st, train_process_wall_seconds=proc[(model, "train")],
                        evaluate_process_wall_seconds=proc[(model, "evaluate")],
                        train_process_extra_seconds=max(0., proc[(model, "train")] - st["training_internal_wall_seconds"]),
                        evaluation_process_extra_seconds=max(0., proc[(model, "evaluate")] - st["evaluation_internal_wall_seconds"]),
                        cuda_and_host_memory=memory, device=server["run"]["device"],
                        memory_scope="Training process only; evaluation peak CUDA memory was not recorded.",
                        torch=server["run"]["torch_version"], python=server["run"]["python"],
                        machine=server["run"]["machine"], checkpoint_step=server["evaluation"]["checkpoint_step"]),
            local_divided_by_server_speed_ratios={
                "steady_training_step": lt["step_seconds_mean"] / st["step_seconds_mean"],
                "training_internal_total": lt["training_internal_wall_seconds"] / st["training_internal_wall_seconds"],
                "evaluation_internal_total": lt["evaluation_internal_wall_seconds"] / st["evaluation_internal_wall_seconds"]})
        raw[model] = (local, server)
    budgets = []
    for n in (5000, 28140):
        per_model = {}
        for model, (local, server) in raw.items():
            val_every = int(server["run"]["args"]["val_every"])
            per_model[model] = {
                "local_internal": estimate_single(local["timing"], n, val_every),
                "server_internal": estimate_single(server["timing"], n, val_every),
                "server_process": estimate_single(server["timing"], n, val_every,
                                                  models[model]["server"]["train_process_extra_seconds"],
                                                  proc[(model, "evaluate")])}
        totals = {}
        for seeds in (1, 3):
            scopes = {}
            for scope in ("local_internal", "server_internal", "server_process"):
                seconds = sum(per_model[m][scope]["total_seconds"] for m in per_model) * 2 * seeds
                scopes[scope] = dict(seconds=seconds, hours=seconds/3600,
                                     planning_upper_seconds=seconds*(1+margin),
                                     planning_upper_hours=seconds*(1+margin)/3600)
            totals[f"{seeds}_seed_{4*seeds}_runs"] = scopes
        budgets.append(dict(steps_per_run=n, per_model=per_model, totals=totals))
    example = raw["lehami"][1]
    auxiliary = auxiliary_measurements(server_path, local_path, str(hardware.get("job_id")),
                                       json.loads(example["run"]["dataset_attributes"]["config_json"]))
    workload = dict(profile_training_steps=example["run"]["args"]["steps"],
                    batch_size=example["run"]["args"]["batch_size"],
                    validation_interval=example["run"]["args"]["val_every"],
                    validation_windows=example["run"]["validation_windows_used"],
                    train_labels=example["evaluation"]["calibration_train_labels"],
                    validation_labels=example["evaluation"]["calibration_val_labels"],
                    mlp_steps=example["evaluation"]["probe_info"].get("mlp", {}).get("optimization_steps"),
                    test_episodes=len({r["episode"] for r in example["evaluation"]["episodes"]}),
                    horizons=sorted({r["horizon_s"] for r in example["evaluation"]["aggregate"]}))
    return dict(schema_version=1, server_directory=str(server_path.resolve()), local_directory=str(local_path.resolve()),
                job_id=hardware.get("job_id"), hardware=hardware, completed=True, warmup_for_statistics=warmup,
                matched_workload=workload,
                auxiliary_measurements=auxiliary,
                models=models, budgets=budgets, planning_margin_fraction=margin,
                method={
                    "speed_ratio": "local time / server time; >1 means server faster; only same internal scope compared",
                    "data_scope": "Measured fixed b=0.2, seed0. Other damping settings and seeds are budget extrapolations, not measured runs.",
                    "precision": "FP32 parameter workload, no autocast; CUDA/cuDNN TF32 defaults disclosed. Backend arithmetic and RNG need not match MPS bitwise.",
                    "overhead": "Training cycle wall minus all timed step walls includes validation, saving, logging and other loop work; grouped overhead is scaled by validation event count. It is not a separately measured validation latency.",
                    "budget": "N*post-warmup mean + measured warmup excess + grouped overhead scaled by validation count + one full evaluation per run. Server process scope additionally includes measured training startup extra and full evaluation process wall.",
                    "local_limit": "Local lacks outer process timer, so its budget omits startup/final output outside internal timers; no matched full-process speedup is claimed.",
                    "uncertainty": f"Upper budget adds a {margin:.0%} planning margin, not a statistical confidence interval. Excludes queue/setup/upload, future searches, long-duration/shared-load drift and pricing.",
                    "memory": "CUDA allocated/reserved are true reported peaks; local MPS current driver allocation is not comparable peak memory."})


def fmt_bool(value: Any) -> str:
    return "未记录" if value is None else str(value)


def markdown(result: dict) -> str:
    h = result["hardware"]
    w = result["matched_workload"]
    lines = ["# 阻尼摆：服务器与本地匹配短测", "",
             f"服务器作业 `{result['job_id']}` 已完成。数据、模型配置、上游源码、训练/验证预算及实际测试轨迹一致；双方均从 **step>20** 重算稳态。", "",
             f"服务器GPU：**{h.get('gpu_name', '未记录')}**；CUDA runtime：`{h.get('cuda_runtime', '未记录')}`；PyTorch：`{h.get('versions', {}).get('torch', '未记录')}`。", "",
             f"参数/训练精度为FP32、未使用autocast；CUDA matmul TF32=`{fmt_bool(h.get('cuda_matmul_allow_tf32'))}`，cuDNN TF32=`{fmt_bool(h.get('cudnn_allow_tf32'))}`。若cuDNN TF32开启，不能称严格全算子IEEE FP32；与MPS比较的是匹配工作量，不是逐位相同的数值路径。", "",
             "## 匹配工作量实测", "",
             "倍数为本地秒数÷服务器秒数，>1表示服务器更快。所有下表耗时都采用两端相同的内部计时范围。", "",
             "| 模型 | 本地稳态秒/步 | 服务器稳态秒/步 | 单步倍数 | 本地训练秒 | 服务器训练秒 | 本地评测秒 | 服务器评测秒 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    auxiliary = result.get("auxiliary_measurements", {})
    if "slurm" in auxiliary:
        job = auxiliary["slurm"]
        lines[3:3] = [f"Slurm完成记录：`JobState={job['state']}`、`ExitCode={job['exit_code']}`，作业总运行 **{job['runtime']}（{job['runtime_seconds']}秒）**。依据已下载的 `scontrol show job` 输出，不使用本集群未启用的sacct accounting；总时长与单独训练/评测进程计时范围不同。", ""]
    for name, data in result["models"].items():
        l, s, ratio = data["local"]["timing"], data["server"]["timing"], data["local_divided_by_server_speed_ratios"]
        lines.append(f"| {name} | {l['step_seconds_mean']:.4f} | {s['step_seconds_mean']:.4f} | {ratio['steady_training_step']:.2f}× | {l['training_internal_wall_seconds']:.2f} | {s['training_internal_wall_seconds']:.2f} | {l['evaluation_internal_wall_seconds']:.2f} | {s['evaluation_internal_wall_seconds']:.2f} |")
    horizons = "/".join(f"{x:g}" for x in w["horizons"])
    lines += ["", f"每模型训练{w['profile_training_steps']}步、batch{w['batch_size']}，固定阻尼b=0.2、种子0；验证每{w['validation_interval']}步、{w['validation_windows']}个窗口。完整评测为{w['train_labels']}/{w['validation_labels']}读出校准标签、{w['mlp_steps']}步MLP、{w['test_episodes']}条测试轨迹及{horizons}秒预测。短测不是收敛比较；最优checkpoint步骤可以因后端数值与随机数不同而不同。", "",
              "## 服务器完整进程与训练显存", "", "| 模型 | 训练进程秒 | 评测进程秒 | 训练CUDA峰值allocated(GiB) | 训练CUDA峰值reserved(GiB) | 稳态后半/前半耗时比 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, data in result["models"].items():
        s = data["server"]; mem = s["cuda_and_host_memory"]
        lines.append(f"| {name} | {s['train_process_wall_seconds']:.2f} | {s['evaluate_process_wall_seconds']:.2f} | {mem['cuda_peak_allocated_bytes']/2**30:.3f} | {mem['cuda_peak_reserved_bytes']/2**30:.3f} | {s['timing']['second_over_first_half_ratio']:.3f} |")
    lines += ["", "训练逐步计时含取数、传输、前后向、参数更新和同步，不含该步后验证/保存/日志；内部训练总时间包含这些循环开销，仍不含启动/导入/hash/建模。评测内部计时早于最终metrics/轨迹写入。服务器外层进程计时包含这些额外开销。**本地没有同样的外层计时，不能据此报告严格端到端加速比。** CUDA两种峰值含义不同，也不能与MPS的末次driver分配量横比峰值。这里峰值仅来自训练进程；评测进程峰值未记录。", "",
              "## 正式预算外推", "", "每次训练后完整评测一次；两阻尼分别训练。同一张GPU顺序执行，小时数也是该计划的近似GPU占用小时。下表服务器范围为点估计至附加计划余量的上界，**不是置信区间**。", "",
              "| 每组步数 | 矩阵 | 本地内部口径小时 | 服务器同口径内部小时 | 服务器完整进程计划小时 |", "|---:|---|---:|---:|---:|"]
    for budget in result["budgets"]:
        for name, total in budget["totals"].items():
            s = total["server_process"]
            label = "1种子×2模型×2阻尼＝4组" if name.startswith("1_") else "3种子×2模型×2阻尼＝12组"
            lines.append(f"| {budget['steps_per_run']} | {label} | {total['local_internal']['hours']:.2f} | {total['server_internal']['hours']:.2f} | {s['hours']:.2f}–{s['planning_upper_hours']:.2f} |")
    lines += ["", "28140步只在当前25,728窗口、batch32、drop_last配置下等于35epoch；5000步约6.2epoch，均不保证收敛。", "",
              "预算方法：N×稳态每步耗时，加实测预热额外开销；训练总时间减逐步时间之和得到验证/保存/日志等合计开销，按每250步一次及末步验证的次数外推；再加一次完整评测。服务器完整进程计划额外计入训练启动和最终评测输出写入。合计开销不是纯验证延迟，外推也不能替代长时实测。", "",
              f"计划余量为{result['planning_margin_fraction']:.0%}，未计排队、首次环境准备、上传、额外调参和未知收费。只实测了b=0.2/种子0；b=0与其他种子的时间属于形状和计算流程相同条件下的外推。若长时或共享负载改变速度，预算需相应更新。", "",
              "配置核对、稳态中位数/p90、分阶段开销、实际checkpoint步骤、硬件信息和逐模型预算分解保存在同名JSON。没有把GPU理论算力换算为时间，也没有给出未经确认的货币费用。"]
    if "simulation" in auxiliary:
        sim = auxiliary["simulation"]
        lines += ["", "## 仿真与数据生成实测", "",
                  f"服务器按同一生成配置生成两组、合计{sim['total_frames']:,}帧，内部总计时 **{sim['server_internal_generation_seconds']:.2f}秒**。该时间包含ODE、渲染、独立数值审计、HDF5写入及hash，不能全称为ODE计算时间；单独ODE累计为{sim['server_ode_simulation_seconds']:.2f}秒。"]
        if "server_process_seconds" in sim:
            lines += [f"服务器生成进程外层 `time real` 为 **{sim['server_process_seconds']:.2f}秒**，含启动和导入。"]
        if "local_internal_generation_seconds" in sim:
            lines += [f"同口径本地内部生成时间为 **{sim['local_internal_generation_seconds']:.2f}秒**；本次服务器用时为本地的{sim['server_divided_by_local_internal_time']:.2f}倍，因此当前仿真无需为提速迁移服务器。这个结论包含CPU、文件系统和依赖差异，不能只归因于某个处理器。"]
        if not sim.get("regenerated_hdf5_hashes_equal_local", True):
            lines += ["服务器重新生成的HDF5文件hash与本地不同，不称逐字节相同数据；两者数值审计均由生成器报告。训练短测使用的是另行传输、SHA256严格一致的原始数据，因此该再生成差异不改变训练比较的数据一致性。"]
    lines += ["", "## 运行位置建议", "",
              "**本地完成仿真、数据检查和小规模调试；服务器单GPU执行正式训练及多种子矩阵。** 本次服务器训练明显更快，但完整评测并没有同步加速。评测可留在本地，也可随服务器训练直接保存结果以减少传输；应预先固定评测设备、数值精度、校准种子和读出预算，全部方法遵守同一协议，避免在比较中随意更换后端。", "",
              "这里没有启动正式12组矩阵；预算以完成的匹配短测为依据。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planning-margin", type=float, default=.25)
    args = parser.parse_args()
    result = compare(args.server, args.local, margin=args.planning_margin)
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "server_runtime_comparison.json"
    md_path = args.output / "server_runtime_comparison.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    md_path.write_text(markdown(result))
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "job_id": result["job_id"],
                      "workload_matched": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
