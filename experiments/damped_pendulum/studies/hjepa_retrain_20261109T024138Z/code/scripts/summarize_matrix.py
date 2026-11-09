"""Validate and aggregate a completed LeHami/LeWM training matrix by seed.

Reads saved metadata/metrics only; does not train, evaluate or contact a server.
One training seed contributes one episode-mean score. Episode dispersion is
never substituted for independent training-seed uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import posixpath
import statistics

MODELS = ("lehami", "lewm")
DAMPINGS = (0., .2)
PROBES = ("ridge", "mlp")
MODES = ("passive", "driven")
HORIZONS = (4., 8., 16., 32.)


class IncompleteRun(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value, label):
    require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value),
            f"Invalid numeric {label}: {value}")
    return float(value)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-9)


def identity(run):
    return (run["args"]["model"], float(run["dataset_attributes"]["damping"]), int(run["args"]["seed"]))


def label(key):
    return f"{key[0]}_b{key[1]:g}_seed{key[2]}"


def recorded_paths_match(left, right):
    """Compare execution-host paths without resolving them on the review host.

    One absolute and one relative path may differ by the execution directory.
    In that case require the entire, multi-component relative suffix; a bare
    filename or parent traversal is insufficient to bind a downloaded result.
    """
    left, right = posixpath.normpath(str(left)), posixpath.normpath(str(right))
    if posixpath.isabs(left) == posixpath.isabs(right):
        return left == right
    absolute, relative = (left, right) if posixpath.isabs(left) else (right, left)
    parts = relative.split("/")
    return len(parts) >= 2 and ".." not in parts and absolute.endswith("/" + relative)


def inspect_history(path, run, summary, expected_steps):
    count, best_step, best = 0, None, float("inf")
    validation_steps = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        count += 1
        require(row["step"] == count, f"Non-contiguous/duplicate training step in {path}")
        finite(row["loss"], "training loss")
        require(finite(row["wall_step_seconds"], "step time") > 0, "Nonpositive step time")
        if "validation_loss" in row:
            score = finite(row["validation_loss"], "validation loss")
            validation_steps.append(count)
            if score < best:
                best, best_step = score, count
    if count < expected_steps:
        raise IncompleteRun(f"History has {count}/{expected_steps} steps")
    require(count == expected_steps, f"History exceeds requested {expected_steps} steps: {path}")
    interval = int(run["args"]["val_every"])
    require(interval > 0, "Invalid validation interval")
    expected_validation = sorted(set(range(interval, expected_steps + 1, interval)) | {expected_steps})
    require(validation_steps == expected_validation, f"Validation schedule mismatch: {path}")
    require(close(summary["best_validation_loss"], best), f"Best validation loss mismatch: {path}")
    return dict(actual_training_steps=count, best_checkpoint_step=best_step, best_validation_loss=best,
                validation_events=len(validation_steps))


def inspect_evaluation(evaluation, key, best_step):
    require(evaluation["model"] == key[0] and int(evaluation["seed"]) == key[2], "Evaluation identity mismatch")
    require(evaluation["independent_training_seeds"] == 1, "A per-run evaluation must contain one training seed")
    require(evaluation["checkpoint_step"] == best_step, "Evaluation is not of validation-selected best checkpoint")
    require(set(evaluation["probe_info"]) == set(PROBES), "Missing/unexpected readouts")
    grouped, seen, test_modes = {}, set(), {}
    for row in evaluation["episodes"]:
        group = (row["probe"], row["mode"], float(row["horizon_s"]))
        require(group in set(itertools.product(PROBES, MODES, HORIZONS)), f"Unexpected evaluation group: {group}")
        sample = (int(row["episode"]), *group)
        require(sample not in seen, f"Duplicate evaluation episode/group: {sample}")
        seen.add(sample)
        ep = int(row["episode"])
        require(ep not in test_modes or test_modes[ep] == row["mode"], "Episode action mode changed")
        test_modes[ep] = row["mode"]
        values = {k: finite(v, k) for k, v in row.items()
                  if k not in {"episode", "probe", "mode", "horizon_s"} and v is not None}
        require({"angle_rmse_rad", "omega_rmse_rad_s", "energy_rmse_J", "reference_budget_rmse_J",
                 "anchor_angle_error_rad", "anchor_omega_error_rad_s"} <= set(values), "Missing required physical metrics")
        require(row["mode"] != "passive" or "passive_energy_increase_fraction" in values,
                "Missing passive energy increase metric")
        grouped.setdefault(group, []).append((ep, values))
    require(set(grouped) == set(itertools.product(PROBES, MODES, HORIZONS)), "Incomplete readout/mode/horizon coverage")
    expected_ids = {mode: sorted(ep for ep, m in test_modes.items() if m == mode) for mode in MODES}
    require(all(expected_ids.values()), "Both passive and driven test episodes are required")
    aggregates = {}
    for row in evaluation["aggregate"]:
        group = (row["probe"], row["mode"], float(row["horizon_s"]))
        require(group in grouped and group not in aggregates, "Unexpected/duplicate aggregate row")
        samples = grouped[group]
        require(sorted(ep for ep, _ in samples) == expected_ids[group[1]], "Test episode coverage changes across horizon/readout")
        require(row["episodes"] == len(samples), "Aggregate episode count mismatch")
        metric_keys = set(samples[0][1])
        require(all(set(v) == metric_keys for _, v in samples), "Metric keys differ across episodes")
        require(set(row["mean"]) == metric_keys, "Aggregate metric keys mismatch")
        mean = {}
        for metric in sorted(metric_keys):
            expected = statistics.mean(v[metric] for _, v in samples)
            actual = finite(row["mean"][metric], metric)
            require(close(actual, expected), f"Stored aggregate does not equal episode mean: {group}/{metric}")
            mean[metric] = actual
        aggregates[group] = dict(episodes=len(samples), means=mean)
    require(set(aggregates) == set(grouped), "Missing aggregate row")
    probe_budget = {}
    for name, info in evaluation["probe_info"].items():
        probe_budget[name] = {k: info[k] for k in ("optimization_steps", "hidden_layers", "label_budget") if k in info}
    signature = dict(calibration_train_labels=evaluation["calibration_train_labels"],
                     calibration_val_labels=evaluation["calibration_val_labels"],
                     probe_budgets=probe_budget, test_episode_modes=sorted(test_modes.items()),
                     groups=[list(x) for x in sorted(grouped)])
    return aggregates, signature


def inspect_run(path, run, expected_steps):
    needed = ("summary.json", "history.jsonl", "evaluation/metrics.json")
    missing = [name for name in needed if not (path/name).is_file()]
    if missing:
        raise IncompleteRun(f"Missing completion files: {missing}")
    key = identity(run)
    require(int(run["args"]["steps"]) == expected_steps, f"Configured steps do not match requested budget: {path}")
    require(run["config"]["benchmark"]["kind"] == key[0], "Model config/identity mismatch")
    require(run["precision"] == "float32", "Formal matrix expects the fixed FP32 protocol")
    require(not run["benchmark_only"] and not run["args"]["benchmark"], "Timing-only benchmark is not a training result")
    digest = run["dataset_sha256"]
    require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Invalid recorded dataset hash")
    summary, evaluation = read(path/"summary.json"), read(path/"evaluation/metrics.json")
    require(recorded_paths_match(evaluation["data"], run["args"]["data"]),
            f"Evaluation dataset path does not match training data: {path}")
    require(recorded_paths_match(evaluation["checkpoint"], posixpath.join(run["args"]["output"], "best.pt")),
            f"Evaluation checkpoint path does not match this run's best.pt: {path}")
    if int(summary["steps"]) < expected_steps:
        raise IncompleteRun(f"Summary has {summary['steps']}/{expected_steps} steps")
    require(summary["steps"] == expected_steps and summary["model"] == key[0], "Summary identity/budget mismatch")
    require(summary["batch_size"] == run["args"]["batch_size"] and summary["device"] == run["device"], "Summary protocol mismatch")
    history = inspect_history(path/"history.jsonl", run, summary, expected_steps)
    groups, evaluation_signature = inspect_evaluation(evaluation, key, history["best_checkpoint_step"])
    common_args = {k: v for k, v in run["args"].items() if k not in {"data", "output", "model", "seed", "device", "warmup"}}
    data_recipe = json.loads(run["dataset_attributes"]["config_json"])
    common = dict(training_args=common_args, data_recipe=data_recipe, upstream=run["upstream"],
                  precision=run["precision"], scheduler=run["scheduler"],
                  gradient_clip_norm=run["gradient_clip_norm"], checkpoint_rule=run["checkpoint_rule"],
                  train_windows=run["train_windows"], validation_windows_used=run["validation_windows_used"],
                  training_device_family=run["device"].split(":")[0], torch_version=run["torch_version"],
                  python_version=run["python"], machine=run["machine"], evaluation=evaluation_signature)
    hashes = {name: file_hash(path/name) for name in ("run.json", *needed)}
    metadata = dict(run_directory=str(path.resolve()), model=key[0], damping=key[1], seed=key[2],
                    dataset_sha256=digest, actual_training_steps=history["actual_training_steps"],
                    best_checkpoint_step=history["best_checkpoint_step"], input_file_sha256=hashes,
                    recorded_evaluation_data=evaluation["data"], recorded_evaluation_checkpoint=evaluation["checkpoint"],
                    training_internal_seconds=finite(summary["wall_seconds"], "training time"),
                    evaluation_internal_seconds=finite(evaluation["total_seconds"], "evaluation time"),
                    parameters=run["parameters"], readout_calibration_train_labels=evaluation["calibration_train_labels"],
                    readout_calibration_val_labels=evaluation["calibration_val_labels"])
    return dict(key=key, run=run, groups=groups, protocol=common, metadata=metadata)


def aggregate_matrix(runs_root, seeds, steps=28140, allow_partial=False):
    require(runs_root.is_dir(), f"Runs root not found: {runs_root}")
    require(len(seeds) in (1, 3) and len(set(seeds)) == len(seeds), "Specify one or three distinct expected seeds")
    require(all(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in seeds),
            "Expected seeds must be nonnegative integers")
    require(steps > 0, "Expected steps must be positive")
    seeds = sorted(seeds)
    expected = set(itertools.product(MODELS, DAMPINGS, seeds))
    discovered, complete, incomplete = {}, {}, []
    for file in sorted(runs_root.rglob("run.json")):
        run = read(file)
        key = identity(run)
        require(key in expected, f"Unexpected run {label(key)} under runs root; select the exact experiment directory")
        require(key not in discovered, f"Duplicate run for {label(key)}; do not select the best duplicate after testing")
        discovered[key] = file.parent
        # Configuration mismatches must not be hidden as merely incomplete.
        require(int(run["args"]["steps"]) == steps, f"Expected {steps} steps but {file} configures {run['args']['steps']}")
        try:
            complete[key] = inspect_run(file.parent, run, steps)
        except IncompleteRun as exc:
            incomplete.append(dict(model=key[0], damping=key[1], seed=key[2], reason=str(exc), run_directory=str(file.parent)))
    absent = [dict(model=m, damping=b, seed=s, reason="No run.json found") for m, b, s in sorted(expected-set(discovered))]
    if (incomplete or absent) and not allow_partial:
        raise IncompleteRun(f"Only {len(complete)}/{len(expected)} runs complete; missing/incomplete: {incomplete+absent}. Use --allow-partial only for an explicitly partial report.")
    protocol, architecture_by_model, hashes_by_damping = None, {}, {}
    for key, result in sorted(complete.items()):
        if protocol is None:
            protocol = result["protocol"]
        require(result["protocol"] == protocol, f"Data/source/training/readout/test/backend protocol mismatch: {label(key)}")
        model, b, _ = key
        arch = dict(config=result["run"]["config"], parameters=result["run"]["parameters"],
                    trainable_parameters=result["run"]["trainable_parameters"])
        require(model not in architecture_by_model or architecture_by_model[model] == arch,
                f"Model architecture changes across seed/damping: {label(key)}")
        architecture_by_model[model] = arch
        digest = result["run"]["dataset_sha256"]
        require(b not in hashes_by_damping or hashes_by_damping[b] == digest, f"Dataset hash differs within b={b}")
        hashes_by_damping[b] = digest
    rows, seed_rows = [], []
    for model, damping, probe, mode, horizon in itertools.product(MODELS, DAMPINGS, PROBES, MODES, HORIZONS):
        group = (probe, mode, horizon)
        available = [(seed, complete[(model, damping, seed)]["groups"][group])
                     for seed in seeds if (model, damping, seed) in complete]
        if not available:
            continue
        metrics = set(available[0][1]["means"])
        require(all(set(g["means"]) == metrics for _, g in available), "Metric keys differ across training seeds")
        values = {}
        for metric in sorted(metrics):
            numbers = [g["means"][metric] for _, g in available]
            values[metric] = dict(mean=statistics.mean(numbers), sample_sd=statistics.stdev(numbers) if len(numbers)>1 else None)
        status = "COMPLETE" if len(available) == len(seeds) else "PARTIAL"
        rows.append(dict(model=model, damping=damping, probe=probe, action_mode=mode, horizon_s=horizon,
                         status=status, expected_seeds=seeds, completed_seeds=[s for s, _ in available],
                         training_seed_count=len(available),
                         episodes_per_seed={str(s): g["episodes"] for s, g in available}, metrics=values))
        for seed, g in available:
            seed_rows.append(dict(model=model, damping=damping, seed=seed, probe=probe, action_mode=mode,
                                  horizon_s=horizon, test_episodes=g["episodes"], **g["means"]))
    metadata = [result["metadata"] for _, result in sorted(complete.items())]
    return dict(schema_version=1, status="COMPLETE" if len(complete)==len(expected) else "PARTIAL",
                runs_root=str(runs_root.resolve()), expected_training_steps=steps, expected_seeds=seeds,
                expected_run_count=len(expected), discovered_run_count=len(discovered), completed_run_count=len(complete),
                absent_runs=absent, incomplete_runs=incomplete,
                protocol_verification=dict(matched_completed_runs=bool(complete), checked_completed_count=len(complete),
                                           shared_protocol=protocol, architecture_by_model=architecture_by_model,
                                           recorded_dataset_hash_by_damping={str(k): v for k, v in hashes_by_damping.items()},
                                           boundary="Checks recorded provenance and hashes of result input files; does not reopen dataset or checkpoint binaries. Original evaluation metadata does not record its device, so evaluation backend equality cannot be proven here."),
                runs=metadata, aggregate=rows, seed_scores=seed_rows,
                total_completed_training_internal_seconds=sum(m["training_internal_seconds"] for m in metadata),
                total_completed_evaluation_internal_seconds=sum(m["evaluation_internal_seconds"] for m in metadata),
                statistical_contract="Each run's test-episode mean is one training-seed observation. Seeds equally weighted. sample_sd is sample SD (ddof=1), null with one seed; no episode-level SD used as training-seed uncertainty. No confidence interval or significance claim.",
                interpretation="Completion means requested steps and evaluation finished, not convergence, physical guarantees or paper reproduction. Partial reports are progress snapshots; compare completed seed sets before interpreting method differences.")


def write_csv(path, rows):
    if rows:
        fields = list(rows[0]) + sorted(set().union(*(row.keys() for row in rows))-set(rows[0]))
    else:
        fields = ["model", "damping", "probe", "action_mode", "horizon_s", "training_seed_count"]
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_outputs(result, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/"matrix_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    csv_rows = []
    for row in result["aggregate"]:
        flat = {k: row[k] for k in ("model", "damping", "probe", "action_mode", "horizon_s", "status", "training_seed_count")}
        flat["completed_seeds"] = ",".join(map(str, row["completed_seeds"]))
        flat["expected_seeds"] = ",".join(map(str, row["expected_seeds"]))
        for metric, data in row["metrics"].items():
            flat[metric+"_mean"] = data["mean"]
            flat[metric+"_sample_sd"] = data["sample_sd"]
        csv_rows.append(flat)
    write_csv(output/"matrix_scores.csv", csv_rows)
    write_csv(output/"matrix_seed_scores.csv", result["seed_scores"])
    lines = ["# LeHami / LeWM 训练矩阵汇总", "",
             f"状态：**{result['status']}**；完成 **{result['completed_run_count']}/{result['expected_run_count']}** 组。预期种子：`{result['expected_seeds']}`；每组要求{result['expected_training_steps']}个优化步。", "",
             "数据/上游源码/模型结构、训练和读出预算、实际测试轨迹按已保存记录核对；同一阻尼的文件hash必须一致，模型配置在种子及阻尼间保持不变。缺失或不完整组不纳入均值。记录不匹配会报错，不会自动舍弃异常结果。", "",
             "**统计单位是独立训练种子。** 每个种子先对测试轨迹取均值，再跨种子等权汇总。SD是训练种子间样本标准差（ddof=1）；单种子SD为null，不填0。episode标准差不作为多种子误差条。", "",
             "## 8秒预测：预设MLP读出", "", "| 模型 | b | 输入 | 完成种子 | 角度RMSE(rad)，均值±SD | 角速度RMSE(rad/s)，均值±SD | 状态 |",
             "|---|---:|---|---|---:|---:|---|"]
    def value(cell):
        return f"{cell['mean']:.4f}" + ("（单种子）" if cell["sample_sd"] is None else f" ± {cell['sample_sd']:.4f}")
    for row in result["aggregate"]:
        if row["probe"] == "mlp" and row["horizon_s"] == 8:
            lines.append(f"| {row['model']} | {row['damping']:g} | {row['action_mode']} | {row['completed_seeds']} | {value(row['metrics']['angle_rmse_rad'])} | {value(row['metrics']['omega_rmse_rad_s'])} | {row['status']} |")
    lines += ["", "完整ridge/MLP、自由/受驱、4/8/16/32秒和能量指标见matrix_scores.csv；逐种子均值见matrix_seed_scores.csv。JSON保留每组实际训练步数、验证选出的checkpoint、输入文件hash和完整核对记录。", "",
              f"已完成组的内部训练总计{result['total_completed_training_internal_seconds']/3600:.3f}小时、内部评测{result['total_completed_evaluation_internal_seconds']/3600:.3f}小时；不含缺失组、排队和进程计时范围之外的开销。", "",
              "达到步数和评测完成不等于收敛。正式解释前检查无阻尼正控、验证曲线、初始状态读出误差和长时预测；不能单凭能量下降判定物理正确。评测原始metadata未记录评测device，本汇总无法独立证明各评测使用相同后端，需结合执行日志确认。"]
    if result["status"] == "PARTIAL":
        lines += ["", "## 未完成项", "", "这是显式允许的部分汇总，不能把完成种子不同的两组当作配对且同预算的最终对照。"]
        for row in result["absent_runs"] + result["incomplete_runs"]:
            lines.append(f"- {row['model']} / b={row['damping']} / seed={row['seed']}: {row['reason']}")
    (output/"matrix_summary.md").write_text("\n".join(lines)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True, help="One or three explicitly expected training seeds")
    p.add_argument("--steps", type=int, default=28140)
    p.add_argument("--allow-partial", action="store_true", help="Explicit progress snapshot; mismatched/duplicate runs still fail")
    a = p.parse_args()
    result = aggregate_matrix(a.runs_root, a.seeds, a.steps, a.allow_partial)
    write_outputs(result, a.output)
    print(json.dumps({"status": result["status"], "completed_runs": result["completed_run_count"],
                      "expected_runs": result["expected_run_count"], "output": str(a.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
