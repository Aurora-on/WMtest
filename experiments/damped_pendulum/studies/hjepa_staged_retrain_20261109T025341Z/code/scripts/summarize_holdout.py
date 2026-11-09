"""Verify downloaded independent-test predictions and write a concise report."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from dpbench.evaluate_holdout import tensor_digest
from dpbench.holdout import save_json, sha256, verify_frozen


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def summarize(study, results, output, old_scores):
    study, results, output = map(Path, (study, results, output))
    frozen = json.loads((study / "frozen_manifest.json").read_text())
    audit = json.loads((study / "audit.json").read_text())
    completion = json.loads((results / "completion.json").read_text())
    runtime = json.loads((results / "evaluation_protocol.json").read_text())
    assert completion["status"] == "COMPLETE" and completion["runs"] == 12
    assert runtime["requested_batch_size"] == 1
    assert runtime["protocol_sha256"] == sha256(study / "protocol.json")
    assert runtime["audit_sha256"] == sha256(study / "audit.json")
    assert runtime["runner_sha256"] == sha256(Path(__file__).resolve().parents[1] / "dpbench/evaluate_holdout.py")
    verify_frozen(study, frozen)
    datasets = {}
    for entry in audit["datasets"]:
        path = study / entry["file"]
        assert sha256(path) == entry["sha256"]
        with h5py.File(path) as f:
            n, frames = len(f["ep_len"]), int(f["ep_len"][0])
            datasets[entry["damping"]] = dict(state=f["state"][:].reshape(n, frames, 2),
                                               mode=f["mode"][:], sha256=entry["sha256"])
    per_seed, runs = [], []
    for expected in frozen["runs"]:
        folder = results / expected["run"]
        run = json.loads((folder / "metrics.json").read_text())
        for key in ("run", "model", "damping", "seed", "checkpoint_step", "training_data_sha256"):
            assert run[key] == expected[key], (folder, key)
        assert run["test_data_sha256"] == datasets[run["damping"]]["sha256"]
        source = study / "frozen/runs/server_47111" / expected["run"]
        checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
        readouts = torch.load(source / "evaluation/readouts.pt", map_location="cpu", weights_only=True)
        assert tensor_digest(checkpoint["state_dict"]) == run["frozen_model_digest"]
        assert tensor_digest(readouts) == run["frozen_readouts_digest"]
        assert run["parameters_and_buffers_unchanged"] and not run["readouts_refitted"]
        assert run["batch_check"]["actual_batch"] == 1
        assert run["canary"]["steps"] == 320 and len(run["canary"]["episodes"]) == 2
        assert run["canary"]["predictions_max_abs_difference"] == run["canary"]["features_max_abs_difference"] == 0
        with np.load(folder / "predictions.npz") as saved:
            ids, names = saved["episode_ids"], saved["probe_names"].tolist()
            predictions, truth = saved["predictions"], saved["truth"]
            assert predictions.shape == (128, 2, 321, 2) and names == ["ridge", "mlp"]
            np.testing.assert_array_equal(ids, audit["paired_splits"]["test_all"])
            np.testing.assert_array_equal(truth, datasets[run["damping"]]["state"][ids, 4:325])
            assert np.isfinite(predictions).all() and np.isfinite(truth).all()
            error = predictions - truth[:, None]
            error[..., 0] = np.arctan2(np.sin(error[..., 0]), np.cos(error[..., 0]))
            for row in run["episodes"]:
                ep, p, steps = row["episode"], names.index(row["probe"]), round(row["horizon_s"] / float(saved["dt"]))
                assert row["mode"] == ("passive" if datasets[run["damping"]]["mode"][ep] == 0 else "driven")
                expected_scores = np.sqrt(np.mean(error[ep, p, 1:steps + 1] ** 2, axis=0))
                np.testing.assert_allclose([row["angle_rmse_rad"], row["omega_rmse_rad_s"]],
                                           expected_scores, atol=1e-10, rtol=1e-10)
        assert len(run["episodes"]) == 128 * 2 * 4
        assert len(run["aggregate"]) == 3 * 2 * 2 * 4
        for cell in run["aggregate"]:
            selected = [r for r in run["episodes"] if r["episode"] in audit["paired_splits"][cell["split"]]
                        and (r["probe"], r["mode"], r["horizon_s"]) == (cell["probe"], cell["mode"], cell["horizon_s"])]
            assert len(selected) == cell["episodes"]
            for key, value in cell["mean"].items():
                np.testing.assert_allclose(value, np.mean([r[key] for r in selected]), rtol=1e-12, atol=1e-12)
            per_seed.append(dict(model=run["model"], damping=run["damping"], seed=run["seed"],
                                 **{k: cell[k] for k in ("split", "probe", "mode", "horizon_s", "episodes")}, **cell["mean"]))
        runs.append(run)
    groups = {}
    keys = ("split", "model", "damping", "probe", "mode", "horizon_s")
    for row in per_seed:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    assert len(groups) == 192
    scores = []
    for group, rows in sorted(groups.items()):
        assert sorted(r["seed"] for r in rows) == [0, 1, 2]
        assert len({r["episodes"] for r in rows}) == 1
        cell = dict(zip(keys, group), episodes=rows[0]["episodes"], training_seeds=3)
        for key in rows[0]:
            if key not in (*keys, "seed", "episodes"):
                values = [r[key] for r in rows]
                cell[key + "_mean"] = float(np.mean(values))
                cell[key + "_sd"] = float(np.std(values, ddof=1))
        scores.append(cell)
    lookup = {tuple(r[k] for k in keys): r for r in scores}
    def cell(model, b, mode, horizon=8., split="test", probe="mlp"):
        return lookup[(split, model, b, probe, mode, horizon)]
    def fmt(c, metric="angle_rmse_rad"):
        return f"{c[metric + '_mean']:.4f} ± {c[metric + '_sd']:.4f}"
    def number(c, metric="angle_rmse_rad"):
        return f"{c[metric + '_mean']:.4f}"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "scores.csv", scores)
    write_csv(output / "seed_scores.csv", per_seed)
    summary = dict(status="COMPLETE", completed_runs=12, runtime=runtime, completion=completion,
                   verification=dict(all_frozen_checkpoint_and_readout_digests_match=True,
                                     all_new_data_hashes_match=True, scores_recomputed_from_saved_predictions=True,
                                     all_12_real_checkpoint_canaries_passed=True, no_parameter_or_readout_updates=True),
                   score_definition="Mean of per-episode time-RMSE; equal mean across three training seeds; sample SD across seeds (ddof=1).",
                   scores=scores)
    save_json(output / "summary.json", summary)
    text = ["# 新独立测试集：冻结模型评测", "",
            "**12/12 组已完成。** LeHami / LeWM × 两种阻尼 × 三个训练种子。使用原检查点及原 ridge、MLP 读出器，未重新训练或拟合。", "",
            "正式测试集每种阻尼 109 条（无外力 45、受控 64）。两模型均只用最初 5 帧及控制输入，从相同时间点自主预测；后续真实图像和物理状态不进入预测。", "",
            "## 8 秒预测：与上一份报告相同的 MLP 读出", "",
            "数值越小越好。先算每条轨迹的时间 RMSE，再对轨迹平均，最后汇总三个训练种子；± 为种子间样本标准差，不是置信区间。", "",
            "| 模型 | 阻尼 b | 输入 | 角度 RMSE（rad） | 角速度 RMSE（rad/s） |",
            "|---|---:|---|---:|---:|"]
    for model in ("lehami", "lewm"):
        for b in (0., .2):
            for mode in ("passive", "driven"):
                c = cell(model, b, mode)
                text.append(f"| {'LeHami' if model == 'lehami' else 'LeWM'} | {b:g} | {'无外力' if mode == 'passive' else '受控'} | {fmt(c)} | {fmt(c, 'omega_rmse_rad_s')} |")
    text += ["", "## 32 秒长时预测：角度 RMSE（rad），MLP 读出", "",
             "| 模型 | 阻尼 b | 无外力 | 受控 |", "|---|---:|---:|---:|"]
    for model in ("lehami", "lewm"):
        for b in (0., .2):
            text.append(f"| {'LeHami' if model == 'lehami' else 'LeWM'} | {b:g} | {fmt(cell(model,b,'passive',32.))} | {fmt(cell(model,b,'driven',32.))} |")
    old = list(csv.DictReader(Path(old_scores).open()))
    old = {(r["model"], float(r["damping"]), r["action_mode"]): float(r["angle_rmse_rad_mean"]) for r in old if r["probe"] == "mlp" and float(r["horizon_s"]) == 8.}
    original_scores = {}
    for run in frozen["runs"]:
        original = json.loads((study / "frozen/runs/server_47111" / run["run"] / "evaluation/metrics.json").read_text())
        for row in original["aggregate"]:
            if row["probe"] == "mlp" and row["horizon_s"] == 8.:
                original_scores.setdefault((run["model"], run["damping"], row["mode"]), []).append(row["mean"]["angle_rmse_rad"])
    for key, values in original_scores.items():
        assert len(values) == 3
        np.testing.assert_allclose(old[key], np.mean(values), rtol=1e-12, atol=1e-12)
    text += ["", "## 去重后的对照：8 秒角度 RMSE 均值（rad），MLP 读出", "",
             "| 模型 | b | 输入 | 旧测试集 | 全部新样本 | 正式去重集 | 严格子集 |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for model in ("lehami", "lewm"):
        for b in (0., .2):
            for mode in ("passive", "driven"):
                text.append(f"| {'LeHami' if model == 'lehami' else 'LeWM'} | {b:g} | {'无外力' if mode == 'passive' else '受控'} | {old[model,b,mode]:.4f} | {number(cell(model,b,mode,split='test_all'))} | {number(cell(model,b,mode))} | {number(cell(model,b,mode,split='test_strict_context'))} |")
    text += ["", "全部新样本每条件 128 条（64/64）；正式集 109 条（45/64）；严格子集 16 条（7/9），初始 4/5 帧图像也无精确重复。严格子集初始能量较高、样本较少；这些分区难度不同，分数变化不能全部归因于泄漏或去重。旧、新样本也不是配对样本。", "",
             "## 可验证的评测检查", "",
             "- 12 个模型、原读出器和新数据哈希均一致；模型参数、缓冲区及读出器在运行前后未变。新入口没有拟合或优化步骤。",
             "- 对每个正式模型各取一条无外力、一条受控轨迹，替换初始化后的全部真实图像及全部物理标签，再预测完整 320 步：潜变量和物理预测的最大差均为 0。这一干预检查覆盖每组两条轨迹，不代表逐条穷举全部测试样本。",
             "- 发现批量与逐条预测存在长时差异后，正式 12 组统一使用 batch=1，与旧评测一致；先前批量尝试未纳入本报告。",
             "- 下载后的全部逐轨迹角度、角速度 RMSE 已由保存的预测和新数据真值重新计算，并核对三个种子的汇总。",
             f"- 运行设备：{runtime['gpu']}；正式模型评测内部总计 {completion['total_run_seconds']/60:.2f} 分钟，其中推演及读出 {completion['inference_seconds']/60:.2f} 分钟；不含排队、传输及前期检查。", "",
             "完整 ridge/MLP、三个分区、4/8/16/32 秒、初始读出误差与能量指标见 [scores.csv](scores.csv)；逐种子结果见 [seed_scores.csv](seed_scores.csv)，完整核查见 [summary.json](summary.json)。", "",
             "这些成绩评价的是当前训练配置下的模型与原读出器组合，不等于论文原任务的复现。精确去重不排除近似样本；参考能量收支误差也不是模型内部结构守恒的证明。"]
    if all(cell("lewm", b, mode, split=split)["angle_rmse_rad_mean"] < cell("lehami", b, mode, split=split)["angle_rmse_rad_mean"]
           for b in (0., .2) for mode in ("passive", "driven") for split in ("test", "test_strict_context")):
        text[6:6] = ["**主要结果：LeWM 在正式测试集的四类条件中，8 秒角度误差均更低；严格子集上的比较顺序相同。** 已发现的精确片段重复不足以单独解释两模型的全部差距。严格子集只有 16 条，仍需保留这一证据范围。", ""]
    with (output / "report.md").open("x") as f:
        f.write("\n".join(text) + "\n")
    print(json.dumps({"status": "COMPLETE", "cells": len(scores), "report": str(output / "report.md")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--old-scores", required=True)
    args = p.parse_args()
    summarize(args.study, args.results, args.output, args.old_scores)
