"""Export measured run costs and physical scores, without treating pilot runs as final results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    costs, scores = [], []
    for file in sorted(a.runs.glob("*/summary.json")):
        run = json.loads((file.parent/"run.json").read_text())
        cost = json.loads(file.read_text())
        evaluation = json.loads((file.parent/"evaluation/metrics.json").read_text())
        b = float(run["dataset_attributes"]["damping"])
        cost.update(run=file.parent.name, damping=b, seed=run["args"]["seed"], evaluation_seconds=evaluation["total_seconds"])
        costs.append(cost)
        for row in evaluation["aggregate"]:
            scores.append(dict(run=file.parent.name, model=cost["model"], damping=b,
                               seed=cost["seed"], training_steps=cost["steps"],
                               probe=row["probe"], action_mode=row["mode"], horizon_s=row["horizon_s"],
                               test_episodes=row["episodes"], **row["mean"]))
    if not costs:
        raise ValueError("No completed evaluated runs found")
    if any(c["steps"] != 500 for c in costs):
        raise ValueError("This narrative template is only for the 500-step pilot. Other runs retain full metrics.json; use a separate report before interpreting convergence.")
    fields = sorted(set().union(*(x.keys() for x in scores)))
    with (a.output/"scores.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader(); writer.writerows(scores)
    budget = {}
    for kind in ("lehami", "lewm"):
        selected = [x for x in costs if x["model"] == kind]
        budget[kind] = dict(mean_step_seconds=float(np.mean([x["step_seconds_mean"] for x in selected])),
                            mean_evaluation_seconds=float(np.mean([x["evaluation_seconds"] for x in selected])))
    compute5000 = sum(x["mean_step_seconds"] for x in budget.values())*5000*2*3/3600
    compute28140 = sum(x["mean_step_seconds"] for x in budget.values())*28140*2*3/3600
    data = dict(costs=costs, measured_training_seconds=sum(x["wall_seconds"] for x in costs),
                measured_evaluation_seconds=sum(x["evaluation_seconds"] for x in costs),
                mean_model_cost=budget, estimated_compute_hours_12_runs_5000_steps=compute5000,
                estimated_compute_hours_12_runs_35_epochs=compute28140,
                estimate_scope="Extrapolation of local measured steps only; excludes future validation, checkpointing, thermal drift, queue, setup; 35epochs=28140steps only for this dataset/batch.")
    (a.output/"costs.json").write_text(json.dumps(data, indent=2))
    lines = ["# 500步基础试跑：实测成本和性能", "", "日期：2026-10-28。四组、一个训练种子，各500步。该预算不足一个epoch，结果只作流程与初期优化诊断，不是收敛比较，也不是论文复现。", "", "## 实测耗时", "", "| 模型 | 阻尼b | 训练秒（含验证/保存） | 平均训练步秒 | 评测秒 |", "|---|---:|---:|---:|---:|"]
    for c in costs:
        lines.append(f"| {c['model']} | {c['damping']} | {c['wall_seconds']:.1f} | {c['step_seconds_mean']:.3f} | {c['evaluation_seconds']:.1f} |")
    lines += ["", f"总训练 {data['measured_training_seconds']/60:.2f} 分钟，总评测 {data['measured_evaluation_seconds']/60:.2f} 分钟；不含安装、源码获取和测试。", "", "## 8秒自主预测：预设MLP读出", "", "每类32条未见测试轨迹；先计算每条轨迹的RMSE，再对轨迹平均。校准只用train/val且不超过8秒训练时域。", "", "| 模型 | b | 输入 | 角度RMSE(rad) | 角速度RMSE(rad/s) | 初始角度读出误差(rad) |", "|---|---:|---|---:|---:|---:|"]
    for r in scores:
        if r["probe"] == "mlp" and r["horizon_s"] == 8:
            lines.append(f"| {r['model']} | {r['damping']} | {r['action_mode']} | {r['angle_rmse_rad']:.3f} | {r['omega_rmse_rad_s']:.3f} | {r['anchor_angle_error_rad']:.3f} |")
    lines += ["", "完整的ridge/MLP、4/8/16/32秒结果在同目录 `scores.csv`。单种子episode离散度不能代替多种子不确定性。", "", "## 结果边界与下一步", "", "LeHami在无阻尼正控中的长时误差仍高，因此这次不能把其阻尼结果解释成纯粹由保守结构造成的失败。当前应优先延长两者训练，检查验证曲线、初始状态读出和无阻尼正控，再做至少三种子比较。只看能量是否下降也不够：轨迹错误或预测静止可获得虚假的单调性。", "", "## 下一轮预算外推", "", f"按本次完整训练步估计，2模型×2阻尼×3种子，每次5000步的纯训练约 **{compute5000:.2f}小时**；相同矩阵若每次35epoch（本数据28140步），约 **{compute28140:.1f}小时**。需另计验证、保存、评测和持续负载下的速度变化；这不是已执行的正式训练。", "", "5000步约6.2epoch，也不能预先保证收敛。35epoch只是预算示例，并非声称等同作者原任务的训练量。", "", "## 本地还是服务器", "", "**本轮基础实验在本地完成；后续先在本地扩大单种子预算。** M5/16GB已证实可运行本配置。若进入约35epoch、三种子和额外模型矩阵，建议服务器单GPU，以避免长时间占用笔记本。当前服务器连接超时，CUDA速度、空闲和费用未测，不给虚构加速比。", "", "MPS的driver allocated读数约2.6–2.9GB，是记录时的分配量，不能称完整任务峰值显存；进程RSS另列于costs.json，统一内存不能简单相加。", "", "代码/操作说明见项目README；相关工作与服务器证据分别见related_work_2026-10-28.md、server_assessment_2026-10-28.md。"]
    (a.output/"experiment_report_2026-10-28.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k: v for k, v in data.items() if k != "costs"}, indent=2))


if __name__ == "__main__":
    main()
