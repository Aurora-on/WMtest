#!/usr/bin/env python3
"""Render aggregate report figures from the sealed evaluation artifacts only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
REVIEW = PROJECT / "studies/staged_damping_d2_launch_20261106T064931Z/review_20261108"
RUN = REVIEW / "run"
INK, MUTED, GRID = "#18354e", "#647a91", "#dce4ed"
ORANGE, BLUE, GREEN, PURPLE = "#d97632", "#24758f", "#498c6c", "#8462a7"
BG = "#f7f9fc"
DAMPING = [0, .05, .1, .2, .5, 1]
STAGES = {"a": "undamped_step20000", "single": "damping_single_step20000", "multi": "damping_multi_step20000"}
plt.rcParams.update({"font.family": "Arial Unicode MS", "axes.unicode_minus": False,
                     "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11,
                     "savefig.facecolor": "white", "axes.facecolor": "white"})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data():
    data, sources = {}, {}
    for arm, stage in STAGES.items():
        root = RUN / "milestones" / stage / "evaluation"
        summary_path = root / "summary.json"
        source = root / "val/trajectories.npz"
        complete_path = root / "complete.json"
        summary = json.loads(summary_path.read_text())
        complete = json.loads(complete_path.read_text())
        assert digest(source) == complete["artifacts"]["val/trajectories.npz"]
        assert summary["checkpoint_step"] == 20000 and summary["q_dim"] == 2
        assert summary["parameters_unchanged"] and not summary["test_split_accessed"]
        with np.load(source, allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        assert np.all(arrays["valid_steps"] == 32)
        np.testing.assert_allclose(arrays["time"], np.arange(33) * .1, atol=1e-12)
        for key in ["truth", "pred", "teacher"]:
            assert arrays[key].shape[1:] == (33, 2) and np.isfinite(arrays[key]).all()
        data[arm] = (arrays, summary)
        sources[arm] = {"npz": str(source), "npz_sha256": digest(source),
                        "summary": str(summary_path), "summary_sha256": digest(summary_path),
                        "completion_manifest_sha256": digest(complete_path)}
    # Shared observation-based readout has exactly the same values at matched conditions.
    for arm, b in [("a", 0), ("single", .2)]:
        z, _ = data[arm]
        zm, _ = data["multi"]
        mi = np.flatnonzero(np.isclose(zm["damping"], b))
        assert np.array_equal(z["group_id"], zm["group_id"][mi])
        np.testing.assert_array_equal(z["truth"], zm["truth"][mi])
        np.testing.assert_array_equal(z["teacher"], zm["teacher"][mi])
    return data, sources


def metrics_for(data, arm, b):
    z, summary = data[arm]
    idx = np.flatnonzero(np.isclose(z["damping"], b))
    assert len(idx) == len(np.unique(z["group_id"][idx])) == 64
    result = {"arm": arm, "damping": b, "n_groups": 64, "time": z["time"][1:].tolist()}
    recorded = summary["evaluation"]["val"]["rollout"]["32"]["groups"][f"damping_{b:g}"]
    for mode, saved_key in [("pred", "metrics"), ("teacher", "teacher_metrics")]:
        delta = z[mode][idx, 1:] - z["truth"][idx, 1:]
        angle = np.arctan2(np.sin(delta[..., 0]), np.cos(delta[..., 0]))
        aggregate = {"angle_mae_rad": float(np.mean(np.abs(angle))),
                     "omega_rmse_rad_s": float(np.sqrt(np.mean(delta[..., 1] ** 2)))}
        for key, value in aggregate.items():
            np.testing.assert_allclose(value, recorded[saved_key][key], rtol=1e-12, atol=1e-12)
        result[mode] = dict(aggregate=aggregate,
                            timewise_angle_mae_rad=np.mean(np.abs(angle), axis=0).tolist(),
                            timewise_omega_rmse_rad_s=np.sqrt(np.mean(delta[..., 1] ** 2, axis=0)).tolist())
    return result


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b8c7d7")
    ax.grid(axis="y", color=GRID, lw=.8, zorder=0)
    ax.tick_params(length=3, color="#b8c7d7")
    ax.set_axisbelow(True)


def title(fig, heading, subtitle):
    fig.text(.075, .945, heading, fontsize=20)
    fig.text(.075, .893, subtitle, fontsize=11.5, color=MUTED)


def aggregate_figure(multi, a, single):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.8), dpi=160)
    fig.subplots_adjust(left=.075, right=.975, top=.73, bottom=.25, wspace=.24)
    title(fig, "固定 20000 步检查点 · 不同阻尼工况的总体误差",
          "每个工况 64 个验证初态组；汇总未来 0.1–3.2 秒（32 帧），排除预测起点")
    handles = [Line2D([], [], c=ORANGE, marker="o", lw=2.2, label="B2 多工况：自主预测"),
               Line2D([], [], c=BLUE, marker="o", mfc="white", ls="--", lw=1.8, label="真实历史读出（共用）"),
               Line2D([], [], c=GREEN, marker="D", ls="none", ms=8, label="A 无阻尼：自主预测"),
               Line2D([], [], c=PURPLE, marker="s", ls="none", ms=8, label="B1 单工况：自主预测")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.067, .85), ncol=2,
               frameon=False, columnspacing=2.0, handlelength=2.3, fontsize=10.5)
    specs = [("angle_mae_rad", "角度 MAE（rad）", .19),
             ("omega_rmse_rad_s", "角速度 RMSE（rad/s）", .55)]
    x = np.arange(6)
    for ax, (key, ylabel, ymax) in zip(axes, specs):
        pred = [v["pred"]["aggregate"][key] for v in multi]
        teacher = [v["teacher"]["aggregate"][key] for v in multi]
        ax.plot(x, pred, color=ORANGE, marker="o", ms=6.5, lw=2.3, zorder=4)
        ax.plot(x, teacher, color=BLUE, marker="o", mfc="white", ms=6, ls="--", lw=1.8, zorder=4)
        for xx, yy in zip(x, pred):
            ax.annotate(f"{yy:.3f}", (xx, yy), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9.5, color=ORANGE)
        for xx, yy in zip(x, teacher):
            label_y = 8 if key == "angle_mae_rad" and yy < .02 else -16
            ax.annotate(f"{yy:.3f}", (xx, yy), xytext=(0, label_y), textcoords="offset points",
                        ha="center", fontsize=9.5, color=BLUE)
        for xx, series, c, marker, offset in [(0, a, GREEN, "D", (14, -4)),
                                               (3, single, PURPLE, "s", (13, -4))]:
            yy = series["pred"]["aggregate"][key]
            ax.scatter(xx, yy, color=c, marker=marker, s=70, edgecolors="white", linewidth=.9, zorder=5)
            ax.annotate(f"{yy:.3f}", (xx, yy), xytext=offset, textcoords="offset points",
                        fontsize=9.5, color=c)
        ax.set(xlim=(-.35, 5.4), ylim=(-.025 * ymax, ymax), ylabel=ylabel,
               xticks=x, xticklabels=[f"{b:g}" for b in DAMPING], xlabel="阻尼系数 b（工况等距排列）")
        tidy(ax)
    fig.text(.075, .145, "真实历史读出：每个时刻重新输入真实历史，经过同一固定读出；用于诊断编码与读出误差。", fontsize=10.3, color=MUTED)
    fig.text(.075, .105, "该诊断不是自主预测，也不构成自主预测误差的严格下界；A / B1 在对应工况的读出结果与 B2 完全一致。", fontsize=10.3, color=MUTED)
    fig.text(.075, .065, "数据范围：已见阻尼验证集，单个训练种子；各数值直接由已保存轨迹重算，并与评价 JSON 逐项一致。", fontsize=10.3, color=MUTED)
    fig.savefig(OUT / "summary_metrics.png")
    plt.close(fig)


def horizon_figure(a, single, multi):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6), dpi=160)
    fig.subplots_adjust(left=.075, right=.975, top=.73, bottom=.22, wspace=.24)
    title(fig, "自主预测误差随预测时间的变化", "每条曲线覆盖同一批 64 个验证初态组；各阶段均取精确第 20000 步检查点")
    series = [(a, GREEN, "A 无阻尼，b=0", "-"),
              (single, PURPLE, "B1 单工况，b=0.2", "-"),
              (multi[3], ORANGE, "B2 多工况，b=0.2", "-"),
              (multi[5], BLUE, "B2 多工况，b=1", "--")]
    handles = []
    for ax, key, ylabel in zip(axes, ["timewise_angle_mae_rad", "timewise_omega_rmse_rad_s"],
                               ["该时刻角度 MAE（rad）", "该时刻角速度 RMSE（rad/s）"]):
        for s, color, label, ls in series:
            line, = ax.plot(s["time"], s["pred"][key], color=color, lw=2.2, ls=ls, label=label)
            if ax is axes[0]:
                handles.append(line)
        ax.set(xlabel="相对预测起点的时间（秒）", ylabel=ylabel, xlim=(.1, 3.2),
               xticks=[.1, .8, 1.6, 2.4, 3.2], ylim=(0, None))
        tidy(ax)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.067, .85), ncol=2,
               frameon=False, columnspacing=2.4, fontsize=10.5)
    fig.text(.075, .12, "每个点只统计该预测时刻的 64 条轨迹，不是从起点累计至当前的平均误差；角度采用圆周距离。", fontsize=10.3, color=MUTED)
    fig.text(.075, .077, "真实历史为 21 帧（2 秒）；随后自主预测 32 步（3.2 秒）。全部轨迹完整，无平滑处理或样本筛选。", fontsize=10.3, color=MUTED)
    fig.savefig(OUT / "horizon_errors.png")
    plt.close(fig)


def box(ax, x, y, w, h, title_text, body, color=BLUE, fontsize=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=.015",
                              fc="white", ec="#cbd7e4", lw=1.2))
    ax.plot([x+.015, x+w-.015], [y+h-.07, y+h-.07], color=color, lw=3, solid_capstyle="round")
    ax.text(x+.018, y+h-.035, title_text, fontsize=12.7, color=color, va="center")
    ax.text(x+.018, y+h-.095, body, fontsize=fontsize, color=INK, va="top", linespacing=1.65)


def pipeline_figure(plan):
    assert plan["stage_a_total_steps"] == 20000 and plan["stage_b_steps_each"] == 20000
    assert plan["stage_b_launch"] == "parallel_after_A_exact_20000_checkpoint_and_readout"
    assert plan["configs"]["single"]["training"]["trainable_modules"] == ["damping"]
    assert plan["configs"]["multi"]["training"]["trainable_modules"] == ["damping", "environment"]
    fig, ax = plt.subplots(figsize=(12.4, 7.0), dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set(position=[0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.text(.045, .945, "二维分阶段训练 · 实际完成的流程", fontsize=21)
    fig.text(.045, .892, "q 与 p 各 2 维；A 封存后，两个阻尼分支并行训练，各自从 0 步到 20000 步", fontsize=11.5, color=MUTED)
    box(ax, .045, .53, .255, .285, "A  无阻尼学习", "b = 0，关闭阻尼\n从已有 10000 步继续训练\n累计到精确第 20000 步\n训练：编码器、初态、势能", GREEN)
    box(ax, .36, .53, .265, .285, "封存 A + 固定读出", "冻结编码器、初态、势能\n固定同一个 k=5 非线性读出\n仅在 A 的训练组上拟合/选 k\n两个分支共用同一封存结果", BLUE)
    box(ax, .70, .56, .255, .255, "B1  单工况阻尼", "b = 0.2\n只训练阻尼网络 D(q,v)\n新增训练 20000 步\n共享模块与读出保持冻结", PURPLE)
    box(ax, .70, .205, .255, .275, "B2  多工况阻尼", "b ∈ {0, .05, .1, .2, .5, 1}\n训练 D(q,v,c) 与环境编码器\nc 为 4 维历史条件特征\n新增 20000 步；其余保持冻结", ORANGE, fontsize=10.9)
    box(ax, .045, .205, .58, .235, "时间与评价设置", "采样间隔 0.1 秒；真实历史 21 帧 = 2 秒\n训练预测长度课程：4 → 8 → 16 → 32 步\n最终验证：每个已见阻尼工况 64 组，自主预测 3.2 秒", BLUE)
    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                    lw=1.6, color="#71869a", shrinkA=3, shrinkB=3))
    arrow((.308, .672), (.35, .672))
    ax.plot([.637, .665, .665], [.672, .672, .34], color="#71869a", lw=1.6)
    arrow((.665, .672), (.692, .672))
    arrow((.665, .34), (.692, .34))
    fig.text(.045, .114, "B1 / B2 独立初始化阻尼分支和优化器；A 的编码器、初态与势能参数哈希及共用读出均已核验。", fontsize=10.5, color=MUTED)
    fig.text(.045, .071, "非线性读出仅用于物理量评价；阻尼训练仍使用 LeHami 式潜空间监督。图中不包含未见阻尼或长时域测试。", fontsize=10.5, color=MUTED)
    fig.savefig(OUT / "pipeline.png", facecolor=BG)
    plt.close(fig)


def convergence_figure(history):
    assert history["block_size_updates"] == 250
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 7.0), dpi=160, sharey=True)
    fig.subplots_adjust(left=.075, right=.975, top=.715, bottom=.245, wspace=.16)
    title(fig, "训练过程 · 潜空间预测损失与验证分数",
          "训练曲线取连续 250 步的均值；验证曲线保留每次原始验证记录；纵轴采用统一对数刻度")
    panels = [("stage_a", "A  无阻尼", GREEN), ("stage_b_single", "B1  单工况阻尼", PURPLE),
              ("stage_b_multi", "B2  多工况阻尼", ORANGE)]
    handles = []
    for ax, (name, heading, heading_color) in zip(axes, panels):
        arm = history["arms"][name]
        blocks, vals = arm["blocks"], arm["validation"]
        assert arm["last_step"] == 20000
        assert all(b["n"] == 250 for b in blocks)
        assert sum(b["n"] for b in blocks) == arm["row_count"]
        tx = np.array([b["end_step"] for b in blocks])
        ty = np.array([b["losses_mean"]["phase_prediction_mse"] for b in blocks])
        vx = np.array([v["step"] for v in vals])
        vy = np.array([v["score"] for v in vals])
        assert np.isfinite(ty).all() and np.isfinite(vy).all() and (ty > 0).all() and (vy > 0).all()
        np.testing.assert_allclose(vy, [v["phase_position_mse"] + v["phase_momentum_mse"] for v in vals], rtol=1e-12, atol=1e-12)
        train, = ax.plot(tx, ty, c=BLUE, lw=2.15, label="训练：phase_prediction_mse（250步均值）")
        val, = ax.plot(vx, vy, c=ORANGE, lw=1.8, marker="o", ms=2.2, label="验证：position_momentum score")
        if name == "stage_a":
            assert arm["first_step"] == 10001
            ax.set_xlim(10000, 20000)
            ax.set_xticks([10000, 15000, 20000], ["10000", "15000", "20000"])
            ax.text(.97, .95, "本次续训 10001–20000 步\n预测长度固定为 32 步", transform=ax.transAxes,
                    va="top", ha="right", fontsize=9.5, color=MUTED, linespacing=1.5)
            handles = [train, val]
        else:
            assert arm["first_step"] == 1
            ax.set_xlim(0, 20000)
            ax.set_xticks([0, 10000, 20000], ["0", "10000", "20000"])
            for boundary in [1000, 4000, 10000]:
                ax.axvline(boundary, color="#aab8c6", lw=.85, ls=":", zorder=0)
            ax.text(.97, .95, "课程：4 → 8 → 16 → 32 步", transform=ax.transAxes,
                    va="top", ha="right", fontsize=9.5, color=MUTED)
        ax.set_title(heading, fontsize=13, color=heading_color, loc="left", pad=12)
        ax.set_xlabel("优化更新步数")
        ax.set_yscale("log")
        ax.set_ylim(5e-4, .32)
        tidy(ax)
    axes[0].set_ylabel("潜空间预测指标（对数刻度）")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.067, .847), ncol=2,
               frameon=False, fontsize=10.5, columnspacing=2.3, handlelength=2.3)
    fig.text(.075, .145, "这些指标衡量 q / p 潜空间预测，不是角度或角速度的物理误差；训练与验证的采样、预测长度口径不同。", fontsize=10.3, color=MUTED)
    fig.text(.075, .105, "B1 / B2 灰色虚线：第 1000、4000、10000 步后的预测长度切换；课程加长可使训练均值出现上跳。", fontsize=10.3, color=MUTED)
    fig.text(.075, .065, "训练均值画在各 250 步区间末端；未绘制含 SIGReg 的总损失。A 的前 10000 步不属于本次续训日志。", fontsize=10.3, color=MUTED)
    fig.savefig(OUT / "training_convergence.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, sources = load_data()
    a = metrics_for(data, "a", 0)
    single = metrics_for(data, "single", .2)
    multi = [metrics_for(data, "multi", b) for b in DAMPING]
    aggregate_figure(multi, a, single)
    horizon_figure(a, single, multi)
    pipeline_path = RUN / "pipeline.json"
    pipeline_figure(json.loads(pipeline_path.read_text())["plan"])
    history_path = OUT.parent / "analysis/training_history_summary.json"
    history = json.loads(history_path.read_text())
    convergence_figure(history)
    values = {"a": a, "single": single, "multi": multi}
    (OUT / "figure_metrics.json").write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n")
    manifest = {"source_files": sources, "pipeline_json": str(pipeline_path),
                "pipeline_sha256": digest(pipeline_path), "script_sha256": digest(Path(__file__)),
                "training_history_json": str(history_path), "training_history_sha256": digest(history_path),
                "training_history_remote_sources": {k: v["sources"] for k, v in history["arms"].items()},
                "training_curve_aggregation": "250-update nonoverlapping means, plotted at block end; raw validation score; A steps 10001..20000 and B steps 1..20000",
                "evaluation_scope": "All 64 validation initial-condition groups per seen damping condition",
                "aggregate_window": "Future frames 1..32 = 0.1..3.2 seconds; anchor excluded",
                "angle_metric": "mean(abs(atan2(sin(pred-truth), cos(pred-truth))))",
                "omega_metric": "sqrt(mean((pred-truth)^2))",
                "horizon_curves": "Pointwise across 64 groups, not cumulative in time",
                "matching_to_saved_json": "Every aggregate pred/teacher metric asserted to tolerance 1e-12",
                "teacher_semantics": "Each time uses its real observation history and the shared frozen readout; diagnostic, not autonomous prediction or a strict error lower bound",
                "model_execution": False, "model_updates": 0, "readout_fitting": False,
                "sample_selection": "None; all 64 groups per condition", "smoothing": False,
                "outputs": {name: digest(OUT / name) for name in ["summary_metrics.png", "horizon_errors.png", "pipeline.png", "training_convergence.png", "figure_metrics.json"]}}
    (OUT / "figure_provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: (OUT / name).stat().st_size for name in manifest["outputs"]}))


if __name__ == "__main__":
    main()
