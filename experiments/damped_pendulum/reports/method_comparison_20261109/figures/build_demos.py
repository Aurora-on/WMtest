#!/usr/bin/env python3
"""Repaint saved physical trajectories; never execute or fit a world model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

FIGURES = Path(__file__).resolve().parent
REPORT = FIGURES.parent
DEMOS = REPORT / "demos"
DATA = REPORT / "data/runs"
METHODS = {"lehami": ("LeHami", "#24758f"), "lewm": ("LeWM", "#2b8a63"),
           "hjepa": ("H-JEPA", "#b94e59"), "ours": ("Ours", "#d97a32")}
CASES = [("a", 0., "无阻尼 A  |  b=0"), ("single", .2, "单工况 B  |  b=0.2"),
         ("multi", .1, "多工况 B  |  b=0.1")]
EXPECTED_COMMON_GROUP = 10000416
TRUTH, INK, MUTED, GRID, BG = "#292e35", "#18354e", "#62788f", "#dce4ed", "#f7f9fc"
plt.rcParams.update({"font.family": "Arial Unicode MS", "axes.unicode_minus": False,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": "#b7c7d7", "savefig.facecolor": "white"})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation(method, stage):
    return DATA / method / f"{method}_seed0" / stage / "evaluation"


def available(method):
    return all(all((evaluation(method, stage) / name).is_file()
                   for name in ["val/trajectories.npz", "summary.json", "complete.json"]) for stage, _, _ in CASES)


def load(method, stage):
    root = evaluation(method, stage)
    source = root / "val/trajectories.npz"
    summary = json.loads((root / "summary.json").read_text())
    complete = json.loads((root / "complete.json").read_text())
    assert complete["status"] == summary["status"] == "completed"
    assert summary["checkpoint_step"] == 20000
    assert sha(source) == complete["artifacts"]["val/trajectories.npz"]
    with np.load(source, allow_pickle=False) as f:
        z = {k: f[k].copy() for k in f.files}
    np.testing.assert_allclose(z["time"], np.arange(33) * .1, rtol=0, atol=1e-12)
    assert z["truth"].shape == z["pred"].shape and z["truth"].shape[1:] == (33, 2)
    return {"method": method, "stage": stage, "arrays": z, "summary": summary,
            "source": str(source), "sha256": sha(source)}


def select(loaded, gid, damping):
    z = loaded["arrays"]
    found = np.flatnonzero((z["group_id"] == gid) & np.isclose(z["damping"], damping))
    assert len(found) == 1
    i = int(found[0])
    assert z["valid_steps"][i] == 32 and np.isfinite(z["pred"][i]).all()
    return {"method": loaded["method"], "stage": loaded["stage"], "group_id": int(gid),
            "damping": float(damping), "energy_band": int(z["energy_band"][i]), "source_row": i,
            "truth": z["truth"][i], "pred": z["pred"][i], "time": z["time"],
            "source": loaded["source"], "source_sha256": loaded["sha256"]}


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, lw=.65, alpha=.75)
    ax.tick_params(length=2.5, width=.6)
    ax.set_axisbelow(True)


def legend(fig, color, where=(.055, .865)):
    handles = [Line2D([], [], color=TRUTH, lw=2.2, label="真实轨迹"),
               Line2D([], [], color=color, lw=2.2, ls="--", label="自主预测")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=where, ncol=2,
               frameon=False, handlelength=2.7, columnspacing=1.8, fontsize=11)


def common_limits(cases):
    limits = []
    for component in range(2):
        vals = np.concatenate([c[k][:, component] for c in cases for k in ("truth", "pred")])
        extent = max(float(np.max(np.abs(vals))) * 1.08, .1)
        limits.append((-extent, extent))
    return limits


def trajectory_figure(method, cases, limits):
    label, color = METHODS[method]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.3), dpi=150)
    fig.subplots_adjust(left=.065, right=.975, top=.73, bottom=.19, hspace=.42, wspace=.22)
    fig.text(.065, .947, f"{label}  |  同一初态的三种训练工况", fontsize=21)
    fig.text(.065, .895, f"固定初态组 {cases[0]['group_id']} · 真实历史 2 秒后自主预测 3.2 秒 · 各阶段精确第 20000 步", fontsize=11.3, color=MUTED)
    legend(fig, color, (.057, .862))
    for col, case in enumerate(cases):
        axes[0, col].set_title(CASES[col][2], fontsize=12, loc="left", pad=12)
        for component, ax in enumerate(axes[:, col]):
            ax.plot(case["time"], case["truth"][:, component], color=TRUTH, lw=2.1)
            ax.plot(case["time"], case["pred"][:, component], color=color, lw=2.1, ls="--")
            ax.set(xlim=(0, 3.2), ylim=limits[component], xticks=[0, .8, 1.6, 2.4, 3.2])
            if col == 0:
                ax.set_ylabel("角度（rad）" if component == 0 else "角速度（rad/s）")
            if component == 1:
                ax.set_xlabel("预测时间（秒）")
            tidy(ax)
    fig.text(.065, .105, "选样规则：公共验证组按 group_id 排序，取上中位组（第 33/64 组）；所有方法、工况使用同一初态组。", fontsize=10, color=MUTED)
    fig.text(.065, .067, "单个固定样本不能代表总体误差；曲线直接来自保存的物理状态读出，未平滑或重新推理。各方法采用相同纵轴范围。", fontsize=10, color=MUTED)
    path = FIGURES / f"{method}_trajectories.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def compact_trajectory_figure(method, cases, limits):
    """PDF-sized plot: text lives in the report; keep only scientific labels."""
    color = METHODS[method][1]
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.6), dpi=180, sharex="col", sharey="row")
    fig.subplots_adjust(left=.083, right=.985, top=.80, bottom=.13, hspace=.28, wspace=.16)
    handles = [Line2D([], [], color=TRUTH, lw=2, label="真实轨迹"),
               Line2D([], [], color=color, lw=2, ls="--", label="自主预测")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1.015),
               ncol=2, frameon=False, handlelength=2.6, fontsize=12, columnspacing=2)
    for col, case in enumerate(cases):
        axes[0, col].set_title(CASES[col][2], fontsize=12, pad=10)
        for component, ax in enumerate(axes[:, col]):
            ax.plot(case["time"], case["truth"][:, component], color=TRUTH, lw=2)
            ax.plot(case["time"], case["pred"][:, component], color=color, lw=2, ls="--")
            ax.set(xlim=(0, 3.2), ylim=limits[component], xticks=[0, 1.6, 3.2])
            ax.tick_params(labelsize=11)
            if col == 0:
                ax.set_ylabel("角度（rad）" if component == 0 else "角速度（rad/s）", fontsize=12)
            if component == 1:
                ax.set_xlabel("时间（秒）", fontsize=11, labelpad=3)
            tidy(ax)
    path = FIGURES / f"{method}_trajectories_compact.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def animation_panel(fig, rectangle, case, title, color, theta_limits, mini_curve=True):
    x, y, w, h = rectangle
    fig.patches.append(FancyBboxPatch((x, y), w, h, transform=fig.transFigure,
        boxstyle="round,pad=.008,rounding_size=.015", fc="white", ec="#d6e1ed", lw=1.0, zorder=-1))
    fig.text(x+.015, y+h-.047, title, fontsize=12, color=INK)
    pend = fig.add_axes([x+.035, y+.27*h, w-.07, .59*h])
    pend.set(xlim=(-1.15, 1.15), ylim=(-1.18, .28), aspect="equal")
    pend.axis("off")
    pend.plot([-.22, .22], [0, 0], color="#91a4b6", lw=3)
    pend.plot(0, 0, "o", color=INK, ms=3)
    true_rod, = pend.plot([], [], color=TRUTH, lw=3)
    pred_rod, = pend.plot([], [], color=color, lw=2.3, ls="--")
    true_bob, = pend.plot([], [], "o", color=TRUTH, ms=9)
    pred_bob, = pend.plot([], [], "o", mfc="white", mec=color, mew=2.0, ms=11)
    trace_true, = pend.plot([], [], color=TRUTH, alpha=.2, lw=1.2)
    trace_pred, = pend.plot([], [], color=color, alpha=.3, lw=1.2, ls=":")
    cursor, true_dot, pred_dot = None, None, None
    if mini_curve:
        ax = fig.add_axes([x+.038, y+.055*h, w-.07, .24*h])
        ax.plot(case["time"], case["truth"][:, 0], color=TRUTH, lw=1.25)
        ax.plot(case["time"], case["pred"][:, 0], color=color, lw=1.3, ls="--")
        ax.set(xlim=(0, 3.2), ylim=theta_limits, xticks=[0, 1.6, 3.2])
        ax.tick_params(labelsize=8)
        ax.set_ylabel("角度", fontsize=8)
        tidy(ax)
        cursor = ax.axvline(0, color="#a3b3c4", lw=.9)
        true_dot, = ax.plot([], [], "o", color=TRUTH, ms=3.5)
        pred_dot, = ax.plot([], [], "o", color=color, ms=3.5)
    def update(frame):
        a, b = case["truth"][frame, 0], case["pred"][frame, 0]
        tx, ty, px, py = np.sin(a), -np.cos(a), np.sin(b), -np.cos(b)
        true_rod.set_data([0, tx], [0, ty]); pred_rod.set_data([0, px], [0, py])
        true_bob.set_data([tx], [ty]); pred_bob.set_data([px], [py])
        past = slice(max(0, frame-7), frame+1)
        trace_true.set_data(np.sin(case["truth"][past, 0]), -np.cos(case["truth"][past, 0]))
        trace_pred.set_data(np.sin(case["pred"][past, 0]), -np.cos(case["pred"][past, 0]))
        if cursor is not None:
            t = case["time"][frame]
            cursor.set_xdata([t, t]); true_dot.set_data([t], [a]); pred_dot.set_data([t], [b])
    return update


def save_animation(fig, updates, timer, path, preview):
    frames = []
    for frame in range(33):
        timer.set_text(f"t = {frame*.1:.1f} s")
        for update in updates:
            update(frame)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
    frames[16].save(preview)
    palette = frames[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
    encoded = [im.quantize(palette=palette) for im in frames]
    encoded[0].save(path, save_all=True, append_images=encoded[1:], duration=[100]*32+[1000],
                    loop=0, disposal=2, optimize=False)
    plt.close(fig)
    with Image.open(path) as check:
        assert check.n_frames == 33
        durations = []
        for frame in range(33):
            check.seek(frame); durations.append(check.info["duration"])
        assert durations == [100]*32+[1000]
    return path, preview


def method_animation(method, cases, limits):
    label, color = METHODS[method]
    fig = plt.figure(figsize=(12.6, 5.8), dpi=100, facecolor=BG)
    fig.text(.04, .94, f"{label}  |  三种工况自主预测演示", fontsize=21)
    fig.text(.04, .885, f"固定初态组 {cases[0]['group_id']} · 黑色为真值，彩色虚线为空心预测摆", fontsize=11, color=MUTED)
    timer = fig.text(.95, .885, "", fontsize=12, ha="right", color=INK)
    updates = [animation_panel(fig, (.04+i*.322, .19, .292, .61), case, CASES[i][2], color, limits[0])
               for i, case in enumerate(cases)]
    fig.text(.04, .105, "按保存的物理角度重绘，非图像解码器输出；单个固定样本不能代表全部验证组。", fontsize=10, color=MUTED)
    fig.text(.04, .062, "播放步长 0.1 秒；到 3.2 秒后停留 1 秒再循环。下方曲线为角度，完整角度/角速度见静态图。", fontsize=10, color=MUTED)
    return save_animation(fig, updates, timer, DEMOS/f"{method}_three_conditions.gif", FIGURES/f"{method}_demo_midpoint.png")


def overview_animation(selected, limits):
    fig = plt.figure(figsize=(10.6, 8.3), dpi=110, facecolor=BG)
    fig.text(.045, .95, "多工况模型对照  |  相同初态、b=0.1", fontsize=21)
    fig.text(.045, .905, f"固定组 {EXPECTED_COMMON_GROUP} · 黑色真值 / 彩色虚线预测 · 自主预测 3.2 秒", fontsize=10.8, color=MUTED)
    timer = fig.text(.95, .865, "", fontsize=12, ha="right")
    updates = []
    for i, method in enumerate(METHODS):
        x, y, w, h = .045 + (i % 2)*.485, .475-(i//2)*.35, .44, .32
        label, color = METHODS[method]
        if method in selected:
            updates.append(animation_panel(fig, (x,y,w,h), selected[method][2], label, color, limits[0]))
        else:
            fig.text(x+.03, y+h*.62, "Ours · 重评结果待下载", fontsize=14, color=MUTED)
            fig.text(x+.03, y+h*.40, "此处没有填入预测数据", fontsize=10, color=MUTED)
    fig.text(.045, .065, "同一公共验证组的上中位样本，非按效果挑选；按保存物理角度重绘，非图像解码。", fontsize=10, color=MUTED)
    fig.text(.045, .032, "步长 0.1 秒；到 3.2 秒后停留 1 秒再循环。", fontsize=10, color=MUTED)
    return save_animation(fig, updates, timer, DEMOS/"comparison_multi.gif", FIGURES/"comparison_multi_midpoint.png")


def select_failure(loaded):
    z = loaded["arrays"]
    scale = float(loaded["summary"]["omega_nrmse_denominator"])
    tw, pw = z["truth"][:, 1:, 1], z["pred"][:, 1:, 1]
    true_power = np.mean(np.lib.stride_tricks.sliding_window_view(tw**2, 10, axis=1), axis=-1)
    pred_power = np.mean(np.lib.stride_tricks.sliding_window_view(pw**2, 10, axis=1), axis=-1)
    dynamic = true_power > (.1*scale)**2
    stopped = dynamic & (pred_power < .01*true_power) & (z["valid_steps"][:, None] >= 32)
    candidates = sorted(np.flatnonzero(stopped.any(1)), key=lambda i: (int(z["group_id"][i]), float(z["damping"][i])))
    if not candidates:
        return None, {"status": "no_stopped_sample"}
    i = int(candidates[0])
    sample = select(loaded, int(z["group_id"][i]), float(z["damping"][i]))
    return sample, {"selection_rule": "Among complete H-JEPA multi records triggering the saved truth-defined stop rule, minimum group_id then minimum damping",
        "candidate_record_count": len(candidates), "source_row": i, "group_id": sample["group_id"], "damping": sample["damping"],
        "omega_scale": scale, "window_samples": 10, "triggered_window_indices_zero_based": np.flatnonzero(stopped[i]).tolist(),
        "criterion": "true window RMS > 0.1*omega_scale AND prediction RMS < 0.1*true RMS; use future frames1..32",
        "first_window_true_rms": float(np.sqrt(true_power[i, np.flatnonzero(stopped[i])[0]])),
        "first_window_pred_rms": float(np.sqrt(pred_power[i, np.flatnonzero(stopped[i])[0]])),
        "interpretation": "Explicit physical-readout failure example, not representative selection and not proof of latent-state stalling"}


def failure_figures(case, detail):
    _, color = METHODS["hjepa"]
    limits = common_limits([case])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), dpi=150)
    fig.subplots_adjust(left=.08, right=.97, top=.68, bottom=.25, wspace=.25)
    fig.text(.08, .935, "H-JEPA 多工况  |  明确标注的停止失败样本", fontsize=18)
    fig.text(.08, .86, f"组 {case['group_id']}，b={case['damping']:g}；从触发停止判据的样本中按最小组号选择", fontsize=10.5, color=MUTED)
    legend(fig, color, (.071, .815))
    for component, ax in enumerate(axes):
        ax.plot(case["time"], case["truth"][:, component], color=TRUTH, lw=2)
        ax.plot(case["time"], case["pred"][:, component], color=color, lw=2, ls="--")
        ax.set(xlim=(0,3.2), ylim=limits[component], xticks=[0,.8,1.6,2.4,3.2], xlabel="预测时间（秒）",
               ylabel="角度（rad）" if component == 0 else "角速度（rad/s）")
        tidy(ax)
    fig.text(.08, .14, "真值保持运动，但预测的物理角度读出在第 1 步后保持不变；角速度读出触发停止判据。", fontsize=10.2, color=MUTED)
    fig.text(.08, .085, "这是失败案例展示；不代表所有样本，也不能单凭物理读出断言潜状态停止。", fontsize=10.2, color=MUTED)
    static = FIGURES / "hjepa_failure_trajectories.png"
    fig.savefig(static); plt.close(fig)
    fig = plt.figure(figsize=(7.4, 6.7), dpi=110, facecolor=BG)
    fig.text(.06,.945,"H-JEPA 多工况 · 停止失败样本",fontsize=19)
    fig.text(.06,.89,f"组 {case['group_id']}，b={case['damping']:g} · 黑色真值 / 红色预测",fontsize=10.8,color=MUTED)
    timer = fig.text(.92,.847,"",fontsize=12,ha="right")
    update = animation_panel(fig,(.13,.2,.74,.60),case,"失败示例：物理读出预测停滞",color,limits[0])
    fig.text(.06,.115,"按最小触发停止组号选择，并非共同中位样本。",fontsize=10,color=MUTED)
    fig.text(.06,.074,"角度重绘；步长0.1秒，3.2秒后停留1秒循环。",fontsize=10,color=MUTED)
    gif, preview = save_animation(fig,[update],timer,DEMOS/"hjepa_failure_demo.gif",FIGURES/"hjepa_failure_midpoint.png")
    return static, gif, preview


def public_case(case):
    return {k:v for k,v in case.items() if k not in ("truth","pred","time")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-ours", action="store_true")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True); DEMOS.mkdir(parents=True, exist_ok=True)
    methods = [m for m in METHODS if available(m)]
    assert all(m in methods for m in ("lehami","lewm","hjepa"))
    if args.require_ours and "ours" not in methods:
        raise FileNotFoundError("Ours needs complete a/single/multi evaluation archives before final four-method rendering")
    loaded = {(m,s):load(m,s) for m in methods for s,_,_ in CASES}
    sets = []
    for m in methods:
        for s,b,_ in CASES:
            z = loaded[m,s]["arrays"]
            sets.append(set(z["group_id"][np.isclose(z["damping"],b)].tolist()))
    common = sorted(set.intersection(*sets))
    assert len(common) == 64 and all(g == set(common) for g in sets)
    gid = int(common[len(common)//2])
    assert gid == EXPECTED_COMMON_GROUP
    selected = {m:[select(loaded[m,s],gid,b) for s,b,_ in CASES] for m in methods}
    # Matched conditions must have byte-for-byte identical physical truth.
    for i in range(3):
        for m in methods[1:]:
            np.testing.assert_array_equal(selected[methods[0]][i]["truth"],selected[m][i]["truth"])
    limits = common_limits([c for cases in selected.values() for c in cases])
    outputs = []
    for m in methods:
        outputs.append(trajectory_figure(m,selected[m],limits))
        outputs.append(compact_trajectory_figure(m,selected[m],limits))
        outputs.extend(method_animation(m,selected[m],limits))
    outputs.extend(overview_animation(selected,limits))
    failure, failure_meta = select_failure(loaded["hjepa","multi"])
    if failure is not None:
        # This precise example is visibly frozen after the first prediction; do
        # not silently reuse the descriptive caption if a future archive changes.
        np.testing.assert_array_equal(failure["pred"][1:,0],np.repeat(failure["pred"][1,0],32))
        outputs.extend(failure_figures(failure,failure_meta))
    manifest = {"methods": methods, "group_id": gid, "common_group_ids": common,
        "selection_rule": "Sorted common validation group IDs; upper median, index len//2 =32 (33rd of64); chosen independently of prediction errors",
        "selected_cases": {m:[public_case(c) for c in cases] for m,cases in selected.items()},
        "shared_axis_limits": limits, "true_trajectories_identical_across_methods": True,
        "gif_frames":33,"physical_dt":.1,"frame_duration_ms":[100]*32+[1000],
        "loop_pause": "After reaching t3.2s, hold last frame1000ms then loop",
        "rendering": "Pendulum repainted from saved physical theta, not image decoder output",
        "model_execution":False,"training_updates":0,"readout_fitting":False,"smoothing":False,
        "failure_example":failure_meta,"script_sha256":sha(__file__),
        "outputs":{str(p.relative_to(REPORT)):sha(p) for p in outputs}}
    (DEMOS/"demo_selection_and_provenance.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"methods":methods,"group_id":gid,"files":[str(p.relative_to(REPORT)) for p in outputs]},ensure_ascii=False))


if __name__ == "__main__":
    main()
