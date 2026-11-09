"""Render saved validation trajectories only; no model execution or fitting."""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from PIL import Image

OUT = Path(__file__).resolve().parent
REVIEW = OUT.parent
MILESTONES = REVIEW / "run/milestones"
GROUP = 10000384
BLUE, ORANGE, INK = "#24758f", "#d97632", "#18354e"
BG = "#f7f9fc"
plt.rcParams.update({"font.family": "Arial Unicode MS", "axes.unicode_minus": False,
                     "font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": "#68798a", "ytick.color": "#68798a"})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    directory = MILESTONES / name / "evaluation"
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["checkpoint_step"] == 20000 and summary["q_dim"] == 2
    assert not summary["test_split_accessed"] and summary["parameters_unchanged"]
    source = directory / "val/trajectories.npz"
    published = json.loads((directory / "complete.json").read_text())
    assert sha(source) == published["artifacts"]["val/trajectories.npz"]
    with np.load(source, allow_pickle=False) as f:
        arrays = {k: f[k].copy() for k in f.files}
    assert np.allclose(arrays["time"], np.arange(33) * .1)
    return arrays, source


def select(arrays, damping):
    matches = np.flatnonzero((arrays["group_id"] == GROUP) & (arrays["damping"] == damping))
    assert len(matches) == 1
    i = int(matches[0])
    assert arrays["energy_band"][i] == 2 and arrays["valid_steps"][i] == 32
    band = (arrays["energy_band"] == 2) & (arrays["damping"] == damping)
    assert GROUP == int(arrays["group_id"][band].min())
    truth, pred = arrays["truth"][i], arrays["pred"][i]
    assert truth.shape == pred.shape == (33, 2)
    assert np.isfinite(truth).all() and np.isfinite(pred).all()
    # These selected swings never cross the angle branch cut.
    assert np.max(np.abs(truth[:, 0])) < np.pi and np.max(np.abs(pred[:, 0])) < np.pi
    return dict(index=i, time=arrays["time"], truth=truth, pred=pred, damping=damping)


def format_curve(ax, t, truth, pred, limits=None):
    ax.plot(t, truth, color=BLUE, lw=2.1, label="真实轨迹")
    ax.plot(t, pred, color=ORANGE, lw=1.9, ls="--", label="自主预测")
    ax.set_xlim(0, 3.2)
    ax.set_xticks([0, 1.6, 3.2])
    if limits:
        ax.set_ylim(*limits)
    ax.set_xlabel("预测时间（秒）", fontsize=9, labelpad=3)
    ax.set_ylabel("角度（rad）", fontsize=9, labelpad=2)
    ax.grid(alpha=.17)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#b8c5d1")
    ax.tick_params(labelsize=9)


def build_animation(cases):
    fig = plt.figure(figsize=(11, 8.2), dpi=110, facecolor=BG)
    fig.text(.055, .956, "二维分阶段训练 · 20000 步最终预测演示", fontsize=20)
    fig.text(.055, .916, "同一预先指定初态组 10000384 · 真实历史 2 秒后自主预测 3.2 秒", fontsize=11, color="#64788b")
    legend = [Line2D([0], [0], color=BLUE, lw=2.5, marker="o", markersize=5, label="真实轨迹"),
              Line2D([0], [0], color=ORANGE, lw=2, ls="--", marker="o", markerfacecolor="white", markersize=5, label="自主预测")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.05, .887), ncol=2, frameon=False, fontsize=11)
    clock = fig.text(.93, .866, "t = 0.0 s", ha="right", fontsize=14, color=INK)
    artists = []
    for j, case in enumerate(cases):
        x = .055 + (j % 2) * .475
        y = .485 - (j // 2) * .38
        fig.add_artist(FancyBboxPatch((x-.014, y-.015), .445, .35, transform=fig.transFigure,
                       boxstyle="round,pad=0.008,rounding_size=0.014", facecolor="white",
                       edgecolor="#dde5ec", lw=1, zorder=-1))
        fig.text(x, y+.303, case["title"], fontsize=12.5)
        ax = fig.add_axes([x+.005, y+.06, .17, .225], facecolor="white")
        ax.set_aspect("equal"); ax.set_xlim(-1.13, 1.13); ax.set_ylim(-1.15, .25); ax.axis("off")
        ax.plot([-.24, .24], [.08, .08], color="#93a4b5", lw=3)
        ax.plot([0, 0], [.08, 0], color="#93a4b5", lw=2)
        ax.plot(0, 0, "o", ms=4, color=INK, zorder=6)
        trail_t, = ax.plot([], [], color=BLUE, lw=1.7, alpha=.23)
        trail_p, = ax.plot([], [], color=ORANGE, lw=1.6, alpha=.26, ls=":")
        rod_t, = ax.plot([], [], color=BLUE, lw=3.2, solid_capstyle="round", zorder=3)
        rod_p, = ax.plot([], [], color=ORANGE, lw=2.1, ls="--", zorder=4)
        bob_t, = ax.plot([], [], "o", color=BLUE, ms=12, zorder=4)
        bob_p, = ax.plot([], [], "o", markerfacecolor="white", markeredgecolor=ORANGE, markeredgewidth=2, ms=9, zorder=5)
        curve = fig.add_axes([x+.232, y+.071, .185, .20], facecolor="white")
        format_curve(curve, case["time"], case["truth"][:, 0], case["pred"][:, 0], (-1.85, 1.85))
        cursor = curve.axvline(0, color="#8598a9", lw=1, alpha=.8)
        dot_t, = curve.plot([], [], "o", color=BLUE, ms=4, zorder=5)
        dot_p, = curve.plot([], [], "o", color=ORANGE, ms=4, zorder=5)
        artists.append((case, trail_t, trail_p, rod_t, rod_p, bob_t, bob_p, cursor, dot_t, dot_p))
    fig.text(.055, .06, "根据物理角度重绘，非图像解码器输出；固定初态组，不能代表全部样本。", fontsize=10, color="#6b7886")
    fig.text(.055, .032, "蓝色实心为真值，橙色空心为预测；下方强阻尼样本采用同一固定读出，需结合总体误差解读。", fontsize=9.5, color="#6b7886")
    frames = []
    for k in range(33):
        clock.set_text(f"t = {k*.1:.1f} s")
        for case, tr_t, tr_p, rod_t, rod_p, bob_t, bob_p, cursor, dot_t, dot_p in artists:
            true, pred = case["truth"][:, 0], case["pred"][:, 0]
            xt, yt = np.sin(true[k]), -np.cos(true[k])
            xp, yp = np.sin(pred[k]), -np.cos(pred[k])
            rod_t.set_data([0, xt], [0, yt]); rod_p.set_data([0, xp], [0, yp])
            bob_t.set_data([xt], [yt]); bob_p.set_data([xp], [yp])
            history = slice(max(0, k-7), k+1)
            tr_t.set_data(np.sin(true[history]), -np.cos(true[history]))
            tr_p.set_data(np.sin(pred[history]), -np.cos(pred[history]))
            cursor.set_xdata([k*.1, k*.1]); dot_t.set_data([k*.1], [true[k]]); dot_p.set_data([k*.1], [pred[k]])
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    # Save a representative still as a fallback and for visual inspection.
    frames[16].save(OUT / "staged_demo_midpoint.png")
    frames[0].save(OUT / "staged_prediction_demo.gif", save_all=True, append_images=frames[1:],
                   duration=100, loop=0, optimize=True, disposal=2)
    plt.close(fig)


def multi_static(multi):
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 8.8), dpi=150, facecolor="white")
    fig.subplots_adjust(left=.08, right=.97, top=.84, bottom=.12, hspace=.67, wspace=.26)
    fig.text(.08, .955, "多工况阻尼模型 · 六种阻尼的固定样本", fontsize=19)
    fig.text(.08, .914, "第 20000 步 · 同一初态组 10000384（能段 2）· 真值与自主预测", fontsize=11, color="#64788b")
    handles = [Line2D([0], [0], color=BLUE, lw=2, label="真实轨迹"),
               Line2D([0], [0], color=ORANGE, lw=2, ls="--", label="自主预测")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.98, .89), ncol=2, frameon=False)
    for ax, damping in zip(axes.flat, [0., .05, .1, .2, .5, 1.]):
        case = select(multi, damping)
        format_curve(ax, case["time"], case["truth"][:, 0], case["pred"][:, 0], (-1.85, 1.85))
        ax.set_title(f"b = {damping:g}", loc="left", fontsize=12, pad=8)
        ax.set_xticks([0, .8, 1.6, 2.4, 3.2])
    fig.text(.08, .058, "固定样本按初始能段最小 group ID 选择，不按预测误差挑选；所有子图使用同一纵轴范围。", fontsize=10, color="#6b7886")
    fig.text(.08, .03, "横轴零点为原始轨迹 2.0 秒；这是物理状态读出后的预测，尚未包含长时或未见阻尼测试。", fontsize=10, color="#6b7886")
    fig.savefig(OUT / "multi_damping_six_conditions.png", dpi=150)
    plt.close(fig)


def main():
    arrays, sources = {}, {}
    for key, name in {"a": "undamped_step20000", "single": "damping_single_step20000", "multi": "damping_multi_step20000"}.items():
        arrays[key], path = load(name)
        sources[key] = dict(path=str(path), sha256=sha(path))
    cases = []
    for arm, b, title in [("a", 0., "A：无阻尼模型  |  b = 0"),
                           ("single", .2, "B1：单阻尼模型  |  b = 0.2"),
                           ("multi", .2, "B2：多阻尼模型  |  b = 0.2"),
                           ("multi", 1., "B2：多阻尼模型  |  b = 1")]:
        case = select(arrays[arm], b)
        case.update(arm=arm, title=title)
        cases.append(case)
    assert np.array_equal(cases[1]["truth"], cases[2]["truth"])
    build_animation(cases)
    multi_static(arrays["multi"])
    manifest = dict(group_id=GROUP, energy_band=2, sample_rule="Minimum group_id in energy_band2, predeclared before results",
                    checkpoint_steps={k:20000 for k in arrays}, horizon_steps=32, physical_dt=.1,
                    source_files=sources, gif_panels=[dict(arm=c["arm"], damping=c["damping"], source_row=c["index"]) for c in cases],
                    no_model_execution=True, no_fitting=True, optimizer_updates=0,
                    rendered_from="Saved physical angle truth and decoded autonomous predictions; not a learned image decoder",
                    outputs={name:sha(OUT/name) for name in ["staged_prediction_demo.gif", "staged_demo_midpoint.png", "multi_damping_six_conditions.png"]})
    (OUT / "visual_provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({name:(OUT/name).stat().st_size for name in manifest["outputs"]}))


if __name__ == "__main__":
    main()
