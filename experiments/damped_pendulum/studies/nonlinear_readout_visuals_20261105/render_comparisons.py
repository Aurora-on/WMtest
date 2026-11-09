#!/usr/bin/env python3
"""Display frozen linear/nonlinear state readouts without fitting a model.

Actual saved RGB is the reference. All other pendulums are fixed-renderer
redrawings of decoded states, never direct image-decoder output. Observation
readouts and autonomous predictions are displayed separately and labelled.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
INK, MUTED, BG = "#183049", "#526478", "#F4F7FB"
BLUE, ORANGE, GREEN, PURPLE = "#2862AF", "#C26A2A", "#24765B", "#8156A2"
FONT_CANDIDATES = (
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font_file(explicit=None):
    for value in ([explicit] if explicit else []) + list(FONT_CANDIDATES):
        if Path(value).is_file():
            return Path(value)
    raise FileNotFoundError("Pass a Chinese TTF/TTC font using --font")


def load_renderer(code_root):
    path = Path(code_root) / "dpbench/generate.py"
    spec = importlib.util.spec_from_file_location("nonlinear_comparison_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render, path


def validate(a, meta):
    if meta.get("checkpoint_step") != 10000:
        raise ValueError("Both checkpoints must be final step10000")
    hashes = meta.get("checkpoint_sha256", {})
    if not isinstance(hashes, dict) or any(len(str(hashes.get(arm, ""))) != 64 for arm in ("control", "lehami")):
        raise ValueError("Both checkpoint SHA256 values are required")
    if a["true_rgb"].shape != (4, 33, 64, 64, 3) or a["true_rgb"].dtype != np.uint8:
        raise ValueError("Expected four saved 33-frame RGB clips")
    keys = ["truth"] + [f"{arm}_{mode}_{readout}" for arm in ("control", "lehami")
                         for mode in ("teacher", "rollout") for readout in ("ridge", "knn")]
    for key in keys:
        if a[key].shape != (4, 33, 2):
            raise ValueError(f"Bad shape for {key}")
        if (key == "truth" or "teacher" in key) and not np.isfinite(a[key]).all():
            raise ValueError(f"Nonfinite observed/reference states: {key}")
    if a["time"].shape != (33,) or not np.allclose(a["time"], np.arange(33) * .1, atol=1e-8, rtol=0):
        raise ValueError("Expected original dt=.1, without resampling")
    for key in ("energy_band", "group_id", "episode"):
        if a[key].shape != (4,):
            raise ValueError(f"Bad identity vector {key}")
    if sorted(a["energy_band"].tolist()) != [0, 1, 2, 3]:
        raise ValueError("Need one fixed case from each energy band")
    expected = set(zip(a["energy_band"].tolist(), a["group_id"].tolist(), a["episode"].tolist()))
    records = {(int(r["energy_band"]), int(r["group_id"]), int(r["episode"])) for r in meta["selected_records"]}
    if expected != records:
        raise ValueError("Case identities disagree with metadata")
    for arm in ("control", "lehami"):
        valid = a[f"valid_{arm}"]
        if valid.shape != (4,) or not np.issubdtype(valid.dtype, np.integer) or np.any(valid < 0) or np.any(valid > 32):
            raise ValueError(f"Bad valid prefix counts for {arm}")
        for i, count in enumerate(valid):
            for method in ("ridge", "knn"):
                if not np.isfinite(a[f"{arm}_rollout_{method}"][i, :int(count) + 1]).all():
                    raise ValueError("Invalid state inside claimed valid rollout prefix")


def center(draw, text, x, y, font, fill=INK):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=font, fill=fill)


def labels(mode):
    return (("真实视频", "线性观测读出", "非线性观测读出") if mode == "teacher" else
            ("真实视频", "线性读出 · 自主预测", "非线性读出 · 自主预测"))


def keys(arm, mode):
    return ("truth", f"{arm}_{mode}_ridge", f"{arm}_{mode}_knn")


def is_valid(a, key, case, step):
    if not np.isfinite(a[key][case, step]).all():
        return False
    if "rollout" in key:
        return step <= a["valid_" + key.split("_")[0]][case]
    return True


def tile(a, key, case, step, renderer, size):
    if key == "truth":
        rgb = a["true_rgb"][case, step]
    elif is_valid(a, key, case, step):
        rgb = renderer(float(a[key][case, step, 0]), 64, renderer="smooth_v1")
    else:
        return None
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)


def put_tile(image, draw, a, key, case, step, renderer, x, y, size, font):
    picture = tile(a, key, case, step, renderer, size)
    if picture is None:
        draw.rectangle((x, y, x + size, y + size), fill="#FFF1EF")
        center(draw, "INVALID", x + size / 2, y + size / 2 - 12, font, "#A9332C")
    else:
        image.paste(picture, (int(x), int(y)))


def banner(draw, box, mode, font, arm="lehami"):
    if arm == "control":
        text, fill, color = "对照组：非线性读出仍不合格；此图用于比较读出能力。", "#FFF0DD", "#904A16"
    elif mode == "teacher":
        text, fill, color = "每个时刻重新读取真实历史视频；本图比较状态读出，不是自主预测。", "#E5F2E9", "#235F40"
    else:
        text, fill, color = "仅用起始21帧初始化；之后不再读取真实图像。同一潜轨迹，仅更换读出。", "#E8EEF8", "#31547F"
    draw.rounded_rectangle(box, radius=8, fill=fill)
    draw.text((box[0] + 12, box[1] + 8), text, font=font, fill=color)


def filmstrip(a, case, renderer, fontpath, arm="lehami", mode="teacher"):
    width, height = 1450, 940
    image = Image.new("RGB", (width, height), BG); draw = ImageDraw.Draw(image)
    f = {n: ImageFont.truetype(str(fontpath), n) for n in (20, 21, 23, 25, 31)}
    name = "LeHami式目标组" if arm == "lehami" else "对照组"
    band = int(a["energy_band"][case]); ep = int(a["episode"][case])
    draw.text((26, 16), f"{name} · 10000步 · 无阻尼    |    能量层 {band}", font=f[31], fill=INK)
    draw.text((26, 60), f"同一模型、同一视频、同一潜状态，仅更换读出方式 · episode {ep}", font=f[21], fill=MUTED)
    banner(draw, (22, 99, width - 22, 142), mode, f[21], arm)
    steps, left, cell, start, stride = (0, 8, 16, 24, 32), 260, 232, 192, 211
    for col, step in enumerate(steps):
        center(draw, f"t = {a['time'][step]:.1f} 秒", left + (col + .5) * cell, 159, f[23])
    for row, key in enumerate(keys(arm, mode)):
        y = start + row * stride
        draw.rounded_rectangle((18, y, width - 18, y + stride - 10), radius=10, fill="white")
        label = labels(mode)[row] if mode == "teacher" else ("真实视频", "线性自主读出", "非线性自主读出")[row]
        color = (BLUE, ORANGE, GREEN)[row]
        draw.text((32, y + 48), label, font=f[25], fill=color)
        notes = (("保存的原始图像", "仅作最近邻放大") if row == 0 else
                 (("每步读取真实历史", "状态读出后重绘") if mode == "teacher" else
                  ("只看起始21帧", "随后自主推进")))
        for j, text in enumerate(notes):
            draw.text((32, y + 91 + j * 28), text, font=f[20], fill=MUTED)
        for col, step in enumerate(steps):
            x = left + col * cell
            put_tile(image, draw, a, key, case, step, renderer, x + (cell - 160) / 2, y + 8, 160, f[25])
            text = f"θ = {a[key][case, step, 0]:+.2f} rad" if is_valid(a, key, case, step) else "INVALID · 未补帧"
            center(draw, text, x + cell / 2, y + 170, f[20], color)
    draw.text((26, 839), "固定样本：各能量层最小 group_id，不按误差筛选。时间从起始21帧观测末端算起。", font=f[20], fill=MUTED)
    draw.text((26, 869), "读出摆图是潜状态经固定状态读出后的重绘，不是图像解码器生成。", font=f[20], fill=MUTED)
    draw.text((26, 899), "θ=0 向下、正角向右。使用原始0.1秒采样；未插值、未修正轨迹。", font=f[20], fill=MUTED)
    return image


def movie_frame(a, step, renderer, fontpath, mode):
    width, height = 1140, 820
    image = Image.new("RGB", (width, height), BG); draw = ImageDraw.Draw(image)
    f = {n: ImageFont.truetype(str(fontpath), n) for n in (18, 20, 22, 28)}
    draw.text((22, 13), "LeHami式目标组 · 10000步 · 无阻尼", font=f[28], fill=INK)
    draw.text((870, 18), f"t = {a['time'][step]:.1f} / 3.2 秒", font=f[22], fill=INK)
    banner(draw, (18, 57, width - 18, 98), mode, f[20])
    for col, label in enumerate(labels(mode)):
        center(draw, label, 190 + 380 * col, 110, f[22], (BLUE, ORANGE, GREEN)[col])
        note = "原始保存图像" if col == 0 else ("每步看真实历史 · 非预测" if mode == "teacher" else "同一自主轨迹 · 不重置状态")
        center(draw, note, 190 + 380 * col, 143, f[18], MUTED)
    ordered = np.argsort(a["energy_band"])
    for row, case in enumerate((ordered[0], ordered[-1])):
        top = 180 + row * 270
        draw.text((26, top), f"能量层 {int(a['energy_band'][case])} · episode {int(a['episode'][case])}", font=f[20], fill=INK)
        for col, key in enumerate(keys("lehami", mode)):
            x = 380 * col
            put_tile(image, draw, a, key, case, step, renderer, x + 94, top + 30, 192, f[28])
            text = f"θ = {a[key][case, step, 0]:+.2f} rad" if is_valid(a, key, case, step) else "INVALID · 未补帧"
            center(draw, text, x + 190, top + 230, f[20], (BLUE, ORANGE, GREEN)[col])
    draw.text((24, 736), "固定样本：最低／最高能量层的最小group_id。1×播放；0.1秒一帧，无插值。", font=f[18], fill=MUTED)
    draw.text((24, 766), "线性与非线性列均为状态读出后的重绘；模型没有直接生成图像。", font=f[18], fill=MUTED)
    return image


def save_gif(frames, path):
    strip = Image.new("RGB", (frames[0].width, frames[0].height * 3))
    for j, k in enumerate((0, 16, 32)):
        strip.paste(frames[k], (0, j * frames[0].height))
    palette = strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=False, disposal=2)
    with Image.open(path) as saved:
        if saved.n_frames != 33:
            raise RuntimeError("GIF frame count differs from original sampling")


def curves(a, path, fontpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=str(fontpath))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), sharex=True, sharey=True)
    specs = (("truth", "真实角度", BLUE, "-", 2.6),
             ("lehami_teacher_knn", "非线性观测读出（非预测）", GREEN, "--", 1.8),
             ("lehami_rollout_ridge", "自主预测＋线性读出", ORANGE, ":", 2.),
             ("lehami_rollout_knn", "自主预测＋非线性读出", PURPLE, "-", 1.8))
    for ax, case in zip(axes.flat, np.argsort(a["energy_band"])):
        for key, label, color, style, width in specs:
            angle = np.arctan2(np.sin(a[key][case, :, 0]), np.cos(a[key][case, :, 0]))
            if "rollout" in key:
                angle[int(a["valid_lehami"][case]) + 1:] = np.nan
            angle[np.flatnonzero(abs(np.diff(angle)) > np.pi) + 1] = np.nan
            ax.plot(a["time"], angle, label=label, color=color, ls=style, lw=width)
        ax.set_title(f"能量层 {int(a['energy_band'][case])} · episode {int(a['episode'][case])}", fontproperties=font, fontsize=13)
        ax.set_xlim(0, 3.2); ax.set_ylim(-np.pi, np.pi)
        ax.set_xticks([0, .8, 1.6, 2.4, 3.2]); ax.set_yticks([-np.pi, 0, np.pi], ["−π", "0", "π"])
        ax.set_xlabel("从观察末端起的时间（秒）", fontproperties=font, fontsize=11)
        ax.set_ylabel("角度（rad）", fontproperties=font, fontsize=11); ax.grid(alpha=.2)
    handles, legend = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="upper center", bbox_to_anchor=(.5, .947), ncol=2, prop=font, frameon=False)
    fig.suptitle("LeHami式目标组 · 10000步：读出改善与自主预测是两项检查", fontproperties=font, fontsize=18, y=.988)
    fig.text(.5, .035, "绿色虚线逐时刻读取真实历史，不能作为自主预测成绩。橙／紫线来自完全相同的自主潜轨迹。\n四个样本固定选取；本图为验证集3.2秒诊断，不代表更长时间或有阻尼情形。", ha="center", va="center", fontproperties=font, fontsize=11, color=MUTED)
    fig.subplots_adjust(left=.08, right=.98, top=.845, bottom=.13, hspace=.35, wspace=.18)
    fig.savefig(path, dpi=150, facecolor="white"); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=HERE / "comparison_inputs.npz")
    parser.add_argument("--metadata", type=Path, default=HERE / "comparison_meta.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    parser.add_argument("--code-root", type=Path, default=HERE.parent / "control_visuals_20261105/frozen_code")
    parser.add_argument("--font")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory; do not mix different evaluations")
    with np.load(args.inputs, allow_pickle=False) as handle:
        a = {key: handle[key] for key in handle.files}
    meta = json.loads(args.metadata.read_text()); validate(a, meta)
    renderer, renderer_path = load_renderer(args.code_root); fontpath = font_file(args.font)
    args.output_dir.mkdir(parents=True, exist_ok=True); outputs = []
    for arm, band in (("lehami", 1), ("lehami", 3), ("control", 1)):
        case = int(np.flatnonzero(a["energy_band"] == band)[0])
        name = f"{arm}_teacher_band{band}.png"
        filmstrip(a, case, renderer, fontpath, arm=arm).save(args.output_dir / name); outputs.append(name)
    for mode in ("teacher", "rollout"):
        frames = [movie_frame(a, step, renderer, fontpath, mode) for step in range(33)]
        name = f"lehami_{mode}_comparison.gif"; save_gif(frames, args.output_dir / name); outputs.append(name)
        name = f"lehami_{mode}_preview.png"; frames[16].save(args.output_dir / name); outputs.append(name)
    name = "lehami_readout_and_rollout_curves.png"; curves(a, args.output_dir / name, fontpath); outputs.append(name)
    manifest = dict(kind="frozen_nonlinear_readout_comparison", checkpoint_step=10000,
                    checkpoint_sha256=meta["checkpoint_sha256"], selected_records=meta["selected_records"],
                    sampling="4 fixed validation cases, original dt0.1 and times0..3.2, no interpolation",
                    teacher="Each time point sees its real history; not autonomous prediction",
                    rollout="Same latent trajectory decoded by two fixed readouts; only initial21 real frames",
                    display="Stored true RGB; all readout images redrawn by fixed renderer; no image decoder",
                    source_sha256={str(args.inputs):sha(args.inputs), str(args.metadata):sha(args.metadata)},
                    script_sha256=sha(__file__), renderer_sha256=sha(renderer_path), font=str(fontpath),
                    outputs={name:dict(sha256=sha(args.output_dir/name), bytes=(args.output_dir/name).stat().st_size) for name in outputs})
    (args.output_dir / "render_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()), outputs=outputs), ensure_ascii=False))


if __name__ == "__main__":
    main()
