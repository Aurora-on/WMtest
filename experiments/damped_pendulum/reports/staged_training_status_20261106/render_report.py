#!/usr/bin/env python3
"""Render verified staged-status diagram plus existing D2 step10000 trajectories.

Status text comes from the 2026-11-06 14:44 Beijing server inspection reported
by the coordinating agent. This renderer does not connect to a server or train.
Curves are the predeclared minimum-group-id examples, not error-selected cases.
"""
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'studies/dimension_launch_20261105T113319Z/evaluation/d2/val/trajectories.npz'
OUTPUT = HERE / 'staged_status_and_available_trajectories.png'
FONT = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
EXPECTED_SHA = '401008ff99f81b58d4b551dcde4bfcc633c664802e0844023f19dc9ee99678ea'


def draw():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED_SHA:
        raise ValueError('Baseline trajectory artifact changed')
    if not FONT.exists():
        raise FileNotFoundError('Arial Unicode font is required for this report')
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=FONT).get_name(),
                         'axes.unicode_minus': False, 'font.size': 12,
                         'axes.labelsize': 12, 'axes.titlesize': 14,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white'})
    z = np.load(SOURCE, allow_pickle=False)
    indices = z['selected_indices']
    if indices.tolist() != [1, 3, 0, 2]:
        raise ValueError('Predeclared example indices changed')
    for index in indices:
        candidates = np.flatnonzero(z['energy_band'] == z['energy_band'][index])
        if int(index) != int(candidates[np.argmin(z['group_id'][candidates])]):
            raise ValueError('Example is not the declared minimum group_id')
        if int(z['valid_steps'][index]) != 32:
            raise ValueError('Example has an invalid rollout prefix')

    fig = plt.figure(figsize=(14, 11), facecolor='white')
    ink, muted = '#183247', '#566775'
    truth_color, prediction_color = '#17627B', '#D16A32'
    fig.text(.055, .966, '二维分阶段训练：已核实进度与现有轨迹',
             fontsize=23, fontweight='bold', color=ink, va='top')
    fig.text(.055, .922, '2026-11-06 14:44 北京时间服务器核查  ·  当前队列为空',
             fontsize=14, color=muted, va='top')

    ax = fig.add_axes([.045, .696, .925, .201])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    def box(x, y, width, height, title, details, verified=False):
        color = '#1F6A59' if verified else '#687684'
        patch = FancyBboxPatch((x, y), width, height,
                               boxstyle='round,pad=0.008,rounding_size=0.018',
                               facecolor='#EDF7F1' if verified else '#F4F6F8',
                               edgecolor='#73A995' if verified else '#B6C1CA', linewidth=1.4)
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height * .75, title, ha='center', va='center',
                fontsize=15, fontweight='bold', color=color)
        ax.text(x + width / 2, y + height * .35, details, ha='center', va='center',
                fontsize=12.5, color=ink, linespacing=1.4)

    box(.012, .28, .215, .46, '已验证：无阻尼 10000 步', '二维模型 · D = 0\n已有验证集物理评测', True)
    box(.325, .28, .224, .46, 'A 目标：无阻尼 20000 步', '未发现提交或结果\n未发现 A20000 检查点')
    box(.674, .54, .31, .41, 'B1：单阻尼 b = 0.2', '未发现提交或结果')
    box(.674, .035, .31, .41, 'B2：多阻尼', 'b = 0, 0.05, 0.1, 0.2, 0.5, 1\n未发现提交或结果')
    for start, end in [((.24, .51), (.31, .51)), ((.562, .56), (.66, .745)), ((.562, .46), (.66, .24))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=17,
                                    color='#8193A1', lw=1.5, linestyle=(0, (4, 3))))
    ax.text(.276, .69, '续训目标', ha='center', fontsize=10.5, color=muted)
    ax.text(.608, .985, '计划衔接', ha='center', va='top', fontsize=10.5, color=muted)

    fig.text(.075, .662, '现有固定样本：真实角度与自主预测', fontsize=17, fontweight='bold', color=ink)
    fig.text(.075, .633, '仅为 10000 步中间结果，非 A20000 / B 结果', fontsize=14, color='#A2472D')
    grid = fig.add_gridspec(2, 2, left=.075, right=.968, bottom=.132, top=.59, hspace=.49, wspace=.19)
    handles = None
    axes = []
    for j, index in enumerate(indices):
        plot = fig.add_subplot(grid[j // 2, j % 2]); axes.append(plot)
        line_true, = plot.plot(z['time'], z['truth'][index, :, 0], color=truth_color, lw=2.3, label='真实轨迹')
        line_pred, = plot.plot(z['time'], z['pred'][index, :, 0], color=prediction_color, lw=2.1,
                              linestyle=(0, (5, 2.4)), label='自主预测（固定 kNN 读出）')
        handles = [line_true, line_pred]
        plot.set_title(f"能段 {int(z['energy_band'][index])}  ·  group {int(z['group_id'][index])}", loc='left', pad=10, color=ink)
        plot.set_xlabel('相对预测时间（秒）', labelpad=5)
        plot.set_ylabel('角度（rad）', labelpad=7)
        plot.set_xlim(0, 3.2)
        plot.set_xticks([0, .8, 1.6, 2.4, 3.2])
        # Physical angular units are identical; separate scales keep each energy band readable.
        ymax = max(np.max(np.abs(z['truth'][index, :, 0])), np.max(np.abs(z['pred'][index, :, 0])))
        plot.set_ylim(-1.14 * ymax, 1.14 * ymax)
        plot.grid(axis='both', color='#D9E1E7', alpha=.65, lw=.7)
        plot.axhline(0, color='#AAB7C0', lw=.7, zorder=0)
        plot.tick_params(labelsize=11, colors='#465766')
        plot.spines['left'].set_color('#AAB7C0'); plot.spines['bottom'].set_color('#AAB7C0')
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(.971, .638),
               frameon=False, fontsize=12.5, ncol=1, handlelength=3.1, labelspacing=.45)

    fig.text(.075, .067, '样本预先固定：每个初始能段选最小 group_id；全部完成 32 步，未按预测误差挑选。',
             color=muted, fontsize=11.5)
    fig.text(.075, .042, 'dt = 0.1 秒；初始 21 帧之后自主推进。横轴零点对应原始时刻 2.0 秒；各子图纵轴范围不同。',
             color=muted, fontsize=11.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Detect text/axes leaving the canvas before saving; visual QA is performed separately.
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        bbox = text.get_window_extent(renderer)
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.width + 1 or bbox.y1 > fig.bbox.height + 1:
            raise RuntimeError(f'Text outside canvas: {text.get_text()}')
    fig.savefig(OUTPUT, dpi=170)
    plt.close(fig)
    print(OUTPUT)


if __name__ == '__main__':
    draw()
