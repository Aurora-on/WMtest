"""Build the evidence-grounded Chinese staged-training report from saved artifacts."""
from pathlib import Path
import json
from datetime import datetime
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.pagesizes import A4
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
REVIEW = PROJECT / 'studies/staged_damping_d2_launch_20261106T064931Z/review_20261108'
OUT = PROJECT / 'output/pdf/staged_damping_d2_technical_report_20261108.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
audit = json.loads((REVIEW/'completion_audit_20261108.json').read_text())
summary = json.loads((REVIEW/'result_summary_20261108.json').read_text())
recheck = json.loads((REVIEW/'independent_recheck.json').read_text())
history = json.loads((ROOT/'analysis/training_history_summary.json').read_text())
tech_path = ROOT/'analysis/technical_audit.json'
tech = json.loads(tech_path.read_text()) if tech_path.exists() else {}

pdfmetrics.registerFont(TTFont('CJK', '/Library/Fonts/Arial Unicode.ttf'))
pdfmetrics.registerFontFamily('CJK',normal='CJK',bold='CJK',italic='CJK',boldItalic='CJK')
NAVY=colors.HexColor('#153650'); TEAL=colors.HexColor('#167994'); ORANGE=colors.HexColor('#E77732')
MUTED=colors.HexColor('#64788C'); LIGHT=colors.HexColor('#EDF3F7'); LINE=colors.HexColor('#D8E2EA')
W,H=A4; WIDTH=W-88
styles={
 'title':ParagraphStyle('title',fontName='CJK',fontSize=25,leading=34,textColor=NAVY,spaceAfter=12,wordWrap='CJK'),
 'section':ParagraphStyle('section',fontName='CJK',fontSize=18,leading=25,textColor=NAVY,spaceAfter=13,wordWrap='CJK'),
 'sub':ParagraphStyle('sub',fontName='CJK',fontSize=11.4,leading=17,textColor=TEAL,spaceBefore=7,spaceAfter=5,wordWrap='CJK'),
 'body':ParagraphStyle('body',fontName='CJK',fontSize=10,leading=16.2,textColor=NAVY,spaceAfter=7,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='CJK',fontSize=8.4,leading=12.6,textColor=MUTED,spaceAfter=6,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='CJK',fontSize=8.5,leading=12.5,textColor=NAVY,wordWrap='CJK'),
 'headcell':ParagraphStyle('headcell',fontName='CJK',fontSize=8.5,leading=12.5,textColor=colors.white,wordWrap='CJK'),
 'eq':ParagraphStyle('eq',fontName='CJK',fontSize=10.5,leading=18,textColor=NAVY,backColor=LIGHT,borderPadding=8,spaceBefore=5,spaceAfter=12,wordWrap='CJK'),
}
story=[]
page_sections=[]
def clean(s):
    # Arial Unicode lacks modifier capital T and the matrix-order glyph.
    return str(s).replace('ᵀ','<super>T</super>').replace('⪰ 0','≥ 0（半正定）')
def p(s,kind='body'):
    story.append(Paragraph(clean(s),styles[kind]))
def section(n,title):
    if story: story.append(PageBreak())
    page_sections.append(title)
    p(f'{n:02d}  {title}','section')
def sub(s): p(s,'sub')
def table(headers,rows,widths=None):
    data=[[Paragraph(clean(s),styles['headcell']) for s in headers]]+[[Paragraph(clean(s),styles['cell']) for s in row] for row in rows]
    t=Table(data,colWidths=[WIDTH*x for x in widths] if widths else None,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,-1),(-1,-1),0.5,LINE)]))
    story.extend([t,Spacer(1,9)])
def fig(path,cap,maxheight=280):
    path=Path(path)
    with PILImage.open(path) as im: iw,ih=im.size
    scale=min(WIDTH/iw,maxheight/ih)
    story.append(Image(str(path),width=iw*scale,height=ih*scale,hAlign='CENTER'))
    story.append(Spacer(1,5));p(cap,'small')
def f(v,n=4): return f'{v:.{n}f}'
def metric(arm,key): return audit['arms'][arm]['metrics'][key]
def per(b): return audit['arms']['multi']['per_damping'][f'damping_{b}']

# 1. Executive summary.
p('二维耗散世界模型<br/>分阶段训练技术报告','title')
p('无阻尼预训练 → 单工况 / 多工况阻尼适配<br/>结果核查与下一轮对比实验设计','sub')
p('报告日期：2026-11-08　|　作业：47920　|　配置：q、p 各 2 维　|　训练 seed：0','small')
p('本次完整流程已完成。A 阶段从 10,000 步续训至总计 20,000 步；固定 A 的共享网络及非线性读出后，B-single 与 B-multi 并行各训练 20,000 步。全部评测于 2026-11-07 11:54（北京时间）结束，提交至完成约 21 小时 03 分。')
table(['最终模型','条件','角度 MAE / rad','角速度 RMSE / rad·s⁻¹'],[
 ['A 无阻尼','b = 0',f(metric('a','angle_mae_rad')),f(metric('a','omega_rmse_rad_s'))],
 ['B 单工况','b = 0.2',f(metric('single','angle_mae_rad')),f(metric('single','omega_rmse_rad_s'))],
 ['B 多工况','六种 b 汇总',f(metric('multi','angle_mae_rad')),f(metric('multi','omega_rmse_rad_s'))],
],[.26,.20,.25,.29])
p('口径：精确第 20,000 步 checkpoint；64 组验证初态；未来 0.1-3.2 秒平均，非末帧误差。多工况为相同 64 组初态 × 6 种阻尼，不能视为 384 个独立初态。','small')
fig(ROOT/'figures/pipeline.png','图 1. 实际完成的训练流程。B 阶段的 20,000 步是各自新优化器的更新数。',190)
sub('核心判断')
p('① 本验证范围内未触发持续停止判据，全部损失与梯度有限，数值求解失败计数为零。② 在相同 b = 0.2 上，单工况的 3.2 秒平均角度误差比多工况低 47.4%，但短时角度误差并非处处更低。③ 强阻尼 b = 1 的物理读出也退化，现有误差不能全部归因于阻尼网络。')
p('目前证据支持“流程正确执行，短时验证预测可用且单工况适配有效”。尚不支持“早停问题在长时彻底解决”“耗散结构优于外部方法”或“分阶段训练优于联合训练”。本报告后半部分给出验证这些主张的实验方案。')

# 2. Setup.
section(2,'实验设置与实际训练范围')
sub('数据与任务')
p('任务是从 RGB 观测历史初始化潜状态，随后在无未来图像反馈的条件下自主预测被动单摆。使用项目的 smooth_v1 受控耗散变体，不能与 LeHami 原论文的原始 benchmark 直接混称。实际动力学为质点摆，m = 1、L = 1、g = 9.81；本轮动作恒为零。')
table(['项目','实际配置 / 解释'],[
 ['数据划分','训练 384 组；验证 64 组。测试集及未见阻尼未参与训练、读出选择或当前结论。'],
 ['工况','A：b = 0；B-single：b = 0.2；B-multi：b ∈ {0, 0.05, 0.1, 0.2, 0.5, 1}。'],
 ['观测与时间','64 × 64 RGB；Δt = 0.1 s；初始化历史 21 帧，覆盖 2.0 s；速度局部历史 5 帧。'],
 ['仿真与推演','真值 RK4 每帧 20 子步；模型每帧 3 个名义子步，困难求解可二分。观测时间间隔与训练时域未改变。'],
 ['潜状态','q ∈ R²、p ∈ R²，总相空间维数为 4；多工况环境上下文额外为 4 维。'],
],[.24,.76])
sub('训练配置')
table(['项目','实际配置'],[
 ['优化','batch = 32；encoder 学习率 2×10⁻⁵；dynamics 学习率 10⁻⁴；weight decay = 10⁻⁴；梯度裁剪 = 1。'],
 ['调度','warmup 500 步，cosine 下限比例 0.1；A 保留原有优化器 / 随机数 / sampler；两条 B 分支均从新优化器第 0 步启动。'],
 ['课程','1-1,000 步：K=4；1,001-4,000：K=8；4,001-10,000：K=16；10,001-20,000：K=32。K×0.1 s 为预测时域。'],
 ['模块','A 训练 encoder、initializer、potential；B-single 仅训练 damping；B-multi 训练 damping 与 environment。共享参数与 buffers 均冻结。'],
 ['验证 / 选点','每 250 步验证 K=8、K=32；本报告使用 exact last@20,000，未改选更早的 best checkpoint。'],
],[.24,.76])
p('二维是本轮既定配置，并非充分维数的证明。此前维数实验与本轮阻尼适配的训练条件不同，本报告不把维数结果并入同一方法排名。','small')

# 3. Method.
section(3,'模型、学习目标与几何边界')
sub('保守骨架与阻尼适配')
p('A 从图像学习位置表示 q、初始潜速度 / 动量 p 以及势函数 V。潜质量 M=I₂ 是模型假设，不等于已校准的物理质量。B 在相同骨架上增加正半定阻尼；multi 的 c 由初始历史产生并在 rollout 内固定，仅作用于 D。')
p('H(q,p) = ½ pᵀp + V(q)<br/>dq/dt = p<br/>dp/dt = −∇V(q) − D(q,p,c)p<br/>D = LLᵀ ⪰ 0；A 中 D = 0；单工况不使用 c。','eq')
p('连续形式满足 dH/dt = −pᵀDp ≤ 0。本次 frozen_dg 使用逐层对称割线离散梯度及隐式求解，能量账本含求解残差功与链式误差，需按有限容差验收。约束对象为内部 H，不自动等于读出机械能 E(θ,ω) 的平衡。B 初始化 D = 0.01I，避免 L=0 时 LLᵀ 参数化的零梯度。')
sub('LeHami 式损失及监督边界')
p('目标 q* 来自未来图像编码，p* 为五点局部二次拟合导数，二者保持可微；额外未来 2 帧只为末端导数提供监督。p* 不是标注物理角速度。物理角度与角速度不用于世界模型反向传播，只用于独立读出校准和诊断。')
p('L = Lq + Lp + 0.25 L<SUB>SIGReg</SUB> + 0.001 L<SUB>init</SUB><br/>Lq = MSE(q̂, q*)；Lp = MSE(τp̂, τp*)；τ = 0.1 s<br/>SIGReg 作用于目标 [q*, τp*]；L<SUB>init</SUB> 约束初始修正量 δp₀。','eq')
p('B 冻结 encoder / initializer 后，SIGReg 和初始修正正则仍计入标量 loss，但对本阶段可训练的 D / environment 无梯度；其数值仍随 batch 变化。应优先看位置-动量预测损失、运动门槛和物理预测，不以总 loss 跨阶段排名。')
sub('非线性读出改善了什么，尚未证明什么')
p('若 q=e(θ) 为弯曲嵌入，则 d²q/dt²=e′(θ)·d²θ/dt²+e″(θ)·ω²。第二项随速度平方改变，是编码几何引起的加速度。固定单位动能、只依赖 q 的势力未必能充分表示它；单纯增加维数或换成非线性读出均不能自动消除这项兼容性问题。')
p('kNN 读出可以更准确地将弯曲潜状态映射回 θ、ω，改善显示与物理评测；它没有更改潜动力学。当前轨迹变好说明整个“潜预测 + 读出”链路更有效，不能据此宣称几何问题已经从结构上解决。')

# 4. Evaluation.
section(4,'评测协议与完整性核查')
sub('读出只用训练划分拟合')
p('在 A20k 的 384 个训练组内部，288 组拟合标准化和 kNN，96 组选择邻居数；候选 k={1,5,25,50}，选中 k=5、等权平均。特征为 K0 因果历史得到的 [q,p]。A20k、B-single、B-multi 使用逐字节一致的同一读出。')
p('真实历史读出诊断（下文记为 teacher）在每个时刻重新观察真实历史；自主预测只在起点初始化。前者衡量视觉表示与固定读出的适用性，不是未来自主预测，也不是可从总误差中直接相减的独立误差分量。')
sub('指标定义')
table(['指标','计算口径'],[
 ['角度 MAE','对 Δθ 使用 atan2(sin Δθ, cos Δθ) 包裹至圆周最短差，再取绝对值平均。主表统计未来第 1-32 帧，不含锚点。'],
 ['角速度 RMSE / NRMSE','RMSE 为物理角速度误差的均方根；NRMSE 以拟合训练数据角速度的标准差归一化，并非每个阻尼各自重新缩放。'],
 ['持续停止','在约 1 秒的真值动态窗口中，预测速度 RMS 小于真实速度 RMS 的 10% 判为停止。真值动态阈值为训练速度尺度的 10%；排除真实自然静止。'],
 ['物理能量误差','E = ½mL²ω² + mgL(1−cosθ)。未来能量 MAE 以真实锚点能量归一化并取逐轨迹中位数；阻尼下能量下降本身不是错误。'],
],[.24,.76])
sub('执行与溯源检查')
table(['检查','结果'],[
 ['步骤与历史','A 新增 10,000 条、两条 B 各 20,000 条更新；连续无缺步；checkpoint / sampler / optimizer 步数符合计划。'],
 ['冻结与数据','两条 B 的所有共享 A state_dict 张量逐位一致；三份 readout 哈希一致；test 仅做划分、初态、时钟和动作配对完整性检查，未做性能评测。'],
 ['数值与文件','全部日志 loss / gradient 有限；求解失败计数为 0；评测文件与 complete 清单哈希匹配。'],
 ['独立复算','从保存的 NPZ 重算指标，与报告最大差 5.55×10⁻¹⁶。Slurm accounting 未启用，完成依据为流水线状态、连续历史、checkpoint 及评测产物。'],
],[.24,.76])
p('这四类检查证明执行与记录可信，不等价于长期物理正确性。当前样本来自验证集、一个训练种子；不报告跨 seed 置信区间或测试泛化成绩。','small')

# 5. Convergence.
section(5,'训练收敛与计算成本')
fig(ROOT/'figures/training_convergence.png','图 2. 训练曲线由服务器全部 50,000 条新增更新汇总：训练为 250 步均值，验证保留每次结果。两者可能采用不同 K 分布，不能把曲线间距直接解释为泛化差距。',270)
table(['阶段','本次新增更新','训练耗时','最终验证分数','验证门槛通过'],[
 ['A',10000,'7 h 34 min','0.018211','40 / 40'],
 ['B-single',20000,'12 h 36 min','0.003117','80 / 80'],
 ['B-multi',20000,'13 h 15 min','0.008427','77 / 80'],
],[.14,.18,.22,.23,.23])
p('最终验证分数是潜位置-动量预测误差，不是角度误差。多工况仅第 250、500、750 步未通过运动质量门槛，最终第 20,000 步通过；这三次不通过不是求解器数值失败。A 和 single 的最终门槛也通过。')
p('A 最优验证点为 17,250 步，single 为 20,000 步，multi 为 19,750 步；本次仍严格评估计划指定的第 20,000 步。验证曲线显示训练有效，但不能凭末段曲线宣称所有工况已达到最优。')
sub('实际资源消耗及预算解释')
p('两条 B 在各自一张 L40 上并行运行，不能将三段时长相加当作等待时间。三段训练进程累计单卡运行时间约 33.42 小时，含周期验证；不是纯 GPU kernel 利用时间。按 2 张 L40 × 提交至完成 21.05 小时计算，预约预算上界约 42.10 GPU 小时，缺 Slurm 实际起止记录，不能视为精确占用。A 的前 10,000 步不计入本次耗时。')
p('后续方法的成本不能由此直接外推：显式预测器、神经 ODE 和隐式求解器每步开销不同。建议先测各方法 500-1,000 步，再同时报告优化更新数、实际 GPU 小时、参数量和推理耗时。')

# 6. Autonomous prediction.
section(6,'自主预测与同工况配对比较')
fig(ROOT/'figures/horizon_errors.png','图 3. 每个预测时刻的误差曲线，使用相同的 64 组验证初态。曲线上 3.2 秒点是末帧误差；正文主表是 0.1-3.2 秒整段平均，两者不可混用。',290)
sub('单工况与多工况：只在相同 b 上比较')
matched=recheck['matched_b0p2']['horizons']
table(['未来窗口','single 角度 MAE','multi 角度 MAE','single / multi 速度 RMSE'],[
 ['0.1-0.8 s',f(matched['8']['single']['angle_mae_rad']),f(matched['8']['multi']['angle_mae_rad']),f(matched['8']['single']['omega_rmse_rad_s'])+' / '+f(matched['8']['multi']['omega_rmse_rad_s'])],
 ['0.1-3.2 s',f(matched['32']['single']['angle_mae_rad']),f(matched['32']['multi']['angle_mae_rad']),f(matched['32']['single']['omega_rmse_rad_s'])+' / '+f(matched['32']['multi']['omega_rmse_rad_s'])],
],[.19,.24,.24,.33])
p('b=0.2 的两组真值、初态、teacher 特征及读出完全一致。3.2 秒窗口中，single 的角度 MAE / 速度 RMSE 分别低 47.4% / 54.1%；0.8 秒窗口中，multi 的角度误差略小。因此更准确的结论是：单工况在当前较长窗口的累积误差更低。')
p('逐组检验也支持这一差距：3.2 秒时 single 在 60/64 组（93.75%）上角度 MAE 更低。single 的逐组 MAE 中位数 / P90 为 0.0344 / 0.0705 rad；multi 同 b 为 0.0552 / 0.1654 rad。0.8 秒时 single 仅在 32/64 组更低。')
sub('无阻尼续训的改善')
p('A10k 与 A20k 使用相同验证初态与真值。3.2 秒平均角度 MAE 由 0.1937 降到 0.0687 rad，下降 64.5%；角速度 RMSE 由 1.1016 降到 0.2965 rad/s，下降 73.1%。这期间训练课程从 K=16 扩大到 K=32，且各自重新校准了读出，故不能将改善完全归因于“多训练了 10,000 步”。')
p('A、single、multi 分别有 0/1472、0/1472、0/8470 个动态窗口触发持续停止判据。它是本时域和该阈值下的观测，不保证 8 秒或 32 秒仍然不停止。','small')

# 7. Per-condition results.
section(7,'多工况分解与读出适用性')
table(['阻尼 b','自主角度 MAE','自主速度 RMSE','teacher 角度 MAE','速度 RMS 比'],[
 [b,f(per(b)['metrics']['angle_mae_rad']),f(per(b)['metrics']['omega_rmse_rad_s']),f(per(b)['teacher']['angle_mae_rad']),f(per(b)['metrics']['velocity_rms_ratio'],3)] for b in ['0','0.05','0.1','0.2','0.5','1']
],[.12,.23,.24,.24,.17])
fig(ROOT/'figures/summary_metrics.png','图 4. 所有工况使用相同 64 组初态。teacher 是真实历史读出诊断；A 与 single 标记只是对应 b 的额外比较点。所有指标均为未来 0.1-3.2 秒平均。',265)
p('多工况在 b=0 上的角度 MAE 为 0.1027 rad，高于 A 的 0.0687；这说明增加阻尼适配后，保守工况也会受到影响。是否来自环境识别、非零阻尼或潜空间近似误差，需要针对性诊断，当前尚不能归因。')
p('b=1 的自主角度 MAE 为 0.1563 rad；teacher 角度 MAE 已达 0.1061 rad，超过预设 0.1 rad 读出诊断阈值。全工况汇总 teacher 仅 0.0283 rad，会掩盖这一边界。应先检查训练读出对低速 / 小振幅状态的覆盖，再判断阻尼学习是否错误。')

# 8. Demonstration.
section(8,'固定初态的轨迹演示')
fig(REVIEW/'figures/multi_damping_six_conditions.png','图 5. 多工况模型的六种阻尼自主预测。蓝实线为真值，橙虚线为预测；固定 group_id=10000384，初始能段 2，按该能段最小组号预先选定，未按误差挑选。',470)
p('该样本在弱至中等阻尼下较好跟随相位与幅度；强阻尼 b=1 在后段出现偏差及读出跳变。图像展示忠实保留这些偏差，不代表所有验证轨迹。横轴零点是原始轨迹 t=2.0 s，即 21 帧历史的末端。')
p('随附 GIF 对比 A(b=0)、single(b=0.2)、multi(b=0.2)、multi(b=1)。摆杆由保存的物理角度程序化重绘，不是像素解码器生成的图像；没有未来真值反馈或本轮重新推理。')
p('动图文件：review_20261108/figures/staged_prediction_demo.gif。图表复现脚本和输入哈希记录在对应 figures 目录。','small')

# 9. Interpretation.
section(9,'结果解释、局限与优先诊断')
sub('真实物理能量应与真值比较')
table(['模型 / 条件','未来能量相对 MAE 中位数','解读'],[
 ['A，b=0','4.96%','内部守恒不等价于读出机械能严格守恒。'],
 ['single，b=0.2','4.45%','正常衰减不能作为能量漂移失败；需比较真值能量。'],
 ['multi，全工况','11.43%','汇总覆盖弱至强阻尼，需分工况解释。'],
 ['multi，b=1','37.34%','与该工况读出退化同时存在，归因尚未分离。'],
],[.26,.29,.45])
p('A 的“最大相对能量漂移”的轨迹中位数为 10.59%，高于上述平均能量误差；两者分别是最坏时刻的相对变化和对真值的平均误差，不可互换。阻尼下本就存在真实能量下降，不应强求这一漂移接近零。')
sub('已观察事实与待验证假设')
table(['已观察事实','可能原因 / 必要验证'],[
 ['强阻尼 teacher 与自主误差同时变大','先比较各工况潜特征至 A 读出训练集的近邻距离和低速覆盖；可做补充读出校准敏感性分析，但保留当前冻结读出结果。'],
 ['multi 在 b=0 也劣于 A','评估 D 的特征值与 pᵀDp、上下文对 b 的可分性；对同一 checkpoint 关闭 D，只作为反事实诊断。'],
 ['single 的长窗口误差低于 multi','可能有跨工况容量 / 上下文或样本曝光竞争。对等曝光预算及上下文消融后才能归因。'],
 ['短时持续停止未出现','扩展自主预测时域，持续监控速度、相位及包络；不能仅看总 loss 或一个示例。'],
],[.34,.66])
p('目前没有“联合训练 vs 分阶段训练”、简单阻尼 vs 状态依赖阻尼的匹配对照，也未进行多 seed、未见阻尼或测试集验收。不能据此宣称分阶段策略必要、D(q,p) 必要，或已优于 LeHami / LeWM。')
p('不建议直接把当前 D 数值解释成物理阻尼系数 b：它作用于学习到的潜坐标，尺度与几何不同。可解释性应首先建立在自主物理轨迹、衰减包络及功率误差上。')

# 10. External baselines.
section(10,'下一步应比较哪些方法')
p('建议核心外部比较保留四种方法，各自回答一个不同问题。先跑通 LeHami / LeWM 的同数据适配，再加入 H-JEPA；CpAE 用作表征与通用连续动力学的互补参照。论文及官方代码已于 2026-11-08 核查；下面是拟议实验，不是已完成的对比结果。')
table(['方法','回答的问题','实施与公平性','优先级'],[
 ['LeHami [1]','在视觉保守 Hamiltonian 骨架上，显式耗散适配带来了什么？','同一 RGB 数据与被动工况；保持原版不含阻尼。原生协议复现与匹配数据适配分开报告。','必做'],
 ['LeWorldModel / LeWM [2]','没有物理结构的强 JEPA 预测器，能否同样拟合并泛化？','保留其方法目标和训练组成；统一数据、可见历史及读出评测。记录计算预算和模型容量。','必做'],
 ['H-JEPA [3]','已有视觉 port-Hamiltonian / PSD 耗散 JEPA 能做到什么？','保留 PIC / Bures 等原有组成；被动零动作是适配场景。其显式 Euler 不等于离散能量保证。','耗散外部主对照'],
 ['CpAE [4]','连续性友好的视觉表示 + 通用 latent Neural ODE 是否足够？','保留重建与连续性训练，属于不同训练范式；原生两帧输入和当前 21 帧历史须明确对齐或分轨比较。','第二批'],
],[.18,.28,.39,.15])
sub('另设一个便宜且可解释的内部结构对照')
p('固定同一 encoder、initializer、读出、训练划分与 LeHami 式预测损失，把阻尼项替换为参数量相近的无约束残差力 MLP（或通用潜 Neural ODE）。它检验“限制为 PSD 耗散”是否比相近容量的一般修正更有价值。应明确命名为本项目内部对照，不能冒称某篇论文的完整复现。')
sub('暂不作为第一批主基线')
p('仅用真实物理状态训练的 HNN / 耗散 HNN 可作为状态观测上界，却不能与 RGB 模型无说明地排在同一主表。其他离散拉格朗日或物体中心方法可用于相关工作讨论；在尚未确认端到端官方实现或与单摆输入不匹配时，不将其列为首轮训练前置条件。')
p('自己的消融继续固定 LeHami 式损失，不重启此前已取消的“当前对照损失”实验。外部方法若被强制替换成同一损失，就应标为 adapted 版本；原方法的结果需要保留其定义性的训练目标。')

# 11. Next protocol.
section(11,'下一轮实验设计与验收顺序')
sub('第一步：先完成当前模型的诊断与长时验证')
p('保留 Δt=0.1 s、训练课程和 checkpoint，增加未来 8 s、32 s 纯自主评测。现有 val 为 81 帧，扣除初始 2 s 后仅支持未来 6 s；需同 64 组初态生成 101 / 341 帧的 eval-only 轨迹，并验证前 81 帧一致。评测器须分别记录原训练数据身份与新评测 SHA，不能关闭哈希检查或挪用 test 选型。先补 b=1 的读出覆盖与 D=0 反事实诊断。')
sub('第二步：用最少消融检验方法主张')
table(['对照','固定部分','要验证的主张'],[
 ['D=0；常数 γI；D(q)；D(q,p)','同一 A20k、读出、B 数据、优化预算与损失；multi 的 γ 可由相同上下文生成。','是否需要耗散、状态依赖与速度依赖；简单阻尼是否已足够。'],
 ['B 冻结 vs B 联合微调','优先从同一 A20k 开始，再各训练 B20k；读出分别校准。后续从头联合训练需匹配 40k 总预算与数据曝光。','冻结共享表示是否限制适配；从头联合对照另检验阶段策略。'],
 ['multi 有 / 无环境上下文','保持可见原始历史和其他网络相近；真实 b 不作输入。','跨工况能力是否依赖从历史识别环境。'],
 ['PSD 阻尼 vs 无约束残差力','同一冻结 A、相近参数量与训练预算。','收益来自耗散约束还是额外拟合容量。'],
],[.30,.38,.32])
sub('第三步：外部方法的两条评测轨道')
p('原生轨：保留各方法推荐结构 / 目标及官方读出，用于复现可信度，不能跨数据表直接排名。匹配轨：统一同一 RGB 生成器、group 划分、Δt、预测长度、初始可见 raw-frame 范围和读出校准协议；方法间分别拟合自己的读出，不跨潜空间共享 A 的读出权重。')
p('先做 b=0 和单 b=0.2 的 seed=0 试跑，再扩展到多工况。首轮完成并冻结协议后，主结果采用至少 3 个训练种子、相同初态配对，报告均值、标准差及逐组分布。多工况输入不能偷偷包含真实 b；测试插值 b={0.075,0.15,0.35,0.75}、外推 b=1.5 仅在方案锁定后验收。')
sub('预算与统计的必要约束')
p('完整 A→B 单路径总预算为 40k 步，不能只计本次新增更新。主比较同时列实际 GPU 小时并保留方法原生潜维度，容量匹配另列。多工况把曝光分摊到六种 b，须区分等总预算与等每工况曝光。训练稳定性采用多 seed 统计，轨迹重采样不能替代多 seed。')

# 12. Sources.
section(12,'证据索引与参考文献')
p('结果来源为作业 47920 的冻结输出、2026-11-08 下载核查包，以及本次只读获取的训练历史摘要。没有为本报告重训、重新拟合读出或改选 checkpoint。新增图表直接使用保存的 NPZ / JSON；拟议对比实验尚未提交。')
table(['本地证据（相对于实验目录）','用途'],[
 ['studies/staged_damping_d2_launch_20261106T064931Z/<br/>review_20261108/completion_audit_20261108.json','步骤、冻结状态、数值健康、时间、评测哈希。'],
 ['同目录 independent_recheck.json / result_summary_20261108.json','独立物理指标复算、A10k 对比、b=0.2 配对。'],
 ['同目录 run/milestones/*/evaluation/','实际 trajectories.npz、summary、readout 与 provenance；精确 step20000。'],
 ['reports/staged_damping_d2_20261108/analysis/','训练历史 250 步摘要；技术代码审计；基线来源记录。'],
 ['reports/staged_damping_d2_20261108/figures/','绘图脚本、指标复算、输入哈希及所有新图表。'],
],[.64,.36])
p('共同读出 SHA-256：b043b1899a979046a6fbad4e57f183436cdcbeb965a9c95c0ed11d8215d66390。完整 checkpoint、源代码、数据与产物哈希见上述审计文件；短哈希不作为独立完整性校验。','small')
sub('原始论文与官方实现')
refs=[
 ('[1] LeHami: Symplectic Latent Dynamics for World Models', 'https://arxiv.org/abs/2610.15267', 'arXiv:2610.15267v1。官方原生协议与本报告的 K0 标准化 kNN 不同。'),
 ('[2] LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels', 'https://arxiv.org/abs/2603.19312', 'arXiv:2603.19312v3。无显式物理结构的通用视觉 JEPA。'),
 ('[3] Hamiltonian JEPA: Action-Conditioned World Models with an Inherited Control State', 'https://arxiv.org/abs/2609.33497', 'RGB / action 条件的 port-Hamiltonian 结构；官方实现采用显式 Euler。'),
 ('[4] Continuity-Preserving Convolutional Autoencoders for Learning Continuous Latent Dynamical Models from Images', 'https://arxiv.org/abs/2502.00754', 'ICLR 2025；连续性表征、图像重建与 latent Neural ODE，含阻尼单摆。'),
]
for title,url,desc in refs:
 p(f'<link href="{url}" color="#167994">{title}</link>','sub');p(desc,'small')
p('官方实现：<link href="https://github.com/rominay/LeHami" color="#167994">LeHami</link>；<link href="https://github.com/lucas-maes/le-wm" color="#167994">LeWM</link>；<link href="https://github.com/tmz-lab/hjepa" color="#167994">H-JEPA</link>；<link href="https://github.com/Aiqing-Zhu/CpAE" color="#167994">CpAE</link>。固定 commit、源码行号与适配边界见 analysis 中的来源记录；公开实现不代表当前数据适配已验收。','small')

def footer(canvas,doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE);canvas.setLineWidth(.5);canvas.line(44,40,W-44,40)
    canvas.setFont('CJK',7.5);canvas.setFillColor(MUTED)
    canvas.drawString(44,27,'WM · 二维分阶段耗散训练 · 2026-11-08')
    canvas.drawRightString(W-44,27,f'{doc.page}')
    if doc.page>1:
        canvas.drawString(44,H-27,'TECHNICAL REPORT  /  STAGED DAMPING')
    canvas.restoreState()

doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=48,bottomMargin=53,title='二维耗散世界模型分阶段训练技术报告',author='WM research project',subject='Job47920: completed staged training, evidence audit and baseline design')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader
reader=PdfReader(str(OUT))
texts=[page.extract_text() for page in reader.pages]
assert len(reader.pages)==12, f'Unexpected pagination: {len(reader.pages)}'
assert all('\x00' not in t and '\ufffd' not in t for t in texts), 'Missing glyph in PDF text'
manifest={'output':str(OUT),'pages':len(reader.pages),'section_count':12,'page_text_lengths':[len(t) for t in texts],'inputs':{'audit':str(REVIEW/'completion_audit_20261108.json'),'history':str(ROOT/'analysis/training_history_summary.json')},'created_at':datetime.now().astimezone().isoformat()}
(ROOT/'report_build_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(ROOT/'report_extracted_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{t}' for i,t in enumerate(texts)))
print(json.dumps(manifest,ensure_ascii=False))
