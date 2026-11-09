"""Two-page report; numerical cells come only from completed evaluation files."""
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
facts = json.loads((STUDY/'report_facts.json').read_text())
audit = json.loads((STUDY/'lewm_comparison_audit.json').read_text())
training = json.loads((ROOT/'studies/efficient3_fresh_20261104T101014Z/completion_audit_20261105.json').read_text())
out = ROOT/'output/pdf/dissipative_world_model_test_report_20261105.pdf'
out.parent.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('CN','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
W,H = A4
LEFT,RIGHT = 42,42
WIDTH=W-LEFT-RIGHT
INK=HexColor('#172a3b'); MUTED=HexColor('#596877'); TEAL=HexColor('#087f80')
RULE=HexColor('#d6e1e7'); PALE=HexColor('#f0f5f7'); ALERT=HexColor('#a2452d')
styles={
 'body': ParagraphStyle('body',fontName='CN',fontSize=10.1,leading=15.5,textColor=INK,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='CN',fontSize=8.4,leading=12.5,textColor=MUTED,wordWrap='CJK'),
 'caption':ParagraphStyle('caption',fontName='CN',fontSize=9,leading=13,textColor=MUTED,wordWrap='CJK'),
 'section':ParagraphStyle('section',fontName='CN',fontSize=12,leading=17,textColor=TEAL,wordWrap='CJK'),
 'lead':ParagraphStyle('lead',fontName='CN',fontSize=12,leading=18,textColor=ALERT,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='CN',fontSize=9.2,leading=12,textColor=INK,wordWrap='CJK'),
 'head':ParagraphStyle('head',fontName='CN',fontSize=9.2,leading=12,textColor=white,wordWrap='CJK'),
}
c=canvas.Canvas(str(out),pagesize=A4)
c.setTitle('耗散哈密顿世界模型：三子步方案测试简报')
c.setAuthor('实验记录')
y=0

def para(text,style='body',gap=7):
    global y
    p=Paragraph(text,styles[style]); _,h=p.wrap(WIDTH,H)
    if y-h<40: raise RuntimeError(f'Page overflow: {text[:60]} y={y} h={h}')
    p.drawOn(c,LEFT,y-h);y-=h+gap

def heading(text):
    global y
    y-=3;para(text,'section',6)

def table(headers,rows,widths):
    global y
    data=[[Paragraph(str(v),styles['head']) for v in headers]]
    data += [[Paragraph(str(v),styles['cell']) for v in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([
      ('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
      ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),5),
      ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
      ('ROWBACKGROUNDS',(0,1),(-1,-1),[white,PALE]),
      ('LINEBELOW',(0,-1),(-1,-1),.5,RULE),
    ]))
    _,h=t.wrap(WIDTH,H)
    if y-h<40: raise RuntimeError('Table overflow')
    t.drawOn(c,LEFT,y-h);y-=h+8

def page(number,subtitle):
    global y
    c.setFillColor(MUTED);c.setFont('CN',8.5)
    c.drawString(LEFT,H-29,'实验简报  /  2026-11-05  /  三子步 · 从零训练')
    c.drawRightString(W-RIGHT,H-29,f'{number} / 2')
    c.setFillColor(INK);c.setFont('CN',20)
    c.drawString(LEFT,H-62,subtitle)
    c.setStrokeColor(RULE);c.line(LEFT,H-77,W-RIGHT,H-77)
    c.setFont('CN',7.6);c.setFillColor(MUTED)
    c.drawString(LEFT,22,'实验 ID: efficient3_evaluation_20261105T023318Z')
    y=H-91

fmt=lambda v:f'{v:.3f}'
page(1,'测试完成；物理预测尚未达标')
para('三子步求解器已稳定完成训练和长时测试，但模型明显过度耗散。当前结果尚不能证明它学会了正确的物理阻尼。','lead',10)
para('本次从零训练完成 20,000 步，耗时约 16 小时 4 分；使用验证集选出的第 19,250 步检查点。主测试耗时 11 分 21 秒，1,408 / 1,408 条轨迹完整，无求解失败；45.06 万次状态推进均通过数值检查。',gap=8)
para('协议：无外力单摆，m=L=1、g=9.81、采样间隔 0.1 秒。一个模型从 21 帧历史（2 秒）推断环境；此后自主预测，不再输入真实图像。训练阻尼 b={0, 0.05, 0.1, 0.2, 0.5, 1}；测试含已见阻尼 768 条、插值 512 条（0.075 / 0.15 / 0.35 / 0.75）、外推 128 条（1.5）。物理读出器仅在训练标签上拟合、验证集选择，测试时冻结。','small',9)

heading('整体测试：误差越小越好')
rows=[]
for r in facts['main_table']:
    rows.append([f"{r['horizon_s']} 秒",fmt(r['model']['angle_rmse_rad']),fmt(r['fixed_frame']['angle_rmse_rad']),
                 fmt(r['equilibrium']['angle_rmse_rad']),fmt(r['model']['omega_rmse_rad_s']),fmt(r['equilibrium']['omega_rmse_rad_s'])])
table(['预测区间','本模型 θ','固定帧 θ','平衡点 θ','本模型 ω','零速度 ω'],rows,[66,89,89,89,89,WIDTH-422])
para('均为区间内累计 RMSE，汇总所有轨迹和未来时刻，不含起点；θ 使用圆周角差，单位 rad，ω 单位 rad/s。固定帧：保持起点读出角度、速度设为零；平衡点：角度和速度均设为零（已知物理先验）。','small',7)
para('本模型角度误差低于固定帧基线，但 8 秒及 32 秒已高于平衡点基线；速度也未稳定超过零速度基线。训练最长预测为 3.2 秒，而 1 秒内已出现明显误差，问题不只在长时外推。','body',8)

heading('LeWM 参考对比：相同测试轨迹，训练协议不同')
rows=[]
for r in facts['lewm_table']:
    rows.append([f"{r['damping']:g}",f"{r['horizon_s']:g} 秒",fmt(r['ours']['angle_rmse_rad']),
                 f"{fmt(r['lewm_mean']['angle_rmse_rad'])} ± {fmt(r['lewm_sample_sd']['angle_rmse_rad'])}",
                 fmt(r['ours']['omega_rmse_rad_s']),
                 f"{fmt(r['lewm_mean']['omega_rmse_rad_s'])} ± {fmt(r['lewm_sample_sd']['omega_rmse_rad_s'])}"])
table(['阻尼 b','区间','本模型 θ','LeWM θ','本模型 ω','LeWM ω'],rows,[43,49,71,133,71,WIDTH-367])
para('LeWM 为既有卷积多步适配版（W=3、lag=0），按已知阻尼选择专用模型；每种阻尼 3 个训练种子，表中为均值 ± 样本标准差。本模型仅 1 个种子。共重新评测 768 条模型轨迹；episode、真值、时间轴逐项一致，采用事先指定的冻结 MLP 读出。','small',7)
para('参考结果中 LeWM 更准确，尤其 b=0.2。但其历史训练混有外力数据、训练 28,140 步、历史 5 帧，读出标签量和拟合预算也不同，不能据此归因于架构优劣。正式比较仍需在本轮多阻尼数据上统一历史长度、训练和读出预算后重训。','small',0)
c.showPage()

page(2,'主要问题与下一轮修改顺序')
heading('已观察到：衰减过快，而且并非只发生在物理读出端')
b0_1=next(x for x in facts['b0_timepoints'] if x['time_s']==1)
para(f"无阻尼 b=0 本应保持物理能量；逐轨迹以真实初始能量归一化后，1 秒预测能量比的均值仅为 {100*b0_1['pred_mean_energy_fraction']:.1f}%。潜空间总能量 H 也在首个 0.1 秒由均值 34.17 降至 6.47，表明学到的内部动力学本身就在快速耗散。",gap=7)
img_h=WIDTH*2.9/9.6
c.drawImage(str(ROOT/'tmp/pdfs/physical_failure.png'),LEFT,y-img_h,width=WIDTH,height=img_h,mask='auto')
y-=img_h+4
para('左：128 条无阻尼轨迹的平均物理能量比。右：按元数据固定选择的中高能量样本，未按误差挑选；LeWM 固定 seed 0。时间均从 2 秒观察结束时起算。','small',8)
one=next(x for x in facts['teacher'] if x['relative_time_s']==1)
para(f"定位检查：固定选择 44 条轨迹，每个阻尼 × 能量档各 1 条。1 秒时若重新输入真实历史，角度 / 角速度 RMSE 为 {one['teacher_forced']['angle_rmse_rad']:.3f} / {one['teacher_forced']['omega_rmse_rad_s']:.3f}；同批自主预测为 {one['autonomous']['angle_rmse_rad']:.3f} / {one['autonomous']['omega_rmse_rad_s']:.3f}。这支持优先检查自主动力学和潜状态偏移。此处是 44 条轨迹的端点诊断，不是上页的累计测试成绩。",gap=8)
para('数值检查与物理正确性须分开：潜空间能量收支误差最大约 3.63×10<super>-14</super>，但解码后的物理能量约 28.2% 的相邻步会上升超过 10<super>-5</super> J。能量收支成立不等于物理阻尼学得正确。','small',8)

heading('实施事实与建议：先修训练约束，再考虑改模型')
para('1. 优先补齐动量目标。当前损失约束未来 q 和逐帧位移，没有直接约束未来 p；位移误差也未除以时间间隔。建议借鉴 LeHami [1]，用按时间间隔定标的中心 SG 导数构造训练用 p 目标，加入按训练统计量归一化的动量损失。推理起点仍只读历史；先验证无阻尼下 1-3 秒的振幅和相位。',gap=8)
para('2. 单独定位过度耗散。保留 D=LL<super>T</super> 的半正定结构；现实现允许 D=0，不存在强制正阻尼下界。做 D(c) 与 D(q,p,c) 对照，并以已知阻尼条件、无阻尼时关闭 D 作为诊断，区分环境推断与动力学误差。它们是消融诊断，不能替代由历史视频推断环境的主任务。',gap=8)
para('3. 约束读出的运动学一致性。尝试 θ̂=g(q)、ω̂=J<sub>g</sub>(q)p，使预测静止时不再读出非零速度；这可修正读出偏差，无法恢复已丢失的振荡。若仍失败，再研究角度友好潜坐标或可学习质量 M(q) [2]，同步调整动量定义、求解器并复核成本与能量性质。',gap=8)
para('边界：以上改法尚未训练验证；本轮未做固定权重的步长收敛比较，宜用 6 / 12 子步短测排除离散误差。本轮仅常数阻尼，尚须补充位置/速度相关摩擦、同协议 LeWM 和多个种子。当前测试已用于诊断，最终结论应在另留初始条件的独立评测上确认。','small',9)
para('[1] <link href="https://arxiv.org/html/2610.15267v1" color="#087f80">LeHami: Symplectic Latent Dynamics for World Models</link>，附录 C.4、D.1-D.2：因果初始化与训练用中心 SG 动量目标。<br/>[2] <link href="https://arxiv.org/html/2002.08860v3" color="#087f80">Dissipative SymODEN</link>，§3.2-3.5：质量矩阵、角度嵌入与半正定耗散结构；其输入为状态，非本实验的视频。','small',0)
c.save()
reader=PdfReader(str(out))
assert len(reader.pages)==2, len(reader.pages)
text='\n'.join(p.extract_text() for p in reader.pages)
for required in ('1,408','19,250','0.059','0.677','24.7','3.63','LeHami','SymODEN'):
    assert required in text, required
(ROOT/'tmp/pdfs/report_text.txt').write_text(text)
print(json.dumps({'pdf':str(out),'pages':len(reader.pages),'bytes':out.stat().st_size},ensure_ascii=False))
