from pathlib import Path
import json,html,base64,hashlib,sys
sys.path.append("/Users/weizhixin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.pagesizes import A4
from PIL import Image as PILImage
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
PDF=PROJECT/'output/pdf/four_method_comparison_20261109.pdf'
METHODS=['ours','lehami','lewm','hjepa'];LABEL={'ours':'我们的二维模型','lehami':'LeHami','lewm':'LeWM','hjepa':'H-JEPA'}
COLOR={'ours':'#d97a32','lehami':'#24758f','lewm':'#2b8a63','hjepa':'#b94e59'}
FONT='/Library/Fonts/Arial Unicode.ttf'
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
DATA={m:{s:json.loads((ROOT/f'data/runs/{m}/{m}_seed0/{s}/evaluation/summary.json').read_text()) for s in ['a','single','multi']} for m in METHODS}
def group(m,s,h='32',g='all'):return DATA[m][s]['evaluation']['val']['rollout'][h]['groups'][g]
def metrics(m,s):return group(m,s)['metrics']
def textfmt(v,d=3):return f'{v:.{d}f}'
CONTENT=json.loads((ROOT/'report_content.json').read_text())
# Static scientific aggregate plot, six conditions with equal initial groups.
fig,axs=plt.subplots(1,2,figsize=(10,3.3),layout='constrained');bs=[0,.05,.1,.2,.5,1]
for m in METHODS:
 for ax,key,title,unit in zip(axs,['angle_mae_rad','omega_rmse_rad_s'],['多阻尼：角度误差','多阻尼：角速度误差'],['MAE / rad','RMSE / rad/s']):
  vals=[group(m,'multi',g=f'damping_{b:g}')['metrics'][key] for b in bs]
  ax.plot(range(6),vals, 'o--' if m=='hjepa' else 'o-', color=COLOR[m],label=LABEL[m]+('*' if m=='hjepa' else ''),lw=1.8,ms=4)
  ax.set(xticks=range(6),xticklabels=[str(b) for b in bs],xlabel='阻尼系数 b',ylabel=unit,title=title);ax.grid(alpha=.18)
axs[0].legend(fontsize=8,frameon=False);fig.savefig(ROOT/'figures/per_damping.png',dpi=180);plt.close(fig)
# Compact data table also available for audit/reuse.
rows=[]
for m in METHODS:
 for s in ['a','single','multi']:
  g=group(m,s);x=g['metrics'];rows.append(dict(method=m,stage=s,angle_mae_rad=x['angle_mae_rad'],omega_rmse_rad_s=x['omega_rmse_rad_s'],teacher_gate=g['teacher_quality_gate_passed'],stop_fraction=x['trajectory_with_stop_fraction']))
(ROOT/'comparison_metrics.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
# PDF typography.
pdfmetrics.registerFont(TTFont('CJK',FONT));pdfmetrics.registerFontFamily('CJK',normal='CJK',bold='CJK',italic='CJK',boldItalic='CJK')
NAVY=colors.HexColor('#173a4a');GRAY=colors.HexColor('#5c6c76');TEAL=colors.HexColor('#137d72');LIGHT=colors.HexColor('#f1f5f7');W,H=A4;WIDTH=W-80
styles={k:ParagraphStyle(k,fontName='CJK',fontSize=size,leading=lead,textColor=col,spaceAfter=space,wordWrap='CJK') for k,size,lead,col,space in [('title',24,32,NAVY,12),('h',18,25,NAVY,12),('sub',11,17,TEAL,8),('body',10,16,NAVY,8),('small',8.5,13,GRAY,7),('cell',8.5,12.8,NAVY,0),('thead',8.5,12.8,colors.white,0)]}
story=[];web=[]
def p(text,kind='body'):
 story.append(Paragraph(text,styles[kind]));web.append(f'<p class="{kind}">{text}</p>')
def section(title):
 if story:story.append(PageBreak())
 p(title,'h')
def table(headers,rows,widths):
 data=[[Paragraph(str(x),styles['thead']) for x in headers]]+[[Paragraph(str(x),styles['cell']) for x in row] for row in rows]
 t=Table(data,colWidths=[WIDTH*w for w in widths],repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.extend([t,Spacer(1,10)])
 web.append('<table><thead><tr>'+''.join(f'<th>{x}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>')
def fig(path,caption,maxheight=250):
 path=Path(path)
 with PILImage.open(path) as im:iw,ih=im.size
 scale=min(WIDTH/iw,maxheight/ih);story.append(Image(str(path),width=iw*scale,height=ih*scale));p(caption,'small')
 web.append('<figure><img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"><figcaption>'+caption+'</figcaption></figure>')
def gif(path,caption):
 path=Path(path);web.append('<figure><img src="data:image/gif;base64,'+base64.b64encode(path.read_bytes()).decode()+'"><figcaption>'+caption+'</figcaption></figure>')
p('视觉动力学模型对比报告','title');p('我们的二维模型 · LeHami · LeWM · H-JEPA','sub');p('2026-11-09｜已完成训练与验证评测｜seed 0｜未来 3.2 秒','small')
for text in CONTENT['summary']:p(text)
table(['方法','无阻尼','单阻尼 b=0.2','多阻尼汇总'],[[LABEL[m]]+[f"{metrics(m,s)['angle_mae_rad']:.3f} / {metrics(m,s)['omega_rmse_rad_s']:.3f}"+(' *' if not group(m,s)['teacher_quality_gate_passed'] else '') for s in ['a','single','multi']] for m in METHODS],[.24,.24,.26,.26])
p('表中为角度 MAE（rad）/ 角速度 RMSE（rad/s），均越小越好。统计未来 0.1-3.2 秒整段，不是第 3.2 秒的末帧误差。* 表示观测状态读出质量门未通过，该数值仅作诊断。','small')
p('评测采用相同验证初态、相同预测时长和各自的非线性读出。我们的模型本次只重新校准读出并评测，世界模型权重没有更新；不混用此前 A-only 冻结读出结果。')
p('证据范围','sub');p('每阶段 20,000 步；训练 384 组、验证 64 组初态。多阻尼为同一批 64 组初态的六种阻尼版本，不是 384 个独立初态。当前为单种子验证结果，尚未开启测试集、未见阻尼或长时预测评测。')
section('01  工况差异与运行成本')
fig(ROOT/'figures/per_damping.png','六种阻尼分别统计。* H-JEPA 的多阻尼整体读出未通过质量门，虚线仅供诊断；不能直接据此评价动力学算法。',205)
for text in CONTENT['conditions']:p(text)
table(['方法','参数量','三阶段流程耗时','GPU 峰值'],CONTENT['cost_rows'],[.24,.19,.34,.23])
for text in CONTENT['cost_notes']:p(text,'small')
for idx,m in enumerate(METHODS,2):
 section(f'{idx:02d}  {LABEL[m]}：结果与演示')
 p(CONTENT['method'][m]['mechanism'])
 table(['工况','角度 MAE','角速度 RMSE','停止轨迹率','读出检查'],[[sname,f"{metrics(m,s)['angle_mae_rad']:.3f}",f"{metrics(m,s)['omega_rmse_rad_s']:.3f}",f"{100*metrics(m,s)['trajectory_with_stop_fraction']:.1f}%",'通过' if group(m,s)['teacher_quality_gate_passed'] else '未通过'] for s,sname in [('a','无阻尼'),('single','单阻尼'),('multi','多阻尼')]],[.21,.18,.21,.20,.20])
 fig(ROOT/f'figures/{m}_trajectories_compact.png','同一初态的示例：左无阻尼，中单阻尼 b=0.2，右多阻尼模型在 b=0.1。黑色为真值，彩色为预测。曲线直接使用保存的读出状态；误差表采用环绕角差。',250)
 for text in CONTENT['method'][m]['findings']:p(text)
 p('动图见配套 HTML 的本方法章节；预测状态仅用于绘制摆杆示意，不是图像解码重建。','small')
 gif(ROOT/f'demos/{m}_three_conditions.gif',f'{LABEL[m]}：同初态三工况演示。0.1秒/帧，预测3.2秒后停留1秒循环。')
 if m=='hjepa':
  gif(ROOT/'demos/hjepa_failure_demo.gif','额外失败样例：multi，b=0，group_id=10000385。按触发停止判据的最小组编号选择，不代表平均表现；与上述固定中位组演示分开。')
section('06  比较口径与后续工作')
for text in CONTENT['protocol']:p(text)
p('建议的下一步','sub')
for text in CONTENT['next']:p(text)
p('代码与结果来源','sub')
for text in CONTENT['sources']:p(text,'small')
p('主要限制：同数据适配版不等于原论文默认实现；方法维数、参数量和训练目标不同；我们 B 阶段只训练阻尼相关模块，基线 B 阶段全参数微调。当前结果不能单独归因于耗散结构或潜空间维数。','small')
def footer(canvas,doc):
 canvas.setFont('CJK',8);canvas.setFillColor(GRAY);canvas.drawString(40,25,'视觉动力学模型对比 · 2026-11-09');canvas.drawRightString(W-40,25,str(doc.page))
PDF.parent.mkdir(parents=True,exist_ok=True);doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=38,bottomMargin=40,title='四种视觉动力学模型对比报告',author='WM experiment review');doc.build(story,onFirstPage=footer,onLaterPages=footer)
css='body{font-family:-apple-system,BlinkMacSystemFont,"Arial Unicode MS",sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;color:#173a4a;line-height:1.7} .title{font-size:32px;font-weight:700}.h{font-size:25px;margin-top:55px;border-top:2px solid #137d72;padding-top:22px}.sub{font-size:18px;color:#137d72}.small,figcaption{font-size:13px;color:#5c6c76}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dbe5e9}th{background:#173a4a;color:white}tr:nth-child(even){background:#f1f5f7}img{max-width:100%;height:auto}figure{margin:25px 0}@media print{.h{break-before:page}}'
(ROOT/'comparison_report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>四种视觉动力学模型对比</title><style>'+css+'</style><body>'+''.join(web)+'</body></html>')
(ROOT/'build_manifest.json').write_text(json.dumps({'pdf':str(PDF),'html':str(ROOT/'comparison_report.html'),'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),'html_sha256':hashlib.sha256((ROOT/'comparison_report.html').read_bytes()).hexdigest(),'data_summary_hashes':{f'{m}/{s}':hashlib.sha256((ROOT/f'data/runs/{m}/{m}_seed0/{s}/evaluation/summary.json').read_bytes()).hexdigest() for m in METHODS for s in ['a','single','multi']}},indent=2))
print(PDF)
