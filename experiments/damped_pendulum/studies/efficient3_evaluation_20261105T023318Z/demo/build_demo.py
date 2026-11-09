from pathlib import Path
import hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont

R=Path(__file__).resolve().parents[1]
OUT=R/'demo'
OURS=R/'received'
LEWM=R/'lewm_received'
records=json.loads((OURS/'episodes.json').read_text())
a=np.load(OURS/'trajectories.npz')
rows=[]
for b,run in [(0.,'lewm_b0p0_seed0'),(.2,'lewm_b0p2_seed0')]:
 options=[(i,r) for i,r in enumerate(records) if r['file']=='seen' and r['damping']==b and r['energy_band']==2]
 index,rec=min(options,key=lambda pair:(pair[1]['group_id'],pair[1]['episode']))
 legacy=np.load(LEWM/run/'trajectories.npz')
 j=np.flatnonzero(legacy['episode_ids']==rec['episode'])
 assert len(j)==1;j=int(j[0])
 np.testing.assert_array_equal(a['truth'][index],legacy['truth'][j])
 assert a['valid_steps'][index]==320 and legacy['valid_steps'][j]==320
 rows.append(dict(record=rec,our_index=index,legacy_index=j,legacy_run=run,
                  series=[a['truth'][index],a['prediction_model'][index],legacy['prediction_mlp'][j]]))
assert rows[0]['record']['group_id']==rows[1]['record']['group_id']

W,H=1050,770
fontpath='/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
font=lambda size:ImageFont.truetype(fontpath,size)
f_title,f_col,f_body,f_small,f_num=font(28),font(23),font(20),font(17),font(17)
colors=['#2862AF','#CB5B2D','#277B5C']
col_x=[78,402,726];panel_w=292;tops=[180,430];panel_h=236

def center_text(draw,x,y,text,f,fill):
 box=draw.textbbox((0,0),text,font=f);draw.text((x-(box[2]-box[0])/2,y),text,font=f,fill=fill)

def render(k):
 im=Image.new('RGB',(W,H),'#F7F9FC');d=ImageDraw.Draw(im)
 d.text((30,18),'同一测试初态：真实轨迹与自主预测',font=f_title,fill='#14253B')
 d.text((30,60),'冻结模型与读出 · 无插值、无重新拟合 · 播放速度 1×',font=f_body,fill='#53647A')
 time=f'预测后 {k/10:.1f} 秒'
 box=d.textbbox((0,0),time,font=f_col);d.rounded_rectangle((W-30-(box[2]-box[0])-28,18,W-24,58),10,fill='#E6EDF6')
 d.text((W-30-(box[2]-box[0])-14,22),time,font=f_col,fill='#14253B')
 for i,label in enumerate(['真实轨迹','本模型：3 子步','历史 LeWM：seed 0']):
  center_text(d,col_x[i]+panel_w/2,112,label,f_col,colors[i])
  second=['参考仿真','多阻尼共享模型','单阻尼专用模型（参考）'][i]
  center_text(d,col_x[i]+panel_w/2,145,second,f_small,'#596A7F')
 for ri,row in enumerate(rows):
  top=tops[ri]
  d.text((22,top+70),f'b =\n{row["record"]["damping"]:g}',font=f_body,fill='#243851',spacing=6)
  for ci,series in enumerate(row['series']):
   x=col_x[ci];cx=x+panel_w/2;cy=top+113;radius=85
   d.rounded_rectangle((x,top,x+panel_w,top+panel_h),12,fill='white',outline='#D6DEE8',width=1)
   # All panels use the same pivot and rod length; theta=0 points down.
   for yy in range(int(cy),int(cy+radius),8):d.line((cx,yy,cx,yy+4),fill='#B8C3D2',width=1)
   d.arc((cx-radius,cy-radius,cx+radius,cy+radius),0,360,fill='#E5EBF2',width=1)
   theta,omega=map(float,series[k]);ex=cx+radius*math.sin(theta);ey=cy+radius*math.cos(theta)
   d.line((cx,cy,ex,ey),fill=colors[ci],width=4)
   d.ellipse((ex-11,ey-11,ex+11,ey+11),fill=colors[ci])
   d.ellipse((cx-4,cy-4,cx+4,cy+4),fill='#34465F')
   d.line((cx-18,cy-7,cx+18,cy-7),fill='#34465F',width=3)
   center_text(d,cx,top+204,f'θ {theta:+.2f} rad   ω {omega:+.2f} rad/s',f_num,'#44576E')
 # Footer keeps the comparison limitation visible throughout the animation.
 d.text((30,690),'固定选择：能量层 2、最小初态组编号；展示前 8 秒，未按预测误差挑选样本。',font=f_small,fill='#4B5E75')
 d.text((30,718),'历史 LeWM 的训练数据、历史长度与预算不同；此演示不构成严格同协议排名。',font=f_small,fill='#894D24')
 return im

frames=[render(k) for k in range(81)]
# One palette derived from the first frame keeps colors stable across frames.
palette=frames[0].quantize(colors=128,method=Image.Quantize.MEDIANCUT)
indexed=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
indexed[0].save(OUT/'pendulum_comparison.gif',save_all=True,append_images=indexed[1:],duration=100,loop=0,optimize=False,disposal=2)
frames[0].save(OUT/'demo_start.png');frames[40].save(OUT/'demo_midpoint.png');frames[80].save(OUT/'demo_end.png')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[OURS/'trajectories.npz',OURS/'episodes.json',OURS/'manifest.json']
for row in rows:sources.extend([LEWM/row['legacy_run']/'trajectories.npz',LEWM/row['legacy_run']/'manifest.json'])
prov={'selection_rule':'For each b in[0,0.2], energy_band2, smallestgroup_id thenepisode; samegroup acrossbothb; noerrorbasedselection',
 'selected':[{'record':row['record'],'our_evaluation_index':row['our_index'],'historical_run':row['legacy_run'],'historical_index':row['legacy_index']} for row in rows],
 'source_sha256':{str(p.relative_to(R)):sha(p) for p in sources},
 'dimensions':[W,H],'frames':81,'frame_duration_ms':100,'relative_times_seconds':[0,8],
 'coordinate_convention':'x=sin(theta), y=cos(theta) belowpivot; allpanels identicalrodlength',
 'series':['reference_truth','our_frozen_model_mlp','historical_LeWM_seed0_frozen_mlp'],
 'state_values':'Raw saved theta/omega, no interpolation, smoothing, refitting, or modelupdates',
 'time_origin':'Testanchor atphysicalt=2s; reportedtimeisrelativepredictiontime',
 'comparison_class':'Nonmatched historical single-damping specialist reference; differingtrainingdata/history/budget/readout',
 'script_sha256':sha(Path(__file__))}
(OUT/'selection_provenance.json').write_text(json.dumps(prov,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'dimensions':[W,H],'frames':81,'selected':[r['record'] for r in rows],'gif_bytes':(OUT/'pendulum_comparison.gif').stat().st_size},indent=2))
