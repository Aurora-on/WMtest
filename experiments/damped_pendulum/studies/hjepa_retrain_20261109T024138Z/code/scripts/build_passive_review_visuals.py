#!/usr/bin/env python3
"""Create static scientific plots and an offline, truth-labelled pendulum demo.

Presentation only: reads saved predictions; does not train, select checkpoints,
change trajectories, or use future true frames in a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


COLORS = {"truth": "#147d92", "model": "#bf4c36", "persistence": "#b27d14", "equilibrium": "#7a8190"}
NAMES = {"truth": "Reference", "model": "World model + frozen readout", "persistence": "Hold decoded anchor", "equilibrium": "Known equilibrium"}
METHODS = ("model", "persistence", "equilibrium")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_plot(history_path, output):
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "losses" in r]
    steps = np.array([r["step"] for r in rows])
    loss = np.array([r["losses"]["loss"] for r in rows])
    val = [r for r in rows if "validation" in r]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.35))
    axes[0].plot(steps, loss, color="#bcc9d4", lw=.65, alpha=.65, label="Per-update loss")
    if len(loss) >= 50:
        median = np.array([np.median(loss[max(0,i-49):i+1]) for i in range(len(loss))])
        axes[0].plot(steps, median, color="#1e5369", lw=1.6, label="Trailing 50-update median")
    axes[0].set(ylabel="Training objective (latent units)", xlabel="Completed optimizer updates", yscale="log")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot([r["step"] for r in val], [r["validation"]["score"] for r in val], "o-", color="#1e5369", ms=4)
    if val:
        best = min((r for r in val if r["validation"]["gate_passed"]), key=lambda r:r["validation"]["score"])
        axes[1].scatter([best["step"]], [best["validation"]["score"]], marker="*", s=130, color=COLORS["model"], zorder=3)
        axes[1].annotate(f"Selected checkpoint: {best['step']}", (best["step"], best["validation"]["score"]), xytext=(7,12), textcoords="offset points", fontsize=8)
    axes[1].set(ylabel="Validation score (lower is better)", xlabel="Completed optimizer updates", yscale="log")
    for ax in axes:
        ax.axvline(1000, ls=":", color="#7a8190", lw=1)
        ax.axvline(steps[-1]+1, ls="--", color=COLORS["model"], lw=1)
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Training stopped after {steps[-1]:,} / 20,000 updates | failure during horizon-8 stage", fontsize=12)
    fig.text(.5, .015, "The validation score measures latent prediction, not physical angle or damping accuracy.", ha="center", fontsize=8, color="#536172")
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(output / "training_progress.png", dpi=180)
    plt.close(fig)


def comparison_plot(metrics, output):
    families = sorted({r["family"] for r in metrics if r["grouping"] == "family"})
    family_names = {"seen":"Seen damping", "seen_test":"Seen damping", "unseen_interpolation":"Unseen: interpolation", "interpolation":"Unseen: interpolation", "unseen_extrapolation":"Unseen: extrapolation", "extrapolation":"Unseen: extrapolation"}
    fig, axes = plt.subplots(2, len(families), figsize=(11.4, 6.1), squeeze=False)
    overview = []
    for col, family in enumerate(families):
        for method in METHODS:
            rows = sorted([r for r in metrics if r["grouping"] == "family" and r["family"] == family and r["method"] == method and r["support"] == "model_complete"], key=lambda r:r["horizon_s"])
            x = [r["horizon_s"] for r in rows]
            for row, key in ((0,"angle_rmse_rad"), (1,"energy_normalized_rmse")):
                axes[row,col].plot(x, [r.get(key,np.nan) for r in rows], "o-", color=COLORS[method], label=NAMES[method], ms=4, lw=1.6)
            if method == "model":
                overview.extend(rows)
                coverage = ", ".join(f"{r['horizon_s']:g}s:{r['n_valid']}/{r['n_total']}" for r in rows if r['horizon_s'] in (1,8,32))
                axes[1,col].text(.5,-.36,coverage,ha="center",transform=axes[1,col].transAxes, fontsize=7.5)
        axes[0,col].set_title(family_names.get(family,family.replace("_"," ")), fontsize=11)
        for ax in axes[:,col]:
            ax.set_xscale("log", base=2)
            ax.set_xticks([1,2,4,8,16,32], labels=[1,2,4,8,16,32])
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
        axes[1,col].set_xlabel("Autonomous forecast horizon (s)")
    axes[0,0].set_ylabel("Wrapped-angle RMSE (rad)")
    axes[1,0].set_ylabel("Energy error RMSE / initial energy")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,1.005), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Physical forecast comparison | identical completed-rollout support",y=.935,fontsize=12)
    fig.text(.5,.01,"Errors pool all future times up to each horizon; coverage below panels excludes solver failures. No test-time fitting.",ha="center",fontsize=8,color="#536172")
    fig.tight_layout(rect=(0,.08,1,.91))
    fig.savefig(output / "forecast_comparison.png", dpi=180)
    plt.close(fig)
    return overview


def energy(state, physical):
    return .5*physical["mass"]*physical["length"]**2*state[...,1]**2+physical["mass"]*physical["gravity"]*physical["length"]*(1-np.cos(state[...,0]))


def example_indices(records):
    """Fix the smallest initial-condition group in each energy band, before errors."""
    chosen = {}
    for i,r in enumerate(records):
        key = (r["damping"],r["energy_band"])
        if key not in chosen or r["group_id"] < records[chosen[key]]["group_id"]:
            chosen[key] = i
    return [chosen[key] for key in sorted(chosen)]


def example_plot(data, records, physical, output):
    selected = example_indices(records)
    ids = [i for i in selected if records[i]["energy_band"] == 2 and any(np.isclose(records[i]["damping"], b) for b in (0,.2,1.5))]
    fig, axes = plt.subplots(2,len(ids), figsize=(11.4,5.9),squeeze=False)
    t = data["time"]
    for col,i in enumerate(ids):
        r=records[i]
        for method in ("truth",)+METHODS:
            state=data["truth"][i] if method=="truth" else data["prediction_"+method][i]
            axes[0,col].plot(t, np.arctan2(np.sin(state[:,0]),np.cos(state[:,0])), color=COLORS[method], lw=1.2, label=NAMES[method],alpha=.9)
            axes[1,col].plot(t,energy(state,physical)/data["initial_energy"][i],color=COLORS[method],lw=1.2)
        axes[0,col].set_title(f"b={r['damping']:g} | group {r['group_id']}",fontsize=10)
        axes[1,col].set_xlabel("Time after 2 s video history (s)")
        for ax in axes[:,col]:
            ax.grid(alpha=.15); ax.spines[["top","right"]].set_visible(False)
    axes[0,0].set_ylabel("Angle (rad)")
    axes[1,0].set_ylabel("Mechanical energy / initial energy")
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc="upper center",ncol=4,fontsize=8,frameon=False)
    fig.suptitle("Fixed example: smallest group ID in initial energy band 2",y=.93,fontsize=12)
    fig.text(.5,.01,"Examples are selected by group ID, never prediction quality. Model curves use a frozen physical readout; missing suffix = solver failure.",ha="center",fontsize=7.5,color="#536172")
    fig.tight_layout(rect=(0,.04,1,.9));fig.savefig(output/"fixed_example_trajectories.png",dpi=180);plt.close(fig)
    return ids


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多阻尼摆：初步测试演示</title>
<style>*{box-sizing:border-box}body{margin:0;background:#edf2f5;color:#182d3d;font:15px/1.65 system-ui,"PingFang SC",sans-serif}.wrap{max-width:1160px;margin:auto;padding:32px 24px}h1{font-size:29px;margin:0 0 6px}.sub{color:#526677}.notice{background:#fff3dc;border-left:4px solid #c28a24;padding:12px 16px;margin:20px 0}.controls,.card{background:white;border:1px solid #dae3e8;border-radius:12px;padding:16px;margin:16px 0}.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap}button,select{font:inherit;padding:6px 10px;border-radius:6px;border:1px solid #bfcdd6;background:white;color:#193647}button{cursor:pointer}input{flex:1;min-width:160px}canvas{width:100%;display:block}.caption{font-size:13px;color:#596d7b}.stats{font-variant-numeric:tabular-nums}a{color:#147d92}.small{font-size:12px}#status{color:#a13f31;font-weight:600}</style>
<div class="wrap"><h1>一个模型，不同阻尼环境</h1><div class="sub">参考运动 × 世界模型 × 简单基线 · 仅用前 2 秒视频初始化，然后自主预测 32 秒</div>
<div class="notice"><b>这是未完成训练的中途模型。</b>训练在第 2,795 步前向求解时失败，完成 2,794 / 20,000 次更新。此处使用验证集选出的第 <span id="ckpt"></span> 步检查点。演示为<b>冻结物理读出后的摆运动重绘</b>，不是网络生成的未来像素，也不是已验证的阻尼识别结果。</div>
<div class="controls"><label>阻尼 b <select id="damping"></select></label><label>初始能量层 <select id="band"><option value="0">0 · 低</option><option value="1">1 · 中低</option><option value="2" selected>2 · 中高</option><option value="3">3 · 高</option></select></label><button id="play">播放</button><button id="reset">回到起点</button><label>速度 <select id="speed"><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option></select></label><input type="range" id="time" min="0" max="320" value="0"><span id="clock" class="stats"></span></div>
<div id="description" class="caption"></div><div id="status"></div>
<div class="card"><canvas id="pendulum" width="1080" height="300"></canvas><div class="caption">“保持初态”固定模型读出的初始角度和角速度，不执行动力学；“平衡点”始终取 θ=ω=0。颜色与下方曲线一致。</div></div>
<div class="card"><canvas id="curves" width="1080" height="350"></canvas><div class="caption">横轴均为预测时间；黑色竖线对应当前帧。能量使用真实物理参数由读出的角度/角速度计算，并除以该轨迹 t=0 的初始机械能；与模型内部潜储能不同。</div></div>
<div class="caption">样本规则：每个初始能量层固定选择最小 group_id，在所有阻尼条件下配对展示，不根据模型误差挑选。测试标签仅用于绘制参考曲线，未反馈到自主推演。求解失败后停止显示模型，其他方法继续显示。该网页自包含，可离线打开。</div></div>
<script>const DATA=__DATA__;
const colors={truth:'#147d92',model:'#bf4c36',persistence:'#b27d14',equilibrium:'#7a8190'};const labels={truth:'参考运动',model:'世界模型 + 物理读出',persistence:'保持初态',equilibrium:'静止平衡点'};const methods=Object.keys(colors);let playing=false,last=null,position=0;
const $=x=>document.getElementById(x);$('ckpt').textContent=DATA.checkpoint_step;
const bs=[...new Set(DATA.examples.map(x=>x.damping))].sort((a,b)=>a-b);for(const b of bs){let o=document.createElement('option');o.value=b;o.textContent=b;$('damping').append(o)}$('damping').value=.2;
function ex(){return DATA.examples.find(x=>x.damping===+$('damping').value&&x.energy_band===+$('band').value)}
function path(ctx,points,x,y,w,h,min,max,col){ctx.beginPath();let active=false;for(let j=0;j<points.length;j++){let v=points[j];if(v===null||!Number.isFinite(v)){active=false;continue}let px=x+j/(points.length-1)*w,py=y+h-(v-min)/(max-min)*h;if(active)ctx.lineTo(px,py);else ctx.moveTo(px,py);active=true}ctx.strokeStyle=col;ctx.lineWidth=1.7;ctx.stroke()}
function render(){const e=ex();if(!e)return;const k=Math.round(position);$('time').value=k;$('clock').textContent=(k*DATA.dt).toFixed(1)+' / 32.0 s';$('description').textContent=`固定初态组 ${e.group_id} · 初始能量层 ${e.energy_band} · ${e.family_label} · 模型有效推演 ${e.valid_steps*DATA.dt} 秒`;$('status').textContent=k>e.valid_steps?'模型已在此前求解失败；该时间段无有效预测。':'';
let c=$('pendulum'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);methods.forEach((m,i)=>{let mid=135+270*i;ctx.fillStyle=colors[m];ctx.font='16px system-ui';ctx.textAlign='center';ctx.fillText(labels[m],mid,25);ctx.strokeStyle='#d2dce3';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(mid-55,62);ctx.lineTo(mid+55,62);ctx.stroke();let state=e.states[m][k];if(!state||state[0]===null){ctx.fillStyle='#a13f31';ctx.fillText('求解失败：无预测',mid,155);return}let th=state[0],x=mid+115*Math.sin(th),y=70+115*Math.cos(th);ctx.beginPath();ctx.moveTo(mid,70);ctx.lineTo(x,y);ctx.strokeStyle=colors[m];ctx.lineWidth=3;ctx.stroke();ctx.beginPath();ctx.arc(x,y,10,0,Math.PI*2);ctx.fillStyle=colors[m];ctx.fill();ctx.beginPath();ctx.arc(mid,70,3,0,Math.PI*2);ctx.fill();ctx.font='12px system-ui';ctx.fillText('θ = '+th.toFixed(3)+' rad',mid,252);ctx.fillText('ω = '+state[1].toFixed(3)+' rad/s',mid,273)});
c=$('curves');ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);const panes=[{x:65,w:455,key:'angle',label:'角度 / rad'},{x:600,w:455,key:'energy',label:'机械能 / 初始能量'}];panes.forEach(p=>{let vals=methods.flatMap(m=>e[p.key][m]).filter(v=>v!==null&&Number.isFinite(v));let lo=p.key==='energy'?0:Math.min(...vals),hi=Math.max(...vals);let pad=Math.max((hi-lo)*.08,.05);if(p.key!=='energy')lo-=pad;hi+=pad;ctx.font='14px system-ui';ctx.fillStyle='#243e50';ctx.textAlign='left';ctx.fillText(p.label,p.x,22);for(let t=0;t<=32;t+=8){let xx=p.x+p.w*t/32;ctx.strokeStyle='#e4ebef';ctx.beginPath();ctx.moveTo(xx,38);ctx.lineTo(xx,292);ctx.stroke();ctx.textAlign='center';ctx.fillText(t,xx,315)}for(let a=0;a<4;a++){let v=lo+(hi-lo)*a/3,yy=292-254*a/3;ctx.strokeStyle='#e4ebef';ctx.beginPath();ctx.moveTo(p.x,yy);ctx.lineTo(p.x+p.w,yy);ctx.stroke();ctx.textAlign='right';ctx.fillText(v.toFixed(2),p.x-8,yy+4)}for(let m of methods)path(ctx,e[p.key][m],p.x,38,p.w,254,lo,hi,colors[m]);ctx.strokeStyle='#182d3d';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(p.x+p.w*k/320,38);ctx.lineTo(p.x+p.w*k/320,292);ctx.stroke();ctx.textAlign='center';ctx.fillStyle='#526677';ctx.fillText('预测时间 / s',p.x+p.w/2,341)})}
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'暂停':'播放';last=null};$('reset').onclick=()=>{position=0;last=null;render()};$('time').oninput=()=>{position=+$('time').value;render()};for(const id of ['damping','band'])$(id).onchange=()=>{position=0;last=null;render()};function tick(ts){if(playing){if(last!==null)position+=(ts-last)/1000/ DATA.dt*+$('speed').value;last=ts;if(position>=320){position=320;playing=false;$('play').textContent='播放'}render()}else last=null;requestAnimationFrame(tick)}render();requestAnimationFrame(tick);
</script></html>'''


def finite_list(a):
    a=np.asarray(a)
    return np.where(np.isfinite(a), a, None).tolist()


def build_demo(data, records, manifest, output):
    physical=manifest["physical_parameters"]
    examples=[]
    for i in example_indices(records):
        r=records[i]; states={m:data["truth"][i] if m=="truth" else data["prediction_"+m][i] for m in ("truth",)+METHODS}
        label="已见阻尼" if r["file"]=="seen" else ("未见阻尼外推" if r["damping"]>1 else "未见阻尼插值")
        examples.append({**r,"family_label":label,"valid_steps":int(data["valid_steps"][i]),"states":{m:finite_list(v) for m,v in states.items()},"angle":{m:finite_list(np.arctan2(np.sin(v[:,0]),np.cos(v[:,0]))) for m,v in states.items()},"energy":{m:finite_list(energy(v,physical)/data["initial_energy"][i]) for m,v in states.items()}})
    payload={"checkpoint_step":manifest["checkpoint_step"],"dt":manifest["dt"],"examples":examples}
    html=HTML.replace("__DATA__",json.dumps(payload,ensure_ascii=False,separators=(",",":"),allow_nan=False))
    (output/"demo.html").write_text(html)
    return examples


def make_gif(data,records,ids,output):
    font_path=Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    font=ImageFont.truetype(str(font_path),16) if font_path.exists() else ImageFont.load_default()
    small=ImageFont.truetype(str(font_path),12) if font_path.exists() else font
    frames=[]; width,height=840,650
    methods=("truth","model","persistence")
    for k in range(0,len(data["time"]),2):
        img=Image.new("RGB",(width,height),"#f5f8fa");d=ImageDraw.Draw(img)
        d.text((20,12),f"Partial training checkpoint | autonomous forecast {data['time'][k]:4.1f} / 32.0 s",font=font,fill="#182d3d")
        for col,m in enumerate(methods):d.text((75+col*270,45),{"truth":"Reference","model":"Model + readout","persistence":"Hold decoded anchor"}[m],font=font,fill=COLORS[m])
        for row,i in enumerate(ids):
            top=80+row*175;r=records[i]
            d.text((15,top),f"b={r['damping']:g}",font=font,fill="#182d3d")
            for col,m in enumerate(methods):
                center=150+col*270; pivot=top+25
                state=data["truth"][i,k] if m=="truth" else data["prediction_"+m][i,k]
                d.line((center-35,pivot-6,center+35,pivot-6),fill="#bfcbd3",width=2)
                if not np.isfinite(state).all():
                    d.text((center-65,pivot+40),"SOLVER FAILED",font=small,fill=COLORS["model"]);continue
                x=center+95*np.sin(state[0]);y=pivot+95*np.cos(state[0])
                d.line((center,pivot,x,y),fill=COLORS[m],width=3);d.ellipse((x-8,y-8,x+8,y+8),fill=COLORS[m]);d.ellipse((center-3,pivot-3,center+3,pivot+3),fill="#182d3d")
        d.text((20,610),"Fixed paired initial group, energy band 2. Physical-readout redraw; no future-frame feedback.",font=small,fill="#526677")
        d.text((20,629),"Each GIF frame represents 0.2 s; displayed at 2x speed. Full controls in demo.html.",font=small,fill="#526677")
        frames.append(img)
    frames[0].save(output/"demo.gif",save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=False)
    frames[0].save(output/"demo_preview.png")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation",type=Path,default=Path("received/passive_review_20261104/evaluation"))
    p.add_argument("--history",type=Path,default=Path("received/passive_review_20261104/seed0_47541/history.jsonl"))
    p.add_argument("--output",type=Path,default=Path("reports/passive_review_20261104"))
    p.add_argument("--training-only",action="store_true")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titleweight":"semibold","savefig.facecolor":"white"})
    training_plot(args.history,args.output)
    if args.training_only:return
    manifest=json.loads((args.evaluation/"manifest.json").read_text());records=json.loads((args.evaluation/"episodes.json").read_text());metrics=json.loads((args.evaluation/"metrics.json").read_text())
    with np.load(args.evaluation/"trajectories.npz",allow_pickle=False) as f:data={k:f[k] for k in f.files}
    if len(records)!=len(data["truth"]):raise ValueError("Episode metadata does not align with saved trajectories")
    overview=comparison_plot(metrics,args.output)
    ids=example_plot(data,records,manifest["physical_parameters"],args.output)
    examples=build_demo(data,records,manifest,args.output)
    make_gif(data,records,ids,args.output)
    dump(args.output/"visual_provenance.json",{"selection_rule":"Smallest group_id within each initial-energy band for each damping; independent of forecast errors","gif_energy_band":2,"gif_damping":[records[i]["damping"] for i in ids],"gif_records":[records[i] for i in ids],"examples_count":len(examples),"evaluation_manifest_sha256":hash_file(args.evaluation/"manifest.json"),"trajectories_sha256":hash_file(args.evaluation/"trajectories.npz"),"script_sha256":hash_file(Path(__file__)),"checkpoint_step":manifest["checkpoint_step"],"plot_support":"model_complete at each horizon; matched baseline support","physical_state_redraw_not_pixel_generation":True,"overview":overview})
    print(json.dumps({"output":str(args.output.resolve()),"examples":len(examples),"checkpoint_step":manifest["checkpoint_step"]}))


if __name__=="__main__":main()
