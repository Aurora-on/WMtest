"""Frozen checkpoint + actual H5 windows: isolate raster jump and SG velocity."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.generate import render
from dpbench.evaluate import pixels_tensor

parser=argparse.ArgumentParser()
parser.add_argument('--device',default='cpu')
parser.add_argument('--checkpoint',type=Path,default=OUT/'frozen_training/best.pt')
parser.add_argument('--teacher-root',type=Path,default=ROOT/'studies/efficient3_evaluation_20261105T023318Z/teacher_received')
parser.add_argument('--windows',type=Path,default=OUT/'stall_rest_windows.npz')
parser.add_argument('--output',type=Path,default=OUT)
args=parser.parse_args()
torch.set_num_threads(4)
CKPT=args.checkpoint
saved=torch.load(CKPT,map_location='cpu',weights_only=True)
model=DissipativeWorldModel(saved['config']).to(args.device).eval()
model.load_state_dict(saved['state_dict'],strict=True)
for par in model.parameters(): par.requires_grad_(False)
before={k:v.clone() for k,v in model.state_dict().items()}
meta=json.loads((args.teacher_root/'diagnostic.json').read_text())
teacher=np.load(args.teacher_root/'diagnostic_arrays.npz')
windows=np.load(args.windows)
rows=json.loads(str(windows['metadata_json']))
assert str(windows['dataset_sha256'])==meta['dataset_sha256']['unseen']
assert hashlib.sha256(CKPT.read_bytes()).hexdigest()==meta['checkpoint_sha256']

def components(images,actions,intervals):
    n=len(images)
    assert 0<n<=32
    # Preserve original image count per encoder batch; pad without adding data.
    images=torch.cat((images,images[-1:].repeat(32-n,1,1,1,1)))
    actions=torch.cat((actions,actions[-1:].repeat(32-n,1,1)))
    intervals=torch.cat((intervals,intervals[-1:].repeat(32-n,1,1)))
    images,actions,intervals=(x.to(args.device) for x in (images,actions,intervals))
    with torch.no_grad():
        q,p,encoded,correction=model.initialize(images,actions,intervals)
    # Independent direct stencil reconstruction from encoded q history.
    local_dt=intervals[:,-4:]
    times=torch.cat([local_dt.new_zeros(32,1),local_dt[...,0].cumsum(1)],1).double()
    span=times[:,-1:];tau=(times-times[:,-1:])/span
    design=torch.stack([torch.ones_like(tau),tau,tau.square()],-1)
    coeff=torch.linalg.solve(design.transpose(1,2)@design,
        design.transpose(1,2)@encoded[:,-5:].double())
    fitted=coeff[:,1]/span
    np.testing.assert_allclose((fitted+correction.double())[:n].cpu().numpy(),p[:n].cpu().numpy(),atol=1e-13,rtol=1e-13)
    return {'q':q[:n].cpu().numpy(),'p':p[:n].cpu().numpy(),'q_history':encoded[:n].cpu().numpy(),
        'sg_fitted_velocity':fitted[:n].cpu().numpy(),'bounded_correction':correction[:n].cpu().numpy()}

images=torch.stack([pixels_tensor(x,'cpu') for x in windows['pixels']])
comp=components(images,torch.from_numpy(windows['actions']),torch.from_numpy(windows['dt']))
target=[]
for row in rows:
    k=next(i for i,r in enumerate(meta['selected_records']) if r['file']=='unseen' and r['episode']==row['episode'])
    j=int(np.flatnonzero(teacher['relative_time']==row['relative_time_s'])[0])
    target.append(teacher['teacher_features'][k,j])
features=np.concatenate((comp['q'],comp['p']),axis=-1)
target=np.array(target)
# Cross-device FP32 encoder check; preserve rather than hide actual discrepancy.
max_error=float(np.abs(features-target).max())
teacher_passed=bool(np.allclose(features,target,atol=1e-5,rtol=1e-5))
if args.device=='cuda':
    np.testing.assert_allclose(features,target,atol=1e-5,rtol=1e-5)

results=[]
for i,row in enumerate(rows):
    actual=windows['pixels'][i]
    regenerated=np.stack([render(th) for th in windows['states'][i,:,0]])
    pixel_equal=bool(np.array_equal(actual,regenerated))
    assert pixel_equal
    results.append(dict(**row,
        states=windows['states'][i].tolist(),
        theta_abs_max_rad=float(np.max(abs(windows['states'][i,:,0]))),
        omega_abs_max_rad_s=float(np.max(abs(windows['states'][i,:,1]))),
        pixel_changes_per_interval=np.any(actual[1:]!=actual[:-1],axis=-1).sum(axis=(1,2)).tolist(),
        unique_frames=int(np.unique(actual.reshape(21,-1),axis=0).shape[0]),
        regenerated_pixels_exactly_match_dataset=pixel_equal,
        q_history=comp['q_history'][i].tolist(),
        sg_fitted_velocity=comp['sg_fitted_velocity'][i].tolist(),
        bounded_correction=comp['bounded_correction'][i].tolist(),
        initialized_p=comp['p'][i].tolist(),
        prior_teacher_feature_max_difference=float(np.abs(features[i]-target[i]).max())))

static_angles=np.array([-1e-12,0.,1e-12])
static_images=torch.stack([pixels_tensor(np.stack([render(th)]*21),'cpu') for th in static_angles])
static=components(static_images,torch.zeros(3,20,1),torch.full((3,20,1),.1))
assert np.array_equal(render(0.),render(1e-12))
stationary=[]
for i,angle in enumerate(static_angles):
    stationary.append(dict(angle_rad=float(angle),q=static['q'][i].tolist(),
        sg_fitted_velocity=static['sg_fitted_velocity'][i].tolist(),
        bounded_correction=static['bounded_correction'][i].tolist(),
        initialized_p=static['p'][i].tolist()))
diff_q=static['q'][2]-static['q'][0]
unchanged=all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
assert unchanged
summary=dict(status='completed',device=args.device,checkpoint_sha256=meta['checkpoint_sha256'],
    dataset_sha256=str(windows['dataset_sha256']),
    actual_windows=len(rows),original_cuda_teacher_reproduced=teacher_passed,
    cross_device_feature_max_abs_error=max_error,feature_comparison_atol=1e-5,feature_comparison_rtol=1e-5,
    parameters_unchanged=unchanged,optimizer_updates=0,
    actual_window_results=results,exactly_static_synthetic_histories=stationary,
    near_zero_q_jump=diff_q.tolist(),near_zero_q_jump_rms=float(np.sqrt(np.mean(diff_q**2))),
    interpretation='Actual late-rest dataset pixels jump despite tiny physical motion; frozen encoder and SG stencil turn them into artificial initialized momentum. Static histories separately show learned correction need not vanish. This explains a local initializer/data inconsistency; it does not prove it is the main source of b=0 autonomous stall.')
args.output.mkdir(parents=True,exist_ok=True)
suffix='' if args.device=='cuda' else '_cpu'
(args.output/f'renderer_initializer_check{suffix}.json').write_text(json.dumps(summary,indent=2)+'\n')
np.savez_compressed(args.output/f'renderer_initializer_arrays{suffix}.npz',**comp,prior_teacher_features=target)
print(json.dumps({k:v for k,v in summary.items() if k not in ['actual_window_results','exactly_static_synthetic_histories']},indent=2))
for row in results:
    if row['relative_time_s']==32:
        print(row['episode'],'theta',row['theta_abs_max_rad'],'framechanges',row['pixel_changes_per_interval'],
            'SG',row['sg_fitted_velocity'],'correction',row['bounded_correction'],'p',row['initialized_p'])
print('static',stationary)
