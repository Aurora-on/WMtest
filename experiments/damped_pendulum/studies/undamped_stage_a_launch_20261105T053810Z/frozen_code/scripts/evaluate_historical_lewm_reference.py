#!/usr/bin/env python3
"""Frozen historical LeWM specialists on current b=0/.2 test episodes.

This is a NONMATCHED HISTORICAL REFERENCE, not a controlled model comparison.
No optimizer, model update, readout fit or current-test selection is performed.
The predeclared primary readout is the historical MLP; ridge is also reported.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpbench.evaluate import pixels_tensor, decode_state
from dpbench.evaluate_holdout import load_readouts, tensor_digest
from dpbench.evaluate_passive import summary_metrics
from dpbench.models import load_checkpoint
from dpbench.passive_family import sha256

LABEL = 'nonmatched_historical_single_damping_specialist_reference'
ANCHOR, HISTORY = 20, 5
HORIZONS = [1., 2., 4., 8., 16., 32.]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def historical_input_batch(path, ids, steps):
    """Only t=1.6..2s RGB and actions; never reads states or future imagery."""
    histories, past, actions = [], [], []
    with h5py.File(path, 'r') as f:
        for ep in ids:
            off, count = int(f['ep_offset'][ep]), int(f['ep_len'][ep])
            if steps < 1 or ANCHOR + steps >= count:
                raise ValueError('Requested horizon exceeds episode length')
            lo, anchor = off + ANCHOR - HISTORY + 1, off + ANCHOR
            histories.append(f['pixels'][lo:anchor + 1])
            past.append(f['action'][lo:anchor])
            actions.append(f['action'][anchor:anchor + steps])
    result = (np.stack(histories), np.stack(past), np.stack(actions))
    if np.any(result[1]) or np.any(result[2]):
        raise ValueError('Historical reference is restricted to passive tests')
    return result


@torch.no_grad()
def predict_frozen(model, probes, inputs, device):
    images, past, actions = inputs
    images = torch.stack([pixels_tensor(x, device) for x in images])
    z = model.rollout_features(images, torch.as_tensor(actions, device=device),
                               past_actions=torch.as_tensor(past, device=device)).cpu().numpy()
    predictions = {name: decode_state(probe.predict(z)) for name, probe in probes.items()}
    finite = np.isfinite(z).all(axis=-1)
    valid = np.where(finite.all(axis=1), z.shape[1] - 1, finite.argmin(axis=1) - 1)
    for i, end in enumerate(valid):
        for prediction in predictions.values():
            prediction[i, end + 1:] = np.nan
    return predictions, z[:, 0], valid


def bound_runs(manifest, legacy_root):
    runs = [r for r in manifest['runs'] if r['model'] == 'lewm']
    expected = {(b, seed) for b in (0., .2) for seed in (0, 1, 2)}
    if len(runs) != 6 or {(r['damping'], r['seed']) for r in runs} != expected:
        raise ValueError('Exactly six preexisting specialist runs are required')
    root = Path(legacy_root).resolve()
    for run in runs:
        required = {f"runs/server_47111/{run['run']}/{suffix}"
                    for suffix in ('best.pt', 'evaluation/readouts.pt', 'evaluation/calibration.json', 'run.json')}
        listed = {item['relative_path']: item for item in run['files']}
        if not required <= set(listed):
            raise ValueError('Historical manifest lacks provenance files')
        for relative in sorted(required):
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Unsafe historical manifest path')
            if sha256(path) != listed[relative]['sha256']:
                raise ValueError(f'Historical frozen file changed: {relative}')
    return runs


def evaluate(args):
    if args.output.exists():
        raise FileExistsError('Use a new historical-reference output directory')
    torch.set_num_threads(args.threads)
    if sha256(args.seen) != args.seen_sha256:
        raise ValueError('Current seen dataset hash mismatch')
    historical = json.loads(args.frozen_manifest.read_text())
    runs = bound_runs(historical, args.legacy_root)
    with h5py.File(args.seen, 'r') as f:
        dt = float(f.attrs['dt'])
        if not np.isclose(dt, .1) or int(f.attrs['history_required']) != 21:
            raise ValueError('Expected current passive-family21-frame protocol')
        physical = {k: float(f.attrs[k]) for k in ('mass', 'length', 'gravity')}
        all_test = f['splits/test'][:]
        subsets = {b: [int(ep) for ep in all_test if np.isclose(float(f['damping'][ep]), b, rtol=0, atol=1e-12)]
                   for b in (0., .2)}
        if any(len(ids) != 128 for ids in subsets.values()):
            raise ValueError('Expected all128 currenttest episodes per selected damping')
        groups = [{int(f['group_id'][ep]) for ep in ids} for ids in subsets.values()]
        if groups[0] != groups[1] or len(groups[0]) != 128:
            raise ValueError('Damping subsets must use the same128 independentinitial groups')
    steps = round(max(HORIZONS) / dt)
    sources = ['scripts/evaluate_historical_lewm_reference.py', 'dpbench/models.py',
               'dpbench/evaluate.py', 'dpbench/evaluate_holdout.py', 'dpbench/evaluate_passive.py',
               'dpbench/generate.py', 'dpbench/passive_family.py', 'third_party/sources.json']
    protocol = dict(status='running', comparison_class=LABEL, primary_probe='mlp',
        probe_selection='MLP predeclared; both original frozen readouts reported; no current-test selection',
        seen_sha256=args.seen_sha256, frozen_manifest_sha256=sha256(args.frozen_manifest),
        source_hashes={n: sha256(ROOT / n) for n in sources},
        historical_root=str(args.legacy_root.resolve()), seen=str(args.seen.resolve()),
        anchor_frame=ANCHOR, anchor_time_s=2., history_frames=HISTORY, history_time_s=.4,
        horizons_s=HORIZONS, episodes_per_model=128, models=6, expected_model_episode_rollouts=768,
        device=args.device, requested_batch_size=args.batch_size, model_updates=0, readout_fitting=False,
        limitations=['Single-damping specialists selected using known test damping versus currentmulti-dampingmodel',
                    'Differenttrainingdata, forcingmix, historylength, budgets andreadoutcalibration',
                    'Only b0 and b0.2; no claim aboutinterpolation or extrapolation',
                    'Historicalmodels areLeHami-author convolutional multistep LeWM adaptations'])
    args.output.mkdir(parents=True)
    write_json(args.output / 'manifest.json', protocol)
    started = time.perf_counter()
    all_metrics, results = [], []
    for run in runs:
        directory = args.legacy_root / 'runs/server_47111' / run['run']
        legacy_run = json.loads((directory / 'run.json').read_text())
        legacy_calibration = json.loads((directory / 'evaluation/calibration.json').read_text())
        if any(not np.isclose(float(legacy_run['dataset_attributes'][key]), value)
               for key, value in physical.items()):
            raise ValueError('Historical physical parameters differ from current test')
        model, checkpoint = load_checkpoint(directory / 'best.pt', args.device)
        model.eval().requires_grad_(False)
        if (checkpoint['kind'], checkpoint['seed'], checkpoint['step'], checkpoint['dataset_sha256']) != (
                'lewm', run['seed'], run['checkpoint_step'], run['training_data_sha256']):
            raise ValueError('Historicalcheckpoint metadata mismatch')
        if not np.isclose(model.dt, dt):
            raise ValueError('Historical frame clock differs')
        probes = load_readouts(directory / 'evaluation/readouts.pt')
        before_model = tensor_digest(model.state_dict())
        before_probes = tensor_digest({k: p.serialize() for k, p in probes.items()})
        ids = subsets[run['damping']]
        pilot_ids = ids[:args.batch_size]
        pilot, _, _ = predict_frozen(model, probes, historical_input_batch(args.seen, pilot_ids, steps), args.device)
        differences = []
        for i, ep in enumerate(pilot_ids[:2]):
            single, _, _ = predict_frozen(model, probes, historical_input_batch(args.seen, [ep], steps), args.device)
            for mode in probes:
                delta = pilot[mode][i] - single[mode][0]
                angle = np.arctan2(np.sin(delta[:, 0]), np.cos(delta[:, 0]))
                differences.append((float(abs(angle).max()), float(abs(delta[:, 1]).max())))
        da, dw = np.max(differences, axis=0)
        actual_batch = args.batch_size if np.isfinite([da,dw]).all() and da <= .001 and dw <= .01 else 1
        prediction_chunks = {mode: [] for mode in probes}
        anchors, valid_chunks = [], []
        for offset in range(0, len(ids), actual_batch):
            chunk = ids[offset:offset + actual_batch]
            pred, anchor_features, valid = predict_frozen(model, probes,
                historical_input_batch(args.seen, chunk, steps), args.device)
            for mode in probes:
                prediction_chunks[mode].append(pred[mode])
            anchors.append(anchor_features); valid_chunks.append(valid)
        # Read physical truth only after prediction; it never reaches model/readout inference.
        truth, initial_energy, records = [], [], []
        with h5py.File(args.seen, 'r') as f:
            for ep in ids:
                off = int(f['ep_offset'][ep])
                truth.append(f['state'][off + ANCHOR:off + ANCHOR + steps + 1])
                initial_energy.append(float(f['energy'][off]))
                records.append(dict(file='seen', episode=ep, family='seen', damping=run['damping'],
                    energy_band=int(f['energy_band'][ep]), group_id=int(f['group_id'][ep])))
        truth = np.stack(truth); initial_energy = np.asarray(initial_energy)
        predictions = {mode: np.concatenate(chunks) for mode, chunks in prediction_chunks.items()}
        valid_steps = np.concatenate(valid_chunks)
        metrics = summary_metrics(predictions, truth, initial_energy, records, HORIZONS, dt, physical, valid_steps)
        for row in metrics:
            row.update(seed=run['seed'], specialist_damping=run['damping'], comparison_class=LABEL,
                       primary=row['method']=='mlp')
        if before_model != tensor_digest(model.state_dict()) or before_probes != tensor_digest({k: p.serialize() for k,p in probes.items()}):
            raise ValueError('Historical model or readout state changed')
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        output = args.output / run['run']; output.mkdir()
        np.savez_compressed(output / 'trajectories.npz', truth=truth, **{f'prediction_{k}':v for k,v in predictions.items()},
            episode_ids=np.asarray(ids), valid_steps=valid_steps, initial_energy=initial_energy,
            anchor_features=np.concatenate(anchors), time=np.arange(steps+1)*dt)
        write_json(output / 'episodes.json', records); write_json(output / 'metrics.json', metrics)
        result = dict(run=run['run'], comparison_class=LABEL, seed=run['seed'], damping=run['damping'],
            checkpoint_step=checkpoint['step'], checkpoint_sha256=sha256(directory / 'best.pt'),
            readout_sha256=sha256(directory / 'evaluation/readouts.pt'), episodes=len(ids),
            full_rollouts=int((valid_steps==steps).sum()), state_and_readouts_unchanged=True,
            complete_readout_rollouts={mode:int(np.isfinite(pred).all(axis=(1,2)).sum())
                                      for mode,pred in predictions.items()},
            historical_protocol={'training_steps':legacy_run['args']['steps'],
                'training_data_sha256':checkpoint['dataset_sha256'], 'history_frames':5,
                'train_labels':len(legacy_calibration['train']), 'val_labels':len(legacy_calibration['val']),
                'probe_optimization_steps':legacy_calibration['probes']['mlp'].get('optimization_steps')},
            readouts_refitted=False, batch_check={'actual_batch_size':actual_batch,
                'max_angle_difference':float(da) if np.isfinite(da) else None,
                'max_omega_difference':float(dw) if np.isfinite(dw) else None,
                'tolerances':[.001,.01]})
        write_json(output / 'manifest.json', result)
        all_metrics.extend(metrics); results.append(result)
        print(json.dumps(dict(phase='run_completed', **result)), flush=True)
        del model, probes, checkpoint
        gc.collect()
        if args.device=='cuda':torch.cuda.empty_cache()
    bound_runs(historical, args.legacy_root)
    assert sha256(args.seen)==args.seen_sha256
    aggregate=[]
    keys=('angle_rmse_rad','omega_rmse_rad_s','energy_rmse_J','energy_normalized_rmse',
          'endpoint_angle_mae_rad','endpoint_omega_mae_rad_s','anchor_angle_mae_rad','anchor_omega_mae_rad_s')
    for b in (0.,.2):
        for mode in ('ridge','mlp'):
            for horizon in HORIZONS:
                selected=[r for r in all_metrics if r['specialist_damping']==b and r['method']==mode
                          and r['horizon_s']==horizon and r['grouping']=='all' and r['support']=='all_available']
                assert len(selected)==3
                aggregate.append(dict(comparison_class=LABEL, damping=b, probe=mode, primary=mode=='mlp',
                    horizon_s=horizon, seeds=[r['seed'] for r in selected],
                    minimum_coverage=min(r['coverage'] for r in selected),
                    mean={k:float(np.mean([r[k] for r in selected])) for k in keys if all(k in r for r in selected)},
                    seed_sd={k:float(np.std([r[k] for r in selected],ddof=1)) for k in keys if all(k in r for r in selected)}))
    write_json(args.output / 'metrics.json', all_metrics)
    write_json(args.output / 'summary.json', dict(comparison_class=LABEL, primary_probe='mlp', aggregate=aggregate,
        scope='Pooled trajectory/time RMSE perseed, then seed mean and sampleSD. Nonmatched historical reference.'))
    protocol.update(status='completed', elapsed_seconds=time.perf_counter()-started, results=results,
                    completed_model_episode_rollouts=sum(r['full_rollouts'] for r in results))
    write_json(args.output / 'manifest.json', protocol)
    print(json.dumps({'phase':'completed','comparison_class':LABEL,'seconds':protocol['elapsed_seconds']}),flush=True)
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('seen','frozen-manifest','legacy-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seen-sha256',required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if min(args.batch_size,args.threads)<1:parser.error('Positivebatchsize andthreads required')
    if not args.execute:
        print(json.dumps({'status':'plan_only','comparison_class':LABEL,'models':6,'model_episode_rollouts':768,
            'horizons_s':HORIZONS,'primary_probe':'mlp','model_updates':0,'readout_fitting':False},indent=2))
        return
    evaluate(args)


if __name__=='__main__':main()
