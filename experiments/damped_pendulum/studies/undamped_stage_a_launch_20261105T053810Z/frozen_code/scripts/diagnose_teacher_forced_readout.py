#!/usr/bin/env python3
"""Bounded frozen-readout diagnostic on predeclared true-image windows.

This is not an autonomous prediction score: future true images are deliberately
re-encoded to separate readout/observation error from accumulated rollout error.
No world-model updates, readout fitting, or test-based model selection occur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.evaluate import Readout, decode_state, pixels_tensor
from dpbench.train_dissipative import source_hashes

RELATIVE_TIMES = (0., .1, .5, 1., 2., 4., 8., 16., 32.)


def sha256(filename):
    h = hashlib.sha256()
    with Path(filename).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_readout(filename, selected):
    saved = torch.load(filename, map_location='cpu', weights_only=True)[selected]
    network = None
    if saved['mode'] == 'mlp':
        width = int(saved['network']['0.weight'].shape[0])
        network = torch.nn.Sequential(
            torch.nn.Linear(len(saved['mean']), width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 3))
        network.load_state_dict(saved['network'], strict=True)
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
    elif saved['mode'] != 'ridge':
        raise ValueError('Only frozen ridge or mlp readouts are supported')
    numpy_value = lambda key: None if saved[key] is None else saved[key].numpy()
    return Readout(saved['mode'], numpy_value('mean'), numpy_value('scale'),
                   numpy_value('ymean'), numpy_value('yscale'),
                   weights=numpy_value('weights'), network=network)


def metrics(prediction, truth):
    angle = np.arctan2(np.sin(prediction[..., 0] - truth[..., 0]),
                       np.cos(prediction[..., 0] - truth[..., 0]))
    omega = prediction[..., 1] - truth[..., 1]
    return dict(angle_rmse_rad=float(np.sqrt(np.mean(angle ** 2))),
                angle_mae_rad=float(np.mean(abs(angle))),
                omega_rmse_rad_s=float(np.sqrt(np.mean(omega ** 2))),
                omega_mae_rad_s=float(np.mean(abs(omega))))


def initialize_fixed_batch(model, images, actions, intervals, device, batch_size):
    """Keep the original evaluation's encoder batch shape, including the tail.

    A different tail batch may select a different FP32 CUDA convolution kernel;
    the causal velocity fit can amplify that encoder roundoff. Padding uses only
    copies of the final requested window, and all padded outputs are discarded.
    """
    count = len(images)
    if not 0 < count <= batch_size or len(actions) != count or len(intervals) != count:
        raise ValueError('A nonempty aligned batch must fit the fixed batch size')
    pad = batch_size - count
    padded_images = images + [images[-1]] * pad
    padded_actions = actions + [actions[-1]] * pad
    padded_intervals = intervals + [intervals[-1]] * pad
    q, p, _, _ = model.initialize(
        torch.stack(padded_images).to(device),
        torch.as_tensor(np.stack(padded_actions), device=device),
        torch.as_tensor(np.stack(padded_intervals), device=device))
    return torch.cat((q, p), -1)[:count].cpu().numpy()


def run(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError('Choose a new diagnostic output directory')
    torch.set_num_threads(args.threads)
    manifest = json.loads((args.eval_results / 'manifest.json').read_text())
    if manifest['status'] != 'completed' or manifest['checkpoint_step'] != 19250:
        raise ValueError('Requires the completed, preselected step19250 evaluation')
    if args.batch_size != manifest['batch_size']:
        raise ValueError('Diagnostic batch size must equal the original evaluation batch size')
    checkpoint_hash, readout_hash = sha256(args.checkpoint), sha256(args.readouts)
    if checkpoint_hash != manifest['checkpoint_sha256'] or readout_hash != manifest['readout_sha256']:
        raise ValueError('Checkpoint/readout differ from the completed evaluation')
    paths = {'seen': args.seen, 'unseen': args.unseen}
    dataset_hashes = {name: sha256(filename) for name, filename in paths.items()}
    if dataset_hashes != manifest['dataset_sha256']:
        raise ValueError('Datasets differ from the completed evaluation')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved['source_hashes'] != source_hashes():
        raise ValueError('Model/training source differs from the checkpoint')
    model = DissipativeWorldModel(saved['config']).to(args.device).eval()
    model.load_state_dict(saved['state_dict'], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    original_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    probe = load_readout(args.readouts, manifest['selected_readout'])
    records = json.loads((args.eval_results / 'episodes.json').read_text())
    # Choose by immutable grouping metadata only; do not inspect any prediction error.
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault((record['damping'], record['energy_band']), []).append((index, record))
    chosen = [min(groups[key], key=lambda pair: (pair[1]['group_id'], pair[1]['episode']))
              for key in sorted(groups)]
    if len(chosen) != 44:
        raise ValueError('Expected exactly11 damping values x4 energy bands')
    steps = np.asarray([round(t / model.dt) for t in RELATIVE_TIMES])
    if not np.allclose(steps * model.dt, RELATIVE_TIMES):
        raise ValueError('Relative diagnostic times must match the frame interval')
    with np.load(args.eval_results / 'trajectories.npz') as arrays:
        indices = np.asarray([index for index, _ in chosen])
        if not (arrays['valid_steps'][indices] >= steps[-1]).all():
            raise ValueError('Selected autonomous trajectories must have complete32s support')
        truth = arrays['truth'][indices][:, steps].copy()
        autonomous = arrays['prediction_model'][indices][:, steps].copy()
        autonomous_features = arrays['features'][indices][:, steps].copy()
    jobs = [(episode_index, time_index, record, int(relative_step))
            for episode_index, (_, record) in enumerate(chosen)
            for time_index, relative_step in enumerate(steps)]
    features = np.empty((44, len(steps), model.q_dim * 2))
    with h5py.File(args.seen, 'r') as seen, h5py.File(args.unseen, 'r') as unseen:
        handles = {'seen': seen, 'unseen': unseen}
        with torch.no_grad():
            for start in range(0, len(jobs), args.batch_size):
                chunk = jobs[start:start + args.batch_size]
                images, actions, intervals = [], [], []
                for _, _, record, relative_step in chunk:
                    f = handles[record['file']]
                    episode = record['episode']
                    off = int(f['ep_offset'][episode])
                    anchor = off + model.history - 1 + relative_step
                    lo = anchor - model.history + 1
                    if anchor >= off + int(f['ep_len'][episode]):
                        raise ValueError('Selected true-image window is out of bounds')
                    images.append(pixels_tensor(f['pixels'][lo:anchor + 1], 'cpu'))
                    actions.append(f['action'][lo:anchor])
                    intervals.append(f['dt_frame'][lo:anchor])
                batch_features = initialize_fixed_batch(
                    model, images, actions, intervals, args.device, args.batch_size)
                for (episode_index, time_index, _, _), value in zip(chunk, batch_features):
                    features[episode_index, time_index] = value
    teacher = decode_state(probe.predict(features))
    if not np.isfinite(teacher).all() or not np.isfinite(features).all():
        raise FloatingPointError('Nonfinite teacher-forced state/readout')
    # The zero-time teacher initialization must reproduce the original evaluation.
    # Original and diagnostic encoder batch shapes now agree, including the tail.
    np.testing.assert_allclose(features[:, 0], autonomous_features[:, 0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(teacher[:, 0], autonomous[:, 0], rtol=1e-4, atol=1e-4)
    parameter_unchanged = all(torch.equal(value.detach().cpu(), original_state[key])
                              for key, value in model.state_dict().items())
    if not parameter_unchanged or sha256(args.checkpoint) != checkpoint_hash or sha256(args.readouts) != readout_hash:
        raise RuntimeError('Frozen model/readout changed during diagnostic')
    rows = []
    for time_index, sec in enumerate(RELATIVE_TIMES):
        for name, mask in [('all', np.ones(44, dtype=bool))] + [
                (f'b={value:g}', np.asarray([record['damping'] == value for _, record in chosen]))
                for value in sorted(set(record['damping'] for _, record in chosen))]:
            rows.append(dict(group=name, relative_time_s=sec, n=int(mask.sum()),
                teacher_forced=metrics(teacher[mask, time_index], truth[mask, time_index]),
                autonomous=metrics(autonomous[mask, time_index], truth[mask, time_index]),
                q_feature_difference_rmse=float(np.sqrt(np.mean((features[mask, time_index, :model.q_dim]
                    - autonomous_features[mask, time_index, :model.q_dim]) ** 2))),
                p_feature_difference_rmse=float(np.sqrt(np.mean((features[mask, time_index, model.q_dim:]
                    - autonomous_features[mask, time_index, model.q_dim:]) ** 2)))))
    source_files = ('scripts/diagnose_teacher_forced_readout.py', 'dpbench/evaluate.py')
    result = dict(status='completed', checkpoint_sha256=checkpoint_hash,
        readout_sha256=readout_hash, dataset_sha256=dataset_hashes,
        source_hashes={**source_hashes(), **{name: sha256(ROOT / name) for name in source_files}},
        checkpoint_step=saved['step'], selected_readout=manifest['selected_readout'],
        selection='First group_id/episode in each damping x energy-band cell; no error-based selection',
        selected_records=[dict(evaluation_index=index, **record) for index, record in chosen],
        relative_times_seconds=list(RELATIVE_TIMES), true_image_windows=len(jobs),
        encoder_batch_size=args.batch_size, final_batch_padding='Repeat final true window; discard padding outputs',
        model_parameters_unchanged=parameter_unchanged, optimizer_updates=0, readout_refit=False,
        scope='Posthoc diagnostic with true future image histories; NOT autonomous prediction and NOT a replacement test score',
        zero_time_matches_original=True,
        zero_time_max_feature_difference=float(np.max(abs(features[:, 0] - autonomous_features[:, 0]))),
        zero_time_max_decoded_difference=float(np.max(abs(teacher[:, 0] - autonomous[:, 0]))),
        zero_time_feature_atol=1e-5, zero_time_decoded_atol=1e-4,
        metrics=rows, elapsed_seconds=time.perf_counter()-started)
    args.output.mkdir(parents=True)
    (args.output / 'diagnostic.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.output / 'diagnostic_arrays.npz', teacher_features=features,
        autonomous_features=autonomous_features, truth=truth, teacher_prediction=teacher,
        autonomous_prediction=autonomous, relative_time=np.asarray(RELATIVE_TIMES))
    print(json.dumps({key: result[key] for key in ('status', 'true_image_windows', 'model_parameters_unchanged',
                                                'optimizer_updates', 'elapsed_seconds')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('checkpoint', 'readouts', 'seen', 'unseen', 'eval-results', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('Positive batch size and thread count required')
    run(args)
