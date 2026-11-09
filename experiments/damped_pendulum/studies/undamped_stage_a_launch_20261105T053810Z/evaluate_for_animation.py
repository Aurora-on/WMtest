#!/usr/bin/env python3
"""Frozen step4000 train-only ridge calibration and val-only animation inputs.

No world-model optimizer, no test split access, no checkpoint selection. Future
true images enter only the explicitly named teacher readout diagnostic.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

ALPHAS = (1e-6, 1e-4, 1e-2, 1., 100.)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def split_training_records(records, fit_count=288, seed=5705):
    ordered = sorted(records, key=lambda row: row['group_id'])
    if len({r['group_id'] for r in ordered}) != len(ordered) or not 0 < fit_count < len(ordered):
        raise ValueError('Training groups must be unique and support disjoint nonempty partitions')
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    return ([ordered[i] for i in permutation[:fit_count]],
            [ordered[i] for i in permutation[fit_count:]])


def select_animation_records(records):
    """Selection uses immutable metadata only; never a prediction/error."""
    bands = sorted({r['energy_band'] for r in records})
    if bands != [0, 1, 2, 3]:
        raise ValueError('Expected four initial-energy bands')
    return [min((r for r in records if r['energy_band'] == band),
                key=lambda r: r['group_id']) for band in bands]


def ridge_coefficients(x, y, alpha):
    """Minimize mean((XW+b-Y)^2) + alpha*sum(W^2), intercept unpenalized.

    mean is over N times target_dimensions, so the normal equation adds
    N*target_dimensions*alpha to slope diagonals, and zero to the intercept.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    design = np.column_stack([x, np.ones(len(x))])
    penalty = np.eye(design.shape[1])
    penalty[-1, -1] = 0
    return np.linalg.solve(design.T @ design + len(x) * y.shape[1] * alpha * penalty,
                           design.T @ y)


def choose_ridge(x, y, xv, yv):
    arrays = [np.asarray(a, dtype=np.float64) for a in (x, y, xv, yv)]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite calibration feature/target')
    x, y, xv, yv = arrays
    xm, xs = x.mean(0), np.maximum(x.std(0), 1e-6)
    ym, ys = y.mean(0), np.maximum(y.std(0), 1e-6)
    xf, yf, vs, vy = (x - xm) / xs, (y - ym) / ys, (xv - xm) / xs, (yv - ym) / ys
    candidates = []
    for alpha in ALPHAS:
        weights = ridge_coefficients(xf, yf, alpha)
        score = float(np.mean((np.column_stack([vs, np.ones(len(vs))]) @ weights - vy) ** 2))
        candidates.append(dict(alpha=alpha, selection_normalized_mse=score, weights=weights))
    best = min(candidates, key=lambda row: row['selection_normalized_mse'])
    readout = dict(x_mean=xm, x_scale=xs, y_mean=ym, y_scale=ys, weights=best['weights'])
    return readout, best['alpha'], [{k: v for k, v in row.items() if k != 'weights'} for row in candidates]


def predict_readout(readout, features):
    features = np.asarray(features)
    flat = features.reshape(-1, features.shape[-1])
    good = np.isfinite(flat).all(1)
    predicted = np.full((len(flat), 2), np.nan)
    standardized = (flat[good] - readout['x_mean']) / readout['x_scale']
    y = (np.column_stack([standardized, np.ones(len(standardized))]) @ readout['weights']
         * readout['y_scale'] + readout['y_mean'])
    predicted[good] = np.column_stack([np.arctan2(y[:, 0], y[:, 1]), y[:, 2]])
    return predicted.reshape(*features.shape[:-1], 2)


def physical_metrics(pred, truth, omega_scale):
    if not len(pred):
        return dict(angle_mae_rad=None, angle_rmse_rad=None, omega_rmse_rad_s=None, omega_nrmse=None)
    angle = np.arctan2(np.sin(pred[..., 0] - truth[..., 0]), np.cos(pred[..., 0] - truth[..., 0]))
    omega_rmse = float(np.sqrt(np.mean((pred[..., 1] - truth[..., 1]) ** 2)))
    return dict(angle_mae_rad=float(np.mean(abs(angle))),
                angle_rmse_rad=float(np.sqrt(np.mean(angle ** 2))), omega_rmse_rad_s=omega_rmse,
                omega_nrmse=omega_rmse / max(float(omega_scale), 1e-6))


def load_frozen_helpers(code_root):
    code_root = Path(code_root).resolve()
    for name, module in list(sys.modules.items()):
        if name == 'dpbench' or name.startswith('dpbench.'):
            filename = getattr(module, '__file__', None)
            if filename and not Path(filename).resolve().is_relative_to(code_root):
                raise RuntimeError('dpbench was already imported from a different code root')
    sys.path.insert(0, str(code_root))
    from dpbench.dissipative_model import DissipativeWorldModel
    from dpbench.evaluate_passive import calibration_features, autonomous_rollout, EVALUATION_DIAGNOSTICS
    from dpbench.evaluate import pixels_tensor
    from dpbench.train_dissipative import source_hashes
    return DissipativeWorldModel, calibration_features, autonomous_rollout, pixels_tensor, source_hashes, EVALUATION_DIAGNOSTICS


def verify_evaluation_sources(code_root, manifest_path):
    files = json.loads(Path(manifest_path).read_text())['files']
    names = ('dpbench/__init__.py', 'dpbench/evaluate_passive.py', 'dpbench/evaluate.py',
             'dpbench/models.py', 'dpbench/data.py', 'dpbench/generate.py', 'dpbench/passive_family.py')
    actual = {name: sha256(Path(code_root) / name) for name in names}
    if any(files.get(name) != digest for name, digest in actual.items()):
        raise ValueError('Frozen evaluation helpers differ from the launch code_manifest')
    return actual


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError('Use a new evaluation output directory')
    if args.batch_size != 32:
        raise ValueError('Frozen feature extraction uses batch32, matching the training/qualification convention')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    manifest_path = args.code_manifest or args.code_root.parent / 'code_manifest.json'
    evaluation_hashes = verify_evaluation_sources(args.code_root, manifest_path)
    (Model, calibration, autonomous, pixels_tensor, source_hashes, diagnostic_names) = load_frozen_helpers(args.code_root)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved.get('kind') != 'dissipative' or saved.get('step') != 4000 or saved.get('validation_status') != 'completed':
        raise ValueError('Requires the completed step4000 checkpoint, without test-based selection')
    hashes = source_hashes()
    if saved.get('source_hashes') != hashes:
        raise ValueError('Frozen training sources do not exactly match checkpoint hashes')
    data_hash = sha256(args.data)
    if data_hash != saved.get('dataset_sha256'):
        raise ValueError('Data SHA256 differs from the trained dataset')
    if (saved['config']['core']['damping_mode'] != 'none'
            or saved['config']['environment_context_dim'] != 0):
        raise ValueError('Requires the undamped stage-A model')
    torch.set_num_threads(args.threads)
    model = Model(saved['config']).to(args.device).eval()
    model.load_state_dict(saved['state_dict'], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.history != 21 or model.dt != .1:
        raise ValueError('This animation protocol fixes history21 and dt=.1s')
    initial_hash = state_hash(model)
    with h5py.File(args.data, 'r') as f:
        if f.attrs.get('training_scope') != 'undamped_stage_a':
            raise ValueError('Requires undamped_stage_a data')
        records = {}
        # Deliberately never read splits/test or test observations/labels.
        for split in ('train', 'val'):
            records[split] = sorted([dict(episode=int(ep), group_id=int(f['group_id'][ep]),
                                                energy_band=int(f['energy_band'][ep]))
                                     for ep in f['splits/' + split][:]], key=lambda row: row['group_id'])
    if len(records['train']) != 384 or len(records['val']) != 64:
        raise ValueError('Expected full384 train and64 val episodes')
    fit_records, select_records = split_training_records(records['train'])
    selected_records = select_animation_records(records['val'])
    args.output.mkdir(parents=True)
    def phase(name):
        print(json.dumps(dict(phase=name, elapsed_seconds=time.perf_counter() - started)), flush=True)
    phase('calibrate_train_fit_288_groups')
    fit_pairs = [(r['episode'], t) for r in fit_records for t in range(20, 81)]
    select_pairs = [(r['episode'], t) for r in select_records for t in range(20, 81)]
    x, y = calibration(model, args.data, fit_pairs, args.batch_size)
    phase('calibrate_train_selection_96_groups')
    xv, yv = calibration(model, args.data, select_pairs, args.batch_size)
    readout, alpha, candidates = choose_ridge(x, y, xv, yv)
    np.savez_compressed(args.output / 'readout.npz', **readout, alpha=np.array(alpha),
                        fit_group_ids=np.array([r['group_id'] for r in fit_records]),
                        selection_group_ids=np.array([r['group_id'] for r in select_records]))
    phase('val_teacher_readout_diagnostic')
    val_records = records['val']
    teacher_pairs = [(r['episode'], t) for r in val_records for t in range(20, 53)]
    tx, _ = calibration(model, args.data, teacher_pairs, args.batch_size)
    teacher_features = tx.reshape(64, 33, -1)
    teacher = predict_readout(readout, teacher_features)
    phase('val_autonomous_K32')
    truths, trajectories, valid, energies, failures = [], [], [], [], []
    diagnostics = {name: [] for name in diagnostic_names}
    with h5py.File(args.data, 'r') as f:
        for start in range(0, len(val_records), args.batch_size):
            chunk = val_records[start:start + args.batch_size]
            images, actions, intervals = [], [], []
            for record in chunk:
                offset = int(f['ep_offset'][record['episode']])
                if int(f['ep_len'][record['episode']]) < 53:
                    raise ValueError('Val episode cannot cover anchor20 + K32')
                images.append(pixels_tensor(f['pixels'][offset:offset + 21], 'cpu'))
                actions.append(f['action'][offset:offset + 52])
                intervals.append(f['dt_frame'][offset:offset + 52])
                truths.append(f['state'][offset + 20:offset + 53])
            action = torch.as_tensor(np.stack(actions), device=args.device)
            dt = torch.as_tensor(np.stack(intervals), device=args.device)
            # No future image tensor is accepted by the autonomous API.
            out = autonomous(model, torch.stack(images).to(args.device), action[:, :20], dt[:, :20],
                             action[:, 20:52], dt[:, 20:52], diagnostic_names=diagnostic_names)
            trajectories.append(out['features']); valid.extend(out['valid_steps'].tolist())
            energies.append(out['latent_energy'])
            for key in diagnostics:
                diagnostics[key].append(out['diagnostics'][key])
            for failure in out['failures']:
                failures.append({**failure, 'evaluation_index': start + failure['batch_index'],
                                 'episode': chunk[failure['batch_index']]['episode']})
        selected_indices = [next(i for i, r in enumerate(val_records) if r['episode'] == s['episode'])
                            for s in selected_records]
        true_rgb = np.stack([f['pixels'][int(f['ep_offset'][r['episode']]) + 20:
                                            int(f['ep_offset'][r['episode']]) + 53] for r in selected_records])
    features, truth, valid_steps = np.concatenate(trajectories), np.asarray(truths), np.asarray(valid)
    pred = predict_readout(readout, features)
    teacher_quality = physical_metrics(teacher[:, 1:], truth[:, 1:], readout['y_scale'][2])
    quality_passed = teacher_quality['angle_mae_rad'] <= .1 and teacher_quality['omega_nrmse'] <= .15
    quality_status = ('读出质量通过预设检查；预测仍需独立评估' if quality_passed
                      else '读出质量不足：本图不能单独判断动力学')
    rollout = {}
    for horizon in (8, 32):
        eligible = (valid_steps >= horizon) & np.isfinite(pred[:, :horizon + 1]).all(axis=(1, 2))
        rollout[str(horizon)] = dict(horizon_steps=horizon, horizon_seconds=horizon * model.dt,
            n_total=64, n_complete=int(eligible.sum()), coverage=float(eligible.mean()),
            metrics=physical_metrics(pred[eligible, 1:horizon + 1], truth[eligible, 1:horizon + 1], readout['y_scale'][2]),
            support='Complete trajectories for this horizon only; failures retained separately and in NPZ')
    final_hash = state_hash(model)
    if final_hash != initial_hash:
        raise RuntimeError('Frozen model changed during diagnostic evaluation')
    checkpoint_hash = sha256(args.checkpoint)
    metadata = dict(episode=np.array([r['episode'] for r in val_records]),
                    group_id=np.array([r['group_id'] for r in val_records]),
                    energy_band=np.array([r['energy_band'] for r in val_records]))
    np.savez_compressed(args.output / 'trajectories.npz', truth=truth, pred=pred, teacher=teacher,
        features=features, teacher_features=teacher_features, valid_steps=valid_steps,
        time=np.arange(33) * model.dt, latent_energy=np.concatenate(energies),
        selected_indices=np.array(selected_indices), **metadata,
        **{'diagnostic_' + key: np.concatenate(value) for key, value in diagnostics.items()})
    selected = np.array(selected_indices)
    np.savez_compressed(args.output / 'animation_inputs.npz', true_rgb=true_rgb,
        truth=truth[selected], teacher_prediction=teacher[selected], autonomous_prediction=pred[selected],
        time=np.arange(33) * model.dt, valid_steps=valid_steps[selected],
        **{key: value[selected] for key, value in metadata.items()})
    for record, index in zip(selected_records, selected_indices):
        record.update(evaluation_index=index, valid_steps=int(valid_steps[index]),
                      solver_failures=[f for f in failures if f['evaluation_index'] == index])
    animation_meta = dict(checkpoint_step=4000, checkpoint_sha256=checkpoint_hash,
        selected_readout='ridge', selected_alpha=alpha, teacher_readout_quality=teacher_quality,
        readout_quality_passed=bool(quality_passed), quality_status_cn=quality_status,
        interpretation='diagnostic' if quality_passed else 'inconclusive_readout_quality',
        selected_records=selected_records, selection='Minimum group_id in each initial-energy band; no error-based selection',
        duration_seconds=3.2, dt_seconds=model.dt)
    write_json(args.output / 'animation_meta.json', animation_meta)
    summary = dict(status='completed', checkpoint_step=4000, checkpoint_sha256=checkpoint_hash,
        dataset_sha256=data_hash, source_hashes=hashes, script_sha256=sha256(__file__),
        evaluation_source_hashes=evaluation_hashes, code_manifest_sha256=sha256(manifest_path),
        code_root=str(args.code_root.resolve()), device=args.device, torch=str(torch.__version__),
        model_hash_before=initial_hash, model_hash_after=final_hash, parameters_unchanged=True,
        world_model_optimizer_created=False, world_model_updates=0, test_split_accessed=False,
        calibration=dict(fit_groups=288, selection_groups=96, frames_per_group=61, seed=5705,
            fit_labels=len(fit_pairs), selection_labels=len(select_pairs), selected_alpha=alpha,
            candidates=candidates, standardization='Fit partition only; per-feature/target std floor1e-6',
            objective='mean over samples and3 targets of standardized squared error + alpha*sum(slopes^2); intercept unpenalized'),
        teacher_readout_quality=teacher_quality, readout_quality_passed=bool(quality_passed),
        quality_status_cn=quality_status, rollout=rollout, failures=failures,
        metrics_definition='Future frames only, excluding anchor. Equal episode/time weighting on complete support. omegaNRMSE=RMSE/train288 fit-partition omega std (floor1e-6).',
        omega_nrmse_denominator=float(readout['y_scale'][2]),
        selected_records=selected_records, elapsed_seconds=time.perf_counter() - started,
        scope='Frozen step4000 diagnostic at val anchor20, K32=3.2s; not full stage-A acceptance or long-horizon test.')
    write_json(args.output / 'summary.json', summary)
    phase('completed')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code-root', 'checkpoint', 'data', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--code-manifest', type=Path,
                        help='Launch code_manifest.json; defaults to code-root parent/code_manifest.json')
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
