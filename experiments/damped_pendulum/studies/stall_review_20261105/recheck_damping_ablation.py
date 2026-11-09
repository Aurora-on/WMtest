"""Independently summarize existing frozen damping interventions; no training.

Recomputes readout error, latent motion, potential and kinetic energy from saved
trajectories. Does not run or alter any optimizer, weights or original outputs.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from dpbench.efficient_dissipative_core import EfficientDissipativeCore
from dpbench.evaluate import decode_state
from dpbench.train_dissipative import source_hashes
from scripts.diagnose_teacher_forced_readout import load_readout

torch.set_num_threads(2)
SOURCE = ROOT / 'studies/stall_diagnosis_20261105'
EVAL = ROOT / 'studies/efficient3_evaluation_20261105T023318Z'
checkpoint = SOURCE / 'frozen_training/best.pt'
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['source_hashes'] == source_hashes()
config = saved['config']
core = EfficientDissipativeCore(q_dim=config['q_dim'],
                              context_dim=config['environment_context_dim'],
                              **config['core'])
state = {k[5:]: v for k, v in saved['state_dict'].items() if k.startswith('core.')}
core.load_state_dict(state, strict=True)
core.eval()
for parameter in core.parameters():
    parameter.requires_grad_(False)
probe = load_readout(SOURCE / 'frozen_training/readouts.pt', 'mlp')
previous = json.loads((SOURCE / 'step_sensitivity.json').read_text())
arrays = np.load(SOURCE / 'step_sensitivity_arrays.npz')
teacher = np.load(EVAL / 'teacher_received/diagnostic_arrays.npz')['teacher_features']
selected = [i for i, row in enumerate(previous['selected']) if row['damping'] == 0.]
truth = arrays['truth'][selected]
results = []
for name in ('dg12', 'dg12_D01', 'dg12_Dzero'):
    z = arrays[name][selected]
    decoded = decode_state(probe.predict(z))
    assert np.allclose(decoded, arrays[name + '_physical'][selected], atol=1e-6, rtol=1e-6)
    error = np.arctan2(np.sin(decoded[:, 1:, 0] - truth[:, 1:, 0]),
                       np.cos(decoded[:, 1:, 0] - truth[:, 1:, 0]))
    q, p = torch.tensor(z[..., :2]), torch.tensor(z[..., 2:])
    with torch.no_grad():
        potential = core.potential(q.reshape(-1, 2)).numpy().reshape(z.shape[:2])
        restoring = core.potential_gradient(q.reshape(-1, 2)).norm(dim=-1).numpy().reshape(z.shape[:2])
    kinetic = .5 * np.sum(z[..., 2:] ** 2, axis=-1)
    energy = potential + kinetic
    path_length = np.linalg.norm(np.diff(z[..., :2], axis=1), axis=-1).sum(axis=1)
    chord_length = np.linalg.norm(z[:, -1, :2] - z[:, 0, :2], axis=-1)
    rows = []
    for i, idx in enumerate(selected):
        rows.append(dict(
            evaluation_index=previous['selected'][idx]['evaluation_index'],
            energy_band=previous['selected'][idx]['energy_band'],
            cumulative_angle_rmse_rad=float(np.sqrt(np.mean(error[i] ** 2))),
            cumulative_omega_rmse_rad_s=float(np.sqrt(np.mean((decoded[i, 1:, 1] - truth[i, 1:, 1]) ** 2))),
            q_initial=z[i, 0, :2].tolist(), q_final=z[i, -1, :2].tolist(),
            p_final=z[i, -1, 2:].tolist(),
            q_path_length=float(path_length[i]), q_chord_length=float(chord_length[i]),
            path_straightness=float(chord_length[i] / path_length[i]),
            potential_initial=float(potential[i, 0]), potential_final=float(potential[i, -1]),
            kinetic_initial=float(kinetic[i, 0]), kinetic_final=float(kinetic[i, -1]),
            learned_energy_initial=float(energy[i, 0]), learned_energy_final=float(energy[i, -1]),
            learned_energy_max_abs_change=float(np.max(np.abs(energy[i] - energy[i, 0]))),
            restoring_force_initial=float(restoring[i, 0]), restoring_force_final=float(restoring[i, -1]),
        ))
    aggregate = dict(
        variant=name, n=len(selected), duration_seconds=8,
        cumulative_angle_rmse_rad=float(np.sqrt(np.mean(error ** 2))),
        cumulative_omega_rmse_rad_s=float(np.sqrt(np.mean((decoded[:, 1:, 1] - truth[:, 1:, 1]) ** 2))),
        final_q_step_rms=float(np.sqrt(np.mean(np.diff(z[..., :2], axis=1)[:, -1] ** 2))),
        final_p_rms=float(np.sqrt(np.mean(z[:, -1, 2:] ** 2))),
        final_potential_mean=float(potential[:, -1].mean()),
        final_restoring_force_mean=float(restoring[:, -1].mean()),
        final_q_norm_mean=float(np.linalg.norm(z[:, -1, :2], axis=-1).mean()),
        rows=rows,
    )
    old = next(r for r in previous['variants'] if r['name'] == name)
    old_metric = next(r for r in old['metrics'] if r['damping'] == 0. and r['time_s'] == 8)
    assert abs(aggregate['cumulative_angle_rmse_rad'] - old_metric['cumulative']['angle_rmse_rad']) < 1e-8
    results.append(aggregate)

unchanged = all(torch.equal(v, core.state_dict()[k]) for k, v in state.items())
assert unchanged
output = dict(
    scope='Independent calculation from existing frozen 8-second DG12 intervention arrays; no new optimization, integration or test-based model selection.',
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    source_hashes_match_checkpoint=True,
    weights_unchanged=unchanged,
    optimizer_updates=0,
    sampling='Four metadata-selected genuinely undamped episodes, one per energy band, selected before inspecting prediction error.',
    limits='Intervening on D changes the learned model; readout may extrapolate outside its calibrated latent manifold. This is a mechanism probe, not a corrected-model benchmark. Four episodes do not establish population performance.',
    true_encoded_q_norm_range=[float(np.linalg.norm(teacher[..., :2], axis=-1).min()),
                               float(np.linalg.norm(teacher[..., :2], axis=-1).max())],
    variants=results,
    input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  (SOURCE / 'step_sensitivity.json', SOURCE / 'step_sensitivity_arrays.npz',
                   SOURCE / 'frozen_training/readouts.pt')},
)
(HERE / 'damping_ablation_recheck.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: v for k, v in output.items() if k != 'variants'}, indent=2))
for row in results:
    print(json.dumps({k: v for k, v in row.items() if k != 'rows'}, indent=2))
