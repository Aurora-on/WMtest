"""Run one frozen dimension arm under Slurm, with qualification and exact resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def verify_files(root, manifest):
    for relative, expected in manifest['bundle_files_sha256'].items():
        if digest(root / relative) != expected:
            raise RuntimeError(f'Frozen file changed: {relative}')


def checkpoint_audit(path, step, dimension, config):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['step'] != step or saved['sampler_next_step'] != step:
        raise RuntimeError('Checkpoint step/sampler mismatch')
    if saved['validation_status'] != 'completed':
        raise RuntimeError('Checkpoint validation is incomplete')
    if saved['train_config'] != config or saved['config']['q_dim'] != dimension:
        raise RuntimeError('Checkpoint configuration mismatch')
    states = saved['optimizer']['state']
    if not states:
        raise RuntimeError('Optimizer has no state')
    optimizer_steps = sorted({int(v['step']) for v in states.values() if 'step' in v})
    if optimizer_steps != [step]:
        raise RuntimeError(f'Optimizer step mismatch: {optimizer_steps}')
    return dict(step=step, optimizer_steps=optimizer_steps, sha256=digest(path),
                validation_completed=True, dimension=dimension)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dimension', type=int, choices=(8, 32), required=True)
    args = parser.parse_args()
    root = args.manifest.resolve().parent
    manifest = json.loads(args.manifest.read_text())
    verify_files(root, manifest)
    code = root / 'code'
    sys.path.insert(0, str(code))
    from dpbench.train_dissipative import read_config
    config_path = root / 'configs' / f'lehami_d{args.dimension}_s0.json'
    config = read_config(config_path)
    arm_root = root / 'runs' / f'd{args.dimension}'
    arm_root.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, PENDULUM_CODE_DIR=str(code), PYTHONPATH=str(code))

    def run(command):
        verify_files(root, manifest)
        print('RUN', ' '.join(map(str, command)), flush=True)
        subprocess.run(list(map(str, command)), cwd=code, env=environment, check=True)

    run([sys.executable, root / 'qualify_dimensions.py', '--config', config_path,
         '--data', manifest['data'], '--reference-run', manifest['reference_run'],
         '--reference-qualification', manifest['reference_qualification'],
         '--output', arm_root / 'cuda_qualification.json'])
    resume = None
    audits = []
    for stop, segment in [(4000, 'first4000'), (10000, 'to10000')]:
        output = arm_root / segment
        command = [sys.executable, '-m', 'dpbench.train_dissipative',
                   '--config', config_path, '--data', manifest['data'],
                   '--output', output, '--device', 'cuda', '--stop-after-step', stop,
                   '--execute']
        if resume is not None:
            command += ['--resume', resume]
        run(command)
        resume = output / 'last.pt'
        audits.append(checkpoint_audit(resume, stop, args.dimension, config))
        (arm_root / 'completion_audit.json').write_text(json.dumps(dict(
            dimension=args.dimension, completed_steps=stop, checkpoints=audits,
            training_complete=stop == 10000, physical_evaluation_complete=False,
            job_id=os.environ.get('SLURM_JOB_ID')), indent=2) + '\n')


if __name__ == '__main__':
    main()
