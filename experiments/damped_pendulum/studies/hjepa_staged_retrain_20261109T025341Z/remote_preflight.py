#!/usr/bin/env python3
"""Validate uploaded staged recipe and Slurm syntax without submitting jobs."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from verify_bundle import verify
from dpbench.comparison_protocol import digest, source_manifest
from dpbench.comparison_hjepa import HJEPAComparison
from scripts.run_hjepa_retrain import prepare


def main():
    torch.set_num_threads(4)
    plan = json.loads((ROOT / 'training_plan.json').read_text())
    report = dict(status='checking', package=verify(), checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  data_sha256={}, stages={}, slurm_test_only={},
                  formal_training_started=False, optimizer_created=False, jobs_submitted=False,
                  gpu_profile_status='pending allocated job', parent_checkpoint_expected_but_not_required_for_upload=True)
    for name, path in plan['data'].items():
        sha = digest(path)
        if sha != plan['expected_data_sha256'][name]:
            raise RuntimeError(f'Dataset changed: {name}')
        report['data_sha256'][name] = sha
    for stage in plan['stages']:
        cfg = json.loads((ROOT / 'code/configs/hjepa_staged_retrain_20261109' / f'{stage}.json').read_text())
        parent = None if stage == 'a' else ROOT / 'runs/a'
        manifest, checked = prepare(cfg, plan['data'][stage], plan['data']['a'], 'cpu', parent_output=parent)
        model = HJEPAComparison(cfg['model'])
        params = list(model.predictor.r.parameters())
        assert params and all(p.requires_grad for p in params)
        report['stages'][stage] = dict(preflight=checked, pipeline_manifest=manifest,
                                      dissipation_parameters_trainable=sum(p.numel() for p in params))
        del model
        command = ['sbatch', '--test-only']
        if stage != 'a': command.append('--kill-on-invalid-dep=yes')
        command.append(str(ROOT / f'submit_{stage}.sh'))
        result = subprocess.run(command, capture_output=True, text=True)
        report['slurm_test_only'][stage] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode:
            raise RuntimeError(f'Slurm test-only failed for {stage}: {result.stderr}')
    report['source_files_verified'] = len(source_manifest())
    report['status'] = 'ready_for_explicit_staged_submission'
    (ROOT / 'remote_preflight.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('status','formal_training_started','jobs_submitted','source_files_verified')}))


if __name__ == '__main__':
    main()
