#!/usr/bin/env python3
"""Check the uploaded code/data on the login node; no model optimization or Slurm submission."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from verify_bundle import verify
from dpbench.comparison_protocol import digest, preflight, source_manifest
from dpbench.damping_stage_data import validate_pair
from dpbench.comparison_hjepa import HJEPAComparison


def main():
    package = verify()
    plan = json.loads((ROOT / 'training_plan.json').read_text())
    result = dict(status='checking', package=package, checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  preflight={}, data_sha256={}, paired={}, slurm_test_only={},
                  formal_training_started=False, optimizer_created=False, jobs_submitted=False,
                  gpu_profile_status='pending allocation at explicit job submission')
    for name, path in plan['data'].items():
        sha = digest(path)
        if sha != plan['expected_data_sha256'][name]:
            raise RuntimeError(f'Dataset changed: {name}')
        result['data_sha256'][name] = sha
    for condition in plan['conditions']:
        cfg = json.loads((ROOT / 'code/configs/hjepa_retrain_20261109' / f'{condition}.json').read_text())
        result['preflight'][condition] = preflight(plan['data'][condition], cfg)
        result['paired'][condition] = validate_pair(plan['data']['a'], plan['data'][condition], multi=condition=='multi')
        # Construction verifies pinned official sources and trainability, without an optimizer.
        model = HJEPAComparison(cfg['model'])
        params = list(model.predictor.r.parameters())
        assert params and all(p.requires_grad for p in params)
        result['preflight'][condition]['dissipation_parameters_trainable'] = sum(p.numel() for p in params)
        del model
        check = subprocess.run(['sbatch', '--test-only', str(ROOT / f'submit_{condition}.sh')],
                               capture_output=True, text=True)
        result['slurm_test_only'][condition] = dict(returncode=check.returncode, stdout=check.stdout, stderr=check.stderr)
        if check.returncode:
            raise RuntimeError(f'Slurm test-only failed for {condition}: {check.stderr}')
    result['source_files_verified'] = len(source_manifest())
    result['status'] = 'ready_for_explicit_submission'
    (ROOT / 'remote_preflight.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'formal_training_started', 'jobs_submitted', 'source_files_verified')}))


if __name__ == '__main__':
    main()
