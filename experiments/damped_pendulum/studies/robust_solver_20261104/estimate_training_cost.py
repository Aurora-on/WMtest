#!/usr/bin/env python3
"""Plan GPU wall time from actual qualification updates, never from loss values.

The low/high scenarios use observed p10/p90 step times, except K4 uses the saved
min/max profile summary. They are not confidence intervals or guarantees about
later learned dynamics. K8/K16 use normal training; K32 uses the forced-horizon
base-LR stress.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values, fraction):
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    left = math.floor(at)
    return ordered[left] + (ordered[math.ceil(at)] - ordered[left]) * (at - left)


def summarize(values, source):
    if not values or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError(f'Missing or invalid timings: {source}')
    return dict(source=source, observations=len(values), mean=statistics.mean(values),
                low=percentile(values, .1), high=percentile(values, .9),
                low_high_statistic='observed p10/p90',
                maximum=max(values), seconds_unit='per optimizer update')


def estimate(root, resume_step):
    profile = json.loads((root / 'fresh_profile.json').read_text())
    status = json.loads((root / 'continuation/status.json').read_text())
    stress_manifest = json.loads((root / 'k32_stress/manifest.json').read_text())
    normal = rows(root / 'continuation/history.jsonl')
    stress = rows(root / 'k32_stress/history.jsonl')
    if (status['status'] != 'budget_completed' or status['completed_steps'] != 4250
            or len(normal) != 1500 or stress_manifest['status'] != 'completed'
            or len(stress) != 500 or not stress_manifest['ready_numerically']):
        raise ValueError('Complete qualification evidence is required for final timing estimates')
    cfg = stress_manifest['config']['training']
    if not 0 <= resume_step < cfg['steps']:
        raise ValueError('Resume step must precede the configured training budget')
    measured = {'4': dict(source='fresh 24-update profile (only four timed K4 updates)',
                         observations=profile['measurements']['4']['timed_updates'],
                         mean=profile['measurements']['4']['mean_seconds'],
                         low=profile['measurements']['4']['min_seconds'],
                         high=profile['measurements']['4']['max_seconds'],
                         low_high_statistic='observed min/max (only summary available)',
                         maximum=profile['measurements']['4']['max_seconds'],
                         seconds_unit='per optimizer update')}
    for horizon in (8, 16):
        measured[str(horizon)] = summarize(
            [r['step_seconds'] for r in normal if r['horizon_steps'] == horizon],
            f'normal schedule, {horizon}-step prediction')
    measured['32'] = summarize([r['seconds'] for r in stress],
                               '500 K32 stress updates from trained weights, full base LR')
    validation = summarize([r['validation_seconds'] for r in normal
                            if r['validation_seconds'] > 0], 'scheduled complete validation')
    checkpoint = summarize([r['checkpoint_seconds'] for r in normal
                            if r['checkpoint_seconds'] > 0], 'scheduled checkpoint write')
    remaining = {}
    previous = 0
    for stage in cfg['curriculum']:
        count = max(0, stage['until_step'] - max(resume_step, previous))
        remaining[str(stage['horizon'])] = remaining.get(str(stage['horizon']), 0) + count
        previous = stage['until_step']
    validations = cfg['steps'] // cfg['val_every'] - resume_step // cfg['val_every']
    scenarios = {}
    for key in ('low', 'mean', 'high'):
        update_seconds = sum(count * measured[h][key] for h, count in remaining.items())
        overhead_seconds = validations * (validation[key] + checkpoint[key])
        scenarios[key] = dict(optimizer_hours=update_seconds / 3600,
                              validation_and_checkpoint_hours=overhead_seconds / 3600,
                              total_hours=(update_seconds + overhead_seconds) / 3600)
    return dict(resume_step=resume_step, target_step=cfg['steps'],
                remaining_updates_by_horizon=remaining, remaining_validation_calls=validations,
                measured_update_seconds=measured, validation_seconds=validation,
                checkpoint_seconds=checkpoint, planning_scenarios=scenarios,
                caveats=[
                    'Observed timing scenarios, not confidence intervals or a future runtime guarantee.',
                    'Learned dynamics, line search and subdivision may change later iteration costs.',
                    'K32 evidence is a 500-update stress run, not the whole late-training stage.',
                    'Excludes queue delay, preflight/source checks and new physical evaluation.',
                    'Does not count a fresh validation when resuming an already validated checkpoint.'
                ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--resume-step', type=int, default=4250)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = estimate(args.results, args.resume_step)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['planning_scenarios'], indent=2))
