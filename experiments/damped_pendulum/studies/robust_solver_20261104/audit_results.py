#!/usr/bin/env python3
"""Offline, independent audit of downloaded solver qualification evidence.

This does not run training or use SSH. Missing evidence is not success. Run with
this project's Python (numpy/torch). The expected result layout is the server's
robust_solver_20261104_v2 root, downloaded without renaming its subdirectories.
The numerical checks do not establish physical accuracy or global convergence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finite_tree(x):
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(finite_tree(v) for v in x)
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, torch.Tensor):
        return bool(torch.isfinite(x).all())
    return True


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


class Auditor:
    def __init__(self, results, source_root, original_checkpoint, manifest):
        self.results, self.source_root = Path(results), Path(source_root)
        self.original_checkpoint, self.hash_manifest = Path(original_checkpoint), Path(manifest)
        self.checks, self.summary = [], {}

    def check(self, name, condition, evidence=None, category='numerical'):
        self.checks.append(dict(name=name, status='pass' if condition else 'fail', evidence=evidence,category=category))

    def section(self, name, function):
        try:
            function()
        except FileNotFoundError as exc:
            self.checks.append(dict(name=name, status='missing', evidence=str(exc)))
        except Exception as exc:
            self.checks.append(dict(name=name, status='fail', evidence=f'{type(exc).__name__}: {exc}'))

    def json(self, relative):
        return json.loads((self.results / relative).read_text())

    def rows(self, relative):
        return [json.loads(s) for s in (self.results / relative).read_text().splitlines() if s.strip()]

    def load(self, relative):
        return torch.load(self.results / relative, map_location='cpu', weights_only=True)

    def energy_model(self, relative):
        # Recompute H from saved q,p and the bound model, not only stored H values.
        sys.path.insert(0,str(self.source_root))
        from dpbench.dissipative_model import DissipativeWorldModel
        checkpoint=self.load(relative)
        model=DissipativeWorldModel(checkpoint['config']).eval()
        model.load_state_dict(checkpoint['state_dict'])
        model.requires_grad_(False)
        return model

    def recomputed_energy(self, model, q, p, stored, tag):
        with torch.no_grad():
            value=model.core.energy(torch.from_numpy(np.array(q,copy=True)),
                                    torch.from_numpy(np.array(p,copy=True))).numpy()
        self.check(tag+'_energy_recomputed_from_states', np.isfinite(value).all() and
                   np.all(abs(value-stored)<=1e-10*(1+abs(value)+abs(stored))))
        return value

    def checkpoint_contract(self, checkpoint, config, data_hash, hashes, runtime, tag):
        # This is the actual trainer's read-only validator. Supplying the recorded
        # CUDA runtime does not initialize CUDA or run/resume training on this CPU.
        sys.path.insert(0,str(self.source_root))
        from dpbench.train_dissipative import validate_checkpoint
        validate_checkpoint(checkpoint,config,data_hash,hashes,torch.device('cuda'),runtime)
        self.check(tag+'_trainer_resume_contract', True)

    def diagnostic_shapes(self, z, tag):
        required={'solver_residual','solver_residual_ratio','solver_split_depth','solver_iterations',
                  'solver_replay_relative_error','solver_failed_attempts','solver_backtracks','solver_leaf_steps',
                  'dissipation','work','energy_residual','numerical_balance_defect','residual_work','chain_defect'}
        self.check(tag+'_diagnostic_shapes', required<=set(z.files) and
                   all(z[k].shape==(32,320) for k in required if k in z.files))

    def representation_gate(self, validation, config, tag, require_pass=False):
        """Recompute the configured noncollapse gate; it is not prediction accuracy."""
        cfg=config['training'];groups=validation['groups'];required=[];consistent=bool(groups)
        horizon_counts={h:0 for h in cfg['val_horizons']}
        for name,row in groups.items():
            match=re.fullmatch(r'h(\d+)_(?:regime\d+_)?mode\d+_time(\d+)',name)
            if not match:raise ValueError(f'Unknown validation group label: {name}')
            horizon,time_bin=map(int,match.groups())
            horizon_counts[horizon]=horizon_counts.get(horizon,0)+row['samples']
            dimensions=sorted(k.removeprefix('target_mean_') for k in row if k.startswith('target_mean_'))
            if not dimensions:raise ValueError('Validation group missing target moment statistics')
            std_min=math.sqrt(min(max(0.,row[f'target_second_moment_{d}']-row[f'target_mean_{d}']**2) for d in dimensions))
            motion_rms=math.sqrt(max(0.,row['target_motion_mse']))
            consistent=(consistent and abs(std_min-row['target_std_min'])<=1e-12*(1+std_min) and
                        abs(motion_rms-row['target_motion_rms'])<=1e-12*(1+motion_rms))
            gate=std_min>=cfg['minimum_target_std'] and motion_rms>=cfg['minimum_target_motion']
            gate_required=cfg.get('validation_gate_time_bins') is None or time_bin in cfg['validation_gate_time_bins']
            consistent=consistent and row['gate_passed']==gate and row['gate_required']==gate_required
            if gate_required:required.append((name,gate,row))
        passed=bool(required) and all(item[1] for item in required)
        self.check(tag+'_gate_statistics_consistent', finite_tree(validation) and consistent and
                   validation['gate_passed']==passed and set(horizon_counts)==set(cfg['val_horizons']) and
                   all(n==cfg['val_windows'] for n in horizon_counts.values()))
        if require_pass:
            self.check(tag+'_representation_gate', passed,
                       {'required_groups':len(required),'failed_groups':[name for name,gate,_ in required if not gate],
                        'minimum_required_target_std':min((row['target_std_min'] for _,_,row in required),default=None),
                        'minimum_required_target_motion':min((row['target_motion_rms'] for _,_,row in required),default=None),
                        'scope':'Configured initial time bins; nondegenerate representation only, not prediction accuracy'},
                       category='representation')
        return passed

    def sources(self):
        hashes = json.loads(self.hash_manifest.read_text())
        mismatch = [p for p, h in hashes.items() if sha256(self.source_root / p) != h]
        self.check('frozen_source_files', not mismatch, dict(count=len(hashes), mismatch=mismatch))
        if (self.results / 'code_hashes.json').exists():
            self.check('downloaded_source_manifest_matches', self.json('code_hashes.json') == hashes)
        cuda = (self.results / 'cuda_tests.log').read_text()
        self.check('cuda_numerical_suite', bool(re.search(r'\b17 passed\b', cuda)) and 'FAILED' not in cuda, cuda[-300:])

    def profile(self):
        p = self.json('fresh_profile.json')
        run=self.json('continuation/run.json')
        self.check('fresh_profile_complete', p['status'] == 'completed' and p['optimizer_updates_completed'] == 24)
        measurements = p['measurements']
        self.check('fresh_all_horizons_updated', set(measurements) == {'4','8','16','32'} and
                   all(v['parameters_actually_changed'] and v['optimizer_state_entries'] > 0 and
                       v['losses_finite'] for v in measurements.values()))
        self.check('fresh_profile_finite', finite_tree(p))
        self.check('fresh_profile_source_config_data_runtime', p['effective_config']==run['train_config'] and
                   p['config_sha256']==run['config_sha256'] and p['dataset_sha256']==run['dataset_sha256'] and
                   p['runtime']==run['runtime'] and p['source_sha256']==
                   {**run['source_hashes'],'scripts/profile_dissipative.py':sha256(self.source_root/'scripts/profile_dissipative.py')})
        self.summary['fresh_profile_step_seconds'] = {k:v['mean_seconds'] for k,v in measurements.items()}

    def migration(self):
        original = torch.load(self.original_checkpoint, map_location='cpu', weights_only=True)
        migrated, report = self.load('migration/last.pt'), self.json('migration/migration.json')
        self.check('migration_input_hash', report['original_checkpoint_sha256'] == sha256(self.original_checkpoint))
        kept = ('state_dict','optimizer','rng','sampler_next_step','validation_indices','dataset_sha256')
        self.check('migration_preserves_training_state', all(equal_tree(original[k], migrated[k]) for k in kept), list(kept))
        self.check('migration_selection_reset', migrated['validation_status'] == 'not_run' and
                   migrated['validation'] is None and migrated['best_score'] is None and migrated['best_step'] is None)
        self.check('migration_step_and_schedule', migrated['step'] == 2750 and
                   migrated['train_config']['training'] == original['train_config']['training'])
        self.check('migration_strict_original_tolerances', all(migrated['config']['core'][k] == original['config']['core'][k]
                   for k in ('residual_atol','residual_rtol','chain_atol','strict','substeps','integrator')))
        self.check('migration_robust_controls', migrated['config']['core']['solver'] == 'backtracking' and
                   migrated['config']['core']['newton_iterations'] == 12 and migrated['config']['core']['max_bisections'] == 4)
        observed = {k:sha256(self.source_root/k) for k in migrated['source_hashes']}
        self.check('migration_new_source_binding', observed == migrated['source_hashes'])
        self.check('migration_provenance_binding', report['new_source_hashes']==migrated['source_hashes'] and
                   report['original_source_hashes']==original['source_hashes'] and
                   report['new_config_sha256']==migrated['config_sha256'] and
                   report['original_config_sha256']==original['config_sha256'] and
                   report['migration_script_sha256']==sha256(self.source_root/'scripts/migrate_solver_checkpoint.py') and
                   migrated['solver_migration']==report)
        expected_config=json.loads((self.source_root/'configs/dissipative_passive_robust.json').read_text())
        self.checkpoint_contract(migrated,expected_config,original['dataset_sha256'],observed,original['runtime'],'migration')

    def continuation(self):
        status = self.json('continuation/status.json')
        run = self.json('continuation/run.json')
        rows = self.rows('continuation/history.jsonl')
        checkpoint = self.load('continuation/last.pt')
        self.check('continuation_budget_complete', status['status'] == 'budget_completed' and
                   status['completed_steps'] == 4250 and status['optimizer_steps_this_run'] == 1500 and
                   status['full_budget_completed'] is False)
        self.check('continuation_complete_step_sequence', [x['step'] for x in rows] == list(range(2751,4251)))
        self.check('continuation_correct_curriculum', all(x['horizon_steps'] == (8 if x['step'] <= 4000 else 16) for x in rows))
        cfg=checkpoint['train_config']['training']
        def expected_lr(step):
            if cfg['warmup_steps'] and step<=cfg['warmup_steps']:
                return step/cfg['warmup_steps']
            progress=(step-cfg['warmup_steps'])/max(1,cfg['steps']-cfg['warmup_steps'])
            return cfg['minimum_lr_ratio']+(1-cfg['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*progress))
        self.check('continuation_original_full_budget_lr', cfg['steps']==20000 and
                   all(abs(x['lr_factor']-expected_lr(x['step']))<=1e-14 for x in rows))
        self.check('continuation_finite_log', finite_tree(rows))
        self.check('continuation_residual_bounds', all(x['losses']['solver_residual_ratio'] <= 1+1e-10 and
                   x['losses']['solver_split_depth'] <= 4 for x in rows))
        due = [x['step'] for x in rows if 'validation' in x]
        self.check('scheduled_validations_present', due == list(range(3000,4251,250)), due)
        reval = self.json('continuation/resume_validation.json')
        self.check('migration_mandatory_revalidation', reval['step'] == 2750 and reval['optimizer_steps'] == 0)
        self.representation_gate(reval['validation'],checkpoint['train_config'],'migration_revalidation')
        for row in rows:
            if 'validation' in row:
                self.representation_gate(row['validation'],checkpoint['train_config'],f'validation_{row["step"]}')
        self.representation_gate(checkpoint['validation'],checkpoint['train_config'],'continuation_final',require_pass=True)
        self.check('continuation_resumable_checkpoint', checkpoint['step'] == 4250 and checkpoint['sampler_next_step'] == 4250 and
                   checkpoint['validation_status'] == 'completed' and bool(checkpoint['optimizer']['state']) and
                   checkpoint['train_config']['training']['steps'] == 20000 and finite_tree(checkpoint['state_dict']) and
                   finite_tree(checkpoint['optimizer']))
        self.check('continuation_source_binding', checkpoint['source_hashes'] == run['source_hashes'] ==
                   {k:sha256(self.source_root/k) for k in checkpoint['source_hashes']})
        migrated=self.load('migration/last.pt')
        self.checkpoint_contract(checkpoint,migrated['train_config'],migrated['dataset_sha256'],
                                 migrated['source_hashes'],migrated['runtime'],'continuation')
        self.check('continuation_migrated_contract_preserved', all(checkpoint[k]==migrated[k] for k in
                   ('config','train_config','validation_indices','dataset_sha256','runtime','seed','train_frames','solver_migration')) and
                   all(checkpoint[k]==run[k] for k in ('config','train_config','config_sha256','dataset_sha256','runtime','validation_indices')))
        self.check('continuation_parameters_actually_changed', not equal_tree(checkpoint['state_dict'],migrated['state_dict']) and
                   any(x['gradient_norm']>0 for x in rows))
        if checkpoint['best_score'] is not None:
            best=self.load('continuation/best.pt')
            self.check('continuation_resume_best_pair', best['step']==checkpoint['best_step'] and
                       best['best_score']==checkpoint['best_score'] and best['step']<=checkpoint['step'] and
                       all(best[k]==checkpoint[k] for k in ('source_hashes','dataset_sha256','train_config','config_sha256','runtime')))
        self.summary['continuation'] = dict(elapsed_seconds=status.get('elapsed_seconds'), best_step=checkpoint['best_step'],
            last_validation_score=checkpoint['validation']['score'], last_validation_gate=checkpoint['validation']['gate_passed'],
            mean_seconds_by_horizon={str(h):float(np.mean([x['step_seconds'] for x in rows if x['horizon_steps']==h]))
                                     for h in (8,16) if any(x['horizon_steps']==h for x in rows)},
            max_solver_split_depth=max(x['losses']['solver_split_depth'] for x in rows),
            max_replay_relative_error=max(x['losses']['solver_replay_relative_error'] for x in rows),
            validation_trend=[{'step':r['step'],'score':r['validation']['score'],'gate_passed':r['validation']['gate_passed']}
                              for r in rows if 'validation' in r],
            resumed_model_path='continuation/last.pt')

    def stress(self):
        m = self.json('k32_stress/manifest.json')
        rows = self.rows('k32_stress/history.jsonl')
        self.check('stress_complete', m['status'] == 'completed' and m['optimizer_updates_completed'] == 500 and m['ready_numerically'])
        self.check('stress_checkpoint_binding', m['checkpoint_step'] == 4250 and m['checkpoint_sha256'] == sha256(self.results/'continuation/last.pt'))
        self.check('stress_step_sequence', [r['stress_update'] for r in rows] == list(range(1,501)) and
                   [r['sampler_absolute_step'] for r in rows] == list(range(4251,4751)) and all(r['horizon']==32 for r in rows))
        self.check('stress_finite_log', finite_tree(rows))
        self.check('stress_residual_bounds', all(r['losses']['solver_residual_ratio'] <= 1+1e-10 and
                   r['losses']['solver_split_depth'] <= 4 for r in rows))
        self.check('stress_scope', m['test_samples_read'] is False and m['physical_readout_fitted'] is False)
        self.check('stress_source_binding', m['source_hashes'] == {k:sha256(self.source_root/k) for k in m['source_hashes']} and
                   m['script_sha256'] == sha256(self.source_root/'scripts/qualify_robust_solver.py'))
        weights = self.load('k32_stress/stress_weights.pt')
        self.check('stress_artifact_not_formal_resume', weights['kind']=='robust_solver_stress_weights' and
                   weights['updates']==500 and 'optimizer' not in weights and finite_tree(weights['state_dict']))
        checkpoint=self.load('continuation/last.pt')
        self.check('stress_config_data_runtime_binding', m['config']==checkpoint['train_config'] and
                   m['data_sha256']==checkpoint['dataset_sha256'] and m['runtime']==checkpoint['runtime'])
        self.representation_gate(m['validation'],checkpoint['train_config'],'stress_final',require_pass=True)
        self.check('stress_parameters_actually_changed', weights['base_checkpoint_sha256']==sha256(self.results/'continuation/last.pt') and
                   weights['config']==checkpoint['config'] and not equal_tree(weights['state_dict'],checkpoint['state_dict']) and
                   any(x['gradient_norm']>0 for x in rows))
        self.summary['stress'] = dict(mean_update_seconds=m['mean_update_seconds'], elapsed_seconds=m['elapsed_seconds'],
            validation_score=m['validation']['score'], validation_gate=m['validation']['gate_passed'],
            max_solver_split_depth=max(x['losses']['solver_split_depth'] for x in rows),
            max_replay_relative_error=max(x['losses']['solver_replay_relative_error'] for x in rows))

    def rollout_arrays(self):
        m = self.json('k32_stress/manifest.json')
        model=self.energy_model('k32_stress/stress_weights.pt')
        self.check('rollout_manifest_coverage', m['long_rollout_total'] == m['long_rollout_completed'] == 384 and
                   m['long_rollout_steps'] == 320 and not m['long_rollout_failures'])
        files = sorted((self.results/'k32_stress').glob('rollout_*.npz'))
        self.check('rollout_batch_files_present', [p.name for p in files] == [f'rollout_{i:04d}.npz' for i in range(0,384,32)])
        totals=0;max_identity=0.;max_num_defect=0.;max_depth=0.;max_ratio=0.;norms=[]
        for path in files:
            with np.load(path,allow_pickle=False) as z:
                self.diagnostic_shapes(z,path.name)
                self.check(path.name+'_finite_and_complete', all(np.isfinite(z[k]).all() for k in z.files) and
                           z['features'].shape == (32,321,4) and z['latent_energy'].shape==(32,321) and
                           z['valid_steps'].shape==(32,) and np.all(z['valid_steps']==320))
                h=self.recomputed_energy(model,z['features'][...,:2],z['features'][...,2:],z['latent_energy'],path.name)
                d=z['dissipation'];w=z['work'];e=z['numerical_balance_defect']
                actual=np.diff(h,axis=1)-w+d
                scale=1+abs(h[:,:-1])+abs(h[:,1:])+abs(d)+abs(w)
                identity=abs(actual-e)
                sequence_error=abs(h[:,-1]-h[:,0]-w.sum(1)+d.sum(1)-e.sum(1))
                sequence_scale=1+abs(h[:,-1])+abs(h[:,0])+abs(w).sum(1)+abs(d).sum(1)
                self.check(path.name+'_ledger', np.all(identity<=1e-10*scale) and
                           np.all(abs(e-z['residual_work']-z['chain_defect'])<=1e-10*scale) and
                           np.all(abs(actual-z['energy_residual'])<=1e-10*scale) and
                           np.all(sequence_error<=1e-10*sequence_scale))
                self.check(path.name+'_acceptance', np.all(z['solver_residual_ratio']<=1+1e-10) and
                           np.all(z['solver_split_depth']<=4) and np.all(d>=-1e-10) and np.all(abs(w)<=1e-12))
                totals+=len(z['valid_steps']);max_identity=max(max_identity,float(identity.max()))
                max_num_defect=max(max_num_defect,float(abs(e).max()));max_depth=max(max_depth,float(z['solver_split_depth'].max()))
                max_ratio=max(max_ratio,float(z['solver_residual_ratio'].max()));norms.append(np.linalg.norm(z['features'],axis=-1).ravel())
        self.summary['rollout'] = dict(trajectories=totals, transitions=totals*320,
            max_ledger_identity_error=max_identity,max_numerical_defect=max_num_defect,
            max_split_depth=max_depth,max_residual_ratio=max_ratio,
            feature_norm_quantiles=np.quantile(np.concatenate(norms),[.5,.95,1.]).tolist() if norms else [])

    def strict_rollout(self):
        """Require the formal checkpoint itself to pass without batch shrinking."""
        prefix='strict_rollout_4250'
        m=self.json(prefix+'/manifest.json')
        checkpoint=self.load('continuation/last.pt')
        model=self.energy_model('continuation/last.pt')
        initial=self.json(prefix+'/initial_histories.json')
        self.check('strict_rollout_complete', m['status']=='completed' and m['ready_numerically'] and
                   m['checkpoint_step']==4250 and m['expected_episodes']==m['completed_episodes']==384 and
                   m['rollout_steps']==320 and m['autonomous_transitions']==122880 and m['batch_size']==32)
        self.check('strict_rollout_checkpoint_binding', m['checkpoint_sha256']==sha256(self.results/'continuation/last.pt') and
                   m['config']==checkpoint['train_config'] and m['data_sha256']==checkpoint['dataset_sha256'] and
                   m['runtime']==checkpoint['runtime'])
        self.check('strict_rollout_source_binding', m['source_hashes']==checkpoint['source_hashes'] and
                   m['script_sha256']==sha256(self.source_root/'scripts/audit_strict_robust_rollouts.py'))
        self.check('strict_rollout_execution_contract', m['strict_batch'] and m['model_state_unchanged'] and
                   m['optimizer_updates']==m['batch_retries']==m['samples_skipped']==0 and
                   m['parameter_updates_enabled'] is False and m['physical_readout_fitted'] is False and
                   m['test_samples_read'] is False and m['future_pixels_read'] is False)
        self.check('strict_rollout_finite_manifest', finite_tree(m))
        ids=[x['episode'] for x in initial]
        self.check('strict_rollout_unique_initials', len(ids)==len(set(ids))==384)
        files=sorted((self.results/prefix).glob('batch_*.npz'))
        self.check('strict_rollout_files_present', [p.name for p in files]==[f'batch_{i:04d}.npz' for i in range(0,384,32)])
        actual_ids=[];max_identity=0.;max_sequence_error=0.;max_defect=0.;norms=[]
        for path in files:
            with np.load(path,allow_pickle=False) as z:
                tag='strict_'+path.name
                self.diagnostic_shapes(z,tag)
                self.check(tag+'_finite_and_complete', all(np.isfinite(z[k]).all() for k in z.files) and
                           z['q'].shape==z['p'].shape==(32,321,2) and z['latent_energy'].shape==(32,321) and
                           z['context'].shape==(32,4) and z['episodes'].shape==(32,))
                h=self.recomputed_energy(model,z['q'],z['p'],z['latent_energy'],tag)
                d=z['dissipation'];w=z['work'];e=z['numerical_balance_defect']
                actual=np.diff(h,axis=1)-w+d
                scale=1+abs(h[:,:-1])+abs(h[:,1:])+abs(d)+abs(w)
                identity=abs(actual-e)
                sequence=h[:,-1]-h[:,0]-w.sum(1)+d.sum(1)
                sequence_error=abs(sequence-e.sum(1))
                sequence_scale=1+abs(h[:,-1])+abs(h[:,0])+abs(w).sum(1)+abs(d).sum(1)
                self.check(tag+'_ledger', np.all(identity<=1e-10*scale) and
                           np.all(abs(actual-z['energy_residual'])<=1e-10*scale) and
                           np.all(abs(e-z['residual_work']-z['chain_defect'])<=1e-10*scale) and
                           np.all(sequence_error<=1e-10*sequence_scale))
                self.check(tag+'_acceptance', np.all(z['solver_residual_ratio']<=1+1e-10) and
                           np.all(z['solver_split_depth']<=4) and np.all(d>=-1e-10) and np.all(abs(w)<=1e-12))
                actual_ids.extend(z['episodes'].tolist())
                max_identity=max(max_identity,float(identity.max()))
                max_sequence_error=max(max_sequence_error,float(sequence_error.max()))
                max_defect=max(max_defect,float(abs(e).max()))
                norms.append(np.linalg.norm(np.concatenate((z['q'],z['p']),axis=-1),axis=-1).ravel())
        self.check('strict_rollout_saved_order_and_coverage', actual_ids==ids and len(actual_ids)==384)
        self.summary['strict_rollout_4250']=dict(trajectories=len(actual_ids), transitions=len(actual_ids)*320,
            max_frame_ledger_identity_error=max_identity,max_sequence_ledger_identity_error=max_sequence_error,
            max_numerical_defect=max_defect,elapsed_seconds=m['elapsed_seconds'],
            feature_norm_quantiles=np.quantile(np.concatenate(norms),[.5,.95,1.]).tolist() if norms else [],
            diagnostic_maxima=m['diagnostic_maxima'],solver_quantiles=m['solver_quantiles'])

    def run(self):
        for name in ('sources','profile','migration','continuation','stress','rollout_arrays','strict_rollout'):
            self.section(name,getattr(self,name))
        numerical_pass=all(c['status']=='pass' for c in self.checks if c.get('category')!='representation')
        gates=[c for c in self.checks if c.get('category')=='representation']
        representation_pass=({c['name'] for c in gates}=={'continuation_final_representation_gate','stress_final_representation_gate'} and
                             all(c['status']=='pass' for c in gates))
        return dict(numerical_evidence_pass=numerical_pass,representation_gates_pass=representation_pass,
            training_readiness_pass=numerical_pass and representation_pass,checks=self.checks,summary=self.summary,
            required_human_completion_review=[
                'Both final representation gates must pass. Gate success means configured early-time representations retain variance and motion; it does not mean prediction scores improved or physical damping was learned.',
                'Review validation-score trends, planning-replay errors and resource cost; local success is not a convergence or physical-accuracy guarantee.',
                'The 320-step arrays here belong to stress-updated weights. Formal continuation must use continuation/last.pt, never stress_weights.pt.',
                'The separate strict_rollout_4250 evidence uses normal checkpoint 4250 without batch shrinking; stress-model rollout still has failure-isolation semantics.',
                'Inspect authoritative terminal process state and source snapshot. Do not infer job completion only from a manifest.',
                'Provide an executable normal-training resume command/config, final source hashes and qualified runtime estimate.'
            ])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--source-root',type=Path,default=PROJECT)
    p.add_argument('--original-checkpoint',type=Path,default=PROJECT/'received/passive_review_20261104/seed0_47541/last.pt')
    p.add_argument('--hash-manifest',type=Path,default=PROJECT/'studies/robust_solver_20261104/code_hashes.json')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    report=Auditor(args.results,args.source_root,args.original_checkpoint,args.hash_manifest).run()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'numerical_evidence_pass':report['numerical_evidence_pass'],
        'representation_gates_pass':report['representation_gates_pass'],'training_readiness_pass':report['training_readiness_pass'],
        'checks':len(report['checks']),'unpassed':[x for x in report['checks'] if x['status']!='pass']},ensure_ascii=False))
    return 0 if report['training_readiness_pass'] else 1

if __name__=='__main__':
    raise SystemExit(main())
