#!/usr/bin/env python3
"""Copy a reviewable research snapshot; never move or modify the source workspace."""
import argparse, hashlib, json, shutil
from pathlib import Path
from datetime import datetime, timezone

TEXT = {'.py','.sh','.slurm','.json','.md','.txt','.csv','.yaml','.yml','.toml','.ini','.cfg','.xml','.patch','.js','.html','.lock'}
MEDIA = {'.png','.gif','.svg','.pdf','.jpg','.jpeg','.webp'}
SPECIAL = {'LICENSE','NOTICE','COPYING','SHA256SUMS','REPORT_SHA256SUMS','uv-nosync'}
SKIP = {'.git','.venv','__pycache__','.pytest_cache','node_modules','.DS_Store'}

def decision(rel, size):
    parts = rel.parts
    if any(x in SKIP for x in parts) or rel.name.startswith('.env'): return 'cache_or_environment'
    if rel.name == '.gitignore': return 'replaced_by_repository_policy'
    project = parts[1]; inner = parts[2:]
    if not inner: return 'empty'
    if project == 'damped_pendulum':
        if inner[0] in {'data','runs','received','tmp'}: return 'runtime_dataset_or_checkpoints'
        if 'ours_complete' in inner or 'ours_partial' in inner: return 'duplicate_result_export'
        if inner[0] == 'references': return 'reference_material_not_project_output'
    else:
        if inner[0] == 'setup': return 'machine_specific_install_cache'
        if inner[0] == 'upstream' and (len(inner)<2 or inner[1] != 'LeHami'): return 'duplicate_upstream_tree'
        if 'paper_figures' in inner: return 'upstream_results_not_our_run'
    if rel.suffix == '.npz':
        allowed = 'reports/method_comparison_20261109/data/runs/' in rel.as_posix()
        allowed |= any(x in inner for x in ('nonlinear_readout_visuals_20261105','dimension_launch_20261105T113319Z'))
        if not allowed: return 'raw_or_redundant_array'
    elif rel.suffix == '.log':
        if size > 100_000: return 'verbose_runtime_log'
    elif rel.suffix not in TEXT | MEDIA and rel.name not in SPECIAL:
        return 'weights_data_archive_or_nonessential_binary'
    if size > 15_000_000: return 'large_artifact'
    return None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--destination',required=True,type=Path)
    a=ap.parse_args();src=a.source.resolve();dst=a.destination.resolve()
    if src==dst or not (src/'experiments/damped_pendulum').is_dir(): raise ValueError('Distinct destination and WM source required')
    copied=[];excluded=[]
    for project in ('damped_pendulum','lehami_official_reproduction'):
        for p in sorted((src/'experiments'/project).rglob('*')):
            if not p.is_file(): continue
            rel=p.relative_to(src);size=p.stat().st_size
            if p.is_symlink(): reason='symlink'
            else: reason=decision(rel,size)
            if reason: excluded.append({'path':str(rel),'bytes':size,'reason':reason});continue
            out=dst/rel;out.parent.mkdir(parents=True,exist_ok=True)
            sha=hashlib.sha256(p.read_bytes()).hexdigest()
            if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest()!=sha: raise FileExistsError(out)
            if not out.exists():shutil.copy2(p,out)
            if hashlib.sha256(out.read_bytes()).hexdigest()!=sha: raise RuntimeError('Copy integrity error')
            copied.append({'path':str(rel),'bytes':size,'sha256':sha})
    (dst/'docs').mkdir(exist_ok=True)
    report={'schema_version':1,'captured_at_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Damped-pendulum experiments and related official LeHami reproduction; other WM projects excluded',
            'copied':copied,'excluded':excluded,'source_files_modified':False,
            'note':'Historical completion manifests may refer to intentionally omitted checkpoints, readouts or datasets; this is a curated source/results snapshot, not an entire frozen training bundle.'}
    (dst/'docs/snapshot_inventory.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'files':len(copied),'bytes':sum(x['bytes'] for x in copied),'excluded':len(excluded)}))

if __name__=='__main__':main()
