#!/usr/bin/env python3
"""Read-only verification of curated source/result and publication file hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def verify():
    original = json.loads((ROOT / 'docs/snapshot_inventory.json').read_text())
    added = json.loads((ROOT / 'docs/repository_files.json').read_text())
    errors = []
    entries = original['copied'] + added['files']
    seen = set()
    for item in entries:
        rel = item['path']
        path = (ROOT / rel).resolve()
        if rel in seen or not path.is_relative_to(ROOT):
            errors.append({'path': rel, 'error': 'duplicate_or_invalid_path'})
            continue
        seen.add(rel)
        if not path.is_file():
            errors.append({'path': rel, 'error': 'missing'})
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            errors.append({'path': rel, 'error': 'hash_mismatch'})
    report = {'passed': not errors, 'checked_files': len(entries), 'errors': errors,
              'scope': 'Curated snapshot bytes only, not model accuracy or full historical run completeness'}
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1

if __name__ == '__main__':
    raise SystemExit(verify())
