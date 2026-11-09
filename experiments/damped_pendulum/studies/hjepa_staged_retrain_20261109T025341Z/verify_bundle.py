#!/usr/bin/env python3
"""Verify immutable uploaded package files before any preflight or launch."""
import hashlib
import json
from pathlib import Path

def verify():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'package_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f'Missing or out-of-package file: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Package hash mismatch: {name}')
    return dict(status='verified', files=len(manifest['files']), formal_training_started=False)

if __name__ == '__main__':
    print(json.dumps(verify()))
