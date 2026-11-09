"""Fetch or verify the minimal, byte-exact pinned upstream source snapshot.

This is a source subset, not a complete clone. No model code is rewritten.
Run `python scripts/bootstrap_upstream.py --verify` for offline verification.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1] / "third_party"


def git_hash(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    count = 0
    for repo in json.loads((ROOT / "sources.json").read_text()):
        for file in repo["files"]:
            path = ROOT / repo["directory"] / file["path"]
            if not path.exists() and not args.verify:
                url = f"https://raw.githubusercontent.com/{repo['repository']}/{repo['commit']}/{file['path']}"
                data = urlopen(url, timeout=60).read()
                if git_hash(data) != file["git_blob_sha1"]:
                    raise RuntimeError(f"Upstream hash mismatch: {url}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            if not path.exists() or git_hash(path.read_bytes()) != file["git_blob_sha1"]:
                raise RuntimeError(f"Missing/modified upstream source: {path}")
            count += 1
    print(f"Verified {count} files against pinned Git blob hashes.")


if __name__ == "__main__":
    main()
