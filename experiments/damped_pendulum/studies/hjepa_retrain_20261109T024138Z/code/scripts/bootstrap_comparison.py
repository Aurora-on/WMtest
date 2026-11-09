#!/usr/bin/env python3
"""Fetch missing pinned official source files, or verify entirely offline."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]/"third_party"


def blob_hash(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--verify",action="store_true");args=p.parse_args()
    sources=json.loads((ROOT/"sources.json").read_text())+[json.loads((ROOT/"hjepa_source.json").read_text())]
    counts={}
    for source in sources:
        repo=source["repository"].removeprefix("https://github.com/")
        base=(ROOT/source["directory"]).resolve();count=0
        for entry in source["files"]:
            path=(base/entry["path"]).resolve()
            if not path.is_relative_to(base):raise ValueError("Invalid upstream path")
            if not path.exists() and not args.verify:
                url=f"https://raw.githubusercontent.com/{repo}/{source['commit']}/{entry['path']}"
                data=urlopen(url,timeout=30).read()
                if blob_hash(data)!=entry["git_blob_sha1"]:raise RuntimeError(f"Downloaded source differs: {entry['path']}")
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            if not path.is_file() or blob_hash(path.read_bytes())!=entry["git_blob_sha1"]:
                raise RuntimeError(f"Missing or modified pinned source: {path}")
            count+=1
        counts[repo]={"commit":source["commit"],"verified_files":count}
    print(json.dumps({"verified":True,"repositories":counts},indent=2))

if __name__=="__main__":main()
