#!/usr/bin/env python3
"""Prepare paired A/single/multi data; default is a read-only plan."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dpbench.passive_family import generate_undamped, plan, validate_config
from dpbench.damping_stage_data import generate, validate_pair
from dpbench.comparison_protocol import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[1]/"configs/passive_family_smooth_v1.json")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args(); config=validate_config(json.loads(args.config.read_text()))
    paths={"a":args.output/"a/undamped.h5","single":args.output/"single/damping.h5","multi":args.output/"multi/seen.h5"}
    if not args.execute:
        print(json.dumps({"training_started":False,"dataset_generation_started":False,
                          "plan":plan(config),"paths":{k:str(v) for k,v in paths.items()}},indent=2));return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Generate into a new directory; existing datasets are accepted directly by run_comparison.py")
    args.output.mkdir(parents=True,exist_ok=True)
    generate_undamped(config,args.output/"a")
    generate(config,args.output/"single",damping=.2,multi=False)
    generate(config,args.output/"multi",damping=.2,multi=True)
    pairs={name:validate_pair(paths["a"],paths[name],multi=name=="multi") for name in ("single","multi")}
    record={"training_started":False,"data":{name:{"path":str(path.resolve()),"sha256":digest(path)} for name,path in paths.items()},"pairs":pairs}
    (args.output/"comparison_data.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))

if __name__=="__main__": main()
