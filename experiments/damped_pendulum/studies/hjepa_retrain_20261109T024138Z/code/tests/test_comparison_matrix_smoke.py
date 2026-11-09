"""Run every model/condition path on real tiny generated data, not benchmark data."""
import json
from pathlib import Path
import pytest
import torch
from dpbench import passive_family, damping_stage_data
from dpbench.comparison_protocol import make_config, digest
from dpbench.train_comparison import execute_run


@pytest.fixture(scope="module")
def paired_matrix_data(tmp_path_factory):
    root=tmp_path_factory.mktemp("comparison_matrix")
    project=Path(__file__).resolve().parents[1]
    config=json.loads((project/"configs/passive_family_smooth_v1.json").read_text())
    config.update(groups={"train":8,"val":4,"test":4},test_frames=81)
    passive_family.generate_undamped(config,root/"a")
    damping_stage_data.generate(config,root/"single",multi=False)
    damping_stage_data.generate(config,root/"multi",multi=True)
    paths={"a":root/"a/undamped.h5","single":root/"single/damping.h5","multi":root/"multi/seen.h5"}
    for stage in ("single","multi"):
        damping_stage_data.validate_pair(paths["a"],paths[stage],multi=stage=="multi")
    return paths


@pytest.mark.parametrize("method",["lehami","lewm","hjepa"])
def test_every_method_transfers_to_both_damping_conditions(method,paired_matrix_data,tmp_path):
    paths=paired_matrix_data; parent=None
    for stage in ("a","single","multi"):
        config=make_config(method,stage,steps=1,tiny=True)
        config["data"]["expected_groups"]={"train":8,"val":4}
        config["training"].update(batch_size=2,threads=1,val_every=1,checkpoint_every=1,
                                   val_windows=18,val_horizons=[4],warmup_steps=1,
                                   curriculum=[{"until_step":1,"horizon":4}])
        checkpoint=execute_run(paths[stage],tmp_path/stage,config,device="cpu",init_from=parent,
                               expected_parent_step=1,
                               expected_parent_data_sha256=digest(paths["a"]) if parent else None)
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
        assert saved["step"]==1 and saved["validation_status"]=="completed"
        assert saved["fixture_only"] is True
        assert all(torch.isfinite(v).all() for v in saved["state_dict"].values())
        if stage=="a":parent=checkpoint
        else:
            assert saved["parent_transfer"]["sha256"]==digest(parent)
            assert saved["parent_transfer"]["independently_verified_data_sha256"]==digest(paths["a"])
            assert {int(s["step"]) for s in saved["optimizer"]["state"].values()}=={1}
