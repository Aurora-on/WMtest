"""Frozen historical reference uses the shared t=2 anchor without refitting."""
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dpbench.evaluate import Readout, decode_state, pixels_tensor
from dpbench.evaluate_holdout import tensor_digest
from dpbench.models import build_model
from dpbench.passive_family import sha256
from scripts.evaluate_historical_lewm_reference import historical_input_batch, predict_frozen, bound_runs


def test_input_boundary_is_last_five_of_initial_twenty_one(tmp_path):
    filename=tmp_path/'data.h5'
    pixels=np.broadcast_to(np.arange(50,dtype=np.uint8)[:,None,None,None],(50,64,64,3)).copy()
    with h5py.File(filename,'w') as f:
        f['ep_offset']=[0,25];f['ep_len']=[25,25];f['pixels']=pixels
        f['action']=np.zeros((50,1),dtype=np.float32);f['state']=np.zeros((50,2))
    before=historical_input_batch(filename,[0,1],4)
    np.testing.assert_array_equal(before[0][:,:,0,0,0],[[16,17,18,19,20],[41,42,43,44,45]])
    assert before[1].shape==(2,4,1) and before[2].shape==(2,4,1)
    with h5py.File(filename,'r+') as f:
        f['pixels'][21:25]=173;f['pixels'][46:50]=173;f['state'][:]=999
    after=historical_input_batch(filename,[0,1],4)
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='exceeds'):historical_input_batch(filename,[0],5)
    with h5py.File(filename,'r+') as f:f['action'][20]=1
    with pytest.raises(ValueError,match='passive'):historical_input_batch(filename,[0],4)


def test_prediction_matches_native_frozen_rollout_and_does_not_update():
    torch.set_num_threads(2);torch.manual_seed(17)
    model=build_model('lewm',device='cpu').eval().requires_grad_(False)
    rng=np.random.default_rng(4)
    inputs=(rng.integers(0,256,(2,5,64,64,3),dtype=np.uint8),np.zeros((2,4,1),np.float32),np.zeros((2,4,1),np.float32))
    probe=Readout('ridge',np.zeros(192),np.ones(192),np.zeros(3),np.ones(3),weights=rng.normal(size=(193,3)))
    digest=tensor_digest(model.state_dict())
    prediction,anchor,valid=predict_frozen(model,{'ridge':probe},inputs,'cpu')
    expected=model.rollout_features(torch.stack([pixels_tensor(x,'cpu') for x in inputs[0]]),
        torch.from_numpy(inputs[2]),torch.from_numpy(inputs[1])).numpy()
    np.testing.assert_array_equal(anchor,expected[:,0])
    np.testing.assert_array_equal(prediction['ridge'],decode_state(probe.predict(expected)))
    assert valid.tolist()==[4,4]
    assert tensor_digest(model.state_dict())==digest and all(p.grad is None for p in model.parameters())


def test_manifest_binding_rejects_changed_historical_readout(tmp_path):
    runs=[]
    for b in (0.,.2):
        for seed in range(3):
            name=f'lewm_b{str(b).replace(".","p")}_seed{seed}'
            files=[]
            for suffix in ('best.pt','evaluation/readouts.pt','evaluation/calibration.json','run.json'):
                relative=f'runs/server_47111/{name}/{suffix}'
                target=tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(name)
                files.append({'relative_path':relative,'sha256':sha256(target)})
            runs.append({'run':name,'model':'lewm','seed':seed,'damping':b,'files':files})
    assert len(bound_runs({'runs':runs},tmp_path))==6
    changed=tmp_path/runs[0]['files'][1]['relative_path'];changed.write_text('changed')
    with pytest.raises(ValueError,match='changed'):bound_runs({'runs':runs},tmp_path)
