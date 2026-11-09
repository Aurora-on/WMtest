"""Diagnostic wrapper tests; no optimizer updates or production model changes."""
from pathlib import Path

import pytest
import torch

from dpbench.dissipative_core import DissipativeCore, SolverError
from dpbench.dissipative_model import DissipativeWorldModel
from scripts.replay_passive_failure import capture_failed_substeps, ReplayBudgetReached, main


def test_capture_reconstructs_identical_original_failure_and_restores_hook(tmp_path):
    torch.manual_seed(21)
    core = DissipativeCore(hidden_dim=8,newton_iterations=1,residual_atol=0.,residual_rtol=0.,context_dim=2)
    q = torch.tensor([[1.,2.]],dtype=torch.float64)
    p = torch.tensor([[4.,-3.]],dtype=torch.float64)
    action = torch.zeros(1,1,dtype=torch.float64)
    h = torch.full((1,1),.05,dtype=torch.float64)
    context = torch.tensor([[.1,.2]],dtype=torch.float64)
    old_step = DissipativeCore._step
    with pytest.raises(SolverError) as original:
        core._step(q,p,action,h,True,context)
    with capture_failed_substeps(tmp_path,2750,60):
        with pytest.raises(SolverError) as captured:
            core._step(q,p,action,h,True,context)
    assert str(captured.value) == str(original.value)
    assert DissipativeCore._step is old_step
    blob = torch.load(tmp_path/'failure_capture.pt',weights_only=True)
    restored = DissipativeCore(**blob['core_config'])
    restored.load_state_dict(blob['core_state_dict'])
    with pytest.raises(SolverError) as replayed:
        restored._step(**blob['input'],track_gradient=blob['track_gradient'])
    assert str(replayed.value) == str(original.value)
    for k,v in core.state_dict().items():
        assert torch.equal(v,blob['core_state_dict'][k])
    assert all(x.grad is None for x in core.parameters())


def test_budget_stops_before_next_training_loss_but_not_validation(tmp_path,monkeypatch):
    calls=[]
    def fake_loss(self,*args,**kwargs):
        calls.append(self.training)
        return {'loss':torch.tensor(1.)}
    monkeypatch.setattr(DissipativeWorldModel,'loss',fake_loss)
    model=DissipativeWorldModel({'encoder_width':8,'sigreg_projections':4,'core':{'hidden_dim':8}})
    with capture_failed_substeps(tmp_path,2750,max_updates=2) as state:
        model.loss(None,1)
        model.eval()
        with torch.no_grad():
            model.loss(None,1)
        model.train()
        model.loss(None,1)
        with pytest.raises(ReplayBudgetReached,match='2 optimizer updates'):
            model.loss(None,1)
        assert state['training_loss_calls']==2
        assert state['training_step']==2752
    assert calls == [True,False,True]
    assert DissipativeWorldModel.loss is fake_loss
    assert not (tmp_path/'failure_capture.pt').exists()


def test_default_cli_is_read_only_plan(tmp_path):
    resume=tmp_path/'last.pt'; torch.save({'step':2750},resume)
    output=tmp_path/'not_created'
    config=Path(__file__).resolve().parents[1]/'configs/dissipative_passive_family.json'
    plan=main(['--resume',str(resume),'--data','not_needed.h5','--config',str(config),'--output',str(output)])
    assert plan['mode']=='plan_only'
    assert plan['max_optimizer_updates']==60
    assert plan['latest_allowed_optimizer_step']==2810
    assert not plan['optimizer_updates_enabled']
    assert not output.exists()
