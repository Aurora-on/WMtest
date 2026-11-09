"""Read-only paired legacy-forward and static-initializer check; no optimizer."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.evaluate import pixels_tensor
from dpbench.generate import render

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
checkpoint = root / 'studies/stall_diagnosis_20261105/frozen_training/best.pt'
torch.set_num_threads(4)
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
spec = importlib.util.spec_from_file_location('before_stall_model', out/'before/dpbench/dissipative_model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# Resolve the unchanged pinned SIGReg dependency in the project, not in the
# source-only backup directory. This changes resource lookup, not model code.
module.__file__ = str(root/'dpbench/dissipative_model.py')
old = module.DissipativeWorldModel(saved['config']).eval()
new = DissipativeWorldModel(saved['config']).eval()
old.load_state_dict(saved['state_dict'])
new.load_state_dict(saved['state_dict'])
a = np.load(root/'studies/stall_diagnosis_20261105/stall_rest_windows.npz')
images = torch.stack([pixels_tensor(x, 'cpu') for x in a['pixels'][:2]])
past_action, past_dt = torch.from_numpy(a['actions'][:2]), torch.from_numpy(a['dt'][:2])
future, future_dt = torch.zeros(2, 4, 1), torch.full((2, 4, 1), .1)
with torch.no_grad():
    x = old.rollout(images, future, past_action, past_dt, future_dt)
    y = new.rollout(images, future, past_action, past_dt, future_dt)
errors = {key: float((x[key]-y[key]).abs().max()) for key in ['q', 'p', 'correction', 'context']}
assert all(value == 0 for value in errors.values()), errors
fixed = DissipativeWorldModel(dict(saved['config'], velocity_initializer='motion_gated')).eval()
fixed.load_state_dict(saved['state_dict'])
static = pixels_tensor(np.stack([render(0.)]*21), 'cpu')[None].repeat(2, 1, 1, 1, 1)
with torch.no_grad():
    legacy_p = new.initialize(static, torch.zeros(2, 20, 1), torch.full((2, 20, 1), .1))[1]
    fixed_p = fixed.initialize(static, torch.zeros(2, 20, 1), torch.full((2, 20, 1), .1))[1]
assert torch.count_nonzero(fixed_p) == 0
assert all(torch.equal(saved['state_dict'][key], value) for key, value in fixed.state_dict().items())
result = dict(checkpoint_step=saved['step'], checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              legacy_vs_backup_forward_max_abs=errors,
              static_legacy_p=legacy_p.tolist(), static_fixed_p=fixed_p.tolist(),
              scope='CPU two captured histories x four steps; backup model against new legacy defaults; '
                    'shared pinned SIGReg and default-compatible core; initializer switch is diagnostic, not retraining',
              optimizer_updates=0, parameters_unchanged=True)
(out/'compatibility_and_static_check.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
