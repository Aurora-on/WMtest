import sys, torch
from pathlib import Path
checkpoint = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
assert checkpoint["step"] == checkpoint["sampler_next_step"] == 4000
assert checkpoint["validation_status"] == "completed"
assert checkpoint["optimizer"]["state"]
assert checkpoint["train_config"]["training"]["steps"] == 20000
print("Resume checkpoint completed step4000 with optimizer and validation state", flush=True)
