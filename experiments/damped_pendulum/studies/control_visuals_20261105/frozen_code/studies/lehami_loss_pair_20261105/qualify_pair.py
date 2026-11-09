"""CPU qualification of the actual pair, synthetic RGB, no optimizer or updates."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.train_dissipative import configure_trainable_modules, read_config, source_hashes


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    def forbidden(*args, **kwargs):
        raise AssertionError("Qualification must never construct an optimizer")
    torch.optim.AdamW = forbidden
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(6205)
    pixels = torch.rand(2, 55, 3, 64, 64, generator=generator)
    batch = dict(pixels=pixels, action=torch.zeros(2, 54, 1),
                 dt_frame=torch.full((2, 54, 1), .1))
    report = dict(scope="Actual full-size configured models; synthetic RGB; CPU; no physical validation",
                  optimizer_created=False, optimizer_updates=0, training_started=False,
                  cuda_qualified=False, source_sha256=source_hashes(), arms={})
    for arm in ("control", "lehami"):
        config = read_config(ROOT / f"configs/dissipative_loss_pair_{arm}_v1.json")
        torch.manual_seed(config["training"]["seed"])
        model = DissipativeWorldModel(config["model"])
        trainability = configure_trainable_modules(model, config["training"])
        initial = state_hash(model)
        rows = []
        for horizon in (4, 8, 32):
            model.zero_grad(set_to_none=True)
            output = model.loss(batch, horizon)
            assert all(value.numel() == 1 and torch.isfinite(value).all() for value in output.values())
            output["loss"].backward()
            absent = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
            # Raw V's additive output bias is a gauge: forces and discrete
            # energy differences are independent of it. It has no gradient.
            assert absent == ["core.potential_net.0.4.bias"]
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            assert state_hash(model) == initial
            rows.append(dict(horizon=horizon, loss=float(output["loss"].detach()),
                             phase_prediction_mse=float(output["phase_prediction_mse"]),
                             finite_gradients=True, parameters_without_gradient=absent,
                             missing_gradient_reason="Additive raw potential constant does not affect dynamics",
                             frozen_parameters_have_no_gradient=True, parameters_unchanged=True))
        report["arms"][arm] = dict(initial_state_sha256=initial, trainability=trainability, checks=rows)
    assert report["arms"]["control"]["initial_state_sha256"] == report["arms"]["lehami"]["initial_state_sha256"]
    report["identical_initial_state"] = True
    # Different losses must not change the common diagnostic before any update.
    for control, lehami in zip(report["arms"]["control"]["checks"], report["arms"]["lehami"]["checks"]):
        assert control["phase_prediction_mse"] == lehami["phase_prediction_mse"]
    report["identical_initial_common_scores"] = True
    output = Path(__file__).with_name("full_config_qualification.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(status="passed", output=str(output), arms=2, horizons=[4, 8, 32],
                          optimizer_updates=0, identical_initial_state=True)))


if __name__ == "__main__":
    main()
