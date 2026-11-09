"""Official-core visual baselines under an explicitly adapted history contract."""
from __future__ import annotations
import copy
import torch

KINDS = ("lehami", "lewm", "hjepa")


def default_comparison_model_config(kind, tiny=False):
    if kind not in KINDS:
        raise ValueError(f"Unknown comparison model: {kind}")
    if kind == "hjepa":
        from .comparison_hjepa import default_hjepa_config
        return default_hjepa_config(tiny=tiny)
    from .comparison_visual_baselines import default_model_config
    return default_model_config(kind, tiny=tiny)


def build_comparison_model(config, device="cpu"):
    config = copy.deepcopy(config)
    kind = config.get("kind")
    dev = torch.device(device)
    if dev.type not in {"cpu", "cuda"}:
        raise ValueError("Comparison qualification supports CPU/CUDA explicitly")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; no silent CPU fallback")
    if kind == "hjepa":
        from .comparison_hjepa import HJEPAComparison
        model = HJEPAComparison(config)
    elif kind in ("lehami", "lewm"):
        from .comparison_visual_baselines import LeHamiComparison, LeWMComparison
        model = (LeHamiComparison if kind == "lehami" else LeWMComparison)(config)
    else:
        raise ValueError(f"Unknown comparison model: {kind}")
    return model.to(dev)
