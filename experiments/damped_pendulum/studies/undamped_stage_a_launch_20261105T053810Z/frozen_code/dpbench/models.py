"""Pinned official LeHami and LeWM pendulum models with a small local runner.

The original builders and losses are executed verbatim except the unused
stable_pretraining import (only required by the unselected ViT factory). This
conv-only adapter is NOT the authors' Lightning training entry point and the
LeWM arm is the LeHami authors' multistep pendulum adaptation of LeWM.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party"
ANCHOR = 4
HISTORY = 5


def verify_sources():
    manifest = json.loads((UPSTREAM / "sources.json").read_text())
    for repo in manifest:
        for entry in repo["files"]:
            path = UPSTREAM / repo["directory"] / entry["path"]
            data = path.read_bytes()
            blob = b"blob " + str(len(data)).encode() + b"\0" + data
            if hashlib.sha1(blob).hexdigest() != entry["git_blob_sha1"]:
                raise RuntimeError(f"Upstream source hash mismatch: {path}")
    return manifest


def _builders():
    verify_sources()
    module_name = "lehami.builders"
    if module_name in sys.modules:
        return sys.modules[module_name]
    package_dir = UPSTREAM / "LeHami/lehami/models"
    sys.path.insert(0, str(package_dir))
    sys.path.insert(0, str(UPSTREAM / "le-wm"))
    import lehami  # noqa: F401; loads the byte-exact upstream package
    path = package_dir / "lehami/builders.py"
    code = path.read_text()
    unused = "import stable_pretraining as spt"
    if code.count(unused) != 1:
        raise RuntimeError("Unexpected upstream import layout; review the adapter.")
    code = code.replace(unused, "spt = None  # explicit conv-only adapter")
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = "lehami"
    sys.modules[module_name] = module
    exec(compile(code, str(path), "exec"), module.__dict__)
    def disallow_vit(*args, **kwargs):
        raise RuntimeError("This adapter only supports the upstream convolutional builders.")
    module._vit = disallow_vit
    return module


def validate_device(device):
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select a device explicitly.")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; select a device explicitly.")
    if dev.type not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unsupported device: {device}")
    return dev


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elif torch.device(device).type == "mps":
        torch.mps.synchronize()


def default_config(kind, dt=0.1, unroll=8):
    if kind not in ("lehami", "lewm"):
        raise ValueError("kind must be lehami or lewm")
    if dt <= 0 or unroll < 3:
        raise ValueError("dt must be positive and unroll >= 3 for official SG targets")
    base = dict(img_size=64, embed_dim=192, data=dict(action_dim=1),
                history_size=1 if kind == "lehami" else 3,
                loss=dict(sigreg=dict(weight=0.25, kwargs=dict(knots=17, num_proj=1024))),
                benchmark=dict(kind=kind, physical_dt=dt, unroll=unroll,
                               anchor=ANCHOR, history=HISTORY, training_window=unroll + 7,
                               diagnostic_device="cpu"))
    if kind == "lehami":
        base["model"] = dict(name="lehami_sg_sep_sreg_q_ms_conv", latent_dim=32,
                             enc_window=1, enc_variant="full", fd_dt=1., dt_ref=dt,
                             momentum_weight=1., sigreg_mode="q",
                             ham_flow=dict(hidden_dim=1024, n_layers=3, dt=1.,
                                           phase_space=True, separable=True, learnable_scale=True))
    else:
        base["model"] = dict(name="lewm_ms_conv", latent_dim=64,
                             enc_window=3, enc_variant="free", enc_lag=0,
                             predictor=dict(depth=6, heads=16, mlp_dim=2048,
                                            dim_head=64, dropout=0.1, emb_dropout=0.))
    return base


class Baseline(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
        self.cfg = OmegaConf.create(self.config)
        self.kind = self.cfg.benchmark.kind
        self.unroll = int(self.cfg.benchmark.unroll)
        self.window = self.unroll + 7
        self.dt = float(self.cfg.benchmark.physical_dt)
        self.builders = _builders()
        if self.cfg.benchmark.get("diagnostic_device") != "cpu":
            raise ValueError("This adapter explicitly computes monitoring SVD on CPU")
        # The upstream SVD is monitoring-only under no_grad, not a loss term.
        # MPS lacks this operator in torch 2.8. This explicit dispatch changes
        # diagnostics placement only; no global backend fallback is enabled.
        if not hasattr(self.builders, "_dpbench_original_rank"):
            self.builders._dpbench_original_rank = self.builders._participation_rank
            original_rank = self.builders._dpbench_original_rank
            def diagnostic_rank_cpu(x):
                with torch.no_grad():
                    return original_rank(x.detach().cpu()).to(x.device)
            self.builders._participation_rank = diagnostic_rank_cpu
        self.model, self.sigreg, self.forward_fn = self.builders.REGISTRY[self.cfg.model.name](self.cfg)
        self.last_diagnostics = {}

    def log_dict(self, values, **kwargs):
        self.last_diagnostics = values

    @staticmethod
    def normalize(pixels):
        # Input contract is RGB float32 in [0,1]; common fixed scaling, no test statistics.
        return pixels.float().mul(2).sub(1)

    def loss(self, batch, stage="train"):
        if batch["pixels"].shape[1] != self.window:
            raise ValueError(f"Expected exactly {self.window} common raw frames")
        if "action" not in batch or "dt_frame" not in batch:
            raise ValueError("Explicit action and dt_frame fields are mandatory")
        sl = slice(1, None) if self.kind == "lehami" else slice(0, self.unroll + 5)
        selected = {k: batch[k][:, sl] for k in ("pixels", "action", "dt_frame")}
        selected["pixels"] = self.normalize(selected["pixels"])
        return self.forward_fn(self, selected, stage)

    def _encode(self, history):
        enc = self.builders._encode_sequence(self.model.encoder, self.normalize(history))
        b, t = enc.shape[:2]
        return self.model.projector(enc.reshape(b * t, -1)).reshape(b, t, -1)

    @torch.no_grad()
    def rollout_features(self, history, actions, past_actions=None):
        """Return anchor and autonomous features, [B, K+1, F].

        history is [B,5,3,64,64]; actions[:,k] acts from anchor+k to +1.
        past_actions is [B,4,1], covering history frame intervals 0..3.
        LeWM features include its complete three-latent recurrent context.
        No future images, physical state, or re-encoding enters this method.
        """
        if self.training:
            raise RuntimeError("Call eval() before extracting rollout features")
        if history.ndim != 5 or history.shape[1] != HISTORY:
            raise ValueError("history must contain exactly five RGB frames")
        b, steps = actions.shape[:2]
        if actions.shape != (b, steps, 1) or b != history.shape[0]:
            raise ValueError("actions must have shape [B, K, 1]")
        if past_actions is None:
            if torch.count_nonzero(actions).item():
                raise ValueError("Controlled rollouts require past_actions, not fabricated zero history")
            past_actions = actions.new_zeros(b, HISTORY - 1, 1)
        if past_actions.shape != (b, HISTORY - 1, 1):
            raise ValueError("past_actions must have shape [B, 4, 1]")
        z = self._encode(history[:, 1:] if self.kind == "lehami" else history)
        if self.kind == "lehami":
            from lehami.derivative import p_causal_last
            qp = torch.cat([z[:, -1], p_causal_last(z, 1.)], -1)
            features = [qp]
            for k in range(steps):
                act = self.model.action_encoder(actions[:, k:k + 1])
                qp = self.model.predictor(qp[:, None], act).squeeze(1)
                features.append(qp)
        else:
            context = z
            all_actions = torch.cat([past_actions[:, -2:], actions], 1)
            features = [context.flatten(1)]
            for k in range(steps):
                act = self.model.action_encoder(all_actions[:, k:k + 3])
                pred = self.model.predict(context, act)[:, -1]
                context = torch.cat([context[:, 1:], pred[:, None]], 1)
                features.append(context.flatten(1))
        return torch.stack(features, 1)

    def provenance(self):
        return dict(upstream=verify_sources(),
                    adaptation="Official conv builders and losses; unused stable_pretraining import skipped; monitoring-only SVD explicitly on CPU; local FP32 trainer; common raw anchor; LeWM free W3 lag0.",
                    label="LeHami original predictor" if self.kind == "lehami" else "LeWM / LeHami authors' conv multistep pendulum adaptation, free W3",
                    parameters=sum(p.numel() for p in self.parameters()),
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    config=self.config)


def build_model(kind, dt=0.1, device="cpu", unroll=8, config=None):
    return Baseline(config or default_config(kind, dt, unroll)).to(validate_device(device))


def load_checkpoint(path, device="cpu"):
    # Only load checkpoints generated locally by this runner; never untrusted pickle.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["kind"] == "dissipative":
        from dpbench.dissipative_model import DissipativeWorldModel
        if torch.device(device).type not in {"cpu", "cuda"}:
            raise ValueError("Dissipative FP64 dynamics support CPU/CUDA, not MPS")
        model = DissipativeWorldModel(checkpoint["config"]).to(validate_device(device))
    else:
        model = build_model(checkpoint["kind"], device=device, config=checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
