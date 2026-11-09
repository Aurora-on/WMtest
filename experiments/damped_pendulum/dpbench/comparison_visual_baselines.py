"""Pinned official cores for the matched-history visual comparison.

These are explicitly *adapted* baselines, not the papers' native training runs.
LeHami keeps its learned separable Hamiltonian, leapfrog, SG momentum targets,
and joint phase-space SIGReg. Its only added module initializes momentum from
the complete 21-frame history. LeWM uses the original JEPA/ARPredictor modules
with the LeHami authors' official convolutional frontend, 64-dimensional
latents, 21-frame attention, and the original one-step teacher-forced objective.
There is no learned damping, physical-state input, or condition-label input.

The two extra future frames are available only to training targets. Inference
receives precisely the initial history and actions, then self-feeds. LeWM's
last-two-latent readout features are a diagnostic projection, not its full state.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from dpbench import models as pinned


PINNED_MANIFEST_SHA256 = "562de60456c15396b4536985a066ba261158ea3dbaf4acf5d2ffb046aaf5e788"
HISTORY = 21
TARGET_TAIL = 2


def default_model_config(kind: str, tiny: bool = False) -> dict[str, Any]:
    """JSON-only model config; ``tiny`` is an explicit CPU fixture, not a baseline."""
    if kind not in {"lehami", "lewm"}:
        raise ValueError("kind must be lehami or lewm")
    cfg = {
        "kind": kind, "history": HISTORY, "dt": .1,
        "target_tail_frames": TARGET_TAIL, "encoder": "official_conv",
        "embed_dim": 16 if tiny else 192,
        "latent_dim": (2 if tiny else 32) if kind == "lehami" else (4 if tiny else 64),
        "projector_hidden_dim": 32 if tiny else 2048,
        "sigreg": {"weight": .25 if kind == "lehami" else .09,
                   "knots": 5 if tiny else 17, "num_proj": 8 if tiny else 1024},
        "tiny_fixture": bool(tiny),
    }
    if kind == "lehami":
        cfg.update({
            "objective": "native_multistep_joint_phase", "sigreg_mode": "joint",
            "fd_dt": 1., "momentum_weight": 1.,
            "history_initializer_hidden_dim": 16 if tiny else 128,
            "ham_flow": {"hidden_dim": 16 if tiny else 1024,
                         "n_layers": 2 if tiny else 3, "dt": 1.,
                         "phase_space": True, "separable": True,
                         "learnable_scale": True},
        })
    else:
        cfg.update({
            "objective": "native_one_step_shifted_contexts",
            "readout_feature": "last_two_latents",
            "predictor": {"depth": 1 if tiny else 6, "heads": 2 if tiny else 16,
                          "mlp_dim": 32 if tiny else 2048,
                          "dim_head": 8 if tiny else 64,
                          "dropout": 0. if tiny else .1, "emb_dropout": 0.},
        })
    return cfg


def _merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if key not in out:
            raise ValueError(f"Unknown comparison model setting: {key}")
        out[key] = _merge(out[key], value) if isinstance(out[key], dict) else value
    return out


def _verified_builders():
    manifest = pinned.UPSTREAM / "sources.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != PINNED_MANIFEST_SHA256:
        raise RuntimeError("Pinned upstream source manifest changed; review before loading")
    # Verifies every vendored file against its pinned Git blob, even if modules
    # were imported earlier in this process. No source files are rewritten.
    pinned.verify_sources()
    builders = pinned._builders()
    expected = {
        "jepa": pinned.UPSTREAM / "le-wm/jepa.py",
        "module": pinned.UPSTREAM / "le-wm/module.py",
        "lehami.builders": pinned.UPSTREAM / "LeHami/lehami/models/lehami/builders.py",
        "lehami.derivative": pinned.UPSTREAM / "LeHami/lehami/models/lehami/derivative.py",
        "lehami.ham_flow": pinned.UPSTREAM / "LeHami/lehami/models/lehami/ham_flow.py",
    }
    for name, source in expected.items():
        loaded = importlib.import_module(name)
        if Path(getattr(loaded, "__file__", "")).resolve() != source.resolve():
            raise RuntimeError(f"Upstream module name collision for {name}; refusing unverified code")
    return builders


class _VisualComparison(nn.Module):
    kind: str

    def __init__(self, config: dict | None = None):
        super().__init__()
        overrides = copy.deepcopy(config or {})
        self.config = json.loads(json.dumps(_merge(
            default_model_config(self.kind, bool(overrides.get("tiny_fixture", False))), overrides)))
        cfg = self.config
        if cfg["kind"] != self.kind:
            raise ValueError(f"Expected kind={self.kind}")
        if cfg["history"] != HISTORY or cfg["target_tail_frames"] != TARGET_TAIL:
            raise ValueError("Comparison requires exactly 21 history and 2 target-tail frames")
        if cfg["dt"] != .1:
            raise ValueError("This matched comparison fixes dt=0.1 seconds")
        if cfg["encoder"] != "official_conv":
            raise ValueError("Only the explicitly declared official_conv adaptation is supported")
        self.builders = _verified_builders()
        # These imports resolve to the byte-verified original LeWM source tree
        # installed in sys.path by the existing conv-only upstream loader.
        from jepa import JEPA
        from module import Embedder, MLP, SIGReg
        self._JEPA, self._Embedder, self._MLP = JEPA, Embedder, MLP
        self.sigreg = SIGReg(knots=cfg["sigreg"]["knots"], num_proj=cfg["sigreg"]["num_proj"])
        self.history = HISTORY
        self.dt = cfg["dt"]
        self.latent_dim = int(cfg["latent_dim"])
        self.last_diagnostics = {}

    def _components(self, predictor, action_dim: int, *, prediction_projection: bool):
        cfg = self.config
        mlp = lambda width, output: self._MLP(
            input_dim=width, hidden_dim=cfg["projector_hidden_dim"], output_dim=output,
            norm_fn=nn.BatchNorm1d)
        self.model = self._JEPA(
            encoder=self.builders.ConvEncoder(embed_dim=cfg["embed_dim"]),
            predictor=predictor,
            action_encoder=self._Embedder(input_dim=1, emb_dim=action_dim),
            projector=mlp(cfg["embed_dim"], self.latent_dim),
            pred_proj=mlp(self.latent_dim, self.latent_dim) if prediction_projection else None,
        )

    @staticmethod
    def _pixels(pixels):
        if pixels.ndim != 5 or tuple(pixels.shape[2:]) != (3, 64, 64):
            raise ValueError("pixels must have shape [B,T,3,64,64]")
        if not pixels.is_floating_point() or not torch.isfinite(pixels).all():
            raise ValueError("pixels must be finite floating point RGB in [0,1]")
        if bool(((pixels < 0) | (pixels > 1)).any()):
            raise ValueError("pixels must lie in [0,1]")

    def _encode(self, pixels):
        self._pixels(pixels)
        b, t = pixels.shape[:2]
        normalized = pixels.float().mul(2).sub(1)
        if self.training:
            if b < 2:
                raise ValueError("Training BatchNorm requires batch size >= 2")
            # Official weights/modules are unchanged. Temporal batching is an
            # explicit adaptation: BatchNorm pools only across examples at the
            # same raw time, never across a sample's past/future frames.
            return torch.stack([self.model.projector(self.model.encoder(normalized[:, i]))
                                for i in range(t)], 1)
        enc = self.model.encoder(normalized.reshape(b*t, 3, 64, 64))
        return self.model.projector(enc).reshape(b, t, self.latent_dim)

    def _batch(self, batch, horizon):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not all(k in batch for k in ("pixels", "action", "dt_frame")):
            raise ValueError("pixels, action, and dt_frame are mandatory")
        pixels, actions, dt = (batch[k] for k in ("pixels", "action", "dt_frame"))
        self._pixels(pixels)
        b, t = pixels.shape[:2]
        if t != HISTORY + horizon + TARGET_TAIL:
            raise ValueError("Expected exactly 21 + horizon + 2 frames")
        if actions.shape != (b, t-1, 1) or dt.shape != (b, t-1, 1):
            raise ValueError("action and dt_frame must be [B,T-1,1]")
        if not torch.isfinite(actions).all() or not torch.isfinite(dt).all():
            raise ValueError("Actions and clocks must be finite")
        if not torch.allclose(dt, torch.full_like(dt, self.dt), atol=1e-7, rtol=0):
            raise ValueError("All observation intervals must equal dt=0.1")
        return pixels, actions

    def _history(self, history, past_actions):
        if self.training:
            raise RuntimeError("Call eval() before extracting causal readout features")
        self._pixels(history)
        if history.shape[1] != HISTORY:
            raise ValueError("Inference history must contain exactly 21 RGB frames")
        b = history.shape[0]
        if past_actions is None:
            # This suite's documented optional default is passive zero action.
            past_actions = history.new_zeros((b, HISTORY-1, 1))
        if past_actions.shape != (b, HISTORY-1, 1) or not torch.isfinite(past_actions).all():
            raise ValueError("past_actions must be finite [B,20,1]")
        return past_actions

    def _future(self, history, actions, past_actions):
        if actions.ndim != 3 or actions.shape[0] != history.shape[0] or actions.shape[-1] != 1:
            raise ValueError("actions must have shape [B,K,1]")
        if not torch.isfinite(actions).all():
            raise ValueError("actions must be finite")
        if past_actions is None and bool(torch.count_nonzero(actions)):
            raise ValueError("Nonzero future actions require explicit past_actions")
        return self._history(history, past_actions)

    def _output(self, prediction, sigreg, **extra):
        out = {"loss": prediction + self.config["sigreg"]["weight"] * sigreg,
               "prediction_loss": prediction, "pred_loss": prediction,
               "sigreg_loss": sigreg, **extra}
        self.last_diagnostics = {k: v.detach() for k, v in out.items()}
        return out

    def provenance(self):
        _verified_builders()
        return {
            "kind": self.kind, "label": f"{self.kind}: official-core matched visual adaptation",
            "upstream": pinned.verify_sources(), "source_manifest_sha256": PINNED_MANIFEST_SHA256,
            "config": copy.deepcopy(self.config),
            "history_frames": HISTORY, "physical_labels_used_by_world_model": False,
            "normalization": "fixed RGB [0,1] to [-1,1]; no fitted observation statistics",
            "encoder": "LeHami authors' pinned ConvEncoder; not original LeWM ViT",
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "loader_adaptation": "skip unused stable_pretraining import; prohibit ViT factory",
            "batchnorm_temporal_policy": "train encoder/projector separately per raw frame across B; eval uses frozen running statistics and may batch time",
            "tiny_fixture": self.config["tiny_fixture"],
        }


class LeHamiComparison(_VisualComparison):
    """Conservative official LeHami flow with an explicit full-history initializer."""
    kind = "lehami"

    def __init__(self, config=None):
        super().__init__(config)
        from lehami.ham_flow import HamiltonianPredictor
        cfg = self.config
        if cfg["sigreg_mode"] != "joint" or cfg["objective"] != "native_multistep_joint_phase":
            raise ValueError("LeHami comparison requires joint q,p SIGReg and native multistep loss")
        if cfg["fd_dt"] != 1. or cfg["ham_flow"]["dt"] != 1.:
            raise ValueError("Official LeHami clock is frame-unit h=dt/0.1=1")
        if not cfg["ham_flow"]["phase_space"] or not cfg["ham_flow"]["separable"]:
            raise ValueError("The official conservative separable phase-space flow is required")
        self._components(HamiltonianPredictor(
            embed_dim=2*self.latent_dim, latent_dim=self.latent_dim,
            action_emb_dim=2*self.latent_dim, **cfg["ham_flow"]),
            action_dim=2*self.latent_dim, prediction_projection=False)
        self.history_initializer = nn.Sequential(
            nn.Linear(HISTORY*self.latent_dim + HISTORY-1, cfg["history_initializer_hidden_dim"]),
            nn.Tanh(), nn.Linear(cfg["history_initializer_hidden_dim"], self.latent_dim))
        nn.init.zeros_(self.history_initializer[-1].weight)
        nn.init.zeros_(self.history_initializer[-1].bias)
        self.feature_dim = 2*self.latent_dim

    def _initial(self, z, past_actions):
        from lehami.derivative import p_causal_last
        correction = self.history_initializer(torch.cat([z.flatten(1), past_actions.to(z).flatten(1)], -1))
        p = p_causal_last(z[:, -4:], self.config["fd_dt"]) + correction
        return torch.cat([z[:, -1], p], -1), correction

    def _rollout(self, initial, actions):
        qp = initial
        outputs = [qp]
        for k in range(actions.shape[1]):
            act = self.model.action_encoder(actions[:, k:k+1])
            qp = self.model.predictor(qp[:, None], act).squeeze(1)
            outputs.append(qp)
        return torch.stack(outputs, 1)

    def loss(self, batch, horizon: int, stage: str = "train"):
        from lehami.derivative import p_interp_series
        pixels, actions = self._batch(batch, horizon)
        # Separate calls prevent future frames from entering initialization via
        # train-mode BatchNorm statistics. The same encoder remains differentiable.
        z_history = self._encode(pixels[:, :HISTORY])
        z_future = self._encode(pixels[:, HISTORY:])
        z = torch.cat([z_history, z_future], 1)
        p_ref = p_interp_series(z, self.config["fd_dt"])
        initial, correction = self._initial(z_history, actions[:, :HISTORY-1])
        pred = self._rollout(initial, actions[:, HISTORY-1:HISTORY-1+horizon])[:, 1:]
        position = F.mse_loss(pred[..., :self.latent_dim], z[:, HISTORY:HISTORY+horizon])
        momentum = F.mse_loss(pred[..., self.latent_dim:], p_ref[:, HISTORY:HISTORY+horizon])
        phase = torch.cat([z[:, 2:-2], p_ref[:, 2:-2]], -1)
        reg = self.sigreg(phase.transpose(0, 1))
        return self._output(position + self.config["momentum_weight"]*momentum, reg,
                            position_loss=position, momentum_loss=momentum,
                            initializer_correction_rms=correction.square().mean().sqrt().detach(),
                            target_position_std=z.detach().std(), target_momentum_std=p_ref.detach().std())

    @torch.no_grad()
    def observed_features(self, history, past_actions=None):
        past_actions = self._history(history, past_actions)
        return self._initial(self._encode(history), past_actions)[0]

    @torch.no_grad()
    def rollout_features(self, history, actions, past_actions=None):
        past_actions = self._future(history, actions, past_actions)
        initial = self._initial(self._encode(history), past_actions)[0]
        return self._rollout(initial, actions)

    def provenance(self):
        return {**super().provenance(),
                "native_core": "LeHami HamiltonianPredictor: learned separable K(p/s)+V(q,u), leapfrog; no damping",
                "adaptation": "q32/p32 by default; full21 history momentum correction added to official causal-last4 SG, zero-initialized output; shared architecture across A/B",
                "objective": "official multistep q and SG-p MSE, differentiable targets, joint q,p SIGReg weight .25; frame-unit momentum",
                "batchnorm_boundary": "per-raw-time encoder/projector BatchNorm; no future statistics in initializer",
                "readout_features": "complete current phase [q,p], including inferred initial correction"}


class LeWMComparison(_VisualComparison):
    """Original LeWM predictor/objective with declared conv/history adaptation."""
    kind = "lewm"

    def __init__(self, config=None):
        super().__init__(config)
        from module import ARPredictor
        if self.config["objective"] != "native_one_step_shifted_contexts":
            raise ValueError("LeWM comparison preserves native one-step teacher-forced prediction")
        if self.config["readout_feature"] != "last_two_latents":
            raise ValueError("LeWM readout feature must be the declared last-two-latent projection")
        self._components(ARPredictor(
            num_frames=HISTORY, input_dim=self.latent_dim, hidden_dim=self.latent_dim,
            output_dim=self.latent_dim, **self.config["predictor"]),
            action_dim=self.latent_dim, prediction_projection=True)
        self.feature_dim = 2*self.latent_dim

    def _teacher_predictions(self, z, actions, horizon):
        contexts = torch.stack([z[:, j:j+HISTORY] for j in range(horizon)], 1)
        controls = torch.stack([actions[:, j:j+HISTORY] for j in range(horizon)], 1)
        b = z.shape[0]
        context = contexts.reshape(b*horizon, HISTORY, self.latent_dim)
        act = self.model.action_encoder(controls.reshape(b*horizon, HISTORY, 1))
        if not self.training:
            return self.model.predict(context, act).reshape_as(contexts)
        raw = self.model.predictor(context, act).reshape_as(contexts)
        # The original pred_proj also contains BatchNorm. Pooling all shifted
        # contexts would leak later teacher-forced states into early outputs.
        # Pool only predictions with the same absolute source-frame time.
        times = torch.arange(horizon, device=z.device)[:, None] + torch.arange(HISTORY, device=z.device)[None]
        projected = torch.empty_like(raw)
        for time in range(HISTORY+horizon-1):
            selected = times == time
            values = raw[:, selected]
            projected[:, selected] = self.model.pred_proj(values.reshape(-1, self.latent_dim)).reshape_as(values)
        return projected

    def loss(self, batch, horizon: int, stage: str = "train"):
        pixels, actions = self._batch(batch, horizon)
        # LeWM has no derivative target and does not consume the extra two tail
        # frames. K controls how many shifted teacher-forced context windows are
        # supervised, NOT a self-fed K-step training objective.
        z = self._encode(pixels[:, :HISTORY+horizon])
        targets = torch.stack([z[:, j+1:j+HISTORY+1] for j in range(horizon)], 1)
        pred = self._teacher_predictions(z, actions, horizon)
        prediction = F.mse_loss(pred, targets)
        reg = self.sigreg(z.transpose(0, 1))
        return self._output(prediction, reg, target_position_std=z.detach().std())

    @torch.no_grad()
    def observed_features(self, history, past_actions=None):
        self._history(history, past_actions)
        return self._encode(history)[:, -2:].flatten(1)

    @torch.no_grad()
    def rollout_features(self, history, actions, past_actions=None):
        past_actions = self._future(history, actions, past_actions)
        context = self._encode(history)
        all_actions = torch.cat([past_actions, actions], 1)
        features = [context[:, -2:].flatten(1)]
        for k in range(actions.shape[1]):
            act = self.model.action_encoder(all_actions[:, k:k+HISTORY])
            pred = self.model.predict(context, act)[:, -1:]
            context = torch.cat([context[:, 1:], pred], 1)
            features.append(context[:, -2:].flatten(1))
        return torch.stack(features, 1)

    def provenance(self):
        return {**super().provenance(),
                "native_core": "original lucas-maes/le-wm JEPA, ARPredictor, Embedder, MLP and SIGReg",
                "adaptation": "LeHami-author ConvEncoder replaces original ViT; latent64 replaces original192; attention context21 replaces3; matched 64x64 RGB",
                "objective": "native one-step teacher-forced MSE over K shifted21 contexts plus SIGReg weight .09; not LeHami-author multistep LeWM objective",
                "readout_features": "last two 64D latents, diagnostic projection; full recurrent state is21 latents; no past action features",
                "batchnorm_boundary": "encoder/projector train per raw time; pred_proj pools only matching absolute source times across shifted contexts; replaces official flattened temporal pooling; inference uses fixed running statistics"}


def build_visual_baseline(kind: str, config: dict | None = None):
    classes = {"lehami": LeHamiComparison, "lewm": LeWMComparison}
    if kind not in classes:
        raise ValueError("kind must be lehami or lewm")
    return classes[kind](config)
