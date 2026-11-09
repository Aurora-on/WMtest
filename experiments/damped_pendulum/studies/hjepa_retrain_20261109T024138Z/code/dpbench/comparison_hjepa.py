"""Matched-history H-JEPA adaptation using pinned, executed official core code.

The official pH predictor, fixed subspace, Bures isotropy loss and PIC loss are
loaded byte-for-byte. The visual frontend is the pinned LeHami ConvEncoder,
followed by a causal 21-frame GRU adapter. This is a 64px passive-pendulum
adaptation, not a reproduction of the authors' ViT/control benchmark.

Training computes CNN batch statistics separately for each raw time index.
Future target frames therefore cannot alter the anchor through BatchNorm.
The official auxiliary PIC loss uses observed target phase contexts; those
contexts never enter the autonomous prediction branch or inference rollout.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn


UPSTREAM = Path(__file__).resolve().parents[1] / "third_party"
COMMIT = "fd4f7de927daa7831a6d51e95e0959a813c13901"


def verify_hjepa_sources():
    manifest = json.loads((UPSTREAM / "hjepa_source.json").read_text())
    if manifest["commit"] != COMMIT or manifest["directory"] != "hjepa":
        raise RuntimeError("Unexpected H-JEPA source manifest")
    for entry in manifest["files"]:
        path = UPSTREAM / "hjepa" / entry["path"]
        data = path.read_bytes()
        blob = b"blob " + str(len(data)).encode() + b"\0" + data
        if (len(data) != entry["bytes"]
                or hashlib.sha1(blob).hexdigest() != entry["git_blob_sha1"]):
            raise RuntimeError(f"H-JEPA source hash mismatch: {path}")
    return manifest


def _official_parts():
    verify_hjepa_sources()
    modules = []
    for filename in ("module", "hamiltonian_predictor", "hjepa_loss"):
        # Never bind the ambiguous upstream name 'module': LeWM uses it too.
        name = "_dpbench_hjepa_official_" + filename
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, UPSTREAM / "hjepa" / (filename + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[name] = module
        modules.append(sys.modules[name])
    return tuple(modules)


def default_hjepa_config(tiny=False):
    if type(tiny) is not bool:
        raise ValueError("tiny must be boolean")
    return dict(kind="hjepa", history=21, dt=0.1, physical_dt=0.1, predictor_dt=1.0,
                encoder_embed_dim=16 if tiny else 192,
                percept_dim=8 if tiny else 192, rank=4 if tiny else 32,
                projector_hidden=16 if tiny else 2048,
                dynamics_hidden=16 if tiny else 256,
                dynamics_layers=2 if tiny else 3,
                head_init_std=1e-3, rho_init=1e-2,
                seed=0, passive_only=True, tiny_fixture=tiny,
                history_adapter="causal_window_gru",
                cnn_batchnorm_policy="separate_raw_time_batches")


def _resolve(config):
    config = copy.deepcopy(config or {})
    result = default_hjepa_config(tiny=config.get("tiny_fixture", False))
    unknown = set(config) - set(result)
    if unknown:
        raise ValueError(f"Unknown H-JEPA configuration: {sorted(unknown)}")
    result.update(config)
    if result["kind"] != "hjepa" or result["history"] != 21:
        raise ValueError("H-JEPA comparison requires kind=hjepa and history=21")
    integer_fields = ("encoder_embed_dim", "percept_dim", "rank", "projector_hidden",
                      "dynamics_hidden", "dynamics_layers")
    if any(type(result[k]) is not int or result[k] < 1 for k in integer_fields):
        raise ValueError("H-JEPA dimensions/layers must be positive integers")
    if result["rank"] % 2 or result["rank"] > result["percept_dim"]:
        raise ValueError("rank must be even and no greater than percept_dim")
    for key in ("dt", "physical_dt", "predictor_dt", "head_init_std", "rho_init"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if (result["dt"] != result["physical_dt"] or result["dt"] != .1
            or result["predictor_dt"] != 1.0):
        raise ValueError("This protocol fixes physical_dt=.1 and normalized predictor_dt=1")
    if type(result["seed"]) is not int or result["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if result["passive_only"] is not True:
        raise ValueError("This comparison adapter only supports passive actions")
    if (result["history_adapter"] != "causal_window_gru"
            or result["cnn_batchnorm_policy"] != "separate_raw_time_batches"):
        raise ValueError("Unsupported history or BatchNorm policy")
    return result


class _AnchorLossView:
    """Execute official loss on one matched anchor rather than all clip starts."""
    def __init__(self, owner, previous):
        self.owner = owner
        self.previous = previous
        self.predictor = owner.predictor

    def history(self, emb):
        # The official context's zero-padded first difference is avoided by
        # supplying the actually available, causal pre-anchor encoded state.
        return self.owner.phase_context(torch.cat([self.previous[:, None], emb], 1))[:, 1:]

    def predict_aligned(self, emb, act, num_preds):
        if emb.shape[1] != num_preds + 1:
            raise ValueError("The matched H-JEPA loss has one anchor")
        predictions = self.owner._predict_states(self.previous, emb[:, 0], act[:, :num_preds])
        return predictions[:, 1:, None, :].transpose(1, 2)


class HJEPAComparison(nn.Module):
    kind = "hjepa"

    def __init__(self, config=None):
        super().__init__()
        self.config = _resolve(config)
        c = self.config
        self.history = c["history"]
        self.dt = c["dt"]
        self.feature_dim = 2 * c["rank"]
        official, dynamics, losses = _official_parts()
        from dpbench.models import _builders
        self.encoder = _builders().ConvEncoder(embed_dim=c["encoder_embed_dim"])
        self.history_encoder = nn.GRU(c["encoder_embed_dim"], c["encoder_embed_dim"], batch_first=True)
        self.projector = official.MLP(c["encoder_embed_dim"], c["projector_hidden"], c["percept_dim"])
        self.frame_norm = nn.LayerNorm(c["percept_dim"], elementwise_affine=False)
        self.state_proj = official.FixedRandomSubspace(c["percept_dim"], c["rank"], seed=c["seed"])
        self.phase_context = official.CausalPhaseContext(c["rank"])
        self.action_encoder = official.NetAction(1, net_steps=1)
        self.iso = official.IsotropyLoss()
        self.predictor = dynamics.ImplicitHamiltonianPredictor(
            c["rank"], 1, hidden=c["dynamics_hidden"], layers=c["dynamics_layers"],
            dt=c["predictor_dt"], head_init_std=c["head_init_std"], rho_init=c["rho_init"])
        self._upstream_loss = losses.hjepa_loss

    @staticmethod
    def _validate_pixels(pixels):
        if pixels.ndim != 5 or tuple(pixels.shape[2:]) != (3, 64, 64):
            raise ValueError("pixels must have shape [B,T,3,64,64]")
        if pixels.shape[0] < 1 or not pixels.is_floating_point():
            raise ValueError("pixels must be a nonempty floating point RGB batch")
        if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
            raise ValueError("pixels must be finite RGB in [0,1]")

    @staticmethod
    def _validate_actions(actions, batch, time, name):
        if actions.shape != (batch, time, 1) or not torch.isfinite(actions).all():
            raise ValueError(f"{name} must have shape [B,{time},1] and be finite")
        if torch.count_nonzero(actions).item():
            raise ValueError("Passive H-JEPA adaptation requires all actions to be zero")

    def _encode_states(self, pixels):
        """Return states from pre-anchor to last frame; each uses <=21 past frames.

        The pre-anchor at raw index19 repeats index0 once on its left: this
        supplies its missing older frame without reading outside the contract.
        Every anchor/target at raw index >=20 uses exactly21 causal frames.
        No target frame contributes to any earlier frame's CNN batch statistics.
        """
        batch, time = pixels.shape[:2]
        if time < self.history:
            raise ValueError("Encoding requires at least 21 raw frames")
        normalized = pixels.float().mul(2).sub(1)
        if self.training:
            encoded = torch.stack([self.encoder(normalized[:, t]) for t in range(time)], 1)
        else:
            encoded = self.encoder(normalized.flatten(0, 1)).unflatten(0, (batch, time))
        ends = torch.arange(self.history - 2, time, device=pixels.device)
        offsets = torch.arange(-self.history + 1, 1, device=pixels.device)
        indices = (ends[:, None] + offsets[None, :]).clamp_min(0)
        windows = encoded[:, indices].flatten(0, 1)
        _, hidden = self.history_encoder(windows)
        percept = self.frame_norm(self.projector(hidden[-1])).unflatten(0, (batch, len(ends)))
        return percept, self.state_proj.encode(percept)

    def _predict_states(self, previous, state, actions):
        predictions = [state]
        for t in range(actions.shape[1]):
            context = torch.cat([state, state - previous], -1)
            nxt = self.predictor(state, actions[:, t], context=context)
            previous, state = state, nxt
            predictions.append(state)
        return torch.stack(predictions, 1)

    def loss(self, batch, horizon, stage="train"):
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        for key in ("pixels", "action", "dt_frame"):
            if key not in batch:
                raise ValueError(f"Explicit {key} is required")
        pixels = batch["pixels"]
        self._validate_pixels(pixels)
        n, time = pixels.shape[:2]
        if time != self.history + horizon + 2:
            raise ValueError("Expected exactly21+horizon+2 common raw frames")
        self._validate_actions(batch["action"], n, time - 1, "action")
        frame_dt = batch["dt_frame"]
        if (frame_dt.shape != (n, time - 1, 1) or not torch.isfinite(frame_dt).all()
                or not torch.allclose(frame_dt, torch.full_like(frame_dt, self.dt), rtol=1e-5, atol=1e-7)):
            raise ValueError("dt_frame must contain physical interval .1 for every transition")
        # The common SG target-only tail is not required by this method.
        percept, states = self._encode_states(pixels[:, :self.history + horizon])
        acts = self.action_encoder(batch["action"][:, self.history - 1:self.history - 1 + horizon])
        out = self._upstream_loss(_AnchorLossView(self, states[:, 0]), states[:, 1:],
                                  percept[:, 1:], acts, num_preds=horizon, iso=self.iso)
        return dict(out, prediction_loss=out["pred_loss"])

    def _initial_states(self, history, past_actions):
        self._validate_pixels(history)
        if history.shape[1] != self.history:
            raise ValueError("history must contain exactly21 raw frames")
        if past_actions is not None:
            self._validate_actions(past_actions, history.shape[0], self.history - 1, "past_actions")
        _, states = self._encode_states(history)
        return states[:, 0], states[:, 1]

    @torch.no_grad()
    def observed_features(self, history, past_actions=None):
        if self.training:
            raise RuntimeError("Call eval() before extracting observed features")
        previous, state = self._initial_states(history, past_actions)
        return torch.cat([state, state - previous], -1)

    @torch.no_grad()
    def rollout_features(self, history, actions, past_actions=None):
        if self.training:
            raise RuntimeError("Call eval() before extracting rollout features")
        previous, state = self._initial_states(history, past_actions)
        if actions.ndim != 3:
            raise ValueError("actions must have shape [B,K,1]")
        self._validate_actions(actions, history.shape[0], actions.shape[1], "actions")
        encoded_actions = self.action_encoder(actions) if actions.shape[1] else actions.float()
        predicted = self._predict_states(previous, state, encoded_actions)
        preceding = torch.cat([previous[:, None], predicted[:, :-1]], 1)
        return torch.cat([predicted, predicted - preceding], -1)

    def provenance(self):
        from dpbench.models import verify_sources
        return dict(label="H-JEPA matched-history official-core adaptation",
                    upstream=verify_hjepa_sources(), visual_frontend_upstream=verify_sources(),
                    config=copy.deepcopy(self.config), feature_dim=self.feature_dim,
                    parameters=sum(p.numel() for p in self.parameters()),
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    executed_official_components=["ImplicitHamiltonianPredictor", "FixedRandomSubspace",
                        "CausalPhaseContext", "NetAction", "MLP", "IsotropyLoss", "hjepa_loss", "pic_loss"],
                    adaptations=["LeHami ConvEncoder replaces original ViT;64px RGB scaled to[-1,1]",
                        "21-frame causal GRU windows replace native per-frame perceptual encoding",
                        "Previous anchor window repeats its oldest available raw frame once",
                        "CNN training BatchNorm is computed separately per raw time index",
                        "One matched anchor instead of all sequence starts in the official loss",
                        "Native PIC uses observed target phases only in its auxiliary training loss",
                        "Inference only encodes initial21 frames; autonomous two-state recurrence",
                        "Feature readout uses[s_t,s_t-s_previous], the complete recurrence state",
                        "Passive zero actions only;physical dt=.1, native normalized Euler dt=1",
                        "Native predictor state-detach behavior preserved;not an official benchmark reproduction"])
