"""Causal visual JEPA with position/velocity dependent dissipative dynamics.

Future frames are targets only. GroupNorm acts within an individual image,
so encoding a target frame cannot alter the initial state via batch statistics.
The encoder is FP32 and the mechanical latent recurrence is FP64. There is no
decoder, teacher forcing, physical-label supervision, or optimizer in this file.
"""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn

from dpbench.dissipative_core import DissipativeCore

EFFICIENT_CORE_DEFAULTS = dict(picard_iterations=6, picard_check_every=2,
                               fd_relative_step=6e-6, implicit_rcond_min=1e-8,
                               adjoint_rtol=1e-10)


def default_model_config():
    return dict(history=5, velocity_history=5, environment_context_dim=0,
                dt=0.1, q_dim=2, encoder_width=32,
                velocity_correction_scale=1.0, sigreg_weight=0.25,
                motion_weight=1.0, initializer_weight=1e-3,
                velocity_initializer="legacy", motion_loss_mode="displacement",
                motion_scale_floor=0.1, momentum_weight=0.0,
                sigreg_knots=17, sigreg_projections=256,
                variance_floor=1e-4,
                core=dict(hidden_dim=32, action_dim=1,
                          damping_mode="position_velocity", potential_mode="softplus",
                          integrator="discrete_gradient", substeps=2,
                          newton_iterations=6, residual_atol=1e-8,
                          residual_rtol=1e-7, chain_atol=1e-10, strict=True,
                          solver="full", line_search_max_backtracks=12,
                          line_search_shrink=0.5, line_search_c1=1e-4, max_bisections=0))


def resolve_model_config(config=None):
    defaults = default_model_config()
    supplied = copy.deepcopy(config or {})
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown model options: {sorted(unknown)}")
    core_supplied = supplied.pop("core", {})
    if (not isinstance(core_supplied, dict)
            or set(core_supplied) - (set(defaults["core"]) | set(EFFICIENT_CORE_DEFAULTS))):
        raise ValueError("Unknown or invalid core configuration")
    defaults["core"].update(core_supplied)
    if defaults["core"]["integrator"] == "frozen_dg":
        for key, value in EFFICIENT_CORE_DEFAULTS.items():
            defaults["core"].setdefault(key, value)
    elif set(core_supplied) & set(EFFICIENT_CORE_DEFAULTS):
        raise ValueError("Efficient-core options require integrator=frozen_dg")
    defaults.update(supplied)
    for key in ("history", "velocity_history", "q_dim", "encoder_width", "sigreg_knots", "sigreg_projections"):
        if isinstance(defaults[key], bool) or not isinstance(defaults[key], int) or defaults[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 3 <= defaults["velocity_history"] <= defaults["history"] or defaults["q_dim"] < 2:
        raise ValueError("Require 3 <= velocity_history <= history and q_dim>=2")
    context_dim = defaults["environment_context_dim"]
    if isinstance(context_dim, bool) or not isinstance(context_dim, int) or context_dim < 0:
        raise ValueError("environment_context_dim must be a nonnegative integer")
    if defaults["dt"] <= 0 or defaults["encoder_width"] < 8 or defaults["encoder_width"] % 8:
        raise ValueError("dt must be positive; encoder_width must be a positive multiple of 8")
    if defaults["sigreg_knots"] < 2 or defaults["sigreg_projections"] < 1:
        raise ValueError("Invalid SIGReg discretization")
    real_options = ("dt", "sigreg_weight", "motion_weight", "initializer_weight",
                    "velocity_correction_scale", "variance_floor", "motion_scale_floor", "momentum_weight")
    if any(isinstance(defaults[k], bool) or not isinstance(defaults[k], (int, float))
           or not math.isfinite(defaults[k]) for k in real_options):
        raise ValueError("Real-valued model options must be finite numbers")
    if any(defaults[k] < 0 for k in ("sigreg_weight", "motion_weight", "initializer_weight",
                                    "velocity_correction_scale", "momentum_weight")) or defaults["variance_floor"] <= 0:
        raise ValueError("Loss weights must be nonnegative and variance_floor positive")
    if defaults["motion_scale_floor"] <= 0:
        raise ValueError("motion_scale_floor must be positive")
    if (not isinstance(defaults["velocity_initializer"], str)
            or defaults["velocity_initializer"] not in {"legacy", "motion_gated"}):
        raise ValueError("Unknown velocity_initializer")
    if (not isinstance(defaults["motion_loss_mode"], str)
            or defaults["motion_loss_mode"] not in {"displacement", "normalized_velocity"}):
        raise ValueError("Unknown motion_loss_mode")
    core = defaults["core"]
    for key in ("hidden_dim", "action_dim", "substeps", "newton_iterations"):
        if isinstance(core[key], bool) or not isinstance(core[key], int) or core[key] < 1:
            raise ValueError(f"core.{key} must be a positive integer")
    for key in ("residual_atol", "residual_rtol", "chain_atol"):
        if isinstance(core[key], bool) or not isinstance(core[key], (int, float)) or not math.isfinite(core[key]) or core[key] < 0:
            raise ValueError(f"core.{key} must be finite and nonnegative")
    for key in ("line_search_max_backtracks", "max_bisections"):
        if isinstance(core[key], bool) or not isinstance(core[key], int) or core[key] < 0:
            raise ValueError(f"core.{key} must be a nonnegative integer")
    for key, upper in (("line_search_shrink", 1.0), ("line_search_c1", 0.5)):
        if (isinstance(core[key], bool) or not isinstance(core[key], (int, float))
                or not math.isfinite(core[key]) or not 0 < core[key] < upper):
            raise ValueError(f"Invalid core.{key}")
    if core["solver"] not in {"full", "backtracking", "picard"}:
        raise ValueError("Unknown nonlinear solver")
    if core["max_bisections"] and core["solver"] not in {"backtracking", "picard"}:
        raise ValueError("Bisection requires backtracking")
    if core["solver"] == "backtracking" and core["integrator"] != "discrete_gradient":
        raise ValueError("Backtracking requires discrete_gradient")
    if core["integrator"] not in {"discrete_gradient", "rk4", "frozen_dg"}:
        raise ValueError("Unknown integrator")
    if (core["integrator"] == "frozen_dg") != (core["solver"] == "picard"):
        raise ValueError("frozen_dg and picard must be selected together")
    if core["integrator"] == "frozen_dg":
        for key in ("picard_iterations", "picard_check_every"):
            if isinstance(core[key], bool) or not isinstance(core[key], int) or core[key] < 1:
                raise ValueError(f"core.{key} must be a positive integer")
        for key in ("fd_relative_step", "implicit_rcond_min", "adjoint_rtol"):
            if isinstance(core[key], bool) or not isinstance(core[key], (int, float)) or not math.isfinite(core[key]) or not 0 < core[key] < 1:
                raise ValueError(f"Invalid core.{key}")
    if core["damping_mode"] not in {"none", "constant", "position", "position_velocity"}:
        raise ValueError("Unknown damping mode")
    if not isinstance(core["potential_mode"], str) or core["potential_mode"] not in {"softplus", "raw"}:
        raise ValueError("Unknown potential mode")
    if defaults["core"].get("action_dim", 1) != 1:
        raise ValueError("This dataset adapter has exactly one action channel")
    if defaults["core"].get("strict", True) is not True:
        raise ValueError("The visual training path requires strict solver checks")
    return defaults


def _official_sigreg(knots, projections):
    # Read-only reuse of the pinned original class, not a similarly named loss.
    from dpbench.models import verify_sources
    verify_sources()
    path = Path(__file__).resolve().parents[1] / "third_party/le-wm/module.py"
    spec = importlib.util.spec_from_file_location("_dpbench_dissipative_sigreg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SIGReg(knots=knots, num_proj=projections)


class FrameEncoder(nn.Module):
    def __init__(self, width, q_dim):
        super().__init__()
        layers = []
        channels = [3, width, 2 * width, 3 * width, 4 * width]
        for inc, outc in zip(channels, channels[1:]):
            layers.extend([nn.Conv2d(inc, outc, 3, stride=2, padding=1),
                           nn.GroupNorm(8, outc), nn.GELU()])
        self.net = nn.Sequential(*layers, nn.Flatten(),
                                 nn.Linear(4 * width * 4 * 4, 128), nn.GELU(),
                                 nn.Linear(128, q_dim))

    def forward(self, frames):
        if frames.ndim != 4 or tuple(frames.shape[1:]) != (3, 64, 64):
            raise ValueError("Encoder expects [N,3,64,64] RGB")
        return self.net(frames.float() * 2 - 1)


class DissipativeWorldModel(nn.Module):
    kind = "dissipative"

    def __init__(self, config=None):
        super().__init__()
        self.config = resolve_model_config(config)
        c = self.config
        self.history, self.dt, self.q_dim = c["history"], c["dt"], c["q_dim"]
        self.velocity_history = c["velocity_history"]
        self.environment_context_dim = c["environment_context_dim"]
        self.encoder = FrameEncoder(c["encoder_width"], self.q_dim)
        history_features = self.history * self.q_dim + 2 * (self.history - 1)
        self.velocity_correction = nn.Sequential(
            nn.Linear(history_features, 64),
            nn.Tanh(), nn.Linear(64, self.q_dim))
        nn.init.zeros_(self.velocity_correction[-1].weight)
        nn.init.zeros_(self.velocity_correction[-1].bias)
        core_type = DissipativeCore
        if c["core"]["integrator"] == "frozen_dg":
            from dpbench.efficient_dissipative_core import EfficientDissipativeCore
            core_type = EfficientDissipativeCore
        self.core = core_type(q_dim=self.q_dim, context_dim=self.environment_context_dim,
                              **c["core"])
        self.environment_encoder = None
        if self.environment_context_dim:
            self.environment_encoder = nn.Sequential(
                nn.Linear(history_features, 64), nn.Tanh(),
                nn.Linear(64, self.environment_context_dim), nn.Tanh())
            # Nonzero output weights preserve a gradient path into the history
            # encoder from the first backward pass. This is not a b readout.
            nn.init.normal_(self.environment_encoder[-2].weight, std=0.02)
            nn.init.zeros_(self.environment_encoder[-2].bias)
        self.sigreg = _official_sigreg(c["sigreg_knots"], c["sigreg_projections"])

    def encoder_parameters(self):
        return list(self.encoder.parameters())

    def dynamics_parameters(self):
        parameters = list(self.core.parameters()) + list(self.velocity_correction.parameters())
        if self.environment_encoder is not None:
            parameters += list(self.environment_encoder.parameters())
        return parameters

    def encode_frames(self, frames):
        if frames.ndim != 5:
            raise ValueError("frames must have shape [B,T,3,64,64]")
        b, t = frames.shape[:2]
        return self.encoder(frames.reshape(b * t, *frames.shape[2:])).reshape(b, t, self.q_dim)

    def _history_features(self, encoded, past_actions, past_dt):
        return torch.cat([encoded.flatten(1), past_actions.float().flatten(1),
                          past_dt.float().flatten(1)], -1)

    def infer_context(self, encoded_history, past_actions, past_dt):
        """Infer a fixed environment context from causal history, without b labels."""
        if self.environment_encoder is None:
            return None
        features = self._history_features(encoded_history, past_actions, past_dt)
        return self.environment_encoder(features).double()

    @staticmethod
    def _quadratic_velocity(encoded, intervals, evaluation_index):
        """Local quadratic derivative at an observed point in physical seconds.

        Time and positions are centered before the solve, in FP64. Intervals
        may vary by time and sample. This helper is used only by new options;
        the legacy initializer retains its original floating-point recurrence.
        """
        intervals = intervals.double()
        times = torch.cat([intervals.new_zeros(intervals.shape[0], 1),
                           intervals[..., 0].cumsum(1)], 1)
        span = times[:, -1:]
        tau = (times - times[:, evaluation_index:evaluation_index + 1]) / span
        design = torch.stack([torch.ones_like(tau), tau, tau.square()], -1)
        values = encoded.double() - encoded[:, evaluation_index:evaluation_index + 1].double()
        coefficients = torch.linalg.solve(design.transpose(1, 2) @ design,
                                          design.transpose(1, 2) @ values)
        return coefficients[:, 1] / span

    @staticmethod
    def _static_observation(frames):
        """Detect identical observed images, not equality of FP32 encodings.

        Repeated inputs can produce slightly different convolution roundoff in
        different batch positions. Pixel equality remains well-defined then.
        This is a zero-motion invariant, not a claim of subpixel observability.
        """
        return (frames == frames[:, -1:]).flatten(1).all(1)

    def initialize(self, history, past_actions, past_dt):
        """Quadratic causal fit in physical time plus bounded learned correction.

        The last observed q is kept exactly; only its velocity is estimated.
        M=I, so p=v. The local fit uses the trailing velocity_history images;
        a longer environment history must not turn it into a fit over a period.
        """
        b = history.shape[0]
        if history.shape[1] != self.history:
            raise ValueError(f"Exactly {self.history} initial frames are required")
        for value, name in ((past_actions, "past_actions"), (past_dt, "past_dt")):
            if value.shape != (b, self.history - 1, 1) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite [B,{self.history - 1},1]")
        if (past_dt <= 0).any():
            raise ValueError("History intervals must be positive")
        encoded = self.encode_frames(history)
        q_history = encoded.double()
        local_dt = past_dt[:, -(self.velocity_history - 1):]
        if self.config["velocity_initializer"] == "motion_gated":
            local_q = q_history[:, -self.velocity_history:]
            fitted_velocity = self._quadratic_velocity(local_q, local_dt, self.velocity_history - 1)
            # Center all history features: absolute position offsets cannot
            # create a correction. The head shape/state_dict is unchanged.
            centered = (q_history - q_history[:, -1:]).float()
            features = self._history_features(centered, past_actions, past_dt)
            local_velocity = (local_q[:, 1:] - local_q[:, :-1]) / local_dt.double()
            power = local_velocity.square().mean(dim=(1, 2), keepdim=False)[:, None]
            gate = power / (power + self.config["motion_scale_floor"] ** 2)
            correction = (self.config["velocity_correction_scale"]
                          * self.velocity_correction(features).tanh().double() * gate)
            static = self._static_observation(history[:, -self.velocity_history:])[:, None]
            correction = torch.where(static, torch.zeros_like(correction), correction)
            momentum = torch.where(static, torch.zeros_like(fitted_velocity), fitted_velocity + correction)
            return q_history[:, -1], momentum, encoded, correction
        times = torch.cat([local_dt.new_zeros(b, 1), local_dt[..., 0].cumsum(1)], 1).double()
        span = times[:, -1:]
        tau = (times - times[:, -1:]) / span
        design = torch.stack([torch.ones_like(tau), tau, tau.square()], -1)
        coefficients = torch.linalg.solve(design.transpose(1, 2) @ design,
                                          design.transpose(1, 2) @ q_history[:, -self.velocity_history:])
        fitted_velocity = coefficients[:, 1] / span
        features = self._history_features(encoded, past_actions, past_dt)
        correction = self.config["velocity_correction_scale"] * self.velocity_correction(features).tanh()
        return q_history[:, -1], fitted_velocity + correction.double(), encoded, correction

    def rollout(self, history, actions, past_actions, past_dt, future_dt):
        """Only initial RGB and explicit past/future actions and intervals enter."""
        b, steps = actions.shape[:2]
        if actions.shape != (b, steps, 1) or future_dt.shape != actions.shape:
            raise ValueError("Future actions and dt must have shape [B,K,1]")
        q, p, encoded, correction = self.initialize(history, past_actions, past_dt)
        context = self.infer_context(encoded, past_actions, past_dt)
        q_values, p_values, diagnostics = [q], [p], {}
        for k in range(steps):
            q, p, current = self.core(q, p, actions[:, k], future_dt[:, k], context=context)
            q_values.append(q)
            p_values.append(p)
            for key, value in current.items():
                diagnostics.setdefault(key, []).append(value)
        return dict(q=torch.stack(q_values, 1), p=torch.stack(p_values, 1),
                    history_encoding=encoded, correction=correction, context=context,
                    diagnostics={k: torch.stack(v, 1) for k, v in diagnostics.items()})

    def predict_batch(self, batch, horizon):
        # Do not route other batch fields, even if supplied by an external caller.
        pixels, action, dt = (batch[key] for key in ("pixels", "action", "dt_frame"))
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive number of frame intervals")
        n = self.history + horizon
        if pixels.shape[1] < n or action.shape[1] < n - 1 or dt.shape[1] < n - 1:
            raise ValueError("Window does not cover history and requested horizon")
        anchor = self.history - 1
        result = self.rollout(pixels[:, :self.history], action[:, anchor:n - 1],
                              action[:, :anchor], dt[:, :anchor], dt[:, anchor:n - 1])
        # Target encoding happens after the complete recurrence; no batch-statistics layer.
        future_targets = self.encode_frames(pixels[:, self.history:n])
        result["target"] = torch.cat([result["history_encoding"][:, -1:], future_targets], 1).double()
        result["all_encoded"] = torch.cat([result["history_encoding"], future_targets], 1)
        return result

    def _momentum_targets(self, batch, result):
        """Training-only detached central derivatives at anchor/future times.

        Use five observations (two on each side) when available, otherwise
        three (one on each side). The final observation is excluded because
        it has no right neighbor. Irregular dt is used in every fit. No true
        physical state or future observation enters initialize/rollout.
        Returns aligned predicted p and detached target p, each [B,N,q_dim].
        """
        encoded = result["all_encoded"]
        targets, indices = [], []
        for index in range(self.history - 1, encoded.shape[1] - 1):
            radius = 2 if index >= 2 and index + 2 < encoded.shape[1] else 1
            lo, hi = index - radius, index + radius + 1
            target = self._quadratic_velocity(encoded[:, lo:hi].detach(),
                                              batch["dt_frame"][:, lo:hi - 1], radius)
            static = self._static_observation(batch["pixels"][:, lo:hi])[:, None]
            target = torch.where(static, torch.zeros_like(target), target)
            targets.append(target.detach())
            indices.append(index - self.history + 1)
        if not targets:
            return result["p"][:, :0], result["p"][:, :0].detach()
        return result["p"][:, indices], torch.stack(targets, 1)

    def loss(self, batch, horizon):
        """JEPA objective plus optional velocity and detached momentum targets.

        normalized_velocity uses v=delta_q/dt and per-window
        mean((v_pred-v_target)^2)/stopgrad(max(mean(v_target^2),floor^2)).
        Momentum uses the same normalization on detached central fitted
        derivatives, including the anchor but excluding unsupported tail points.

        Validation velocity powers/error/cross are means over B,T,q. Their
        dynamic_* counterparts zero windows with target RMS<=floor and retain
        the full-B denominator. Aggregate these means with batch-size weights.
        *_count are scalar sums; sum them directly. The fixed stall criterion
        is predicted RMS<0.1*target RMS in a dynamic window. Motion magnitude
        alone does not imply correctness: error and signed cross must accompany
        the power ratio. All diagnostics are detached; loss terms retain grads.
        """
        result = self.predict_batch(batch, horizon)
        predicted, target = result["q"], result["target"]
        prediction = (predicted[:, 1:] - target[:, 1:]).square().mean()
        predicted_delta = predicted[:, 1:] - predicted[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        c = self.config
        intervals = batch["dt_frame"][:, self.history - 1:self.history - 1 + horizon].double()
        if not torch.isfinite(intervals).all() or (intervals <= 0).any():
            raise ValueError("Future intervals must be finite and positive")
        predicted_velocity, target_velocity = predicted_delta / intervals, target_delta / intervals
        window_target_power = target_velocity.square().mean(dim=(1, 2))
        velocity_error = (predicted_velocity - target_velocity).square()
        # These are visual-latent derivatives, not physical velocity labels.
        # New loss is mean_b mean_t,d(error_v^2) / stopgrad(max(power_v, floor^2)).
        if c["motion_loss_mode"] == "normalized_velocity":
            scale = window_target_power.detach().clamp_min(c["motion_scale_floor"] ** 2)
            motion = (velocity_error.mean(dim=(1, 2)) / scale).mean()
        else:
            motion = (predicted_delta - target_delta).square().mean()
        reg = self.sigreg(result["all_encoded"].transpose(0, 1))
        initial_penalty = result["correction"].square().mean()
        loss = (prediction + c["motion_weight"] * motion + c["sigreg_weight"] * reg
                + c["initializer_weight"] * initial_penalty)
        momentum, momentum_count = prediction.new_zeros(()), prediction.new_zeros(())
        if c["momentum_weight"]:
            predicted_momentum, target_momentum = self._momentum_targets(batch, result)
            momentum_count = prediction.new_tensor(target_momentum.shape[0] * target_momentum.shape[1])
            if target_momentum.shape[1]:
                momentum_scale = target_momentum.square().mean(dim=(1, 2)).clamp_min(c["motion_scale_floor"] ** 2)
                momentum = ((predicted_momentum - target_momentum).square().mean(dim=(1, 2))
                            / momentum_scale.detach()).mean()
            loss = loss + c["momentum_weight"] * momentum
        target_std = target[:, 1:].detach().flatten(0, 1).std(0, unbiased=False)
        variance = target_std.square().mean().clamp_min(c["variance_floor"])
        persistence = (target[:, :1] - target[:, 1:]).detach().square().mean()
        output = dict(loss=loss, prediction_loss=prediction, motion_loss=motion,
                      sigreg_loss=reg, initializer_loss=initial_penalty,
                      target_std_min=target_std.min(), target_std_mean=target_std.mean(),
                      target_motion_rms=target_delta.detach().square().mean().sqrt(),
                      target_motion_mse=target_delta.detach().square().mean(),
                      persistence_mse=persistence, prediction_mse=prediction.detach(),
                      normalized_prediction_mse=prediction.detach() / variance,
                      persistence_ratio=prediction.detach() / persistence.clamp_min(c["variance_floor"]))
        # Powers/cross products are mean over B,T,q. Dynamic variants zero
        # nondynamic windows but KEEP the full-B denominator, so validation can
        # aggregate exactly with batch-size weights. *_count are sums, not means.
        # A dynamic window has target RMS > motion_scale_floor. A stalled dynamic
        # window has predicted RMS < 0.1 * target RMS (fixed diagnostic threshold).
        pv, tv = predicted_velocity.detach(), target_velocity.detach()
        target_power = tv.square().mean(dim=(1, 2))
        predicted_power = pv.square().mean(dim=(1, 2))
        dynamic = target_power > c["motion_scale_floor"] ** 2
        stalled = dynamic & (predicted_power < .01 * target_power)
        window_metrics = dict(predicted_velocity_power=predicted_power,
                              target_velocity_power=target_power,
                              velocity_error_power=(pv - tv).square().mean(dim=(1, 2)),
                              velocity_cross_power=(pv * tv).mean(dim=(1, 2)))
        for key, value in window_metrics.items():
            output[key] = value.mean()
            output["dynamic_" + key] = (value * dynamic).mean()
        output.update(momentum_loss=momentum, momentum_point_count=momentum_count,
                      dynamic_window_count=dynamic.sum().to(prediction),
                      stalled_dynamic_window_count=stalled.sum().to(prediction),
                      window_count=prediction.new_tensor(len(predicted)))
        for index in range(self.q_dim):
            coordinate = target[:, 1:, index].detach()
            output[f"target_mean_{index}"] = coordinate.mean()
            output[f"target_second_moment_{index}"] = coordinate.square().mean()
        for key, value in result["diagnostics"].items():
            if key in {"solver_residual", "chain_defect_max", "energy_residual", "numerical_balance_defect",
                       "solver_residual_ratio", "solver_split_depth", "solver_replay_relative_error"}:
                output[key] = value.abs().max()
            else:
                output[key] = value.mean()
        return output

    @torch.no_grad()
    def rollout_features(self, history, actions, past_actions=None):
        """Compatibility with frozen physical readouts; returns full (q,p).

        Existing dpbench evaluators use the dataset's constant dt. The training
        interface and rollout() above additionally accept explicit varying dt.
        """
        if self.training:
            raise RuntimeError("Call eval() before physical readout evaluation")
        if past_actions is None:
            raise ValueError("Explicit past_actions are required for causal initialization")
        b, steps = actions.shape[:2]
        past_dt = history.new_full((b, self.history - 1, 1), self.dt)
        future_dt = history.new_full((b, steps, 1), self.dt)
        out = self.rollout(history, actions, past_actions, past_dt, future_dt)
        return torch.cat([out["q"], out["p"]], -1)

    def provenance(self):
        return dict(label="Dissipative JEPA with a finite-step energy ledger",
                    config=copy.deepcopy(self.config), parameters=sum(p.numel() for p in self.parameters()),
                    observation="RGB/action/dt only; no physical targets",
                    initializer=f"{self.config['velocity_initializer']}: "
                                f"{self.velocity_history}-frame local causal quadratic fit in seconds; "
                                f"bounded correction from {self.history}-frame history",
                    motion_objective=self.config["motion_loss_mode"],
                    momentum_target=("detached central quadratic derivative, 5-point preferred/3-point boundary; "
                                     "anchor/future points with both neighbors only"
                                     if self.config["momentum_weight"] else "disabled"),
                    environment_context=(f"{self.environment_context_dim}-dimensional context inferred from "
                                         f"{self.history} causal frames/actions/dt; fixed during rollout; "
                                         "conditions damping only; no true damping labels"
                                         if self.environment_context_dim else "disabled"),
                    encoder="new single-frame GroupNorm convolutional encoder; not unchanged LeWM architecture",
                    regularizer="pinned upstream LeWM SIGReg; jointly differentiable target branch",
                    precision="FP32 perception, FP64 dynamics; CPU/CUDA",
                    energy_scope="learned latent storage, not calibrated physical energy",
                    gradient=("implicit accepted-root gradient; first-order outer derivatives only"
                              if self.config["core"]["integrator"] == "frozen_dg"
                              else "differentiate the actual finite solver iterations"))
