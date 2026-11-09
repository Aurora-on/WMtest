"""Explicit ordinary/discrete differentiation of scalar Sequential potentials.

Supported networks contain nested Sequential, Linear, Tanh, Softplus and Identity
modules. The discrete chain rule replaces each elementwise activation derivative
by a symmetric divided difference, then propagates a scalar output cotangent
through the layers. No ``torch.autograd.grad`` call or nested backward is needed;
ordinary PyTorch autograd still differentiates these formulas with respect to
the endpoints and the original network parameters.

At well-separated endpoints the chain rule is an exact-arithmetic identity. For
small activation differences a fourth-degree symmetric Taylor expansion avoids
cancellation in the value *and in its parameter/input derivatives*. It has an
O(delta**6) slope remainder, not an exact discrete identity. The returned actual
chain defect must be checked by the time integrator. Softplus's thresholded
linear branch is honored: a threshold crossing always uses the actual endpoint
secant and is never silently replaced by a smooth Softplus approximation.

These helpers do not add host/device finite checks to the hot path. The caller
must reject nonfinite outputs and a chain defect exceeding its stated tolerance.
They do not turn an arbitrary module into a supported potential or guarantee
that a nonlinear time step has a unique root.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class DiscreteGradientResult:
    gradient: torch.Tensor
    value_start: torch.Tensor
    value_end: torch.Tensor
    chain_defect: torch.Tensor


def _layers(potential: nn.Module) -> tuple[nn.Module, ...]:
    if type(potential) is nn.Sequential:
        return tuple(layer for child in potential for layer in _layers(child))
    # Exact types deliberately reject custom subclasses with overridden forward.
    if type(potential) not in (nn.Linear, nn.Tanh, nn.Softplus, nn.Identity):
        raise TypeError(f"Unsupported potential layer: {type(potential).__name__}")
    if type(potential) is nn.Softplus:
        if not math.isfinite(potential.beta) or potential.beta <= 0:
            raise ValueError("Softplus beta must be finite and positive")
        if not math.isfinite(potential.threshold):
            raise ValueError("Softplus threshold must be finite")
    return (potential,)


def _check_inputs(q: torch.Tensor, other: torch.Tensor | None = None) -> None:
    if not isinstance(q, torch.Tensor) or q.ndim < 1 or q.shape[-1] < 1:
        raise ValueError("Potential input must have shape [..., dimensions]")
    if q.dtype not in (torch.float32, torch.float64):
        raise TypeError("Explicit potential derivatives require float32 or float64")
    if other is not None and (not isinstance(other, torch.Tensor)
                              or other.shape != q.shape or other.dtype != q.dtype
                              or other.device != q.device):
        raise ValueError("Potential endpoints must have identical shape/dtype/device")


def _slope(layer: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if type(layer) is nn.Tanh:
        return 1 - y.square()
    if type(layer) is nn.Softplus:
        scaled = layer.beta * x
        return torch.where(scaled > layer.threshold, torch.ones_like(x),
                           torch.sigmoid(scaled))
    if type(layer) is nn.Identity:
        return torch.ones_like(x)
    raise TypeError("A slope is only defined here for supported activations")


def _secant(layer: nn.Module, a: torch.Tensor, b: torch.Tensor,
            fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
    """Symmetric endpoint secant with safe arithmetic on every evaluated branch."""
    if type(layer) is nn.Identity:
        return torch.ones_like(a)
    delta = b - a
    # Midpoint this way avoids overflow from a+b for equal, large finite inputs.
    middle = 0.5 * a + 0.5 * b
    # Series contains terms through delta**4. This activation-scale threshold
    # makes the omitted delta**6 term of order machine precision before constants.
    cutoff = torch.finfo(a.dtype).eps ** (1 / 6)
    if type(layer) is nn.Tanh:
        small = delta.abs() <= cutoff
        regular = ~small
        # Use safe branch operands BEFORE division/powers; output masking alone
        # would not prevent NaN gradients in an unused singular expression.
        safe_delta = torch.where(regular, delta, torch.ones_like(delta))
        divided = (fb - fa) / safe_delta
        series_delta = torch.where(small, delta, torch.zeros_like(delta))
        t = torch.tanh(middle)
        t2 = t.square()
        first = 1 - t2
        third = 2 * first * (3 * t2 - 1)
        fifth = first * (16 - 120 * t2 + 120 * t2.square())
        series = first + series_delta.square() * third / 24
        series = series + series_delta.pow(4) * fifth / 1920
        return torch.where(small, series, divided)
    if type(layer) is nn.Softplus:
        beta = layer.beta
        linear_a, linear_b = beta * a > layer.threshold, beta * b > layer.threshold
        both_linear = linear_a & linear_b
        crossing = linear_a != linear_b
        small = (delta.abs() <= cutoff / beta) & ~crossing & ~both_linear
        regular = ~small & ~both_linear
        safe_delta = torch.where(regular, delta, torch.ones_like(delta))
        divided = (fb - fa) / safe_delta
        # βδ, rather than δ, is the local activation scale.
        scaled_delta = beta * torch.where(small, delta, torch.zeros_like(delta))
        s = torch.sigmoid(beta * middle)
        common = s * (1 - s)
        third = common * (1 - 2 * s)
        fifth = common * (1 - 14 * s + 36 * s.square() - 24 * s.pow(3))
        series = s + scaled_delta.square() * third / 24
        series = series + scaled_delta.pow(4) * fifth / 1920
        return torch.where(both_linear, torch.ones_like(delta),
                           torch.where(small, series, divided))
    raise TypeError("A secant is only defined here for supported activations")


def potential_value_and_gradient(potential: nn.Module, q: torch.Tensor
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scalar V(q) and its explicit ordinary gradient, both differentiable.

    Output shapes are ``q.shape[:-1]`` and ``q.shape``. No network state or
    gradients are cached, detached, or re-registered in another module.
    """
    _check_inputs(q)
    layers = _layers(potential)
    value, trace = q, []
    for layer in layers:
        next_value = layer(value)
        derivative = None if type(layer) is nn.Linear else _slope(layer, value, next_value)
        trace.append((layer, derivative))
        value = next_value
    if value.shape[-1] != 1:
        raise ValueError("Potential must produce exactly one scalar per input")
    gradient = torch.ones_like(value)
    for layer, derivative in reversed(trace):
        gradient = gradient @ layer.weight if type(layer) is nn.Linear else gradient * derivative
    return value.squeeze(-1), gradient


def potential_discrete_gradient(potential: nn.Module, q: torch.Tensor,
                               q_next: torch.Tensor) -> DiscreteGradientResult:
    """Return symmetric layerwise discrete gradient and measured chain defect.

    ``chain_defect = V(q_next)-V(q)-gradient·(q_next-q)`` is computed from the
    actual network endpoint outputs. Its shape is ``q.shape[:-1]``. Endpoints
    are evaluated together so each network layer is called once on a pair axis.
    """
    _check_inputs(q, q_next)
    layers = _layers(potential)
    pair, trace = torch.stack((q, q_next), dim=-2), []
    for layer in layers:
        next_pair = layer(pair)
        if type(layer) is nn.Linear:
            derivative = None
        else:
            derivative = _secant(layer, pair[..., 0, :], pair[..., 1, :],
                                 next_pair[..., 0, :], next_pair[..., 1, :])
        trace.append((layer, derivative))
        pair = next_pair
    if pair.shape[-1] != 1:
        raise ValueError("Potential must produce exactly one scalar per input")
    start, end = pair[..., 0, 0], pair[..., 1, 0]
    gradient = torch.ones_like(pair[..., 0, :])
    for layer, derivative in reversed(trace):
        gradient = gradient @ layer.weight if type(layer) is nn.Linear else gradient * derivative
    defect = end - start - (gradient * (q_next - q)).sum(-1)
    return DiscreteGradientResult(gradient, start, end, defect)
