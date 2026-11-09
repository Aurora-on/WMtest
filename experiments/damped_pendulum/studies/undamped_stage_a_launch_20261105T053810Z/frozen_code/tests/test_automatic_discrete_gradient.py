"""Local numerical checks for explicit layerwise discrete differentiation.

No optimizer updates or training data are used. These checks concern values,
state/parameter derivatives, finite small differences, and threshold semantics.
"""
import pytest
import torch
from torch import nn

from dpbench.automatic_discrete_gradient import (
    potential_discrete_gradient,
    potential_value_and_gradient,
)


def potential():
    torch.manual_seed(2404)
    return nn.Sequential(
        nn.Sequential(nn.Linear(2, 7), nn.Tanh(), nn.Linear(7, 5), nn.Tanh(), nn.Linear(5, 1)),
        nn.Softplus(),
    ).double()


def points():
    return torch.tensor([[.4, -.3], [-.7, 1.2], [2., -.5]], dtype=torch.float64)


def test_explicit_gradient_matches_autograd_and_its_input_derivative():
    net, q = potential(), points().requires_grad_()
    value, explicit = potential_value_and_gradient(net, q)
    ordinary = torch.autograd.grad(net(q).sum(), q, create_graph=True)[0]
    torch.testing.assert_close(value, net(q).squeeze(-1), rtol=0, atol=0)
    torch.testing.assert_close(explicit, ordinary, atol=2e-15, rtol=2e-13)
    he = torch.autograd.grad(explicit.sum(), q, retain_graph=True)[0]
    ha = torch.autograd.grad(ordinary.sum(), q)[0]
    torch.testing.assert_close(he, ha, atol=2e-15, rtol=2e-13)


@pytest.mark.parametrize('distance', [0., 1e-12, 1e-8, 1e-4, .2, 3.])
def test_symmetry_chain_rule_and_diagonal_limit(distance):
    net, q = potential(), points()
    r = q + distance * torch.tensor([[.3, -.7]], dtype=q.dtype)
    forward, reverse = potential_discrete_gradient(net, q, r), potential_discrete_gradient(net, r, q)
    torch.testing.assert_close(forward.gradient, reverse.gradient, rtol=0, atol=0)
    torch.testing.assert_close(forward.value_start, net(q).squeeze(-1), atol=2e-15, rtol=2e-13)
    torch.testing.assert_close(forward.value_end, net(r).squeeze(-1), atol=2e-15, rtol=2e-13)
    assert forward.chain_defect.abs().max() < 2e-14
    assert torch.isfinite(forward.gradient).all()
    if distance == 0:
        _, ordinary = potential_value_and_gradient(net, q)
        torch.testing.assert_close(forward.gradient, ordinary, atol=2e-15, rtol=2e-13)


def test_diagonal_endpoint_derivatives_equal_half_the_hessian_contraction():
    net, q = potential(), points().requires_grad_()
    r = q.detach().clone().requires_grad_()
    discrete = potential_discrete_gradient(net, q, r).gradient
    first, second = torch.autograd.grad(discrete.sum(), (q, r))
    _, ordinary = potential_value_and_gradient(net, q)
    hessian_contraction = torch.autograd.grad(ordinary.sum(), q)[0]
    torch.testing.assert_close(first, .5 * hessian_contraction, atol=3e-15, rtol=3e-13)
    torch.testing.assert_close(second, .5 * hessian_contraction, atol=3e-15, rtol=3e-13)


@pytest.mark.parametrize('near', [False, True])
def test_endpoint_and_network_parameter_gradients_match_directional_differences(near):
    net, q = potential(), points().requires_grad_()
    r = (q.detach() + (1e-8 if near else .4)).requires_grad_()
    parameter = net[0][2].weight

    def objective():
        result = potential_discrete_gradient(net, q, r)
        return (result.gradient * torch.tensor([.3, -.7], dtype=q.dtype)).sum() + .2 * result.value_end.sum()

    inputs = (q, r, parameter)
    analytical = torch.autograd.grad(objective(), inputs)
    epsilon = 1e-6
    for value, grad in zip(inputs, analytical):
        direction = torch.linspace(-.3, .8, value.numel(), dtype=value.dtype).reshape(value.shape)
        baseline = value.detach().clone()
        with torch.no_grad():
            value.copy_(baseline + epsilon * direction)
            plus = objective()
            value.copy_(baseline - epsilon * direction)
            minus = objective()
            value.copy_(baseline)
        numerical = (plus - minus) / (2 * epsilon)
        assert torch.isfinite(grad).all()
        torch.testing.assert_close((grad * direction).sum(), numerical, rtol=3e-5, atol=3e-8)


def test_softplus_threshold_crossing_uses_actual_piecewise_values():
    # A low threshold deliberately magnifies the discontinuity in PyTorch's
    # numerical linear approximation, making a smooth replacement detectable.
    net = nn.Softplus(beta=2., threshold=5.)
    q = torch.tensor([[2.5 - 1e-5]], dtype=torch.float64, requires_grad=True)
    r = torch.tensor([[2.5 + 1e-5]], dtype=torch.float64, requires_grad=True)
    result = potential_discrete_gradient(net, q, r)
    expected = (net(r) - net(q)) / (r - q)
    torch.testing.assert_close(result.gradient, expected, rtol=0, atol=0)
    assert result.chain_defect.abs().max() < 1e-16
    assert result.gradient.item() < 0  # The thresholded function really jumps.
    assert all(torch.isfinite(x).all() for x in torch.autograd.grad(result.gradient.sum(), (q, r)))


@pytest.mark.parametrize('activation', [nn.Tanh(), nn.Softplus(beta=1.7, threshold=20.)])
def test_saturated_and_equal_endpoints_have_finite_gradients(activation):
    q = torch.tensor([[-1000.], [-20.], [0.], [20.], [1000.]], dtype=torch.float64, requires_grad=True)
    r = q.detach().clone().requires_grad_()
    result = potential_discrete_gradient(activation, q, r)
    grads = torch.autograd.grad(result.gradient.sum() + result.value_end.sum(), (q, r))
    assert torch.isfinite(result.gradient).all()
    assert torch.isfinite(result.chain_defect).all()
    assert all(torch.isfinite(g).all() for g in grads)
    torch.testing.assert_close(result.chain_defect, torch.zeros_like(result.chain_defect), rtol=0, atol=0)


def test_helpers_do_not_call_autograd_grad(monkeypatch):
    net, q = potential(), points().requires_grad_()
    def forbidden(*args, **kwargs):
        raise AssertionError('Nested automatic differentiation is not allowed')
    monkeypatch.setattr(torch.autograd, 'grad', forbidden)
    _, ordinary = potential_value_and_gradient(net, q)
    result = potential_discrete_gradient(net, q, q + .2)
    (ordinary.sum() + result.gradient.sum()).backward()
    assert torch.isfinite(q.grad).all()
    assert net[0][0].weight.grad is not None


def test_unsupported_or_nonscalar_potential_is_rejected():
    q = points()
    with pytest.raises(TypeError, match='Unsupported'):
        potential_discrete_gradient(nn.Sequential(nn.Linear(2, 1).double(), nn.ReLU()), q, q)
    with pytest.raises(ValueError, match='one scalar'):
        potential_discrete_gradient(nn.Linear(2, 3).double(), q, q)
    with pytest.raises(ValueError, match='identical'):
        potential_discrete_gradient(potential(), q, q[:1])
