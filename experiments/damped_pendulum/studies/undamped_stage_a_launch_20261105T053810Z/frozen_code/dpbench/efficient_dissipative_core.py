"""Predicted-coefficient discrete gradients with Picard and implicit backward.

This is a NEW discrete map, not an equivalent implementation of midpoint D(q,v).
The continuous model and parameters are unchanged. D and B are predicted at the
midpoint once per accepted leaf and remain differentiable with respect to its
initial state, context and parameters. Forward root iterations have no training
graph. Backward differentiates the converged-root equation once per leaf.

The SPD Picard preconditioner handles the frozen damping exactly. Nonconvergent
samples use finite-difference Newton/Armijo for THE SAME residual, then bounded
subdivision. Original physical residual and chain-defect acceptance are retained.
No optimizer is run by this module. Only first-order outer gradients are supported.
"""
from __future__ import annotations

import math

import torch
from torch.autograd.function import once_differentiable

from dpbench.automatic_discrete_gradient import (
    potential_discrete_gradient, potential_value_and_gradient,
)
from dpbench.dissipative_core import DissipativeCore, SolverError


class _ImplicitRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, root, q, p, h, damping, drive, core, *parameters):
        ctx.core = core
        ctx.save_for_backward(root, q, p, h, damping, drive, *parameters)
        return root.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        root, q0, p0, h0, d0, b0, *parameters = ctx.saved_tensors
        core = ctx.core
        with torch.enable_grad():
            v, q, p, h, damping, drive = [x.detach().requires_grad_(True)
                                         for x in (root, q0, p0, h0, d0, b0)]
            residual, _ = core._frozen_residual(v, q, p, h, damping, drive)
            rows = [torch.autograd.grad(residual[:, j].sum(), v, retain_graph=True)[0]
                    for j in range(core.q_dim)]
            jacobian = torch.stack(rows, -2).detach()
            identity = torch.eye(core.q_dim, dtype=q.dtype, device=q.device).expand_as(jacobian)
            rhs = torch.cat((upstream[..., None], identity), -1)
            solved, info = torch.linalg.solve_ex(jacobian.transpose(-1, -2), rhs)
            adjoint, inverse_t = solved[..., 0], solved[..., 1:]
            condition = (jacobian.abs().sum(-1).amax(-1)
                         * inverse_t.abs().sum(-2).amax(-1))
            correction = (inverse_t.transpose(-1, -2) @ residual.detach()[..., None]).squeeze(-1)
            root_error_limit = 4 * (core.residual_atol + core.residual_rtol * torch.maximum(
                p.detach().abs().amax(-1), (2*root-p.detach()).abs().amax(-1)))
            defect = (jacobian.transpose(-1, -2) @ adjoint[..., None]).squeeze(-1) - upstream
            scale = (1 + upstream.abs().amax(-1)
                     + jacobian.abs().sum(-2).amax(-1) * adjoint.abs().amax(-1))
            accepted = ((info == 0) & torch.isfinite(solved).all(dim=(-2, -1))
                        & torch.isfinite(jacobian).all(dim=(-2, -1))
                        & torch.isfinite(condition) & (condition <= 1 / core.implicit_rcond_min)
                        & torch.isfinite(correction).all(-1)
                        & (correction.abs().amax(-1) <= root_error_limit)
                        & (defect.abs().amax(-1) <= core.adjoint_rtol * scale))
            if not accepted.all():
                raise SolverError('Implicit backward rejected singular/ill-conditioned Jacobian, root correction or bad adjoint')
            differentiable_parameters = [x for x in parameters if x.requires_grad]
            targets = [q, p, h, damping, drive, *differentiable_parameters]
            gradients = torch.autograd.grad(residual, targets, grad_outputs=-adjoint.detach(),
                                            allow_unused=True)
            finite = [torch.isfinite(x).all() for x in gradients if x is not None]
            if finite and not torch.stack(finite).all():
                raise SolverError('Nonfinite implicit-root input/parameter gradient')
            state_gradients = list(gradients[:5])
            par_iter = iter(gradients[5:])
            parameter_gradients = [next(par_iter) if x.requires_grad else None for x in parameters]
        return (None, *state_gradients, None, *parameter_gradients)


class EfficientDissipativeCore(DissipativeCore):
    """Same learned H/D/B, different second-order predicted-coefficient map."""

    def __init__(self, *, picard_iterations=6, picard_check_every=2,
                 fd_relative_step=6e-6, implicit_rcond_min=1e-8,
                 adjoint_rtol=1e-10, **base):
        integrator = base.pop('integrator', 'frozen_dg')
        solver = base.pop('solver', 'picard')
        if integrator != 'frozen_dg' or solver != 'picard':
            raise ValueError('Efficient core requires integrator=frozen_dg and solver=picard')
        for name, value in (('picard_iterations', picard_iterations),
                            ('picard_check_every', picard_check_every)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name, value in (('fd_relative_step', fd_relative_step),
                            ('implicit_rcond_min', implicit_rcond_min),
                            ('adjoint_rtol', adjoint_rtol)):
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f'{name} must be finite and between zero and one')
        base.setdefault('newton_iterations', 12)
        base.setdefault('max_bisections', 4)
        super().__init__(integrator='discrete_gradient', solver='backtracking', **base)
        self.integrator, self.solver = integrator, solver
        self.picard_iterations, self.picard_check_every = picard_iterations, picard_check_every
        self.fd_relative_step = fd_relative_step
        self.implicit_rcond_min, self.adjoint_rtol = implicit_rcond_min, adjoint_rtol
        self.config.update(integrator=integrator, solver=solver, picard_iterations=picard_iterations,
                           picard_check_every=picard_check_every, fd_relative_step=fd_relative_step,
                           implicit_rcond_min=implicit_rcond_min, adjoint_rtol=adjoint_rtol)

    def potential_gradient(self, q):
        return potential_value_and_gradient(self.potential_net, q)[1]

    def discrete_gradient(self, q, q_next):
        return potential_discrete_gradient(self.potential_net, q, q_next).gradient

    def _predict_coefficients(self, q, p, action, h, context):
        # Only the root solver freezes these tensors. Their upstream graph is kept.
        d0 = self.damping(q, p, context)
        force0 = (self.input_matrix(q) @ action[..., None]).squeeze(-1)
        identity = torch.eye(self.q_dim, dtype=q.dtype, device=q.device)
        matrix = identity + .5 * h[..., None] * d0
        rhs = p - .5 * h * (self.potential_gradient(q) - force0)
        predicted, code = torch.linalg.solve_ex(matrix, rhs[..., None])
        predicted = predicted.squeeze(-1)
        middle = q + .5 * h * p
        damping = self.damping(middle, predicted, context)
        drive = (self.input_matrix(middle) @ action[..., None]).squeeze(-1)
        finite = ((code == 0) & torch.isfinite(predicted).all(-1)
                  & torch.isfinite(damping).all(dim=(-2, -1)) & torch.isfinite(drive).all(-1))
        finite = (finite & torch.isfinite(d0).all(dim=(-2, -1))
                  & torch.isfinite(matrix).all(dim=(-2, -1)) & torch.isfinite(rhs).all(-1)
                  & torch.isfinite(middle).all(-1) & torch.isfinite(force0).all(-1))
        if not finite.all():
            raise SolverError('Nonfinite predicted midpoint coefficients')
        return damping, drive, predicted

    def _frozen_residual(self, v, q, p, h, damping, drive):
        gradient = self.discrete_gradient(q, q + h * v)
        force = (damping @ v[..., None]).squeeze(-1)
        return v - p + .5 * h * (gradient + force - drive), gradient

    def _ledger(self, q, p, h, damping, drive, v):
        _, gradient = self._frozen_residual(v, q, p, h, damping, drive)
        qn, pn = q + h * v, 2 * v - p
        force = (damping @ v[..., None]).squeeze(-1)
        dq, dp = qn - q, pn - p
        rq, rp = dq - h * v, dp + h * (gradient + force - drive)
        residual = torch.cat((rq, rp), -1).abs().amax(-1)
        work = h.squeeze(-1) * (drive * v).sum(-1)
        dissipation = h.squeeze(-1) * (force * v).sum(-1)
        before, after = self.energy(q, p), self.energy(qn, pn)
        chain = after - before - (gradient * dq).sum(-1) - (v * dp).sum(-1)
        residual_work = (gradient * rq).sum(-1) + (v * rp).sum(-1)
        tolerance = self.residual_atol + self.residual_rtol * torch.maximum(
            p.abs().amax(-1), pn.abs().amax(-1))
        chain_limit = self.chain_atol * (1 + before.abs() + after.abs())
        values = torch.stack((residual, work, dissipation, chain, residual_work,
                              before, after, tolerance, chain_limit), -1)
        finite = torch.isfinite(values).all(-1) & torch.isfinite(torch.cat((qn, pn, gradient), -1)).all(-1)
        accepted = finite & (residual <= tolerance) & (chain.abs() <= chain_limit)
        ratio = torch.where(tolerance > 0, residual / tolerance.clamp_min(torch.finfo(q.dtype).tiny),
                            torch.where(residual == 0, torch.zeros_like(residual), torch.full_like(residual, float('inf'))))
        return accepted, dict(work=work, dissipation=dissipation, residual_work=residual_work,
                              chain_defect=chain, chain_defect_max=chain.abs(),
                              numerical_balance_defect=chain + residual_work,
                              solver_residual=residual, solver_residual_ratio=ratio)

    def _root_threshold(self, v, p):
        return .25 * (self.residual_atol + self.residual_rtol * torch.maximum(
            p.abs().amax(-1), (2 * v - p).abs().amax(-1)))

    def _fd_jacobian(self, v, q, p, h, damping, drive):
        # Differentiate only G with respect to its ENDPOINT. Linear terms exact.
        del p, drive
        b, d = q.shape
        endpoint = q + h * v
        eps = self.fd_relative_step * (1 + endpoint.abs())
        perturb = torch.diag_embed(eps)
        endpoints = torch.cat((endpoint[:, None, :] + perturb,
                               endpoint[:, None, :] - perturb), 1)
        starts = q[:, None, :].expand(-1, 2 * d, -1)
        gradients = self.discrete_gradient(starts.reshape(-1, d), endpoints.reshape(-1, d)).reshape(b, 2*d, d)
        derivative = ((gradients[:, :d] - gradients[:, d:]) / (2 * eps[..., None])).transpose(-1, -2)
        identity = torch.eye(d, dtype=q.dtype, device=q.device)
        return identity + .5 * h[..., None] * damping + .5 * h.square()[..., None] * derivative

    def _newton_fallback(self, v, q, p, h, damping, drive):
        counts = {k: q.new_zeros(len(q)) for k in ('solver_newton_iterations', 'solver_backtracks', 'solver_line_search_evals')}
        alive = torch.ones(len(q), device=q.device, dtype=torch.bool)
        for _ in range(self.newton_iterations):
            residual, _ = self._frozen_residual(v, q, p, h, damping, drive)
            alive = alive & torch.isfinite(residual).all(-1)
            active = alive & (residual.abs().amax(-1) > self._root_threshold(v, p))
            if not active.any():
                break
            ids = active.nonzero(as_tuple=True)[0]
            jacobian = self._fd_jacobian(v[ids], q[ids], p[ids], h[ids], damping[ids], drive[ids])
            finite_j = torch.isfinite(jacobian).all(dim=(-2, -1))
            safe_j = torch.where(finite_j[:, None, None], jacobian,
                                 torch.eye(self.q_dim, dtype=q.dtype, device=q.device))
            direction, code = torch.linalg.solve_ex(safe_j, residual[ids, :, None])
            direction = direction.squeeze(-1)
            good = finite_j & (code == 0) & torch.isfinite(direction).all(-1)
            alive[ids[~good]] = False
            pending, direction = ids[good], direction[good]
            counts['solver_newton_iterations'][ids] += 1
            phi0 = .5 * residual[pending].square().sum(-1)
            alpha = q.new_ones(len(pending))
            for backtrack in range(self.line_search_max_backtracks + 1):
                if not len(pending):
                    break
                trial = v[pending] - alpha[:, None] * direction
                rr, _ = self._frozen_residual(trial, q[pending], p[pending], h[pending], damping[pending], drive[pending])
                phi = .5 * rr.square().sum(-1)
                sufficient = torch.isfinite(trial).all(-1) & torch.isfinite(phi)
                sufficient = sufficient & (phi <= (1 - 2*self.line_search_c1*alpha) * phi0)
                counts['solver_line_search_evals'][pending] += 1
                v[pending[sufficient]] = trial[sufficient]
                rejected = ~sufficient
                if backtrack == self.line_search_max_backtracks:
                    alive[pending[rejected]] = False
                    break
                counts['solver_backtracks'][pending[rejected]] += 1
                pending, direction = pending[rejected], direction[rejected]
                phi0, alpha = phi0[rejected], alpha[rejected] * self.line_search_shrink
        return v, counts

    @torch.no_grad()
    def _solve_leaf(self, q, p, h, damping, drive, initial):
        identity = torch.eye(self.q_dim, dtype=q.dtype, device=q.device)
        matrix = identity + .5 * h[..., None] * damping
        factor, factor_info = torch.linalg.cholesky_ex(matrix)
        factor_good = (factor_info == 0) & torch.isfinite(factor).all(dim=(-2, -1))
        safe_factor = torch.where(factor_good[:, None, None], factor, identity)
        v = initial.clone()
        alive = factor_good & torch.isfinite(v).all(-1)
        counts = {k: q.new_zeros(len(q)) for k in ('solver_picard_iterations', 'solver_newton_iterations',
                   'solver_backtracks', 'solver_line_search_evals')}
        for iteration in range(self.picard_iterations):
            residual, gradient = self._frozen_residual(v, q, p, h, damping, drive)
            finite = torch.isfinite(residual).all(-1) & torch.isfinite(gradient).all(-1)
            alive = alive & finite
            active = alive & (residual.abs().amax(-1) > self._root_threshold(v, p))
            if iteration % self.picard_check_every == 0 and not active.any():
                break
            rhs = p - .5 * h * (gradient - drive)
            # No invalid inactive RHS is passed to the solve, even in the detached path.
            rhs = torch.where(active[:, None], rhs, torch.zeros_like(rhs))
            trial = torch.cholesky_solve(rhs[..., None], safe_factor).squeeze(-1)
            good = active & torch.isfinite(trial).all(-1)
            counts['solver_picard_iterations'] += active.to(q.dtype)
            alive = alive & (~active | good)
            v = torch.where(good[:, None], trial, v)
        accepted, _ = self._ledger(q, p, h, damping, drive, v)
        bad = (~accepted).nonzero(as_tuple=True)[0]
        if len(bad):
            candidate, extra = self._newton_fallback(v[bad].clone(), q[bad], p[bad], h[bad], damping[bad], drive[bad])
            v = v.index_copy(0, bad, candidate)
            for key, value in extra.items():
                counts[key] = counts[key].index_add(0, bad, value)
        accepted, ledger = self._ledger(q, p, h, damping, drive, v)
        return v, accepted, ledger, counts

    def _interval(self, q, p, action, h, context, track_gradient, depth=0):
        damping, drive, initial = self._predict_coefficients(q, p, action, h, context)
        root, accepted, ledger, counts = self._solve_leaf(
            q.detach(), p.detach(), h.detach(), damping.detach(), drive.detach(), initial.detach())
        good, bad = accepted.nonzero(as_tuple=True)[0], (~accepted).nonzero(as_tuple=True)[0]
        qn, pn = torch.zeros_like(q), torch.zeros_like(p)
        if len(good):
            v = root[good]
            if track_gradient:
                v = _ImplicitRoot.apply(v, q[good], p[good], h[good], damping[good], drive[good],
                                        self, *tuple(self.potential_net.parameters()))
            qn = qn.index_copy(0, good, q[good] + h[good] * v)
            pn = pn.index_copy(0, good, 2*v - p[good])
        info = {key: torch.where(accepted, value, torch.zeros_like(value)) for key, value in ledger.items()}
        info.update(counts)
        info.update(solver_iterations=counts['solver_picard_iterations'] + counts['solver_newton_iterations'],
                    solver_failed_attempts=(~accepted).to(q.dtype),
                    solver_leaf_steps=accepted.to(q.dtype),
                    solver_split_depth=torch.where(accepted, q.new_full((len(q),), depth), q.new_zeros(len(q))),
                    solver_replay_iterations=q.new_zeros(len(q)), solver_replay_relative_error=q.new_zeros(len(q)))
        if len(bad):
            if depth >= self.max_bisections:
                raise SolverError(f'Predicted-D solve exhausted bisection depth {depth}; failed={len(bad)}/{len(q)}')
            bc = None if context is None else context[bad]
            qm, pm, left = self._interval(q[bad], p[bad], action[bad], h[bad]/2, bc, track_gradient, depth+1)
            qe, pe, right = self._interval(qm, pm, action[bad], h[bad]/2, bc, track_gradient, depth+1)
            child = self._merge_ledgers(left, right)
            qn, pn = qn.index_copy(0, bad, qe), pn.index_copy(0, bad, pe)
            costs = set(counts) | {'solver_iterations', 'solver_failed_attempts'}
            for key in info:
                info[key] = (info[key].index_add(0, bad, child[key]) if key in costs
                             else info[key].index_copy(0, bad, child[key]))
        return qn, pn, info

    def forward(self, q, p, action, dt, context=None):
        if torch.is_inference_mode_enabled():
            raise RuntimeError('Use torch.no_grad(), not inference_mode(), for this core')
        if q.ndim != 2 or q.shape[-1] != self.q_dim or p.shape != q.shape:
            raise ValueError('q,p must have shape [B,q_dim]')
        if q.device.type not in {'cpu', 'cuda'} or q.dtype not in {torch.float32, torch.float64}:
            raise ValueError('Only CPU/CUDA float32/float64 states are supported')
        if (p.dtype != q.dtype or p.device != q.device or self.factor_basis.dtype != q.dtype
                or self.factor_basis.device != q.device):
            raise ValueError('States and core must share dtype/device')
        action = torch.as_tensor(action, dtype=q.dtype, device=q.device)
        if action.shape != (len(q), self.action_dim):
            raise ValueError('action must have shape [B,action_dim]')
        dt = torch.as_tensor(dt, dtype=q.dtype, device=q.device)
        if dt.ndim == 0:
            dt = dt.expand(len(q), 1)
        elif dt.ndim == 1:
            dt = dt[:, None]
        if dt.shape != (len(q), 1):
            raise ValueError('dt must be scalar/[B]/[B,1]')
        valid = torch.stack([torch.isfinite(x).all() for x in (q,p,action,dt)])
        if not valid.all() or not (dt > 0).all():
            raise ValueError('Finite inputs and positive dt required')
        context = self._context(q, context)
        track_gradient = torch.is_grad_enabled()
        initial_energy = self.energy(q,p).detach()
        if not torch.isfinite(initial_energy).all():
            raise SolverError('Nonfinite initial energy')
        h = dt / self.substeps
        diagnostics = None
        for _ in range(self.substeps):
            q,p,current = self._interval(q,p,action,h,context,track_gradient)
            diagnostics = current if diagnostics is None else self._merge_ledgers(diagnostics,current)
        with torch.no_grad():
            diagnostics['energy_delta'] = self.energy(q,p) - initial_energy
            diagnostics['energy_residual'] = diagnostics['energy_delta'] - diagnostics['work'] + diagnostics['dissipation']
            finite = torch.stack([torch.isfinite(x).all() for x in (q,p,*diagnostics.values())])
            if not finite.all():
                raise SolverError('Nonfinite final state or accumulated predicted-D ledger')
        return q,p,{k:v.detach() for k,v in diagnostics.items()}
