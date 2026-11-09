"""Finite-step dissipative Hamiltonian core, with explicit numerical accounting.

M=I is a latent coordinate convention, not a physical mass calibration.  The
force is -D(q,v,c)v, D=LL^T, with optional fixed environment context c.
Context changes only D, not H or the input port. Newton iterations are differentiated as executed;
this module never substitutes a converged-root gradient for a finite solve.
The optional backtracking solver plans detached per-sample acceptance branches,
then differentiates only their accepted Newton iterates and subdivision tree.
The default parameters use float64.  No optimizer or training is run here.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class SolverError(RuntimeError):
    """A step failed its configured dynamics or chain-rule tolerance."""


def _network(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, outputs))


class DissipativeCore(nn.Module):
    """Batched autonomous transition (q,p,u,dt) -> (q_next,p_next,diagnostics).

    Shapes: q,p [B,d], action [B,a], dt scalar, [B], or [B,1].
    Optional context has shape [B,context_dim] and is fixed across substeps.
    Diagnostics are detached per-sample tensors.  Work/dissipation/residual
    work and chain defect are summed across substeps; solver_residual and
    chain_defect_max are the maximum absolute defects over all substeps;
    solver_iterations counts active Newton updates summed over substeps.
    energy_residual is the actual full-step dH-work+dissipation.
    numerical_balance_defect is chain_defect+residual_work for the discrete
    gradient step, and the actual energy residual for the RK4 control arm.

    A tiny-displacement midpoint branch avoids catastrophic divided
    differences; its measured chain defect is checked, never called exact.
    Strict mode raises on failed implicit steps rather than silently returning
    them.  The RK4 arm uses the same learned H/D/B and integrates state, work
    and dissipation with the same RK4 stages.  It has no discrete balance
    guarantee: normal truncation error is reported, not rejected.  Its solver
    and chain-rule diagnostics are zero placeholders (not applicable), with
    the integrator identified by self.config['integrator'].
    Inference under torch.no_grad is supported; torch.inference_mode is not,
    since Newton and the potential gradient require local autograd.

    solver='full' preserves the original recurrence. solver='backtracking'
    uses raw residual Armijo descent, per-sample bounded subdivision and the
    original residual/chain acceptance. Line-search and subdivision decisions
    are piecewise constant in backward; gradients are of that executed branch,
    not an exact root derivative. Failed/rejected trial graphs are discarded.
    solver_iterations counts attempted Newton directions including retries;
    solver_replay_iterations counts updates along accepted paths. Backtracks,
    line-search residual evaluations, failed attempts and accepted leaf steps
    are summed; split depth and accepted residual/tolerance ratio take maxima.
    """

    def __init__(self, q_dim=2, hidden_dim=32, action_dim=1,
                 damping_mode="position_velocity", substeps=2,
                 newton_iterations=6, residual_atol=1e-8,
                 residual_rtol=1e-7, chain_atol=1e-10, strict=True,
                 integrator="discrete_gradient", context_dim=0, solver="full",
                 line_search_max_backtracks=12, line_search_shrink=0.5,
                 line_search_c1=1e-4, max_bisections=0, potential_mode="softplus"):
        super().__init__()
        if any(not isinstance(x, int) or x < 1 for x in
               (q_dim, hidden_dim, action_dim, substeps, newton_iterations)):
            raise ValueError("dimensions, substeps and newton_iterations must be positive integers")
        if damping_mode not in {"none", "constant", "position", "position_velocity"}:
            raise ValueError("Unknown damping_mode")
        if not isinstance(potential_mode, str) or potential_mode not in {"softplus", "raw"}:
            raise ValueError("potential_mode must be softplus or raw")
        if integrator not in {"discrete_gradient", "rk4"}:
            raise ValueError("integrator must be discrete_gradient or rk4")
        if solver not in {"full", "backtracking"}:
            raise ValueError("Unknown nonlinear solver")
        for name, value in (("line_search_max_backtracks", line_search_max_backtracks),
                            ("max_bisections", max_bisections)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 < line_search_shrink < 1 or not 0 < line_search_c1 < 0.5:
            raise ValueError("Require 0 < shrink < 1 and 0 < Armijo c1 < 0.5")
        if solver == "backtracking" and (not strict or integrator != "discrete_gradient"):
            raise ValueError("Backtracking requires strict discrete-gradient integration")
        if max_bisections and solver != "backtracking":
            raise ValueError("Adaptive bisection requires the backtracking solver")
        if isinstance(context_dim, bool) or not isinstance(context_dim, int) or context_dim < 0:
            raise ValueError("context_dim must be a nonnegative integer")
        if any(not math.isfinite(x) or x < 0 for x in
               (residual_atol, residual_rtol, chain_atol)):
            raise ValueError("solver tolerances must be finite and nonnegative")
        self.config = dict(q_dim=q_dim, hidden_dim=hidden_dim, action_dim=action_dim,
                           damping_mode=damping_mode, potential_mode=potential_mode, substeps=substeps,
                           newton_iterations=newton_iterations, residual_atol=residual_atol,
                           residual_rtol=residual_rtol, chain_atol=chain_atol, strict=bool(strict),
                           integrator=integrator, context_dim=context_dim, solver=solver,
                           line_search_max_backtracks=line_search_max_backtracks,
                           line_search_shrink=line_search_shrink, line_search_c1=line_search_c1,
                           max_bisections=max_bisections)
        self.q_dim, self.action_dim = q_dim, action_dim
        self.context_dim = context_dim
        self.damping_mode = damping_mode
        self.potential_mode = potential_mode
        self.integrator = integrator
        self.solver = solver
        self.line_search_max_backtracks = line_search_max_backtracks
        self.line_search_shrink, self.line_search_c1 = line_search_shrink, line_search_c1
        self.max_bisections = max_bisections
        self.substeps, self.newton_iterations = substeps, newton_iterations
        self.residual_atol, self.residual_rtol = residual_atol, residual_rtol
        self.chain_atol, self.strict = chain_atol, bool(strict)
        potential = _network(q_dim, hidden_dim, 1)
        # Raw output is already lower bounded for each fixed set of parameters:
        # the last hidden Tanh gives V(q) >= bias - ||last_weight||_1. It may be
        # negative; storage nonnegativity can be restored by a q-independent
        # constant, which does not change forces or the finite-step ledger.
        # Avoiding an outer Softplus avoids its extra sigmoid force attenuation.
        # Identity preserves legacy state_dict keys; selecting raw nevertheless
        # changes the force field and is an explicit new model configuration.
        self.potential_net = nn.Sequential(
            potential, nn.Softplus() if potential_mode == "softplus" else nn.Identity())
        rows, cols = torch.tril_indices(q_dim, q_dim)
        basis = torch.zeros(len(rows), q_dim, q_dim)
        basis[torch.arange(len(rows)), rows, cols] = 1
        self.register_buffer("factor_basis", basis)
        initial_factor = (rows == cols).float() * math.sqrt(0.1)
        self.register_parameter("constant_factor", None)
        self.damping_net = None
        if damping_mode == "constant" and context_dim == 0:
            self.constant_factor = nn.Parameter(initial_factor)
        elif damping_mode != "none":
            state_inputs = {"constant": 0, "position": q_dim,
                            "position_velocity": 2 * q_dim}[damping_mode]
            inputs = state_inputs + context_dim
            self.damping_net = _network(inputs, hidden_dim, len(rows))
            with torch.no_grad():
                nn.init.normal_(self.damping_net[-1].weight, std=0.01)
                self.damping_net[-1].bias.copy_(initial_factor)
        self.input_net = _network(q_dim, hidden_dim, q_dim * action_dim)
        with torch.no_grad():
            nn.init.normal_(self.input_net[-1].weight, std=0.01)
            self.input_net[-1].bias.copy_(torch.eye(q_dim, action_dim).reshape(-1))
        self.double()

    def potential(self, q):
        """Smooth lower-bounded V (not necessarily positive/coercive in raw mode)."""
        return self.potential_net(q).squeeze(-1)

    def energy(self, q, p):
        return self.potential(q) + 0.5 * p.square().sum(-1)

    def _context(self, q, context):
        """Validate context without caching state or detaching its gradients."""
        if self.context_dim == 0:
            if context is not None:
                raise ValueError("context must be None when context_dim=0")
            return None
        if context is None:
            raise ValueError("context is required when context_dim>0")
        context = torch.as_tensor(context, dtype=q.dtype, device=q.device)
        if context.shape != (*q.shape[:-1], self.context_dim) or not torch.isfinite(context).all():
            raise ValueError("context must be finite with shape [B,context_dim]")
        return context

    def damping(self, q, v, context=None):
        # With context, constant means D(c): constant along a rollout but
        # allowed to differ across environments. Other modes add q and/or v.
        context = self._context(q, context)
        if self.damping_mode == "none":
            return q.new_zeros(*q.shape[:-1], self.q_dim, self.q_dim)
        if self.constant_factor is not None:
            raw = self.constant_factor.expand(*q.shape[:-1], -1)
        else:
            if self.damping_mode == "constant":
                features = context
            else:
                features = q if self.damping_mode == "position" else torch.cat([q, v], -1)
                if context is not None:
                    features = torch.cat([features, context], -1)
            raw = self.damping_net(features)
        factor = torch.einsum("...k,kij->...ij", raw, self.factor_basis)
        return factor @ factor.transpose(-1, -2)

    def input_matrix(self, q):
        return self.input_net(q).reshape(*q.shape[:-1], self.q_dim, self.action_dim)

    def discrete_gradient(self, q, q_next):
        """Gonzalez discrete gradient of V; caller must enable autograd."""
        delta = q_next - q
        middle = (q + q_next) * 0.5
        if not middle.requires_grad:
            middle = middle.requires_grad_(True)
        gradient = torch.autograd.grad(self.potential(middle).sum(), middle,
                                       create_graph=True)[0]
        numerator = self.potential(q_next) - self.potential(q) - (gradient * delta).sum(-1)
        norm2 = delta.square().sum(-1)
        # This is a branch, not an epsilon added to the denominator.  On the
        # small branch we explicitly audit the resulting chain-rule defect.
        threshold2 = (10 * math.sqrt(torch.finfo(q.dtype).eps)
                      * (1 + q.detach().norm(dim=-1))).square()
        small = norm2.detach() <= threshold2
        denominator = torch.where(small, torch.ones_like(norm2), norm2)
        correction = torch.where(small, torch.zeros_like(numerator), numerator / denominator)
        return gradient + correction.unsqueeze(-1) * delta

    def _residual(self, velocity, q, p, action, h, context=None):
        q_next = q + h * velocity
        middle = (q + q_next) * 0.5
        gradient = self.discrete_gradient(q, q_next)
        damping_force = (self.damping(middle, velocity, context) @ velocity.unsqueeze(-1)).squeeze(-1)
        drive = (self.input_matrix(middle) @ action.unsqueeze(-1)).squeeze(-1)
        residual = velocity - p + 0.5 * h * (gradient + damping_force - drive)
        return residual, gradient, damping_force, drive

    _MAX_DIAGNOSTICS = frozenset({"solver_residual", "solver_residual_ratio", "solver_replay_relative_error",
                                  "chain_defect_max", "solver_split_depth"})

    def _implicit_ledger(self, q, p, action, h, context, velocity):
        """Original physical acceptance, independent of line-search merit."""
        _, gradient, force, drive = self._residual(velocity, q, p, action, h, context)
        q_next, p_next = q + h * velocity, 2 * velocity - p
        dq, dp = q_next - q, p_next - p
        r_q, r_p = dq - h * velocity, dp - h * (-gradient - force + drive)
        residual = torch.cat((r_q, r_p), -1).abs().amax(-1)
        work = h.squeeze(-1) * (drive * velocity).sum(-1)
        dissipation = h.squeeze(-1) * (force * velocity).sum(-1)
        before, after = self.energy(q, p), self.energy(q_next, p_next)
        chain = after - before - (gradient * dq).sum(-1) - (velocity * dp).sum(-1)
        residual_work = (gradient * r_q).sum(-1) + (velocity * r_p).sum(-1)
        tolerance = self.residual_atol + self.residual_rtol * torch.maximum(
            p.abs().amax(-1), p_next.abs().amax(-1))
        chain_limit = self.chain_atol * (1 + before.detach().abs() + after.detach().abs())
        finite = torch.isfinite(torch.cat((q_next, p_next, gradient, force, drive), -1)).all(-1)
        for value in (residual, work, dissipation, chain, residual_work, before, after, tolerance, chain_limit):
            finite = finite & torch.isfinite(value)
        accepted = finite & (residual.detach() <= tolerance.detach()) & (chain.detach().abs() <= chain_limit)
        # A zero configured tolerance means exact acceptance, without division by zero.
        ratio = torch.where(tolerance > 0, residual / tolerance.clamp_min(torch.finfo(q.dtype).tiny),
                            torch.where(residual == 0, torch.zeros_like(residual), torch.full_like(residual, float("inf"))))
        info = dict(work=work, dissipation=dissipation, residual_work=residual_work,
                    chain_defect=chain, numerical_balance_defect=chain + residual_work,
                    chain_defect_max=chain.abs(), solver_residual=residual,
                    solver_residual_ratio=ratio)
        return q_next, p_next, accepted, {key: value.detach() for key, value in info.items()}

    def _plan_newton(self, q, p, action, h, context):
        """Choose detached per-sample Newton steps, never building a training graph.

        The merit is raw ||F||^2/2. Dynamic physical tolerances are used only for
        convergence, not as variable scaling in the Armijo objective. Singular,
        nonfinite or exhausted attempts are returned as failures to the splitter.
        """
        velocity = p.detach().clone().requires_grad_(True)
        alive = torch.ones(len(q), device=q.device, dtype=torch.bool)
        counts = {k: q.new_zeros(len(q)) for k in
                  ("solver_iterations", "solver_backtracks", "solver_line_search_evals")}
        alphas = []
        for _ in range(self.newton_iterations):
            residual = self._residual(velocity, q, p, action, h, context)[0]
            finite = torch.isfinite(residual).all(-1)
            alive = alive & finite
            # Half the physical acceptance threshold gives a replay-rounding margin.
            threshold = 0.25 * (self.residual_atol + self.residual_rtol * torch.maximum(
                p.abs().amax(-1), (2 * velocity - p).abs().amax(-1)))
            active = alive & (residual.detach().abs().amax(-1) > threshold.detach())
            if not active.any():
                break
            ids = active.nonzero(as_tuple=True)[0]
            rows = [torch.autograd.grad(residual[:, j].sum(), velocity,
                      retain_graph=True, create_graph=False)[0] for j in range(self.q_dim)]
            jacobian = torch.stack(rows, -2).detach()[ids]
            finite_j = torch.isfinite(jacobian).all(dim=(-2, -1))
            # solve_ex does not throw for a singular row; other samples remain valid.
            safe_j = torch.where(finite_j[:, None, None], jacobian,
                                 torch.eye(self.q_dim, dtype=q.dtype, device=q.device))
            update, linear_info = torch.linalg.solve_ex(safe_j, residual.detach()[ids, :, None])
            update = update.squeeze(-1)
            good = finite_j & (linear_info == 0) & torch.isfinite(update).all(-1)
            alive[ids[~good]] = False
            pending, direction = ids[good], update[good]
            counts["solver_iterations"][ids] += 1
            chosen_alpha = q.new_zeros(len(q))
            next_velocity = velocity.detach().clone()
            alpha = q.new_ones(len(pending))
            old_phi = 0.5 * residual.detach()[pending].square().sum(-1)
            for backtrack in range(self.line_search_max_backtracks + 1):
                if not len(pending):
                    break
                trial = (velocity.detach()[pending] - alpha[:, None] * direction).requires_grad_(True)
                trial_residual = self._residual(trial, q[pending], p[pending], action[pending],
                    h[pending], None if context is None else context[pending])[0].detach()
                phi = 0.5 * trial_residual.square().sum(-1)
                sufficient = torch.isfinite(phi) & torch.isfinite(trial.detach()).all(-1)
                sufficient = sufficient & (phi <= (1 - 2 * self.line_search_c1 * alpha) * old_phi)
                counts["solver_line_search_evals"][pending] += 1
                accepted_ids = pending[sufficient]
                chosen_alpha[accepted_ids] = alpha[sufficient]
                next_velocity[accepted_ids] = trial.detach()[sufficient]
                rejected = ~sufficient
                if backtrack == self.line_search_max_backtracks:
                    alive[pending[rejected]] = False
                    break
                counts["solver_backtracks"][pending[rejected]] += 1
                pending, direction = pending[rejected], direction[rejected]
                old_phi, alpha = old_phi[rejected], alpha[rejected] * self.line_search_shrink
            alphas.append(chosen_alpha)
            velocity = next_velocity.requires_grad_(True)
        q_next, p_next, accepted, ledger = self._implicit_ledger(q, p, action, h, context, velocity)
        # The original residual/chain ledger is authoritative. A direction may
        # fail while the last finite iterate already meets physical acceptance
        # (the internal Newton target deliberately has extra rounding margin).
        return q_next.detach(), p_next.detach(), accepted, alphas, ledger, counts

    def _merge_ledgers(self, first, second):
        return {key: (torch.maximum(first[key], second[key]) if key in self._MAX_DIAGNOSTICS
                      else first[key] + second[key]) for key in first}

    def _plan_interval(self, q, p, action, h, context, depth=0):
        q_next, p_next, ok, alphas, ledger, counts = self._plan_newton(q, p, action, h, context)
        good, bad = ok.nonzero(as_tuple=True)[0], (~ok).nonzero(as_tuple=True)[0]
        plan = dict(good=good, bad=bad, alphas=alphas)
        info = {key: torch.where(ok, value, torch.zeros_like(value)) for key, value in ledger.items()}
        info.update(counts)
        info.update(solver_failed_attempts=(~ok).to(q.dtype), solver_leaf_steps=ok.to(q.dtype),
                    solver_split_depth=torch.where(ok, q.new_full((len(q),), depth), q.new_zeros(len(q))),
                    solver_replay_iterations=torch.where(ok,
                        sum(((alpha > 0).to(q.dtype) for alpha in alphas), q.new_zeros(len(q))),
                        torch.zeros_like(q[:, 0])))
        if len(bad):
            if depth >= self.max_bisections:
                raise SolverError(f"Backtracking implicit solve exhausted bisection depth {depth}; "
                    f"failed={len(bad)}/{len(q)}, residual_ratio_max="
                    f"{ledger['solver_residual_ratio'][bad].max().item():.6g}")
            bad_context = None if context is None else context[bad]
            qm, pm, left, li = self._plan_interval(q[bad], p[bad], action[bad], h[bad] / 2, bad_context, depth + 1)
            qe, pe, right, ri = self._plan_interval(qm, pm, action[bad], h[bad] / 2, bad_context, depth + 1)
            children = self._merge_ledgers(li, ri)
            q_next, p_next = q_next.index_copy(0, bad, qe), p_next.index_copy(0, bad, pe)
            for key in info:
                if key in counts or key == "solver_failed_attempts":
                    info[key] = info[key].index_add(0, bad, children[key])
                else:
                    info[key] = info[key].index_copy(0, bad, children[key])
            plan.update(left=left, right=right)
        return q_next, p_next, plan, info

    def _replay_leaf(self, q, p, action, h, context, alphas):
        """Differentiate only the executed, accepted branch of the solve."""
        velocity = p.clone()
        if not velocity.requires_grad:
            velocity.requires_grad_(True)
        for alpha in alphas:
            ids = (alpha > 0).nonzero(as_tuple=True)[0]
            if not len(ids):
                continue
            v = velocity[ids]
            residual = self._residual(v, q[ids], p[ids], action[ids], h[ids],
                                      None if context is None else context[ids])[0]
            rows = [torch.autograd.grad(residual[:, j].sum(), v, create_graph=True,
                                       retain_graph=True)[0] for j in range(self.q_dim)]
            update = torch.linalg.solve(torch.stack(rows, -2), residual.unsqueeze(-1)).squeeze(-1)
            velocity = velocity.index_copy(0, ids, v - alpha[ids, None] * update)
        q_next, p_next, accepted, ledger = self._implicit_ledger(q, p, action, h, context, velocity)
        if not accepted.all():
            raise SolverError("Accepted Newton path failed original tolerance during differentiable replay")
        return q_next, p_next, ledger

    def _replay_interval(self, q, p, action, h, context, plan):
        good, bad = plan["good"], plan["bad"]
        q_next, p_next = torch.zeros_like(q), torch.zeros_like(p)
        ledger = None
        if len(good):
            qg, pg, gi = self._replay_leaf(q[good], p[good], action[good], h[good],
                None if context is None else context[good], [a[good] for a in plan["alphas"]])
            q_next, p_next = q_next.index_copy(0, good, qg), p_next.index_copy(0, good, pg)
            ledger = {key: value.new_zeros(len(q)).index_copy(0, good, value) for key, value in gi.items()}
        if len(bad):
            bc = None if context is None else context[bad]
            qm, pm, li = self._replay_interval(q[bad], p[bad], action[bad], h[bad]/2, bc, plan["left"])
            qe, pe, ri = self._replay_interval(qm, pm, action[bad], h[bad]/2, bc, plan["right"])
            bi = self._merge_ledgers(li, ri)
            q_next, p_next = q_next.index_copy(0, bad, qe), p_next.index_copy(0, bad, pe)
            if ledger is None:
                ledger = {key: value.new_zeros(len(q)) for key, value in bi.items()}
            for key, value in bi.items():
                ledger[key] = ledger[key].index_copy(0, bad, value)
        return q_next, p_next, ledger

    def _robust_step(self, q, p, action, h, track_gradient, context):
        # Planning keeps all discrete accept/retry decisions out of the gradient.
        # Failed attempts and rejected nonfinite candidates never enter backward.
        with torch.enable_grad():
            qd, pd = q.detach(), p.detach()
            cd = None if context is None else context.detach()
            q_next, p_next, plan, info = self._plan_interval(qd, pd, action.detach(), h.detach(), cd)
            info["solver_replay_relative_error"] = q.new_zeros(len(q))
            if track_gradient:
                planned = torch.cat((q_next, p_next), -1)
                q_next, p_next, replay_ledger = self._replay_interval(q, p, action, h, context, plan)
                info.update(replay_ledger)
                replayed = torch.cat((q_next.detach(), p_next.detach()), -1)
                info["solver_replay_relative_error"] = (replayed - planned).abs().amax(-1) / (1 + planned.abs().amax(-1))
        return q_next, p_next, {key: value.detach() for key, value in info.items()}

    def _step(self, q, p, action, h, track_gradient, context=None):
        if self.solver == "backtracking":
            return self._robust_step(q, p, action, h, track_gradient, context)
        # Clone makes velocity a separate autograd node: taking dF/dvelocity
        # must hold the initial p fixed even when the predictor equals p.
        velocity = p.clone()
        if not velocity.requires_grad:
            velocity.requires_grad_(True)
        iterations = q.new_zeros(q.shape[0])
        for _ in range(self.newton_iterations):
            residual, _, _, _ = self._residual(velocity, q, p, action, h, context)
            if not torch.isfinite(residual).all():
                raise SolverError("Non-finite Newton residual")
            tolerance = 0.5 * (self.residual_atol + self.residual_rtol
                              * torch.maximum(p.abs().amax(-1), (2 * velocity - p).abs().amax(-1)))
            active = residual.detach().abs().amax(-1) > tolerance.detach()
            if not active.any():
                # Remaining masked updates would all be identically zero.
                # Stop without changing any per-sample state or its gradient.
                break
            rows = [torch.autograd.grad(residual[:, j].sum(), velocity,
                                        create_graph=track_gradient, retain_graph=True)[0]
                    for j in range(self.q_dim)]
            jacobian = torch.stack(rows, -2)
            if not torch.isfinite(jacobian).all() or not torch.isfinite(residual).all():
                raise SolverError("Non-finite Newton residual/Jacobian")
            try:
                update = torch.linalg.solve(jacobian, residual.unsqueeze(-1)).squeeze(-1)
            except RuntimeError as exc:
                raise SolverError("Newton Jacobian solve failed") from exc
            velocity = velocity - torch.where(active.unsqueeze(-1), update, torch.zeros_like(update))
            iterations = iterations + active.to(q.dtype)
            if not track_gradient:
                velocity = velocity.detach().requires_grad_(True)
        _, gradient, damping_force, drive = self._residual(velocity, q, p, action, h, context)
        q_next, p_next = q + h * velocity, 2 * velocity - p
        dq, dp = q_next - q, p_next - p
        r_q = dq - h * velocity
        r_p = dp - h * (-gradient - damping_force + drive)
        dynamics_residual = torch.cat([r_q, r_p], -1).abs().amax(-1)
        work = h.squeeze(-1) * (drive * velocity).sum(-1)
        dissipation = h.squeeze(-1) * (damping_force * velocity).sum(-1)
        energy_delta = self.energy(q_next, p_next) - self.energy(q, p)
        chain_defect = energy_delta - (gradient * dq).sum(-1) - (velocity * dp).sum(-1)
        residual_work = (gradient * r_q).sum(-1) + (velocity * r_p).sum(-1)
        tolerance = self.residual_atol + self.residual_rtol * torch.maximum(
            p.abs().amax(-1), p_next.abs().amax(-1))
        chain_limit = self.chain_atol * (1 + self.energy(q, p).detach().abs()
                                         + self.energy(q_next, p_next).detach().abs())
        if not torch.isfinite(torch.cat([q_next, p_next], -1)).all():
            raise SolverError("Non-finite proposed state")
        if any(not torch.isfinite(value).all() for value in
               (gradient, damping_force, drive, work, dissipation, energy_delta,
                chain_defect, residual_work, dynamics_residual, tolerance, chain_limit)):
            raise SolverError("Non-finite implicit force, energy or numerical ledger")
        if self.strict and ((dynamics_residual.detach() > tolerance.detach()).any()
                            or (chain_defect.detach().abs() > chain_limit).any()):
            raise SolverError("Implicit step failed tolerance: "
                              f"dynamics_max={dynamics_residual.detach().max().item():.3g}, "
                              f"chain_max={chain_defect.detach().abs().max().item():.3g}")
        diagnostics = dict(work=work, dissipation=dissipation,
                           residual_work=residual_work, chain_defect=chain_defect,
                           numerical_balance_defect=residual_work + chain_defect,
                           chain_defect_max=chain_defect.abs(),
                           solver_residual=dynamics_residual, solver_iterations=iterations,
                           solver_residual_ratio=dynamics_residual / tolerance.clamp_min(torch.finfo(q.dtype).tiny),
                           solver_replay_iterations=iterations,
                           solver_backtracks=torch.zeros_like(iterations),
                           solver_line_search_evals=torch.zeros_like(iterations),
                           solver_failed_attempts=torch.zeros_like(iterations),
                           solver_leaf_steps=torch.ones_like(iterations),
                           solver_split_depth=torch.zeros_like(iterations))
        diagnostics["solver_replay_relative_error"] = torch.zeros_like(iterations)
        if not track_gradient:
            q_next, p_next = q_next.detach(), p_next.detach()
        return q_next, p_next, {key: value.detach() for key, value in diagnostics.items()}

    def _rhs(self, q, p, action, track_gradient, context=None):
        """Derivative of (q,p,W,D) for the ordinary-integrator control arm."""
        if not q.requires_grad:
            q = q.clone().requires_grad_(True)
        gradient = torch.autograd.grad(self.potential(q).sum(), q,
                                       create_graph=track_gradient)[0]
        force_d = (self.damping(q, p, context) @ p.unsqueeze(-1)).squeeze(-1)
        drive = (self.input_matrix(q) @ action.unsqueeze(-1)).squeeze(-1)
        result = (p, -gradient - force_d + drive,
                  (p * drive).sum(-1), (p * force_d).sum(-1))
        if not all(torch.isfinite(x).all() for x in result):
            raise SolverError("Non-finite RK4 stage derivative/power")
        return result if track_gradient else tuple(x.detach() for x in result)

    def _rk4_step(self, q, p, action, h, track_gradient, context=None):
        k1 = self._rhs(q, p, action, track_gradient, context)
        k2 = self._rhs(q + h * k1[0] / 2, p + h * k1[1] / 2, action, track_gradient, context)
        k3 = self._rhs(q + h * k2[0] / 2, p + h * k2[1] / 2, action, track_gradient, context)
        k4 = self._rhs(q + h * k3[0], p + h * k3[1], action, track_gradient, context)
        increment = tuple((a + 2 * b + 2 * c + d) / 6
                          for a, b, c, d in zip(k1, k2, k3, k4))
        q_next, p_next = q + h * increment[0], p + h * increment[1]
        work, dissipation = h.squeeze(-1) * increment[2], h.squeeze(-1) * increment[3]
        energy_delta = self.energy(q_next, p_next) - self.energy(q, p)
        defect = energy_delta - work + dissipation
        if not all(torch.isfinite(x).all() for x in (q_next, p_next, work, dissipation, defect)):
            raise SolverError("Non-finite RK4 state/energy accounting")
        zeros = torch.zeros_like(work)
        diagnostics = dict(work=work, dissipation=dissipation,
                           numerical_balance_defect=defect,
                           residual_work=zeros, chain_defect=zeros,
                           chain_defect_max=zeros, solver_residual=zeros,
                           solver_iterations=zeros, solver_replay_iterations=zeros,
                           solver_residual_ratio=zeros, solver_backtracks=zeros,
                           solver_line_search_evals=zeros, solver_failed_attempts=zeros,
                           solver_leaf_steps=torch.ones_like(zeros), solver_split_depth=zeros)
        diagnostics["solver_replay_relative_error"] = zeros
        if not track_gradient:
            q_next, p_next = q_next.detach(), p_next.detach()
        return q_next, p_next, {key: value.detach() for key, value in diagnostics.items()}

    def forward(self, q, p, action, dt, context=None):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Use torch.no_grad(), not inference_mode(), for the implicit core")
        if q.ndim != 2 or q.shape[-1] != self.q_dim or q.shape != p.shape:
            raise ValueError("q and p must have matching shape [B,q_dim]")
        if q.device.type not in {"cpu", "cuda"}:
            raise ValueError("This implicit core supports CPU/CUDA; MPS is not validated")
        if q.dtype not in {torch.float32, torch.float64} or q.dtype != self.factor_basis.dtype:
            raise ValueError("State and core dtype must match (float32 or float64)")
        if p.dtype != q.dtype or p.device != q.device or self.factor_basis.device != q.device:
            raise ValueError("States and core must share dtype/device")
        action = torch.as_tensor(action, dtype=q.dtype, device=q.device)
        if action.shape != (q.shape[0], self.action_dim):
            raise ValueError("action must have shape [B,action_dim]")
        dt = torch.as_tensor(dt, dtype=q.dtype, device=q.device)
        if dt.ndim == 0:
            dt = dt.expand(q.shape[0], 1)
        elif dt.shape == (q.shape[0],):
            dt = dt[:, None]
        if dt.shape != (q.shape[0], 1) or not torch.isfinite(dt).all() or (dt <= 0).any():
            raise ValueError("dt must be finite, positive, scalar/[B]/[B,1]")
        if any(not torch.isfinite(x).all() for x in (q, p, action)):
            raise ValueError("Inputs must be finite")
        context = self._context(q, context)
        track_gradient = torch.is_grad_enabled()
        h = dt / self.substeps
        energy_initial = self.energy(q, p).detach()
        if not torch.isfinite(energy_initial).all():
            raise SolverError("Non-finite initial energy")
        diagnostics = None
        with torch.enable_grad():
            for _ in range(self.substeps):
                advance = self._step if self.integrator == "discrete_gradient" else self._rk4_step
                q, p, current = advance(q, p, action, h, track_gradient, context)
                if diagnostics is None:
                    diagnostics = current
                else:
                    for key in current:
                        if key in self._MAX_DIAGNOSTICS:
                            diagnostics[key] = torch.maximum(diagnostics[key], current[key])
                        else:
                            diagnostics[key] = diagnostics[key] + current[key]
        with torch.no_grad():
            diagnostics["energy_delta"] = self.energy(q, p) - energy_initial
            diagnostics["energy_residual"] = (diagnostics["energy_delta"]
                                               - diagnostics["work"] + diagnostics["dissipation"])
            if (not torch.isfinite(torch.cat((q, p), -1)).all()
                    or any(not torch.isfinite(value).all() for value in diagnostics.values())):
                raise SolverError("Non-finite macro-step state or accumulated energy/solver ledger")
        return q, p, diagnostics
