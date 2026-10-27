"""Execution modes: "reference" (solver only), "surrogate" (network only), "hybrid" (network, with solver interventions).

All modes advance in surrogate steps (one surrogate step = `spacing` solver steps). An intervention replaces the
surrogate's proposal for that step by `spacing` solver steps taken FROM THE CURRENT HYBRID STATE (the solver never
sees the true trajectory, so cost accounting is honest). Triggers, checked in this order:
  - "every": steps t = every, 2 every, ... are always solver steps (fixed re-anchor schedule);
  - "residual": system.residual(s, proposal, params with dt -> spacing * dt) > residual_tol;
  - "energy": |mean_square(proposal) / energy_ref - 1| > energy_tol (energy_ref defaults to mean_square(s0));
  - "nonfinite": the proposal contains NaN/Inf (always on in hybrid mode).
Each intervention is a Trajectory event {"step", "kind": "intervention", "reason", "value"}. Parameters are never
changed. Cost = solver steps used = (number of "reference" steps) * spacing; see cost()."""
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from kolmo.core import Trajectory
from kolmo.rollout import coarse, detect_failures, mean_square, rollout


@dataclass
class HybridPolicy:
    mode: str = "hybrid"  # "reference" | "surrogate" | "hybrid"
    every: int = 0  # fixed re-anchor period in surrogate steps (0 = off)
    residual_tol: float | None = None
    energy_tol: float | None = None
    energy_ref: float | None = None

    def validate(self):
        if self.mode not in ("reference", "surrogate", "hybrid"):
            raise ValueError("mode must be reference, surrogate or hybrid")
        if self.every < 0:
            raise ValueError("every must be >= 0")
        return self

    @property
    def name(self):
        if self.mode != "hybrid":
            return self.mode
        parts = [f"every{self.every}"] if self.every else []
        parts += [f"res>{self.residual_tol:g}"] if self.residual_tol is not None else []
        parts += [f"E>{self.energy_tol:g}"] if self.energy_tol is not None else []
        return "hybrid[" + ",".join(parts or ["nonfinite"]) + "]"


def cost(traj: Trajectory, spacing: int) -> dict:
    """Solver steps used, and as a fraction of a pure reference run of the same length."""
    n_ref = traj.source.count("reference")
    return {"solver_steps": n_ref * spacing, "fraction_of_reference": n_ref / max(len(traj.source), 1),
            "interventions": sum(e["kind"] == "intervention" for e in traj.events)}


def run(policy: HybridPolicy, system, params, surrogate, s0, n, **failure_kw) -> Trajectory:
    """n surrogate steps from s0 under `policy`. `surrogate` is a models.Surrogate (unused in reference mode)."""
    policy.validate()
    p = system.validate(params)
    k = surrogate.spacing
    solver = coarse(system.stepper(p), k)
    if policy.mode == "reference":
        return rollout(solver, s0, n, "reference", **failure_kw)
    net = surrogate.stepper(p)
    if policy.mode == "surrogate":
        return rollout(net, s0, n, "surrogate", **failure_kw)

    p_coarse = {**p, "dt": p["dt"] * k}  # residual over one surrogate step
    s = jnp.asarray(s0, jnp.float32)
    e_ref = policy.energy_ref if policy.energy_ref is not None else float(mean_square(s0))
    t = time.perf_counter()
    jax.block_until_ready((solver(s), net(s)))
    compile_s = time.perf_counter() - t
    out, source, events = [s], [], []
    t = time.perf_counter()
    for step in range(1, n + 1):
        reason, value = None, None
        if policy.every and step % policy.every == 0:
            reason = "every"
        else:
            prop = net(s)
            if not bool(jnp.isfinite(prop).all()):
                reason = "nonfinite"
            elif policy.residual_tol is not None:
                value = float(system.residual(s, prop, p_coarse))
                reason = "residual" if not value <= policy.residual_tol else None
            if reason is None and policy.energy_tol is not None:
                value = float(mean_square(prop) / e_ref - 1)
                reason = "energy" if not abs(value) <= policy.energy_tol else None
        if reason:
            s = solver(s)
            source.append("reference")
            events.append({"step": step, "kind": "intervention", "reason": reason, "value": value})
        else:
            s = prop
            source.append("surrogate")
        out.append(s)
    jax.block_until_ready(s)
    secs = time.perf_counter() - t
    states = np.asarray(jnp.stack(out))
    return Trajectory(states, source, events + detect_failures(states, **failure_kw), secs, compile_s)
