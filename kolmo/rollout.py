"""Run a Stepper (reference solver or surrogate) for n steps and return a core.Trajectory with failure events.

Timing: `compile_seconds` is the wall time of one warm-up call (compilation + one step, result discarded);
`seconds` is the wall time of the n recorded steps afterwards (steady state, synchronised at the end).
Failure checks run after the loop on the stored states so they never perturb the timing."""
import time

import jax
import jax.numpy as jnp
import numpy as np

from kolmo.core import Stepper, Trajectory


def coarse(step: Stepper, k: int) -> Stepper:
    """One surrogate step = k solver steps, fused into one jitted call."""
    if k == 1:
        return step
    return jax.jit(lambda s: jax.lax.fori_loop(0, k, lambda _, x: step(x), s))


def mean_square(states) -> np.ndarray:
    """Mean of state**2 over (C, H, W), float64. The generic 'energy' used for drift checks: for a temperature field
    it is the mean square temperature, for vorticity it is twice the enstrophy density. Units: field units squared."""
    s = np.asarray(states, np.float64)
    return np.mean(s * s, axis=(-3, -2, -1))


def detect_failures(states, ref_scale=None, blowup=10.0, ref_energy=None, energy_tol=None) -> list[dict]:
    """First step of each failure kind in `states` (T+1, C, H, W):
    - "nonfinite": any NaN/Inf;
    - "blowup": max|state| > blowup * ref_scale (ref_scale defaults to max|states[0]|);
    - "energy_drift": |E_t / E_ref,t - 1| > energy_tol, E = mean_square, ref_energy scalar or per-step array
      (only checked when both ref_energy and energy_tol are given)."""
    s = np.asarray(states)
    finite = np.isfinite(s).all(axis=(1, 2, 3))
    scale = float(np.abs(s[0]).max()) if ref_scale is None else float(ref_scale)
    with np.errstate(invalid="ignore", over="ignore"):
        peak = np.abs(np.where(np.isfinite(s), s, 0)).max(axis=(1, 2, 3))
        checks = {"nonfinite": ~finite, "blowup": finite & (peak > blowup * scale)}
        if ref_energy is not None and energy_tol is not None:
            drift = mean_square(s) / np.asarray(ref_energy, np.float64) - 1
            checks["energy_drift"] = ~(np.abs(drift) <= energy_tol)
    events = []
    for kind, bad in checks.items():
        if bad.any():
            t = int(np.argmax(bad))
            detail = {"nonfinite": "NaN/Inf in state", "blowup": f"max|s|={peak[t]:.3g} > {blowup}x{scale:.3g}",
                      "energy_drift": f"|E/E_ref-1| > {energy_tol}"}[kind]
            events.append({"step": t, "kind": "failure", "reason": kind, "detail": detail})
    return sorted(events, key=lambda e: e["step"])


def rollout(step: Stepper, s0, n: int, source="reference", **failure_kw) -> Trajectory:
    """States 0..n from s0 under `step`; failure events via detect_failures(**failure_kw)."""
    s0 = jnp.asarray(s0, jnp.float32)
    t = time.perf_counter()
    jax.block_until_ready(step(s0))
    compile_s = time.perf_counter() - t
    out, s = [s0], s0
    t = time.perf_counter()
    for _ in range(n):
        s = step(s)
        out.append(s)
    jax.block_until_ready(s)
    secs = time.perf_counter() - t
    states = np.asarray(jnp.stack(out))
    return Trajectory(states, [source] * n, detect_failures(states, **failure_kw), secs, compile_s)
