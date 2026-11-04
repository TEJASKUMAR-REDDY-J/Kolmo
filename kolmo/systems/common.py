"""Shared helpers of the spectral systems: periodic grid, 2/3-rule mask, lax.scan rollout with compile timing."""
import time

import jax
import jax.numpy as jnp
import numpy as np

from kolmo.core import Trajectory


def grid(n, L):
    """rfft2 wavenumbers on an n x n periodic grid of side L (row = y, column = x) and the 2/3-rule mask."""
    K = (n - 1) // 3
    mx, my = np.fft.rfftfreq(n, 1 / n)[None, :], np.fft.fftfreq(n, 1 / n)[:, None]
    kx, ky = 2 * np.pi / L * mx, 2 * np.pi / L * my
    k2 = kx ** 2 + ky ** 2
    return {"kx": kx, "ky": ky, "k2": k2, "inv_k2": np.where(k2 > 0, 1 / np.where(k2 > 0, k2, 1), 0),
            "dealias": ((np.abs(mx) <= K) & (np.abs(my) <= K)).astype(np.float32)}


def scan_rollout(step, state0, steps, every=1):
    """Run `steps` steps of a pure step s -> s in one lax.scan, keeping every `every`-th state. A 4-D state0
    (B, C, H, W) is vmapped. compile_seconds = XLA compile of the scan, seconds = run of the compiled program."""
    state0 = jnp.asarray(state0, jnp.float32)
    fn = jax.vmap(step) if state0.ndim == 4 else step

    def run(s):
        def body(s, _):
            s = jax.lax.fori_loop(0, every, lambda i, x: fn(x), s)
            return s, s
        return jax.lax.scan(body, s, None, length=steps // every)[1]

    t0 = time.perf_counter()
    compiled = jax.jit(run).lower(state0).compile()
    t1 = time.perf_counter()
    traj = compiled(state0).block_until_ready()
    t2 = time.perf_counter()
    states = np.concatenate([np.asarray(state0)[None], np.asarray(traj)])
    events = [] if np.isfinite(states[-1]).all() else [{"step": steps, "kind": "nonfinite", "reason": "blow-up"}]
    return Trajectory(states=states, source=["reference"] * (len(states) - 1), events=events,
                      seconds=t2 - t1, compile_seconds=t1 - t0)
