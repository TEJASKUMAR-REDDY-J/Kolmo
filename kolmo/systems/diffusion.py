"""Heat diffusion dT/dt = kappa * laplacian(T) on a periodic square: the smallest complete Kolmo system.

Reference method: exact in time per Fourier mode (T_k(t + dt) = T_k(t) exp(-kappa |k|^2 dt)), so the only error is
float32 round-off. It exists to validate the shared interfaces with a second domain and to test surrogate and
evaluation code against a system whose answer is known exactly."""
import math

import jax
import jax.numpy as jnp
import numpy as np


class Diffusion:
    name = "diffusion"
    fields = ("temperature",)
    DEFAULTS = {"n": 64, "L": 2 * math.pi, "kappa": 0.01, "dt": 0.05, "modes": 6}

    def validate(self, params):
        p = {**self.DEFAULTS, **(params or {})}
        if int(p["n"]) < 8 or int(p["n"]) % 2:
            raise ValueError("n must be an even number >= 8")
        for k in ("L", "kappa", "dt"):
            if not float(p[k]) > 0:
                raise ValueError(f"{k} must be positive")
        p["n"], p["modes"] = int(p["n"]), int(p["modes"])
        return p

    def _k2(self, p):
        k = 2 * math.pi / p["L"] * np.fft.fftfreq(p["n"], 1.0 / p["n"])
        kx, ky = np.meshgrid(k, k, indexing="xy")
        return jnp.asarray(kx ** 2 + ky ** 2, jnp.float32)

    def initial(self, params, seed):
        p = self.validate(params)
        rng = np.random.default_rng(seed)
        x = np.arange(p["n"]) * p["L"] / p["n"]
        X, Y = np.meshgrid(x, x, indexing="xy")
        T = np.zeros_like(X)
        for _ in range(p["modes"]):  # a few random low modes
            kx, ky = rng.integers(1, 4, 2) * 2 * math.pi / p["L"]
            T += rng.normal() * np.cos(kx * X + rng.uniform(0, 2 * math.pi)) * np.cos(ky * Y + rng.uniform(0, 2 * math.pi))
        return jnp.asarray(T[None], jnp.float32)

    def stepper(self, params):
        p = self.validate(params)
        decay = jnp.exp(-p["kappa"] * self._k2(p) * p["dt"])
        return jax.jit(lambda s: jnp.fft.irfft2(jnp.fft.rfft2(s) * decay[:, : p["n"] // 2 + 1], s=s.shape[-2:]).astype(jnp.float32))

    def diagnostics(self, state, params):
        s = np.asarray(state[0], np.float64)
        return {"mean": float(s.mean()), "variance": float(s.var()), "max_abs": float(np.abs(s).max())}

    def residual(self, prev, nxt, params):
        """|| (nxt - prev)/dt - kappa lap((prev + nxt)/2) || / || (nxt - prev)/dt ||  (Crank-Nicolson form)."""
        p = self.validate(params)
        lap = lambda s: jnp.fft.irfft2(-self._k2(p)[:, : p["n"] // 2 + 1] * jnp.fft.rfft2(s), s=s.shape[-2:])
        dTdt = (nxt - prev) / p["dt"]
        r = dTdt - p["kappa"] * lap(0.5 * (prev + nxt))
        return float(jnp.linalg.norm(r) / (jnp.linalg.norm(dTdt) + 1e-30))


if __name__ == "__main__":  # exactness check: one mode decays at exp(-kappa k^2 t)
    sys_ = Diffusion(); p = sys_.validate({"n": 32, "kappa": 0.1, "dt": 0.1})
    x = np.arange(32) * p["L"] / 32
    s0 = jnp.asarray(np.cos(2 * x)[None, None, :].repeat(32, 1), jnp.float32)
    step = sys_.stepper(p); s = s0
    for _ in range(10):
        s = step(s)
    assert abs(float(s[0, 0, 0]) - math.exp(-0.1 * 4 * 1.0)) < 1e-5, float(s[0, 0, 0])
    assert sys_.residual(s0, step(s0), p) < 2e-2  # CN vs exact: O((k^2 kappa dt)^2)
    print("ok")
