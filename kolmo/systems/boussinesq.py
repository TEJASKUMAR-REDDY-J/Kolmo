"""2D Boussinesq convection, vorticity + temperature, two-way coupled, on a periodic square (free-fall units):

    dw/dt + u . grad w = nu lap w + d(theta)/dx,     nu    = sqrt(Pr / Ra)
    dtheta/dt + u . grad theta = kappa lap theta,    kappa = 1 / sqrt(Ra Pr)
    lap psi = -w, u = d psi/dy, v = -d psi/dx; gravity along -y, buoyancy +theta e_y

Units: length H (L is in units of H), temperature dT, velocity sqrt(g alpha dT H), time H / that velocity. Same
numerics as fluid2d (pseudo-spectral, 2/3 rule, IF-RK3 with diffusion exact) and optional Brinkman obstacle held at
temperature T_body (split substep, first order). See docs/physics.md."""
import math

import jax
import jax.numpy as jnp
import numpy as np

from kolmo.systems.common import grid, scan_rollout
from kolmo.systems.fluid2d import CFL_STABLE


class Boussinesq:
    name = "boussinesq"
    fields = ("vorticity", "temperature")
    DEFAULTS = {"n": 64, "L": 2 * math.pi, "Ra": 1e5, "Pr": 1.0, "dt": None, "cfl_max": 0.5,
                "ic": "thermal", "ic_amp": 1.0, "ic_r": 0.4, "ic_delta": 0.1, "ic_eps": 0.05, "ic_m": 1,
                "mask": None, "T_body": 1.0, "eta": 1e-3}
    INFO = {"n": (16, 512, "cells", "grid size"), "L": (0, None, "H", "box side"),
            "Ra": (0, None, "1", "Rayleigh number g alpha dT H^3 / (nu kappa)"), "Pr": (0, None, "1", "Prandtl number nu/kappa"),
            "dt": (0, None, "H/U_ff", "time step (None: from CFL)"), "cfl_max": (0, CFL_STABLE, "1", "CFL safety limit"),
            "ic": (None, None, "", "thermal | rayleigh_taylor | zero"), "ic_amp": (0, None, "dT", "IC temperature amplitude"),
            "ic_r": (0, None, "H", "thermal radius"), "ic_delta": (0, None, "H", "RT interface thickness"),
            "ic_eps": (0, None, "H", "RT interface displacement"), "ic_m": (1, None, "1", "RT perturbation mode"),
            "mask": (None, None, "", "(n, n) bool obstacle"), "T_body": (None, None, "dT", "obstacle temperature"),
            "eta": (0, None, "H/U_ff", "penalisation time")}

    def validate(self, params):
        p = {**self.DEFAULTS, **(params or {})}
        for k in ("nu", "kappa", "D", "cfl_estimate"):
            p.pop(k, None)
        n = p["n"] = int(p["n"])
        if n < 16 or n % 2:
            raise ValueError("n must be an even number >= 16")
        for k in ("L", "Ra", "Pr", "cfl_max", "ic_amp", "ic_r", "ic_delta", "ic_eps", "T_body", "eta"):
            p[k] = float(p[k])
        if not (p["L"] > 0 and p["Ra"] > 0 and p["Pr"] > 0 and p["eta"] > 0):
            raise ValueError("L, Ra, Pr and eta must be positive")
        if not 0 < p["cfl_max"] <= CFL_STABLE:
            raise ValueError(f"cfl_max must be in (0, {CFL_STABLE:.3f}]")
        if p["ic"] not in ("thermal", "rayleigh_taylor", "zero"):
            raise ValueError("ic must be 'thermal', 'rayleigh_taylor' or 'zero'")
        p["ic_m"] = int(p["ic_m"])
        p["nu"], p["kappa"] = math.sqrt(p["Pr"] / p["Ra"]), 1 / math.sqrt(p["Ra"] * p["Pr"])
        if p["mask"] is not None:
            p["mask"] = np.asarray(p["mask"], bool)
            if p["mask"].shape != (n, n) or not p["mask"].any() or p["mask"].all():
                raise ValueError(f"mask must be a non-empty, non-full ({n}, {n}) bool array")
        dx = p["L"] / n
        # free-fall speed estimate sqrt(dT * size), doubled: heuristic, diagnostics()["cfl"] is the real value
        A = max(abs(p["ic_amp"]) if p["ic"] != "zero" else 0.0, abs(p["T_body"]) if p["mask"] is not None else 0.0)
        S = 2 * math.sqrt(A * p["L"] / 4) if A else 0.0
        dt_max = p["cfl_max"] * dx / S if S else 0.1
        if p["dt"] is None:
            m = 10 ** math.floor(math.log10(dt_max) - 1)
            p["dt"] = math.floor(dt_max / m) * m
        p["dt"] = float(p["dt"])
        if not p["dt"] > 0:
            raise ValueError("dt must be positive")
        if p["dt"] > dt_max * (1 + 1e-9):
            raise ValueError(f"dt={p['dt']:.4g} violates the CFL limit: estimated max(|u|+|v|)={S:.4g}, dx={dx:.4g}, "
                             f"CFL={p['dt'] * S / dx:.3g} > cfl_max={p['cfl_max']}; use dt <= {dt_max:.4g}")
        p["cfl_estimate"] = p["dt"] * S / dx
        return p

    def _setup(self, p):
        g = {k: jnp.asarray(v, jnp.float32) for k, v in grid(p["n"], p["L"]).items()}
        g["chi"] = jnp.asarray(p["mask"] if p["mask"] is not None else np.zeros((p["n"],) * 2), jnp.float32)
        g["lam"] = -jnp.stack([p["nu"] * g["k2"], p["kappa"] * g["k2"]])
        return g

    @staticmethod
    def _irf(a):
        return jnp.fft.irfft2(a, s=(a.shape[-2], a.shape[-2]))

    def _vel(self, wh, g):
        ph = wh * g["inv_k2"]
        return self._irf(1j * g["ky"] * ph), self._irf(-1j * g["kx"] * ph)

    def _N(self, vh, g):
        """Spectral explicit terms of both equations: advection (dealiased) and the buoyancy torque d theta/dx."""
        u, v = self._vel(vh[0], g)
        adv = lambda a: jnp.fft.rfft2(u * self._irf(1j * g["kx"] * a) + v * self._irf(1j * g["ky"] * a))
        return jnp.stack([-adv(vh[0]) + 1j * g["kx"] * vh[1], -adv(vh[1])]) * g["dealias"]

    def _split(self, vh, p, g):
        """Exact penalisation over dt: velocity -> 0 and theta -> T_body inside the mask."""
        u, v = self._vel(vh[0], g)
        damp = 1 - g["chi"] * (1 - math.exp(-p["dt"] / p["eta"]))
        wh = 1j * g["kx"] * jnp.fft.rfft2(v * damp) - 1j * g["ky"] * jnp.fft.rfft2(u * damp)
        th = p["T_body"] + (self._irf(vh[1]) - p["T_body"]) * damp
        return jnp.stack([wh, jnp.fft.rfft2(th)]) * g["dealias"]

    def _step_fn(self, p):
        g, h = self._setup(p), p["dt"]
        E1, E2 = jnp.exp(g["lam"] * h), jnp.exp(g["lam"] * h / 2)
        obst = p["mask"] is not None

        def step(s):
            vh = jnp.fft.rfft2(s) * g["dealias"]
            if obst:
                vh = self._split(vh, p, g)
            k1 = self._N(vh, g)
            k2 = self._N(E2 * (vh + 0.5 * h * k1), g)
            k3 = self._N(E1 * (vh - h * k1) + 2 * h * E2 * k2, g)
            vh = (E1 * (vh + h / 6 * k1) + h / 6 * (4 * E2 * k2 + k3)) * g["dealias"]
            return self._irf(vh).astype(jnp.float32)
        return step

    def initial(self, params, seed):
        """thermal: Gaussian warm blob amp*exp(-r^2/(2 ic_r^2)) at (L/2, L/4); rayleigh_taylor: theta =
        amp*tanh(sin(2 pi (y + eps cos(2 pi m x / L)) / L) L / (2 pi delta)), unstable (warm below cold) interface
        at y = L/2, stable one at y = 0; zero. Deterministic (seed unused: the ICs have no random part)."""
        p = self.validate(params)
        n, L = p["n"], p["L"]
        x = np.arange(n) * L / n
        X, Y = np.meshgrid(x, x, indexing="xy")
        if p["ic"] == "thermal":
            th = p["ic_amp"] * np.exp(-((X - L / 2) ** 2 + (Y - L / 4) ** 2) / (2 * p["ic_r"] ** 2))
        elif p["ic"] == "rayleigh_taylor":
            yy = Y + p["ic_eps"] * np.cos(2 * np.pi * p["ic_m"] * X / L)
            th = p["ic_amp"] * np.tanh(np.sin(2 * np.pi * yy / L) * L / (2 * np.pi * p["ic_delta"]))
        else:
            th = np.zeros((n, n))
        return jnp.asarray(np.stack([np.zeros((n, n)), th]), jnp.float32)

    def stepper(self, params):
        return jax.jit(self._step_fn(self.validate(params)))

    def batched_stepper(self, params):
        return jax.jit(jax.vmap(self._step_fn(self.validate(params))))

    def rollout(self, params, state0, steps, every=1):
        return scan_rollout(self._step_fn(self.validate(params)), state0, steps, every)

    def diagnostics(self, state, params):
        """Box means, density 1: energy 0.5<u^2+v^2>, enstrophy 0.5<w^2>, buoyancy_flux <v theta> (energy input),
        theta_mean, theta_var 0.5<theta^2>, thermal_dissipation kappa<|grad theta|^2>, max_abs_vorticity, cfl,
        centroid_y = sum(theta+ y)/sum(theta+) (warm-fluid height, no periodic unwrapping)."""
        p = self.validate(params)
        g = self._setup(p)
        s = jnp.asarray(state, jnp.float32)
        vh = jnp.fft.rfft2(s)
        u, v = self._vel(vh[0], g)
        th = s[1]
        tx, ty = self._irf(1j * g["kx"] * vh[1]), self._irf(1j * g["ky"] * vh[1])
        y = jnp.arange(p["n"])[:, None] * p["L"] / p["n"]
        tp = jnp.maximum(th, 0)
        d = {"energy": 0.5 * jnp.mean(u ** 2 + v ** 2), "enstrophy": 0.5 * jnp.mean(s[0] ** 2),
             "buoyancy_flux": jnp.mean(v * th), "theta_mean": jnp.mean(th), "theta_var": 0.5 * jnp.mean(th ** 2),
             "thermal_dissipation": p["kappa"] * jnp.mean(tx ** 2 + ty ** 2),
             "max_abs_vorticity": jnp.max(jnp.abs(s[0])),
             "cfl": p["dt"] * jnp.max(jnp.abs(u) + jnp.abs(v)) * p["n"] / p["L"],
             "centroid_y": jnp.sum(tp * y) / (jnp.sum(tp) + 1e-30)}
        return {k: float(val) for k, val in d.items()}

    def residual(self, prev, nxt, params):
        """Variation-of-constants residual of both equations over one step (same definition as kolmo.systems.pde):
        R = (v1 - E v~)/dt - [phi1 N(v~) + phi2 (N(v1) - N(v~))], v~ = penalisation substep of v0 (identity without
        mask), r = ||R|| / (||(v1 - v0)/dt|| + ||(v~ - v0)/dt|| + ||0.5 (N~ + N1)|| + ||0.5 L (v~ + v1)||)."""
        p = self.validate(params)
        g = self._setup(p)
        v0, v1 = (jnp.fft.rfft2(jnp.asarray(s, jnp.float32)) * g["dealias"] for s in (prev, nxt))
        vt = self._split(v0, p, g) if p["mask"] is not None else v0
        z = g["lam"] * p["dt"]
        zs = jnp.where(z == 0, 1.0, z)
        phi1 = jnp.where(z == 0, 1.0, jnp.expm1(z) / zs)
        phi2 = jnp.where(z == 0, 0.5, (jnp.expm1(z) - z) / zs ** 2)
        Nt, N1 = self._N(vt, g), self._N(v1, g)
        R = (v1 - jnp.exp(z) * vt) / p["dt"] - (phi1 * Nt + phi2 * (N1 - Nt))
        nrm = lambda a: jnp.linalg.norm(a.ravel())
        den = nrm((v1 - v0) / p["dt"]) + nrm((vt - v0) / p["dt"]) + nrm(0.5 * (Nt + N1)) + nrm(0.5 * g["lam"] * (vt + v1))
        return float(nrm(R) / (den + 1e-30))

    def describe(self):
        params = {k: {"default": None if k == "mask" else self.DEFAULTS[k], "min": a, "max": b, "unit": u, "help": h}
                  for k, (a, b, u, h) in self.INFO.items()}
        return {"name": self.name, "fields": list(self.fields),
                "equations": "dw/dt + u.grad w = sqrt(Pr/Ra) lap w + dtheta/dx; dtheta/dt + u.grad theta = "
                             "lap theta / sqrt(Ra Pr); lap psi = -w (free-fall units, buoyancy +theta e_y)",
                "params": params,
                "checks": ["dE/dt = <v theta> - 2 nu Z", "d(0.5<theta^2>)/dt = -kappa <|grad theta|^2>",
                           "mean temperature conserved (no obstacle)"]}
