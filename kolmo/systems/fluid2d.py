"""2D incompressible Navier-Stokes in vorticity form on a periodic square: the Kolmo reference fluid solver.

    dw/dt + (U_inf e_x + u) . grad w = nu lap w - mu w + f  [+ penalisation and sponge, applied as a split substep]
    lap psi = -w,   u = d psi/dy,   v = -d psi/dx,   w = dv/dx - du/dy

Pseudo-spectral (rfft2) with the 2/3 rule; time stepping is integrating-factor RK3 (Kutta's 3-stage scheme): viscosity
and drag are integrated exactly per Fourier mode, advection and forcing explicitly, third order in time. Obstacles use
Brinkman volume penalisation solved exactly over each step (Lie splitting, first order). Equations, units, stability
limits and measured accuracy/performance are in docs/physics.md."""
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from kolmo.core import Trajectory

CFL_STABLE = 3 * math.sqrt(3) / (2 * math.pi)  # IF-RK3 + 2/3 dealiasing linear advection bound on dt*max(|u|+|v|)/dx
DYNAMIC = ("nu", "mu", "A", "dt", "U_inf", "eta", "sponge_rate", "chi")  # may differ across a batch (vmapped)
STATIC = ("n", "L", "kf", "sponge")  # must be equal across a batch


class Fluid2D:
    name = "fluid2d"
    fields = ("vorticity",)
    DEFAULTS = {
        "n": 64, "L": 2 * math.pi,
        "nu": None, "Re": None,  # give one; nu = 1e-3 when neither is given (Re definitions: docs/physics.md)
        "mu": 0.0,  # linear drag
        "A": 0.0, "kf": 4,  # Kolmogorov body force f_x = -A sin(k y), k = 2 pi kf / L  ->  f_w = A k cos(k y)
        "dt": None, "cfl_max": 0.5,  # dt = None picks the largest dt with estimated CFL <= cfl_max
        "ic": None, "ic_amp": 1.0, "ic_k": (1, 6), "tg_k": 1,  # ic: "random" | "taylor_green" | "zero"
        "mask": None, "U_inf": 0.0, "eta": 1e-3,  # obstacle (H, W) bool, free stream along +x, penalisation time
        "sponge": None, "sponge_rate": None,  # fringe strip width (fraction of L) at the right edge, rate 1/time
    }

    # ------------------------------------------------------------------ parameters
    def validate(self, params):
        p = {**self.DEFAULTS, **(params or {})}
        n = p["n"] = int(p["n"])
        if n < 16 or n % 2:
            raise ValueError("n must be an even number >= 16")
        L = p["L"] = float(p["L"])
        if not L > 0:
            raise ValueError("L must be positive")
        dx, K = L / n, (n - 1) // 3  # K: largest mode kept by the 2/3 rule
        for k in ("mu", "ic_amp", "U_inf", "A", "cfl_max", "eta"):
            p[k] = float(p[k])
        if p["mu"] < 0 or p["ic_amp"] < 0 or p["U_inf"] < 0:
            raise ValueError("mu, ic_amp and U_inf must be >= 0")
        if not 0 < p["cfl_max"] <= CFL_STABLE:
            raise ValueError(f"cfl_max must be in (0, {CFL_STABLE:.3f}] (linear stability bound of IF-RK3)")
        if not p["eta"] > 0:
            raise ValueError("eta must be positive")
        p["kf"], p["tg_k"] = int(p["kf"]), int(p["tg_k"])
        lo, hi = p["ic_k"] = tuple(int(v) for v in p["ic_k"])
        if not (1 <= p["kf"] <= K and 1 <= p["tg_k"] <= K and 1 <= lo <= hi <= K):
            raise ValueError(f"kf, tg_k and ic_k must lie in [1, {K}] (2/3-rule band for n={n})")

        mask = p["mask"]
        if mask is not None:
            mask = p["mask"] = np.asarray(mask, bool)
            if mask.shape != (n, n):
                raise ValueError(f"mask must have shape ({n}, {n}), got {mask.shape}")
            if not mask.any() or mask.all():
                raise ValueError("mask must be neither empty nor full")
        p["D"] = float(mask.any(axis=1).sum() * dx) if mask is not None else None  # cross-stream extent
        p["ic"] = p["ic"] or ("zero" if mask is not None else "random")
        if p["ic"] not in ("random", "taylor_green", "zero"):
            raise ValueError("ic must be 'random', 'taylor_green' or 'zero'")

        # viscosity from nu or Re = U_ref L_ref / nu
        k = 2 * math.pi * p["kf"] / L
        if mask is not None:
            U_ref, L_ref = p["U_inf"], p["D"]
        elif p["A"] != 0:
            U_ref, L_ref = math.sqrt(abs(p["A"]) / k), 1 / k
        else:
            U_ref, L_ref = p["ic_amp"], L / (2 * math.pi)
        nu, Re = p["nu"], p["Re"]
        if nu is None and Re is None:
            nu = 1e-3
        elif nu is None:
            if not (float(Re) > 0 and U_ref > 0):
                raise ValueError("Re must be positive and needs a nonzero reference velocity (U_inf, A or ic_amp)")
            nu = U_ref * L_ref / float(Re)
        nu = p["nu"] = float(nu)
        if not nu > 0:
            raise ValueError("nu must be positive")
        if Re is not None and U_ref > 0 and abs(U_ref * L_ref / nu / float(Re) - 1) > 1e-6:
            raise ValueError(f"nu={nu} and Re={Re} disagree (Re = U_ref L_ref / nu = {U_ref * L_ref / nu:.6g})")
        p["Re"] = U_ref * L_ref / nu if U_ref > 0 else math.inf

        # CFL: dt * max(|u| + |v|) / dx <= cfl_max, with max(|u|+|v|) estimated a priori
        S = p["U_inf"] * (2 if mask is not None else 1)  # flow speeds up around a body
        S += {"random": 4.0, "taylor_green": 1.0, "zero": 0.0}[p["ic"]] * p["ic_amp"]  # random: peak ~ 4 x rms
        if p["A"]:  # laminar amplitude, capped by a multiple of the turbulent scale sqrt(A/k) (heuristic)
            S += min(abs(p["A"]) / (nu * k * k + p["mu"]), 8 * math.sqrt(abs(p["A"]) / k))
        dt_max = p["cfl_max"] * dx / S if S > 0 else 0.1
        if p["dt"] is None:
            m = 10 ** math.floor(math.log10(dt_max) - 1)
            p["dt"] = math.floor(dt_max / m) * m  # rounded down to 2 significant digits
        p["dt"] = float(p["dt"])
        if not p["dt"] > 0:
            raise ValueError("dt must be positive")
        if p["dt"] > dt_max * (1 + 1e-9):
            raise ValueError(
                f"dt={p['dt']:.4g} violates the CFL limit: estimated max(|u|+|v|)={S:.4g}, dx={dx:.4g}, "
                f"CFL={p['dt'] * S / dx:.3g} > cfl_max={p['cfl_max']}; use dt <= {dt_max:.4g}")
        p["cfl_estimate"] = p["dt"] * S / dx

        if p["sponge"] is None:
            p["sponge"] = 0.1 if mask is not None else 0.0
        p["sponge"] = float(p["sponge"])
        if not 0 <= p["sponge"] < 0.5:
            raise ValueError("sponge (width as a fraction of L) must be in [0, 0.5)")
        if p["sponge_rate"] is None:  # ~5 e-folds of damping for vorticity crossing the strip at U_inf
            p["sponge_rate"] = 10 * max(p["U_inf"], 1.0) / (p["sponge"] * L) if p["sponge"] else 0.0
        p["sponge_rate"] = float(p["sponge_rate"])
        return p

    # ------------------------------------------------------------------ spectral setup
    def _setup(self, p):
        """Static grid arrays g (depend on n, L, kf, sponge) and dynamic float32 parameters q (vmappable)."""
        n, L = p["n"], p["L"]
        K = (n - 1) // 3
        mx, my = np.fft.rfftfreq(n, 1 / n)[None, :], np.fft.fftfreq(n, 1 / n)[:, None]
        kx, ky = 2 * np.pi / L * mx, 2 * np.pi / L * my
        k2 = kx ** 2 + ky ** 2
        x = np.arange(n) * L / n
        k = 2 * np.pi * p["kf"] / L
        s = (x / L - (1 - p["sponge"])) / max(p["sponge"], 1e-30)  # 0..1 across the strip at the right edge
        g = {
            "kx": kx, "ky": ky, "k2": k2, "inv_k2": np.where(k2 > 0, 1 / np.where(k2 > 0, k2, 1), 0),
            "dealias": ((np.abs(mx) <= K) & (np.abs(my) <= K)).astype(np.float32),
            "f_hat": k * np.fft.rfft2(np.cos(k * x)[:, None] * np.ones((1, n))),  # times A: rfft2 of A k cos(k y)
            "fx_unit": -np.sin(k * x)[:, None] * np.ones((1, n)),  # times A: velocity-form force f_x
            "sponge": np.where((s >= 0) & (s <= 1), 0.5 * (1 - np.cos(2 * np.pi * s)), 0.0)[None, :] * np.ones((n, 1)),
        }
        g = {key: jnp.asarray(v, jnp.complex64 if np.iscomplexobj(v) else jnp.float32) for key, v in g.items()}
        q = {key: jnp.float32(p[key]) for key in DYNAMIC if key != "chi"}
        q["chi"] = jnp.asarray(p["mask"] if p["mask"] is not None else np.zeros((n, n)), jnp.float32)
        split = p["mask"] is not None or p["sponge"] > 0
        return g, q, split

    @staticmethod
    def _irf(a):
        return jnp.fft.irfft2(a, s=(a.shape[-2], a.shape[-2]))

    def _velocity(self, wh, g):
        """Physical velocity (u, v) of the periodic part (zero mean; U_inf excluded)."""
        ph = wh * g["inv_k2"]
        return self._irf(1j * g["ky"] * ph), self._irf(-1j * g["kx"] * ph)

    def _advection(self, wh, q, g):
        """Spectral -(U_inf + u, v) . grad w, 2/3-dealiased (wh must already be band-limited)."""
        u, v = self._velocity(wh, g)
        wx, wy = self._irf(1j * g["kx"] * wh), self._irf(1j * g["ky"] * wh)
        return -jnp.fft.rfft2((u + q["U_inf"]) * wx + v * wy) * g["dealias"]

    def _split(self, wh, q, g):
        """Exact solution over dt of du/dt = -chi/eta (U_inf e_x + u) followed by dw/dt = -sigma(x) w, then curl
        and 2/3 filter. The mean flow U_inf is re-imposed afterwards (a uniform pressure gradient holds it)."""
        u, v = self._velocity(wh, g)
        damp = 1 - q["chi"] * (1 - jnp.exp(-q["dt"] / q["eta"]))
        wh = (1j * g["kx"] * jnp.fft.rfft2(v * damp) - 1j * g["ky"] * jnp.fft.rfft2((u + q["U_inf"]) * damp))
        w = self._irf(wh * g["dealias"]) * jnp.exp(-q["sponge_rate"] * g["sponge"] * q["dt"])
        return jnp.fft.rfft2(w) * g["dealias"]

    def _step(self, w, q, g, split):
        """One step state (1, n, n) -> (1, n, n): optional split penalisation/sponge, then IF-RK3 (Kutta)."""
        wh = jnp.fft.rfft2(w[0])
        if split:
            wh = self._split(wh, q, g)
        h, lam = q["dt"], -(q["nu"] * g["k2"] + q["mu"])
        E1, E2 = jnp.exp(lam * h), jnp.exp(lam * h / 2)
        f = q["A"] * g["f_hat"]
        N = lambda a: self._advection(a, q, g) + f
        k1 = N(wh)
        k2 = N(E2 * (wh + 0.5 * h * k1))
        k3 = N(E1 * (wh - h * k1) + 2 * h * E2 * k2)
        wh = (E1 * (wh + h / 6 * k1) + h / 6 * (4 * E2 * k2 + k3)) * g["dealias"]
        return self._irf(wh)[None].astype(jnp.float32)

    # ------------------------------------------------------------------ System protocol
    def initial(self, params, seed):
        p = self.validate(params)
        n, L = p["n"], p["L"]
        x = np.arange(n) * L / n
        X, Y = np.meshgrid(x, x, indexing="xy")
        if p["ic"] == "zero":
            w = np.zeros((n, n))
        elif p["ic"] == "taylor_green":  # u = U0 sin(kx) cos(ky), v = -U0 cos(kx) sin(ky)
            k = 2 * np.pi * p["tg_k"] / L
            w = 2 * k * p["ic_amp"] * np.sin(k * X) * np.sin(k * Y)
        else:  # white noise restricted to the integer-wavenumber shell lo <= |m| <= hi, scaled to rms |u| = ic_amp
            lo, hi = p["ic_k"]
            mx, my = np.fft.rfftfreq(n, 1 / n)[None, :], np.fft.fftfreq(n, 1 / n)[:, None]
            m = np.hypot(mx, my)
            wh = np.fft.rfft2(np.random.default_rng(seed).standard_normal((n, n))) * ((m >= lo) & (m <= hi))
            kx, ky = 2 * np.pi / L * mx, 2 * np.pi / L * my
            ph = wh / np.where(m > 0, kx ** 2 + ky ** 2, 1)
            u, v = np.fft.irfft2(1j * ky * ph, s=(n, n)), np.fft.irfft2(-1j * kx * ph, s=(n, n))
            w = np.fft.irfft2(wh, s=(n, n)) * p["ic_amp"] / max(np.sqrt(np.mean(u ** 2 + v ** 2)), 1e-300)
        return jnp.asarray(w[None], jnp.float32)

    def stepper(self, params):
        g, q, split = self._setup(self.validate(params))
        return jax.jit(lambda s: self._step(s, q, g, split))

    def batched_stepper(self, params, batch_params=None):
        """(B, 1, n, n) -> (B, 1, n, n). With batch_params (a list of B dicts overriding `params`), every DYNAMIC
        parameter (nu, mu, A, dt, U_inf, eta, sponge_rate, mask) may differ per member and is vmapped; STATIC ones
        (n, L, kf, sponge width) must agree. Without it, one parameter set is applied to a batch of states."""
        fn, q = self._batched_fn(params, batch_params)
        return jax.jit(lambda s: fn(s, q))

    def _batched_fn(self, params, batch_params):
        p = self.validate(params)
        g, q, split = self._setup(p)
        if not batch_params:
            return jax.vmap(lambda s, q: self._step(s, q, g, split), in_axes=(0, None)), q
        ps = [self.validate({**params, **bp}) for bp in batch_params]
        for pb in ps:
            if any(pb[k] != p[k] for k in STATIC) or (pb["mask"] is None) != (p["mask"] is None):
                raise ValueError(f"batch members must share {STATIC} and the presence of a mask")
        qs = [self._setup(pb)[1] for pb in ps]
        q = {k: jnp.stack([qb[k] for qb in qs]) for k in q}
        return jax.vmap(lambda s, q: self._step(s, q, g, split)), q

    def rollout(self, params, state0, steps, every=1, batch_params=None):
        """Advance `steps` steps inside one lax.scan, keeping every `every`-th state. state0 is (1, n, n), or
        (B, 1, n, n) for a batch (then Trajectory.states is (T+1, B, 1, n, n)). compile_seconds is the XLA
        compilation of the whole scan; seconds is the steady-state run of the compiled program."""
        state0 = jnp.asarray(state0, jnp.float32)
        if state0.ndim == 4:
            fn, q = self._batched_fn(params, batch_params)
        else:
            g, q, split = self._setup(self.validate(params))
            fn = lambda s, q: self._step(s, q, g, split)

        def run(s, q):
            def body(s, _):
                s = jax.lax.fori_loop(0, every, lambda i, x: fn(x, q), s)
                return s, s
            return jax.lax.scan(body, s, None, length=steps // every)[1]

        t0 = time.perf_counter()
        compiled = jax.jit(run).lower(state0, q).compile()
        t1 = time.perf_counter()
        traj = compiled(state0, q).block_until_ready()
        t2 = time.perf_counter()
        states = np.concatenate([np.asarray(state0)[None], np.asarray(traj)])
        events = [] if np.isfinite(states[-1]).all() else [{"step": steps, "kind": "nonfinite", "reason": "blow-up"}]
        if state0.ndim == 3 and not events:  # the a-priori CFL estimate of validate() can be beaten by the flow
            cfl = self.diagnostics(states[-1], params)["cfl"]
            if cfl > CFL_STABLE:
                events.append({"step": steps, "kind": "cfl", "reason": f"final CFL {cfl:.3g} > {CFL_STABLE:.3g}"})
        return Trajectory(states=states, source=["reference"] * (len(states) - 1), events=events,
                          seconds=t2 - t1, compile_seconds=t1 - t0)

    def diagnostics(self, state, params):
        """Per-area means <.> over the box; density 1. energy = 0.5 <u^2 + v^2> of the periodic part (U_inf
        excluded), enstrophy = 0.5 <w^2>, forcing_power = <u f_x>, cfl = dt max(|U_inf + u| + |v|) / dx.
        With a mask: drag/lift = force of the fluid on the body per unit depth, (1 - exp(-dt/eta))/dt *
        integral(chi (U_inf + u, v)) dA, i.e. the momentum the next penalisation substep removes, per unit time;
        cd/cl = force / (0.5 U_inf^2 D) with D the mask's cross-stream extent."""
        p = self.validate(params)
        g, q, _ = self._setup(p)
        w = jnp.asarray(state, jnp.float32)[0]
        u, v = self._velocity(jnp.fft.rfft2(w), g)
        dx = p["L"] / p["n"]
        d = {
            "energy": 0.5 * jnp.mean(u ** 2 + v ** 2), "enstrophy": 0.5 * jnp.mean(w ** 2),
            "max_abs_vorticity": jnp.max(jnp.abs(w)), "mean_vorticity": jnp.mean(w),
            "forcing_power": p["A"] * jnp.mean(u * g["fx_unit"]),
            "cfl": p["dt"] * jnp.max(jnp.abs(u + p["U_inf"]) + jnp.abs(v)) / dx,
        }
        if p["mask"] is not None:
            c = (1 - math.exp(-p["dt"] / p["eta"])) / p["dt"] * dx * dx
            d["drag"], d["lift"] = c * jnp.sum(q["chi"] * (u + p["U_inf"])), c * jnp.sum(q["chi"] * v)
            if p["U_inf"] > 0:
                d["cd"], d["cl"] = (d[k] / (0.5 * p["U_inf"] ** 2 * p["D"]) for k in ("drag", "lift"))
        return {k: float(v) for k, v in d.items()}

    def residual(self, prev, nxt, params):
        """Relative residual of the vorticity equation over one step, in variation-of-constants form (linear terms
        exact, nonlinear and forcing terms by the trapezoid rule), per Fourier mode:

            R = [w1 - E w~] / dt - 0.5 [E N(w~) + N(w1)],   E = exp(-(nu |k|^2 + mu) dt),   w~ = S(w0)
            N(w) = -(U_inf e_x + u) . grad w + f      (spectral, 2/3-dealiased)
            r = ||R|| / (||Dw|| + ||P|| + sum_T ||0.5 [T(w~) + T(w1)]||)
            Dw = (w1 - w0)/dt,  P = (w~ - w0)/dt,  T in {advection, viscous nu lap w, drag -mu w, forcing}

        S is the split penalisation/sponge substep (identity without mask and sponge), so P is the penalisation
        and sponge contribution exactly as the reference scheme applies it. Norms are L2 over the grid. The
        denominator is the size of the whole balance, so r stays meaningful for steady states (Dw -> 0). For the
        reference solver r is O(dt^2) (trapezoid vs RK3 on N) plus float32 round-off, and exactly 0 for purely
        linear decay; it measures physical consistency of a transition, not prediction error. The transition may span
        more than one stable solver step (a surrogate stride): the CFL limit is not applied here, only to stepping."""
        p = self.validate({**params, "dt": None})
        p["dt"] = float(params["dt"]) if params.get("dt") is not None else p["dt"]
        g, q, split = self._setup(p)
        w0, w1 = (jnp.fft.rfft2(jnp.asarray(s, jnp.float32)[0]) * g["dealias"] for s in (prev, nxt))
        wt = self._split(w0, q, g) if split else w0
        terms = lambda a: (self._advection(a, q, g), -p["nu"] * g["k2"] * a, -p["mu"] * a, p["A"] * g["f_hat"])
        T = [self._irf(0.5 * (a + b)) for a, b in zip(terms(wt), terms(w1))]
        Dw = self._irf(w1 - w0) / p["dt"]
        P = self._irf(wt - w0) / p["dt"]
        E = jnp.exp(-(p["nu"] * g["k2"] + p["mu"]) * p["dt"])
        N = lambda a: self._advection(a, q, g) + p["A"] * g["f_hat"]
        R = self._irf((w1 - E * wt) / p["dt"] - 0.5 * (E * N(wt) + N(w1)))
        nrm = jnp.linalg.norm
        return float(nrm(R) / (nrm(Dw) + nrm(P) + sum(nrm(t) for t in T) + 1e-30))


def cylinder_mask(n, L, center, radius):
    """Boolean (n, n) disc on the grid (row = y, column = x), for tests and examples."""
    x = np.arange(n) * L / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    return (X - center[0]) ** 2 + (Y - center[1]) ** 2 <= radius ** 2


if __name__ == "__main__":  # throughput: python -m kolmo.systems.fluid2d [n ...]
    import sys
    sys_ = Fluid2D()
    for n in [int(a) for a in sys.argv[1:]] or [64, 128, 256]:
        p = sys_.validate({"n": n, "nu": 1e-3, "A": 1.0, "kf": 4, "mu": 0.05, "ic_amp": 1.0})
        s0 = sys_.initial(p, 0)
        steps = {64: 2000, 128: 500, 256: 150}.get(n, 200)
        tr = sys_.rollout(p, s0, steps, every=steps)
        print(f"n={n:4d} dt={p['dt']:.3g} compile {tr.compile_seconds:.2f} s, steady {steps / tr.seconds:.1f} steps/s "
              f"({steps} steps in {tr.seconds:.2f} s)")
        B = 8  # vmapped batch of seeds
        tb = sys_.rollout(p, jnp.stack([sys_.initial(p, i) for i in range(B)]), steps // 4, every=steps // 4)
        print(f"       batch of {B} seeds: compile {tb.compile_seconds:.2f} s, steady "
              f"{B * (steps // 4) / tb.seconds:.1f} member-steps/s")
