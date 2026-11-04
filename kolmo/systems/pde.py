"""Spec-driven pseudo-spectral PDE system: du_i/dt = rhs_i(u) on a periodic square, rhs written as text.

Grammar (parsed with `ast` against a whitelist, never eval): field names, parameter names, `pi`, numbers,
`+ - * / **`, unary minus, and the one-argument functions lap, dx, dy, grad2 (= |grad u|^2), exp, tanh, sin, cos.

Numerics: the linear constant-coefficient part of each equation is found automatically as the Fourier symbol of the
Jacobian of rhs_i with respect to u_i at u = 0 (the grammar has no x-dependence, so that Jacobian is
translation-invariant and diagonal in Fourier space), optionally made more dissipative by `stabilize`
(S lap u moved from the explicit to the exact part). That part is integrated exactly by ETDRK4 (Cox-Matthews, with
the coefficients evaluated by a complex contour mean, Kassam-Trefethen); the remainder N = rhs - L u is explicit.
2/3-rule dealiasing of N and of the state. See docs/physics.md."""
import ast
import math
import operator

import jax
import jax.numpy as jnp
import numpy as np

from kolmo.systems.common import grid, scan_rollout

FUNCS = ("lap", "dx", "dy", "grad2", "exp", "tanh", "sin", "cos")
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow}
GROW_MAX = 2.0  # max (linear growth rate) * dt
STIFF_MAX = 2.0  # max |J_N| * dt for the explicit part (RK4 real-axis bound 2.78, imaginary 2.83)

LIBRARY = {
    "cahn_hilliard": {
        "name": "cahn_hilliard", "fields": ["u"], "rhs": {"u": "lap(u**3 - u) - gamma*lap(lap(u))"},
        "params": {"gamma": 1.0}, "n": 64, "L": 32.0, "dt": 0.03, "ic": "noise:0.1", "stabilize": {"u": 1.5},
        "energy": "0.25*(u**2 - 1)**2 + 0.5*gamma*grad2(u)",
        "checks": ["mass conserved to round-off", "free energy non-increasing"],
        "help": {"gamma": "interface energy (interface width ~ sqrt(2 gamma))"}},
    "allen_cahn": {
        "name": "allen_cahn", "fields": ["u"], "rhs": {"u": "eps2*lap(u) + u - u**3"},
        "params": {"eps2": 1.0}, "n": 64, "L": 32.0, "dt": 0.1, "ic": "noise:0.1",
        "energy": "0.5*eps2*grad2(u) + 0.25*(u**2 - 1)**2",
        "checks": ["free energy non-increasing", "|u| <= 1 (maximum principle)"],
        "help": {"eps2": "squared interface width"}},
    "gray_scott": {
        "name": "gray_scott", "fields": ["u", "v"],
        "rhs": {"u": "Du*lap(u) - u*v**2 + F*(1 - u)", "v": "Dv*lap(v) + u*v**2 - (F + k)*v"},
        "params": {"Du": 2e-5, "Dv": 1e-5, "F": 0.04, "k": 0.06}, "n": 64, "L": 0.32, "dt": 1.0,
        "ic": {"u": "const:1+bump:-0.5:0.03+noise:0.01", "v": "bump:0.25:0.03"},
        "checks": ["u, v >= 0 (positivity)"],
        "help": {"Du": "diffusivity of u", "Dv": "diffusivity of v", "F": "feed rate", "k": "kill rate"}},
    "kuramoto_sivashinsky": {
        "name": "kuramoto_sivashinsky", "fields": ["u"], "rhs": {"u": "-lap(u) - lap(lap(u)) - 0.5*grad2(u)"},
        "params": {}, "n": 64, "L": 16 * math.pi, "dt": 0.1, "ic": "noise:0.1",
        "checks": ["linear growth rate |k|^2 - |k|^4"]},
    "swift_hohenberg": {
        "name": "swift_hohenberg", "fields": ["u"], "rhs": {"u": "(r - 1)*u - 2*lap(u) - lap(lap(u)) - u**3"},
        "params": {"r": 0.2}, "n": 64, "L": 16 * math.pi, "dt": 0.5, "ic": "noise:0.1",
        "energy": "-0.5*r*u**2 + 0.5*(u + lap(u))**2 + 0.25*u**4",
        "checks": ["Lyapunov functional non-increasing", "linear growth rate r - (1 - |k|^2)^2"],
        "help": {"r": "distance above onset"}},
    "burgers": {
        "name": "burgers", "fields": ["u"], "rhs": {"u": "nu*lap(u) - u*(dx(u) + dy(u))"},
        "params": {"nu": 0.1}, "n": 64, "L": 2 * math.pi, "dt": 0.01, "ic": "sin:1:1",
        "energy": "0.5*u**2", "checks": ["mean conserved", "L2 energy non-increasing"],
        "help": {"nu": "viscosity"}},
}


def compile_expr(src, names):
    """Whitelisted AST -> function env -> array. Raises ValueError naming the offending symbol or syntax."""
    try:
        tree = ast.parse(str(src), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"cannot parse {src!r}: {e.msg}") from None

    def build(node):
        if isinstance(node, ast.Expression):
            return build(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            v = float(node.value)
            return lambda env: v
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"unknown symbol {node.id!r} in {src!r}; allowed: {sorted(names)} and {list(FUNCS)}")
            return lambda env, k=node.id: env[k]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            a, b = build(node.left), build(node.right)
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) and type(node.right.value) is int:
                m = node.right.value
                return lambda env: a(env) ** m  # integer power (exact, differentiable at 0)
            f = _BINOPS[type(node.op)]
            return lambda env: f(a(env), b(env))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            a, s = build(node.operand), -1.0 if isinstance(node.op, ast.USub) else 1.0
            return lambda env: s * a(env)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in FUNCS:
                raise ValueError(f"unknown function {node.func.id!r} in {src!r}; allowed: {list(FUNCS)}")
            if len(node.args) != 1 or node.keywords:
                raise ValueError(f"{node.func.id}() takes exactly one argument in {src!r}")
            a = build(node.args[0])
            return lambda env, fn=node.func.id: env["ops"][fn](a(env))
        raise ValueError(f"unsupported syntax {type(node).__name__} in {src!r}; allowed: fields, params, numbers, "
                         f"+ - * / **, {', '.join(FUNCS)}")
    return build(tree)


class PDE:
    """One spec -> one System. `params` passed to validate may override n, L, dt, ic and the spec's parameters."""

    def __init__(self, spec=None):
        spec = dict(spec or LIBRARY["cahn_hilliard"])
        self.spec = spec
        self.name = str(spec.get("name", "pde"))
        self.fields = tuple(spec["fields"])
        rhs = spec.get("rhs", {})
        if set(rhs) != set(self.fields):
            raise ValueError(f"rhs must give exactly one expression per field {self.fields}, got {sorted(rhs)}")
        self.DEFAULTS = {"n": 64, "L": 2 * math.pi, "dt": 0.01, "ic": "noise:0.1",
                         **{k: spec[k] for k in ("n", "L", "dt", "ic") if k in spec}, **spec.get("params", {})}
        names = set(self.fields) | set(spec.get("params", {})) | {"pi"}
        for k in self.DEFAULTS:
            if k in FUNCS or (k in self.fields):
                raise ValueError(f"parameter name {k!r} clashes with a field or function")
        self._rhs = [compile_expr(rhs[f], names) for f in self.fields]
        self._energy = compile_expr(spec["energy"], names) if spec.get("energy") else None
        self._stab = [float(spec.get("stabilize", {}).get(f, 0.0)) for f in self.fields]
        self._cache = {}

    # ------------------------------------------------------------------ parameters and setup
    def validate(self, params):
        p = {**self.DEFAULTS, **(params or {})}
        for k in ("growth_rate", "stiffness", "_skip"):  # derived keys of an already validated dict
            p.pop(k, None)
        unknown = set(p) - set(self.DEFAULTS) - {"seed"}
        if unknown:
            raise ValueError(f"unknown parameter(s) {sorted(unknown)}; known: {sorted(self.DEFAULTS)}")
        p["n"] = int(p["n"])
        if p["n"] < 16 or p["n"] % 2:
            raise ValueError("n must be an even number >= 16")
        for k in set(self.DEFAULTS) - {"n", "ic"}:
            p[k] = float(p[k])
        if not (p["L"] > 0 and p["dt"] > 0):
            raise ValueError("L and dt must be positive")
        key = repr(sorted((k, v) for k, v in p.items()))
        if key not in self._cache:
            self._cache[key] = self._check(p)
        p.update(self._cache[key])
        return p

    def _env(self, state, p, g):
        n = p["n"]
        rf, irf = jnp.fft.rfft2, lambda a: jnp.fft.irfft2(a, s=(n, n))
        full = lambda a: jnp.broadcast_to(jnp.asarray(a, jnp.float32), (n, n))
        d = lambda a, k: irf(1j * k * rf(full(a)))
        ops = {"lap": lambda a: irf(-g["k2"] * rf(full(a))), "dx": lambda a: d(a, g["kx"]),
               "dy": lambda a: d(a, g["ky"]), "grad2": lambda a: d(a, g["kx"]) ** 2 + d(a, g["ky"]) ** 2,
               "exp": jnp.exp, "tanh": jnp.tanh, "sin": jnp.sin, "cos": jnp.cos}
        env = {k: p[k] for k in self.spec.get("params", {})}
        env.update({f: state[i] for i, f in enumerate(self.fields)}, pi=math.pi, ops=ops)
        return env

    def _R(self, state, p, g):
        env = self._env(state, p, g)
        n = p["n"]
        return jnp.stack([jnp.broadcast_to(jnp.asarray(r(env), jnp.float32), (n, n)) for r in self._rhs])

    def _grid(self, p):
        return {k: jnp.asarray(v, jnp.float32) for k, v in grid(p["n"], p["L"]).items()}

    def _symbol(self, p, g):
        """(C, n, n//2+1) complex128 symbol of the exactly treated linear part (memoised per parameter set)."""
        key = repr(sorted((k, v) for k, v in p.items() if k not in ("ic", "dt", "growth_rate", "stiffness")))
        if key in self._cache:
            return self._cache[key]
        n, C = p["n"], len(self.fields)
        zero = jnp.zeros((C, n, n), jnp.float32)
        jvp = jax.jit(lambda d: jax.jvp(lambda s: self._R(s, p, g), (zero,), (d,))[1])
        lam = [jnp.fft.rfft2(jvp(zero.at[i, 0, 0].set(1.0))[i]) - self._stab[i] * g["k2"] for i in range(C)]
        lam = self._cache[key] = np.asarray(jnp.stack(lam), np.complex128)
        if not np.isfinite(lam).all():
            raise ValueError("the linearisation of rhs at u = 0 is not finite (e.g. division by a field); "
                             "rewrite the rhs so it is defined at 0")
        return lam

    def _check(self, p):
        """Linear growth and explicit-stiffness limits; cached per parameter set."""
        g = self._grid(p)
        lam = self._symbol(p, g) * g["dealias"]
        grow = float(lam.real.max())
        if grow * p["dt"] > GROW_MAX:
            raise ValueError(f"linear instability too fast for dt: max growth rate {grow:.4g}, rate*dt = "
                             f"{grow * p['dt']:.3g} > {GROW_MAX}; use dt <= {GROW_MAX / grow:.4g}")
        # power iteration on the Jacobian of the explicit part N = rhs - L u at the seed-0 initial state
        u0 = self.initial({**p, "_skip": True}, 0)
        lam_j = jnp.asarray(lam, jnp.complex64)
        R = lambda s: self._R(s, p, g)

        @jax.jit
        def JN(v):
            jv = jax.jvp(R, (u0,), (v,))[1]
            out = (jnp.fft.rfft2(jv) - lam_j * jnp.fft.rfft2(v)) * g["dealias"]
            return jnp.fft.irfft2(out, s=v.shape[-2:])
        v = jnp.fft.irfft2(jnp.fft.rfft2(jax.random.normal(jax.random.PRNGKey(0), u0.shape)) * g["dealias"],
                           s=u0.shape[-2:])
        rho = 0.0
        for _ in range(30):
            w = JN(v)
            rho = max(rho, float(jnp.linalg.norm(w) / (jnp.linalg.norm(v) + 1e-30))) if _ >= 20 else rho
            v = w / (jnp.linalg.norm(w) + 1e-30)
        if rho * p["dt"] > STIFF_MAX:
            raise ValueError(f"explicit (nonlinear) part too stiff for dt: |J_N| ~ {rho:.4g} at the initial state, "
                             f"|J_N|*dt = {rho * p['dt']:.3g} > {STIFF_MAX}; use dt <= {STIFF_MAX / rho:.4g}"
                             f" or increase 'stabilize'")
        return {"growth_rate": grow, "stiffness": rho}

    def _coeffs(self, p, g):
        """ETDRK4 coefficients by the contour mean of Kassam & Trefethen (M = 32 points, radius 1)."""
        h = p["dt"]
        z = self._symbol(p, g)[..., None] * h + np.exp(2j * np.pi * (np.arange(32) + 0.5) / 32)
        ez, m = np.exp(z), lambda a: a.mean(-1)
        c = {"E": np.exp(z.mean(-1)), "E2": np.exp(z.mean(-1) / 2), "Q": h * m((np.exp(z / 2) - 1) / z),
             "f1": h * m((-4 - z + ez * (4 - 3 * z + z ** 2)) / z ** 3), "f2": h * m((2 + z + ez * (z - 2)) / z ** 3),
             "f3": h * m((-4 - 3 * z - z ** 2 + ez * (4 - z)) / z ** 3), "lam": z.mean(-1) / h,
             "phi1": m((ez - 1) / z), "phi2": m((ez - 1 - z) / z ** 2)}
        return {k: jnp.asarray(v, jnp.complex64) for k, v in c.items()}

    # ------------------------------------------------------------------ System protocol
    def initial(self, params, seed):
        """ic: one string for all fields or {field: string}; '+'-joined terms const:c | noise:A (white, 2/3 band,
        std A, seeded) | bump:A:r (Gaussian, centre of box) | sin:A:m (A sin(2 pi m x / L)) |
        spots:A:r:count (Gaussian bumps at seeded random positions, the same for every field) | zero."""
        p = params if params.get("_skip") else self.validate(params)
        n, L = int(p["n"]), float(p["L"])
        x = np.arange(n) * L / n
        X, Y = np.meshgrid(x, x, indexing="xy")
        ics = p["ic"] if isinstance(p["ic"], dict) else {f: p["ic"] for f in self.fields}
        dealias = grid(n, L)["dealias"]
        out = []
        for i, f in enumerate(self.fields):
            u = np.zeros((n, n))
            for term in str(ics.get(f, "zero")).replace(" ", "").split("+"):
                kind, *a = term.split(":")
                try:
                    a = [float(v) for v in a]
                    if kind == "zero":
                        continue
                    elif kind == "const":
                        u += a[0]
                    elif kind == "noise":
                        r = np.fft.irfft2(np.fft.rfft2(np.random.default_rng([seed, i]).standard_normal((n, n))) * dealias, s=(n, n))
                        u += a[0] * r / r.std()
                    elif kind == "bump":
                        u += a[0] * np.exp(-((X - L / 2) ** 2 + (Y - L / 2) ** 2) / (2 * a[1] ** 2))
                    elif kind == "sin":
                        u += a[0] * np.sin(2 * np.pi * a[1] * X / L)
                    elif kind == "spots":  # spots:A:r:count, Gaussian bumps at seeded positions shared by all fields
                        for cx, cy in np.random.default_rng([seed, 7]).uniform(0, L, (int(a[2]), 2)):
                            dx, dy = (X - cx + L / 2) % L - L / 2, (Y - cy + L / 2) % L - L / 2
                            u += a[0] * np.exp(-(dx ** 2 + dy ** 2) / (2 * a[1] ** 2))
                    else:
                        raise ValueError
                except (ValueError, IndexError):
                    raise ValueError(f"bad ic term {term!r} for field {f}; use const:c, noise:A, bump:A:r, sin:A:m, spots:A:r:count, zero") from None
            out.append(u)
        return jnp.asarray(np.stack(out), jnp.float32)

    def _step_fn(self, p):
        g, c = self._grid(p), None
        c = self._coeffs(p, g)
        n, D = p["n"], g["dealias"]
        rf, irf = jnp.fft.rfft2, lambda a: jnp.fft.irfft2(a, s=(n, n))
        N = lambda vh: (rf(self._R(irf(vh), p, g)) - c["lam"] * vh) * D

        def step(s):
            v = rf(s) * D
            Nv = N(v)
            a = c["E2"] * v + c["Q"] * Nv
            Na = N(a)
            b = c["E2"] * v + c["Q"] * Na
            Nb = N(b)
            cc = c["E2"] * a + c["Q"] * (2 * Nb - Nv)
            Nc = N(cc)
            v = c["E"] * v + c["f1"] * Nv + 2 * c["f2"] * (Na + Nb) + c["f3"] * Nc
            return irf(v * D).astype(jnp.float32)
        return step

    def stepper(self, params):
        return jax.jit(self._step_fn(self.validate(params)))

    def batched_stepper(self, params):
        """(B, C, n, n) -> (B, C, n, n) for one parameter set (e.g. a batch of seeds)."""
        return jax.jit(jax.vmap(self._step_fn(self.validate(params))))

    def rollout(self, params, state0, steps, every=1):
        return scan_rollout(self._step_fn(self.validate(params)), state0, steps, every)

    def diagnostics(self, state, params):
        """Per field box mean, variance, min, max; `energy` = box mean of the spec's energy density if given;
        growth_rate / stiffness = validate's linear-growth and explicit-Jacobian estimates (1/time)."""
        p = self.validate(params)
        s = jnp.asarray(state, jnp.float32)
        d = {}
        for i, f in enumerate(self.fields):
            d.update({f"mean_{f}": jnp.mean(s[i]), f"var_{f}": jnp.var(s[i]), f"min_{f}": jnp.min(s[i]),
                      f"max_{f}": jnp.max(s[i])})
        if self._energy is not None:
            d["energy"] = jnp.mean(jnp.broadcast_to(self._energy(self._env(s, p, self._grid(p))), s.shape[-2:]))
        d["growth_rate"], d["stiffness"] = p["growth_rate"], p["stiffness"]
        return {k: float(v) for k, v in d.items()}

    def residual(self, prev, nxt, params):
        """Variation-of-constants residual per Fourier mode: linear part exact, N interpolated linearly in time
        and integrated exactly against exp(L (dt - s)) (the ETD2 quadrature, accurate for stiff slaved modes):
            R = (v1 - E v0)/dt - [phi1(L dt) N0 + phi2(L dt) (N1 - N0)],  phi1 = (e^z - 1)/z, phi2 = (e^z - 1 - z)/z^2
            r = ||R|| / (||(v1 - v0)/dt|| + ||0.5 (N0 + N1)|| + ||0.5 L (v0 + v1)||), L2 over all fields.
        O(dt^2) for the reference solver; a physical-consistency metric, not a prediction error."""
        p = self.validate(params)
        g = self._grid(p)
        c = self._coeffs(p, g)
        n, D = p["n"], g["dealias"]
        rf = jnp.fft.rfft2
        N = lambda vh: (rf(self._R(jnp.fft.irfft2(vh, s=(n, n)), p, g)) - c["lam"] * vh) * D
        v0, v1 = (rf(jnp.asarray(s, jnp.float32)) * D for s in (prev, nxt))
        N0, N1 = N(v0), N(v1)
        R = (v1 - c["E"] * v0) / p["dt"] - (c["phi1"] * N0 + c["phi2"] * (N1 - N0))
        nrm = lambda a: jnp.linalg.norm(a.ravel())
        den = nrm((v1 - v0) / p["dt"]) + nrm(0.5 * (N0 + N1)) + nrm(0.5 * c["lam"] * (v0 + v1))
        return float(nrm(R) / (den + 1e-30))

    def describe(self):
        info = self.spec.get("help", {})
        params = {k: {"default": v, "min": 0.0 if k not in ("ic",) else None, "max": None, "unit": "nondim",
                      "help": info.get(k, "")} for k, v in self.spec.get("params", {}).items()}
        params.update({"n": {"default": self.DEFAULTS["n"], "min": 16, "max": 512, "unit": "cells", "help": "grid size"},
                       "L": {"default": self.DEFAULTS["L"], "min": 0.0, "max": None, "unit": "length", "help": "box side"},
                       "dt": {"default": self.DEFAULTS["dt"], "min": 0.0, "max": None, "unit": "time", "help": "time step"},
                       "ic": {"default": self.DEFAULTS["ic"], "min": None, "max": None, "unit": "",
                              "help": "const:c | noise:A | bump:A:r | sin:A:m, '+'-joined, per field or shared"}})
        eqs = "; ".join(f"d{f}/dt = {self.spec['rhs'][f]}" for f in self.fields)
        return {"name": self.name, "fields": list(self.fields), "equations": eqs, "params": params,
                "checks": list(self.spec.get("checks", []))}
