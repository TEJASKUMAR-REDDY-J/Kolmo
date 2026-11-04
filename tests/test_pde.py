"""Spec-driven PDE system and its library. Run: python -m pytest tests/test_pde.py -q -s"""
import math

import numpy as np
import pytest

from kolmo.systems import make, describe, SYSTEMS, LIBRARY
from kolmo.systems.pde import PDE


def _run(name, steps, every, **over):
    S = make(name)
    p = S.validate(over)
    tr = S.rollout(p, S.initial(p, 0), steps, every=every)
    return S, p, tr, [S.diagnostics(s, p) for s in tr.states]


def test_grammar_whitelist():
    base = {"name": "t", "fields": ["u"], "params": {"a": 1.0}}
    for bad, msg in [("u + b", "unknown symbol"), ("__import__('os')", "unknown function"), ("u.real", "unsupported"),
                     ("lap(u, u)", "exactly one"), ("u if u else u", "unsupported"), ("u +", "cannot parse")]:
        with pytest.raises(ValueError, match=msg):
            PDE({**base, "rhs": {"u": bad}})
    S = PDE({**base, "rhs": {"u": "a*lap(u) - dx(u)*dy(u) + tanh(u)**2 - exp(-u) + sin(pi*u)*cos(u) + grad2(u)/2"}})
    assert np.isfinite(np.asarray(S.stepper({"dt": 0.01})(S.initial({}, 0)))).all()


def test_registry_and_describe():
    assert set(SYSTEMS) == {"fluid2d", "diffusion", "boussinesq", "pde"}
    for name in list(SYSTEMS) + list(LIBRARY):
        d = describe(name)
        assert {"name", "fields", "equations", "params", "checks"} <= set(d)
        assert all({"default", "min", "max", "unit", "help"} <= set(v) for v in d["params"].values())
    with pytest.raises(ValueError, match="unknown system"):
        make("nope")


def test_stiffness_refusal():
    with pytest.raises(ValueError, match="too stiff.*use dt <="):
        make("cahn_hilliard").validate({"dt": 0.1})
    with pytest.raises(ValueError, match="linear instability"):
        make("kuramoto_sivashinsky").validate({"dt": 10.0})


def test_cahn_hilliard_mass_and_energy():
    S, p, tr, d = _run("cahn_hilliard", 1000, 50)  # t = 30
    mass = [x["mean_u"] for x in d]
    E = np.array([x["energy"] for x in d])
    dm, rise = max(abs(m - mass[0]) for m in mass), np.diff(E).max()
    r = S.residual(tr.states[-1], S.stepper(p)(tr.states[-1]), p)
    print(f"CH: max |mean(u) - mean(u0)| = {dm:.2e}, energy {E[0]:.4f} -> {E[-1]:.4f}, max rise {rise:.2e}, "
          f"min/max u {d[-1]['min_u']:.3f}/{d[-1]['max_u']:.3f}, residual {r:.1e}")
    assert dm < 1e-6 and rise <= 0 and r < 1e-3


def test_kuramoto_sivashinsky_linear_growth():
    S, p, tr, _ = _run("kuramoto_sivashinsky", 100, 100, ic="sin:1e-4:4")  # k = 2 pi 4 / (16 pi) = 0.5, t = 10
    k, T = 0.5, 100 * p["dt"]
    amp = lambda s: 2 * np.abs(np.fft.rfft2(s[0])[0, 4]) / s[0].size
    measured = math.log(amp(tr.states[-1]) / amp(tr.states[0])) / T
    print(f"KS growth rate measured {measured:.6f}, theory k^2 - k^4 = {k * k - k ** 4:.6f}")
    assert abs(measured / (k * k - k ** 4) - 1) < 1e-3


def test_etdrk4_order():
    # Allen-Cahn from noise: nonlinear and stiff enough that the error stays above the float32 floor (~3e-5),
    # unlike KS/Burgers at these sizes whose ETDRK4 error is already at round-off for dt <= 0.2.
    S = make("allen_cahn")
    base = {"n": 32, "ic": "noise:0.5"}
    s0 = S.initial(base, 3)
    T, dts = 4.0, [0.4, 0.2, 0.1]
    fin = lambda dt: S.rollout({**base, "dt": dt}, s0, int(round(T / dt)), every=int(round(T / dt))).states[-1]
    ref = fin(0.0125)
    errs = [np.abs(fin(dt) - ref).max() for dt in dts]
    order = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    print(f"ETDRK4 on Allen-Cahn: max-abs errors {['%.2e' % e for e in errs]} at dt {dts}, fitted order {order:.2f}")
    assert order > 3.3


def test_gray_scott_positivity():
    S, p, tr, d = _run("gray_scott", 2000, 200)
    lo = min(min(x["min_u"], x["min_v"]) for x in d)
    print(f"GS: min over 2000 steps of min(u, v) = {lo:.2e}; final v range {d[-1]['min_v']:.3f}..{d[-1]['max_v']:.3f}")
    assert lo > -1e-6


def test_allen_cahn_swift_hohenberg_burgers_lyapunov():
    for name, steps, check in [("allen_cahn", 500, "bounded"), ("swift_hohenberg", 400, None), ("burgers", 300, "mean")]:
        S, p, tr, d = _run(name, steps, steps // 20)
        E = np.array([x["energy"] for x in d])
        print(f"{name}: energy {E[0]:.5f} -> {E[-1]:.5f}, max rise {np.diff(E).max():.2e}, "
              f"range {d[-1]['min_u']:.4f}..{d[-1]['max_u']:.4f}, mean drift {abs(d[-1]['mean_u'] - d[0]['mean_u']):.1e}")
        assert np.diff(E).max() <= 1e-7 * abs(E).max()
        if check == "bounded":
            assert max(max(abs(x["min_u"]), abs(x["max_u"])) for x in d) <= 1 + 1e-3
        if check == "mean":
            assert abs(d[-1]["mean_u"] - d[0]["mean_u"]) < 1e-6
