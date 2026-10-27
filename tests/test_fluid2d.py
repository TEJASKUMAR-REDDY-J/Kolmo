"""Verification of the Fluid2D reference solver. Run: python -m pytest tests/test_fluid2d.py -q  (or python -m tests.test_fluid2d)."""
import math

import numpy as np
import pytest

from kolmo.systems.fluid2d import Fluid2D, cylinder_mask

S = Fluid2D()


def test_taylor_green_exact_decay():
    p = S.validate({"n": 32, "ic": "taylor_green", "nu": 0.05, "mu": 0.1, "dt": 0.05})
    s0 = S.initial(p, 0)
    tr = S.rollout(p, s0, 100)
    exact = np.asarray(s0) * math.exp(-(2 * p["nu"] + p["mu"]) * 100 * p["dt"])  # |k|^2 = 2 for tg_k = 1, L = 2 pi
    err = np.abs(tr.states[-1] - exact).max() / np.abs(exact).max()
    print(f"taylor-green rel max err {err:.2e}")
    assert err < 1e-4


def test_kolmogorov_laminar_steady_state():
    p = S.validate({"n": 32, "ic": "zero", "nu": 0.05, "mu": 0.2, "A": 1.0, "kf": 2})
    tr = S.rollout(p, S.initial(p, 0), int(40 / p["dt"]), every=int(40 / p["dt"]))  # 16 e-folds of 1/(nu k^2 + mu)
    k = 2.0
    y = np.arange(32) * 2 * math.pi / 32
    exact = p["A"] * k * np.cos(k * y)[:, None] / (p["nu"] * k * k + p["mu"])
    err = np.abs(tr.states[-1][0] - exact).max() / np.abs(exact).max()
    print(f"kolmogorov laminar amplitude rel err {err:.2e} (exact amplitude {exact.max():.4f})")
    assert err < 1e-4


def test_energy_budget():
    p = S.validate({"n": 64, "nu": 2e-3, "mu": 0.05, "A": 1.0, "kf": 4, "ic_amp": 1.0})
    tr = S.rollout(p, S.initial(p, 0), 200)
    E = lambda j: S.diagnostics(tr.states[j], p)["energy"]
    worst = 0.0
    for i in range(20, 200, 20):
        d = S.diagnostics(tr.states[i], p)
        dEdt = (E(i + 1) - E(i - 1)) / (2 * p["dt"])  # central difference, O(dt^2)
        budget = d["forcing_power"] - 2 * p["nu"] * d["enstrophy"] - 2 * p["mu"] * d["energy"]
        scale = abs(d["forcing_power"]) + 2 * p["nu"] * d["enstrophy"] + 2 * p["mu"] * d["energy"]
        worst = max(worst, abs(dEdt - budget) / scale)
    r = S.residual(tr.states[-2], tr.states[-1], p)
    print(f"energy budget worst |dE/dt - (P - 2 nu Z - 2 mu E)| / (|P| + 2 nu Z + 2 mu E) = {worst:.2e}; residual {r:.2e}")
    assert worst < 1e-3
    assert r < 1e-2


def test_temporal_convergence_order():
    # dt range chosen so truncation error stays above the float32 accumulation floor (~1e-4 max-abs after ~500 steps)
    base = {"n": 32, "nu": 0.01, "ic_amp": 1.0, "ic_k": (1, 4), "cfl_max": 0.82}
    s0 = S.initial(base, 4)
    T, dts = 1.2, [0.04, 0.03, 0.02]
    final = lambda dt: S.rollout({**base, "dt": dt}, s0, int(round(T / dt)), every=int(round(T / dt))).states[-1]
    ref = final(0.005)
    errs = [np.abs(final(dt) - ref).max() for dt in dts]
    order = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    print(f"temporal max-abs errors at dt={dts}: {['%.2e' % e for e in errs]}, fitted order {order:.2f}")
    assert 2.7 < order < 3.5


def test_determinism():
    p = S.validate({"n": 32, "nu": 1e-3, "A": 0.5})
    a, b, c = S.initial(p, 7), S.initial(p, 7), S.initial(p, 8)
    assert np.array_equal(np.asarray(a), np.asarray(b)) and not np.array_equal(np.asarray(a), np.asarray(c))
    ta, tb = S.rollout(p, a, 20).states, S.rollout(p, b, 20).states
    assert np.array_equal(ta, tb)


def test_cfl_refusal_and_reynolds():
    with pytest.raises(ValueError, match="use dt <=") as e:
        S.validate({"n": 64, "dt": 1.0})
    suggested = float(str(e.value).split("use dt <= ")[1])
    assert S.validate({"n": 64, "dt": suggested * 0.999})["cfl_estimate"] <= 0.5
    p = S.validate({"n": 64, "Re": 500})
    assert abs(p["nu"] - 1 / 500) < 1e-12  # unforced: U_ref = ic_amp = 1, L_ref = L / 2 pi = 1
    with pytest.raises(ValueError, match="disagree"):
        S.validate({"n": 64, "Re": 500, "nu": 1e-2})


def test_batched_stepper_matches_single():
    p = S.validate({"n": 32, "nu": 1e-2, "A": 0.5})
    s = np.stack([np.asarray(S.initial(p, i)) for i in range(3)])
    seeds = np.asarray(S.batched_stepper(p)(s))
    nus = (1e-2, 2e-2, 4e-2)
    per_nu = np.asarray(S.batched_stepper(p, [{"nu": v, "Re": None} for v in nus])(s))
    for i, nu in enumerate(nus):
        assert np.allclose(seeds[i], np.asarray(S.stepper(p)(s[i])), atol=1e-5)
        assert np.allclose(per_nu[i], np.asarray(S.stepper({**p, "nu": nu, "Re": None})(s[i])), atol=1e-5)


def test_penalised_cylinder():
    """Re = U D / nu = 20 cylinder of nominal D = 1 in a 4 pi box (blockage ~8 %, periodic array), sponge 10 %,
    t = 20 (Cd settled to < 1e-3). n = 64 and 192 are compared because their staircase masks have the same
    cross-stream extent (D = 0.982); at other n the staircase D jumps by up to 10 % and dominates the Cd change."""
    L, out = 4 * math.pi, {}
    for n in (64, 192):
        m = cylinder_mask(n, L, (L / 4, L / 2), 0.5)
        p = S.validate({"n": n, "L": L, "mask": m, "U_inf": 1.0, "Re": 20})
        steps = int(round(20 / p["dt"]))
        tr = S.rollout(p, S.initial(p, 0), steps, every=steps)
        d = S.diagnostics(tr.states[-1], p)
        r = S.residual(tr.states[-1], S.stepper(p)(tr.states[-1]), p)
        g, _, _ = S._setup(p)
        u, v = S._velocity(np.fft.rfft2(tr.states[-1][0]).astype(np.complex64), g)
        speed = np.hypot(np.asarray(u) + p["U_inf"], np.asarray(v)) / p["U_inf"]
        inner = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
        out[n] = (d["cd"], speed[m].mean())
        print(f"cylinder n={n}: D={p['D']:.3f} cd={d['cd']:.4f} |cl|={abs(d['cl']):.1e} residual={r:.1e} "
              f"max|u|/U inside (1-cell eroded)={speed[inner].max():.3f} mean|u|/U inside={speed[m].mean():.3f}")
        assert 1.8 < d["cd"] < 2.6 and abs(d["cl"]) < 1e-3 and speed[inner].max() < 0.15 and r < 1e-2
    assert abs(out[192][0] / out[64][0] - 1) < 0.05  # Cd changes by < 5 % at 3x resolution
    assert out[192][1] < 0.5 * out[64][1]  # leakage into the body falls with dx


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.perf_counter()
            fn()
            print(f"PASS {name} ({time.perf_counter() - t:.1f} s)")
