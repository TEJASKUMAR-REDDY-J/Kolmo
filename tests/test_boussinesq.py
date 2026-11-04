"""Boussinesq vorticity-temperature system. Run: python -m pytest tests/test_boussinesq.py -q -s"""
import math

import numpy as np

from kolmo.systems.boussinesq import Boussinesq
from kolmo.systems.fluid2d import cylinder_mask

B = Boussinesq()


def test_energy_and_temperature_budgets():
    p = B.validate({"ic": "thermal"})
    tr = B.rollout(p, B.initial(p, 0), 60)
    D = lambda j: B.diagnostics(tr.states[j], p)
    eE = eT = 0.0
    for i in range(10, 60, 10):
        d, a, b = D(i), D(i - 1), D(i + 1)
        dE, dT = (b["energy"] - a["energy"]) / (2 * p["dt"]), (b["theta_var"] - a["theta_var"]) / (2 * p["dt"])
        eE = max(eE, abs(dE - (d["buoyancy_flux"] - 2 * p["nu"] * d["enstrophy"]))
                 / (abs(d["buoyancy_flux"]) + 2 * p["nu"] * d["enstrophy"]))
        eT = max(eT, abs(dT + d["thermal_dissipation"]) / d["thermal_dissipation"])
    r = B.residual(tr.states[-2], tr.states[-1], p)
    print(f"Boussinesq budgets: kinetic {eE:.1e}, thermal variance {eT:.1e} (relative), residual {r:.1e}")
    assert eE < 1e-3 and eT < 2e-3 and r < 1e-2


def test_rising_thermal():
    p = B.validate({"ic": "thermal"})
    steps = int(4 / p["dt"])
    tr = B.rollout(p, B.initial(p, 0), steps, every=steps // 4)
    d = [B.diagnostics(s, p) for s in tr.states]
    yc = [x["centroid_y"] for x in d]
    th = tr.states[-1][1]
    xs = np.arange(p["n"]) * p["L"] / p["n"]
    xc = (np.maximum(th, 0) * xs[None, :]).sum() / np.maximum(th, 0).sum()
    print(f"thermal: centroid_y {['%.3f' % y for y in yc]}, x-centroid drift {abs(xc - p['L'] / 2):.1e}, "
          f"mean theta drift {abs(d[-1]['theta_mean'] - d[0]['theta_mean']):.1e}")
    assert all(np.diff(yc) > 0) and yc[-1] - yc[0] > 1.0
    assert abs(xc - p["L"] / 2) < 1e-3 and abs(d[-1]["theta_mean"] - d[0]["theta_mean"]) < 1e-6


def test_rayleigh_taylor_growth():
    p = B.validate({"ic": "rayleigh_taylor"})
    steps = int(4 / p["dt"])
    tr = B.rollout(p, B.initial(p, 0), steps, every=steps // 8)
    E = np.array([B.diagnostics(s, p)["energy"] for s in tr.states])
    rates = np.diff(np.log(E[2:])) / (p["dt"] * (steps // 8))
    print(f"RT: kinetic energy {['%.1e' % e for e in E]}, log-growth rates {np.round(rates, 2).tolist()}")
    assert E[-1] > 100 * E[1] and (rates > 0).all()


def test_heated_obstacle_plume():
    n, L = 64, 2 * math.pi
    m = cylinder_mask(n, L, (L / 2, L / 4), 0.4)
    p = B.validate({"ic": "zero", "mask": m, "T_body": 1.0})
    steps = int(4 / p["dt"])
    tr = B.rollout(p, B.initial(p, 0), steps, every=steps)
    s = tr.states[-1]
    col = slice(n // 2 - 3, n // 2 + 4)
    above, below = s[1][n // 4 + 8:n // 4 + 20, col].mean(), s[1][n // 4 - 12:n // 4 - 6, col].mean()
    g = B._setup(p)
    u, v = B._vel(np.fft.rfft2(s[0]).astype(np.complex64), g)
    inner = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    speed = np.hypot(np.asarray(u), np.asarray(v))
    print(f"plume: theta above {above:.3f}, below {below:.3f}, inside {s[1][m].mean():.3f}, "
          f"max speed inside {speed[inner].max():.3f} vs max overall {speed.max():.3f}")
    assert above > 0.05 and above > 5 * abs(below) and abs(s[1][m].mean() - 1) < 0.02
    assert speed[inner].max() < 0.2 * speed.max()
