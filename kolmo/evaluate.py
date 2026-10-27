"""Evaluation of surrogate rollouts against reference trajectories. All metrics are computed per trajectory and then
aggregated over trajectories (median and mean; a non-finite value counts as +inf, JSON writes null for non-finite).

Definitions (s = predicted state, r = reference state at the same surrogate step t, sums over all C x H x W values):
  rel_l2[t]        = ||s_t - r_t||_2 / ||r_t||_2, dimensionless (denominator: reference norm at the same step)
  rel_l2_field[t,c]= same, restricted to channel c
  corr[t]          = Pearson correlation of s_t and r_t over all values (1 = perfect pattern match, scale-blind)
  horizons         = rel_l2 at steps 1, 10, 50, 100, 200, 500, 1000 (those <= T)
  time_to_failure  = first step t >= 1 with rel_l2[t] > threshold (default 0.5) or a non-finite state; null if none
                     within the T evaluated steps (censored; the count of censored trajectories is reported)
  residual[t]      = system.residual(s_{t-1}, s_t, params with dt -> spacing*dt): physical inconsistency of the
                     transition, dimensionless; the same for the reference transition is the solver's own floor
  energy[t]        = mean of state^2 (rollout.mean_square, field units^2); energy_drift[t] = E_s[t] / E_r[t] - 1
  nonphysical      = number of steps whose state is non-finite or has max|s| > blowup * max_t |r_t|
  first_step_only  = rel_l2 of the REFERENCE solver restarted from the surrogate's step-1 prediction: the error a
                     perfect model would show after inheriting only the first-step error (chaotic amplification
                     floor); the gap to rel_l2 is error added by the surrogate's later steps
  runtime          = steady-state wall seconds per surrogate step (compile excluded) for reference (spacing solver
                     steps, fused) and surrogate, batch 1, float32 on this CPU; speedup = reference / surrogate"""
import json
import math
import platform

import numpy as np

from kolmo.hybrid import HybridPolicy, cost, run
from kolmo.rollout import coarse, mean_square, rollout

HORIZONS = (1, 10, 50, 100, 200, 500, 1000)


def rel_l2(pred, ref, axes=(-3, -2, -1)):
    p, r = np.asarray(pred, np.float64), np.asarray(ref, np.float64)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        e = np.sqrt(((p - r) ** 2).sum(axes) / (r ** 2).sum(axes))
    return np.where(np.isfinite(e), e, np.inf)


def correlation(pred, ref):
    p = np.asarray(pred, np.float64).reshape(len(pred), -1)
    r = np.asarray(ref, np.float64).reshape(len(ref), -1)
    p, r = p - p.mean(1, keepdims=True), r - r.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        return (p * r).sum(1) / np.sqrt((p * p).sum(1) * (r * r).sum(1))


def time_to_failure(err, threshold=0.5):
    bad = ~(np.asarray(err)[1:] <= threshold)
    return int(np.argmax(bad)) + 1 if bad.any() else None


def _agg(rows):
    a = np.asarray(rows, np.float64)
    a = np.where(np.isnan(a), np.inf, a)
    with np.errstate(invalid="ignore"):
        return {"median": np.median(a, 0), "mean": a.mean(0)}


def _at(series, T):
    return {str(h): series[h] for h in HORIZONS if h <= T}


def trajectory_metrics(system, pred, ref, params, spacing, threshold=0.5, blowup=10.0, with_residual=True):
    """Per-trajectory metric dict for pred, ref of shape (T+1, C, H, W)."""
    T = len(ref) - 1
    pc = {**params, "dt": params["dt"] * spacing}
    err = rel_l2(pred, ref)
    finite = np.isfinite(pred).all(axis=(1, 2, 3))
    with np.errstate(invalid="ignore"):
        peak = np.where(finite, np.abs(np.nan_to_num(pred)).max(axis=(1, 2, 3)), np.inf)
    m = {"rel_l2": err, "rel_l2_field": rel_l2(pred, ref, (-2, -1)), "corr": correlation(pred, ref),
         "ttf": time_to_failure(err, threshold),
         "energy_pred": mean_square(pred), "energy_ref": mean_square(ref),
         "nonphysical": int((peak > blowup * np.abs(ref).max()).sum())}
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        m["energy_drift"] = m["energy_pred"] / m["energy_ref"] - 1
    if with_residual:
        res = lambda a, b: float(system.residual(a, b, pc)) if np.isfinite(b).all() else np.inf
        try:
            m["residual_pred"] = np.array([res(pred[t - 1], pred[t]) for t in range(1, T + 1)])
            m["residual_ref"] = np.array([res(ref[t - 1], ref[t]) for t in range(1, T + 1)])
        except ValueError as e:  # e.g. a solver whose validate() refuses dt = spacing * dt (CFL)
            m.pop("residual_pred", None)
            m["residual_unavailable"] = str(e)
    m["diagnostics"] = {str(h): {"pred": system.diagnostics(pred[h], params) if finite[h] else None,
                                 "ref": system.diagnostics(ref[h], params)} for h in (0,) + HORIZONS if h <= T}
    return m


def summarize(ms, T, threshold):
    out = {"n_traj": len(ms), "steps": T, "threshold": threshold}
    for key in ("rel_l2", "corr", "energy_pred", "energy_ref", "energy_drift", "residual_pred", "residual_ref",
                "first_step_only"):
        if key in ms[0]:
            out[key] = _agg([m[key] for m in ms])
    if "residual_unavailable" in ms[0]:
        out["residual_unavailable"] = ms[0]["residual_unavailable"]
    out["rel_l2_field"] = _agg([m["rel_l2_field"] for m in ms])["median"]  # (T+1, C)
    out["horizons"] = _at(out["rel_l2"]["median"], T)
    ttf = [m["ttf"] for m in ms]
    hit = [t for t in ttf if t is not None]
    out["time_to_failure"] = {"per_traj": ttf, "censored": ttf.count(None),
                              "median_observed": float(np.median(hit)) if hit else None,
                              "min": min(hit) if hit else None}
    d = out["energy_drift"]["median"]
    out["energy_drift_final"] = float(d[-1])
    out["energy_drift_max_abs"] = float(np.max(np.abs(d)))
    out["nonphysical_steps"] = int(sum(m["nonphysical"] for m in ms))
    if "residual_pred" in out:
        out["residual_ratio_median"] = float(np.median(out["residual_pred"]["median"]) /
                                             max(np.median(out["residual_ref"]["median"]), 1e-30))
    diag = [m["diagnostics"] for m in ms]
    out["diagnostics"] = {h: {who: _mean_dicts([d[h][who] for d in diag]) for who in ("pred", "ref")} for h in diag[0]}
    return out


def _mean_dicts(ds):
    ds = [d for d in ds if d is not None]
    return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]} if ds else None


def evaluate(system, ds, sur, splits=("test", "heldout"), steps=None, threshold=0.5, policies=(), blowup=10.0,
             verbose=True):
    """Full evaluation of one surrogate on dataset splits plus hybrid policies (on the test split)."""
    res = {"definitions": __doc__, "splits": {}, "hybrid": []}
    runtimes = []
    for name in [s for s in splits if s in ds.splits]:
        sp = ds.splits[name]
        T = sp.states.shape[1] - 1 if steps is None else min(steps, sp.states.shape[1] - 1)
        ms = []
        for i in range(len(sp.seeds)):
            p, ref = sp.params[i], sp.states[i, :T + 1]
            tr = rollout(sur.stepper(p), ref[0], T, "surrogate")
            runtimes.append(tr)
            m = trajectory_metrics(system, tr.states, ref, p, sur.spacing, threshold, blowup)
            fs = rollout(coarse(system.stepper(system.validate(p)), sur.spacing), tr.states[1], T - 1).states
            m["first_step_only"] = np.concatenate([[0.0], rel_l2(fs, ref[1:])])
            ms.append(m)
        res["splits"][name] = summarize(ms, T, threshold)
        res["splits"][name]["params"] = [{k: p[k] for k in ds.config.vary} for p in sp.params]
        if verbose:
            s = res["splits"][name]
            print(f"  {name}: relL2 @ " + ", ".join(f"{h}: {v:.3g}" for h, v in s["horizons"].items()) +
                  f" | ttf censored {s['time_to_failure']['censored']}/{s['n_traj']}", flush=True)
    sp = ds.splits["test"]
    T = sp.states.shape[1] - 1 if steps is None else min(steps, sp.states.shape[1] - 1)
    ref_run = run(HybridPolicy("reference"), system, sp.params[0], sur, sp.states[0, 0], T)
    sur_s = float(np.median([tr.seconds / max(len(tr.source), 1) for tr in runtimes]))
    ref_s = ref_run.seconds / T
    res["runtime"] = {"reference_s_per_step": ref_s, "reference_compile_s": ref_run.compile_seconds,
                      "surrogate_s_per_step": sur_s,
                      "surrogate_compile_s": float(np.median([tr.compile_seconds for tr in runtimes])),
                      "speedup": ref_s / sur_s,
                      "conditions": f"{platform.processor() or platform.machine()}, CPU JAX, batch 1, float32, grid "
                                    f"{tuple(sp.states.shape[-2:])}, 1 surrogate step = {sur.spacing} solver steps, "
                                    "compile excluded, machine shared with other processes"}
    for pol in policies:
        try:
            trs = [run(pol, system, sp.params[i], sur, sp.states[i, 0], T) for i in range(len(sp.seeds))]
        except ValueError as e:  # e.g. residual trigger refused by the system at this spacing
            res.setdefault("hybrid_unavailable", []).append({"policy": pol.name, "config": vars(pol), "reason": str(e)})
            if verbose:
                print(f"  {pol.name}: unavailable ({e})", flush=True)
            continue
        errs = [rel_l2(tr.states, sp.states[i, :T + 1]) for i, tr in enumerate(trs)]
        agg = _agg(errs)
        c = [cost(tr, sur.spacing) for tr in trs]
        res["hybrid"].append({
            "policy": pol.name, "config": vars(pol), "rel_l2": agg, "horizons": _at(agg["median"], T),
            "max_rel_l2_median": float(np.max(agg["median"])),
            "time_to_failure": [time_to_failure(e, threshold) for e in errs],
            "cost_fraction": float(np.mean([x["fraction_of_reference"] for x in c])),
            "solver_steps_mean": float(np.mean([x["solver_steps"] for x in c])), "reference_solver_steps": T * sur.spacing,
            "interventions_by_reason": _count([e["reason"] for tr in trs for e in tr.events if e["kind"] == "intervention"]),
            "s_per_step": float(np.median([tr.seconds / T for tr in trs]))})
        if verbose:
            h = res["hybrid"][-1]
            print(f"  {h['policy']}: max median relL2 {h['max_rel_l2_median']:.3g} at {100 * h['cost_fraction']:.1f}% "
                  "of reference solver steps", flush=True)
    return res


def _count(xs):
    return {x: xs.count(x) for x in sorted(set(xs))}


def table(results):
    """Generalisation table rows: one per (model, split)."""
    rows = []
    for model, r in results.items():
        for split, s in r["splits"].items():
            rows.append({"model": model, "split": split, "n_traj": s["n_traj"], **{f"relL2@{h}": v for h, v in s["horizons"].items()},
                         "ttf_median": s["time_to_failure"]["median_observed"],
                         "ttf_censored": f"{s['time_to_failure']['censored']}/{s['n_traj']}",
                         "energy_drift_final": s["energy_drift_final"], "nonphysical_steps": s["nonphysical_steps"]})
    return rows


def clean(x):
    """JSON-safe copy: numpy -> lists/floats, non-finite -> None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in (x.tolist() if isinstance(x, np.ndarray) else x)]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(clean(obj), f, indent=1)


def plot(results, outdir):
    """error_vs_horizon.png, energy_vs_time.png, hybrid_tradeoff.png for {model_name: evaluate() result}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, r in results.items():
        for split, s in r["splits"].items():
            e = np.asarray(s["rel_l2"]["median"], float)
            ax.semilogy(np.arange(1, len(e)), e[1:], "-" if split == "test" else "--", label=f"{model} / {split}")
        if "test" in r["splits"]:
            f = np.asarray(r["splits"]["test"]["first_step_only"]["median"], float)
            ax.semilogy(np.arange(1, len(f)), f[1:], ":", lw=1, label=f"{model} / test, first-step-only")
    ax.axhline(r["splits"]["test"]["threshold"] if "test" in r["splits"] else 0.5, color="k", lw=0.8)
    ax.set(xlabel="surrogate step", ylabel="relative L2 error (median over trajectories)", title="Error vs horizon")
    ax.legend(fontsize=7); fig.tight_layout()
    files.append(f"{outdir}/error_vs_horizon.png"); fig.savefig(files[-1], dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    first = True
    for model, r in results.items():
        s = r["splits"].get("test")
        if s is None:
            continue
        if first:
            ax.plot(s["energy_ref"]["median"], "k-", lw=2, label="reference")
            first = False
        ax.plot(s["energy_pred"]["median"], label=model)
    ax.set(xlabel="surrogate step", ylabel="mean square of state (median over test trajectories)", title="Energy vs time")
    ax.legend(fontsize=8); fig.tight_layout()
    files.append(f"{outdir}/energy_vs_time.png"); fig.savefig(files[-1], dpi=120); plt.close(fig)

    if any(r["hybrid"] for r in results.values()):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, r in results.items():
            xs = [100 * h["cost_fraction"] for h in r["hybrid"]]
            ys = [h["max_rel_l2_median"] for h in r["hybrid"]]
            ax.plot(xs, ys, "o-", label=model)
            for x, y, h in zip(xs, ys, r["hybrid"]):
                ax.annotate(h["policy"], (x, y), fontsize=6)
        ax.set(xlabel="solver steps used (% of reference)", ylabel="max over steps of median relative L2", yscale="log",
               title="Hybrid cost vs error")
        ax.legend(fontsize=8); fig.tight_layout()
        files.append(f"{outdir}/hybrid_tradeoff.png"); fig.savefig(files[-1], dpi=120); plt.close(fig)
    return files
