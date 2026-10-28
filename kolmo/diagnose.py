"""Rule-based diagnosis of evaluation results (no learned components). Input: {model_name: evaluate.evaluate() result}
(dicts or the JSON written by the experiment, nulls = non-finite). Output: list of findings
  {"level": "critical" | "warning" | "info", "model", "fact": measured statement, "evidence": the numbers,
   "interpretation": what it suggests (hypothesis, not measurement), "next": suggested next experiment}.
Thresholds are module constants so the rules are inspectable."""
import math

import numpy as np

DRIFT_TOL = 0.05  # |energy drift| above this is reported
SHIFT_RATIO = 2.0  # held-out / test error ratio that counts as distribution shift
RESIDUAL_RATIO = 10.0  # surrogate / reference residual ratio that counts as physically inconsistent
SAME_RATE = 0.25  # relative difference of growth rates treated as "the same"
BETTER = 0.8  # error ratio below which one model is called better at a horizon
SPEC_TOL = 0.1  # dex: time-mean log-spectrum error counted as faithful
STAT_TOL = 0.1  # relative error of time-mean energy/enstrophy counted as faithful


def _arr(x):
    return np.array([np.inf if v is None else v for v in x], float)


def growth_fit(err, threshold, t_min=1):
    """Least-squares fit log(err_t) = a + r t over steps t >= t_min with 0 < err_t <= threshold.
    Returns (r per surrogate step, R^2, number of points) or None if fewer than 5 points."""
    e = _arr(err)
    t = np.arange(len(e))
    ok = (t >= t_min) & np.isfinite(e) & (e > 0) & (e <= threshold)
    if ok.sum() < 5:
        return None
    y = np.log(e[ok])
    r, a = np.polyfit(t[ok], y, 1)
    r2 = 1 - ((y - (a + r * t[ok])) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    return float(r), float(r2), int(ok.sum())


def _f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}g}"


def diagnose_model(model, r):
    out = []
    add = lambda level, fact, ev, interp, nxt: out.append(
        {"level": level, "model": model, "fact": fact, "evidence": ev, "interpretation": interp, "next": nxt})
    test = r["splits"].get("test")
    if test is None:
        return out
    thr, T = test["threshold"], test["steps"]
    e = _arr(test["rel_l2"]["median"])
    fit = growth_fit(e, thr)
    fs = growth_fit(test["first_step_only"]["median"], thr) if "first_step_only" in test else None
    if fit and fit[0] > 0:
        rate, r2, n = fit
        ev = {"rate_per_step": rate, "doubling_steps": math.log(2) / rate, "r2_log_fit": r2, "points": n,
              "err_step1": float(e[1]), "first_step_only_rate": fs[0] if fs else None}
        fact = (f"median test error grows by a factor e^{rate:.3g} per step (doubles every {math.log(2) / rate:.3g} steps; "
                f"log-linear fit R^2={r2:.2f} on {n} steps with error <= {thr}), from {e[1]:.3g} at step 1")
        if r2 < 0.9:
            interp = "growth is not well described by a single exponential (sub- or super-exponential phases)"
        elif fs and fs[0] > 0 and abs(rate - fs[0]) <= SAME_RATE * fs[0]:
            interp = ("growth rate matches the reference solver restarted from the step-1 prediction: error growth is "
                      "dominated by the dynamics amplifying the first-step error, not by later model steps")
        else:
            interp = ("growth rate differs from the first-step-only reference rate "
                      f"({_f(fs[0]) if fs else 'n/a'}/step): the surrogate's own repeated errors drive the growth")
        add("info", fact, ev, interp, "reduce one-step error (capacity/data) if rates match; rollout-aware training or "
            "noise injection if the surrogate's rate is higher")
    ttf = test["time_to_failure"]
    if ttf["censored"] < test["n_traj"]:
        add("warning", f"{test['n_traj'] - ttf['censored']}/{test['n_traj']} test trajectories exceed relative L2 {thr} "
            f"within {T} steps (earliest step {ttf['min']}, median of failing {_f(ttf['median_observed'])})",
            {"per_traj": ttf["per_traj"], "threshold": thr}, "usable autoregressive horizon is bounded by time-to-failure",
            "hybrid re-anchoring at a period below the earliest failure step")
    else:
        add("info", f"no test trajectory exceeds relative L2 {thr} within {T} steps",
            {"max_median_err": float(np.max(e))}, "horizon limit not reached; evaluate longer rollouts", f"evaluate > {T} steps")
    if test["nonphysical_steps"]:
        add("critical", f"{test['nonphysical_steps']} non-finite or blown-up states over the test rollouts",
            {"nonphysical_steps": test["nonphysical_steps"]}, "numerical instability of the learned map",
            "rollout-aware training, smaller spacing, or spectral filtering of the output")
    d = test["energy_drift_final"]
    if d is not None and abs(d) > DRIFT_TOL:
        up = d > 0
        add("warning", f"energy (mean square) at step {T} is {100 * d:+.1f}% vs the reference (max |drift| "
            f"{100 * (test['energy_drift_max_abs'] or float('inf')):.1f}%)",
            {"drift_final": d, "drift_max_abs": test["energy_drift_max_abs"]},
            "spurious energy injection by the surrogate" if up else "spurious dissipation: predictions smooth/blur out",
            "energy-penalty or spectrum loss; check the high-wavenumber spectrum of predictions")
    se, st = test.get("spectrum_err_timemean"), test.get("stats") or {}
    rel = {k: st[k]["mean_rel_err"] for k in ("energy", "enstrophy", "variance") if k in st}
    if se is not None or rel:
        ok = se is not None and se < SPEC_TOL and all(v is not None and abs(v) < STAT_TOL for v in rel.values())
        add("info" if ok else "warning",
            f"rollout statistics over {T} steps: time-mean log-spectrum error {_f(se)} dex (k <= n/3); time-mean "
            "relative error " + ", ".join(f"{k} {_f(v)}" for k, v in rel.items()),
            {"spectrum_err_dex": se, "stats": st},
            "statistically faithful: long rollouts usable for statistics even after pointwise divergence" if ok else
            "long-rollout statistics are biased: the surrogate drifts to a different attractor or amplitude",
            "spectrum or energy-statistics loss term; longer rollout-aware training")
    rr = test.get("residual_ratio_median")
    if rr is not None and rr > RESIDUAL_RATIO:
        add("warning", f"median physics residual of surrogate transitions is {rr:.3g}x the reference solver's own",
            {"residual_ratio": rr}, "transitions are physically inconsistent even where the error looks small",
            "add a residual penalty to the loss or use the residual as a hybrid trigger")
    ho = r["splits"].get("heldout")
    if ho:
        common = [h for h in test["horizons"] if h in ho["horizons"]]
        ratios = {h: (ho["horizons"][h] or math.inf) / max(test["horizons"][h] or math.inf, 1e-30) for h in common}
        worst = max(ratios, key=ratios.get) if ratios else None
        only_ho = ttf["censored"] == test["n_traj"] and ho["time_to_failure"]["censored"] < ho["n_traj"]
        if worst and (ratios[worst] > SHIFT_RATIO or only_ho):
            add("warning", ("fails only on held-out parameters; " if only_ho else "") +
                f"held-out-parameter error is {ratios[worst]:.3g}x the in-distribution test error at step {worst}",
                {"ratio_by_horizon": ratios, "heldout_params": ho.get("params")},
                "distribution shift: the surrogate does not extrapolate in the conditioning parameter",
                "train on a wider parameter range or add the held-out regime; check conditioning is used")
    hyb = [h for h in r["hybrid"] if h["policy"].startswith("hybrid")]
    good = [h for h in hyb if h["max_rel_l2_median"] is not None and h["max_rel_l2_median"] <= thr]
    if good:
        h = min(good, key=lambda h: h["cost_fraction"])
        add("info", f"{h['policy']} keeps median relative L2 below {_f(h['max_rel_l2_median'])} for {T} steps at "
            f"{100 * h['cost_fraction']:.1f}% of reference solver steps",
            {"policy": h["config"], "cost_fraction": h["cost_fraction"], "max_rel_l2_median": h["max_rel_l2_median"],
             "interventions": h["interventions_by_reason"]},
            "cheapest tested hybrid policy that stays under the failure threshold", "bisect the policy period/threshold "
            "between this and the next cheaper policy")
    elif hyb:
        add("warning", f"no tested hybrid policy keeps median relative L2 below {thr} for {T} steps",
            {h["policy"]: [h["cost_fraction"], h["max_rel_l2_median"]] for h in hyb},
            "re-anchoring from the drifted state cannot remove accumulated (or chaotic) divergence",
            "more frequent anchors, or compare against statistics rather than pointwise error past predictability")
    return out


def compare(results):
    """Pairwise: every model vs the first (baseline) on the test split, horizon by horizon."""
    names = list(results)
    out = []
    if len(names) < 2:
        return out
    base = results[names[0]]["splits"].get("test")
    for m in names[1:]:
        t = results[m]["splits"].get("test")
        if base is None or t is None:
            continue
        ratios = {h: (t["horizons"][h] or math.inf) / max(base["horizons"][h] or math.inf, 1e-30)
                  for h in base["horizons"] if h in t["horizons"]}
        better = [h for h, q in ratios.items() if q < BETTER]
        worse = [h for h, q in ratios.items() if q > 1 / BETTER]
        tb, tm = base["time_to_failure"], t["time_to_failure"]
        fact = (f"{m} / {names[0]} test error ratio by horizon: " + ", ".join(f"{h}: {q:.3g}" for h, q in ratios.items()) +
                f"; failing trajectories {t['n_traj'] - tm['censored']}/{t['n_traj']} vs "
                f"{base['n_traj'] - tb['censored']}/{base['n_traj']}")
        if better and not worse:
            interp, level = f"{m} is better at horizons {better} and not worse anywhere", "info"
        elif better and worse:
            interp, level = f"trade-off: {m} better at {better}, worse at {worse}", "info"
        elif worse:
            interp, level = f"{m} is worse at horizons {worse} and better nowhere", "warning"
        else:
            interp, level = f"no difference beyond a factor {1 / BETTER:.2g} at any horizon", "info"
        out.append({"level": level, "model": f"{m} vs {names[0]}", "fact": fact,
                    "evidence": {"ratios": ratios, "ttf": {names[0]: tb["per_traj"], m: tm["per_traj"]}},
                    "interpretation": interp, "next": "repeat with more training seeds to separate effect from noise"})
    return out


def diagnose(results):
    return [f for m, r in results.items() for f in diagnose_model(m, r)] + compare(results)


def to_markdown(findings):
    lines = []
    for f in findings:
        lines += [f"- **[{f['level']}] {f['model']}**", f"  - Measured: {f['fact']}",
                  f"  - Interpretation: {f['interpretation']}", f"  - Next: {f['next']}"]
    return "\n".join(lines)
