"""Hybrid study on the Kolmogorov 64^2 test split: JEPA latent vs 4-step CNN vs 1-step CNN under kolmo.hybrid policies.

For the latent model, hybrid re-anchoring = latent rollout until the solver step, then re-encode the solver state
(LatentSurrogate.stepper re-encodes any input that is not its own last output). Errors: all test trajectories, one
run each (deterministic). Timing: test trajectory 0, all policies + pure reference run back to back, repeated 3x,
median of compile-excluded wall seconds. Run (repo root): PYTHONPATH=. python benchmarks/latent_hybrid_study.py [OUT_DIR]"""
import json
import os
import shutil
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kolmo.evaluate import rel_l2, write_json
from kolmo.hybrid import HybridPolicy, cost, run
from kolmo.surrogates import data, latent, train

BASE = "runs/kolmogorov64_long_horizon-88294366/checkpoints"
MODELS = {"jepa_latent": lambda: latent.load("runs/latent_kolmogorov64/jepa/latent")[0],
          "cnn_rollout4": lambda: train.load(f"{BASE}/cnn_rollout4")[0],
          "cnn_one_step": lambda: train.load(f"{BASE}/cnn_one_step")[0]}
POLICIES = [HybridPolicy("surrogate")] + [HybridPolicy(every=k) for k in (20, 10, 5, 2)] + [HybridPolicy(energy_tol=0.2)]


def main(out="runs/latent_hybrid_study"):
    with open("benchmarks/long_horizon.json") as f:
        cfg = json.load(f)
    ds = data.build(data.DataConfig(**cfg["data"]), "runs/cache")
    system = data.get_system(ds.config.system)
    sp = ds.splits["test"]
    T = cfg["eval_steps"]
    sim_time = T * ds.config.spacing * sp.params[0]["dt"]
    surs = {m: f() for m, f in MODELS.items()}
    rows = []
    for m, sur in surs.items():
        for pol in POLICIES:
            trs = [run(pol, system, sp.params[i], sur, sp.states[i, 0], T) for i in range(len(sp.seeds))]
            med = np.median([rel_l2(tr.states, sp.states[i, :T + 1]) for i, tr in enumerate(trs)], axis=0)
            nonphys = sum(int((np.abs(np.nan_to_num(tr.states, nan=np.inf)).max((1, 2, 3)) >
                               10 * np.abs(sp.states[i, :T + 1]).max()).sum()) for i, tr in enumerate(trs))
            rows.append({"model": m, "policy": pol.name, "max_rel_l2": float(np.max(med)),
                         "mean_rel_l2": float(np.mean(med[1:])), "cost_fraction": float(np.mean([cost(tr, ds.config.spacing)
                         ["fraction_of_reference"] for tr in trs])), "nonphysical_steps": nonphys, "seconds": []})
            print(m, pol.name, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
    ref_s = []
    for rep in range(3):  # ponytail: interleaved repeats share machine load; median of 3 is the noise control
        ref_s.append(run(HybridPolicy("reference"), system, sp.params[0], surs["cnn_one_step"], sp.states[0, 0], T).seconds)
        for r in rows:
            pol = next(p for p in POLICIES if p.name == r["policy"])
            r["seconds"].append(run(pol, system, sp.params[0], surs[r["model"]], sp.states[0, 0], T).seconds)
    ref = float(np.median(ref_s))
    for r in rows:
        r["wall_s_per_time_unit"] = float(np.median(r["seconds"])) / sim_time
        r["speedup_vs_reference"] = ref / float(np.median(r["seconds"]))
    res = {"reference_wall_s_per_time_unit": ref / sim_time, "reference_seconds": ref_s, "sim_time_units": sim_time,
           "n_traj": len(sp.seeds), "steps": T, "rows": rows}
    os.makedirs(out, exist_ok=True)
    write_json(res, f"{out}/results.json")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for (m, c), mk in zip({"jepa_latent": "C0", "cnn_rollout4": "C1", "cnn_one_step": "C2"}.items(), "os^"):
        rs = sorted((r for r in rows if r["model"] == m), key=lambda r: r["cost_fraction"])
        ax.plot([r["cost_fraction"] for r in rs], [r["mean_rel_l2"] for r in rs], "-" + mk, color=c, label=f"{m} (mean)")
        ax.plot([r["cost_fraction"] for r in rs], [r["max_rel_l2"] for r in rs], ":" + mk, color=c, mfc="none",
                label=f"{m} (max)")
    ax.axhline(np.sqrt(2), color="grey", lw=0.8, ls="--", label="decorrelated, right energy")
    ax.set(yscale="log", xlabel="fraction of steps taken by the solver (reference = 1, error 0)",
           ylabel="median relL2 over 200 steps", title="Kolmogorov 64², test split: error vs solver cost")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/pareto.png", dpi=130)
    os.makedirs("docs/images", exist_ok=True)
    shutil.copy(f"{out}/pareto.png", "docs/images/pareto.png")
    print(f"reference {ref / sim_time:.4f} s per time unit; wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
