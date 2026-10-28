"""Assistant tools: thin, structured wrappers over real Kolmo code (kolmo.surrogates.data systems, kolmo.experiments,
kolmo.diagnose). Every result is a JSON-safe dict; the agent passes it to the model verbatim, so every number the
model may quote comes from here. Resource limits are enforced here, not trusted to the model."""
import json
import os
import time
from dataclasses import asdict

from kolmo import diagnose as dg
from kolmo import evaluate as ev
from kolmo import experiments
from kolmo.surrogates import data

LIMITS = {"max_grid": 128, "max_data_steps": 400, "max_train_steps": 5000, "max_models": 4,
          "confirm_seconds": 60, "max_seconds": 1800}
# Measured on this machine (2-core shared CPU, JAX CPU, float32); see docs/assistant.md for the sources.
TRAIN_S_PER_CELL_SAMPLE = 7e-6  # runs/ui-tiny-diffusion-v2: cnn w8 d2, 300 steps x batch 8 x 32^2 cells in 17.1 s
REF_S_PER_CELL_STEP = {"diffusion": 1.3e-7, "fluid2d": 6e-7}  # 1.36e-4 s/step at 32^2; ~400 steps/s at 64^2 (physics.md)
OVERHEAD_S = {"run": 10.0, "model": 8.0}  # imports, data cache, plots / per-model compile + evaluation


def list_systems():
    out = {}
    for name in data.SYSTEMS:
        s = data.get_system(name)
        out[name] = {"fields": list(s.fields), "defaults": {k: (list(v) if isinstance(v, tuple) else v)
                                                            for k, v in s.DEFAULTS.items() if k != "mask"}}
    return out


def _cfg(config, out_dir):
    c = experiments.ExperimentConfig(**{**config, "out_dir": out_dir})
    d, m, h = c.parsed()
    return c, d, m


def validate_config(config, out_dir="runs"):
    """Kolmo's own validation (ExperimentConfig.parsed + the system's validate) plus the assistant's limits."""
    try:
        c, d, m = _cfg(config, out_dir)
        p = data.get_system(d.system).validate(d.params)
        problems = [f"grid n={p['n']} exceeds max_grid {LIMITS['max_grid']}" if p["n"] > LIMITS["max_grid"] else "",
                    f"data steps*spacing={d.steps * d.spacing} exceeds {LIMITS['max_data_steps']}"
                    if d.steps * d.spacing > LIMITS["max_data_steps"] else "",
                    f"{len(m)} models exceed max_models {LIMITS['max_models']}" if len(m) > LIMITS["max_models"] else ""]
        problems += [f"model {k}: {t.steps} training steps exceed {LIMITS['max_train_steps']}"
                     for k, t in m.items() if t.steps > LIMITS["max_train_steps"]]
        problems = [x for x in problems if x]
        if problems:
            return {"ok": False, "error": "; ".join(problems), "limits": LIMITS}
        return {"ok": True, "config": asdict(c), "system_params": {k: v for k, v in p.items() if k != "mask"}}
    except (ValueError, TypeError, KeyError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def estimate_cost(config, out_dir="runs"):
    """Seconds ~ overhead + reference data generation + training, from the measured rates above (order of magnitude)."""
    try:
        _, d, m = _cfg(config, out_dir)
        n = data.get_system(d.system).validate(d.params)["n"]
    except (ValueError, TypeError, KeyError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    cells = n * n
    nvary = max([len(v) for v in d.vary.values()] or [1])
    traj = (len(d.train_seeds) + len(d.val_seeds) + len(d.test_seeds)) * nvary + len(d.test_seeds) * len(next(iter(d.heldout.values()), []))
    ref = traj * (d.burn_in + d.steps * d.spacing) * cells * REF_S_PER_CELL_STEP.get(d.system, 6e-7)
    train = sum(t.steps * t.batch * t.unroll for t in m.values()) * cells * TRAIN_S_PER_CELL_SAMPLE
    s = OVERHEAD_S["run"] + OVERHEAD_S["model"] * len(m) + ref + train
    return {"ok": True, "seconds": round(s, 1), "breakdown": {"reference_data_s": round(ref, 2), "training_s": round(train, 1),
            "overhead_s": OVERHEAD_S["run"] + OVERHEAD_S["model"] * len(m)}, "grid_cells": cells, "trajectories": traj,
            "needs_confirmation": s > LIMITS["confirm_seconds"], "limits": LIMITS}


def run_experiment(config, out_dir="runs", confirmed=False):
    """Validates, estimates, then runs kolmo.experiments.run. `confirmed` comes from the user, never from the model."""
    v = validate_config(config, out_dir)
    if not v["ok"]:
        return v
    e = estimate_cost(config, out_dir)
    if e["seconds"] > LIMITS["max_seconds"]:
        return {"ok": False, "error": f"estimated {e['seconds']} s exceeds max_seconds {LIMITS['max_seconds']}"}
    if e["needs_confirmation"] and not confirmed:
        return {"ok": False, "needs_confirmation": True, "estimated_seconds": e["seconds"],
                "error": "expensive run: waiting for the user's confirmation"}
    t = time.perf_counter()
    out, _, findings = experiments.run(experiments.ExperimentConfig(**v["config"]), verbose=False)
    return {"ok": True, "run_dir": out, "seconds": round(time.perf_counter() - t, 1), "estimated_seconds": e["seconds"],
            "n_findings": len(findings)}


def _results(run_dir):
    with open(os.path.join(run_dir, "results.json")) as f:
        return json.load(f)


def _sig(x):
    return float(f"{x:.4g}") if isinstance(x, float) else x


def read_results(run_dir):
    """Compact view of results.json: generalisation table, runtime, training summary (4 significant digits)."""
    try:
        r = _results(run_dir)
    except OSError as e:
        return {"ok": False, "error": str(e)}
    table = [{k: _sig(v) for k, v in row.items()} for row in ev.table(r)]
    runtime = {m: {k: _sig(v[k]) for k in ("reference_s_per_step", "surrogate_s_per_step", "speedup")} for m, v in
               ((m, x["runtime"]) for m, x in r.items())}
    training = {m: {"train_minutes": _sig(x["training"]["train_minutes"]), "evaluations": len(x["training"]["history"]),
                    "last_step": x["training"]["history"][-1]["step"]} for m, x in r.items()}
    hybrid = {m: [{"policy": h["policy"], "cost_fraction": _sig(h["cost_fraction"]),
                   "max_rel_l2_median": _sig(h["max_rel_l2_median"])} for h in x["hybrid"]] for m, x in r.items() if x["hybrid"]}
    return {"ok": True, "table": table, "runtime": runtime, "training": training, "hybrid": hybrid}


def diagnose(run_dir):
    """kolmo.diagnose rules over the run's results.json (facts = measured, interpretation = hypothesis)."""
    try:
        f = dg.diagnose(_results(run_dir))
    except OSError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "findings": json.loads(json.dumps(ev.clean(
        [{k: x[k] for k in ("level", "model", "fact", "interpretation", "next", "evidence")} for x in f])))}


FUNCS = {"list_systems": list_systems, "validate_config": validate_config, "estimate_cost": estimate_cost,
         "run_experiment": run_experiment, "read_results": read_results, "diagnose": diagnose}
_CFG = {"type": "object", "description": "kolmo.experiments.ExperimentConfig: {name, data: DataConfig fields (system, params, "
        "steps, spacing, train_seeds, val_seeds, test_seeds, vary, heldout, cond), models: {name: TrainConfig fields "
        "(model: {kind: cnn|fno, ...}, unroll, steps, batch, ...)}, hybrid, eval_steps, threshold}"}
_RUN = {"type": "object", "properties": {"run_dir": {"type": "string"}}, "required": ["run_dir"]}
_ARGS = {"list_systems": {"type": "object", "properties": {}},
         "validate_config": {"type": "object", "properties": {"config": _CFG}, "required": ["config"]},
         "read_results": _RUN, "diagnose": _RUN}
_ARGS["estimate_cost"] = _ARGS["run_experiment"] = _ARGS["validate_config"]
SCHEMAS = [{"type": "function", "function": {"name": k, "description": (f.__doc__ or k).split("\n")[0], "parameters": _ARGS[k]}}
           for k, f in FUNCS.items()]


def call(name, args, out_dir="runs", confirmed=False):
    """Dispatch a model tool call. out_dir and confirmation are set by the agent; model-supplied values are ignored."""
    if name not in FUNCS:
        return {"ok": False, "error": f"unknown tool {name!r}; available: {sorted(FUNCS)}"}
    a = {k: v for k, v in (args or {}).items() if k in ("config", "run_dir")}
    if name in ("validate_config", "estimate_cost", "run_experiment"):
        a["out_dir"] = out_dir
    if name == "run_experiment":
        a["confirmed"] = confirmed
    try:
        return FUNCS[name](**a)
    except TypeError as e:
        return {"ok": False, "error": f"bad arguments for {name}: {e}"}
