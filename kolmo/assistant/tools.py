"""Assistant tools over real Kolmo code (kolmo.systems, kolmo.surrogates, kolmo.evaluate, kolmo.diagnose).

A Task holds one research task's memory (specs, datasets, models, results), budgets and event emitter. Every tool
returns a JSON-safe dict with "ok"; failures come back as {"ok": false, "error"} so the model can fix its call.
The task directory is laid out like an experiment run (checkpoints/, results.json, report.md, *.png) so the model
registry can register its models. Recovery is automatic and reported in "recovered":
  generate_data: unstable / non-finite reference data -> halve dt (up to 3 times);
  train: non-finite loss or validation error -> lr / 3 and unroll 1, retried once."""
import json
import math
import os
import time

import jax
import numpy as np

from kolmo import diagnose as dg
from kolmo import evaluate as ev
from kolmo.surrogates import data, registry, train as tr

LIMITS = {"max_grid": 128, "max_traj_steps": 400, "max_trajectories": 64, "max_train_minutes": 15}
KINDS = {"cnn": {"kind": "cnn", "width": 16, "depth": 3}, "fno": {"kind": "fno", "width": 16, "modes": 8, "depth": 3},
         "nca_hybrid": {"kind": "nca_hybrid", "hidden": 8, "width": 48, "modes": 8},
         "latent": {"kind": "latent", "latent": 24, "patch": 4, "width": 24}}


def _systems():
    from kolmo import systems
    return systems


def _pde():
    from kolmo.systems import pde
    return pde


def _short(x, n=300):
    s = json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + "…"


class Task:
    def __init__(self, dir, root, emit, max_compute_s, should_stop=None):
        self.dir, self.root, self.emit, self.max_compute_s, self.should_stop = dir, root, emit, max_compute_s, should_stop
        self.calls, self.compute_s = 0, 0.0
        self.sources, self.removed, self.figures, self.report = [], [], [], None
        self.specs, self.datasets, self.models, self.results = {}, {}, {}, {}
        os.makedirs(os.path.join(dir, "checkpoints"), exist_ok=True)

    def left(self):
        return self.max_compute_s - self.compute_s

    def call(self, name, args):
        self.calls += 1
        t = time.perf_counter()
        if name not in FUNCS:
            res = {"ok": False, "error": f"unknown tool {name!r}; available: {sorted(FUNCS)}"}
        elif self.left() <= 0 and name in ("generate_data", "train", "evaluate"):
            res = {"ok": False, "error": "compute budget exhausted: compare, write_report and answer with what you have"}
        elif set(args or {}) - set(_ARGS[name][0]) or set(_ARGS[name][1]) - set(args or {}):
            res = {"ok": False, "error": f"bad arguments for {name}: unknown {sorted(set(args or {}) - set(_ARGS[name][0]))}, "
                                         f"missing {sorted(set(_ARGS[name][1]) - set(args or {}))}; allowed {sorted(_ARGS[name][0])}"}
        else:
            try:
                res = FUNCS[name](self, **(args or {}))
            except (ValueError, TypeError, KeyError, RuntimeError, FloatingPointError, SyntaxError) as e:
                res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.compute_s += time.perf_counter() - t
        res = json.loads(json.dumps(ev.clean(res), default=str))
        res["budget"] = {"tool_calls_used": self.calls, "compute_s_used": round(self.compute_s),
                         "compute_s_left": max(0, round(self.left()))}
        self.sources.append(json.dumps(res))
        self.emit({"type": "tool", "name": name, "args": args, "ok": res.get("ok", False),
                   "summary": res.get("error") or res.get("summary") or _short({k: v for k, v in res.items() if k != "budget"})})
        return res

    def metrics(self):
        return {m: _compact(r)["test"] for m, r in self.results.items() if "test" in r["splits"]}

    def system(self, ref):
        return data.get_system(self.specs.get(ref, ref))


# -------------------------------------------------------------------------------------------------------- tools
def plan(task, steps):
    """Announce (or revise) the plan: a list of short steps."""
    task.emit({"type": "plan", "steps": [str(s) for s in steps]})
    return {"ok": True, "summary": f"plan with {len(steps)} steps"}


def set_zen(task, on):
    """Ask the UI to enter (true) or leave (false) zen mode during long work."""
    task.emit({"type": "zen", "on": bool(on)})
    return {"ok": True, "zen": bool(on)}


def list_systems(task):
    """Registered systems and the PDE library."""
    out = {}
    for name in _systems().SYSTEMS:
        try:
            d = data.get_system(name).describe()
            out[name] = {"fields": d["fields"], "equations": d["equations"]}
        except Exception as e:  # noqa: BLE001 (the spec-driven "pde" system needs a spec)
            out[name] = {"note": f"needs a spec: use compose_pde ({type(e).__name__})"}
    lib = getattr(_pde(), "LIBRARY", {})
    return {"ok": True, "systems": out, "pde_library": {k: v.get("rhs") for k, v in lib.items()},
            "summary": f"{len(out)} systems, {len(lib)} library PDEs"}


def describe_system(task, name):
    """Equations, fields, parameters (default, min, max, unit, help) and guaranteed invariants of a system."""
    return {"ok": True, **task.system(name).describe()}


def compose_pde(task, spec):
    """Compose a PDE system from a spec (or a library name, optionally with overrides); validated by kolmo.systems.make."""
    if isinstance(spec, str):
        spec = dict(_pde().LIBRARY[spec])
    elif "rhs" not in spec and spec.get("name") in getattr(_pde(), "LIBRARY", {}):
        spec = {**_pde().LIBRARY[spec["name"]], **spec}
    s = data.get_system(spec)
    s.validate({})  # stiffness / stability of the spec's own parameters
    d = s.describe()
    task.specs[spec["name"]] = spec
    return {"ok": True, "system": spec["name"], "spec": spec, "describe": d,
            "summary": f"composed {spec['name']}: fields {d['fields']}"}


def generate_data(task, system, params=None, steps=40, spacing=1, burn_in=0, n_train=8, n_val=2, n_test=3,
                  vary=None, heldout=None):
    """Reference trajectories on disjoint train/val/test seeds. Unstable dt is halved automatically."""
    spec = task.specs.get(system)
    params = dict(params or {})
    sysobj = task.system(system)
    n = int(sysobj.validate(params)["n"])
    if n > LIMITS["max_grid"] or steps * spacing + burn_in > LIMITS["max_traj_steps"] or \
            n_train + n_val + n_test > LIMITS["max_trajectories"]:
        return {"ok": False, "error": f"over limits {LIMITS} (n={n}, solver steps={steps * spacing + burn_in}, "
                                      f"trajectories={n_train + n_val + n_test})"}
    recovered = []
    for attempt in range(4):
        cfg = data.DataConfig(system=spec or system, params=params, steps=int(steps), spacing=int(spacing),
                              burn_in=int(burn_in), train_seeds=list(range(n_train)),
                              val_seeds=list(range(100, 100 + n_val)), test_seeds=list(range(200, 200 + n_test)),
                              vary=vary or {}, heldout=heldout or {})
        try:
            ds = data.build(cfg.validate(), os.path.join(task.root, "cache"), verbose=False)
            if all(np.isfinite(sp.states).all() and np.abs(sp.states).max() < 1e6 for sp in ds.splits.values()):
                break
            err = "non-finite or exploding reference states"
        except (ValueError, FloatingPointError) as e:
            err = str(e)
            if not any(w in err.lower() for w in ("dt", "stab", "cfl", "stiff", "finite")):
                raise
        if attempt == 3:
            return {"ok": False, "error": f"reference solver unstable after halving dt 3 times: {err}", "recovered": recovered}
        params["dt"] = float(sysobj.validate(params)["dt"]) / 2
        recovered.append(f"{err[:120]} -> dt halved to {params['dt']:.4g}")
    did = f"d{len(task.datasets) + 1}"
    task.datasets[did] = (ds, system)
    shape = list(ds.splits["train"].states.shape)
    return {"ok": True, "dataset": did, "system": system, "shape_train": shape, "fields": list(ds.fields),
            "trajectories": {k: len(v.seeds) for k, v in ds.splits.items()}, "steps": int(steps), "spacing": int(spacing),
            "dt": ds.splits["train"].params[0].get("dt"), "state_std": [float(x) for x in ds.std], "recovered": recovered,
            "summary": f"{did}: {system} train {shape}" + (f"; recovered: {recovered}" if recovered else "")}


def train(task, dataset, kind="cnn", budget_minutes=2.0, unroll=1, lr=1e-3, name=None, batch=8, model=None):
    """Train a surrogate (cnn | fno | latent | nca_hybrid) for at most budget_minutes. NaN -> lr/3, unroll 1, once."""
    if kind not in KINDS:
        return {"ok": False, "error": f"unknown kind {kind!r}; known: {sorted(KINDS)}"}
    ds, system = task.datasets[dataset]
    minutes = max(0.1, min(float(budget_minutes), LIMITS["max_train_minutes"], task.left() / 60 - 0.5))
    name = name or (kind if unroll == 1 else f"{kind}_k{unroll}")
    arch = {**KINDS[kind], **(model or {})}
    if "modes" in arch:
        arch["modes"] = min(arch["modes"], ds.splits["train"].states.shape[-1] // 2)
    if kind == "latent":
        from kolmo.surrogates import latent
        arch["out"] = len(ds.fields)
    recovered, t = [], time.perf_counter()
    for attempt in range(2):
        cls = latent.LatentConfig if kind == "latent" else tr.TrainConfig
        cfg = cls(model=arch, unroll=int(unroll), steps=100000, batch=int(batch), lr=float(lr), eval_every=50,
                  val_horizon=20, patience=6, max_minutes=minutes)
        t = time.perf_counter()
        if kind == "latent":  # ponytail: latent.train has no progress callback yet
            sur, hist = latent.train(ds, cfg, verbose=False)
        else:
            sur, hist = tr.fit(ds, cfg, verbose=False, should_stop=task.should_stop,
                               on_progress=lambda p: task.emit({"type": "progress", "model": name, **p}))
        vals = [h["val_rollout_rel_l2"] for h in hist]
        bad = not all(math.isfinite(h["train_loss"]) for h in hist) or not all(math.isfinite(v) and v < 1e3 for v in vals)
        if not bad or attempt:
            break
        recovered.append(f"non-finite training (lr {lr}, unroll {unroll}) -> retried with lr {lr / 3:.3g}, unroll 1")
        lr, unroll = lr / 3, 1
    mins = (time.perf_counter() - t) / 60
    tr.save(os.path.join(task.dir, "checkpoints", name), sur, ds, cfg, hist, train_minutes=mins)
    task.models[name] = (sur, dataset, kind)
    best = min(hist, key=lambda h: h["val_rollout_rel_l2"] if math.isfinite(h["val_rollout_rel_l2"]) else 1e30)
    return {"ok": not bad, "model": name, "kind": kind, "weights": sum(int(np.size(x)) for x in jax.tree_util.tree_leaves(sur.params)),
            "optimizer_steps": hist[-1]["step"], "minutes": round(mins, 2), "best_val_rollout_rel_l2": best["val_rollout_rel_l2"],
            "best_at_step": best["step"], "stopped": hist[-1].get("stopped"), "recovered": recovered,
            **({"error": "training stayed non-finite after recovery"} if bad else {}),
            "summary": f"{name}: best val rollout relL2 {best['val_rollout_rel_l2']:.4g} at step {best['step']}"}


def _compact(r):
    out = {}
    for split, s in r["splits"].items():
        out[split] = {"rel_l2_at": s["horizons"], "steps_to_failure_median": s["time_to_failure"]["median_observed"],
                      "failed": f"{s['n_traj'] - s['time_to_failure']['censored']}/{s['n_traj']}", "steps": s["steps"],
                      "energy_drift_final": s.get("energy_drift_final"), "nonphysical_steps": s.get("nonphysical_steps"),
                      "residual_ratio_median": s.get("residual_ratio_median")}
    rt = r.get("runtime") or {}
    out["runtime"] = {k: rt.get(k) for k in ("reference_s_per_step", "surrogate_s_per_step", "speedup")}
    return out


def evaluate(task, model, steps=None, threshold=0.5):
    """Long-horizon rollout from held-out test seeds (and held-out parameters if the dataset has them)."""
    sur, did, _ = task.models[model]
    ds, system = task.datasets[did]
    task.results[model] = ev.clean(ev.evaluate(task.system(system), ds, sur, steps=steps, threshold=threshold, verbose=False))
    ev.write_json(task.results, os.path.join(task.dir, "results.json"))
    c = _compact(task.results[model])
    return {"ok": True, "model": model, **c, "summary": f"{model}: test relL2 {c['test']['rel_l2_at']}, "
            f"median steps to failure {c['test']['steps_to_failure_median']}"}


def _pick(task, models):
    missing = [m for m in models or [] if m not in task.results]
    if missing:
        raise KeyError(f"not evaluated: {missing}; evaluated: {sorted(task.results)}")
    return {m: task.results[m] for m in (models or task.results)}


def compare(task, models=None):
    """Side-by-side table of evaluated models plus kolmo.diagnose findings (fact = measured, interpretation = hypothesis)."""
    res = _pick(task, models)
    if not res:
        return {"ok": False, "error": "no evaluated models; call evaluate first"}
    rows = [{k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in row.items()} for row in ev.table(res)]
    f = dg.diagnose(res)
    return {"ok": True, "table": rows, "findings": [{k: x[k] for k in ("level", "model", "fact", "interpretation", "next")}
                                                     for x in f], "summary": f"{len(res)} models, {len(f)} findings"}


def make_figure(task, models=None):
    """Error-vs-horizon and energy plots (PNG) of evaluated models, saved in the task directory."""
    res = _pick(task, models)
    if not res:
        return {"ok": False, "error": "no evaluated models; call evaluate first"}
    paths = ev.plot(res, task.dir)
    task.figures = sorted(set(task.figures) | set(paths))
    return {"ok": True, "figures": paths, "summary": ", ".join(os.path.basename(p) for p in paths)}


def registry_op(task, op="list", model=None, id=None, system=None):
    """Model registry (runs/registry.json): op list | register (a model trained and evaluated in this task) | remove (id)."""
    if op == "list":
        return {"ok": True, "entries": registry.list(system, root=task.root)}
    if op == "register":
        return {"ok": True, "entry": registry.register(task.dir, model, root=task.root)}
    if op == "remove":
        registry.remove(id, root=task.root)
        return {"ok": True, "removed": id}
    return {"ok": False, "error": f"unknown op {op!r}"}


def write_report(task, markdown, title=None):
    """Write report.md in the task directory. Numbers not found in tool results are replaced by [unverified]."""
    from kolmo.assistant.agent import audit, sections
    text, removed = audit(sections(markdown), task.sources)
    task.removed += removed
    figs = "".join(f"\n![{os.path.basename(p)}]({os.path.basename(p)})" for p in task.figures)
    path = os.path.join(task.dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title or 'Assistant report'}\n\n{text}\n" + (f"\n## Figures\n{figs}\n" if figs else "") +
                (f"\n> Audit: {len(removed)} number(s) not found in tool results were replaced by [unverified].\n" if removed else ""))
    task.report = path
    return {"ok": True, "report": path, "unverified_numbers": removed, "summary": f"report written ({len(removed)} unverified)"}


FUNCS = {f.__name__: f for f in (plan, set_zen, list_systems, describe_system, compose_pde, generate_data, train,
                                 evaluate, compare, make_figure, registry_op, write_report)}
_S, _I, _N, _B, _O = {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"}, {"type": "object"}
_L = {"type": "array", "items": _S}
_ARGS = {
    "plan": ({"steps": _L}, ["steps"]),
    "set_zen": ({"on": _B}, ["on"]),
    "list_systems": ({}, []),
    "describe_system": ({"name": _S}, ["name"]),
    "compose_pde": ({"spec": {"description": "PDE spec {name, fields, rhs: {field: expr}, params, n, L, dt, ic}, or a "
                              "library PDE name"}}, ["spec"]),
    "generate_data": ({"system": {**_S, "description": "registered system or composed PDE name"}, "params": _O,
                       "steps": _I, "spacing": _I, "burn_in": _I, "n_train": _I, "n_val": _I, "n_test": _I, "vary": _O,
                       "heldout": _O}, ["system"]),
    "train": ({"dataset": _S, "kind": {"type": "string", "enum": list(KINDS)}, "budget_minutes": _N, "unroll": _I,
               "lr": _N, "name": _S, "batch": _I, "model": _O}, ["dataset", "kind"]),
    "evaluate": ({"model": _S, "steps": _I, "threshold": _N}, ["model"]),
    "compare": ({"models": _L}, []),
    "make_figure": ({"models": _L}, []),
    "registry_op": ({"op": {"type": "string", "enum": ["list", "register", "remove"]}, "model": _S, "id": _S,
                     "system": _S}, ["op"]),
    "write_report": ({"markdown": _S, "title": _S}, ["markdown"]),
}
SCHEMAS = [{"type": "function", "function": {"name": k, "description": (FUNCS[k].__doc__ or k).split("\n")[0],
            "parameters": {"type": "object", "properties": p, "required": r, "additionalProperties": False}}} for k, (p, r) in _ARGS.items()]
