"""Reproducible experiments: `python -m kolmo.experiments run <config.json>`.

Config JSON = ExperimentConfig fields: {"name", "data": DataConfig fields, "models": {name: TrainConfig fields},
"hybrid": [HybridPolicy fields], "eval_steps", "threshold", "out_dir"}. The first model is the comparison baseline.
Run directory <out_dir>/<name>-<config hash>/ gets config.json, env.json, checkpoints/<model>.{npz,json},
results.json, diagnosis.json, plots (*.png) and report.md. A checkpoint whose stored train and data config match is
reused instead of retrained (delete checkpoints/ to force retraining)."""
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from kolmo import diagnose as dg
from kolmo import evaluate as ev
from kolmo.hybrid import HybridPolicy
from kolmo.surrogates import data, train


@dataclass
class ExperimentConfig:
    name: str
    data: dict
    models: dict
    hybrid: list = field(default_factory=list)
    eval_steps: int | None = None  # surrogate steps evaluated (default: whole test trajectories)
    threshold: float = 0.5  # time-to-failure relative L2 threshold
    out_dir: str = "runs"

    def parsed(self):
        """Validated (DataConfig, {name: TrainConfig}, [HybridPolicy]); raises ValueError on any bad field."""
        if not self.name or not self.models:
            raise ValueError("name and at least one model are required")
        try:
            d = data.DataConfig(**self.data).validate()
            m = {k: train.TrainConfig(**v).validate() for k, v in self.models.items()}
            h = [HybridPolicy(**p).validate() for p in self.hybrid]
        except TypeError as e:
            raise ValueError(f"bad config field: {e}") from None
        if not 0 < self.threshold:
            raise ValueError("threshold must be positive")
        return d, m, h

    def hash(self):
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:8]


def environment():
    import flax, jax, matplotlib, optax
    return {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "jax": jax.__version__, "jax_devices": [str(d) for d in jax.devices()],
            "flax": flax.__version__, "optax": optax.__version__, "numpy": np.__version__,
            "matplotlib": matplotlib.__version__, "kolmo": train.VERSION}


def run(cfg: ExperimentConfig, verbose=True):
    dcfg, mcfgs, policies = cfg.parsed()
    out = os.path.join(cfg.out_dir, f"{cfg.name}-{cfg.hash()}")
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    ev.write_json(asdict(cfg), f"{out}/config.json")
    ev.write_json(environment(), f"{out}/env.json")
    t0 = time.perf_counter()
    ds = data.build(dcfg, os.path.join(cfg.out_dir, "cache"), verbose)
    system = data.get_system(dcfg.system)
    results = {}
    for name, tc in mcfgs.items():
        ck = f"{out}/checkpoints/{name}"
        meta = None
        if os.path.exists(ck + ".json"):
            with open(ck + ".json") as f:
                meta = json.load(f)
        if meta and meta["train"] == asdict(tc) and meta["data_hash"] == ds.hash:
            sur, meta = train.load(ck)
            if verbose:
                print(f"model {name}: reusing {ck}")
        else:
            if verbose:
                print(f"model {name}: training {tc.model} unroll={tc.unroll} pushforward={tc.pushforward}", flush=True)
            t = time.perf_counter()
            sur, hist = train.train(ds, tc, verbose)
            train.save(ck, sur, ds, tc, hist, train_minutes=(time.perf_counter() - t) / 60)
            meta = train.load(ck)[1]
        if verbose:
            print(f"model {name}: evaluating", flush=True)
        results[name] = ev.evaluate(system, ds, sur, steps=cfg.eval_steps, threshold=cfg.threshold, policies=policies,
                                    verbose=verbose)
        results[name]["training"] = {"history": meta["history"], "train_minutes": meta.get("train_minutes")}
    ev.write_json(results, f"{out}/results.json")
    clean = json.loads(json.dumps(ev.clean(results)))
    findings = dg.diagnose(clean)
    ev.write_json(findings, f"{out}/diagnosis.json")
    plots = ev.plot(clean, out)
    with open(f"{out}/report.md", "w") as f:
        f.write(report(cfg, clean, findings, plots, time.perf_counter() - t0))
    if verbose:
        print(f"wrote {out}")
    return out, clean, findings


def report(cfg, results, findings, plots, seconds):
    rows = ev.table(results)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else ("n/a" if v is None else str(v))
    lines = [f"# {cfg.name}", "", f"Reproduce: `python -m kolmo.experiments run <run dir>/config.json` "
             f"(config hash {cfg.hash()}); total wall time {seconds / 60:.1f} min.", "",
             "## Generalisation table (test = held-out seeds, heldout = held-out parameter values)", "",
             "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r.get(k)) for k in keys) + " |" for r in rows]
    lines += ["", "## Training", ""]
    for m, r in results.items():
        h = r["training"]["history"]
        best = min(h, key=lambda x: x["val_rollout_rel_l2"] if x["val_rollout_rel_l2"] is not None else float("inf"))
        lines.append(f"- {m}: {len(h)} evaluations, last step {h[-1]['step']}, best val rollout relL2 "
                     f"{fmt(best['val_rollout_rel_l2'])} at step {best['step']}, "
                     f"{fmt(r['training']['train_minutes'])} min" + (f", stopped: {h[-1]['stopped']}" if "stopped" in h[-1] else ""))
    lines += ["", "## Runtime", ""]
    for m, r in results.items():
        rt = r["runtime"]
        lines.append(f"- {m}: reference {fmt(rt['reference_s_per_step'])} s/step, surrogate "
                     f"{fmt(rt['surrogate_s_per_step'])} s/step, speedup {fmt(rt['speedup'])}x "
                     f"(surrogate compile {fmt(rt['surrogate_compile_s'])} s); {rt['conditions']}")
    if any(r["hybrid"] for r in results.values()):
        lines += ["", "## Hybrid execution (test split)", "",
                  "| model | policy | solver cost % | max median relL2 | interventions |", "|---|---|---|---|---|"]
        for m, r in results.items():
            lines += [f"| {m} | {h['policy']} | {100 * h['cost_fraction']:.1f} | {fmt(h['max_rel_l2_median'])} | "
                      f"{h['interventions_by_reason']} |" for h in r["hybrid"]]
            lines += [f"| {m} | {h['policy']} | unavailable: {h['reason'][:100]} | | |" for h in r.get("hybrid_unavailable", [])]
    lines += ["", "## Diagnosis (rule-based; Measured = numbers, Interpretation = hypothesis)", "", dg.to_markdown(findings),
              "", "## Plots", ""] + [f"![{os.path.basename(p)}]({os.path.basename(p)})" for p in plots]
    return "\n".join(lines) + "\n"


def load_config(path) -> ExperimentConfig:
    with open(path) as f:
        d = json.load(f)
    try:
        return ExperimentConfig(**d)
    except TypeError as e:
        raise ValueError(f"bad experiment config {path}: {e}") from None


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "run":
        sys.exit("usage: python -m kolmo.experiments run <config.json>")
    run(load_config(sys.argv[2]))
