"""Model registry (docs/contracts.md): <root>/registry.json lists trained surrogates; entries point into runs/.

Entry: {"id": "<run dir name>/<model>", "name", "system", "kind", "path" (checkpoint without extension), "created"
(ISO time), "params" (number of weights), "metrics" ({"one_step", "steps_to_failure", "error_at_10",
"surrogate_ms_per_step", "speedup"} from the run's results.json test split, null where not evaluated),
"report" (path to the run's report.md or null)}. remove() only unregisters; checkpoint files stay."""
import datetime
import json
import os

import numpy as np

from kolmo.surrogates import data, train

ROOT = "runs"


def _path(root):
    return os.path.join(root, "registry.json")


def _read(root):
    if not os.path.exists(_path(root)):
        return []
    with open(_path(root)) as f:
        return json.load(f)


def _write(entries, root):
    os.makedirs(root, exist_ok=True)
    with open(_path(root), "w") as f:
        json.dump(entries, f, indent=1)


def list(system=None, root=ROOT):  # noqa: A001 (name fixed by the contract)
    return [e for e in _read(root) if system is None or e["system"] == system]


def register(run_dir, name=None, root=ROOT):
    """Register model `name` of an experiment run directory (or its only model when name is None)."""
    ck = os.path.join(run_dir, "checkpoints")
    names = sorted(f[:-5] for f in os.listdir(ck) if f.endswith(".json"))
    if name is None:
        if len(names) != 1:
            raise ValueError(f"{run_dir} holds models {names}; pass name")
        name = names[0]
    if name not in names:
        raise ValueError(f"no checkpoint {name!r} in {run_dir} (have {names})")
    path = os.path.join(ck, name)
    with open(path + ".json") as f:
        meta = json.load(f)
    with np.load(path + ".npz") as z:
        n_params = int(sum(z[k].size for k in z.files))
    metrics = {"one_step": None, "steps_to_failure": None, "error_at_10": None, "surrogate_ms_per_step": None,
               "speedup": None}
    res = os.path.join(run_dir, "results.json")
    if os.path.exists(res):
        with open(res) as f:
            r = json.load(f).get(name)
        if r and "test" in r["splits"]:
            t = r["splits"]["test"]
            metrics.update(one_step=t["horizons"].get("1"), error_at_10=t["horizons"].get("10"),
                           steps_to_failure=t["time_to_failure"]["median_observed"],
                           surrogate_ms_per_step=1e3 * r["runtime"]["surrogate_s_per_step"],
                           speedup=r["runtime"]["speedup"])
    report = os.path.join(run_dir, "report.md")
    entry = {"id": f"{os.path.basename(os.path.normpath(run_dir))}/{name}", "name": name,
             "system": data.system_name(meta["data"]["system"]), "kind": meta["arch"]["kind"], "path": path,
             "created": datetime.datetime.now().isoformat(timespec="seconds"), "params": n_params,
             "metrics": metrics, "report": report if os.path.exists(report) else None}
    entries = [e for e in _read(root) if e["id"] != entry["id"]] + [entry]
    _write(entries, root)
    return entry


def load(id, root=ROOT):  # noqa: A002
    """(Surrogate, checkpoint metadata) of a registered model."""
    for e in _read(root):
        if e["id"] == id:
            return train.load(e["path"])
    raise KeyError(f"model {id!r} is not registered")


def remove(id, root=ROOT):  # noqa: A002
    entries = _read(root)
    if not any(e["id"] == id for e in entries):
        raise KeyError(f"model {id!r} is not registered")
    _write([e for e in entries if e["id"] != id], root)
