"""Surrogate datasets from any kolmo System.

One trajectory = validated params + seed -> initial(params, seed), `burn_in` solver steps discarded, then `steps`
surrogate steps recorded, each = `spacing` solver steps of size params["dt"] (states in physical units, float32).
Splits: train/val/test use disjoint seed lists and every combination of `vary` values; split "heldout" uses the test
seeds with every combination of `heldout` values (parameter values never seen in training).
Normalisation: per-channel mean/std over all train states; conditioning params mean/std over train trajectories
(std 0 -> 1). Cached as runs/cache/data-<hash>.npz, hash = sha1 of the system name + config JSON."""
import hashlib
import importlib
import itertools
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from kolmo.rollout import coarse

SYSTEMS = {"diffusion": "kolmo.systems.diffusion:Diffusion", "fluid2d": "kolmo.systems.fluid2d:Fluid2D"}


def get_system(name):
    mod, cls = SYSTEMS[name].split(":")
    return getattr(importlib.import_module(mod), cls)()


@dataclass
class DataConfig:
    system: str = "diffusion"
    params: dict = field(default_factory=dict)  # base parameters (system defaults complete the rest)
    steps: int = 50  # recorded surrogate steps per trajectory (states 0..steps)
    spacing: int = 1  # solver steps per surrogate step
    burn_in: int = 0  # solver steps discarded after the initial condition
    train_seeds: list = field(default_factory=lambda: list(range(0, 8)))
    val_seeds: list = field(default_factory=lambda: [100, 101])
    test_seeds: list = field(default_factory=lambda: [200, 201, 202])
    vary: dict = field(default_factory=dict)  # {param: [values]} used in train/val/test
    heldout: dict = field(default_factory=dict)  # {param: [values]} only in split "heldout"
    cond: list = field(default_factory=list)  # param names fed to the model as constant channels

    def validate(self):
        if self.system not in SYSTEMS:
            raise ValueError(f"unknown system {self.system!r}; known: {sorted(SYSTEMS)}")
        if self.steps < 1 or self.spacing < 1 or self.burn_in < 0:
            raise ValueError("steps and spacing must be >= 1, burn_in >= 0")
        seeds = [set(self.train_seeds), set(self.val_seeds), set(self.test_seeds)]
        if not all(seeds) or any(a & b for a, b in itertools.combinations(seeds, 2)):
            raise ValueError("train/val/test seeds must be non-empty and disjoint")
        if self.heldout and set(self.heldout) - set(self.vary) - set(self.params):
            raise ValueError("heldout params must also appear in params or vary")
        for name, vals in self.heldout.items():
            if set(vals) & set(self.vary.get(name, [self.params.get(name)])):
                raise ValueError(f"heldout values of {name} overlap training values")
        if set(self.cond) - set(self.vary) - set(self.heldout):
            raise ValueError("cond params must be ones that vary (else they carry no information)")
        return self

    def hash(self):
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class Split:
    states: np.ndarray  # (N, steps + 1, C, H, W) float32, physical units
    cond: np.ndarray  # (N, P) float32 raw values of cfg.cond params
    params: list  # validated params per trajectory
    seeds: list


@dataclass
class Dataset:
    config: DataConfig
    splits: dict  # name -> Split
    mean: np.ndarray  # (C,) per-channel mean of train states
    std: np.ndarray  # (C,)
    cond_mean: np.ndarray  # (P,)
    cond_std: np.ndarray  # (P,)
    fields: tuple = ()

    @property
    def hash(self):
        return self.config.hash()


def _combos(base, vary):
    keys = list(vary)
    return [{**base, **dict(zip(keys, vals))} for vals in itertools.product(*vary.values())] if keys else [dict(base)]


def trajectory(system, params, seed, steps, spacing, burn_in=0):
    """(steps + 1, C, H, W) float32 reference trajectory at surrogate spacing."""
    p = system.validate(params)
    step = system.stepper(p)
    s = system.initial(p, seed)
    if burn_in:
        s = coarse(step, burn_in)(s)
    big = coarse(step, spacing)
    out = [s]
    for _ in range(steps):
        s = big(s)
        out.append(s)
    return np.asarray(np.stack(out), np.float32)


def _make_split(system, cfg, seeds, combos):
    runs = [(system.validate(p), s) for p in combos for s in seeds]
    states = np.stack([trajectory(system, p, s, cfg.steps, cfg.spacing, cfg.burn_in) for p, s in runs])
    cond = np.array([[float(p[c]) for c in cfg.cond] for p, _ in runs], np.float32).reshape(len(runs), len(cfg.cond))
    return Split(states, cond, [p for p, _ in runs], [s for _, s in runs])


def build(cfg: DataConfig, cache_dir="runs/cache", verbose=True) -> Dataset:
    """Generate (or load from cache) the dataset described by cfg."""
    cfg.validate()
    system = get_system(cfg.system)
    path = os.path.join(cache_dir, f"data-{cfg.system}-{cfg.hash()}.npz")
    if os.path.exists(path):
        z = np.load(path)
        splits = {name: Split(z[f"{name}/states"], z[f"{name}/cond"], json.loads(str(z[f"{name}/params"])),
                              [int(v) for v in z[f"{name}/seeds"]]) for name in json.loads(str(z["names"]))}
        if verbose:
            print(f"data: loaded {path}")
    else:
        base = {**cfg.params}
        todo = {"train": (cfg.train_seeds, _combos(base, cfg.vary)), "val": (cfg.val_seeds, _combos(base, cfg.vary)),
                "test": (cfg.test_seeds, _combos(base, cfg.vary))}
        if cfg.heldout:
            todo["heldout"] = (cfg.test_seeds, _combos(base, {**cfg.vary, **cfg.heldout}))
        splits = {name: _make_split(system, cfg, seeds, combos) for name, (seeds, combos) in todo.items()}
        os.makedirs(cache_dir, exist_ok=True)
        arrays = {"names": np.array(json.dumps(list(splits)))}
        for name, sp in splits.items():
            arrays.update({f"{name}/states": sp.states, f"{name}/cond": sp.cond,
                           f"{name}/params": np.array(json.dumps(sp.params)), f"{name}/seeds": np.array(sp.seeds)})
        np.savez(path, **arrays)
        if verbose:
            print(f"data: generated {path} " + ", ".join(f"{k}={v.states.shape}" for k, v in splits.items()))
    tr = splits["train"]
    mean = tr.states.mean(axis=(0, 1, 3, 4), dtype=np.float64).astype(np.float32)
    std = tr.states.std(axis=(0, 1, 3, 4), dtype=np.float64).astype(np.float32)
    cmean = tr.cond.mean(0).astype(np.float32)
    cstd = tr.cond.std(0).astype(np.float32)
    cstd[cstd == 0] = 1.0
    return Dataset(cfg, splits, mean, np.maximum(std, 1e-12), cmean, cstd, tuple(system.fields))
