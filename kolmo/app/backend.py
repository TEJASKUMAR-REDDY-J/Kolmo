"""The only place the UI touches the rest of Kolmo. Everything the window needs goes through here, so a change in a
solver, surrogate, hybrid, evaluation, diagnosis or geometry module is absorbed in this file and nowhere else.

Optional components are imported lazily; a missing or broken one raises `Unavailable` with a message the UI shows
verbatim, instead of inventing data.

Step convention: the UI counts SURROGATE steps. One surrogate step = `spacing` solver steps (from the checkpoint;
1 in reference mode), and the reference runs fused over the same `spacing` (kolmo.rollout.coarse), exactly as
kolmo.hybrid and kolmo.evaluate do."""
import importlib
import inspect
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIRS = (ROOT / "runs", ROOT / "checkpoints")  # experiments write runs/<name>-<hash>/checkpoints/<model>.{npz,json}
TRAIN_HELP = "No surrogate checkpoint for this system. Train one with:  python -m kolmo.experiments run <config.json>"
KEEP_STATES_BYTES = 400e6  # states kept for the post-run diagnosis up to this size (both trajectories)
RESIDUAL_MAX_STEPS = 500  # per-step physics residuals in the diagnosis only up to this many steps (they cost ~a step each)


class Unavailable(RuntimeError):
    """A component is not there; str(e) tells the user what is missing and how to get it."""


def _import(name):
    """(module, None) or (None, reason). Import errors inside other modules must not take the UI down."""
    try:
        return importlib.import_module(name), None
    except Exception as e:  # noqa: BLE001 - the reason is shown to the user
        return None, f"{name}: {type(e).__name__}: {e}"


def _need(name):
    m, why = _import(name)
    if m is None:
        raise Unavailable(f"Not available: {why}")
    return m


def failure_threshold():
    """Relative L2 at which a rollout counts as failed: evaluate.time_to_failure's default (0.5)."""
    m, _ = _import("kolmo.evaluate")
    return inspect.signature(m.time_to_failure).parameters["threshold"].default if m else 0.5


def drift_tol():
    """|energy drift| above which kolmo.diagnose reports it (DRIFT_TOL)."""
    m, _ = _import("kolmo.diagnose")
    return getattr(m, "DRIFT_TOL", 0.05)


def device():
    """'Local (CPU)' etc. from JAX's default device."""
    import jax
    return f"Local ({jax.devices()[0].platform.upper()})"


def experiments():
    """Experiment run directories written by kolmo.experiments, newest first:
    [{"name", "dir", "rows" (evaluate.table), "n_findings", "report" (path or None), "runtime"}]."""
    ev = _import("kolmo.evaluate")[0]
    root, out = ROOT / "runs", []
    for res in sorted(root.glob("*/results.json"), key=lambda p: p.stat().st_mtime, reverse=True) if root.is_dir() else []:
        try:
            results = json.loads(res.read_text(encoding="utf-8"))
            diag = res.parent / "diagnosis.json"
            n = len(json.loads(diag.read_text(encoding="utf-8"))) if diag.exists() else 0
        except (OSError, ValueError):
            continue
        rep = res.parent / "report.md"
        out.append({"name": res.parent.name, "dir": str(res.parent), "rows": ev.table(results) if ev else [],
                    "n_findings": n, "report": str(rep) if rep.exists() else None,
                    "runtime": {m: r.get("runtime", {}) for m, r in results.items()}})
    return out


def all_checkpoints():
    """[(system label, checkpoint label, stem)] over every available system."""
    return [(lbl, c, stem) for lbl, s, _ in systems() if s is not None for c, stem in checkpoints(s)]


# ---------------------------------------------------------------- systems and parameters

@dataclass
class Param:
    key: str
    label: str
    unit: str = ""
    lo: float = 0
    hi: float = 1
    decimals: int = 0  # 0 -> integer
    choices: tuple = ()  # non-empty -> drop-down
    tip: str = ""
    auto: bool = False  # lo (spin) / "auto" (drop-down) means None: the system picks the value


# Nondimensional units: L = length unit, T = time unit.
PARAMS = {
    "fluid2d": [
        Param("n", "Grid", "cells", choices=(32, 64, 128, 256), tip="n x n periodic grid"),
        Param("Re", "Reynolds number Re", "", 1, 1e6, 1, tip="Re = U_ref L_ref / ν. Reference scales: free stream and "
              "body size with an obstacle, else forcing, else initial amplitude (docs/physics.md)"),
        Param("A", "Forcing amplitude A", "L/T²", 0, 100, 3, tip="Kolmogorov body force f_x = −A sin(k_f y)"),
        Param("kf", "Forcing wavenumber k_f", "", 1, 64),
        Param("mu", "Linear drag μ", "1/T", 0, 10, 3),
        Param("U_inf", "Free stream U∞", "L/T", 0, 10, 3, tip="uniform flow along +x, needed for flow past obstacles"),
        Param("ic", "Initial condition", "", choices=("random", "taylor_green", "zero"), auto=True,
              tip="auto: zero with an obstacle, random otherwise"),
        Param("ic_amp", "Initial amplitude", "L/T", 0, 10, 3, tip="rms velocity of the initial field"),
        Param("dt", "Time step Δt", "T", 0, 1, 5, auto=True, tip="auto = largest Δt with estimated CFL ≤ cfl_max"),
    ],
    "diffusion": [
        Param("n", "Grid", "cells", choices=(32, 64, 128, 256), tip="n x n periodic grid"),
        Param("L", "Domain side", "L", 0.1, 100, 3),
        Param("kappa", "Diffusivity κ", "L²/T", 1e-5, 10, 5),
        Param("dt", "Time step Δt", "T", 1e-4, 10, 4),
        Param("modes", "Initial modes", "", 1, 50, tip="random low Fourier modes in the initial condition"),
    ],
}
# UI starting point. Fluid 2D: forced Kolmogorov flow as in `python -m kolmo.systems.fluid2d` (nu 1e-3, A 1, kf 4, mu 0.05).
UI_DEFAULTS = {"fluid2d": {"nu": 1e-3, "A": 1.0, "kf": 4, "mu": 0.05}}
HEADER_EXTRAS = {"fluid2d": ("Re", "nu", "dt", "cfl_estimate"), "diffusion": ("kappa", "dt")}
DESCRIPTIONS = {
    "fluid2d": "2D incompressible Navier–Stokes, vorticity form, periodic box; pseudo-spectral, IF-RK3, 2/3 dealiasing.",
    "diffusion": "∂T/∂t = κ∇²T on a periodic square, solved exactly per Fourier mode (error = float32 round-off).",
}


def systems():
    """[(label, instance or None, reason if None)] in display order."""
    out = []
    m, why = _import("kolmo.systems.fluid2d")
    out.append(("Fluid 2D", m.Fluid2D() if m else None, why))
    from kolmo.systems.diffusion import Diffusion
    out.append(("Diffusion", Diffusion(), None))
    return out


def system_by_name(name):
    for _, s, _ in systems():
        if s is not None and s.name == name:
            return s
    raise Unavailable(f"system {name!r} is not available")


def param_specs(system):
    """Known specs, else one generic spec per numeric default (a new system works before it is described here)."""
    if system.name in PARAMS:
        return PARAMS[system.name]
    out = []
    for k, v in system.validate({}).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out.append(Param(k, k, "", 0, max(10 * v, 10)) if isinstance(v, int) else
                   Param(k, k, "", 0, max(10 * abs(v), 1.0), max(2, 3 - int(math.floor(math.log10(abs(v) or 1))))))
    return out


def defaults(system):
    """Widget values: validated UI defaults, with None kept for 'auto' parameters."""
    raw = {**getattr(system, "DEFAULTS", {}), **UI_DEFAULTS.get(system.name, {})}
    d = system.validate(raw)
    for sp in param_specs(system):
        if sp.auto and raw.get(sp.key) is None:
            d[sp.key] = None
    return d


def describe(system):
    return DESCRIPTIONS.get(system.name, (system.__doc__ or "").strip().split("\n")[0])


def header_keys(system):
    return HEADER_EXTRAS.get(system.name, ())


def supports_obstacle(system):
    return "mask" in getattr(system, "DEFAULTS", {})


# ---------------------------------------------------------------- obstacles

def builtin_obstacle(shape, n, size=0.2, position=(0.25, 0.5)):
    """(n, n) bool [row = y, col = x]; flow along +x. size = diameter / side as a fraction of the domain side."""
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    dx, dy = x - position[0], y - position[1]
    if shape == "circle":
        return dx * dx + dy * dy <= (size / 2) ** 2
    if shape == "square":
        return (np.abs(dx) <= size / 2) & (np.abs(dy) <= size / 2)
    raise ValueError(f"unknown obstacle shape {shape!r}")


def photo_obstacle(path, n, chord=0.25):
    """Photo -> (grid (n, n) bool [row = y upwards], meta). SAM 3 when installed, else GrabCut (kolmo.geometry).
    Slow (SAM 3 runs in a subprocess): call from a worker thread."""
    seg, msk = _need("kolmo.geometry.segment"), _need("kolmo.geometry.mask")
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    segmenter = seg.get_segmenter("auto")
    t0 = time.perf_counter()
    grid, meta = msk.mask_to_grid(segmenter.segment(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), n=n, chord=chord)
    meta.update(segmenter=segmenter.name, segment_seconds=time.perf_counter() - t0)
    return np.flipud(grid), meta  # geometry: row 0 = photo top; solver/display: row = y upwards -> object upright


def obstacle_mask(spec, n):
    """Config obstacle spec -> (n, n) bool or None."""
    kind = (spec or {}).get("kind", "none")
    if kind == "none":
        return None
    if kind == "photo":
        if not spec.get("path"):
            raise ValueError("photo obstacle without a photo path")
        return photo_obstacle(spec["path"], n, spec.get("size", 0.25))[0]
    return builtin_obstacle(kind, n, spec.get("size", 0.2))


# ---------------------------------------------------------------- surrogates and hybrid policies

def checkpoints(system):
    """[(label, stem)] of surrogate checkpoints (<stem>.json + <stem>.npz) trained on this system, newest first."""
    found = []
    for d in CHECKPOINT_DIRS:
        for meta_path in d.rglob("*.json") if d.is_dir() else ():
            if not meta_path.with_suffix(".npz").exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if str(meta.get("version", "")).startswith("kolmo-surrogates") and meta.get("data", {}).get("system") == system.name:
                stem = meta_path.with_suffix("")
                label = f"{stem.parent.parent.name}/{stem.name}" if stem.parent.name == "checkpoints" else stem.name
                found.append((meta_path.stat().st_mtime, label, str(stem)))
    return [(label, stem) for _, label, stem in sorted(found, reverse=True)]


def checkpoint_meta(stem):
    return json.loads(Path(str(stem) + ".json").read_text(encoding="utf-8"))


def describe_checkpoint(stem):
    """One honest line about what the checkpoint was trained on."""
    m = checkpoint_meta(stem)
    d, arch = m.get("data", {}), dict(m.get("arch", {}))
    vary = ", ".join(f"{k} ∈ {v}" for k, v in d.get("vary", {}).items())
    base = ", ".join(f"{k} {v}" for k, v in d.get("params", {}).items())
    hist = m.get("history") or [{}]
    val = hist[-1].get("val_rollout_rel_l2")
    return (f"{arch.pop('kind', '?').upper()} · 1 step = {m.get('spacing', 1)} solver steps\n"
            f"trained on {base or 'system defaults'}{'; ' + vary if vary else ''}"
            + (f"\nlast validation rollout rel. L2 {val:.3g}" if isinstance(val, (int, float)) else ""))


def training_params(stem):
    """Parameters inside the checkpoint's training distribution: base params + the first value of each varied one."""
    d = checkpoint_meta(stem).get("data", {})
    return {**d.get("params", {}), **{k: v[0] for k, v in d.get("vary", {}).items() if v}}


def load_surrogate(stem):
    if not stem:
        raise Unavailable(TRAIN_HELP)
    train = _need("kolmo.surrogates.train")
    try:
        return train.load(str(stem))
    except FileNotFoundError as e:
        raise Unavailable(f"Checkpoint files missing: {e}") from None


def make_policy(spec):
    """{"every", "residual_tol", "energy_tol"} (0 / None = off) -> kolmo.hybrid.HybridPolicy("hybrid", ...)."""
    hy = _need("kolmo.hybrid")
    spec = spec or {}
    pos = lambda v: float(v) if v else None
    return hy.HybridPolicy("hybrid", int(spec.get("every") or 0), pos(spec.get("residual_tol")),
                           pos(spec.get("energy_tol"))).validate()


# ---------------------------------------------------------------- live run

@dataclass
class Timing:
    first_call: float = 0.0  # seconds: compilation + one step
    steady: float = 0.0  # seconds over `count` steady-state steps
    count: int = 0

    @property
    def per_step(self):
        return self.steady / self.count if self.count else None


@dataclass
class Frame:
    step: int  # surrogate step
    t: float  # physical time = step * spacing * dt
    reference: np.ndarray  # (C, H, W)
    candidate: np.ndarray | None  # surrogate or hybrid state; None in reference mode
    source: str  # who produced `candidate` at this step: "reference" | "surrogate" | "" (step 0, reference mode)
    rel_error: float | None  # ||candidate - reference|| / ||reference|| (evaluate.rel_l2)
    diag_reference: dict
    diag_candidate: dict
    events: list  # hybrid interventions / failures at this step


class LiveRun:
    """Yields one Frame per surrogate step so the UI can draw while it runs. Reference and surrogate are stepped live;
    hybrid mode first runs kolmo.hybrid.run (the real policy, not a re-implementation) and then replays its states next
    to the live reference. Only stepper calls are timed (diagnostics and drawing excluded)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.system = system_by_name(cfg["system"])
        params = {k: v for k, v in (cfg.get("params") or {}).items() if v is not None}
        if supports_obstacle(self.system):
            n = int(self.system.validate(params)["n"])
            mask = obstacle_mask(cfg.get("obstacle"), n)
            if mask is not None:
                params["mask"] = mask
        self.params = self.system.validate(params)
        self.obstacle = self.params.get("mask")
        self.mode = cfg.get("mode", "reference")
        self.steps = int(cfg.get("steps", 200))
        self.seed = int(cfg.get("seed", 0))
        self.threshold = failure_threshold()
        self.timing = {"reference": Timing(), "candidate": Timing()}
        self.surrogate = self.meta = self.policy = self.hybrid_traj = None
        self.spacing = 1
        if self.mode in ("surrogate", "hybrid"):
            self.surrogate, self.meta = load_surrogate(cfg.get("checkpoint"))
            self.spacing = int(self.surrogate.spacing)
            if list(self.meta.get("fields", self.system.fields)) != list(self.system.fields):
                raise ValueError(f"checkpoint fields {self.meta.get('fields')} do not match {self.system.fields}")
            # first call = compilation, timed as such; it also catches checkpoints the current model code cannot apply
            probe = self.surrogate.stepper(self.params)
            t0 = time.perf_counter()
            try:
                probe(self.system.initial(self.params, self.seed)).block_until_ready()
            except Exception as e:  # noqa: BLE001
                raise Unavailable(f"Checkpoint {Path(str(cfg.get('checkpoint'))).name} does not run with the current "
                                  f"kolmo.surrogates code ({type(e).__name__}: {str(e)[:160]}). Retrain it with "
                                  "python -m kolmo.experiments run <config.json>.") from None
            self.timing["candidate"].first_call = time.perf_counter() - t0
        if self.mode == "hybrid":
            self.policy = make_policy(cfg.get("policy"))
        elif self.mode not in ("reference", "surrogate"):
            raise ValueError(f"unknown mode {self.mode!r}")
        self._ref_step = _need("kolmo.rollout").coarse(self.system.stepper(self.params), self.spacing)
        self._net = probe if self.mode == "surrogate" else None  # already compiled
        n_vals = int(np.prod(self.system.initial(self.params, self.seed).shape))
        self.keep = 8 * (self.steps + 1) * n_vals <= KEEP_STATES_BYTES  # float32, two trajectories
        self.ref_states, self.cand_states, self.events = [], [], []
        self.phase = ""

    @staticmethod
    def _timed(fn, x, timing):
        t0 = time.perf_counter()
        y = fn(x)
        y.block_until_ready()
        dt = time.perf_counter() - t0
        if timing.first_call == 0.0:
            timing.first_call = dt
        else:
            timing.steady += dt
            timing.count += 1
        return y

    def _frame(self, k, ref, cand, source, events=()):
        r = np.asarray(ref)
        c = None if cand is None else np.asarray(cand)
        err = None
        if c is not None:
            with np.errstate(all="ignore"):
                err = float(np.sqrt(((c.astype(np.float64) - r) ** 2).sum() / max((r.astype(np.float64) ** 2).sum(), 1e-300)))
            err = err if math.isfinite(err) else math.inf
        if self.keep:
            self.ref_states.append(r)
            if c is not None:
                self.cand_states.append(c)
        diag = lambda s: self.system.diagnostics(s, self.params)
        dc = {}
        if c is not None:
            try:
                dc = diag(c) if np.isfinite(c).all() else {}
            except (ValueError, FloatingPointError):
                dc = {}
        self.events += list(events)
        return Frame(k, k * self.spacing * float(self.params["dt"]), r, c, source, err, diag(r), dc, list(events))

    def __iter__(self):
        s0 = self.system.initial(self.params, self.seed)
        replay = None
        if self.mode == "hybrid":
            self.phase = "hybrid rollout"
            tr = _need("kolmo.hybrid").run(self.policy, self.system, self.params, self.surrogate, s0, self.steps)
            self.hybrid_traj, self.phase = tr, ""
            self.timing["candidate"] = Timing(tr.compile_seconds, tr.seconds, self.steps)
            by_step = {}
            for e in tr.events:
                by_step.setdefault(int(e["step"]), []).append(e)
            replay = (tr.states, tr.source, by_step)
        yield self._frame(0, s0, s0 if self.mode != "reference" else None, "", replay[2].get(0, []) if replay else ())
        ref, cand = s0, s0
        for k in range(1, self.steps + 1):
            ref = self._timed(self._ref_step, ref, self.timing["reference"])
            if self._net is not None:
                cand = self._timed(self._net, cand, self.timing["candidate"])
                yield self._frame(k, ref, cand, "surrogate")
            elif replay is not None:
                yield self._frame(k, ref, replay[0][k], replay[1][k - 1], replay[2].get(k, []))
            else:
                yield self._frame(k, ref, None, "")


# ---------------------------------------------------------------- post-run summary and diagnosis

def summarize(run, stopped=False):
    """Measured facts and rule-based findings for a finished run.
    Returns {"facts": [str], "findings": [diagnose finding + "source"], "notes": [str]} (notes = why something is absent)."""
    facts, findings, notes = [], [], []
    ev = _need("kolmo.evaluate")
    ro = _need("kolmo.rollout")
    T = len(run.ref_states) - 1
    if run.keep and T >= 1:
        ref = np.stack(run.ref_states)
        bad = ro.detect_failures(ref)
        facts += [f"Reference solver: {e['reason']} at step {e['step']} ({e['detail']})" for e in bad]
        if "cfl" in (run.system.diagnostics(ref[-1], run.params) or {}):
            cfl = max(run.system.diagnostics(ref[t], run.params)["cfl"] for t in range(0, T + 1, max(1, T // 50)))
            facts.append(f"Reference max CFL (sampled) {cfl:.3g}")
    if run.mode != "reference" and run.keep and len(run.cand_states) == T + 1 and T >= 1:
        pred = np.stack(run.cand_states)
        err = ev.rel_l2(pred, ref)
        ttf = ev.time_to_failure(err, run.threshold)
        facts.append(f"Relative L2 first exceeds {run.threshold:g} at step {ttf}" if ttf else
                     f"Relative L2 stays ≤ {run.threshold:g} for all {T} steps (max {np.max(err):.3g})")
        facts += [f"{run.mode.capitalize()}: {e['reason']} at step {e['step']} ({e['detail']})"
                  for e in ro.detect_failures(pred, ref_scale=float(np.abs(ref).max()))]
        if run.mode == "hybrid":
            c = _need("kolmo.hybrid").cost(run.hybrid_traj, run.spacing)
            reasons = {}
            for e in run.hybrid_traj.events:
                if e["kind"] == "intervention":
                    reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
            facts.append(f"{c['interventions']} interventions ({', '.join(f'{k} {v}' for k, v in reasons.items()) or 'none'});"
                         f" solver steps {c['solver_steps']} = {100 * c['fraction_of_reference']:.1f}% of a reference run")
        if run.mode == "surrogate" and not stopped:
            findings += _diagnose_single(run, pred, ref, ev, ro)
        elif run.mode == "hybrid":
            notes.append("Rule-based diagnosis of a live run covers surrogate mode; hybrid policies are diagnosed over "
                         "a test set by kolmo.experiments (shown below when the checkpoint has one).")
    elif run.mode != "reference" and not run.keep:
        notes.append("Run too long to keep its states, so no post-run diagnosis.")
    if stopped:
        notes.append("Run stopped early: facts cover the steps computed.")
    if run.mode != "reference":
        findings += experiment_findings(run.cfg.get("checkpoint"))
    return {"facts": facts, "findings": findings, "notes": notes}


def _diagnose_single(run, pred, ref, ev, ro):
    """kolmo.diagnose rules on a one-trajectory evaluation of this run (same metrics as kolmo.evaluate)."""
    dg = _need("kolmo.diagnose")
    T = len(ref) - 1
    m = ev.trajectory_metrics(run.system, pred, ref, run.params, run.spacing, run.threshold,
                              with_residual=T <= RESIDUAL_MAX_STEPS)
    fs = ro.rollout(ro.coarse(run.system.stepper(run.params), run.spacing), pred[1], T - 1).states
    m["first_step_only"] = np.concatenate([[0.0], ev.rel_l2(fs, ref[1:])])
    name = Path(str(run.cfg.get("checkpoint"))).name
    res = {name: {"splits": {"test": ev.summarize([m], T, run.threshold)}, "hybrid": []}}
    res = json.loads(json.dumps(ev.clean(res)))
    out = dg.diagnose(res)
    for f in out:
        f["source"] = f"this run (1 trajectory, seed {run.seed}{'' if T <= RESIDUAL_MAX_STEPS else ', no residuals'})"
    return out


def experiment_findings(stem):
    """Findings that kolmo.experiments wrote for this checkpoint's model (diagnosis.json next to checkpoints/)."""
    if not stem:
        return []
    stem = Path(str(stem))
    path = stem.parent.parent / "diagnosis.json"
    if stem.parent.name != "checkpoints" or not path.exists():
        return []
    try:
        fs = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = [dict(f) for f in fs if str(f.get("model", "")).split(" vs ")[0] == stem.name or
           str(f.get("model", "")).endswith(f" vs {stem.name}")]
    for f in out:
        f["source"] = f"experiment {stem.parent.parent.name} (test-set evaluation)"
    return out


# ---------------------------------------------------------------- configs

def save_config(cfg, path):
    Path(path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if "system" not in cfg:
        raise ValueError("not a Kolmo config: no 'system' key")
    return cfg
