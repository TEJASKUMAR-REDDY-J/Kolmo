# Interfaces between Kolmo modules (build against these)

## Systems — `kolmo/systems/__init__.py`

- `SYSTEMS: dict[str, type]` maps a name to a System class.
- `make(name_or_spec) -> System` takes a registered name, or a PDE spec dict (see below).
- Every System adds `describe() -> dict` with keys `name`, `fields`, `equations` (plain text), `params`, `checks`.
  - `params` maps a parameter name to `{default, min, max, unit, help}`.
  - `checks` lists invariants it guarantees, e.g. "mass conserved".
- Systems:
  - `fluid2d` (exists; option `beta` for a beta-plane gets added);
  - `diffusion` (exists);
  - `boussinesq` (vorticity + temperature, two-way coupled);
  - `pde` (spec-driven).

## PDE spec — `kolmo/systems/pde.py`

```json
{"name": "cahn_hilliard", "fields": ["u"],
 "rhs": {"u": "lap(u**3 - u) - gamma*lap(lap(u))"},
 "params": {"gamma": 0.01}, "n": 64, "L": 6.283, "dt": 0.01, "ic": "noise:0.1"}
```

- **Grammar:** fields, params, numbers, `+ - * / **`, `lap`, `dx`, `dy`, `grad2` (= |∇u|²), `exp`, `tanh`, `sin`, `cos`.
- **Parsing:** done with Python's `ast` against a whitelist. Never `eval`.
- **Numerics:** the solver splits off the linear constant-coefficient part (it treats it exactly in Fourier space) from
  the nonlinear remainder.
- **Validation:** `validate` reports a clear error for unknown symbols, stiffness or instability.
- **Library:** `cahn_hilliard`, `allen_cahn`, `gray_scott` (2 fields), `kuramoto_sivashinsky`, `swift_hohenberg`, `burgers`.

## Model registry — `kolmo/surrogates/registry.py`

- `list(system=None) -> [entry]`. An entry has `id`, `name`, `system`, `kind` (`cnn` | `fno` | `latent` | `nca_hybrid`),
  `path`, `created`, `params`, `metrics` (`one_step`, `steps_to_failure`, …) and `report` (path to md).
- `register(run_dir, name=None) -> entry`
- `load(id) -> (surrogate, meta)`
- `remove(id)`
- The registry lives in `runs/registry.json`; entries point into `runs/`.

## Training progress

- `kolmo.surrogates.train.fit(..., on_progress=None)` calls `on_progress` with a dict:
  `{"phase": "data"|"train"|"eval", "step", "total", "loss", "val_rollout_error", "eta_s"}`.
- It calls it at most 4 times per second.
- It checks the cancel flag `should_stop()`, if given.

## Assistant events (for the UI) — `kolmo/assistant/agent.py`

- `run(question, on_event=cb)` emits dicts with these `type` values:
  - `plan`: `steps: [str]`
  - `status`: `text`
  - `tool`: `name`, `args`, `ok`, `summary`
  - `progress`: same keys as training progress
  - `result`: `metrics`, `figures: [path]`, `report: path`
  - `zen`: `on: bool`
  - `error`: `text`
  - `done`
- The UI honours `zen` events. If the user turned zen off during this task, the UI ignores later `zen on` events until
  the user re-enables zen.
