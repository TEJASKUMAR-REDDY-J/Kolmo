# Kolmo desktop UI

    python -m kolmo

One window, dark theme, no tabs. PySide6 6.7 (QtCore/QtGui/QtWidgets only), NumPy, no other dependencies. All
computation runs in worker threads; the window never waits on the solver.

## Layout

**Left column — what to run**

| Section | What it does |
|---|---|
| System | Fluid 2D (`kolmo.systems.fluid2d.Fluid2D`) or Diffusion. A system whose module fails to import is listed greyed out with the reason as a tooltip. |
| Parameters | The system's parameters with units (L = length unit, T = time unit, nondimensional). The system's own `validate()` decides what is allowed (CFL, 2/3-rule band, Re needs a reference velocity, ...) and its message is shown under Run. `auto` = the system chooses (Fluid 2D: Δt from the CFL limit; initial condition zero with an obstacle, random otherwise). Fluid 2D starts from forced Kolmogorov flow (ν 1e-3, A 1, k_f 4, μ 0.05: the solver's benchmark case). |
| Initial condition | Random seed passed to `system.initial`. |
| Obstacle | None, circle, square (centred at x = 0.25 L, y = 0.5 L; size = diameter or side as a fraction of the domain) or **From photo…**: `kolmo.geometry` segments the photo (SAM 3 when installed, else GrabCut) and maps it onto the grid with the object's streamwise length = size × L. Shows the mask thumbnail, cell count and segmenter. Only systems with a `mask` parameter accept obstacles; flow past a body needs U∞ > 0. |
| Execution | Reference / Surrogate / Hybrid. Surrogate and Hybrid need a checkpoint: the picker lists every `kolmo.surrogates.train` checkpoint (`<name>.json` + `<name>.npz`) under `runs/` and `checkpoints/` trained on the selected system, with its architecture, spacing and training distribution. Parameters outside the training distribution are flagged (⚠) with a one-click *Use the training parameters*. Hybrid shows the `kolmo.hybrid.HybridPolicy` fields: re-anchor period, residual tolerance, energy tolerance (`off` = disabled; non-finite proposals always trigger the solver). |
| Run | Steps (surrogate steps; 1 surrogate step = `spacing` solver steps), Run / Pause / Reset. Inputs are locked while a run is live, so the screen always matches the configuration that produced it; changing the setup afterwards clears the screen. |
| Config | Save / open the whole setup as JSON (system, parameters, seed, obstacle, mode, checkpoint, policy, steps). A photo obstacle stores the photo path and is re-segmented on open. |

**Header**: system and mode, the validated parameters actually used (including derived ones: Re, ν, Δt, the a-priori
CFL estimate), seed, spacing, obstacle size; step counter, physical time t, and run status.

**Centre: three field panels**, each titled with exactly what it shows:

1. **Reference solver**: the system's own stepper.
2. **Surrogate prediction** / **Hybrid prediction**: the network rollout from the same initial state, or the hybrid
   trajectory. The subtitle says who produced the current step (network, or solver intervention).
3. **Absolute error |prediction − reference|**, per cell.

Panels 1 and 2 share one colour scale, set from the reference only and only ever widened during a run (a decaying field
visibly fades; a blown-up prediction saturates instead of rescaling the reference away). Signed fields use a diverging
map with a dark zero (after Crameri's *berlin*), magnitudes a sequential map (after *inferno*); the error panel's scale
is 0 to the current frame's maximum. Every panel has a colour bar with its numeric range. Images are drawn y-up
(x →, y ↑), first field channel. Obstacles are solid grey. In reference mode panels 2 and 3 show an empty state
instead of anything invented; without a checkpoint they say how to train one.

**Bottom row**

- **Relative L2 error vs step** (log scale): ‖prediction − reference‖ / ‖reference‖ over all cells (the definition of
  `kolmo.evaluate.rel_l2`), the failure threshold line (`kolmo.evaluate.time_to_failure`'s default, 0.5), purple
  dotted lines at hybrid interventions, red dotted lines at detected failures.
- **Diagnostic vs step**: any quantity from `system.diagnostics` (energy, enstrophy, CFL, drag/lift with an obstacle,
  variance, ...) for reference (solid) and prediction (dashed); optional log scale. Long runs are drawn with per-pixel
  min/max decimation, so spikes are never hidden.
- **Runtime**: steady-state ms per step for the reference solver and the surrogate (or hybrid), first-call time
  (compilation + one step) separately, and the steady-state speed-up. Only stepper calls are timed (each synchronised
  with `block_until_ready`); diagnostics and drawing are excluded. Hybrid time is `kolmo.hybrid.run`'s own timing,
  policy checks included. Measured live on this machine, while the UI and anything else on the machine are running.
- **Diagnosis** (after a run): *measured facts* (time to failure, failures from `kolmo.rollout.detect_failures`,
  hybrid interventions and solver cost from `kolmo.hybrid.cost`, max CFL) and *rule-based findings* from
  `kolmo.diagnose`: for a surrogate run, the same rules applied to a one-trajectory evaluation of this run
  (`kolmo.evaluate.trajectory_metrics` + `summarize`, labelled "1 trajectory"); for a checkpoint made by
  `kolmo.experiments`, the findings of its test-set evaluation (`diagnosis.json`), labelled with the experiment.
  Interpretations are shown as hypotheses, as `kolmo.diagnose` defines them.

## How the modes run

- **Reference**: `system.stepper` stepped live.
- **Surrogate**: the reference fused over `spacing` solver steps (`kolmo.rollout.coarse`) and `Surrogate.stepper`
  stepped live, side by side.
- **Hybrid**: `kolmo.hybrid.run` (the real policy, not a re-implementation) computes the hybrid trajectory first
  (status "Hybrid rollout…"), then it is replayed next to the live reference.

## Code

| File | Role |
|---|---|
| `kolmo/app/backend.py` | The only file that imports the other Kolmo modules: systems, parameter specs, obstacles, checkpoints, `LiveRun`, post-run summary. API changes elsewhere are absorbed here. |
| `kolmo/app/widgets.py` | `FieldView` (QImage Indexed8 + numpy colour LUT) and `LineChart` (QPainter). No matplotlib. |
| `kolmo/app/window.py` | Main window, worker threads (`Runner`, `Task`), styling. |
| `kolmo/__main__.py` | `python -m kolmo`. |

## Test

    python tests/gui_smoke.py

Runs offscreen (`QT_QPA_PLATFORM=offscreen`, Windows fonts via `QT_QPA_FONTDIR`): starts the window, runs a Diffusion
reference rollout and checks that panels, charts and timings update, pauses and resets a long run, checks the
no-checkpoint empty state, runs Fluid 2D with and without an obstacle, and saves screenshots to `runs/ui/`.
