# Kolmo desktop UI

    python -m kolmo          # Space = run / pause / resume, R = reset, E = error panel

PySide6 6.7 (QtCore/QtGui/QtWidgets only) + NumPy. All computation runs in worker threads. Design reference:
`docs/design/reference.webp` (its numbers are illustrative; the app shows only measured values or a pending state).

## Look

Violet night palette over an animated "fabric" backdrop (numpy ribbons at 96 x 56, upscaled; 6 fps; each frame repaints
only the gaps between cards, cards re-sample the blurred backdrop every 5 s). Measured offscreen on this laptop:
~5 % of one core while animating; it pauses while a simulation runs (clean timings), when the window is inactive or
minimised, and can be switched off (⚙ → Animated background, remembered). Cards are frosted glass: blurred backdrop,
translucent fill, hairline border, top highlight; their background is cached so content repaints stay cheap.

## Layout

- **Top bar**: brand; system selector (a system whose module fails to import is greyed out with the reason); device
  badge from JAX (`Local (CPU)`); ⚙ settings.
- **Left nav**: Simulate · Experiments (tables from `runs/*/results.json`) · Assistant · Models (every checkpoint with
  what it was trained on) · Geometry (photo → mask via `kolmo.geometry`, becomes the Simulate obstacle) · Reports
  (`report.md` viewer). Pages read run files; with none they say how to create them.
- **Simulate (centre)**
  - Field card: **Reference (numerical)** and **Surrogate (neural)** / **Hybrid (neural + solver)** with one shared
    vertical colour bar (set from the reference only, only widens during a run, so decay fades and a blown-up
    prediction saturates), optional **Absolute error** panel with its own scale (E). y-up, obstacles grey.
    Transport: ▶/❚❚, reset, scrubber over the recorded steps (end = follow live), step and physical time.
  - Metric cards (measured, else a pending caption): relative L2 error at the shown step; energy drift (mean square
    of prediction / reference − 1, flagged beyond `kolmo.diagnose.DRIFT_TOL`); steady-state speed-up with both
    ms/step; stability horizon (first step with error > `evaluate.time_to_failure` threshold, or "> N so far").
  - **Error over rollout** (log): the current run plus up to 4 earlier runs of this session (dashed) for comparison,
    failure threshold, hybrid interventions (yellow dotted) and detected failures (red). **Diagnostic**: any
    `system.diagnostics` quantity, reference vs prediction.
- **Right column**: **Setup** (parameters with units and `auto`, seed, obstacle none/circle/square/photo with
  thumbnail, execution mode, checkpoint picker with training distribution and out-of-distribution flags, hybrid policy
  fields, steps, Run, save/open config JSON) and **Assistant** (offline · rule-based): measured facts and
  `kolmo.diagnose` findings after each run, a plan from the findings' "next" steps, Run experiment, and a command box
  (run / explain / plan / train). `kolmo/app/assistant.py` is the provider seam for a hosted model later; nothing
  calls a network API.

## Modes

Reference: `system.stepper` live. Surrogate: reference fused over the checkpoint's `spacing` (`rollout.coarse`) and
`Surrogate.stepper`, live side by side. Hybrid: `kolmo.hybrid.run` (the real policy) first, then replayed next to the
live reference. A checkpoint that the current `kolmo.surrogates` code cannot apply is refused with a retrain hint.
Runtime: only stepper calls are timed (`block_until_ready`), first call (compilation) separately.

## Code and test

`kolmo/app/backend.py` (the only importer of other Kolmo modules), `widgets.py` (backdrop, glass, fields, charts),
`window.py`, `assistant.py`, `kolmo/__main__.py`. `python tests/gui_smoke.py` runs offscreen and saves screenshots to
`runs/ui/` (Diffusion reference, pause/reset, surrogate + hybrid when a diffusion checkpoint exists, Fluid 2D forced
and past a cylinder, Experiments page).
