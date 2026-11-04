# UI progress

Done
- Glass/fabric redesign per `docs/design/reference.webp`: animated backdrop (~5 % of one core, pauses during runs /
  inactive / off-switch), frosted cached cards, nav, top bar with system + device badge + settings, field card with
  shared ω colour bar, error panel toggle, transport + scrubber, 4 metric cards, error-over-rollout comparison chart,
  diagnostic chart, Setup and offline rule-based Assistant, Experiments / Models / Geometry / Reports pages.
- Real modules integrated in `kolmo/app/backend.py`: fluid2d (params, obstacles via `mask`), rollout.coarse, hybrid.run
  and cost, evaluate (rel_l2, time_to_failure, trajectory_metrics, summarize, table), diagnose, experiments outputs,
  surrogates.train.load, geometry (segment + mask_to_grid, flipped to y-up).
- Shortcuts Space / R / E; tooltips; empty states; no fabricated numbers.
- `tests/gui_smoke.py` passes (offscreen). Screenshots: `runs/ui/*.png` (latest: 02_diffusion_reference,
  04_surrogate, 04_hybrid, 04_hybrid_assistant, 06_fluid_cylinder, 07_experiments); `runs/ui/small/` = downscaled.

Not done / next
- Smooth transitions (page fades, value tweening) not implemented; hover/press states are QSS only.
- Photo obstacle path not exercised in the test (needs a photo; SAM 3 optional) — try Geometry → Choose photo.
- Animation cost measured offscreen only; check on screen with Task Manager.
- Mock-up's Simulation/Comparison/Analysis/Rollout tabs not built (one Simulate page instead).

Update (contracts round)
- Static layout: nav + assistant fixed, only the centre (Setup card, fields, metrics, charts, Training) scrolls.
- Systems from kolmo.systems.SYSTEMS/make + pde.LIBRARY; auto-form from describe() (ranges, units, help); field selector
  for multi-field systems (Boussinesq, Gray–Scott).
- Models page: registry (runs/registry.json) + unregistered checkpoints, filter, metrics, Run vs reference, Open report
  (markdown dialog), Register.
- Training panel: phase chips, loss + val-rollout charts, progress, ETA, cancel (when agent.run takes should_stop),
  summary line (assistant.training_summary). Fed by agent "progress" events.
- Zen (Z / button / agent zen event, with the user-off-during-task rule), 250 ms animations, dimmed backdrop.
- Agent activity: plan steps (pending/running/done/failed, spinner), tool chips, progress bar, report link; agent runs
  via backend.agent_run (agent.run(on_event) or legacy ask()).
- Next: real training from the UI (train.fit(on_progress, should_stop) now exists) — Models → Train button.
