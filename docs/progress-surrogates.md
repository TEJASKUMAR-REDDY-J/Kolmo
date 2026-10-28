# Progress: surrogates, evaluation, hybrid execution

## Done and verified

- `kolmo/surrogates/{data,models,train}.py`, `kolmo/rollout.py`, `kolmo/hybrid.py`, `kolmo/evaluate.py`,
  `kolmo/diagnose.py`, `kolmo/experiments.py`; docs `docs/surrogates.md`, `docs/evaluation.md`.
- `python -m pytest tests/test_surrogates.py -q` -> `10 passed` (Diffusion system: data splits/cache/stats, model
  shapes + identity init + periodic equivariance, deterministic training that beats persistence, checkpoint
  round trip, rollout failure events, hybrid modes/cost/events, metric definitions, diagnosis rules, end-to-end
  experiment writing every artifact).
- CPU performance findings (measured, 64x64, batch 8): convolutions inside `lax.scan` ran ~90x slower on XLA CPU
  (CNN gradient 11.7 s vs 0.13 s straight-line), so training unrolls K and the validation horizon in Python.
  Training step after the fix: CNN (width 32, depth 4) K=4 0.64 s; FNO (width 24, modes 12) K=1 0.26 s, K=4 1.0 s.
  Inference per surrogate step: CNN 5.5 ms, FNO 13 ms.
- Flax 0.8.4 + JAX 0.7.1 incompatibility (`Trace.level`) shimmed in `models.py`.

## Known limitations

- `Fluid2D.residual` re-validates params incl. CFL, so the residual over a surrogate step (dt = 20 x 0.006) is refused;
  residual metric and residual-triggered hybrid are reported as unavailable for Kolmogorov runs. Request to the solver
  owner: a residual that does not apply the CFL check (dt only divides).

## Kolmogorov experiments

Done: runs/kolmogorov64_long_horizon-88294366 (one-step CNN vs rollout-aware CNN) and runs/kolmogorov64_fno-9ebbb269 (FNO). Numbers and findings: docs/evaluation.md, Results.

Not run (pending): benchmarks/baseline.json on its own (its model is identical to cnn_one_step in long_horizon), rollout-aware FNO, pushforward variant, multiple training seeds.

## Next steps
1. Rollout-aware FNO (unroll 4, about 1 s/step -> about 1200 steps in 20 min), fine-tuned from the one-step FNO checkpoint.
2. Statistical metrics past the predictability horizon (energy spectrum, enstrophy statistics).
3. Ask the solver owner for a residual without the CFL check so the residual trigger can be used.
