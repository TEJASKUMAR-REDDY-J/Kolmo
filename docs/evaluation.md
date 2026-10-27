# Evaluation, diagnosis and experiments

Code: `kolmo/evaluate.py`, `kolmo/diagnose.py`, `kolmo/experiments.py`, configs in `benchmarks/`.

## Metric definitions

Notation: `s_t` predicted state, `r_t` reference state at the same surrogate step t (one surrogate step = `spacing`
solver steps), norms are L2 over all C x H x W values. Every metric is computed per trajectory, then aggregated over
trajectories as median and mean; a non-finite value counts as +inf, and JSON writes `null` for non-finite numbers.

| metric | definition | units / denominator |
|---|---|---|
| `rel_l2[t]` | `||s_t - r_t|| / ||r_t||` | dimensionless; denominator = reference norm at the same step |
| `rel_l2_field[t, c]` | same, channel c only | dimensionless |
| `corr[t]` | Pearson correlation of `s_t` and `r_t` over all values | dimensionless, 1 = perfect pattern, blind to amplitude |
| `horizons` | median `rel_l2` at steps 1, 10, 50, 100, 200, 500, 1000 (those <= T) | dimensionless |
| `time_to_failure` | first step t >= 1 with `rel_l2[t] > threshold` (default 0.5) or a non-finite state | surrogate steps; `null` = not reached within T (censored, count reported) |
| `residual[t]` | `system.residual(s_{t-1}, s_t, params with dt -> spacing dt)`; the same for `r` gives the solver's floor | dimensionless (system-defined); `residual_unavailable` when the system refuses the coarse dt |
| `energy[t]` | mean of `state^2` (`rollout.mean_square`) | field units^2 (vorticity: 2 x enstrophy density) |
| `energy_drift[t]` | `E_s[t] / E_r[t] - 1` | dimensionless; denominator = reference energy at the same step |
| `nonphysical_steps` | steps with a non-finite state or `max|s_t| > 10 max_t |r_t|`, summed over trajectories | count |
| `first_step_only[t]` | `rel_l2` of the reference solver restarted from the surrogate's step-1 prediction | dimensionless; the error a perfect model would show after inheriting only the first-step error |
| `diagnostics` | `system.diagnostics` (Fluid2D: kinetic energy, enstrophy, CFL, ...) at step 0 and the horizons, mean over trajectories | system units |
| runtime | steady-state wall seconds per surrogate step, compile excluded, batch 1, float32, this CPU (shared); reference = `spacing` fused solver steps; `speedup` = reference / surrogate | seconds; conditions string stored with the numbers |
| hybrid | per policy: median `rel_l2` over time, its maximum, time-to-failure, `cost_fraction` = solver steps used / solver steps of a pure reference run, interventions by reason | dimensionless / count |

Generalisation table: one row per (model, split), `test` = held-out seeds at training parameter values, `heldout` =
held-out parameter values (with test seeds).

## Diagnosis rules (`kolmo/diagnose.py`)

Each finding has `level`, `model`, `fact` (measured), `evidence` (the numbers), `interpretation` (hypothesis) and
`next` (suggested experiment). Rules:

1. Error growth: least-squares fit `log rel_l2 = a + r t` over steps with error <= threshold (>= 5 points); reports
   r, doubling steps `ln 2 / r`, R^2. Compares r with the growth rate of `first_step_only`: within 25 % -> the dynamics
   amplify the first-step error (intrinsic); otherwise the surrogate's repeated errors drive the growth.
2. Time-to-failure: number of trajectories that cross the threshold, earliest and median step.
3. Non-finite / blown-up states -> critical (instability).
4. Energy drift at the last step beyond 5 %: up -> spurious energy injection, down -> spurious dissipation.
5. Residual ratio surrogate / reference > 10 -> physically inconsistent transitions.
6. Held-out parameter error > 2x the test error at some horizon (or failures only on held-out parameters) ->
   distribution shift.
7. Hybrid: cheapest policy that keeps the median error under the threshold, with its cost; or a warning if none does.
8. Model comparison: every model vs the first one, error ratio by horizon and failure counts.

## Experiments

`python -m kolmo.experiments run <config.json>` writes `runs/<name>-<hash>/` with `config.json`, `env.json`,
`checkpoints/`, `results.json`, `diagnosis.json`, `error_vs_horizon.png`, `energy_vs_time.png`,
`hybrid_tradeoff.png` and `report.md`. Running the run directory's own `config.json` reproduces the same directory
(existing checkpoints with identical train and data configs are reused; delete `checkpoints/` to retrain).

Results of the benchmark runs are in the section below; anything not run is marked pending.

## Results
