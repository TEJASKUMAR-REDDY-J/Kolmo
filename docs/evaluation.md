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

### Kolmogorov flow 64x64, one-step CNN vs rollout-aware CNN (`benchmarks/long_horizon.json`, run `runs/kolmogorov64_long_horizon-88294366`)

Setup: Fluid2D n=64, nu in {0.002, 0.003} (conditioning channel), mu=0.05, A=1, kf=4, dt=0.006, burn-in 2000 solver
steps, 1 surrogate step = 10 solver steps (0.06 time units; median relative change of the field per surrogate step
measured on a spacing-20 dataset was 40 %, which made spacing 20 unlearnable for this CNN, so spacing 10 was used).
Train 20 trajectories (10 seeds x 2 nu), val 4, test 8 (held-out seeds), heldout 4 (nu = 0.005), 200 steps each.
CNN width 32, depth 4, 3x3, pos=4; batch 8, Adam 1e-3 cosine; rollout-aware = unroll 4, full backprop. Wall time
34.8 min total.

| model | split | relL2@1 | @10 | @50 | @100 | @200 | median ttf (steps) | nonphysical steps | energy drift @200 |
|---|---|---|---|---|---|---|---|---|---|
| one-step | test | 0.193 | 0.858 | 1.21 | 4.18 | 1.51e3 | 4 | 689 | +2.7e6 (blow-up) |
| one-step | heldout nu | 0.153 | 0.802 | 1.25 | 2.94 | 1.06e3 | 4.5 | 332 | +1.2e6 |
| rollout K=4 | test | 0.208 | 0.791 | 1.17 | 1.40 | 2.20 | 5 | 0 | +2.75 (x3.75 energy) |
| rollout K=4 | heldout nu | 0.177 | 0.783 | 1.58 | 2.27 | 6.23 | 4.5 | 0 | +37.9 |

First-step-only floor (reference solver restarted from the surrogate's step-1 prediction), test median:
0.19-0.21 at step 1, 0.22 at 10, 0.46-0.50 at 50, 1.03-1.05 at 100, 1.37-1.45 at 200. A perfect model inheriting only the
first-step error would cross 0.5 around step 50; both surrogates cross it at step 4-5, so their failure is error added
by their own later steps, not chaos alone.

Training: one-step stopped by patience at step 1800 (best val rollout relL2 0.811 at 800, 5.8 min); rollout-aware
stopped by the 20-min cap at step 1600 (best 0.757 at 1600, 21 min, still improving).
Runtime (shared CPU, batch 1): reference 7.1-9.7 ms per surrogate step, CNN 5.2-7.0 ms, speedup 1.38-1.39x.

Hybrid (test split, max over 200 steps of the median relL2; cost = solver steps / pure reference):

| policy | cost % | one-step CNN | rollout CNN |
|---|---|---|---|
| surrogate only | 0 | 1.51e3 | 2.20 |
| re-anchor every 20 | 5 | 1.44 | 1.77 |
| every 10 | 10 | 1.45 | 1.71 |
| every 5 | 20 | 1.44 | 1.61 |
| every 2 | 50 | 1.36 | 1.29 |
| energy drift > 20 % | 29 / 68 | 1.42 | 1.42 |
| residual > 0.05 | - | unavailable (Fluid2D.residual refuses dt = 0.06 by its CFL check) | same |

Findings (measured): rollout-aware training removed the blow-up (0 vs 689 non-finite/blown-up states, error at 200 steps
2.2 vs 1.5e3) but did not lengthen time-to-failure (5 vs 4 steps) and is slightly worse at step 1 (0.208 vs 0.193). It
still injects energy (+275 % at step 200). Re-anchoring removes the blow-up at 5 % cost but no policy keeps the error
under 0.5: re-anchoring from a drifted state cannot undo divergence, and values near 1.4 (= sqrt 2) mean decorrelated
fields. Held-out nu = 0.005: error at step 200 2.84x the in-distribution error for the rollout model (distribution shift).
Interpretation: the binding constraint is the one-step error (~20 %); next step is to reduce it (longer training,
larger/FNO model, smaller spacing) and to add statistical metrics (spectra, energy statistics) for horizons beyond
predictability.

### FNO one-step on the same data (`benchmarks/fno.json`, run `runs/kolmogorov64_fno-9ebbb269`)

FNO width 24, modes 12, depth 4, pos=4, one-step, 2000-step budget (stopped by patience at 1400, best val rollout relL2
0.41 at step 400, 11.5 min).

| split | relL2@1 | @10 | @50 | @100 | @200 | ttf per test trajectory | nonphysical | energy drift @200 |
|---|---|---|---|---|---|---|---|---|
| test | 0.149 | 0.376 | 1.33 | 1.40 | 1.54 | 17, 11, 11, 14, 14, 16, 16, 19 (median 15) | 0 | +0.63 |
| heldout nu | 0.089 | 0.318 | 1.31 | 1.69 | 1.70 | median 16 | 0 | +1.14 |

First-step-only floor (test): 0.149 / 0.103 / 0.301 / 0.861 at steps 1 / 10 / 50 / 100.
Hybrid: surrogate only 1.69, every 10 (10 % cost) 1.56, every 2 (50 % cost) 1.45 (max of median relL2).
Runtime: FNO 12.8 ms vs reference 8.4 ms per surrogate step: speedup 0.655x (slower than the solver at 64x64 on CPU).
Measured: the FNO's time-to-failure is 3x the CNNs' (15 vs 4-5 steps) and its 10-step error is less than half (0.38 vs
0.79-0.86). This comes from one-step training, so here the architecture mattered more than rollout-aware training.
