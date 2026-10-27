# Surrogates: data, models, training, execution modes

Code: `kolmo/surrogates/{data,models,train}.py`, `kolmo/rollout.py`, `kolmo/hybrid.py`. Tests: `tests/test_surrogates.py`
(Diffusion system, about 1 minute: `python -m pytest tests/test_surrogates.py -q`).

## Data (`kolmo/surrogates/data.py`)

`DataConfig` (all fields JSON-serialisable):

| field | meaning |
|---|---|
| `system` | `"diffusion"` or `"fluid2d"` (registry `SYSTEMS`) |
| `params` | base parameters; the system's `validate()` fills defaults |
| `steps` | surrogate steps recorded per trajectory (states 0..steps) |
| `spacing` | solver steps per surrogate step (surrogate dt = spacing x params["dt"]) |
| `burn_in` | solver steps discarded after the initial condition (lets turbulence develop) |
| `train_seeds`, `val_seeds`, `test_seeds` | disjoint seed lists (checked) |
| `vary` | `{param: [values]}`: every combination is used in train, val and test |
| `heldout` | `{param: [values]}`: split `heldout` = test seeds x these values; must not overlap training values |
| `cond` | parameter names given to the network as conditioning channels (must vary) |

Normalisation statistics come from the train split only: per-channel mean/std over all train states, and mean/std of
the conditioning values over train trajectories (std 0 -> 1). Datasets are cached at
`runs/cache/data-<system>-<sha1 of the config JSON>.npz` (`runs/` is gitignored); a changed config gives a new file.

## Models (`kolmo/surrogates/models.py`) - model card

* **Input**: `z` of shape `(B, C, H, W)`, float32, normalised state `z = (s - mean_c) / std_c` with the train-split
  per-channel statistics stored in the checkpoint; `c` of shape `(B, P)`, normalised parameters
  `(p - cond_mean) / cond_std`. Physical units are those of the system (Diffusion: temperature; Fluid2D: vorticity,
  1/time).
* **Conditioning**: each parameter becomes one constant plane; `pos = k` adds `sin/cos(j x)`, `sin/cos(j y)` planes for
  `j = 1..k` (needed for spatially fixed forcing such as Kolmogorov `cos(4 y)`, which a translation-equivariant network
  cannot locate otherwise). Network input channels: `C + P + 4 pos`.
* **Output**: `z_next = z + f(z, c)`, shape `(B, C, H, W)`, normalised. The last layer is zero-initialised, so an untrained
  network is the persistence model. One application = one surrogate step = `spacing` solver steps.
* **ResCNN** (`kind: "cnn"`): `depth` periodic (wrap-padded) `kernel x kernel` convolutions with `width` channels and GELU,
  then a zero-initialised convolution to C channels. Receptive field `(depth + 1)(kernel - 1) + 1` cells.
* **FNO** (`kind: "fno"`): 1x1 lift to `width`; `depth` blocks `h <- GELU(SpectralConv(h) + 1x1(h))`, where SpectralConv
  multiplies the lowest `modes x modes` Fourier coefficients (both signs of ky) by learned complex matrices; 1x1
  projection (hidden 2 width) to C, zero-initialised. Global receptive field.
* **Physical wrapper**: `Surrogate.stepper(params)` returns a jitted `(C, H, W) -> (C, H, W)` map in physical units
  (normalise, apply, de-normalise). It never modifies `params`.
* **Checkpoint**: `<name>.npz` (Flax params flattened, keys joined by `/`) + `<name>.json` with `version`
  (`kolmo-surrogates 0.1`), `arch` (the `build()` spec), `spacing`, `fields`, `norm` (mean, std, cond_names, cond_mean,
  cond_std), `data` (DataConfig) and `data_hash`, `train` (TrainConfig), `history`, `train_minutes`, library versions.
  Load with `kolmo.surrogates.train.load(path_without_extension)`.
* **Inference requirements**: JAX + Flax (linen) + NumPy, CPU is enough. Same grid size as training for the CNN weights
  to be meaningful in physical units (the CNN is local in cells; the FNO is resolution-independent in its weights but
  was only evaluated at the training resolution). Inputs outside the training parameter range are extrapolation.
* **Compatibility note**: Flax 0.8.4 reads `Trace.level`, which JAX 0.7 removed; `models.py` contains a one-line shim
  (`flax.core.tracers.trace_level`) so linen modules work. Remove it when Flax is upgraded.

## Training (`kolmo/surrogates/train.py`)

`TrainConfig`: `model` (arch spec), `unroll` K (1 = one-step, K > 1 = rollout-aware), `pushforward`, `steps`, `batch`,
`lr` (Adam, cosine decay to 0, global-norm clip 1), `seed`, `eval_every`, `val_horizon`, `patience`, `max_minutes`.

* Loss (normalised units): mean over k = 1..K of `MSE(z_k, target_k)` with `z_k = net(z_{k-1}, c)` from a random train
  window. `pushforward = true` applies stop-gradient to every step input: the inputs come from the model's own rollout,
  but each gradient passes through a single network application (Brandstetter et al. 2022).
* Early stopping: every `eval_every` steps the validation rollout error (mean over val trajectories and surrogate steps
  1..`val_horizon` of the physical-units relative L2, rolled out from step 0) is computed; the best parameters are kept;
  training stops after `patience` evaluations without improvement or after `max_minutes`.
* Determinism: windows are drawn from `numpy.random.default_rng(seed)`, weights from `PRNGKey(seed)`; repeated runs on
  this CPU give identical histories (tested).

## Execution modes (`kolmo/rollout.py`, `kolmo/hybrid.py`)

* `rollout(step, s0, n)` -> `core.Trajectory`. `compile_seconds` = one warm-up call (compile + one step);
  `seconds` = the n recorded steps. Failure events (first step of each kind): `nonfinite`, `blowup`
  (max|s| > 10 x max|s0| by default), `energy_drift` (if a reference energy and tolerance are given).
* `hybrid.run(HybridPolicy(...), system, params, surrogate, s0, n)`; modes `reference`, `surrogate`, `hybrid`.
  In hybrid mode a step becomes a solver step (`spacing` solver steps from the current hybrid state) when: the step is
  a multiple of `every`; or the proposal's `system.residual` (with dt -> spacing x dt) exceeds `residual_tol`; or the
  proposal's mean square deviates from `energy_ref` (default: that of s0) by more than `energy_tol`; or the proposal
  is non-finite. Every intervention is an event `{"step", "kind": "intervention", "reason", "value"}`; parameters are
  never altered. `hybrid.cost(traj, spacing)` gives solver steps used and the fraction of a pure reference run.
* The solver never sees the true trajectory during a hybrid run: re-anchoring advances the solver from the possibly
  drifted state, so its cost is real and it cannot undo divergence that already happened.
* Fluid2D's `residual()` re-validates its parameters including the CFL limit, so it refuses dt = spacing x dt; the
  residual metric and the residual trigger are then reported as unavailable (recorded in the results, never invented).
