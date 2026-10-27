# Progress: reference fluid solver (fluid2d)

Paused on owner request. No background processes are running. All files are in a working state.

## Done and verified

Files:
* `kolmo/systems/fluid2d.py`: `Fluid2D` (System protocol, name "fluid2d", fields ("vorticity",)), `cylinder_mask`,
  a throughput benchmark in `__main__`.
* `tests/test_fluid2d.py`: 8 tests.
* `docs/physics.md`: equations, units, Re definitions, discretisation, stability, ICs, penalisation, diagnostics,
  residual, API, verification tables, performance, limitations.

Method: pseudo-spectral (rfft2) with the 2/3 rule. Time stepping is integrating-factor RK3 (Kutta): viscosity and drag
are exact, advection and forcing explicit, third order. The CFL bound is 0.827 and `cfl_max` defaults to 0.5. Obstacles
use Brinkman penalisation with the free stream U_inf, solved exactly as a split substep (first order), plus a
raised-cosine vorticity sponge. `batched_stepper` uses vmap over seeds or over the dynamic params. `rollout` runs one
lax.scan with AOT compile timing kept separate.

Real outputs:
* `python -m pytest tests/test_fluid2d.py -q -s`: `8 passed in 101.26s`.
* `python -m tests.test_fluid2d`: all 8 `PASS` lines, exit code 0.
  * Taylor-Green: relative error 9.72e-06.
  * Kolmogorov laminar: relative error 7.44e-06.
  * Energy budget: worst 1.58e-04, residual 2.35e-04.
  * Temporal order: fitted 3.06, from errors 8.41e-3 / 3.52e-3 / 1.01e-3.
  * Cylinder at Re = 20:
    * n = 64: Cd 2.1789, residual 7.5e-4, max inside speed 0.094, mean 0.078.
    * n = 192: Cd 2.2238, residual 1.9e-4, max inside speed 0.041, mean 0.027.
* `python -m kolmo.systems.fluid2d`, two runs on the shared, contended CPU, in steps/s:
  * 64^2: 446 and 367.
  * 128^2: 91 and 148.
  * 256^2: 54 and 65.
  * Compile time: 1-2.6 s.

## Not finished (cosmetic only, no code impact)

These edits were blocked by tool-permission timeouts:
1. `docs/physics.md`, paragraph after the cylinder table:
   * Replace "Cd settles to < 1e-3 by t ~ 15" with "Cd changes by < 0.1 % between t = 15 and t = 20". The measurement
     at n = 64 went from 2.1802 to 2.1789.
   * Replace "~3-10 % above the unbounded-cylinder literature value of about 2.0-2.1" with "~2-9 % above the commonly
     quoted value of about 2.05 (not re-derived here)".
2. `tests/test_fluid2d.py`, docstring of `test_penalised_cylinder`: replace "(Cd settled to < 1e-3)" with "(Cd
   changes < 0.1 % over the last 5 time units)".

## Next steps to resume

1. Apply the two edits above.
2. Optional: a mask that scales with the grid, or a smoothed mask, to get a clean Cd convergence study. Today Cd is
   dominated by the staircase extent D, which ranges from 0.92 to 1.08 across n.
3. Optional: rerun the benchmark on an idle machine and replace the performance table.
