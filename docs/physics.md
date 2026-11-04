# Physics reference: `fluid2d` (kolmo/systems/fluid2d.py)

The reference fluid solver. Every number in this file was produced on the project laptop (i3-10110U, 2 cores /
4 threads, JAX 0.7.1 CPU, float32) by the commands quoted next to it.

## Governing equations

2D incompressible Navier-Stokes in vorticity form on the periodic square [0, L)^2 (default L = 2 pi):

    dw/dt + (U_inf e_x + u) . grad w = nu lap w - mu w + f_w  +  [penalisation + sponge]
    lap psi = -w,    u = d psi/dy,    v = -d psi/dx,    w = dv/dx - du/dy

* `w` vorticity (the single state field, `fields = ("vorticity",)`), `u = (u, v)` the periodic, zero-mean velocity.
* `U_inf` a uniform free stream along +x (default 0). The periodic box cannot represent a mean velocity through the
  vorticity, so it is imposed separately and held fixed (equivalently, a uniform pressure gradient sustains it).
* `nu` kinematic viscosity, `mu` linear (Rayleigh / bottom) drag.
* Kolmogorov forcing. Body force per unit mass `f = (-A sin(k_f y), 0)`, whose curl is `f_w = A k_f cos(k_f y)`, with
  `k_f = 2 pi kf / L` (`kf` = integer number of periods across the box, default 4) and `A` an acceleration
  (default 0 = unforced). Its exact laminar steady state is `w = A k_f cos(k_f y) / (nu k_f^2 + mu)`,
  `u = -A sin(k_f y) / (nu k_f^2 + mu)` (the advection of a single shear mode vanishes identically).

### Units

Nondimensional, density 1. Length unit: the box is `L` (with the default L = 2 pi wavenumbers are integers).
Velocity unit: whatever the user sets through `ic_amp` (rms speed of the random IC), `U_inf` or `A`. Time =
length / velocity. Forces in `diagnostics` are per unit depth (2D), density 1.

### Reynolds number (`Re` is accepted instead of `nu`)

`Re = U_ref L_ref / nu`; `validate` fills whichever of `nu` / `Re` is missing (nu = 1e-3 if neither), and rejects a
pair that disagrees.

| case | U_ref | L_ref |
|---|---|---|
| obstacle (`mask` given) | `U_inf` | `D` = cross-stream extent of the mask (rows containing a solid cell x dx) |
| forced (`A != 0`) | `sqrt(A / k_f)` | `1 / k_f` (so Re = sqrt(A / k_f^3) / nu, the usual Kolmogorov-flow definition) |
| otherwise | `ic_amp` | `L / 2 pi` |

## Discretisation

* Fourier pseudo-spectral on an n x n grid (`n` even >= 16), `rfft2` over (y, x) (row = y, column = x).
  Velocity from `psi_hat = w_hat / |k|^2` (mean mode 0).
* Advection `-(U_inf + u) w_x - v w_y` is computed in physical space from spectral derivatives and de-aliased by the
  2/3 rule: modes with |m_x| or |m_y| > K = floor((n-1)/3) are zeroed in the product, and the state itself is kept
  in that band after every step. Because the truncation is an exact Galerkin projection of the quadratic term,
  advection conserves energy and enstrophy to round-off (this is what makes the energy-budget test tight).
* Time stepping: integrating-factor RK3. With `lam(k) = -(nu |k|^2 + mu)` and `E(h) = exp(lam h)` the linear terms
  are integrated exactly per mode; advection + forcing `N` use Kutta's 3-stage third-order scheme in the
  integrating-factor variable (all exponentials decay, no `exp(+...)` factors):

        k1 = N(w);  k2 = N(E(h/2)(w + h/2 k1));  k3 = N(E(h)(w - h k1) + 2h E(h/2) k2)
        w+ = E(h)(w + h/6 k1) + h/6 (4 E(h/2) k2 + k3)

  Global order 3 (measured 3.06, see Verification). Viscosity and drag impose no stability limit.
* Stability: for 2/3-dealiased advection the largest eigenvalue modulus is (2 pi / 3 dx) max(|u| + |v|); RK3 is
  stable on the imaginary axis up to sqrt(3), so the bound is `CFL = dt max(|U_inf + u| + |v|) / dx <= 3 sqrt(3) /
  2 pi = 0.827` (`CFL_STABLE`). `validate` applies a safety limit `cfl_max` (default 0.5) to an a-priori estimate
  of max(|u|+|v|): `U_inf` (x2 with an obstacle) + 4 ic_amp (random IC) or ic_amp (Taylor-Green) + min(laminar
  Kolmogorov amplitude, 8 sqrt(A / k_f)). A too-large `dt` raises `ValueError` that quotes the estimate and the
  largest admissible `dt`; `dt = None` picks that value rounded down to 2 significant digits. The estimate is a
  heuristic: `diagnostics()["cfl"]` is the real value, and `rollout` adds an event if the final state exceeds
  `CFL_STABLE` or is non-finite.

## Initial conditions (`ic`)

* `"random"` (default without obstacle): white noise from `numpy.random.default_rng(seed)` restricted to the
  integer-wavenumber shell `ic_k = (lo, hi)` (default 1..6), scaled to rms speed `ic_amp`. Deterministic: same seed
  and parameters give a bit-identical state.
* `"taylor_green"`: `u = U0 sin(kx) cos(ky), v = -U0 cos(kx) sin(ky)`, `w = 2 k U0 sin(kx) sin(ky)`, `k = 2 pi tg_k / L`,
  `U0 = ic_amp`. An exact Navier-Stokes solution decaying as `exp(-(2 nu k^2 + mu) t)`.
* `"zero"` (default with obstacle): fluid at rest relative to the free stream (impulsive start).

## Obstacles: Brinkman penalisation with a free stream

`mask` is a boolean (n, n) array (row = y, column = x; e.g. from a photo). In the obstacle frame the fluid velocity
`U_inf e_x + u` is penalised towards 0 inside the mask:

    du/dt = ... - (chi / eta)(U_inf e_x + u),     chi = mask (0/1), eta = permeability time (default 1e-3)

The penalisation (and the optional sponge) is applied as a split substep S before the RK3 step and solved exactly:
`u <- u exp(-chi dt / eta)`, vorticity = spectral curl, 2/3 filter, then `U_inf` is re-imposed. This is
unconditionally stable for any `eta`; with the default `eta << dt` it is effectively direct forcing (velocity set to
zero inside the mask once per step), and the curl of the masked velocity is the vortex sheet that the RK3 step then
diffuses and advects into the boundary layer.

Sponge / fringe: with a mask, a strip of width `sponge` x L (default 0.1) at the right edge of the box (just upstream
of x = 0 through periodicity) damps vorticity as `dw/dt = -sponge_rate sigma(x) w`, `sigma` a raised-cosine bump
from 0 to 1 and back; default `sponge_rate = 10 U_inf / (sponge L)`, i.e. ~5 e-folds for a wake crossing the strip.
The free stream re-enters clean (velocity -> U_inf). Place the body well left of the strip.

## Diagnostics (`diagnostics(state, params)`)

Box means `<.>`, density 1:

| key | meaning |
|---|---|
| `energy` | `E = 0.5 <u^2 + v^2>` of the periodic part (U_inf excluded), per unit area |
| `enstrophy` | `Z = 0.5 <w^2>` |
| `max_abs_vorticity`, `mean_vorticity` | `max |w|`, `<w>` (the latter is 0 to round-off in a periodic box) |
| `forcing_power` | `P = <u . f> = -A <u sin(k_f y)>` |
| `cfl` | `dt max(|U_inf + u| + |v|) / dx` of this state |
| `drag`, `lift` (mask only) | force of the fluid on the body per unit depth: `(1 - exp(-dt/eta))/dt * integral chi (U_inf + u, v) dA`, the momentum the next penalisation substep removes per unit time |
| `cd`, `cl` (mask and U_inf > 0) | `drag / (0.5 U_inf^2 D)`, `lift / (0.5 U_inf^2 D)` |

Energy budget (exact for the Galerkin-truncated system): `dE/dt = P - 2 nu Z - 2 mu E`.

## Residual (`residual(prev, nxt, params)`)

Physical-consistency metric of one transition `w0 -> w1` (not a prediction error):

    R = (w1 - w0)/dt - P - 0.5 [F(S(w0)) + F(w1)],     P = (S(w0) - w0)/dt
    F(w) = -(U_inf e_x + u) . grad w + nu lap w - mu w + f_w
    r = ||R|| / (||(w1 - w0)/dt|| + ||P|| + sum_T ||0.5 [T(S(w0)) + T(w1)]||),   T in {advection, viscous, drag, forcing}

`S` is the penalisation/sponge substep (identity without them). Normalising by the size of the whole balance keeps
`r` meaningful for steady states. For the reference solver `r` is the trapezoid-vs-RK3 difference, O(dt^2), plus
float32 round-off. A surrogate whose transitions give `r` far above the reference value is violating the equation.

Measured reference values: Taylor-Green 7.5e-6 (round-off); decaying 2D turbulence n = 64, nu = 1e-3, dt = 0.012
(CFL 0.34-0.41) 1.3e-3; forced turbulence n = 64 (energy-budget test) 2.4e-4; penalised cylinder 7.5e-4 (n = 64),
3.6e-4 (n = 128), 1.9e-4 (n = 192). An earlier plain-trapezoid definition (viscous term also by trapezoid) grew to
1.2e-2 / 3.2e-2 / 5.2e-2 for the cylinder at n = 128 / 192 / 256, because grid-scale modes have `nu |k|^2 dt ~ 1`;
the variation-of-constants form removes that.

## API beyond the System protocol

* `batched_stepper(params, batch_params=None)`: `(B, 1, n, n) -> (B, 1, n, n)`. Without `batch_params`, vmaps one
  parameter set over a batch of states (e.g. seeds). With a list of B override dicts, every parameter in `DYNAMIC`
  (`nu, mu, A, dt, U_inf, eta, sponge_rate`, the mask) is vmapped per member; `STATIC` ones (`n, L, kf, sponge`) must
  agree.
* `rollout(params, state0, steps, every=1, batch_params=None)` -> `kolmo.core.Trajectory`: the whole loop runs in
  one `lax.scan` (inner `fori_loop` of `every` steps); `compile_seconds` is the ahead-of-time XLA compile of the scan,
  `seconds` the run of the compiled program. A batched `state0` gives `states` of shape (T+1, B, 1, n, n).
* `cylinder_mask(n, L, center, radius)`: disc mask helper.

## Verification (tests/test_fluid2d.py)

Command: `python -m pytest tests/test_fluid2d.py -q -s` (also runs as `python -m tests.test_fluid2d`).

Result of the last run: `8 passed in 101.26s`. Printed measurements:

| test | setup | measured | asserted |
|---|---|---|---|
| Taylor-Green decay | n = 32, nu = 0.05, mu = 0.1, dt = 0.05, 100 steps | max rel error 9.7e-6 vs `exp(-(2 nu k^2 + mu) t)` | < 1e-4 |
| Kolmogorov laminar | n = 32, nu = 0.05, mu = 0.2, A = 1, kf = 2, from rest, t = 40 | amplitude rel error 7.4e-6 (exact 5.0000) | < 1e-4 |
| energy budget | n = 64, nu = 2e-3, mu = 0.05, A = 1, kf = 4, random IC, 200 steps | worst `|dE/dt - (P - 2 nu Z - 2 mu E)| / (|P| + 2 nu Z + 2 mu E)` = 1.6e-4 (dE/dt by central difference) | < 1e-3 |
| temporal order | n = 32, nu = 0.01, random IC (shell 1-4), T = 1.2, ref dt = 0.005 | max-abs errors 8.41e-3 / 3.52e-3 / 1.01e-3 at dt = 0.04 / 0.03 / 0.02, fitted order 3.06 | in (2.7, 3.5) |
| determinism | n = 32, seeds 7, 7, 8 | same seed bit-identical (IC and 20-step rollout), different seed differs | exact |
| CFL refusal, Re | dt = 1 at n = 64 | `ValueError` quoting the admissible dt; that dt accepted; Re = 500 -> nu = 2e-3; inconsistent nu/Re rejected | exact |
| batched stepper | 3 seeds; 3 viscosities vmapped | equals the single stepper | atol 1e-5 |
| penalised cylinder | Re = 20, nominal D = 1, L = 4 pi, U_inf = 1, sponge 10 %, t = 20 | see below | Cd in (1.8, 2.6), \|Cl\| < 1e-3, inside speed < 0.15, residual < 1e-2, Cd change < 5 % from n = 64 to 192, mean leakage halves |

Cylinder (staircase disc at x = L/4; Cd from the penalisation force, normalised with the mask extent D):

| n | points per D | mask D | dt | Cd (t = 20) | max \|u\|/U inside, 1-cell eroded | mean \|u\|/U inside mask |
|---|---|---|---|---|---|---|
| 64 | 5 | 0.982 | 0.049 | 2.179 | 0.094 | 0.078 |
| 96 | 7.6 | 0.916 | 0.032 | 2.228 | 0.072 | 0.053 |
| 128 | 10 | 1.080 | 0.024 | 2.099 | 0.053 | 0.042 |
| 192 | 15 | 0.982 | 0.016 | 2.224 | 0.041 | 0.027 |
| 256 | 20 | 1.031 | 0.012 | 2.165 | 0.046 | 0.021 |

(n = 96 and 256 rows come from a separate run of the same setup, not from the test.) Cd changes by
< 0.1 % between t = 15 and t = 20. What this shows, honestly: Cd lies in 2.10-2.23, i.e. ~2-9 % above the commonly
quoted value of about 2.05 at Re = 20 (not re-derived here), which is the expected direction for 8 % blockage in a periodic array; Cl is zero to
round-off as the symmetric steady wake requires; the leakage inside the body falls roughly like dx. There is no
clean monotone convergence of Cd: it is dominated by the staircase extent D (which jumps between 0.92 and 1.08 as n
changes), and at identical mask extent (n = 64 vs 192) Cd changed by +2.1 %. Converged-geometry studies need a
mask that scales with the grid (or a smoothed mask) and are not done here.

## Performance

Command: `python -m kolmo.systems.fluid2d` (forced turbulence, nu = 1e-3, A = 1, kf = 4, mu = 0.05, dt from the CFL
estimate; a single `lax.scan` per size). Compile = XLA compile of the scan (excluded from the rate); steady =
compiled program only. The laptop was shared with three other engineers' jobs during measurement, so two
consecutive runs are given; treat the spread as the uncertainty.

| n | compile (s) | steady, run 1 (steps/s) | steady, run 2 (steps/s) | batch of 8 seeds, run 1 / 2 (member-steps/s) |
|---|---|---|---|---|
| 64 | 1.45 / 2.00 | 446 | 367 | 704 / 458 |
| 128 | 0.94 / 1.77 | 91 | 148 | 90 / 232 |
| 256 | 2.57 / 1.55 | 54 | 65 | 46 / 41 |

An earlier quieter measurement at n = 64 (unforced, 200 steps) ran at ~850 steps/s. One step costs ~17 real FFTs
of size n^2 (15 in the three RK stages + 2 for the physical-space state); obstacles add ~6. vmapping over seeds
helps at 64^2 where per-call overhead dominates, not at 256^2 where the FFTs already saturate 2 cores. The first
call of a plain `stepper(params)` also compiles (~1 s); time it separately from the loop.

## Limitations

* Penalisation is first order at the boundary: the mask is a staircase on the grid, so the effective body differs
  from the intended one by O(dx) (the test cylinder of nominal D = 1 has mask extent D = 0.98 at n = 64 and 1.08 at
  n = 128), and the Lie splitting of penalisation and RK3 is first order in dt. The spectral curl of the masked
  velocity rings (Gibbs) at the boundary and leaves a small velocity inside the body (measured above).
* Periodic box: the domain is an infinite periodic array of bodies (blockage D/L) and the wake re-enters from the
  left unless the sponge removes it; the sponge damps vorticity, not the full velocity defect, and is not perfectly
  non-reflecting. Drag coefficients are therefore those of a confined, periodic array, not an isolated cylinder.
* The a-priori CFL estimate in `validate` is a heuristic for turbulent and obstacle flows; check
  `diagnostics()["cfl"]` or the `rollout` events for long runs.
* float32: accumulated round-off sets a floor of ~1e-4 (max-abs, |w| ~ 4) after ~500 steps; temporal-convergence
  studies must use dt large enough to stay above it. The residual has a matching floor ~ eps / (dt * rate).
* No subgrid model: at high Re the 2/3 truncation is the only small-scale sink besides viscosity, so under-resolved
  runs pile up enstrophy at the cutoff (watch `max_abs_vorticity`). `nu` must be > 0.
* Square grid only (H = W = n), one scalar field, time-independent forcing.

## Beta-plane and vortices (fluid2d)

`beta` adds `-beta v` to the vorticity equation (Coriolis parameter f = f0 + beta y, +y = north). In Fourier space it
is the linear term `i beta k_x / |k|^2 w_hat`, folded into the integrating factor, so it is exact and adds no time-step
limit. A constant f0 (f-plane) has no effect on 2D incompressible flow (its curl is f0 div u = 0), so it is not a
parameter. `ic = "vortices"` with `vortices = [(x, y, R, w0), ...]` places Gaussian vortices `w0 exp(-r^2/R^2)`
(periodic minimal image, mean removed).

Measured (tests/test_fluid2d.py):
* Rossby wave cos(2x + y), beta = 2, t = 5: phase moved -4.00 rad (sigma = -beta k_x / |k|^2, westward); max error
  1.4e-5 against the exact damped wave.
* Cyclone R = 0.5, w0 = 5, n = 64, t = 4: the vorticity peak stays put with beta = 0 and moves by (-1.08, +0.79) with
  beta = 2, i.e. north-west, the beta-drift direction of a northern-hemisphere cyclone.
* Co-rotating pair R = 0.4, w0 = 5, nu = 1e-3: separation 1.2 merges (peak count 2 -> 1 between t = 15 and 20);
  separation 2.0 stays two vortices to t = 30. The merger threshold was not scanned finer.

## Spec-driven PDEs (kolmo/systems/pde.py)

`make(spec)` or `make("<library name>")`. A spec gives `fields`, one `rhs` expression per field, `params`, `n`, `L`,
`dt` and `ic`. It may also give `stabilize` ({field: S}), `energy` (a density in the same grammar, whose box mean is
reported as `energy`) and `checks`.

Grammar: fields, params, `pi`, numbers, `+ - * / **`, unary minus, and `lap dx dy grad2 exp tanh sin cos`. Expressions
are parsed by `ast` against a whitelist and never evaluated as Python. Unknown names, functions or syntax raise
`ValueError` naming them.

* Linear part: the Fourier symbol of d(rhs_i)/d(u_i) at u = 0, obtained with `jax.jvp` on a delta function. The grammar
  has no x-dependence, so this Jacobian is translation-invariant and diagonal in Fourier space. `stabilize` moves
  `S lap u` from the explicit to the exact part. Cahn-Hilliard uses S = 1.5, which balances the explicit Jacobian
  between u = 0 and u = +-1. The remainder N = rhs - L u is explicit, and cross-field linear terms stay explicit.
* Time stepping: ETDRK4 (Cox & Matthews, J. Comput. Phys. 176, 2002). Its coefficients come from the complex contour
  mean of Kassam & Trefethen (2005), with 32 points on a unit circle, which avoids cancellation in the phi-functions at
  small |L dt|. The scheme is fourth order and keeps the ODE's fixed points for any dt. The 2/3 rule dealiases N and
  the state. This is not alias-free for cubic terms, but it is standard practice.
* Checks in `validate`, cached per parameter set:
  * linear growth resolved: `max Re(L) * dt <= 2`;
  * explicit stiffness: a 30-step power iteration on the Jacobian of N at the seed-0 initial state must give
    `|J_N| * dt <= 2` (the RK4 bounds are 2.78 on the real axis and 2.83 on the imaginary axis).

  Errors quote the admissible dt. The stiffness estimate holds for the initial state only. For Cahn-Hilliard, the
  choice of S also covers the later u = +-1 state.
* `residual`: per Fourier mode, the linear part is exact, and N, taken as linear in time, is integrated exactly against
  exp(L(dt - s)): `R = (v1 - E v0)/dt - [phi1 N0 + phi2 (N1 - N0)]`. It is normalised by the size of all terms. A plain
  trapezoid in the integrating-factor variable gave 4.9e-2 for Swift-Hohenberg, because it is inaccurate for slaved
  high modes (|L dt| >> 1); the phi-form gives 1.5e-6.
* IC strings, joined by `+`, either one string or {field: string}:
  * `const:c`;
  * `noise:A`: white noise in the 2/3 band, std A, seeded by (seed, field);
  * `bump:A:r`: Gaussian at the centre;
  * `sin:A:m`: A sin(2 pi m x / L);
  * `zero`.

Library (defaults in `LIBRARY`):

| name | equation |
|---|---|
| `cahn_hilliard` | u_t = lap(u^3 - u) - gamma lap^2 u |
| `allen_cahn` | u_t = eps2 lap u + u - u^3 |
| `gray_scott` | two fields, Pearson-type F = 0.04, k = 0.06 |
| `kuramoto_sivashinsky` | u_t = -lap u - lap^2 u - \|grad u\|^2 / 2 (in 2D the mean drifts down by <\|grad u\|^2>/2) |
| `swift_hohenberg` | u_t = r u - (1 + lap)^2 u - u^3 |
| `burgers` | u_t = nu lap u - u (u_x + u_y) |

Measured (tests/test_pde.py, `python -m pytest tests/test_pde.py -q -s`):

| check | result |
|---|---|
| Cahn-Hilliard, 1000 steps (t = 30) | max change of mean(u) 2.5e-7 (round-off); free energy 0.3052 -> 0.1957 and never rises; residual 3.4e-6 |
| KS linear growth, mode k = 0.5, amplitude 1e-4, t = 10 | measured 0.187500 vs k^2 - k^4 = 0.187500 |
| ETDRK4 order (Allen-Cahn, n = 32, T = 4) | errors 6.0e-2 / 6.35e-3 / 3.73e-4 at dt 0.4 / 0.2 / 0.1; fitted order 3.66 (last pair 4.1) |
| Gray-Scott positivity, 2000 steps | min(u, v) = 1.1e-13 >= 0 |
| Allen-Cahn / Swift-Hohenberg / Burgers | energy never rises (largest step change -1e-15 / -1.4e-5 / -5.7e-3); Allen-Cahn stays in [-1, 1]; Burgers mean drift 1.5e-8 |
| stiffness refusal | Cahn-Hilliard at dt = 0.1 and KS at dt = 10 are rejected with the admissible dt |

KS and Burgers at these sizes cannot be used for an order test: their ETDRK4 error is already at the float32 floor
(~3e-5) for dt <= 0.2. The library defaults use modest resolution: the Cahn-Hilliard interface spans ~3 cells and
Gray-Scott fronts ~2 cells at n = 64. Raise n for quantitative work.

### Integrator choice (literature check, Nov 2026)

* ETDRK4 with contour-integral coefficients is still the standard method for stiff semilinear PDEs on periodic domains
  (Kassam & Trefethen, "Fourth-order time-stepping for stiff PDEs", SIAM J. Sci. Comput. 26, 2005;
  people.maths.ox.ac.uk/trefethen/fourth-order.pdf).
* The JAX library Exponax (Koehler et al.; github.com/Ceyron/exponax, fkoehler.site/exponax) is built on Fourier
  pseudo-spectral ETDRK for the same reason.
* A 2025 survey of exponential Runge-Kutta methods (arXiv:2507.04024) gives no reason to change.
* fluid2d and boussinesq keep IF-RK3. Their stiff part is plain diffusion, which IF integrates exactly, and their
  explicit part is advection, where RK3's imaginary-axis stability matters more than stiff accuracy.

## Boussinesq convection (kolmo/systems/boussinesq.py)

    dw/dt + u . grad w = sqrt(Pr/Ra) lap w + d(theta)/dx
    dtheta/dt + u . grad theta = lap(theta) / sqrt(Ra Pr),     lap psi = -w

* Units: free-fall units with length H, temperature dT and velocity sqrt(g alpha dT H). Gravity points along -y and
  buoyancy is +theta e_y, whose curl is d(theta)/dx. The mean temperature drives no mean flow, because the
  hydrostatic part drops out of the curl.
* Numerics: as in fluid2d (2/3 rule, IF-RK3 with both diffusions exact). The CFL limit uses a free-fall speed estimate
  of 2 sqrt(dT L/4).
* Obstacle: an optional `mask` is penalised to zero velocity and temperature `T_body` (split substep, first order).
* ICs:
  * `thermal`: a Gaussian warm blob at (L/2, L/4).
  * `rayleigh_taylor`: theta = A tanh(sin(2 pi (y + eps cos(2 pi m x/L))/L) L/(2 pi delta)), warm below cold. It is
    unstable at y = L/2 and stable at y = 0.
  * `zero`.
* Diagnostics: energy, enstrophy, buoyancy_flux <v theta>, theta_mean, theta_var = 0.5<theta^2>,
  thermal_dissipation kappa<|grad theta|^2>, max_abs_vorticity, cfl, centroid_y.
* Budgets: `dE/dt = <v theta> - 2 nu Z` and `d(0.5<theta^2>)/dt = -kappa <|grad theta|^2>`.

Measured (tests/test_boussinesq.py, Ra = 1e5, Pr = 1, n = 64, L = 2 pi, dt = 0.019):

| check | result |
|---|---|
| budgets (central difference, 60 steps) | kinetic 3.8e-5, thermal variance 4.5e-4 (relative); residual 5.3e-5 |
| rising thermal, t = 4 | centroid height 1.571 -> 1.683 -> 1.999 -> 2.445 -> 2.909 (monotone); x-centroid stays at L/2 to 2.7e-4; mean theta unchanged |
| Rayleigh-Taylor, eps = 0.05, m = 1 | kinetic energy 7.8e-5 -> 5.9e-2 over t = 3.5; energy growth rate 1.7-1.9 per time unit (exponential phase) |
| heated disc (T_body = 1), t = 4 | theta 0.116 above the body vs 0.003 below, 0.997 inside; max speed inside 0.038 vs 0.618 outside |

Limitations:
* Ra = 1e5 at n = 64 is near the resolution limit; raise n for higher Ra.
* There are no top or bottom walls and the periodic box has no imposed mean gradient, so there is no Rayleigh-Benard
  Nusselt validation.
* The RT growth rate is not compared with a theory value (diffuse interface, viscous).
* The obstacle shares fluid2d's first-order penalisation limits.

### All physics tests

`python -m pytest tests/test_boussinesq.py tests/test_pde.py tests/test_fluid2d.py -q -s` gave `22 passed in 91.93s`.
