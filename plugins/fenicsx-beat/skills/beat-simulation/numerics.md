# Numerics: scheme, splitting, time step, mesh, backends

Background and evidence: https://finsberg.github.io/fenicsx-beat/docs/math_background.html
("Choice of cell scheme"), the CLI guide https://finsberg.github.io/fenicsx-beat/docs/cli.html
("Cell schemes and splitting") and the
[verification demo](https://finsberg.github.io/fenicsx-beat/demos/verification.html).

## Defaults (beat 0.14 and later)

| Setting | CLI | Python |
|---|---|---|
| Cell scheme (gotranx) | `cell.scheme = "rush_larsen_2"` | the scheme you generate the model code with, e.g. `gotranx.schemes.Scheme.rush_larsen_2`; `beat.c_backend.from_ode` defaults to `rush_larsen_2` |
| Splitting | `solver.splitting = "strang"` | `beat.SplittingSolver(pde, ode)` is Strang |
| Merged half-steps | `solver.merge_half_steps` defaults to true for Strang | off; `merge_half_steps=True` opts in, then call `solver.synchronize()` before reading or saving the state |
| PDE time stepping | `solver.pde.theta = 0.5` (Crank–Nicolson) | `theta` 0.5 in the PDE model's parameters |
| Linear solver | `solver.pde.linear_solver = "direct"` (LU, MUMPS) | direct |

Why these defaults: `rush_larsen_2` is second order, costs about what `generalized_rush_larsen` costs,
was clearly more accurate in beat's tests at typical time steps, and steps the non-gating states with an
explicit Runge–Kutta step, which preserves linear invariants of those states. Strang splitting with a
second-order cell scheme and theta 0.5 is second order in time; merging the half-steps keeps second
order at the cost of one ODE step per time step.

## When to deviate

- **Non-finite values or oscillations at the wanted dt** → `rush_larsen_2` steps $v$ explicitly. Halve
  `solver.dt`, or switch to `cell.scheme = "generalized_rush_larsen"` (exponential update for every
  state, more robust) with `solver.splitting = "godunov"` (first order, cheapest).
- **Reading $v$ after every step** (activation tracking, probes every step, in Python) → plain Strang
  (no merging): with merging, the state between synchronizations is half a step off.
- **Reproducing results from beat 0.13 or earlier** → `cell.scheme = "generalized_rush_larsen"`,
  `solver.splitting = "godunov"` (the old defaults).
- **Stiff states** (a few very fast gates) → a `hybrid_rush_larsen` scheme with `cell.stiff_states`.
- **First-order scheme with Strang** → the CLI warns: Strang is only second order with a second-order
  cell scheme and theta 0.5.

## Time step

- Physiological models (TP06, ToR-ORd): `solver.dt` 0.01–0.05 ms is typical; the templates use 0.05 ms.
  FitzHugh–Nagumo / Mitchell–Schaeffer tolerate larger steps (0.1 ms), but check.
- Check convergence by halving dt on a short run and comparing activation times or $v$ traces.
- Stimulus duration and start should be multiples of dt (or several dt long).

## Mesh size

- The upstroke front is well under 1 mm wide in tissue. A mesh that does not resolve it gives a wave
  that is too slow; far too coarse and the wave does not propagate at all.
- Physiological models with literature conductivities: about 0.1–0.25 mm for converged conduction
  velocity (the Niederer benchmark compares 0.1, 0.2 and 0.5 mm). Whole hearts at that resolution
  need clusters.
- Coarser meshes (0.5–1 mm and above): calibrate the conductivities so the wave has the wanted speed
  on that mesh (`beat.conduction_velocity.calibrate_conductivities`), or use reaction-eikonal, which
  is built for coarse meshes.
- Always check propagation in a smoke run, and say which mesh size the result is valid for.

## Conductivities and units

- Presets: `ep.conductivity = { preset = "Niederer" }` (or `"Bishop"`, `"Neic2017_wedge"`, … — the
  names `beat.conductivities.default_conductivities` accepts), or explicit intra-/extracellular
  longitudinal and transverse values (`sigma_il`, `sigma_it`, `sigma_el`, `sigma_et`), combined into
  the monodomain $M$ by harmonic mean.
- $\chi$ (`ep.chi`, e.g. `"1400 cm**-1"`) and $C_m$ (`ep.C_m`, e.g. `"1 uF/cm**2"`) scale the PDE;
  stimulus amplitudes are current densities (`"uA/cm**2"`). Keep the mesh unit (`geometry.unit`)
  consistent.

## Backends for the cell-model step

| Backend | When |
|---|---|
| NumPy (gotranx Python code), default | everything; simplest |
| Numba | single-cell runs (`beat.single_cell` uses it when available) |
| Compiled C (`beat.c_backend.from_ode`), optional OpenMP | large meshes where the ODE step dominates |
| Irksome (`beat.IrksomeODESolver`, `beat.IrksomeMonodomainModel`) | fully implicit Runge–Kutta, stiff problems, studies of time stepping |
| dolfinx-external-operator (`beat.ExternalOperatorODESolver`) | the cell step expressed as a UFL external operator |

## MPI and performance

- `mpirun -n N beat run config.toml` or `mpirun -n N python script.py`; results do not depend on N
  beyond round-off (chaotic problems such as spiral breakup can still diverge).
- Generate a gmsh-based geometry once (`beat geometry config.toml`, serially) before a parallel run.
- `beat post` can run on any number of ranks, independent of the run.
- `output.performance = true` writes a timing summary (`performance.json`).
- Large 3D runs: `solver.pde.linear_solver = "iterative"` (CG + algebraic multigrid) scales better
  than the direct solver.
