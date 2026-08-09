# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`fenicsx-beat` (`beat` package) — cardiac electrophysiology simulator on FEniCSx/`dolfinx`. Solves the
Monodomain model (reaction-diffusion PDE + cell-model ODEs) via operator splitting. Requires a working
`dolfinx` install (container/conda); not pip-installable standalone. No Makefile — use `pip`/`pytest`/
`pre-commit` directly.

## Commands

```bash
pip install -e ".[test]"                                    # editable install + test deps
pytest -v                                                    # full suite
pytest tests/test_monodomain.py::test_name -v                # single test
pytest --cov=beat --cov-report=html --cov-report=term -v     # with coverage (as CI does)
mpirun -n 2 pytest -v -m "not skip_in_parallel"               # MPI subset (as main-mpi.yml does)
pre-commit run --all-files                                   # ruff, ruff-format, mypy, cspell, etc.
```

- Tests that can't run under multi-rank MPI: mark `@pytest.mark.skip_in_parallel` (see
  `tests/test_monodomain.py`).
- CI dolfinx matrix: `stable` + `nightly` in `main.yml`, `stable` only in `main-mpi.yml`.

### `irksome` extra — install gotcha

`.[irksome]` (pulled by `.[test]`/`.[docs]`) installs `irksome[dolfinx]` from PyPI, but **that release
is broken**: it ignores `backend="dolfinx"` and imports `firedrake` anyway. Fix (what CI does, after
`pip install .[test]`):
```bash
pip install "irksome[dolfinx] @ git+https://github.com/firedrakeproject/Irksome.git"
```
Skipping this doesn't fail loudly — `tests/test_irksome_*.py` use `pytest.importorskip("irksome")`, so
they just silently skip.

## Architecture

Operator-split PDE/ODE solve per timestep:

- **PDE** (`base_model.py`, `monodomain_model.py`): `BaseModel` assembles a
  `dolfinx.fem.petsc.LinearProblem` from `variational_forms(dt)` (theta-scheme). `MonodomainModel`
  implements `Cm*dv/dt - div(M*grad(v)) - I_stim = 0` with conductivity `M` and a `Stimulus`
  (`stimulation.py`).
- **ODE** (`odesolver.py`): `ODESystemSolver` steps a cell-model RHS (`fun(states, t, parameters, dt)`,
  usually `gotranx`-generated from `odes/*.ode`/CellML). `BaseDolfinODESolver` /
  `DolfinODESolver` / `DolfinMultiODESolver` (per-region) bridge the numpy state array to/from dolfinx
  `Function`s via `local_project`.
- **Coupling** (`monodomain_solver.py`): `MonodomainSplittingSolver(pde, ode)` drives Godunov/Strang
  splitting: ODE step → project to PDE → PDE step → project back → optional corrective ODE step if
  `theta != 1.0`.
- **Irksome backend** (`irksome_model.py`, `irksome_odesolver.py`, `irksome` extra):
  `IrksomeMonodomainModel` / `IrksomeODESolver` / `IrksomeMultiODESolver` are drop-in alternatives that
  advance the PDE/ODE step with a fully implicit Runge–Kutta Butcher tableau instead of theta-rule/
  explicit; freely mixable with the default classes in `MonodomainSplittingSolver`. See install
  gotcha above.
- **Conductivities/units** (`conductivities.py`, `units.py`): `pint` quantities → `M` via
  `define_conductivity_tensor`; `default_conductivities` has literature sets ("Niederer", "Bishop").
- **Geometry** (`geometry.py`): `Geometry` NamedTuple + 2D/3D slab helpers; real ventricular geometries
  come from `cardiac-geometriesx`/`ukb-atlas`/`fenicsx-ldrb` (`demos`/`docs` extras).
- **ECG** (`ecg.py`): `ECGRecovery` + R-peak/T-wave detection from simulated extracellular potentials.
- **Single-cell** (`single_cell.py`): standalone ODE-only cell-model solving (numba-jitted if
  available), for pacing protocols independent of tissue PDEs.
- **Telemetry** (`telemetry.py`): `BaseMonitor`/`NullMonitor`/`PerformanceMonitor`, threaded through
  `step()` via `monitor.track_time(...)`.
- **CLI** (`cli.py`, entry point `beat`): `run`/`ecg`/`post` subcommands are stubs (`NotImplemented`).

Dolfinx API compat: `base_model.py`/`odesolver.py`/`utils.py`/`ecg.py` branch on
`packaging.version.Version(dolfinx.__version__)` — check for an existing `_dolfinx_version >=
Version(...)` branch before assuming one API shape.

## Demos and docs

- `demos/*.py` are jupytext "light format" scripts (`# # Title` / `# ...` → markdown cells, `# +`/`# -`
  bracket code cells) executed directly as notebooks by Jupyter Book. **Do not** wrap demo logic in
  `def main(): ...` / `if __name__ == "__main__"` — it silently never runs when built into the docs.
- `docs/math_background.md` is the canonical notation reference ($v$, $M$, $\chi$, $C_m$, $I_{ion}$,
  $I_{stim}$, $\theta$, matching `beat` API arg names) — keep new demos/docs consistent with it rather
  than inventing new symbols.
- `demos/index.md` is a flat, grouped index of every demo. When adding a demo, add it to both this and
  `_toc.yml`.
- `odes/` holds third-party CellML/`.ode` cell-model sources (not part of the `beat` package).
