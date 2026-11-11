# Code structure

This page lists the classes in `beat`, what each one is for and how they fit together. The notation
follows the [mathematical background](math_background.md).

## The big picture

A tissue simulation in `beat` is three objects:

- a **PDE model** that advances the transmembrane potential $v$ by diffusion (e.g. `MonodomainModel`),
- an **ODE solver** that advances the cell model at every point (e.g. `DolfinODESolver`), and
- a **coupling solver**, `SplittingSolver`, that alternates between the two by operator
  splitting.

Everything else plugs in around these: an eikonal solve gives activation times that drive the cell
model through a foot current (the reaction-eikonal model), `ActivationTracker` records activation
times from $v$, and electrode recorders compute extracellular potentials from $v$.

```text
         solve_eikonal / EikonalSolver ──► t_a (optional: reaction-eikonal)
                                            │
  ODE solver (DolfinODESolver, ...) ──► FootCurrentODESolver(ode, t_a)   (still an ODE solver)
         │ v_ode ◄── pde_to_ode / ode_to_pde ──► v_pde  ══ same Function ══  pde.v
         │                                                                   │
         └──────────────── SplittingSolver(pde or None, ode) ────────────────┘
                                         │ solver.v
            ┌────────────────────────────┼─────────────────────────────┐
            ▼                            ▼                             ▼
     ActivationTracker          ElectrodeRecorder               PointEvaluator /
     (activation times)   (ElectrodePotentials,          scifem.evaluate_function
                           LeadFieldElectrodes)               (probes)
```

## Protocols

The coupling solver and the electrogram tools only depend on small interfaces (Python
`typing.Protocol`s). Any class with these members works, including your own.

- `PDEModel` (`beat.monodomain_solver`): a `v` property, `assign_previous()` and
  `step((t0, t1))`. Implemented by `MonodomainModel`, `BidomainModel` and `IrksomeMonodomainModel`.
- `ODESolver` (`beat.monodomain_solver`): `v_ode`, `v_pde`, `to_dolfin()`, `from_dolfin()`,
  `ode_to_pde()`, `pde_to_ode()`, `step(t0, dt)`, `states_to_dolfin(names)` and
  `load_all_states(functions)`. `ode.v_pde` must be the same Function as `pde.v`
  (pass `v_pde=pde.state`). Implemented by every `BaseDolfinODESolver` subclass and by
  `FootCurrentODESolver`.
- `ElectrodeRecorder` (`beat.ecg`): calling it returns `{electrode name: potential}`, the same on
  every rank, ready for `LeadSystem.leads`. Implemented by `ElectrodePotentials` and
  `LeadFieldElectrodes`. The two use different conductivity conventions: `ECGRecovery` takes the
  monodomain `M`, the pseudo-bidomain classes the unscaled bidomain tensors.

## Common wirings

Monodomain (see the [FitzHugh–Nagumo demo](../demos/fitzhughnagumo.py)):

```python
pde = beat.MonodomainModel(time=time, mesh=mesh, M=M, I_s=I_s, C_m=C_m)
ode = beat.odesolver.DolfinODESolver(v_ode=dolfinx.fem.Function(V_ode), v_pde=pde.state, **cell)
# cell: fun, init_states, parameters, num_states, v_index (see the demo)
solver = beat.SplittingSolver(pde=pde, ode=ode)
solver.solve((0.0, T), dt)
```

Reaction-eikonal with diffusion, R-E$^+$ (a PDE model without stimulus):

```python
t_a = beat.eikonal.solve_eikonal(mesh, velocities, (f0, s0, n0), sources)
ode = FootCurrentODESolver(cell_solver(v_pde=pde.state), t_a, ExponentialFoot())
solver = beat.SplittingSolver(pde=pde, ode=ode)
```

Reaction-eikonal without diffusion, R-E$^-$ (no PDE; `v_pde` is not used):

```python
ode = FootCurrentODESolver(cell_solver(v_pde=dolfinx.fem.Function(V)), t_a, ExponentialFoot())
solver = beat.SplittingSolver(pde=None, ode=ode)
```

Electrograms in a bath (see the [wedge electrogram demo](../demos/reaction_eikonal_wedge_electrograms.py)):

```python
bath = beat.geometry.get_3D_wedge_bath_geometry(comm, dx)
conductor = VolumeConductor(
    bath.geometry.mesh, bath.cell_tags, TISSUE, BATH, sigma_i_b, sigma_e_b, sigma_b
)
record = LeadFieldElectrodes(conductor, solver.v, sigma_i, {"P_E": wedge.electrode})
solver.step((t, t + dt))
phi_e = record()["P_E"]
```

## Classes by role

### PDE models

| Class | What it is for | Builds on / implements | Demos |
|---|---|---|---|
| `BaseModel` | theta-scheme skeleton: assembles and solves one linear system per step, restart metadata, telemetry | — | (base class) |
| `MonodomainModel` | the monodomain equation | `BaseModel`, `PDEModel` | most demos, e.g. [fitzhughnagumo](../demos/fitzhughnagumo.py) |
| `BidomainModel` | the bidomain equations, $v$ and $u_e$ | `BaseModel`, `PDEModel` | [bidomain_convergence](../demos/bidomain_convergence.py) |
| `IrksomeMonodomainModel` | monodomain advanced by an implicit Runge–Kutta method (`irksome` extra) | `MonodomainModel`, `PDEModel` | [irksome_model_gotranx](../demos/irksome_model_gotranx.py) |
| `BidomainState` | `(v, u_e)` of a `BidomainModel` | `NamedTuple` | — |
| `Results`, `Status` | return value of `BaseModel.solve` | `NamedTuple`, `Enum` | — |

### ODE solvers (cell models)

| Class | What it is for | Builds on / implements | Demos |
|---|---|---|---|
| `BaseDolfinODESolver` | cell states on a dolfinx space; projection between `v_ode` and `v_pde` | `ODESolver` | (base class) |
| `DolfinODESolver` | one cell model everywhere, NumPy states, `fun(states, t, parameters, dt)` | `BaseDolfinODESolver` | most demos, e.g. [niederer_benchmark](../demos/niederer_benchmark.py) |
| `DolfinMultiODESolver` | one cell model per region marker | `BaseDolfinODESolver` | [lv_endocardial](../demos/lv_endocardial.py), [slab](../demos/slab.py) |
| `ODESystemSolver` | the NumPy stepper inside `DolfinODESolver` | — | — |
| `IrksomeODESolver`, `IrksomeMultiODESolver` | cell model advanced by an implicit Runge–Kutta method (`irksome` extra) | `BaseDolfinODESolver` | [irksome_model_gotranx](../demos/irksome_model_gotranx.py) |
| `ExternalOperatorODESolver`, `ExternalOperatorMultiODESolver` | cell model evaluated through `dolfinx-external-operator` | `BaseDolfinODESolver` | [external_operator_gotranx](../demos/external_operator_gotranx.py) |
| `CScheme` | a compiled C cell model, usable as `fun` (from `beat.c_backend.compile_scheme` / `from_ode`) | callable `fun` | [c_backend](../demos/c_backend.py) |
| `ODEResults` | unused | `NamedTuple` | — |

### Coupling

| Class | What it is for | Builds on / implements | Demos |
|---|---|---|---|
| `SplittingSolver` | Godunov/Strang splitting of a `PDEModel` and an `ODESolver`, or the ODE solver alone (`pde=None`); restart | uses `PDEModel`, `ODESolver` | almost every demo |
| `MonodomainSplittingSolver` | deprecated name of `SplittingSolver`; warns | `SplittingSolver` | — |
| `PDEModel`, `ODESolver` | the protocols above | `Protocol` | — |

### Eikonal and reaction-eikonal

| Class | What it is for | Builds on / implements | Demos |
|---|---|---|---|
| `EikonalSolver` | activation times by the Fast Iterative Method; set up once, solve for many sources (`solve_eikonal` is the one-call form) | C kernel | [eikonal](../demos/eikonal.py) |
| `Source` | an eikonal source: where, and when | dataclass | [eikonal](../demos/eikonal.py), the wedge demos |
| `ExponentialFoot` | parameters of the foot current | dataclass | the wedge demos |
| `FootCurrentODESolver` | wraps any ODE solver and adds the foot current at $t_a$ | `ODESolver` | [wedge activation](../demos/reaction_eikonal_wedge_activation.py), [repolarization](../demos/reaction_eikonal_wedge_repolarization.py), [electrograms](../demos/reaction_eikonal_wedge_electrograms.py) |

### Activation and probes

| Class | What it is for | Builds on / implements | Demos |
|---|---|---|---|
| `ActivationTracker` | per-dof activation time from $v$ (max $dv/dt$ or threshold crossing), updated each step | — | [wedge activation](../demos/reaction_eikonal_wedge_activation.py) |
| `PointEvaluator` | evaluates Functions at fixed points, locating them once | — | [wedge activation](../demos/reaction_eikonal_wedge_activation.py), [electrograms](../demos/reaction_eikonal_wedge_electrograms.py) |

### Electrograms and ECG

| Class | What it is for | Builds on / implements | Demos |
|---|---|---|---|
| `ECGRecovery` | pseudo-ECG in an infinite homogeneous medium | — | [slab](../demos/slab.py), [biv_endocardial](../demos/biv_endocardial.py) |
| `ElectrodePotentials` | `ECGRecovery` at many named electrodes, one solve per call | `ElectrodeRecorder` | [biv_endocardial](../demos/biv_endocardial.py) |
| `VolumeConductor` | extracellular potential $\phi_e$ in a bounded tissue-and-bath volume (pseudo-bidomain) | — | [electrograms](../demos/reaction_eikonal_wedge_electrograms.py) |
| `LeadFieldElectrodes` | $\phi_e$ at named electrodes via lead fields (reciprocity), cheap per step | `ElectrodeRecorder` | [electrograms](../demos/reaction_eikonal_wedge_electrograms.py) |
| `ElectrodeRecorder` | the protocol above | `Protocol` | — |
| `Lead`, `LeadSystem` | leads as weighted sums of electrode potentials (`twelve_lead()` builds the standard 12) | dataclasses | [biv_endocardial](../demos/biv_endocardial.py) |
| `QTIntervalResult` | return value of `qt_interval` | `NamedTuple` | — |

### Geometry, stimulation and conductivities

| Class | What it is for | Builds on / implements | Demos |
|---|---|---|---|
| `Geometry` | mesh, facet tags, markers and fibre fields `f0, s0, n0` | `NamedTuple` | [slab](../demos/slab.py) |
| `WedgeGeometry` | the Neic 2017 transmural wedge: a `Geometry`, cell tags, stimulus sites, electrode | wraps `Geometry` | the wedge demos, [eikonal](../demos/eikonal.py) |
| `WedgeBathGeometry` | the wedge inside a bath box: a `Geometry`, tissue/bath cell tags, margins | wraps `Geometry` | [electrograms](../demos/reaction_eikonal_wedge_electrograms.py) |
| `Stimulus` | a stimulus current with its measure (`define_stimulus` builds one) | `NamedTuple` | most demos |
| `Conductivities` | longitudinal and transverse conductivity (`get_harmonic_mean_conductivity`) | `NamedTuple` | — |

### Telemetry and single cell

| Class | What it is for | Builds on / implements | Demos |
|---|---|---|---|
| `BaseMonitor` | timing interface threaded through `step()` | ABC | — |
| `NullMonitor` | the default: records nothing | `BaseMonitor` | — |
| `PerformanceMonitor` | timings and Krylov statistics, with a summary | `BaseMonitor` | [c_backend](../demos/c_backend.py), [niederer_benchmark](../demos/niederer_benchmark.py) |
| `NumbaWarning`, `PlottingWarning` | warnings from `beat.single_cell` (single-cell pacing to steady state) | `UserWarning` | — |

The `beat` command-line interface (`beat.cli`) builds these same objects from a TOML file; see
[the CLI guide](cli.md).

When you add a public class, add it to this page: `tests/test_architecture_docs.py` checks that
every public class in `src/beat` is named here.
