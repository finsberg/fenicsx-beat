# # Cell-model stepping via dolfinx-external-operator
#
# This demo introduces `beat.ExternalOperatorODESolver`, an alternative to
# `beat.odesolver.DolfinODESolver` built on
# [`dolfinx-external-operator`](https://github.com/a-latyshev/dolfinx-external-operator)'s
# `FEMExternalOperator`, which lets a cell-model right-hand side be evaluated as a plain array
# operation and spliced back into the dolfinx `Function` machinery. See the
# [mathematical background](../docs/math_background.md) page (Section 7.2) for how it fits into the
# monodomain model and `beat.MonodomainSplittingSolver`.
#
# The headline feature is that it is a genuine drop-in replacement: it calls your existing cell-model
# function with the *exact same* `fun(states, t, parameters, dt)` convention already used by
# `DolfinODESolver` — no UFL-returning variant is needed (unlike the
# [Irksome backend](irksome_model_gotranx.py)). This demo makes that concrete by running the same
# monodomain simulation, with the same `gotranx`-generated ten Tusscher–Panfilov cell model, through
# both solvers, and checking the results match exactly. It also honestly reports the current
# performance trade-off — being built on `dolfinx-external-operator`'s general-purpose UFL machinery
# has a real per-step overhead compared to `DolfinODESolver`'s plain NumPy loop, so this is not (yet) a
# speed optimization. The reason to reach for it instead is forward-looking: the same
# `FEMExternalOperator` machinery is what would let a future cell model supply its own Jacobian and be
# coupled *implicitly* with the PDE in a single monolithic Newton solve, with no operator-splitting
# error at all — something `DolfinODESolver`'s plain array update has no path towards.

from pathlib import Path
import time as pytime

from mpi4py import MPI
import dolfinx
import numpy as np
import gotranx
import numba
import ufl

import beat
from beat.external_operator_odesolver import ExternalOperatorODESolver

# ## The cell model
#
# As in the [Niederer benchmark](niederer_benchmark.py) and other tissue demos, we generate the ten
# Tusscher–Panfilov 2006 epicardial cell model with `gotranx`, using a generalized Rush–Larsen scheme.
# Since `ExternalOperatorODESolver` calls the generated function exactly like `DolfinODESolver` does,
# no special code-generation option is needed here.

model_path = Path("tentusscher_panfilov_2006_epi_cell.py")
if not model_path.is_file():
    here = Path.cwd()
    ode = gotranx.load_ode(
        here
        / ".."
        / "odes"
        / "tentusscher_panfilov_2006"
        / "tentusscher_panfilov_2006_epi_cell.ode",
    )
    code = gotranx.cli.gotran2py.get_code(
        ode,
        scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
    )
    model_path.write_text(code)

import tentusscher_panfilov_2006_epi_cell as model

fun = model.generalized_rush_larsen
v_index = model.state_index("V")

# ## Geometry, PDE and stimulus
#
# A small 2D slab with a scalar conductivity and a stimulus in one corner — the same style of setup as
# the [diffusion](diffusion.py) and [FitzHugh–Nagumo](fitzhughnagumo.py) demos, just with a realistic
# cell model in place of a toy one.

comm = MPI.COMM_WORLD
N = 25
mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.mesh.CellType.triangle)
time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))

tol = 1.0e-10
L = 0.2


def S1_subdomain(x):
    return np.logical_and(x[0] <= L + tol, x[1] <= L + tol)


S1_marker = 1
tdim = mesh.topology.dim
facets = dolfinx.mesh.locate_entities(mesh, tdim, S1_subdomain)
facet_tags = dolfinx.mesh.meshtags(
    mesh,
    tdim,
    facets,
    np.full(len(facets), S1_marker, dtype=np.int32),
)
dx = ufl.dx(domain=mesh, subdomain_data=facet_tags)
stim_amp = dolfinx.fem.Constant(mesh, 50.0)
stim_expr = ufl.conditional(ufl.le(time, 2.0), stim_amp, 0.0)
I_s = beat.Stimulus(expr=stim_expr, dZ=dx, marker=S1_marker)

M = 0.1  # scalar conductivity, chosen for a visible wave on a unit square

# ## Initial conditions
#
# For a short demo we start every point at the model's default resting state, rather than pacing to a
# limit cycle as in the [Niederer benchmark](niederer_benchmark.py) — this keeps the runtime small
# while still exercising a real, 19-state cell model.

V_ode = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
num_states = len(model.init_state_values())
N_ode = V_ode.dofmap.index_map.size_local + V_ode.dofmap.index_map.num_ghosts
init_states = np.zeros((num_states, N_ode))
init_states.T[:] = model.init_state_values()
parameters = model.init_parameter_values(stim_amplitude=0.0)

# ## Running the simulation with ExternalOperatorODESolver
#
# Everything here — `beat.MonodomainModel`, `beat.MonodomainSplittingSolver` — is exactly what the
# other demos use; only the ODE solver class changes.

pde = beat.MonodomainModel(time=time, mesh=mesh, M=M, I_s=I_s)
ode = ExternalOperatorODESolver(
    v_ode=dolfinx.fem.Function(V_ode),
    v_pde=pde.state,
    fun=fun,
    init_states=init_states,
    parameters=parameters,
    num_states=num_states,
    v_index=v_index,
)
solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)

T = 5.0
dt = 0.05
t = 0.0
while t < T + 1e-12:
    solver.step((t, t + dt))
    t += dt

# +
try:
    import pyvista
except ImportError:
    pyvista = None
else:
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(pde.state.function_space))
    grid.point_data["V"] = pde.state.x.array.real
    grid.set_active_scalars("V")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot("external_operator_gotranx.png")
# -

# ## Checking against DolfinODESolver
#
# To confirm this is a genuine drop-in — same PDE, same `gotranx`-generated cell model, unmodified —
# we repeat a short run with the default `DolfinODESolver` from the same initial condition and compare
# the resulting voltage fields directly. This needs its own `time` `dolfinx.fem.Constant` and its own
# stimulus expression tied to it — reusing the first run's `I_s` would silently reference the first
# run's (by-then-stale) time value instead of this run's own.

time_ref = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
stim_expr_ref = ufl.conditional(ufl.le(time_ref, 2.0), stim_amp, 0.0)
I_s_ref = beat.Stimulus(expr=stim_expr_ref, dZ=dx, marker=S1_marker)

pde_ref = beat.MonodomainModel(time=time_ref, mesh=mesh, M=M, I_s=I_s_ref)
ode_ref = beat.odesolver.DolfinODESolver(
    v_ode=dolfinx.fem.Function(V_ode),
    v_pde=pde_ref.state,
    fun=fun,
    init_states=init_states.copy(),
    parameters=parameters,
    num_states=num_states,
    v_index=v_index,
)
solver_ref = beat.MonodomainSplittingSolver(pde=pde_ref, ode=ode_ref)

t = 0.0
while t < T + 1e-12:
    solver_ref.step((t, t + dt))
    t += dt

diff = np.max(np.abs(pde.state.x.array - pde_ref.state.x.array))
print(f"Max |V_ExternalOperatorODESolver - V_DolfinODESolver| = {diff:.3e} mV")

# ## An honest look at performance
#
# `ExternalOperatorODESolver`'s *first* `step()` call pays a one-time cost: internally,
# `evaluate_operands` builds a `dolfinx.fem.Expression` to tabulate the operator's operand, which gets
# JIT-compiled by FFCx (a real C-compiler invocation) the first time it is used, and cached from then
# on. That one-time cost is roughly constant regardless of mesh size, so we report it separately from
# the steady-state (post-warm-up) per-step cost, which is what actually matters for a long-running
# simulation. `DolfinODESolver` has no such compilation step at all — it operates directly on a flat
# NumPy array — so we expect `ExternalOperatorODESolver` to still be slower per step, just not by the
# dramatic amount a naive (un-warmed-up) timing would suggest.

fun_numba = numba.jit(nopython=True)(model.generalized_rush_larsen)
# Warm up the JIT compilation before timing
fun_numba(states=init_states.copy(), t=0.0, parameters=parameters, dt=dt)

n_bench_steps = 10


def time_solver(solver_cls, fun_to_use, **extra):
    ode = solver_cls(
        v_ode=dolfinx.fem.Function(V_ode),
        v_pde=dolfinx.fem.Function(V_ode),
        fun=fun_to_use,
        init_states=init_states.copy(),
        parameters=parameters,
        num_states=num_states,
        v_index=v_index,
        **extra,
    )
    t0 = pytime.perf_counter()
    ode.step(0.0, dt)
    first_step = pytime.perf_counter() - t0

    t0 = pytime.perf_counter()
    t = dt
    for _ in range(n_bench_steps):
        ode.step(t, dt)
        t += dt
    steady_state_per_step = (pytime.perf_counter() - t0) / n_bench_steps
    return first_step, steady_state_per_step


results = {
    ("DolfinODESolver", "plain"): time_solver(beat.odesolver.DolfinODESolver, fun),
    ("DolfinODESolver", "numba"): time_solver(beat.odesolver.DolfinODESolver, fun_numba),
    ("ExternalOperatorODESolver", "plain"): time_solver(ExternalOperatorODESolver, fun),
    ("ExternalOperatorODESolver", "numba"): time_solver(ExternalOperatorODESolver, fun_numba),
}

print(f"\n{N_ode} points, {num_states}-state cell model, {n_bench_steps} steps after warm-up:")
print(f"{'solver':<28}{'fun':<8}{'first step (s)':>16}{'steady-state (s/step)':>24}")
for (solver_name, fun_name), (first_step, steady_state) in results.items():
    print(f"{solver_name:<28}{fun_name:<8}{first_step:>16.4f}{steady_state:>24.5f}")

# The "first step" column shows the one-time FFCx compilation cost described above — large, and
# largest of all for `ExternalOperatorODESolver` with a Numba-jitted `fun`, since the cell-model call
# itself is by far the cheapest part of that combination, so the fixed compilation cost dominates
# completely. The "steady-state" column is the fairer comparison for a long-running simulation:
# `ExternalOperatorODESolver` is consistently slower there too, but only by a modest, roughly constant
# factor (a handful of times, not thousands) — the real, structural cost of routing every step through
# `dolfinx-external-operator`'s general-purpose UFL operand/coefficient evaluation machinery, for what
# is, today, just a flat array update.
#
# If raw throughput of today's *explicit* splitting scheme is all you need, `DolfinODESolver` is the
# better choice on both counts. `ExternalOperatorODESolver` earns its keep once you want something
# `DolfinODESolver` cannot do at all: supply a JAX- or PyTorch-based cell model with automatic
# differentiation, and couple it to the PDE *implicitly* — in a single monolithic Newton solve, with no
# operator-splitting error — instead of the explicit array update shown here. That is a genuinely new
# capability rather than a performance trade-off; see Section 7.2 of the
# [mathematical background](../docs/math_background.md) page.
