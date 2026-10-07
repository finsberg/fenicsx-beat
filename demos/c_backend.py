# # A compiled C backend for cell models
#
# `beat` advances the cell model at every degree of freedom with a function
# `fun(states, t, parameters, dt)` that acts on a `(num_states, num_points)` array (see the
# [mathematical background](../docs/math_background.md) and `beat.odesolver.DolfinODESolver`).
# Most demos use the NumPy code that `gotranx` generates for it, and some compile that code with
# `numba`. `gotranx` can also generate C. `beat.c_backend` compiles the C, adds a loop over the
# points, and wraps the result in a function with the same signature, so that it can be passed as
# `fun` without changing anything else.
#
# In this demo we compare NumPy, Numba, C and C with OpenMP threads, for the ten Tusscher–Panfilov
# 2006 epicardial cell model (as in the [spiral wave demo](spiral_wave.py)) with the generalized
# Rush–Larsen scheme: first the ODE step alone, then a small monodomain simulation. We also check
# that they give the same solution.
#
# The C backend needs a C compiler when it runs. Any working dolfinx installation already has one,
# since FFCx compiles the finite element kernels with it. See the
# [C backend guide](../docs/c_backend.md) for the options.

# +
import os
import time as pytime
from pathlib import Path
from typing import Any

from mpi4py import MPI

import dolfinx
import gotranx
import matplotlib.pyplot as plt
import numpy as np

import beat
import beat.c_backend

try:
    import numba
except ImportError:
    numba = None
# -

comm = MPI.COMM_WORLD


def print0(*args, **kwargs):
    # Print on rank 0 only
    if comm.rank == 0:
        print(*args, **kwargs)


# ## The cell model
#
# We load the `.ode` file once, and generate the NumPy code from it as in the other demos.

# +
ode_file = (
    Path.cwd()
    / ".."
    / "odes"
    / "tentusscher_panfilov_2006"
    / "tentusscher_panfilov_2006_epi_cell.ode"
)
cell_ode = gotranx.load_ode(ode_file)

model_path = Path("tentusscher_panfilov_2006_epi_cell.py")
if not model_path.is_file():
    code = gotranx.cli.gotran2py.get_code(
        cell_ode,
        scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
    )
    model_path.write_text(code)

import tentusscher_panfilov_2006_epi_cell as model

num_states = cell_ode.num_states
v_index = model.state_index("V")
parameters = model.init_parameter_values(stim_amplitude=0.0)
dt = 0.05  # ms
# -

# ## The backends
#
# The NumPy backend is the generated function itself.

backends: dict[str, Any] = {"NumPy": model.generalized_rush_larsen}
build_times = {}

# Numba compiles the same function the first time it is called, so we call it once on a small
# array and time that separately.

if numba is not None:
    tic = pytime.perf_counter()
    fun_numba = numba.jit(nopython=True)(model.generalized_rush_larsen)
    warmup = np.zeros((num_states, 2))
    warmup.T[:] = model.init_state_values()
    fun_numba(states=warmup, t=0.0, parameters=parameters, dt=dt)
    build_times["Numba"] = pytime.perf_counter() - tic
    backends["Numba"] = fun_numba
else:
    print0("numba is not installed, so the Numba backend is left out")

# For the C backend, `beat.c_backend.from_ode` generates the C code with `gotranx`, compiles it on
# rank 0 with `cc -O3 -march=native`, and loads it on all ranks. The library is cached, keyed by a
# hash of the code, the compiler and the flags, so the next run skips the compilation.

tic = pytime.perf_counter()
fun_c = beat.c_backend.from_ode(cell_ode, scheme="generalized_rush_larsen")
build_times["C"] = pytime.perf_counter() - tic
backends["C"] = fun_c
print0(f"The C library is {fun_c.library_path}")

# With `openmp=True` the loop over the points runs in parallel with OpenMP threads, in addition to
# any MPI ranks. The number of threads is `num_threads`, or `OMP_NUM_THREADS` if it is not given.
# Here we use up to four threads.

# +
if beat.c_backend.openmp_available():
    num_threads = min(4, os.cpu_count() or 1)
    tic = pytime.perf_counter()
    backends["C + OpenMP"] = beat.c_backend.from_ode(
        cell_ode,
        scheme="generalized_rush_larsen",
        openmp=True,
        num_threads=num_threads,
    )
    build_times["C + OpenMP"] = pytime.perf_counter() - tic
    print0(f"C + OpenMP uses {backends['C + OpenMP'].max_threads()} threads")
else:
    print0("The C compiler does not support OpenMP, so C + OpenMP is left out")
for name, seconds in build_times.items():
    print0(f"Building the {name} backend took {seconds:.2f} s")
# -

# ## The ODE step alone
#
# We time one step of each backend for 10³, 10⁴ and 10⁵ points, best of five.

# +
sizes = [1_000, 10_000, 100_000]
repeats = 5


def time_ode_step(fun, num_points):
    states = np.zeros((num_states, num_points))
    states.T[:] = model.init_state_values()
    best = np.inf
    for _ in range(repeats):
        tic = pytime.perf_counter()
        fun(states=states, t=0.0, parameters=parameters, dt=dt)
        best = min(best, pytime.perf_counter() - tic)
    return best


step_times = {name: np.array([time_ode_step(f, n) for n in sizes]) for name, f in backends.items()}

print0("Time per step in ms (speed-up over NumPy)")
print0(f"{'points':>8}" + "".join(f"{name:>18}" for name in backends))
for i, n in enumerate(sizes):
    row = "".join(
        f"{1e3 * step_times[name][i]:10.2f} ({step_times['NumPy'][i] / step_times[name][i]:4.1f}x)"
        for name in backends
    )
    print0(f"{n:>8}{row}")

fig, ax = plt.subplots()
for name, times in step_times.items():
    ax.loglog(sizes, 1e9 * times / np.array(sizes), marker="o", label=name)
ax.set_xlabel("Number of points")
ax.set_ylabel("Time per point and step (ns)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.savefig("c_backend_ode_step.png")
# -

# ## A monodomain simulation
#
# Next we run the same small monodomain simulation with each backend: a $20 \times 7$ mm sheet with
# a mesh size of 0.25 mm, stimulated in one corner, for 30 ms. The setup (units, $\chi$, $C_m$ and
# $D$) is that of the [spiral wave demo](spiral_wave.py). A `beat.PerformanceMonitor` records the
# time spent in the ODE step and in the PDE step separately.

# +
L_x, L_y, h = 20.0, 7.0, 0.25  # mm
end_time = 30.0  # ms
mesh = dolfinx.mesh.create_rectangle(
    comm,
    [[0.0, 0.0], [L_x, L_y]],
    [round(L_x / h), round(L_y / h)],
    dolfinx.mesh.CellType.triangle,
)
tdim = mesh.topology.dim
stim_marker = 1
stim_cells = dolfinx.mesh.locate_entities(
    mesh,
    tdim,
    lambda x: np.logical_and(x[0] <= 1.5 + 1e-8, x[1] <= 1.5 + 1e-8),
)
stim_tags = dolfinx.mesh.meshtags(
    mesh,
    tdim,
    stim_cells,
    np.full(len(stim_cells), stim_marker, dtype=np.int32),
)
chi = beat.units.ureg.Quantity(1400.0, "cm**-1")
C_m = beat.units.ureg.Quantity(1.0, "uF/cm**2").to("uF/mm**2").magnitude
D = 0.122  # mm^2 / ms


def run_monodomain(fun):
    time = dolfinx.fem.Constant(mesh, 0.0)
    I_s = [
        beat.stimulation.define_stimulus(
            mesh=mesh,
            chi=chi,
            time=time,
            subdomain_data=stim_tags,
            marker=stim_marker,
            mesh_unit="mm",
            amplitude=50_000.0,
            start=0.0,
            duration=2.0,
        ),
    ]
    pde = beat.MonodomainModel(
        time=time,
        mesh=mesh,
        M=D * C_m,
        I_s=I_s,
        C_m=C_m,
        dx=I_s[0].dZ,
        params={
            "petsc_options": {"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "boomeramg"},
        },
    )
    ode = beat.odesolver.DolfinODESolver(
        v_ode=dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1))),
        v_pde=pde.state,
        fun=fun,
        init_states=model.init_state_values(),
        parameters=parameters,
        num_states=num_states,
        v_index=v_index,
    )
    monitor = beat.PerformanceMonitor(log_frequency=0, comm=comm)
    solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode, monitor=monitor)
    t = 0.0
    for _ in range(round(end_time / dt)):
        solver.step((t, t + dt))
        t += dt
    return pde.state.x.array.copy(), monitor.timings


results = {name: run_monodomain(f) for name, f in backends.items()}
# -

# +
numpy_timings = results["NumPy"][1]
print0(
    f"{'backend':>12}{'total (s)':>12}{'ODE (s)':>10}{'PDE (s)':>10}"
    f"{'ODE speed-up':>14}{'total speed-up':>16}",
)
for name, (_, timings) in results.items():
    print0(
        f"{name:>12}{timings['total_step']:12.2f}{timings['ode_step']:10.2f}"
        f"{timings['pde_step']:10.2f}"
        f"{numpy_timings['ode_step'] / timings['ode_step']:14.1f}"
        f"{numpy_timings['total_step'] / timings['total_step']:16.1f}",
    )
# -

# The ODE step gets faster, but the PDE step takes the same time with every backend, so the overall
# speed-up is smaller than that of the ODE step alone.
#
# ## The backends give the same solution
#
# The backends evaluate the same expressions, and differ only in rounding (the C library and NumPy
# compute `exp` and `log` slightly differently), so the membrane potential should agree to
# round-off.

v_numpy = results["NumPy"][0]
v_peak = comm.allreduce(v_numpy.max(initial=-np.inf), op=MPI.MAX)
differences = {
    name: comm.allreduce(np.max(np.abs(v - v_numpy), initial=0.0), op=MPI.MAX)
    for name, (v, _) in results.items()
}
print0(f"peak membrane potential: {v_peak:.1f} mV")
for name, difference in differences.items():
    print0(f"max |v_{name} - v_NumPy| = {difference:.2e} mV")
assert v_peak > 0.0, "the stimulus did not produce an action potential"
assert all(difference <= 1e-9 for difference in differences.values())

# ## Notes
#
# - **Other schemes and code generation options.** `from_ode` takes the name of any `gotranx`
#   scheme, and passes `codegen_kwargs` on to `gotranx.codegen.CCodeGenerator.scheme`. If you
#   already have C code (for example from `gotranx ode2c`), use `beat.c_backend.compile_scheme`.
# - **The cache.** The libraries are stored in `~/.cache/beat/c_backend`, or in `$BEAT_C_CACHE` if
#   it is set. In parallel, rank 0 compiles and the other ranks load the library from there, so on a
#   cluster the cache directory must be on a file system that all nodes share.
# - **`-march=native`.** The library is compiled inside your job, on the compute node. With a
#   native flag the cache key includes the CPU features the compiler detects (its predefined macros),
#   so a library built on a login node with a different CPU (e.g. with AVX-512) is not reused on the
#   compute nodes. Pass `cflags=("-O3",)` for a portable library.
# - **OpenMP and MPI.** OpenMP threads combine with MPI ranks. Set `num_threads` (or
#   `OMP_NUM_THREADS`) to at most cores-per-node / ranks-per-node. See the
#   [guide](../docs/c_backend.md).
