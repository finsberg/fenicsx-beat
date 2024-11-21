# # Niederer benchmark
# In this example we will use the same setup as in the Niederer benchmark {cite}`land2015verification`.

from pathlib import Path
import json

from mpi4py import MPI

# import beat.conductivities
import dolfinx
import ufl
import numpy as np
import numpy.typing as npt

import pyvista
import gotranx
import matplotlib.pyplot as plt

import beat


def setup_initial_conditions() -> npt.NDArray:
    ic = {
        "V": -85.23,  # mV
        "Xr1": 0.00621,
        "Xr2": 0.4712,
        "Xs": 0.0095,
        "m": 0.00172,
        "h": 0.7444,
        "j": 0.7045,
        "d": 3.373e-05,
        "f": 0.7888,
        "f2": 0.9755,
        "fCass": 0.9953,
        "s": 0.999998,
        "r": 2.42e-08,
        "Ca_i": 0.000126,  # millimolar
        "R_prime": 0.9073,
        "Ca_SR": 3.64,  # millimolar
        "Ca_ss": 0.00036,  # millimolar
        "Na_i": 8.604,  # millimolar
        "K_i": 136.89,  # millimolar
    }
    values = model.init_state_values(**ic)
    return values


comm = MPI.COMM_WORLD
# +
dx = 0.5
dt = 0.05
# Increase T to 100 to reproduce Niederer benchmark
T = 20.0

here = Path.cwd()

model_path = Path("tentusscher_panfilov_2006_epi_cell.py")
if not model_path.is_file():
    here = Path.cwd()
    ode = gotranx.load_ode(
        here
        / ".."
        / "odes"
        / "tentusscher_panfilov_2006"
        / "tentusscher_panfilov_2006_epi_cell.ode"
    )
    code = gotranx.cli.gotran2py.get_code(
        ode, scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen]
    )
    model_path.write_text(code)

import tentusscher_panfilov_2006_epi_cell as model

fun = model.generalized_rush_larsen
init_states = setup_initial_conditions()
parameters = model.init_parameter_values(stim_amplitude=0.0)

mesh_unit = "mm"

Lx = 20.0 * beat.units.ureg("mm").to(mesh_unit).magnitude
Ly = 7 * beat.units.ureg("mm").to(mesh_unit).magnitude
Lz = 3 * beat.units.ureg("mm").to(mesh_unit).magnitude
dx_mm = dx * beat.units.ureg("mm").to(mesh_unit).magnitude

geo = beat.geometry.get_3D_slab_geometry(
    comm=comm,
    Lx=Lx,
    Ly=Ly,
    Lz=Lz,
    dx=dx_mm,
)
ode_space = dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 1))

pyvista.start_xvfb()
plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(geo.mesh))
plotter.add_mesh(grid, show_edges=True)
plotter.show_grid()
plotter.add_axes(line_width=5)
plotter.show_axes()
plotter.view_xy()

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("niederer_mesh.png")

# +
# Surface to volume ratio
conductivities = beat.conductivities.default_conductivities("Niederer")
# # Membrane capacitance
C_m = 1.0 * beat.units.ureg("uF/cm**2")

time = dolfinx.fem.Constant(geo.mesh, 0.0)
L = 1.5 * beat.units.ureg("mm").to(mesh_unit).magnitude
S1_marker = 1
L = 1.5


tol = 1.0e-10


def S1_subdomain(x):
    return np.logical_and(
        np.logical_and(x[0] <= L + tol, x[1] <= L + tol), x[2] <= L + tol
    )


cells = dolfinx.mesh.locate_entities(geo.mesh, geo.mesh.topology.dim, S1_subdomain)
S1_markers = dolfinx.mesh.meshtags(
    geo.mesh,
    geo.mesh.topology.dim,
    cells,
    np.full(len(cells), S1_marker, dtype=np.int32),
)

I_s = beat.stimulation.define_stimulus(
    mesh=geo.mesh,
    chi=conductivities["chi"],
    time=time,
    subdomain_data=S1_markers,
    marker=S1_marker,
    mesh_unit=mesh_unit,
    amplitude=50_000.0,
)


M = beat.conductivities.define_conductivity_tensor(
    f0=geo.f0,
    **conductivities,
)

params = {"preconditioner": "sor", "use_custom_preconditioner": False}

pde = beat.MonodomainModel(
    time=time,
    mesh=geo.mesh,
    M=M,
    I_s=I_s,
    params=params,
    C_m=C_m.to(f"uF/{mesh_unit}**2").magnitude,
    dx=I_s.dZ,
)
ode = beat.odesolver.DolfinODESolver(
    v_ode=dolfinx.fem.Function(ode_space),
    v_pde=pde.state,
    fun=fun,
    init_states=init_states,
    parameters=parameters,
    num_states=len(init_states),
    v_index=model.state_index("V"),
)

# +
solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)
output_dir = Path("results-niederer-benchmark")
output_dir.mkdir(exist_ok=True)
filename = output_dir / f"results-{dt}-{dx}.xdmf"
filename.unlink(missing_ok=True)
filename.with_suffix(".h5").unlink(missing_ok=True)

vtx = dolfinx.io.VTXWriter(
    comm, "niederer_benchmark.bp", [solver.pde.state], engine="BP4"
)

points = {
    "P1": (0, 0, 0),
    "P2": (0.0, Ly, 0.0),
    "P3": (Lx, 0.0, 0.0),
    "P4": (Lx, Ly, 0.0),
    "P5": (0.0, 0.0, Lz),
    "P6": (0.0, Ly, Lz),
    "P7": (Lx, 0.0, Lz),
    "P8": (Lx, Ly, Lz),
    "P9": (Lx / 2, Ly / 2, Lz / 2),
}

# +
activation_times = {p: -1.0 for p in points}
save_freq = int(1.0 / dt)
i = 0
plotter_voltage = pyvista.Plotter()
viridis = plt.get_cmap("viridis")
grid.point_data["V"] = solver.pde.state.vector().get_local()
grid.set_active_scalars("V")
renderer = plotter_voltage.add_mesh(
    grid,
    show_edges=True,
    lighting=False,
    cmap=viridis,
    clim=[-90.0, 40.0],
)
gif_file = Path("niederer_benchmark.gif")
gif_file.unlink(missing_ok=True)
plotter_voltage.open_gif(gif_file.as_posix())

T = 20
# T = 100  # Change to 100 to reproduce Niederer benchmark
t = 0.0
while t < T + 1e-12 and any(at < 0.0 for at in activation_times.values()):
    v = solver.pde.state.x.array
    if i % save_freq == 0:
        print(f"Solve for {t=:.2f}, {v.max() =}, {v.min() = }")
        print(activation_times)
        vtx.write(t)
        grid.point_data["V"] = solver.pde.state.vector().get_local()
        plotter_voltage.write_frame()
    solver.step((t, t + dt))

    for p in points:
        value = beat.utils.evaluate_function(solver.pde.state, [points[p]]).squeeze()
        if value > 0.0 and activation_times[p] < 0.0:
            activation_times[p] = t
    i += 1
    t += dt

plotter_voltage.close()
# -

# ![_](niederer_benchmark.gif)

# Save activation times
activation_times["dx"] = dx
activation_times["dt"] = dt
at_file_name = output_dir / "activation_times.json"
if at_file_name.is_file():
    all_at = json.loads(at_file_name.read_text())
else:
    all_at = []
all_at.append(activation_times)
at_file_name.write_text(json.dumps(all_at, indent=2))

# The activation times are saved in the file `output-niederer-benchmark/activation_times.json`.
# The file contains a list of dictionaries, each dictionary contains the activation times for a specific dx and dt.
# Here are the activation times for the different dx and dt:
#
# |    |   dx |    dt |    P1 |     P2 |     P3 |     P4 |     P5 |     P6 |     P7 |     P8 |     P9 |
# |---:|-----:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
# |  0 |  0.5 | 0.05  | 1.25  | 51.1   | 34.9   | 58.9   | 14.1   | 49.5   | 34     | 56.65  | 26.05  |
# |  1 |  0.5 | 0.01  | 1.22  | 50.85  | 33.96  | 58.05  | 13.98  | 49.36  | 33.07  | 55.91  | 25.64  |
# |  2 |  0.5 | 0.005 | 1.215 | 50.775 | 33.825 | 57.96  | 13.97  | 49.345 | 32.945 | 55.825 | 25.595 |
# |  3 |  0.2 | 0.05  | 1.25  | 29.7   | 32.9   | 40.2   |  9.55  | 30     | 32.95  | 39.9   | 18.9   |
# |  4 |  0.2 | 0.01  | 1.24  | 29.09  | 31.25  | 38.66  |  9.34  | 29.4   | 31.29  | 38.42  | 18.14  |
# |  5 |  0.2 | 0.005 | 1.235 | 29.015 | 31.05  | 38.475 |  9.315 | 29.32  | 31.08  | 38.235 | 18.045 |
# |  6 |  0.1 | 0.05  | 1.25  | 26.85  | 33.3   | 40.35  |  8.4   | 27.5   | 33.85  | 40.55  | 18.95  |
# |  7 |  0.1 | 0.01  | 1.23  | 25.64  | 31.46  | 38.08  |  8.03  | 26.24  | 31.94  | 38.21  | 17.95  |
# |  8 |  0.1 | 0.005 | 1.225 | 25.5   | 31.26  | 37.81  |  7.99  | 26.09  | 31.72  | 37.93  | 17.835 |


#
# ```{bibliography}
# ```
