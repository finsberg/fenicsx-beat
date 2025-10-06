# # Conduction velocity and ECG for slabs
# In this demo we will show how to compute conduction velocity and ECG for a Slab geometry.
#

from pathlib import Path
import shutil
from mpi4py import MPI


import adios4dolfinx


import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import gotranx
import scifem
import beat
import pyvista

import beat.postprocess
from beat.geometry import Geometry

comm = MPI.COMM_WORLD
results_folder = Path("results-slab")
results_folder.mkdir(exist_ok=True)
save_every_ms = 1.0

transverse = False
end_time = 20.0
dt = 0.05
overwrite = False
stim_amp = 5000.0
mesh_unit = "cm"
dx = 0.05 * beat.units.ureg("cm").to(mesh_unit).magnitude
L = 1.0 * beat.units.ureg("cm").to(mesh_unit).magnitude
mesh = beat.geometry.get_3D_slab_mesh(comm=comm, Lx=L, Ly=dx, Lz=dx, dx=dx / 5)


tol = 1.0e-8

marker = 1


def S1_subdomain(x):
    return x[0] <= tol


facets = dolfinx.mesh.locate_entities_boundary(
    mesh,
    mesh.topology.dim - 1,
    S1_subdomain,
)
ffun = dolfinx.mesh.meshtags(
    mesh,
    mesh.topology.dim - 1,
    facets,
    np.full(len(facets), marker, dtype=np.int32),
)

V = dolfinx.fem.functionspace(mesh, ("P", 1))

plotter_markers = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V))
plotter_markers.add_mesh(grid, show_edges=True)
if mesh.geometry.dim == 2:
    plotter_markers.view_xy()

if not pyvista.OFF_SCREEN:
    plotter_markers.show()
else:
    plotter_markers.screenshot(results_folder / "markers.png")


def endo_epi(x):
    return np.where(x[0] < L / 3, 1, np.where(x[0] > 2 * L / 3, 2, 0))


cfun_func = dolfinx.fem.Function(V)
cfun_func.interpolate(endo_epi)


plotter_markers = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V))
grid.point_data["V"] = cfun_func.x.array
plotter_markers.add_mesh(grid, show_edges=True)
if mesh.geometry.dim == 2:
    plotter_markers.view_xy()

if not pyvista.OFF_SCREEN:
    plotter_markers.show()
else:
    plotter_markers.screenshot(results_folder / "endo_epi.png")


with dolfinx.io.XDMFFile(comm, results_folder / "ffun.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ffun, mesh.geometry)


with dolfinx.io.XDMFFile(comm, results_folder / "endo_epi.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(cfun_func)

g_il = 0.16069
g_el = 0.625
g_it = 0.04258
g_et = 0.236

f0, s0, n0 = beat.geometry.get_3D_slab_microstructure(mesh, transverse)

markers = {"ENDO": (marker, 2)}

data = Geometry(
    mesh=mesh,
    ffun=ffun,
    markers=markers,
    f0=f0,
    s0=s0,
    n0=n0,
)
save_freq = round(save_every_ms / dt)


print("Running model")
# Load the model
model_path = Path("ToRORd_dynCl_endo.py")
if not model_path.is_file():
    print("Generate code for cell model")
    here = Path.cwd()
    ode = gotranx.load_ode(here / ".." / "odes" / "torord" / "ToRORd_dynCl_endo.ode")
    code = gotranx.cli.gotran2py.get_code(
        ode,
        scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
    )
    model_path.write_text(code)

import ToRORd_dynCl_endo

model = ToRORd_dynCl_endo.__dict__

# Surface to volume ratio

chi = 1400.0 * beat.units.ureg("cm**-1")

# Membrane capacitance

C_m = 1.0 * beat.units.ureg("uF/cm**2")


print("Get steady states")
nbeats = 2  # Should be set to at least 200
init_states = {
    0: beat.single_cell.get_steady_state(
        fun=model["generalized_rush_larsen"],
        init_states=model["init_state_values"](),
        parameters=model["init_parameter_values"](celltype=2),
        outdir=results_folder / "mid",
        BCL=1000,
        nbeats=nbeats,
        track_indices=[model["state_index"]("v"), model["state_index"]("cai")],
        dt=0.05,
    ),
    1: beat.single_cell.get_steady_state(
        fun=model["generalized_rush_larsen"],
        init_states=model["init_state_values"](),
        parameters=model["init_parameter_values"](celltype=0),
        outdir=results_folder / "endo",
        BCL=1000,
        nbeats=nbeats,
        track_indices=[
            model["state_index"]("v"),
            model["state_index"]("cai"),
            model["state_index"]("nai"),
        ],
        dt=0.05,
    ),
    2: beat.single_cell.get_steady_state(
        fun=model["generalized_rush_larsen"],
        init_states=model["init_state_values"](),
        parameters=model["init_parameter_values"](celltype=1),
        outdir=results_folder / "epi",
        BCL=1000,
        nbeats=nbeats,
        track_indices=[model["state_index"]("v"), model["state_index"]("cai")],
        dt=0.05,
    ),
}
# endo = 0, epi = 1, M = 2
parameters = {
    0: model["init_parameter_values"](i_Stim_Amplitude=0.0, celltype=2),
    1: model["init_parameter_values"](i_Stim_Amplitude=0.0, celltype=0),
    2: model["init_parameter_values"](i_Stim_Amplitude=0.0, celltype=1),
}
fun = {
    0: model["generalized_rush_larsen"],
    1: model["generalized_rush_larsen"],
    2: model["generalized_rush_larsen"],
}
v_index = {
    0: model["state_index"]("v"),
    1: model["state_index"]("v"),
    2: model["state_index"]("v"),
}

time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
I_s = beat.stimulation.define_stimulus(
    mesh=data.mesh,
    chi=chi,
    time=time,
    subdomain_data=data.ffun,
    marker=markers["ENDO"][0],
    amplitude=stim_amp,
    mesh_unit=mesh_unit,
)

M = beat.conductivities.define_conductivity_tensor(
    chi,
    f0=data.f0,
    g_il=g_il,
    g_it=g_it,
    g_el=g_el,
    g_et=g_et,
)

params = {"preconditioner": "sor", "use_custom_preconditioner": False}
pde = beat.MonodomainModel(
    time=time,
    mesh=data.mesh,
    M=M,
    I_s=I_s,
    params=params,
    C_m=C_m.to(f"uF/{mesh_unit}**2").magnitude,
)

V_ode = dolfinx.fem.functionspace(data.mesh, ("P", 1))
ode = beat.odesolver.DolfinMultiODESolver(
    v_ode=dolfinx.fem.Function(V_ode),
    v_pde=pde.state,
    markers=cfun_func,
    num_states={i: len(s) for i, s in init_states.items()},
    fun=fun,
    init_states=init_states,
    parameters=parameters,
    v_index=v_index,
)
t = 0.0
solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)

vtxfname = results_folder / "slab.bp"
checkpointfname = results_folder / "slab_checkpoint.bp"

shutil.rmtree(vtxfname, ignore_errors=True)
shutil.rmtree(checkpointfname, ignore_errors=True)
vtx = dolfinx.io.VTXWriter(
    comm,
    vtxfname,
    [solver.pde.state],
    engine="BP4",
)

adios4dolfinx.write_mesh(checkpointfname, mesh)


def save(t):
    v = solver.pde.state.x.array
    print(f"Solve for {t=:.2f}, {v.max() =}, {v.min() =}")
    vtx.write(t)
    adios4dolfinx.write_function(checkpointfname, solver.pde.state, time=t, name="v")


i = 0
while t < end_time + 1e-12:
    # Make sure to save at the same time steps that is used by Ambit

    if i % save_freq == 0:
        save(t)

    solver.step((t, t + dt))
    i += 1
    t += dt


# ## Compute conduction velocity and ECG

threshold = 0.0
x0 = L * 0.25
x1 = L * 0.75
if mesh.geometry.dim == 2:
    p_ecg = (L * 2.0, dx * 0.5)
    p1 = (x0, dx * 0.5)
    p2 = (x1, dx * 0.5)
else:
    p_ecg = (L * 2.0, dx * 0.5, dx * 0.5)  # type: ignore
    p1 = (x0, dx * 0.5, dx * 0.5)  # type: ignore
    p2 = (x1, dx * 0.5, dx * 0.5)  # type: ignore


# Need to either save the functions on the input mesh using adios4dolfinx.write_function_on_input_mesh or read the mesh again see https://jsdokken.com/adios4dolfinx/docs/original_checkpoint.html

mesh = adios4dolfinx.read_mesh(comm=comm, filename=checkpointfname)
V = dolfinx.fem.functionspace(mesh, ("P", 1))
v = dolfinx.fem.Function(V)


plotter_voltage = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V))
grid.point_data["V"] = v.x.array
viridis = plt.get_cmap("viridis")
sargs = dict(
    title_font_size=25,
    label_font_size=20,
    fmt="%.2e",
    color="black",
    position_x=0.1,
    position_y=0.8,
    width=0.8,
    height=0.1,
)

# +
plotter_voltage.add_mesh(
    grid,
    show_edges=True,
    lighting=False,
    cmap=viridis,
    scalar_bar_args=sargs,
    clim=[-90.0, 40.0],
)

times = beat.postprocess.read_timestamps(comm, checkpointfname, "v")
t1 = np.inf
t2 = np.inf
phie = []
ecg = beat.ecg.ECGRecovery(
    v=v, sigma_b=1.0, C_m=C_m.to(f"uF/{mesh_unit}**2").magnitude, M=M,
)
p_ecg_form = ecg.eval(p_ecg)
gif_file = Path("voltage_slab_time.gif")
gif_file.unlink(missing_ok=True)
plotter_voltage.open_gif(gif_file.as_posix())
for t in times:
    adios4dolfinx.read_function(checkpointfname, v, time=t, name="v")
    ecg.solve()
    phie.append(
        mesh.comm.allreduce(dolfinx.fem.assemble_scalar(p_ecg_form), op=MPI.SUM),
    )

    grid.point_data["V"] = v.x.array
    plotter_voltage.write_frame()
    v1, v2 = scifem.evaluate_function(v, [p1, p2])
    print(f"Read {t=:.2f}, {v1 =}, {v2 =}")

    if v1 > threshold:
        t1 = min(t, t1)
    if v2 > threshold:
        t2 = min(t, t2)

plotter_voltage.close()
# -

# ![volt](voltage_slab_time.gif "volt")

if not np.isclose(t1, t2):
    cv = (x1 - x0) / (t2 - t1) * beat.units.ureg(f"{mesh_unit}/ms")
    msg = (
        f"Conduction velocity = {cv.magnitude:.3f} mm/ms or "  #
        f" {cv.to('m/s').magnitude:.3f} m/s or "  #
        f" {cv.to('cm/s').magnitude:.3f} cm/s"  #
    )
    print(msg)


fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(times, phie)
ax.set_title("ECG recovery")
fig.savefig(results_folder / "ecg_recovery.png")
