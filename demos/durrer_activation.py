# # A Durrer-like activation sequence on a Bi-ventricular ellipsoid
#
# In a healthy heart the ventricles are not activated from one point but by the Purkinje network,
# which delivers the activation to a few early sites on the endocardium. Durrer et al.
# {cite}`durrer1970total` mapped these sites in isolated human hearts: three areas in the left
# ventricle (LV), high on the anterior paraseptal wall, in the centre of the left septal surface
# and low on the posterior paraseptal wall, all within the first 5 ms, and, about 5 ms later, two
# areas in the right ventricle (RV), on the septal surface and on the free wall near the insertion
# of the anterior papillary muscle. From these sites the activation spreads over the endocardium
# much faster than through the wall, while the base, which has few Purkinje fibres, is activated by
# the tissue.
#
# `beat.activation_sequence.EndocardialActivation` turns this description into a stimulus
# $I_{stim}$ for the monodomain model:
#
# 1. a `HeartFrame` (apex, long axis, septal and anterior directions) is found from the marked
#    surfaces of the mesh;
# 2. each `RootSite`, given by a height along the long axis and an angle around the LV axis, is
#    placed on the nearest endocardial vertex;
# 3. the time $t_{endo}$ at which a wave starting at the root sites (after their delays) and
#    travelling over the endocardial surface at a Purkinje-like speed $v_{endo}$ reaches each
#    endocardial vertex is the solution of the eikonal equation on that surface,
#
#    $$ v_{endo} \, \| \nabla_S t_{endo} \| = 1, \qquad t_{endo} = \text{delay at the roots}, $$
#
#    solved with `beat.eikonal.EikonalSolver`;
# 4. every cell with a facet on the endocardium below 85 % of the apex–base height receives the
#    stimulus current for 2 ms from the time the wave reaches it.
#
# This demo builds the sequence on the bi-ventricular ellipsoid of the
# [BiV demo](biv_endocardial.py), looks at it, and uses it to drive a monodomain simulation.

from pathlib import Path

from mpi4py import MPI

import cardiac_geometries
import dolfinx
import gotranx
import matplotlib.pyplot as plt
import numpy as np
import pyvista

import beat
from beat.activation_sequence import DURRER_ROOT_SITES, EndocardialActivation, RootSite

comm = MPI.COMM_WORLD
results_folder = Path("results-durrer-activation")
results_folder.mkdir(exist_ok=True)

# ## The geometry
#
# We use the bi-ventricular ellipsoid from [cardiac_geometries](https://computationalphysiology.github.io/cardiac-geometriesx),
# with fibres. The mesh is in cm and has the facet markers `LV` and `RV` (the endocardium), `EPI`
# and `BASE`. These are the markers `EndocardialActivation.from_geometry` looks for; a realistic
# mesh with valve planes (`MV`, `AV`, `TV`, `PV`) works too, and the pulmonary valve then fixes
# which side is anterior.

geodir = results_folder / "geo"
if not geodir.exists():
    comm.barrier()
    cardiac_geometries.mesh.biv_ellipsoid(
        comm=comm,
        outdir=geodir,
        char_length=0.3,  # Reduce this value to get a finer mesh
        create_fibers=True,
    )
geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)
mesh_unit = "cm"
print(geo.markers)

# ## The root sites
#
# The default root sites are `beat.activation_sequence.DURRER_ROOT_SITES`. The height is a fraction
# of the distance from the apex (0) to the base (1), measured along the normal of the base plane, so
# that the base is at height 1 on both ventricles. A float position is the angle around the LV
# axis in degrees, from the septum (0) towards the anterior wall (90); `"septal"` and `"free"` take
# the vertex of the RV at that height closest to and furthest from the LV axis. The delays are those
# of a human heart.

for name, site in DURRER_ROOT_SITES.items():
    print(f"{name:25s} {site}")

# ## The activation sequence
#
# The conduction velocity can be a pint quantity, here 2 mm/ms, a typical speed for the
# Purkinje-coupled endocardium; plain floats are taken to be in mesh units per ms. Everything is
# computed when the object is created. On this ellipsoid, without a pulmonary valve, the sign of the
# anterior direction is arbitrary; pass `anterior_point` (any point on the anterior side) to fix it.

ureg = beat.units.ureg
activation = EndocardialActivation.from_geometry(
    geo,
    mesh_unit=mesh_unit,
    conduction_velocity=ureg.Quantity(2.0, "mm/ms"),
    max_height=0.85,
)
summary = activation.summary()
if comm.rank == 0:
    print(
        f"{summary['num_stimulated_cells']} cells are stimulated between "
        f"{summary['first_stimulus']:.1f} and {summary['last_stimulus']:.1f} ms"
    )
    for name, root in summary["roots"].items():
        print(f"{name:25s} at {np.round(root['x'], 2)} after {root['delay']} ms")

# The object holds
#
# * `activation.frame`: the `HeartFrame`, whose `coordinates(x)` give the height, angle and distance
#   from the LV axis of any point, and `point(height, angle, radius)` the inverse (to place
#   electrodes, say);
# * `activation.activation_times`: $t_{endo}$ as a P1 function, `inf` away from the endocardium and
#   above `max_height`;
# * `activation.cell_activation_times`: a DG0 function with the time each endocardial cell is
#   stimulated (`inf` for the cells that are not).
#
# Let us look at $t_{endo}$ on the endocardium, with the root sites as spheres. We look at the
# heart from the RV side, through a transparent epicardium, and from the base into the cavities.

V = activation.activation_times.function_space
grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V))
t_endo = activation.activation_times.x.array.copy()
t_endo[~np.isfinite(t_endo)] = np.nan
grid.point_data["t_endo"] = t_endo
surface = grid.extract_surface()
endocardium = surface.threshold(value=(-1.0, 1e6), scalars="t_endo", all_scalars=True)
frame = activation.frame

plotter = pyvista.Plotter(shape=(1, 2), window_size=(1200, 600))
center = frame.base_center - 0.5 * frame.length * frame.long_axis
views = {
    "from the RV side": (frame.septal, frame.long_axis),
    "from the base, into the cavities": (frame.long_axis, frame.septal),
}
for column, (title, (direction, up)) in enumerate(views.items()):
    plotter.subplot(0, column)
    plotter.add_mesh(surface, color="lightgrey", opacity=0.15)
    plotter.add_mesh(
        endocardium,
        scalars="t_endo",
        cmap="plasma",
        scalar_bar_args=dict(title="t_endo (ms)", color="black"),
    )
    for name, root in activation.roots.items():
        plotter.add_mesh(pyvista.Sphere(radius=0.12, center=root["x"]), color="cyan")
    plotter.camera.focal_point = center
    plotter.camera.position = center + 15.0 * direction
    plotter.camera.up = up
    plotter.add_text(title, color="black", font_size=12)
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot(results_folder / "endocardial_activation.png")

# The LV is activated first, from its three septal sites; the RV follows 5 ms later and the last
# endocardial cells to be stimulated are just below `max_height`. The band at the top is the base,
# which is left to the tissue.
#
# ### Your own root sites
#
# Any set of `RootSite`s can be used, and `delay_scale` scales all the delays, for instance by the
# size of the heart relative to a human heart. Here the RV free-wall site is moved to the apex and
# the RV sites are delayed by 10 ms instead of 5:

custom_sites = dict(DURRER_ROOT_SITES)
custom_sites["RV free wall"] = RootSite("RV", height=0.2, position="free", delay=5.0)
custom = EndocardialActivation.from_geometry(
    geo,
    mesh_unit=mesh_unit,
    conduction_velocity=ureg.Quantity(2.0, "mm/ms"),
    root_sites=custom_sites,
    delay_scale=2.0,
)
if comm.rank == 0:
    print("last endocardial stimulus:", round(custom.summary()["last_stimulus"], 1), "ms")

# ## A monodomain simulation driven by the sequence
#
# The cell model is the endocardial ToR-ORd model, as in the [BiV demo](biv_endocardial.py), with
# its own stimulus switched off. To keep the demo short we start from the default initial states
# instead of pacing to steady state.

model_path = Path("ToRORd_dynCl_endo.py")
if not model_path.is_file():
    cell_ode = gotranx.load_ode(Path.cwd() / ".." / "odes" / "torord" / "ToRORd_dynCl_endo.ode")
    code = gotranx.cli.gotran2py.get_code(
        cell_ode, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen]
    )
    model_path.write_text(code)

import ToRORd_dynCl_endo

model = ToRORd_dynCl_endo.__dict__

# ### Calibrating the conductivities
#
# The tissue uses the Bishop conductivities, as in the [BiV demo](biv_endocardial.py), but not as
# they are. The elements of this mesh are about 3 mm across, while the upstroke of the action
# potential is less than 1 mm wide in tissue, and on a mesh that does not resolve the upstroke the
# wave is too slow: with these conductivities and this cell model, a plane wave does not propagate
# at all on a 3 mm mesh, and on a 1 mm mesh it travels at 0.26 mm/ms along the fibres and 0.10
# mm/ms across them, less than half of what is measured in human ventricles. The base, which only
# the tissue activates (it is above `max_height`), is then reached very late.
#
# The usual remedy {cite}`costa2013automatic` is to calibrate the conductivities for the mesh:
# choose them so that a plane wave on a slab, discretised with the mesh size, time step and cell
# model of the simulation, travels at the measured conduction velocities.
# `beat.conduction_velocity.calibrate_conductivities` does this:
#
# 1. `beat.conduction_velocity.measure_conduction_velocity` builds a slab of tetrahedra of size
#    `dx`, with the fibres along it or across it, stimulates one end, and fits the velocity to the
#    activation times (the upward crossing of −30 mV) of the nodes between 40 % and 90 % of its
#    length;
# 2. the conductivities along the fibres ($g_{il}$ and $g_{el}$) are multiplied by a factor
#    $k_l$ and those across them ($g_{it}$ and $g_{et}$) by $k_t$; scaling the intra- and
#    extracellular conductivity by the same factor scales the monodomain conductivity, their
#    harmonic mean, by that factor;
# 3. each iteration measures both velocities $c$ and updates each factor by
#
#    $$ k \leftarrow k \left( \frac{c_{target}}{c} \right)^{1/p}, $$
#
#    where $p = 1/2$ at first (the velocity of a resolved wave is proportional to the square root
#    of the conductivity) and then the slope of $\log c$ against $\log k$ between the last two
#    iterations, which is smaller on a coarse mesh; it stops when both velocities are within 2 % of
#    their targets.
#
# The targets are 0.6 mm/ms along the fibres and 0.3 mm/ms across them, typical of human
# ventricular myocardium and the values of the wedge benchmark of {cite}`neic2017efficient`. The
# mesh size is the characteristic length of the mesh, and the cell model and time step are those of
# the simulation below. Each measurement is a short simulation on a small slab; here the whole
# calibration takes about half a minute.

dt = 0.05
init_states = model["init_state_values"]()
parameters = model["init_parameter_values"](i_Stim_Amplitude=0.0, celltype=0)
calibration = beat.conduction_velocity.calibrate_conductivities(
    ureg.Quantity(0.6, "mm/ms"),
    ureg.Quantity(0.3, "mm/ms"),
    conductivities=beat.conductivities.default_conductivities("Bishop"),
    fun=model["generalized_rush_larsen"],
    init_states=init_states,
    parameters=parameters,
    v_index=model["state_index"]("v"),
    dx=ureg.Quantity(3.0, "mm"),
    dt=dt,
)
if comm.rank == 0:
    for scales, velocities in calibration.history:
        print(
            f"k_l = {scales['longitudinal']:5.2f}, k_t = {scales['transverse']:5.2f}: "
            f"{velocities['longitudinal']:.3f} mm/ms along, "
            f"{velocities['transverse']:.3f} mm/ms across the fibres"
        )
conductivities = calibration.conductivities

# The factors, about 7 along the fibres and 9 across them, are a correction for this mesh, not
# physiology; on a 1 mm mesh they are about 3. The slab is not the heart, so we checked the
# calibration against the activation map that the target velocities give in the heart, from the
# same endocardial times (an eikonal solution with `beat.eikonal.EikonalSolver`): on a 1 mm mesh
# the calibrated simulation is within 1.6 ms of it at 95 % of the nodes, while with the Bishop
# conductivities it is 7.6 ms late at half of them. On this 3 mm mesh the base is activated a few
# milliseconds early. For real simulations, use a mesh of 0.5 mm or finer and calibrate at that
# mesh size.
#
# ### The simulation
#
# The stimulus is what `activation.stimulus` returns: a `beat.Stimulus` that goes straight into
# `MonodomainModel` as `I_s`. It needs the time constant of the PDE and the time step. The
# amplitude is a membrane current density (by default 40 µA/cm² for 2 ms); a volumetric amplitude
# (current per volume) can be given as a pint quantity together with `chi`.
#
# The stimulus is a UFL expression of the PDE's time, so nothing has to be updated during the
# simulation. Each cell's pulse is averaged over the time step around the time at which the PDE
# evaluates the stimulus, so every cell receives the same charge whatever the time step and
# wherever its pulse starts within a step.

C_m = ureg.Quantity(1.0, "uF/cm**2")
M = beat.conductivities.define_conductivity_tensor(f0=geo.f0, **conductivities)
time = dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0))

I_s = activation.stimulus(time, dt, amplitude=ureg.Quantity(40.0, "uA/cm**2"), duration=2.0)

pde = beat.MonodomainModel(
    time=time,
    mesh=geo.mesh,
    M=M,
    I_s=I_s,
    C_m=C_m.to(f"uF/{mesh_unit}**2").magnitude,
)

V_ode = dolfinx.fem.functionspace(geo.mesh, ("P", 1))
num_points = V_ode.dofmap.index_map.size_local + V_ode.dofmap.index_map.num_ghosts
ode = beat.odesolver.DolfinODESolver(
    v_ode=dolfinx.fem.Function(V_ode),
    v_pde=pde.state,
    fun=model["generalized_rush_larsen"],
    init_states=np.repeat(init_states[:, None], num_points, axis=1),
    parameters=parameters,
    num_states=len(init_states),
    v_index=model["state_index"]("v"),
)
solver = beat.SplittingSolver(pde=pde, ode=ode)

# We record the activation time of every node with `beat.activation.ActivationTracker` and run
# until the whole heart is activated.

tracker = beat.activation.ActivationTracker(solver.v, method="threshold", threshold=-30.0)
end_time = 50.0
t = 0.0
for step in range(round(end_time / dt)):
    solver.step((t, t + dt))
    t += dt
    tracker.update(t)
    if step % round(10.0 / dt) == 0:
        v = solver.v.x.array
        print(f"t = {t:5.1f} ms, v in [{v.min():.1f}, {v.max():.1f}] mV")

t_a = tracker.t_a
n_owned = V_ode.dofmap.index_map.size_local
not_activated = comm.allreduce(int(np.isnan(t_a.x.array[:n_owned]).sum()), op=MPI.SUM)
last = comm.allreduce(float(np.nanmax(t_a.x.array[:n_owned])), op=MPI.MAX)
if comm.rank == 0:
    print(f"last activation at {last:.1f} ms; {not_activated} nodes not activated")

# Let us compare the tissue activation map with the endocardial sequence that drives it. On the
# endocardium the tissue follows the sequence closely (the stimulus has to raise $v$ to the
# threshold, which takes a fraction of a millisecond); from there the activation crosses the wall to
# the epicardium, and the base is activated last, from below.
#
# The largest lags are at the sharp edges of the RV cavity. Around a node on such an edge most of
# the tissue lies in cells that touch the endocardium only at the edge, so with the default
# `stimulated_cells="facet"` (only cells with a facet on the endocardium) the node gets less
# stimulus than its neighbours. `stimulated_cells="vertex"` also stimulates the cells that touch the
# endocardium at a vertex or an edge, which removes these lags on coarse meshes.

t_tissue = t_a.x.array.copy()
endo = np.isfinite(activation.activation_times.x.array[:n_owned])
delay = t_tissue[:n_owned][endo] - activation.activation_times.x.array[:n_owned][endo]
if comm.rank == 0:
    print(
        f"tissue minus endocardial time on the stimulated endocardium: "
        f"median {np.nanmedian(delay):.2f} ms, max {np.nanmax(delay):.2f} ms"
    )

grid.point_data["t_a"] = t_tissue
plotter = pyvista.Plotter(shape=(1, 2), window_size=(1200, 600))
for column, side in enumerate((1.0, -1.0)):
    plotter.subplot(0, column)
    plotter.add_mesh(
        grid.extract_surface(),
        scalars="t_a",
        cmap="plasma",
        clim=[0.0, last],
        scalar_bar_args=dict(title="activation time (ms)", color="black"),
    )
    plotter.camera.focal_point = frame.base_center - 0.5 * frame.length * frame.long_axis
    plotter.camera.position = plotter.camera.focal_point + side * 15.0 * frame.septal
    plotter.camera.up = frame.long_axis
    plotter.add_text("RV side" if side > 0 else "LV free wall", color="black", font_size=12)
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot(results_folder / "activation_map.png")

# And as a histogram: the share of the myocardium activated over time, against the share of the
# endocardium reached by the endocardial wave.

if comm.rank == 0:
    fig, ax = plt.subplots(figsize=(6, 4))
    ta = np.sort(t_tissue[:n_owned][np.isfinite(t_tissue[:n_owned])])
    te = np.sort(activation.activation_times.x.array[:n_owned][endo])
    ax.plot(ta, np.arange(1, len(ta) + 1) / len(ta), label="tissue (all nodes)")
    ax.plot(te, np.arange(1, len(te) + 1) / len(te), label="endocardial sequence")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("fraction activated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_folder / "activation_fraction.png")

# ## A whole beat: movie and ECG
#
# Running a whole beat on a fine mesh takes about half an hour, too long for the documentation
# build, so the movie and the ECG below were computed beforehand with
# [`scripts/durrer_activation_full.py`](https://github.com/finsberg/fenicsx-beat/blob/main/scripts/durrer_activation_full.py):
#
# ```shell
# python3 scripts/durrer_activation_full.py
# ```
#
# It uses the same ellipsoid, activation sequence and conduction velocities as above, with two
# changes. The mesh is finer, about 1 mm (`char_length=0.1`, 45 504 nodes instead of 2 834), with
# the conductivities calibrated for that mesh size: on the coarse mesh of this demo the wave front
# is a few elements wide, and near the base, where only the tissue activates the heart, it
# advances in visible jumps from element to element. And for a realistic repolarisation, the
# wall is split into endocardial, mid-myocardial and epicardial layers (30 %, 40 % and 30 %, with
# `beat.utils.expand_layer_biv`, as in the [BiV demo](biv_endocardial.py)), each with its own
# ToR-ORd cell type, and each cell type is paced for 200 beats at 1 Hz to steady state. It
# simulates 500 ms with a time step of 0.05 ms and records the 12-lead ECG every 1 ms, with
# `beat.ecg.ElectrodePotentials` at the torso electrodes of the [BiV demo](biv_endocardial.py).
#
# The movie shows $v$ on the epicardium, seen from the RV side, and on a cut along the long axis
# through both cavities. The LV septum is activated first and the RV 5 ms later; the wave spreads
# over the endocardium, crosses the wall, and reaches the base last, about 40 ms after the first
# root site fires (Durrer et al. measured about 80 ms, in hearts larger than this ellipsoid, which
# is 5.4 cm from apex to base). It has a frame every 2 ms for the first 100 ms and every 6 ms
# after that.
#
# ![Durrer activation movie](../docs/_static/durrer_activation.gif "v over one beat")
#
# The QRS complex lies within the activation (shaded) and the T wave peaks at about 250 ms, when
# the cells repolarise. The amplitudes are in arbitrary units, and since the geometry is an
# idealised ellipsoid in an idealised torso, the shape of each lead should not be compared
# directly with a clinical ECG.
#
# ```{figure} ../docs/_static/durrer_activation_ecg.png
# ---
# name: durrer_activation_ecg
# ---
# The 12-lead ECG of one beat driven by the Durrer activation sequence
# ```
#
# ## Where to go from here
#
# * Realistic geometries: `EndocardialActivation.from_geometry` works on any mesh with the `LV`,
#   `RV` and `EPI` markers and a base (`BASE`, or the valve planes `MV`, `AV`, `TV`, `PV`), e.g. the
#   UK Biobank meshes of the [UKB demo](ukb_atlas.py). On a mesh in mm, leave `mesh_unit="mm"`.
# * Several beats: add the start of each beat with `start`, e.g. one stimulus per beat passed as a
#   list to `I_s`.
# * Different markers: build the object directly with the marker values,
#   `EndocardialActivation(mesh, facet_tags, lv=..., rv=..., epi=..., base=[...])`.
