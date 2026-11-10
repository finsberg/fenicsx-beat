# # The reaction-eikonal model: extracellular potentials and electrograms in a ventricular wedge
#
# This demo reproduces Fig. 7 of Neic et al. {cite}`neic2017efficient`: extracellular potentials
# $\phi_e$ and the electrogram at the electrode $P_E$ for the reaction-diffusion (R-D) model and the
# reaction-eikonal models R-E$^+$ and R-E$^-$ on the wedge of the
# [activation demo](reaction_eikonal_wedge_activation.py), now immersed in a 1 S/m bath.
#
# As in the paper, all three models are *pseudo-bidomain* models (Section 8.4 of the
# [mathematical background](../docs/math_background.md)): the transmembrane potential $v$ comes
# from a monodomain (or R-E) model whose conductivity is augmented in a thin layer along the
# tissue-bath interface to mimic bath loading {cite}`bishop2011representing`, and $\phi_e$ in
# tissue and bath is recovered from $v$ by one elliptic solve whenever it is needed. Electrograms
# at a fixed electrode are computed from a precomputed lead field, at the cost of one integral per
# time step.
#
# The live part runs R-E$^-$ on the 400 µm wedge for 40 ms; the full results (100 µm R-D, R-E$^+$,
# R-E$^-$ and 400 µm R-E$^+$ for 400 ms, from `scripts/reaction_eikonal_wedge*.py`) are shown at the
# end.

# +
from pathlib import Path

from mpi4py import MPI

import dolfinx
import gotranx
import matplotlib.pyplot as plt
import numpy as np

import beat
from beat.eikonal import Source, solve_eikonal
from beat.pseudo_bidomain import LeadFieldElectrodes, VolumeConductor
from beat.reaction_eikonal import ExponentialFoot, FootCurrentODESolver, ODEOnlySolver
from beat.utils import PointEvaluator

comm = MPI.COMM_WORLD
dx = 0.4  # mm
end_time = 40.0  # ms
dt = 0.02  # ms
# -

# ## The wedge and its bath
#
# `get_3D_wedge_bath_geometry` builds a box of bath around the wedge on the same grid, so the
# tissue part of the bath mesh has exactly the wedge's vertices and cells and $v$ moves between the
# two meshes without interpolation error. The paper does not give the size of the bath; we use 1 mm
# on every side (the bath-loading effect saturates at about that thickness
# {cite}`bishop2011representing`, and a thicker bath does not fit in memory at 100 µm on the
# ~32 GB machine used here, about 21 GB free).

# +
wedge = beat.geometry.get_3D_wedge_geometry(comm, dx=dx)
geo = wedge.geometry
mesh = geo.mesh
V = dolfinx.fem.functionspace(mesh, ("P", 1))
layers = beat.geometry.wedge_layer_markers(V)
bath = beat.geometry.get_3D_wedge_bath_geometry(comm, dx=dx, margin=1.0)
g = beat.conductivities.default_conductivities("Neic2017_wedge_bath")


def S(name):
    return g[name].to("S/m").magnitude


sigma_i_bath = beat.conductivities.conductivity_tensor(S("g_il"), S("g_it"), bath.f0)
sigma_e_bath = beat.conductivities.conductivity_tensor(S("g_el"), S("g_et"), bath.f0)
markers = beat.geometry.WEDGE_BATH_MARKERS
conductor = VolumeConductor(
    bath.mesh, bath.cell_tags, markers["TISSUE"], markers["BATH"], sigma_i_bath, sigma_e_bath, S("g_b"),
)
if comm.rank == 0:
    print(f"bath margins {bath.margins} mm, {bath.mesh.topology.index_map(0).size_global} vertices")
# -

# ## R-E$^-$ with an electrogram at $P_E$
#
# The ten Tusscher-Panfilov cell models and the eikonal activation time are set up as in the
# [activation demo](reaction_eikonal_wedge_activation.py). R-E$^-$ has no diffusion, so the
# augmented conductivity does not enter here; it does enter R-D and R-E$^+$ (see the scripts).

# +
ode_dir = Path.cwd().parent / "odes" / "tentusscher_panfilov_2006"
cell_models = {}
for layer, name in {"EPI": "epi", "M": "M", "ENDO": "endo"}.items():
    code = gotranx.cli.gotran2py.get_code(
        gotranx.load_ode(ode_dir / f"tentusscher_panfilov_2006_{name}_cell.ode"),
        scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
    )
    namespace: dict = {}
    exec(code, namespace)
    cell_models[beat.geometry.WEDGE_LAYERS[layer]] = namespace
init = {k: m["init_state_values"]() for k, m in cell_models.items()}
cells = beat.odesolver.DolfinMultiODESolver(
    v_ode=dolfinx.fem.Function(V),
    v_pde=dolfinx.fem.Function(V),
    markers=layers,
    init_states=init,
    parameters={k: m["init_parameter_values"](stim_amplitude=0.0) for k, m in cell_models.items()},
    fun={k: m["generalized_rush_larsen"] for k, m in cell_models.items()},
    num_states={k: len(v) for k, v in init.items()},
    v_index={k: m["state_index"]("V") for k, m in cell_models.items()},
)
cv = (0.594, 0.324, 0.324)  # mm/ms, from scripts/reaction_eikonal_cv.json
sources = [Source((wedge.cell_tags, tag), t0) for tag, t0 in wedge.stimuli.values()]
t_a = solve_eikonal(mesh, cv, (geo.f0, geo.s0, geo.n0), sources)
solver = ODEOnlySolver(FootCurrentODESolver(cells, t_a, ExponentialFoot()))
v = solver.v

assert geo.f0 is not None
sigma_i = beat.conductivities.conductivity_tensor(S("g_il"), S("g_it"), geo.f0)
electrodes = LeadFieldElectrodes(conductor, v, sigma_i, {"P_E": wedge.electrode})
trace_t, trace = [0.0], [electrodes()["P_E"]]
t = 0.0
while t < end_time - 1e-9:
    solver.step((t, t + dt))
    t += dt
    if round(t / dt) % 5 == 0:
        trace_t.append(t)
        trace.append(electrodes()["P_E"])
# -

# The extracellular potential at $t = 40$ ms on the mid-plane $z = 2.5$ mm, next to the
# electrogram so far:

# +
phi_e = conductor.solve(v)
gx, gy = np.meshgrid(np.linspace(0, 25, 101), np.linspace(0, 25, 101))
points = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 2.5)])
phi_grid = PointEvaluator(bath.mesh, points)(phi_e).reshape(gx.shape)
if comm.rank == 0:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    cs = ax0.contourf(gx, gy, phi_grid, levels=21, cmap="jet")
    fig.colorbar(cs, ax=ax0, label=r"$\phi_e$ [mV]")
    ax0.set_aspect("equal")
    ax0.set_title(rf"R-E$^-$ (400 µm), $\phi_e$ at t = {end_time:g} ms")
    ax1.plot(trace_t, trace)
    ax1.set_xlabel("t [ms]")
    ax1.set_ylabel(r"$\phi_e(P_E)$ [mV]")
    ax1.set_title("Electrogram at $P_E$")
    fig.tight_layout()
    plt.show()
# -

# ## Precomputed results
#
# The figure below was computed with the scripts in `scripts/` on 16 MPI ranks (see their
# docstrings): the lead field of $P_E$ (`reaction_eikonal_wedge_leadfield.py`), the bath-loaded
# runs with electrograms and snapshots of $v$ at 40 and 320 ms (`reaction_eikonal_wedge.py`), and
# $\phi_e$ for each snapshot (`reaction_eikonal_wedge_phie.py`).
#
# ![Fig. 7](../docs/_static/reaction_eikonal_fig7.png)
#
# RESULTS-PARAGRAPH (written in Task 10 from the measured deviations; compare with the paper: 90% of
# the R-E$^+$ and 95% of the R-E$^-$ potentials within 0.25 mV and 0.75 mV of R-D at 40 ms, R-E$^+$
# electrograms virtually indistinguishable from R-D also at 400 µm, and R-E$^-$ overestimating the
# late repolarization deflection).
