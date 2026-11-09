# # The reaction-eikonal model: repolarisation in a ventricular wedge
#
# Repolarisation gradients are the sources of the T wave. This demo reproduces Fig. 5 of
# {cite}`neic2017efficient`: at the instant of 50% repolarisation, $t_{r50} = 305$ ms, R-E$^+$
# reproduces the transmural profile of $v$ of the R-D model, while R-E$^-$, which ignores
# diffusion, keeps the sharp steps between the epicardial, M and endocardial layers, and so
# overestimates the gradients. See the [activation demo](reaction_eikonal_wedge_activation.py)
# for the models and set-up.
#
# Simulating 500 ms is too long for the documentation build, so we only run R-E$^-$ — the
# cheapest model — on the 400 µm wedge for `end_time` ms and plot the transmural profile; set
# `end_time = 305.0` to see the steps at $t_{r50}$. The precomputed results (100 µm R-D against
# 400 µm R-E) are shown at the end.

# +
import time as walltime
from pathlib import Path

from mpi4py import MPI

import dolfinx
import gotranx
import matplotlib.pyplot as plt
import numpy as np
import scifem

import beat
from beat.eikonal import Source, solve_eikonal
from beat.reaction_eikonal import ExponentialFoot, FootCurrentODESolver, ODEOnlySolver

comm = MPI.COMM_WORLD
dx, dt = 0.4, 0.05  # mm, ms: R-E- has no diffusion, so a larger time step is fine
end_time = 50.0  # ms; 305.0 reproduces t_r50
CV_L, CV_T = 0.594, 0.324  # mm/ms, from scripts/reaction_eikonal_cv.json
# -

# +
wedge = beat.geometry.get_3D_wedge_geometry(comm, dx=dx)
geo = wedge.geometry
V = dolfinx.fem.functionspace(geo.mesh, ("P", 1))
layers = beat.geometry.wedge_layer_markers(V)
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
sources = [Source((wedge.cell_tags, tag), t0) for tag, t0 in wedge.stimuli.values()]
t_a = solve_eikonal(geo.mesh, (CV_L, CV_T, CV_T), (geo.f0, geo.s0, geo.n0), sources)
solver = ODEOnlySolver(FootCurrentODESolver(cells, t_a, ExponentialFoot()))
start = walltime.perf_counter()
solver.solve((0.0, end_time), dt)
if comm.rank == 0:
    print(f"R-E-, {end_time:g} ms: {walltime.perf_counter() - start:.0f} s")
# -

# The transmural profile along $y = 12.5$, $z = 2.5$ mm, with the layer boundaries:

# +
line_x = np.linspace(0, 25, 501)
line = np.column_stack([line_x, np.full_like(line_x, 12.5), np.full_like(line_x, 2.5)])
v_line = scifem.evaluate_function(solver.v, line).ravel()
if comm.rank == 0:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(line_x, v_line)
    for boundary in (6.25, 17.5):
        ax.axvline(boundary, color="k", linestyle=":")
    ax.set_xlabel("x (transmural, mm)")
    ax.set_ylabel(f"$v$ at t = {end_time:g} ms (mV)")
    fig.savefig("reaction_eikonal_wedge_repolarization.png", bbox_inches="tight")
# -

# ## Precomputed results
#
# Top: $v$ along the transmural line at $t_{r50}$ for R-D (100 µm) and the R-E models (400 µm),
# with the normalised spatial derivative (dotted). Bottom: histograms of the absolute deviation
# $|v_{R-D} - v_{R-E}|$ at $t_{r50}$ over the mid-plane (paper Eq. 34). Right: action potentials
# at the centres of the endocardial and M layers. As in the paper, R-E$^+$ matches R-D closely,
# while R-E$^-$ has steps at the layer boundaries and larger deviations.
#
# ![Fig. 5](../docs/_static/reaction_eikonal_fig5.png)
