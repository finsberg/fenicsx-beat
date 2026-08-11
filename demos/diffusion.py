# # Diffusion in a square domain with a stimulus in the lower left corner
#
# This is the simplest possible tissue-level demo: it solves the monodomain equation with the ionic
# current $I_{ion}$ switched off entirely, so only diffusion driven by a stimulus current remains,
#
# $$
# C_m \frac{\partial v}{\partial t} = \nabla \cdot (M \nabla v) + I_{stim},
# $$
#
# on a unit square domain, using $C_m = 1$ (the default) and a constant, scalar conductivity $M = 1$.
# The stimulus $I_{stim}$ is a constant value of $1$, applied only in the lower left corner of the
# domain for the first $2.5$ ms of a simulation with time step $\Delta t = 0.1$ ms. Since there is no
# `beat.MonodomainSplittingSolver` involved here (there is no ODE system to split against), this demo
# only exercises `beat.MonodomainModel` directly. See the
# [mathematical background](../docs/math_background.md) page for where this equation comes from, and
# the [FitzHugh–Nagumo demo](fitzhughnagumo.py) for the same equation with a non-trivial $I_{ion}$
# added back in.
#

from mpi4py import MPI
import numpy as np
import dolfinx

import pyvista
import beat
import ufl


comm = MPI.COMM_WORLD
N = 20
mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.mesh.CellType.triangle)


tol = 1.0e-10
L = 0.3


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
S = dolfinx.fem.Constant(mesh, 1.0)
I_s = beat.base_model.Stimulus(expr=S, dZ=dx, marker=S1_marker)

time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))

model = beat.MonodomainModel(time=time, mesh=mesh, M=1.0, I_s=I_s, dx=dx)
res = model.solve((0, 2.5), dt=0.1)

u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(res.state.function_space))
u_grid.point_data["u"] = res.state.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    u_plotter.screenshot("diffusion.png")
