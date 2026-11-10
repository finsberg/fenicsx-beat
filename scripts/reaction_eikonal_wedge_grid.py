"""Helpers shared by the bath scripts of the wedge benchmark (Neic et al. 2017, Fig. 7)

P1 fields are exchanged between the tissue mesh and the bath mesh (and between separate MPI runs)
as numpy arrays on the wedge's structured vertex grid: vertex (i, j, k) is at (i hx, j hy, k hz),
with the spacing of beat.geometry.get_3D_wedge_geometry.
"""

import numpy as np

import beat
from beat.conductivities import conductivity_tensor, default_conductivities

SIZES = np.array([25.0, 25.0, 5.0])
ELECTRODE = (25.0, 12.5, 2.5)  # P_E, centre of the endocardial face
SNAPSHOTS = (40.0, 320.0)  # ms, Fig. 7A and 7B
GRID_N = 101
NUM_LAYERS = 3  # augmentation layer, element rings (Bishop & Plank 2011)
MARGIN = 1.0  # mm of bath; 2 mm does not fit in memory at 100 um


def grid_spacing(dx):
    return SIZES / np.rint(SIZES / dx)


def grid_shape(dx):
    return tuple(int(n) + 1 for n in np.rint(SIZES / dx))


def _grid_dofs(V, dx, owned_only):
    h = grid_spacing(dx)
    x = V.tabulate_dof_coordinates()
    if owned_only:
        x = x[: V.dofmap.index_map.size_local]
    ijk = np.rint(x / h).astype(np.int64)
    on = np.all(np.abs(ijk * h - x) < 1e-6 * h.min(), axis=1)
    on &= np.all((ijk >= 0) & (ijk < np.array(grid_shape(dx))), axis=1)
    return np.flatnonzero(on), ijk[on]


def gather_to_grid(u, dx):
    """The values of the P1 function u at the tissue grid vertices, on rank 0 (None elsewhere)"""
    dofs, ijk = _grid_dofs(u.function_space, dx, owned_only=True)
    comm = u.function_space.mesh.comm
    parts = comm.gather((ijk, u.x.array[dofs].copy()), root=0)
    if comm.rank != 0:
        return None
    out = np.full(grid_shape(dx), np.nan)
    for ijk_rank, values in parts:
        out[tuple(ijk_rank.T)] = values
    return out


def scatter_from_grid(u, array, dx):
    """Set the P1 function u to array at the tissue grid vertices and to zero elsewhere

    array must be the same on every rank."""
    dofs, ijk = _grid_dofs(u.function_space, dx, owned_only=False)
    u.x.array[:] = 0.0
    u.x.array[dofs] = array[tuple(ijk.T)]


def sigma_tensors(f0):
    """Intra- and extracellular conductivity tensors of the wedge (S/m, not scaled by chi)"""
    g = default_conductivities("Neic2017_wedge_bath")

    def s(name):
        return g[name].to("S/m").magnitude

    return conductivity_tensor(s("g_il"), s("g_it"), f0), conductivity_tensor(
        s("g_el"),
        s("g_et"),
        f0,
    )


def bath_conductivity():
    return default_conductivities("Neic2017_wedge_bath")["g_b"].to("S/m").magnitude


def midplane_grid():
    """The 101 x 101 grid on z = 2.5 mm used for all maps"""
    gx, gy = np.meshgrid(np.linspace(0, 25, GRID_N), np.linspace(0, 25, GRID_N))
    return gx, gy, np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 2.5)])


def volume_conductor(comm, dx, margin):
    bath = beat.geometry.get_3D_wedge_bath_geometry(comm, dx=dx, margin=margin)
    sigma_i, sigma_e = sigma_tensors(bath.f0)
    markers = beat.geometry.WEDGE_BATH_MARKERS
    conductor = beat.pseudo_bidomain.VolumeConductor(
        bath.mesh,
        bath.cell_tags,
        markers["TISSUE"],
        markers["BATH"],
        sigma_i,
        sigma_e,
        bath_conductivity(),
    )
    return bath, conductor
