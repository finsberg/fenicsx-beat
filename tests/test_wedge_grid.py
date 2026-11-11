"""Round trip of the wedge scripts' structured-grid storage (serial and mpirun -n 2)."""

import sys
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import reaction_eikonal_wedge_grid as grid  # noqa: E402

import beat  # noqa: E402


def test_grid_round_trip_between_wedge_and_bath_meshes():
    comm = MPI.COMM_WORLD
    dx = 0.5
    wedge = beat.geometry.get_3D_wedge_geometry(comm, dx=dx)
    V = dolfinx.fem.functionspace(wedge.geometry.mesh, ("P", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] + 10.0 * x[1] + 100.0 * x[2])
    array = grid.gather_to_grid(u, dx)
    if comm.rank == 0:
        assert array.shape == grid.grid_shape(dx)
        assert not np.isnan(array).any()
    array = comm.bcast(array, root=0)

    bath = beat.geometry.get_3D_wedge_bath_geometry(comm, dx=dx, margin=1.0)
    w = dolfinx.fem.Function(dolfinx.fem.functionspace(bath.geometry.mesh, ("P", 1)))
    grid.scatter_from_grid(w, array, dx)
    x = w.function_space.tabulate_dof_coordinates()
    inside = np.all((x > -1e-9) & (x < np.array([25.0, 25.0, 5.0]) + 1e-9), axis=1)
    assert np.allclose(w.x.array[inside], x[inside, 0] + 10.0 * x[inside, 1] + 100.0 * x[inside, 2])
    assert np.all(w.x.array[~inside] == 0.0)
