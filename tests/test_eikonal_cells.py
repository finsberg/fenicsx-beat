from types import SimpleNamespace

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pytest
from scipy.spatial import cKDTree

from beat.eikonal import _SUBSIMPLICES, Source, _subsimplices, solve_eikonal

CT = dolfinx.mesh.CellType
VL, VT = 0.6, 0.3
NUM_NODES = {
    (CT.triangle, 1): 3,
    (CT.quadrilateral, 1): 4,
    (CT.tetrahedron, 1): 4,
    (CT.hexahedron, 1): 8,
    (CT.triangle, 2): 6,
    (CT.quadrilateral, 2): 9,
    (CT.tetrahedron, 2): 10,
    (CT.hexahedron, 2): 27,
}


def tdim_of(cell_type):
    return 3 if cell_type in (CT.tetrahedron, CT.hexahedron) else 2


def base_data(cell_type, n, degree, bump=0.0):
    """Nodes and cells (on rank 0) of a structured mesh of the unit square/cube, scaled by 2;
    degree-2 nodes that are not vertices are displaced by ``bump * sin(pi x)`` in y."""
    tdim = tdim_of(cell_type)
    if MPI.COMM_WORLD.rank != 0:
        return np.empty((0, tdim)), np.empty((0, NUM_NODES[(cell_type, degree)]), dtype=np.int64)
    if tdim == 3:
        base = dolfinx.mesh.create_box(
            MPI.COMM_SELF,
            [np.zeros(3), 2 * np.ones(3)],
            [n] * 3,
            cell_type,
        )
    else:
        base = dolfinx.mesh.create_rectangle(
            MPI.COMM_SELF,
            [np.zeros(2), 2 * np.ones(2)],
            [n] * 2,
            cell_type,
        )
    V = dolfinx.fem.functionspace(base, ("P", degree))
    nodes = V.tabulate_dof_coordinates()[:, :tdim].copy()
    cells = V.dofmap.list.astype(np.int64)
    if degree == 2 and bump:
        V1 = dolfinx.fem.functionspace(base, ("P", 1))
        vertex = np.zeros(len(nodes), dtype=bool)
        vertex[cKDTree(nodes).query(V1.tabulate_dof_coordinates()[:, :tdim])[1]] = True
        nodes[~vertex, 1] += bump * np.sin(np.pi * nodes[~vertex, 0])
    return nodes, cells


def make_mesh(cell_type, nodes, cells, degree):
    element = basix.ufl.element("Lagrange", cell_type.name, degree, shape=(tdim_of(cell_type),))
    return dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, element, nodes)


def directions(mesh, degrees=30.0):
    th = np.deg2rad(degrees)
    gdim = mesh.geometry.dim
    out = [
        dolfinx.fem.Constant(mesh, np.array([np.cos(th), np.sin(th), 0.0])[:gdim]),
        dolfinx.fem.Constant(mesh, np.array([-np.sin(th), np.cos(th), 0.0])[:gdim]),
    ]
    if gdim == 3:
        out.append(dolfinx.fem.Constant(mesh, np.array([0.0, 0.0, 1.0])))
    return (VL, VT, VT)[:gdim], out


def corner(x):
    return np.all(np.isclose(x[:2], 0.0), axis=0)


def gather_sorted(u):
    V = u.function_space
    n = V.dofmap.index_map.size_local
    comm = V.mesh.comm
    x = np.vstack(comm.allgather(V.tabulate_dof_coordinates()[:n]))
    y = np.concatenate(comm.allgather(u.x.array[:n]))
    order = np.lexsort(np.round(x, 9).T[::-1])
    return x[order], y[order]


@pytest.mark.parametrize("cell_type", [CT.quadrilateral, CT.hexahedron])
def test_planar_front_is_exact(cell_type):
    mesh = make_mesh(cell_type, *base_data(cell_type, 4, 1), 1)
    speeds, dirs = directions(mesh, 0.0)
    t = solve_eikonal(mesh, speeds, dirs, [Source(lambda x: np.isclose(x[0], 0.0))])
    n = t.function_space.dofmap.index_map.size_local
    x = t.function_space.tabulate_dof_coordinates()[:n]
    err = mesh.comm.allreduce(np.abs(t.x.array[:n] - x[:, 0] / VL).max(), op=MPI.MAX)
    assert err < 1e-12


@pytest.mark.parametrize(
    "cell_type, simplex",
    [(CT.quadrilateral, CT.triangle), (CT.hexahedron, CT.tetrahedron)],
)
def test_split_cells_equal_simplex_mesh(cell_type, simplex):
    nodes, cells = base_data(cell_type, 4, 1)
    split = _SUBSIMPLICES[cell_type]
    simplices = cells[:, split].reshape(-1, split.shape[1])
    a = make_mesh(cell_type, nodes, cells, 1)
    b = make_mesh(simplex, nodes, simplices, 1)
    ta = solve_eikonal(a, *directions(a), [Source(corner)])
    tb = solve_eikonal(b, *directions(b), [Source(corner)])
    xa, ya = gather_sorted(ta)
    xb, yb = gather_sorted(tb)
    np.testing.assert_allclose(xa, xb, atol=1e-12)
    np.testing.assert_allclose(ya, yb, atol=1e-5)  # different partitions: equal to ~tol


@pytest.mark.parametrize(
    "cell_type",
    [CT.tetrahedron, CT.hexahedron, CT.triangle, CT.quadrilateral],
)
@pytest.mark.parametrize("bump", [0.0, 0.05])
def test_higher_order_geometry_uses_vertices(cell_type, bump):
    p1 = make_mesh(cell_type, *base_data(cell_type, 3, 1), 1)
    p2 = make_mesh(cell_type, *base_data(cell_type, 3, 2, bump), 2)
    t1 = solve_eikonal(p1, *directions(p1), [Source(corner)])
    t2 = solve_eikonal(p2, *directions(p2), [Source(corner)])
    x1, y1 = gather_sorted(t1)
    x2, y2 = gather_sorted(t2)
    np.testing.assert_allclose(x1, x2, atol=1e-12)
    np.testing.assert_allclose(y1, y2, atol=1e-5)  # different partitions: equal to ~tol


def test_hexahedral_metric_ball_converges():
    errors = []
    for n in (10, 20):
        mesh = dolfinx.mesh.create_box(
            MPI.COMM_WORLD,
            [np.zeros(3), np.array([10.0, 10.0, 6.0])],
            [n, n, round(0.6 * n)],
            CT.hexahedron,
        )
        th = np.deg2rad(30.0)
        f = np.array([np.cos(th), np.sin(th), 0.0])
        Vinv = np.eye(3) / VT**2 + (1 / VL**2 - 1 / VT**2) * np.outer(f, f)

        def dist(x):
            d = x - np.array([[5.0], [5.0], [3.0]])
            return np.sqrt(np.einsum("in,ij,jn->n", d, Vinv, d))

        t = solve_eikonal(mesh, *directions(mesh), [Source(lambda x: dist(x) <= 2.5 + 1e-10)])
        V = t.function_space
        nloc = V.dofmap.index_map.size_local
        exact = np.maximum(0.0, dist(V.tabulate_dof_coordinates()[:nloc].T) - 2.5)
        sq = mesh.comm.allreduce(((t.x.array[:nloc] - exact) ** 2).sum(), op=MPI.SUM)
        errors.append(np.sqrt(sq / V.dofmap.index_map.size_global))
    assert errors[1] < errors[0] and np.log2(errors[0] / errors[1]) >= 0.8, errors


def test_mixed_topology_raises():
    stub = SimpleNamespace(
        topology=SimpleNamespace(entity_types=[[], [CT.tetrahedron, CT.hexahedron]]),
    )
    with pytest.raises(NotImplementedError, match="mixed-topology"):
        _subsimplices(stub)


def test_interval_mesh_raises():
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 4)
    with pytest.raises(ValueError, match="does not support interval"):
        solve_eikonal(
            mesh,
            (VL,),
            (dolfinx.fem.Constant(mesh, np.array([1.0])),),
            [Source(lambda x: x[0] < 1e-9)],
        )
