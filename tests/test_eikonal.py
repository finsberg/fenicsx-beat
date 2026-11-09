from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pytest

import beat
from beat.eikonal import EikonalSolver, Source, solve_eikonal

CT = dolfinx.mesh.CellType
VL, VT = 0.6, 0.3


def box(comm, h, lengths=(10.0, 10.0, 6.0), cell_type=CT.tetrahedron):
    n = [max(1, round(L / h)) for L in lengths]
    if len(lengths) == 2:
        return dolfinx.mesh.create_rectangle(comm, [np.zeros(2), np.array(lengths)], n, cell_type)
    return dolfinx.mesh.create_box(comm, [np.zeros(3), np.array(lengths)], n, cell_type)


def fibers(mesh, degrees=0.0):
    gdim = mesh.geometry.dim
    th = np.deg2rad(degrees)
    f = np.array([np.cos(th), np.sin(th), 0.0])[:gdim]
    s = np.array([-np.sin(th), np.cos(th), 0.0])[:gdim]
    out = [dolfinx.fem.Constant(mesh, f), dolfinx.fem.Constant(mesh, s)]
    if gdim == 3:
        out.append(dolfinx.fem.Constant(mesh, np.array([0.0, 0.0, 1.0])))
    return out


def speeds(mesh):
    return (VL, VT, VT)[: mesh.geometry.dim]


def owned(u):
    return u.x.array[: u.function_space.dofmap.index_map.size_local]


def interpolate(V, f):
    u = dolfinx.fem.Function(V)
    u.interpolate(f)
    return u


def max_error(u, f):
    return u.function_space.mesh.comm.allreduce(
        np.abs(owned(u) - owned(interpolate(u.function_space, f))).max(),
        op=MPI.MAX,
    )


def gather_sorted(u):
    """(coordinates, values) of all owned dofs on every rank, sorted by coordinate"""
    V = u.function_space
    n = V.dofmap.index_map.size_local
    comm = V.mesh.comm
    x = np.vstack(comm.allgather(V.tabulate_dof_coordinates()[:n]))
    y = np.concatenate(comm.allgather(u.x.array[:n]))
    order = np.lexsort(np.round(x, 9).T[::-1])
    return x[order], y[order]


@pytest.mark.parametrize("dim", [2, 3])
def test_planar_front_is_exact(dim):
    mesh = box(
        MPI.COMM_WORLD,
        1.0,
        (10.0, 4.0, 3.0)[:dim],
        CT.triangle if dim == 2 else CT.tetrahedron,
    )
    t = solve_eikonal(mesh, speeds(mesh), fibers(mesh), [Source(lambda x: np.isclose(x[0], 0.0))])
    assert t.name == "activation_time"
    assert max_error(t, lambda x: x[0] / VL) < 1e-12


def test_two_fronts_collide_at_the_right_place():
    mesh = box(MPI.COMM_WORLD, 1.0)
    sources = [
        Source(lambda x: np.isclose(x[0], 0.0), 0.0),
        Source(lambda x: np.isclose(x[0], 10.0), 5.0),
    ]
    t = solve_eikonal(mesh, speeds(mesh), fibers(mesh), sources)
    assert max_error(t, lambda x: np.minimum(x[0] / VL, 5.0 + (10.0 - x[0]) / VL)) < 1e-12


def metric_ball_rms(h, cell_type=CT.tetrahedron):
    mesh = box(MPI.COMM_WORLD, h, cell_type=cell_type)
    th = np.deg2rad(30.0)
    f = np.array([np.cos(th), np.sin(th), 0.0])
    Vinv = np.eye(3) / VT**2 + (1 / VL**2 - 1 / VT**2) * np.outer(f, f)
    x0 = np.array([5.0, 5.0, 3.0])

    def dist(x):
        d = x - x0[:, None]
        return np.sqrt(np.einsum("in,ij,jn->n", d, Vinv, d))

    rho = 2.5
    t = solve_eikonal(
        mesh,
        speeds(mesh),
        fibers(mesh, 30.0),
        [Source(lambda x: dist(x) <= rho + 1e-10)],
    )
    err = owned(t) - owned(interpolate(t.function_space, lambda x: np.maximum(0.0, dist(x) - rho)))
    comm = mesh.comm
    n = t.function_space.dofmap.index_map.size_global
    return np.sqrt(comm.allreduce((err**2).sum(), op=MPI.SUM) / n)


def test_metric_ball_converges():
    errors = [metric_ball_rms(h) for h in (1.0, 0.5, 0.25)]
    orders = [np.log2(e1 / e2) for e1, e2 in zip(errors[:-1], errors[1:])]
    assert all(o >= 0.8 for o in orders), (errors, orders)
    assert errors[-1] < 0.6


def test_velocity_and_fiber_representations_agree():
    mesh = box(MPI.COMM_WORLD, 1.0)
    src = [Source(lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))]
    ref = solve_eikonal(mesh, speeds(mesh), fibers(mesh, 20.0), src)
    DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    P1 = dolfinx.fem.functionspace(mesh, ("P", 1))
    P1v = dolfinx.fem.functionspace(mesh, ("P", 1, (3,)))
    vl_dg0 = interpolate(DG0, lambda x: np.full(x.shape[1], VL))
    vt_p1 = interpolate(P1, lambda x: np.full(x.shape[1], VT))
    vn_const = dolfinx.fem.Constant(mesh, VT)
    th = np.deg2rad(20.0)
    f_p1 = interpolate(P1v, lambda x: np.tile([[np.cos(th)], [np.sin(th)], [0.0]], (1, x.shape[1])))
    f0, s0, n0 = fibers(mesh, 20.0)
    other = solve_eikonal(mesh, (vl_dg0, vt_p1, vn_const), (f_p1, s0, n0), src)
    np.testing.assert_allclose(other.x.array, ref.x.array, atol=1e-12)
    ureg = beat.units.ureg
    pint_t = solve_eikonal(
        mesh,
        (0.6 * ureg("m/s"), 30.0 * ureg("cm/s"), VT),
        fibers(mesh, 20.0),
        src,
    )
    np.testing.assert_allclose(pint_t.x.array, ref.x.array, atol=1e-12)


def test_non_unit_fibers_are_normalised():
    mesh = box(MPI.COMM_WORLD, 1.0)
    src = [Source(lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))]
    f0, s0, n0 = fibers(mesh, 20.0)
    ref = solve_eikonal(mesh, speeds(mesh), (f0, s0, n0), src)
    scaled = solve_eikonal(mesh, speeds(mesh), (3.0 * f0, 0.5 * s0, n0), src)
    np.testing.assert_allclose(scaled.x.array, ref.x.array, atol=1e-12)


def test_meshtags_source_and_overlapping_sources():
    mesh = box(MPI.COMM_WORLD, 1.0)
    fdim = mesh.topology.dim - 1
    facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 0.0))
    tags = dolfinx.mesh.meshtags(mesh, fdim, facets, np.full(len(facets), 7, dtype=np.int32))
    by_tags = solve_eikonal(mesh, speeds(mesh), fibers(mesh), [Source((tags, 7), 1.0)])
    assert max_error(by_tags, lambda x: 1.0 + x[0] / VL) < 1e-12
    overlap = solve_eikonal(
        mesh,
        speeds(mesh),
        fibers(mesh),
        [Source((tags, 7), 3.0), Source(lambda x: np.isclose(x[0], 0.0), 1.0)],
    )
    assert max_error(overlap, lambda x: 1.0 + x[0] / VL) < 1e-12


def test_no_sources_raises():
    mesh = box(MPI.COMM_WORLD, 2.0)
    with pytest.raises(ValueError, match="No source vertices"):
        solve_eikonal(mesh, speeds(mesh), fibers(mesh), [Source(lambda x: x[0] < -1.0)])


def test_wrong_number_of_directions_raises():
    mesh = box(MPI.COMM_WORLD, 2.0)
    with pytest.raises(ValueError, match="one velocity and one direction per"):
        EikonalSolver(mesh, (VL, VT), fibers(mesh)[:2])


def two_disjoint_squares(comm):
    if comm.rank == 0:
        base = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
        V = dolfinx.fem.functionspace(base, ("P", 1))
        x = V.tabulate_dof_coordinates()[:, :2]
        c = V.dofmap.list.astype(np.int64)
        nodes = np.vstack([x, x + [2.0, 0.0]])
        cells = np.vstack([c, c + len(x)])
    else:
        nodes, cells = np.empty((0, 2)), np.empty((0, 3), dtype=np.int64)
    element = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    return dolfinx.mesh.create_mesh(comm, cells, element, nodes)


def test_unreached_vertices():
    mesh = two_disjoint_squares(MPI.COMM_WORLD)
    src = [Source(lambda x: np.isclose(x[0], 0.0))]
    with pytest.raises(ValueError, match="not reached"):
        solve_eikonal(mesh, (VL, VT), fibers(mesh), src)
    with pytest.warns(RuntimeWarning, match="not reached"):
        t = solve_eikonal(mesh, (VL, VT), fibers(mesh), src, allow_unreached=True)
    x = t.function_space.tabulate_dof_coordinates()
    assert np.isinf(t.x.array[x[:, 0] > 1.5]).all()
    assert np.isfinite(t.x.array[x[:, 0] < 1.5]).all()


def test_ghosts_equal_owners():
    mesh = box(MPI.COMM_WORLD, 0.5)
    t = solve_eikonal(
        mesh,
        speeds(mesh),
        fibers(mesh, 30.0),
        [Source(lambda x: (x[0] < 1e-9) & (x[1] < 1e-9))],
    )
    before = t.x.array.copy()
    t.x.scatter_forward()
    np.testing.assert_array_equal(t.x.array, before)


def test_partition_independence():
    # A corner source: on several ranks the source is only in ghost cells of some ranks
    def source(x):
        return (x[0] < 1e-9) & (x[1] < 1e-9) & (x[2] < 1e-9)

    world = box(MPI.COMM_WORLD, 0.5)
    serial = box(MPI.COMM_SELF, 0.5)
    tw = solve_eikonal(world, speeds(world), fibers(world, 30.0), [Source(source)])
    ts = solve_eikonal(serial, speeds(serial), fibers(serial, 30.0), [Source(source)])
    xw, yw = gather_sorted(tw)
    xs, ys = gather_sorted(ts)
    np.testing.assert_allclose(xw, xs, atol=1e-12)
    # Different partitions visit vertices in a different order; values agree to ~tol
    np.testing.assert_allclose(yw, ys, atol=1e-5)


def test_reuse_matches_fresh_solver():
    mesh = box(MPI.COMM_WORLD, 1.0)
    a = [Source(lambda x: np.isclose(x[0], 0.0))]
    b = [Source(lambda x: (x[0] > 10 - 1e-9) & (x[1] > 10 - 1e-9), 2.0)]
    solver = EikonalSolver(mesh, speeds(mesh), fibers(mesh, 30.0))
    solver.solve(a)
    reused = solver.solve(b)
    fresh = EikonalSolver(mesh, speeds(mesh), fibers(mesh, 30.0)).solve(b)
    np.testing.assert_array_equal(reused.x.array, fresh.x.array)
    assert solver.stats["rounds"] >= 1 and solver.stats["updates"] > 0


def test_openmp_matches_default():
    mesh = box(MPI.COMM_WORLD, 0.5)
    src = [Source(lambda x: (x[0] < 1e-9) & (x[1] < 1e-9))]
    ref = solve_eikonal(mesh, speeds(mesh), fibers(mesh, 30.0), src)
    for n in (1, 2):
        omp = solve_eikonal(mesh, speeds(mesh), fibers(mesh, 30.0), src, openmp=True, num_threads=n)
        np.testing.assert_allclose(omp.x.array, ref.x.array, atol=1e-12)
