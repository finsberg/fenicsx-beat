from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import beat.eikonal
from beat.eikonal import FAR, compile_kernel, vertex_connectivity

# (row, column) of the 6 stored components of the symmetric metric
COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    return compile_kernel(cache_dir=tmp_path_factory.mktemp("kernel"), comm=MPI.COMM_SELF)


def simplices(n, lengths):
    """Vertex coordinates (always 3D) and cells of a serial dolfinx triangle/tetrahedron mesh"""
    if len(n) == 2:
        mesh = dolfinx.mesh.create_rectangle(MPI.COMM_SELF, [np.zeros(2), np.array(lengths)], n)
    else:
        mesh = dolfinx.mesh.create_box(MPI.COMM_SELF, [np.zeros(3), np.array(lengths)], n)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    x = np.ascontiguousarray(V.tabulate_dof_coordinates(), dtype=np.float64)
    return x, np.ascontiguousarray(V.dofmap.list, dtype=np.int32)


def metric(fibers, vl, vt):
    """V^{-1} = I / vt^2 + (1 / vl^2 - 1 / vt^2) f f^T per cell, as (xx, yy, zz, xy, xz, yz)"""
    f = fibers / np.linalg.norm(fibers, axis=1, keepdims=True)
    vl, vt = np.broadcast_to(vl, len(f)), np.broadcast_to(vt, len(f))
    full = np.eye(3) / vt[:, None, None] ** 2
    full = full + (1 / vl**2 - 1 / vt**2)[:, None, None] * np.einsum("ci,cj->cij", f, f)
    return np.ascontiguousarray(np.stack([full[:, a, b] for a, b in COMPONENTS], axis=1))


def fixed_sources(x, where, t0=0.0):
    fixed = where(x).astype(np.uint8)
    return fixed, np.where(fixed, t0, FAR)


def run(kernel, x, cells, M, fixed, t, tol=1e-6, max_iter=1_000_000):
    t = t.copy()
    stats = np.zeros(2, dtype=np.int64)
    conn = vertex_connectivity(len(t), cells)
    seeds = np.flatnonzero(fixed).astype(np.int32)
    status = kernel.run(x, cells, M, conn, fixed, t, seeds, tol, max_iter, stats)
    return status, t, stats


def brute_force_update(v, x, cells, M, t, n=100):
    """min over the cells of v of t(y) + ||y - x_v||_{V^-1}, y sampled densely (n per edge) on
    the face (or edge) opposite v, with t linear on it"""
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    if cells.shape[1] == 4:
        keep = i + j <= n
        lam = np.column_stack([i[keep], j[keep], n - i[keep] - j[keep]]) / n
    else:
        lam = np.column_stack([np.arange(n + 1), n - np.arange(n + 1)]) / n
    best = np.inf
    for c in np.flatnonzero((cells == v).any(axis=1)):
        others = cells[c][cells[c] != v]
        m = np.empty((3, 3))
        for q, (a, b) in enumerate(COMPONENTS):
            m[a, b] = m[b, a] = M[c, q]
        d = lam @ x[others] - x[v]
        best = min(best, (lam @ t[others] + np.sqrt(np.einsum("ni,ij,nj->n", d, m, d))).min())
    return best


def test_vertex_connectivity_two_triangles():
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    v2c_ptr, v2c, nb_ptr, nb = vertex_connectivity(4, cells)
    assert v2c_ptr.tolist() == [0, 1, 3, 5, 6]
    assert sorted(v2c[v2c_ptr[1] : v2c_ptr[2]].tolist()) == [0, 1]
    assert nb_ptr.tolist() == [0, 2, 5, 8, 10]
    assert sorted(nb[nb_ptr[1] : nb_ptr[2]].tolist()) == [0, 2, 3]
    assert all(a.dtype == np.int32 for a in (v2c_ptr, v2c, nb_ptr, nb))


@pytest.mark.parametrize("n, lengths", [((10, 4), (5.0, 2.0)), ((10, 4, 4), (5.0, 2.0, 2.0))])
def test_planar_front_is_exact(kernel, n, lengths):
    x, cells = simplices(n, lengths)
    M = metric(np.tile([1.0, 0.0, 0.0], (len(cells), 1)), 0.6, 0.3)
    fixed, t0 = fixed_sources(x, lambda x: np.isclose(x[:, 0], 0.0))
    status, t, _ = run(kernel, x, cells, M, fixed, t0)
    assert status == 0
    np.testing.assert_allclose(t, x[:, 0] / 0.6, rtol=0, atol=1e-12)


def test_colliding_fronts_are_exact(kernel):
    x, cells = simplices((10, 4, 4), (5.0, 2.0, 2.0))
    M = metric(np.tile([1.0, 0.0, 0.0], (len(cells), 1)), 0.6, 0.3)
    fixed = (np.isclose(x[:, 0], 0.0) | np.isclose(x[:, 0], 5.0)).astype(np.uint8)
    t0 = np.where(np.isclose(x[:, 0], 0.0), 0.0, np.where(fixed, 2.0, FAR))
    status, t, _ = run(kernel, x, cells, M, fixed, t0)
    assert status == 0
    np.testing.assert_allclose(
        t,
        np.minimum(x[:, 0] / 0.6, 2.0 + (5.0 - x[:, 0]) / 0.6),
        atol=1e-12,
    )


@pytest.mark.parametrize("n, lengths", [((12, 12), (3.0, 3.0)), ((6, 6, 4), (3.0, 3.0, 2.0))])
def test_solution_is_the_fixed_point_of_the_local_update(kernel, n, lengths):
    x, cells = simplices(n, lengths)
    rng = np.random.default_rng(0)
    fibers = rng.normal(size=(len(cells), 3))
    if len(n) == 2:
        fibers[:, 2] = 0.0
    M = metric(fibers, rng.uniform(0.4, 0.9, len(cells)), rng.uniform(0.2, 0.35, len(cells)))
    fixed, t0 = fixed_sources(x, lambda x: np.isclose(x[:, 0], 0.0) & np.isclose(x[:, 1], 0.0))
    tol = 1e-6
    status, t, _ = run(kernel, x, cells, M, fixed, t0, tol=tol)
    assert status == 0
    assert (t < FAR).all()
    for v in np.flatnonzero(fixed == 0):
        brute = brute_force_update(v, x, cells, M, t)
        assert t[v] <= brute + 10 * tol, (v, t[v], brute)  # never above the local minimum
        assert t[v] >= brute - 1e-3, (v, t[v], brute)  # sampling overestimates by < 1e-3


def test_openmp_build_matches_serial(kernel, tmp_path):
    omp = compile_kernel(openmp=True, num_threads=2, cache_dir=tmp_path, comm=MPI.COMM_SELF)
    assert omp.max_threads() >= 1
    x, cells = simplices((10, 10, 6), (4.0, 4.0, 3.0))
    M = metric(np.tile([0.0, 1.0, 0.0], (len(cells), 1)), 0.6, 0.3)
    fixed, t0 = fixed_sources(x, lambda x: np.all(np.isclose(x, [2.0, 2.0, 1.5]), axis=1))
    _, t_serial, _ = run(kernel, x, cells, M, fixed, t0)
    status, t_omp, _ = run(omp, x, cells, M, fixed, t0)
    assert status == 0
    # Jacobi form: every pass reads t before writing, so threads cannot change the result
    np.testing.assert_allclose(t_omp, t_serial, rtol=0, atol=1e-12)


def test_kernel_is_cached(tmp_path):
    k1 = compile_kernel(cache_dir=tmp_path, comm=MPI.COMM_SELF)
    mtime = k1.library.stat().st_mtime_ns
    k2 = compile_kernel(cache_dir=tmp_path, comm=MPI.COMM_SELF)
    assert k2.library == k1.library
    assert k2.library.stat().st_mtime_ns == mtime
    assert k1.library.name == beat.eikonal.KERNEL_LIBRARY


def test_max_iter_reports_non_convergence(kernel):
    x, cells = simplices((8, 8), (4.0, 4.0))
    M = metric(np.tile([1.0, 0.0, 0.0], (len(cells), 1)), 0.6, 0.3)
    fixed, t0 = fixed_sources(x, lambda x: np.all(np.isclose(x, 0.0), axis=1))
    status, _, stats = run(kernel, x, cells, M, fixed, t0, max_iter=1)
    assert status == 1
    assert stats[0] == 1


def test_num_threads_requires_openmp():
    with pytest.raises(ValueError, match="openmp=True"):
        compile_kernel(num_threads=2, comm=MPI.COMM_SELF)
