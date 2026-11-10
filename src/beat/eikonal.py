"""Activation times from the anisotropic eikonal equation

.. math::

    \\sqrt{\\nabla t_a^T V \\nabla t_a} = 1, \\qquad t_a = t_0 \\text{ on the sources},

with the velocity tensor :math:`V = v_l^2 f f^T + v_t^2 s s^T + v_n^2 n n^T`, solved at the
vertices of the mesh with the Fast Iterative Method (FIM) of Jeong & Whitaker (2008) and
Fu, Kirby & Whitaker (2013), in the Jacobi form. The FIM kernel is C, compiled ahead of time with
:func:`beat.c_backend.build_library` (rank 0 compiles into the hash-keyed cache, every rank loads
it), optionally with OpenMP.
"""

from __future__ import annotations

import ctypes
import numbers
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import numpy as np
import numpy.typing as npt
import pint
import ufl

from . import c_backend

KERNEL_PATH = Path(__file__).with_name("eikonal_kernel.c")
KERNEL_LIBRARY = "libbeat_eikonal.so"
#: Value of vertices not (yet) reached; the kernel uses the same number
FAR = 1e300


class EikonalKernel:
    """The compiled FIM kernel (see ``eikonal_kernel.c``)"""

    def __init__(self, library: Path, num_threads: int | None = None) -> None:
        self.library = library
        self.num_threads = num_threads or 0
        self._lib = ctypes.CDLL(str(library))
        p = ctypes.c_void_p
        fim = self._lib.beat_eikonal_fim
        argtypes: list[Any] = [ctypes.c_int32, ctypes.c_int32, *([p] * 9)]
        argtypes += [p, ctypes.c_int32, ctypes.c_double, ctypes.c_int64, ctypes.c_int32, p]
        fim.argtypes = argtypes
        fim.restype = ctypes.c_int
        self._lib.beat_eikonal_max_threads.restype = ctypes.c_int

    def max_threads(self) -> int:
        """The number of OpenMP threads the kernel would use (1 without OpenMP)"""
        return int(self._lib.beat_eikonal_max_threads())

    def run(
        self,
        x: npt.NDArray[np.float64],
        cells: npt.NDArray[np.int32],
        metric: npt.NDArray[np.float64],
        connectivity: tuple[npt.NDArray[np.int32], ...],
        fixed: npt.NDArray[np.uint8],
        t: npt.NDArray[np.float64],
        seeds: npt.NDArray[np.int32],
        tol: float,
        max_iter: int,
        stats: npt.NDArray[np.int64],
    ) -> int:
        """Run FIM from ``seeds``, updating ``t`` in place; 0 if converged, 1 if not"""
        n = len(t)
        if x.shape != (n, 3) or fixed.shape != (n,) or cells.shape[1] not in (3, 4):
            raise ValueError("Inconsistent array shapes passed to the eikonal kernel")
        arrays: list[Any] = [x, cells, metric, *connectivity, fixed, t, seeds, stats]
        dtypes: list[Any] = [np.float64, np.int32, np.float64, *([np.int32] * 4)]
        dtypes += [np.uint8, np.float64, np.int32, np.int64]
        for a, dtype in zip(arrays, dtypes):
            if a.dtype != dtype or not a.flags.c_contiguous:
                raise ValueError(f"Eikonal kernel arrays must be C-contiguous {np.dtype(dtype)}")
        ptr = [a.ctypes.data for a in arrays]
        status = self._lib.beat_eikonal_fim(
            n,
            cells.shape[1],
            *ptr[:9],
            ptr[9],
            len(seeds),
            tol,
            max_iter,
            self.num_threads,
            ptr[10],
        )
        if status == 2:
            raise MemoryError("The eikonal kernel ran out of memory")
        return int(status)


def compile_kernel(
    *,
    openmp: bool = False,
    num_threads: int | None = None,
    cc: str | None = None,
    cflags: Sequence[str] = c_backend.DEFAULT_CFLAGS,
    cache_dir: str | Path | None = None,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> EikonalKernel:
    """Compile (or load from the cache) the FIM kernel on rank 0 and load it on every rank

    Parameters
    ----------
    openmp : bool, optional
        Compile with -fopenmp and run the update phases in parallel, by default False
    num_threads : int | None, optional
        The number of OpenMP threads, by default OMP_NUM_THREADS or the OpenMP default
    cc, cflags, cache_dir, comm
        As in :func:`beat.c_backend.compile_scheme`
    """
    if isinstance(cflags, str):
        raise TypeError(f"cflags must be a sequence of strings, not a str: {cflags!r}")
    if num_threads is not None and num_threads < 1:
        raise ValueError(f"num_threads must be at least 1, got {num_threads}")
    if num_threads is not None and not openmp:
        raise ValueError("num_threads is only used with OpenMP; pass openmp=True")
    cache = Path(cache_dir) if cache_dir is not None else c_backend.default_cache_dir()
    flags = tuple(cflags) + (c_backend.OPENMP_FLAGS if openmp else ())
    library = c_backend.build_library(
        KERNEL_PATH.read_text(),
        cc=cc or c_backend.default_compiler(),
        cflags=flags,
        cache_dir=cache.resolve(),
        comm=comm,
        library_name=KERNEL_LIBRARY,
        source_name="eikonal_kernel.c",
        description="eikonal kernel",
    )
    return EikonalKernel(library, num_threads)


def vertex_connectivity(
    num_vertices: int,
    cells: npt.NDArray[np.int32],
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Vertex-to-cell and vertex-to-vertex maps of ``cells`` as CSR arrays

    Returns ``(v2c_ptr, v2c, nb_ptr, nb)``: the cells of vertex ``v`` are
    ``v2c[v2c_ptr[v]:v2c_ptr[v + 1]]`` and its neighbours (vertices sharing a cell, without ``v``
    itself) are ``nb[nb_ptr[v]:nb_ptr[v + 1]]``.
    """
    cells = np.asarray(cells)
    flat = cells.ravel()
    order = np.argsort(flat, kind="stable")
    v2c = (order // cells.shape[1]).astype(np.int32)
    v2c_ptr = np.zeros(num_vertices + 1, dtype=np.int32)
    np.cumsum(np.bincount(flat, minlength=num_vertices), out=v2c_ptr[1:])
    i, j = np.triu_indices(cells.shape[1], 1)
    edges = np.concatenate([cells[:, [a, b]] for a, b in zip(i, j)]).astype(np.int64)
    edges = np.vstack([edges, edges[:, ::-1]])
    key = np.unique(edges[:, 0] * num_vertices + edges[:, 1])
    nb_ptr = np.zeros(num_vertices + 1, dtype=np.int32)
    np.cumsum(np.bincount(key // num_vertices, minlength=num_vertices), out=nb_ptr[1:])
    return v2c_ptr, v2c, nb_ptr, (key % num_vertices).astype(np.int32)


# Sub-simplices of each supported cell, in basix reference vertex numbering (P1 dof order):
# quadrilateral v0(0,0) v1(1,0) v2(0,1) v3(1,1) -> two triangles along v0-v3; hexahedron
# v0(000) v1(100) v2(010) v3(110) v4(001) v5(101) v6(011) v7(111) -> six tetrahedra along v0-v7
# (Kuhn split). Splits need not be conforming between cells: each local update is a minimum
# over candidate simplices, so the scheme stays monotone and consistent.
_SUBSIMPLICES: dict[dolfinx.mesh.CellType, npt.NDArray[np.int64]] = {
    dolfinx.mesh.CellType.triangle: np.array([[0, 1, 2]]),
    dolfinx.mesh.CellType.tetrahedron: np.array([[0, 1, 2, 3]]),
    dolfinx.mesh.CellType.quadrilateral: np.array([[0, 1, 3], [0, 2, 3]]),
    dolfinx.mesh.CellType.hexahedron: np.array(
        [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]],
    ),
}
# (row, column) of the 6 stored components of the symmetric 3x3 metric
_METRIC_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class Source:
    """Vertices where :math:`t_a = t_0`

    ``marker`` is either a function ``x -> bool array`` of the coordinates (``x`` has shape
    ``(3, num_points)``, as in :func:`dolfinx.fem.locate_dofs_geometrical`), or a pair
    ``(meshtags, value)``: all vertices of the entities tagged with ``value``.
    """

    marker: (
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]]
        | tuple[dolfinx.mesh.MeshTags, int]
    )
    t0: float = 0.0


def _subsimplices(mesh: Any) -> npt.NDArray[np.int64]:
    if len(mesh.topology.entity_types[-1]) > 1:
        raise NotImplementedError("The eikonal solver does not support mixed-topology meshes")
    cell_type = mesh.topology.cell_type
    if cell_type not in _SUBSIMPLICES:
        raise ValueError(f"The eikonal solver does not support {cell_type.name} cells")
    return _SUBSIMPLICES[cell_type]


def _speed(value: Any, mesh_unit: str) -> Any:
    if isinstance(value, pint.Quantity):
        value = float(value.to(f"{mesh_unit}/ms").magnitude)
    if isinstance(value, numbers.Real) and value <= 0:
        raise ValueError(f"Velocities must be positive, got {value}")
    return value


def _num_local_cells(mesh: dolfinx.mesh.Mesh) -> int:
    imap = mesh.topology.index_map(mesh.topology.dim)
    return imap.size_local + imap.num_ghosts


def cell_metric(
    mesh: dolfinx.mesh.Mesh,
    velocities: Sequence[Any],
    fibers: Sequence[Any],
    mesh_unit: str = "mm",
) -> npt.NDArray[np.float64]:
    """:math:`V^{-1} = \\sum_i v_i^{-2} e_i e_i^T` at the midpoint of every local cell

    ``velocities`` are floats (mesh units per ms), pint quantities, Constants or Functions;
    ``fibers`` are the directions (Constants, Functions or UFL vectors), normalised here. Give
    one velocity and one direction per geometric dimension. Returns ``(num_cells, 6)``
    components ``(xx, yy, zz, xy, xz, yz)``, with ``zz = 1`` on 2D meshes.
    """
    gdim = mesh.geometry.dim
    if len(velocities) != gdim or len(fibers) != gdim:
        raise ValueError(
            f"Give one velocity and one direction per geometric dimension ({gdim}), got "
            f"{len(velocities)} velocities and {len(fibers)} directions",
        )
    Minv = None
    speeds = []
    for v, f in zip(velocities, fibers):
        if f is None:
            raise ValueError("All directions must be given (got None)")
        e = f / ufl.sqrt(ufl.dot(f, f))
        speed = _speed(v, mesh_unit)
        speeds.append(speed)
        term = ufl.outer(e, e) / speed**2
        Minv = term if Minv is None else Minv + term
    assert Minv is not None
    X = basix.geometry(mesh.basix_cell()).mean(axis=0, keepdims=True)
    num_cells = _num_local_cells(mesh)
    values = dolfinx.fem.Expression(Minv, X).eval(mesh, np.arange(num_cells, dtype=np.int32))
    values = np.asarray(values).reshape(num_cells, gdim, gdim)
    # The metric only sees v**2, so check the sign of the velocities themselves
    cells = np.arange(num_cells, dtype=np.int32)
    v_values = dolfinx.fem.Expression(
        ufl.as_vector(
            [
                dolfinx.fem.Constant(mesh, float(v)) if isinstance(v, numbers.Real) else v
                for v in speeds
            ],
        ),
        X,
    ).eval(mesh, cells)
    metric = np.zeros((num_cells, 6))
    metric[:, 2] = 1.0
    for k, (a, b) in enumerate(_METRIC_COMPONENTS):
        if a < gdim and b < gdim:
            metric[:, k] = values[:, a, b]
    positive = (np.asarray(v_values) > 0).all() and (metric[:, :3] > 0).all()
    ok = bool(np.isfinite(metric).all() and positive)
    # Decide collectively, otherwise one rank raises while the others block in a collective
    if not mesh.comm.allreduce(ok, op=MPI.LAND):
        raise ValueError("Velocities must be positive and finite everywhere")
    return metric


class EikonalSolver:
    """Solve the eikonal equation for activation times on ``mesh``

    The mesh, the metric :math:`V^{-1}` and the connectivity are set up once, so
    :meth:`solve` can be called for many source sets.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh (triangles, tetrahedra, quadrilaterals or hexahedra; any geometry degree).
        Only its vertices are used: the result is a P1 Function.
    velocities : Sequence
        Conduction velocities along ``fibers`` (mesh units per ms): floats, pint quantities,
        Constants or (DG0/P1/...) Functions, evaluated at cell midpoints
    fibers : Sequence
        The directions (e.g. ``(f0, s0, n0)``), normalised at the cell midpoints
    mesh_unit : str, optional
        Unit of the mesh coordinates, used to convert pint velocities, by default "mm"
    openmp, num_threads, cc, cflags, cache_dir
        Kernel compilation, see :func:`compile_kernel`
    """

    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        velocities: Sequence[Any],
        fibers: Sequence[Any],
        *,
        mesh_unit: str = "mm",
        openmp: bool = False,
        num_threads: int | None = None,
        cc: str | None = None,
        cflags: Sequence[str] = c_backend.DEFAULT_CFLAGS,
        cache_dir: str | Path | None = None,
    ) -> None:
        split = _subsimplices(mesh)
        self.mesh = mesh
        self.V = dolfinx.fem.functionspace(mesh, ("P", 1))
        num_cells = _num_local_cells(mesh)
        dofs = self.V.dofmap.list[:num_cells]
        self._cells = np.ascontiguousarray(
            dofs[:, split].reshape(-1, split.shape[1]),
            dtype=np.int32,
        )
        parent = np.repeat(np.arange(num_cells), split.shape[0])
        self._metric = np.ascontiguousarray(
            cell_metric(mesh, velocities, fibers, mesh_unit)[parent],
        )
        # P1 dof coordinates are the vertex coordinates, also for higher-order geometry
        self._x = np.ascontiguousarray(self.V.tabulate_dof_coordinates(), dtype=np.float64)
        self._connectivity = vertex_connectivity(len(self._x), self._cells)
        self._kernel = compile_kernel(
            openmp=openmp,
            num_threads=num_threads,
            cc=cc,
            cflags=cflags,
            cache_dir=cache_dir,
            comm=mesh.comm,
        )
        self._work = dolfinx.fem.Function(self.V)
        self.stats: dict[str, int] = {}

    def _source_dofs(self, source: Source) -> npt.NDArray[np.int32]:
        if callable(source.marker):
            return dolfinx.fem.locate_dofs_geometrical(self.V, source.marker)
        tags, value = source.marker
        self.mesh.topology.create_connectivity(tags.dim, self.mesh.topology.dim)
        return dolfinx.fem.locate_dofs_topological(self.V, tags.dim, tags.find(value))

    def _min_exchange(self, t: npt.NDArray[np.float64]) -> None:
        """Every rank's copy of a vertex becomes the minimum over all ranks (PETSc has no
        MIN_VALUES, so take the maximum of -t)"""
        w = self._work
        w.x.array[:] = -t
        w.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.MAX_VALUES, mode=PETSc.ScatterMode.REVERSE)
        w.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        t[:] = -w.x.array

    def solve(
        self,
        sources: Sequence[Source],
        *,
        tol: float = 1e-6,
        max_iter: int | None = None,
        allow_unreached: bool = False,
    ) -> dolfinx.fem.Function:
        """The activation time :math:`t_a` (P1 Function named ``"activation_time"``)

        Parameters
        ----------
        sources : Sequence[Source]
            Where the activation starts; where sources overlap the smallest ``t0`` wins
        tol : float, optional
            FIM convergence tolerance (ms), by default 1e-6
        max_iter : int | None, optional
            Maximum FIM passes per kernel call, by default 4 times the number of local vertices
            plus 1000
        allow_unreached : bool, optional
            Warn instead of raising if some vertices are not reached; they get ``inf``
        """
        if not (np.isfinite(tol) and tol > 0):
            raise ValueError(f"tol must be positive and finite, got {tol}")
        comm = self.mesh.comm
        n = len(self._x)
        n_owned = self.V.dofmap.index_map.size_local
        max_iter = max_iter if max_iter is not None else 4 * n + 1000
        t = np.full(n, FAR)
        for source in sources:
            dofs = self._source_dofs(source)
            t[dofs] = np.minimum(t[dofs], source.t0)
        self._min_exchange(t)
        fixed = (t < FAR).astype(np.uint8)
        if comm.allreduce(int(fixed[:n_owned].sum()), op=MPI.SUM) == 0:
            raise ValueError("No source vertices found")
        seeds = np.flatnonzero(fixed).astype(np.int32)
        stats = np.zeros(2, dtype=np.int64)
        rounds = 0
        while True:
            rounds += 1
            status = self._kernel.run(
                self._x,
                self._cells,
                self._metric,
                self._connectivity,
                fixed,
                t,
                seeds,
                tol,
                max_iter,
                stats,
            )
            if comm.allreduce(status, op=MPI.MAX) != 0:
                raise RuntimeError(f"The eikonal solver did not converge in {max_iter} passes")
            before = t.copy()
            self._min_exchange(t)
            seeds = np.flatnonzero(t < before - tol).astype(np.int32)
            if comm.allreduce(len(seeds), op=MPI.SUM) == 0:
                break
        self.stats = {
            "rounds": rounds,
            "passes": int(comm.allreduce(int(stats[0]), op=MPI.MAX)),
            "updates": int(comm.allreduce(int(stats[1]), op=MPI.SUM)),
        }
        unreached = comm.allreduce(int((t[:n_owned] >= FAR).sum()), op=MPI.SUM)
        if unreached:
            message = f"{unreached} vertices were not reached by any source"
            if not allow_unreached:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            t[t >= FAR] = np.inf
        t_a = dolfinx.fem.Function(self.V, name="activation_time")
        t_a.x.array[:] = t
        return t_a


def solve_eikonal(
    mesh: dolfinx.mesh.Mesh,
    velocities: Sequence[Any],
    fibers: Sequence[Any],
    sources: Sequence[Source],
    *,
    tol: float = 1e-6,
    max_iter: int | None = None,
    allow_unreached: bool = False,
    **solver_kwargs: Any,
) -> dolfinx.fem.Function:
    """Build an :class:`EikonalSolver` (``solver_kwargs`` go to it) and solve once"""
    solver = EikonalSolver(mesh, velocities, fibers, **solver_kwargs)
    return solver.solve(sources, tol=tol, max_iter=max_iter, allow_unreached=allow_unreached)
