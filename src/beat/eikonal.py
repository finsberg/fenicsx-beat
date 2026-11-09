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
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mpi4py import MPI

import numpy as np
import numpy.typing as npt

from . import c_backend

logger = logging.getLogger(__name__)

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
