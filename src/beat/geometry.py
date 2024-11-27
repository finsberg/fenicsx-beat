from typing import NamedTuple

from mpi4py import MPI

import dolfinx
import numpy as np


class Geometry(NamedTuple):
    mesh: dolfinx.mesh.Mesh
    ffun: dolfinx.mesh.MeshTags | None = None
    markers: dict[str, tuple[int, int]] | None = None
    f0: dolfinx.fem.Constant | dolfinx.fem.Function | None = None
    s0: dolfinx.fem.Constant | dolfinx.fem.Function | None = None
    n0: dolfinx.fem.Constant | dolfinx.fem.Function | None = None


def get_2D_slab_microstructure(
    mesh: dolfinx.mesh.Mesh,
    transverse: bool = False,
) -> tuple[dolfinx.fem.Constant, dolfinx.fem.Constant]:
    """
    Get the microstructure of a 2D slab

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh object
    transverse : bool, optional
        Whether the fiber in the slab is transverse, by default False

    Returns
    -------
    tuple[dolfinx.fem.Constant, dolfinx.fem.Constant]
        The fiber and sheet directions
    """
    if transverse:
        f0 = dolfinx.fem.Constant(mesh, (0.0, 1.0))
        s0 = dolfinx.fem.Constant(mesh, (1.0, 0.0))
    else:
        f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0))
        s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0))

    return f0, s0


def get_3D_slab_microstructure(
    mesh: dolfinx.mesh.Mesh,
    transverse: bool = False,
) -> tuple[dolfinx.fem.Constant, dolfinx.fem.Constant, dolfinx.fem.Constant]:
    """
    Get the microstructure of a 3D slab

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh object
    transverse : bool, optional
        Whether the fiber in the slab is transverse, by default False

    Returns
    -------
    tuple[dolfinx.fem.Constant, dolfinx.fem.Constant, dolfinx.fem.Constant]
        The fiber, sheet, and normal directions
    """
    if transverse:
        f0 = dolfinx.fem.Constant(mesh, (0.0, 0.0, 1.0))
        s0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
        n0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    else:
        f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
        s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
        n0 = dolfinx.fem.Constant(mesh, (0.0, 0.0, 1.0))

    return f0, s0, n0


def get_2D_slab_mesh(
    comm: MPI.Intracomm,
    dx: float,
    Lx: float,
    Ly: float,
    cell_type: dolfinx.mesh.CellType = dolfinx.mesh.CellType.triangle,
    dtype: type = np.float64,
):
    """
    Generate a 2D slab mesh

    Parameters
    ----------
    comm : MPI.Intracomm
        The MPI communicator
    dx : float
        The mesh resolution
    Lx : float
        The length of the slab in the x-direction
    Ly : float
        The length of the slab in the y-direction
    cell_type : dolfinx.mesh.CellType, optional
        The celltype, by default dolfinx.mesh.CellType.triangle
    dtype : type, optional
        Data type, by default np.float64

    Returns
    -------
    dolfinx.mesh.Mesh
        The mesh object

    """
    nx = int(np.rint((Lx / dx)))
    ny = int(np.rint((Ly / dx)))
    return dolfinx.mesh.create_rectangle(
        comm=comm,
        points=[np.array([0.0, 0.0]), np.array([Lx, Ly])],
        n=[nx, ny],
        cell_type=cell_type,
        dtype=dtype,
    )


def get_3D_slab_mesh(
    comm: MPI.Intracomm,
    dx: float,
    Lx: float,
    Ly: float,
    Lz: float,
    cell_type: dolfinx.mesh.CellType = dolfinx.cpp.mesh.CellType.tetrahedron,
    dtype: type = np.float64,
):
    nx = int(np.rint((Lx / dx)))
    ny = int(np.rint((Ly / dx)))
    nz = int(np.rint((Lz / dx)))
    return dolfinx.mesh.create_box(
        comm=comm,
        points=[np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        n=[nx, ny, nz],
        cell_type=cell_type,
        dtype=dtype,
    )


def get_3D_slab_geometry(
    comm: MPI.Intracomm,
    dx: float,
    Lx: float,
    Ly: float,
    Lz: float,
    cell_type: dolfinx.mesh.CellType = dolfinx.cpp.mesh.CellType.tetrahedron,
    dtype: type = np.float64,
    transverse: bool = False,
) -> Geometry:
    """Generate a 3D slab geometry

    Parameters
    ----------
    comm : MPI.Intracomm
        The MPI communicator
    dx : float
        The mesh resolution
    Lx : float
        The length of the slab in the x-direction
    Ly : float
        The length of the slab in the y-direction
    Lz : float
        The length of the slab in the z-direction
    cell_type : dolfinx.mesh.CellType, optional
        The celltype, by default dolfinx.mesh.CellType.tetrahedron
    dtype : type, optional
        Data type, by default np.float64
    transverse : bool, optional
        Whether the fiber in the slab is transverse, by default False

    Returns
    -------
    Geometry
        The geometry object
    """
    mesh = get_3D_slab_mesh(comm, dx, Lx, Ly, Lz, cell_type, dtype)
    f0, s0, n0 = get_3D_slab_microstructure(mesh, transverse)
    return Geometry(mesh=mesh, f0=f0, s0=s0, n0=n0)


def get_2D_slab_geometry(
    comm: MPI.Intracomm,
    dx: float,
    Lx: float,
    Ly: float,
    cell_type: dolfinx.mesh.CellType = dolfinx.mesh.CellType.triangle,
    dtype: type = np.float64,
    transverse: bool = False,
) -> Geometry:
    """Generate a 2D slab geometry

    Parameters
    ----------
    comm : MPI.Intracomm
        The MPI communicator
    dx : float
        The mesh resolution
    Lx : float
        The length of the slab in the x-direction
    Ly : float
        The length of the slab in the y-direction
    cell_type : dolfinx.mesh.CellType, optional
        The celltype, by default dolfinx.mesh.CellType.triangle
    dtype : type, optional
        Data type, by default np.float64
    transverse : bool, optional
        Whether the fiber in the slab is transverse, by default False

    Returns
    -------
    Geometry
        The geometry object
    """
    mesh = get_2D_slab_mesh(comm, dx, Lx, Ly, cell_type, dtype)
    f0, s0 = get_2D_slab_microstructure(mesh, transverse)
    return Geometry(mesh=mesh, f0=f0, s0=s0)
