from typing import NamedTuple

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
    comm,
    dx,
    Lx,
    Ly,
    cell_type=dolfinx.cpp.mesh.CellType.triangle,
    dtype=np.float64,
):
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
    comm,
    dx,
    Lx,
    Ly,
    Lz,
    cell_type=dolfinx.cpp.mesh.CellType.tetrahedron,
    dtype=np.float64,
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
    comm,
    dx,
    Lx,
    Ly,
    Lz,
    cell_type=dolfinx.cpp.mesh.CellType.tetrahedron,
    dtype=np.float64,
    transverse=False,
) -> Geometry:
    mesh = get_3D_slab_mesh(comm, dx, Lx, Ly, Lz, cell_type, dtype)
    f0, s0, n0 = get_3D_slab_microstructure(mesh, transverse)
    return Geometry(mesh=mesh, f0=f0, s0=s0, n0=n0)


def get_2D_slab_geometry(
    comm,
    dx,
    Lx,
    Ly,
    Lz,
    cell_type=dolfinx.cpp.mesh.CellType.triangle,
    dtype=np.float64,
    transverse=False,
) -> Geometry:
    mesh = get_2D_slab_mesh(comm, dx, Lx, Ly, cell_type, dtype)
    f0, s0 = get_2D_slab_microstructure(mesh, transverse)
    return Geometry(mesh=mesh, f0=f0, s0=s0)
