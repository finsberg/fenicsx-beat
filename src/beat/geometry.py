from typing import NamedTuple

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt


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


#: Marker values of the transmural layers of the wedge
WEDGE_LAYERS: dict[str, int] = {"EPI": 1, "M": 2, "ENDO": 3}


class WedgeGeometry(NamedTuple):
    """The transmural wedge of Neic et al. (2017)

    ``geometry.markers`` and ``cell_tags`` hold the stimulus regions S1-S3, ``stimuli`` maps
    their names to ``(tag, t0)`` (``t0`` in ms), ``electrode`` is the recording site
    :math:`P_E` and ``length`` the transmural extent (mm).
    """

    geometry: Geometry
    cell_tags: dolfinx.mesh.MeshTags
    stimuli: dict[str, tuple[int, float]]
    electrode: npt.NDArray[np.float64]
    length: float


def wedge_layer_markers(V: dolfinx.fem.FunctionSpace, length: float = 25.0) -> dolfinx.fem.Function:
    """Layer markers (:data:`WEDGE_LAYERS`) on ``V``: epi for the 25% of the wall nearest
    :math:`x = 0`, then 45% M, then 30% endo"""
    markers = dolfinx.fem.Function(V, name="layers")

    def layer(x):
        xi = x[0] / length
        return np.where(
            xi < 0.25,
            WEDGE_LAYERS["EPI"],
            np.where(xi < 0.70, WEDGE_LAYERS["M"], WEDGE_LAYERS["ENDO"]),
        ).astype(float)

    markers.interpolate(layer)
    return markers


def get_3D_wedge_geometry(
    comm: MPI.Intracomm,
    dx: float = 0.4,
    *,
    length: float = 25.0,
    width: float = 25.0,
    thickness: float = 5.0,
    stimulus_size: float = 1.0,
    cell_type: dolfinx.mesh.CellType = dolfinx.mesh.CellType.tetrahedron,
    fiber_space: tuple[str, int] = ("DG", 0),
) -> WedgeGeometry:
    """The wedge benchmark of Neic et al. (2017), in mm

    The box :math:`[0, L] \\times [0, W] \\times [0, H]` has the wall across :math:`x` (epi at
    :math:`x = 0`, endo at :math:`x = L`). The fibres rotate in the :math:`yz`-plane from
    -60 degrees at the epicardium to +60 degrees at the endocardium; the sheet direction is
    :math:`e_x`. The stimuli are full-thickness corner columns: S1 at :math:`(0, W)` and S3 at
    :math:`(L, 0)` at :math:`t = 0`, S2 at :math:`(0, 0)` at :math:`t = 5` ms (read off the
    paper's Figs. 1 and 2). :math:`P_E` is the centre of the endocardial face.
    """
    mesh = get_3D_slab_mesh(comm, dx, length, width, thickness, cell_type)
    family, degree = fiber_space
    V = dolfinx.fem.functionspace(mesh, (family, degree, (3,)))

    def angle(x):
        return np.deg2rad(-60.0 + 120.0 * x[0] / length)

    f0 = dolfinx.fem.Function(V, name="f0")
    f0.interpolate(lambda x: np.vstack([np.zeros(x.shape[1]), np.cos(angle(x)), np.sin(angle(x))]))
    s0 = dolfinx.fem.Function(V, name="s0")
    s0.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros((2, x.shape[1]))]))
    n0 = dolfinx.fem.Function(V, name="n0")
    n0.interpolate(lambda x: np.vstack([np.zeros(x.shape[1]), np.sin(angle(x)), -np.cos(angle(x))]))

    s, eps = stimulus_size, 1e-10
    regions = {
        "S1": (1, 0.0, lambda x: (x[0] <= s + eps) & (x[1] >= width - s - eps)),
        "S2": (2, 5.0, lambda x: (x[0] <= s + eps) & (x[1] <= s + eps)),
        "S3": (3, 0.0, lambda x: (x[0] >= length - s - eps) & (x[1] <= s + eps)),
    }
    tdim = mesh.topology.dim
    indices, values = [], []
    for name, (tag, _, inside) in regions.items():
        cells = dolfinx.mesh.locate_entities(mesh, tdim, inside)
        if comm.allreduce(len(cells)) == 0:
            raise ValueError(f"No cells in the {name} stimulus region; use dx <= stimulus_size")
        indices.append(cells)
        values.append(np.full(len(cells), tag, dtype=np.int32))
    order = np.argsort(np.concatenate(indices))
    cell_tags = dolfinx.mesh.meshtags(
        mesh,
        tdim,
        np.concatenate(indices)[order].astype(np.int32),
        np.concatenate(values)[order],
    )
    geometry = Geometry(
        mesh=mesh,
        markers={name: (tag, tdim) for name, (tag, _, _) in regions.items()},
        f0=f0,
        s0=s0,
        n0=n0,
    )
    return WedgeGeometry(
        geometry=geometry,
        cell_tags=cell_tags,
        stimuli={name: (tag, t0) for name, (tag, t0, _) in regions.items()},
        electrode=np.array([length, width / 2, thickness / 2]),
        length=length,
    )
