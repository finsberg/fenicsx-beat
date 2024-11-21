from __future__ import annotations

import logging

from mpi4py import MPI
import basix
import dolfinx
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def evaluate_function(
    u: dolfinx.fem.Function, points: npt.ArrayLike[np.float64], broadcast=True
) -> npt.NDArray[np.float64]:
    """Evaluate a function at a set of points.

    Args:
        u: The function to evaluate.
        points: The points to evaluate the function at.
        broadcast: If True, the values will be broadcasted to all processes.

    Returns:
        The values of the function evaluated at the points.

    """
    mesh = u.function_space.mesh
    u.x.scatter_forward()
    comm = mesh.comm
    points = np.array(points, dtype=np.float64)
    assert (
        len(points.shape) == 2
    ), f"Expected points to have shape (num_points, dim), got {points.shape}"
    num_points = points.shape[0]
    extra_dim = 3 - mesh.geometry.dim

    # Append zeros to points if the mesh is not 3D
    if extra_dim > 0:
        points = np.hstack((points, np.zeros((points.shape[0], extra_dim))))

    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    # Find cells whose bounding-box collide with the the points
    potential_colliding_cells = dolfinx.geometry.compute_collisions_points(
        bb_tree, points
    )
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh, potential_colliding_cells, points
    )
    points_on_proc = []
    cells = []
    indices = []
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
            indices.append(i)
    indices = np.array(indices, dtype=np.int32)
    points_on_proc = np.array(points_on_proc, dtype=np.float64).reshape(-1, 3)
    cells = np.array(cells, dtype=np.int32)

    values = u.eval(points_on_proc, cells)
    if broadcast:
        bs = u.function_space.dofmap.index_map_bs
        # Create array to store values and fill with -inf
        # to ensure that all points are included in the allreduce
        # with op=MPI.MAX
        u_out = np.ones((num_points, bs), dtype=np.float64) * -np.inf
        # Fill in values for points on this process
        u_out[indices, :] = values
        # Now loop over all processes and find the maximum value
        for i in range(num_points):
            if bs > 1:
                # If block size is larger than 1, loop over blocks
                for j in range(bs):
                    u_out[i, j] = comm.allreduce(u_out[i, j], op=MPI.MAX)
            else:
                u_out[i] = comm.allreduce(u_out[i], op=MPI.MAX)

        return u_out
    else:
        return values


def local_project(
    v: dolfinx.fem.Function,
    V: dolfinx.fem.FunctionSpace,
    u: dolfinx.fem.Function | None = None,
) -> dolfinx.fem.Function | None:
    """Element-wise projection using LocalSolver

    Parameters
    ----------
    v : dolfinx.fem.Function
        Function to be projected
    V : dolfinx.fem.FunctionSpace
        Function space to project into
    u : dolfinx.fem.Function | None, optional
        Optional function to save the projected function, by default None

    Returns
    -------
    dolfinx.fem.Function | None
        The projected function
    """
    if u is None:
        U = dolfinx.fem.Function(V)
    else:
        U = u

    if v.x.array.size == U.x.array.size:
        U.x.array[:] = v.x.array[:]
        return U

    expr = dolfinx.fem.Expression(v, V.element.interpolation_points())
    U.interpolate(expr)
    return U


def parse_element(
    space_string: str, mesh: dolfinx.mesh.Mesh, dim: int
) -> basix.ufl._ElementBase:
    """
    Parse a string representation of a basix element family
    """
    family_str, degree_str = space_string.split("_")
    kwargs = {"degree": int(degree_str), "cell": mesh.ufl_cell().cellname()}
    if dim > 1:
        if family_str in ["Quadrature", "Q", "Quad"]:
            kwargs["value_shape"] = (dim,)
        else:
            kwargs["shape"] = (dim,)

    if family_str in ["Lagrange", "P", "CG"]:
        el = basix.ufl.element(
            family=basix.ElementFamily.P, discontinuous=False, **kwargs
        )
    elif family_str in ["Discontinuous Lagrange", "DG", "dP"]:
        el = basix.ufl.element(
            family=basix.ElementFamily.P, discontinuous=True, **kwargs
        )

    elif family_str in ["Quadrature", "Q", "Quad"]:
        el = basix.ufl.quadrature_element(scheme="default", **kwargs)
    else:
        families = list(basix.ElementFamily.__members__.keys())
        msg = f"Unknown element family: {family_str}, available families: {families}"
        raise ValueError(msg)
    return el


def space_from_string(
    space_string: str, mesh: dolfinx.mesh.Mesh, dim: int = 1
) -> dolfinx.fem.functionspace:
    """
    Constructed a finite elements space from a string
    representation of the space

    Arguments
    ---------
    space_string : str
        A string on the form {family}_{degree} which
        determines the space. Example 'Lagrange_1'.
    mesh : df.Mesh
        The mesh
    dim : int
        1 for scalar space, 3 for vector space.

    Returns
    -------
    df.FunctionSpace
        The function space
    """
    el = parse_element(space_string, mesh, dim)
    return dolfinx.fem.functionspace(mesh, el)
