from __future__ import annotations

import logging
import basix
import dolfinx

# import numpy as np

logger = logging.getLogger(__name__)


def spaces_equal(V: dolfinx.fem.FunctionSpace, W: dolfinx.fem.FunctionSpace) -> bool:
    """Check if two function spaces are equal

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        First function space
    W : dolfinx.fem.FunctionSpace
        Second function space

    Returns
    -------
    bool
        True if the function spaces are equal, False otherwise
    """
    return V.element.signature() == W.element.signature()


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
    if spaces_equal(v.function_space, V):
        if u is not None:
            u.x.array[:] = v.x.array[:]
            return u
        else:
            return v.copy()

    if u is None:
        U = dolfinx.fem.Function(V)
    else:
        U = u

    expr = dolfinx.fem.Expression(v, V.element.interpolation_points())
    U.interpolate(expr)
    return u


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
    """
    el = parse_element(space_string, mesh, dim)
    return dolfinx.fem.functionspace(mesh, el)
