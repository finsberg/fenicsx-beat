from __future__ import annotations

import logging

import basix
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl

logger = logging.getLogger(__name__)


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


def parse_element(space_string: str, mesh: dolfinx.mesh.Mesh, dim: int) -> basix.ufl._ElementBase:
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
        el = basix.ufl.element(family=basix.ElementFamily.P, discontinuous=False, **kwargs)
    elif family_str in ["Discontinuous Lagrange", "DG", "dP"]:
        el = basix.ufl.element(family=basix.ElementFamily.P, discontinuous=True, **kwargs)

    elif family_str in ["Quadrature", "Q", "Quad"]:
        el = basix.ufl.quadrature_element(scheme="default", **kwargs)
    else:
        families = list(basix.ElementFamily.__members__.keys())
        msg = f"Unknown element family: {family_str}, available families: {families}"
        raise ValueError(msg)
    return el


def space_from_string(
    space_string: str,
    mesh: dolfinx.mesh.Mesh,
    dim: int = 1,
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


def expand_layer(
    V: dolfinx.fem.FunctionSpace,
    ft: dolfinx.mesh.MeshTags,
    endo_marker: int,
    epi_marker: int,
    endo_size: float,
    epi_size: float,
    output_mid_marker: int = 0,
    output_endo_marker: int = 1,
    output_epi_marker: int = 2,
) -> dolfinx.fem.Function:
    """Expand the endo and epi markers to the rest of the mesh
    with a given size

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        The function space for your return function
    ft : dolfinx.mesh.MeshTags
        The facet tags
    endo_marker : int
        The endocardium marker
    epi_marker : int
        The epicardium marker
    endo_size : float
        The endocardium size
    epi_size : float
        The epicardium size
    output_mid_marker : int, optional
        The marker to be set for the midwall, by default 0
    output_endo_marker : int, optional
        The marker to be set for the endocardium, by default 1
    output_epi_marker : int, optional
        The marker to be set for the epicardium, by default 2


    Returns
    -------
    dolfinx.fem.Function
        The expanded markers as a function in ``V``
    """
    logger.info("Expanding endo and epi markers to the rest of the mesh")
    logger.debug(
        f"endo_marker: {endo_marker}, epi_marker: {epi_marker}\n"
        f"endo_size: {endo_size}, epi_size: {epi_size}\n"
        f"Output markers: endo: {output_endo_marker}, "
        f"epi: {output_epi_marker} mid: {output_mid_marker}",
    )
    # Find the rest of the laplace solutions

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    mesh = V.mesh

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=mesh)
    L = v * dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)) * ufl.dx(domain=mesh)

    u_endo = dolfinx.fem.Function(V)
    u_endo.x.array[:] = 0
    u_epi = dolfinx.fem.Function(V)
    u_epi.x.array[:] = 1

    endo_dofs = dolfinx.fem.locate_dofs_topological(V, ft.dim, ft.find(endo_marker))
    epi_dofs = dolfinx.fem.locate_dofs_topological(V, ft.dim, ft.find(epi_marker))

    # Apply Dirichlet BC on the outer boundary
    bcs = [
        dolfinx.fem.dirichletbc(0.0, endo_dofs, V),
        dolfinx.fem.dirichletbc(1.0, epi_dofs, V),
    ]

    problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        # form_compiler_options=form_compiler_options,
        # jit_options=jit_options,
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_norm_type": "unpreconditioned",
            "ksp_atol": 1e-15,
            "ksp_rtol": 1e-10,
            "ksp_max_it": 10_000,
            "ksp_error_if_not_converged": False,
        },
    )
    uh = problem.solve()

    arr = uh.x.array.copy()
    uh.x.array[:] = output_mid_marker
    uh.x.array[arr <= endo_size] = output_endo_marker
    uh.x.array[arr >= 1 - epi_size] = output_epi_marker
    uh.name = "endo_epi"

    with dolfinx.io.XDMFFile(mesh.comm, "endo_epi_lv.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(uh)

    return uh


def expand_layer_biv(
    V: dolfinx.fem.FunctionSpace,
    ft: dolfinx.mesh.MeshTags,
    endo_lv_marker: int,
    endo_rv_marker: int,
    epi_marker: int,
    endo_size: float,
    epi_size: float,
    output_mid_marker: int = 0,
    output_endo_marker: int = 1,
    output_epi_marker: int = 2,
) -> dolfinx.fem.Function:
    """Expand the endo and epi markers to the rest of the mesh
    with a given size

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        The function space for your return function
    ft : dolfinx.mesh.MeshTags
        The facet tags
    endo_lv_marker : int
        The LV endocardium marker
    endo_rv_marker : int
        The LV endocardium marker
    epi_marker : int
        The epicardium marker
    endo_size : float
        The endocardium size
    epi_size : float
        The epicardium size
    output_mid_marker : int, optional
        The marker to be set for the midwall, by default 0
    output_endo_marker : int, optional
        The marker to be set for the endocardium, by default 1
    output_epi_marker : int, optional
        The marker to be set for the epicardium, by default 2


    Returns
    -------
    dolfinx.fem.Function
        The expanded markers as a function in ``V``
    """
    logger.info("Expanding endo and epi markers to the rest of the mesh")
    logger.debug(
        f"endo_lv_marker: {endo_lv_marker}, "
        f"endo_rv_marker: {endo_rv_marker}, "
        f"epi_marker: {epi_marker}\n"
        f"endo_size: {endo_size}, epi_size: {epi_size}\n"
        f"Output markers: endo: {output_endo_marker}, "
        f"epi: {output_epi_marker} mid: {output_mid_marker}",
    )
    # Find the rest of the laplace solutions

    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_norm_type": "unpreconditioned",
        "ksp_atol": 1e-15,
        "ksp_rtol": 1e-10,
        "ksp_max_it": 10_000,
        "ksp_error_if_not_converged": False,
    }

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    mesh = V.mesh

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=mesh)
    L = v * dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)) * ufl.dx(domain=mesh)

    u_endo = dolfinx.fem.Function(V)
    u_endo.x.array[:] = 0
    u_epi = dolfinx.fem.Function(V)
    u_epi.x.array[:] = 1

    endo_lv_dofs = dolfinx.fem.locate_dofs_topological(V, ft.dim, ft.find(endo_lv_marker))
    endo_rv_dofs = dolfinx.fem.locate_dofs_topological(V, ft.dim, ft.find(endo_rv_marker))
    epi_dofs = dolfinx.fem.locate_dofs_topological(V, ft.dim, ft.find(epi_marker))

    lv_problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        bcs=[
            dolfinx.fem.dirichletbc(0.0, endo_lv_dofs, V),
            dolfinx.fem.dirichletbc(1.0, epi_dofs, V),
        ],
        petsc_options=petsc_options,
    )
    uh_lv = lv_problem.solve()

    rv_problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        bcs=[
            dolfinx.fem.dirichletbc(0.0, endo_rv_dofs, V),
            dolfinx.fem.dirichletbc(1.0, epi_dofs, V),
        ],
        petsc_options=petsc_options,
    )
    uh_rv = rv_problem.solve()

    # In BiV we have have one epi and two endo solutions
    # We take the minimum of the two endo solutions
    arr = np.min([uh_rv.x.array, uh_lv.x.array], axis=0)
    uh_rv.x.array[:] = output_mid_marker
    uh_rv.x.array[arr <= endo_size] = output_endo_marker
    uh_rv.x.array[arr >= 1 - epi_size] = output_epi_marker
    uh_rv.name = "endo_epi"

    with dolfinx.io.XDMFFile(mesh.comm, "endo_epi_biv.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(uh_rv)

    return uh_rv
