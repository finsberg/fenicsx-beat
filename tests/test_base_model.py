import logging

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import beat


@pytest.fixture
def mesh():
    return dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        2,
        2,
        dolfinx.cpp.mesh.CellType.triangle,
    )


def _warnings_from_building_model(mesh, params, caplog):
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    with caplog.at_level(logging.WARNING, logger="beat.base_model"):
        beat.MonodomainModel(time=time, mesh=mesh, M=1.0, params=params)
    return [r.getMessage() for r in caplog.records if r.name == "beat.base_model"]


def test_unknown_params_key_warns(mesh, caplog):
    messages = _warnings_from_building_model(mesh, {"linear_solver_type": "direct"}, caplog)

    assert any("linear_solver_type" in m for m in messages), messages


def test_known_params_key_does_not_warn(mesh, caplog):
    messages = _warnings_from_building_model(mesh, {"theta": 1.0}, caplog)

    assert messages == []


def test_monodomain_v_is_the_state():
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 2, 2, dolfinx.cpp.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    model = beat.MonodomainModel(time=time, mesh=mesh, M=1.0)

    assert model.v is model.state


def test_ready_made_bcs_are_rejected(mesh):
    """A DirichletBC built outside the model is silently ignored during assembly, so it
    must be refused rather than accepted and disregarded."""
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    dofs = dolfinx.fem.locate_dofs_topological(
        V,
        mesh.topology.dim - 1,
        dolfinx.mesh.exterior_facet_indices(mesh.topology),
    )
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), dofs, V)

    with pytest.raises(TypeError, match="function space object it was built on"):
        beat.MonodomainModel(time=time, mesh=mesh, M=1.0, bcs=[bc])


def test_dirichlet_bcs_are_enforced_and_lifted():
    """v = x is an exact discrete solution of the steady problem, boundary and interior alike.

    With M = 1, no stimulus and v_ = x, the residual of every interior test function
    vanishes for v = x, so the whole P1 solution must reproduce it. Boundary values come
    out right only if the boundary rows are set, and the interior only if the boundary
    columns are lifted out of the right-hand side.
    """
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 8, 8, dolfinx.cpp.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    def v_equals_x_on_the_boundary(model):
        g = dolfinx.fem.Function(model.V)
        g.interpolate(lambda x: x[0])
        dofs = dolfinx.fem.locate_dofs_topological(
            model.V,
            mesh.topology.dim - 1,
            dolfinx.mesh.exterior_facet_indices(mesh.topology),
        )
        return [dolfinx.fem.dirichletbc(g, dofs)]

    model = beat.MonodomainModel(
        time=time,
        mesh=mesh,
        M=1.0,
        params=dict(theta=1.0),
        bcs=v_equals_x_on_the_boundary,
    )
    model.v_.interpolate(lambda x: x[0])
    model.solve((0.0, 0.1), dt=0.05)

    x = model.V.tabulate_dof_coordinates()
    assert np.allclose(model.v.x.array, x[:, 0])
