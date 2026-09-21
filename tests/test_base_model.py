import logging

from mpi4py import MPI

import dolfinx
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
