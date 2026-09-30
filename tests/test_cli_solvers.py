from mpi4py import MPI

import dolfinx
import pytest
from cli_helpers import minimal_config_dict

import beat
from beat.cli.cellmodel import build_cell_model, build_region_markers
from beat.cli.config import Config, ConfigError
from beat.cli.geometry import build_conductivity, build_geometry
from beat.cli.solvers import build_ode, build_pde, check_backends_available, ode_space
from beat.cli.stimulus import build_stimuli

try:
    # Irksome's own import-time check (irksome/ufl/deriv.py:check_irksome_import_order)
    # requires that irksome be imported before any UFL form is ever processed elsewhere in
    # the process (it registers a new UFL node type, which UFL's MultiFunction dispatch
    # cannot accommodate once it has already run once). In production this is guaranteed by
    # check_backends_available() running before build_pde()/build_ode(), but within this
    # single test file the theta-PDE tests (test_default_backends and friends) process UFL
    # forms first, so warm up irksome here at collection time -- before any test body runs --
    # the same way tests/test_irksome_odesolver.py does at its own module top.
    import irksome  # noqa: F401
except ImportError:
    pass


@pytest.fixture
def tmp_path(tmp_path):
    """Override pytest's built-in ``tmp_path`` so every rank shares rank 0's directory.

    See tests/test_cli_cellmodel.py for the full rationale: build_cell_model's on-disk codegen
    cache assumes a shared path across ranks, which pytest's per-process ``tmp_path`` is not
    under ``mpirun -n 2 pytest``.
    """
    return MPI.COMM_WORLD.bcast(tmp_path, root=0)


def assemble(tmp_path, solver=None, cell=None):
    data = minimal_config_dict(tmp_path)
    if solver:
        data["solver"].update(solver)
    if cell:
        data["cell"].update(cell)
    conf = Config.model_validate(data)
    check_backends_available(conf.solver)
    geo = build_geometry(conf.geometry, MPI.COMM_WORLD)
    time = dolfinx.fem.Constant(geo.mesh, 0.0)
    M = build_conductivity(conf.ep, geo, conf.geometry.fibers)
    stim = build_stimuli(conf.stimulus, geo, conf.ep, time, "mm", conf.solver.t_end_ms())
    pde = build_pde(conf.solver, geo.mesh, time, M, stim, C_m=0.01, monitor=None)
    model = build_cell_model(conf.cell, tmp_path / "c", MPI.COMM_WORLD)
    markers = build_region_markers(conf.cell, geo, ode_space(geo.mesh))
    ode = build_ode(conf.solver, model, markers, pde, time, monitor=None)
    return pde, ode


def test_default_backends(tmp_path):
    pde, ode = assemble(tmp_path)
    assert isinstance(pde, beat.MonodomainModel)
    assert isinstance(ode, beat.odesolver.DolfinODESolver)


def test_multi_region_uses_multi_solver(tmp_path):
    _, ode = assemble(
        tmp_path,
        cell={"layers": {"method": "transmural", "endo_markers": ["X0"], "epi_marker": "X1"}},
    )
    assert isinstance(ode, beat.odesolver.DolfinMultiODESolver)


def test_petsc_options_reach_the_pde(tmp_path):
    pde, _ = assemble(
        tmp_path,
        solver={
            "petsc_options": {"ksp_type": "gmres"},
            "pde": {"type": "theta", "linear_solver": "iterative"},
        },
    )
    assert pde.parameters["petsc_options"]["ksp_type"] == "gmres"
    assert pde.parameters["petsc_options"]["pc_type"] == "hypre"


def test_missing_irksome_gives_install_hint(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "irksome", None)
    with pytest.raises(ConfigError, match="git\\+https://github.com/firedrakeproject/Irksome"):
        assemble(tmp_path, solver={"ode": {"type": "irksome"}})


def test_irksome_backends(tmp_path):
    pytest.importorskip("irksome")
    pde, ode = assemble(tmp_path, solver={"pde": {"type": "irksome"}, "ode": {"type": "irksome"}})
    assert isinstance(pde, beat.IrksomeMonodomainModel)
    assert isinstance(ode, beat.IrksomeODESolver)


def test_external_operator_backend(tmp_path):
    pytest.importorskip("dolfinx_external_operator")
    _, ode = assemble(tmp_path, solver={"ode": {"type": "external_operator"}})
    assert isinstance(ode, beat.ExternalOperatorODESolver)
