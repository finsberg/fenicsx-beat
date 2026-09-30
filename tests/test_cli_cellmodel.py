from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
from cli_helpers import minimal_config_dict

from beat.cli.cellmodel import build_cell_model, build_region_markers, state_index
from beat.cli.config import Config, ConfigError
from beat.cli.geometry import build_geometry


@pytest.fixture
def tmp_path(tmp_path):
    """Override pytest's built-in ``tmp_path`` so every rank shares rank 0's directory.

    Every test in this module exercises the on-disk, rank-0-only codegen/steady-state cache
    (`load_module`/`_steady_state`), which assumes ``cache_dir`` is a *shared* path across ranks
    (true in production, where it's an absolute path resolved from the config file). But under
    `mpirun -n 2 pytest`, each rank runs a fully independent pytest process with its own
    ``tmp_path`` fixture value, and that value is NOT rank-synchronized (verified directly: two
    ranks get e.g. ``/tmp/pytest-of-root/pytest-50/test_tmp0`` and ``.../pytest-51/...``). Note
    that a plain local re-bcast *inside* a helper (e.g. `conf_with`) is not enough: reassigning a
    local variable there doesn't change the caller's own ``tmp_path`` used e.g. for
    ``tmp_path / "cache"`` — the override has to happen at the fixture itself so every use of
    ``tmp_path`` in a test body sees the same, shared value.
    """
    return MPI.COMM_WORLD.bcast(tmp_path, root=0)


def conf_with(tmp_path, **cell):
    # `tmp_path` is already shared across ranks (see the `tmp_path` fixture override above), so
    # only rank 0 needs to do the actual `.ode`-file write that `minimal_config_dict` performs;
    # every rank then gets an identical `data` dict via `bcast` (which also synchronizes: no
    # rank can proceed to read the file before rank 0's write is visible to it).
    comm = MPI.COMM_WORLD
    data = minimal_config_dict(tmp_path) if comm.rank == 0 else None
    comm.barrier()
    data = comm.bcast(data, root=0)
    data["cell"].update(cell)
    return Config.model_validate(data)


def test_codegen_is_cached_by_content(tmp_path):
    conf = conf_with(tmp_path)
    m1 = build_cell_model(conf.cell, tmp_path / "cache", MPI.COMM_WORLD)
    files = sorted((tmp_path / "cache").glob("cell_model_*.py"))
    assert len(files) == 1
    build_cell_model(conf.cell, tmp_path / "cache", MPI.COMM_WORLD)
    assert sorted((tmp_path / "cache").glob("cell_model_*.py")) == files
    assert m1.state_names == ["v", "h"]
    assert m1.v_index == 0


def test_parameter_overrides_and_unknown_parameter(tmp_path):
    conf = conf_with(tmp_path, parameters={"tau_in": 0.5})
    model = build_cell_model(conf.cell, tmp_path / "cache", MPI.COMM_WORLD)
    idx = model.module["parameter_index"]("tau_in")
    assert model.parameters[0][idx] == pytest.approx(0.5)
    bad = conf_with(tmp_path, parameters={"tau_typo": 0.5})
    with pytest.raises(ConfigError, match="tau_typo"):
        build_cell_model(bad.cell, tmp_path / "cache", MPI.COMM_WORLD)


def test_unknown_v_name(tmp_path):
    with pytest.raises(ConfigError, match="V"):
        build_cell_model(conf_with(tmp_path, v_name="V").cell, tmp_path / "c", MPI.COMM_WORLD)


def test_state_index_unknown(tmp_path):
    model = build_cell_model(conf_with(tmp_path).cell, tmp_path / "c", MPI.COMM_WORLD)
    with pytest.raises(ConfigError, match="cai"):
        state_index(model, "cai")


def test_regions_get_their_own_parameters(tmp_path):
    conf = conf_with(
        tmp_path,
        layers={"method": "transmural", "endo_markers": ["X0"], "epi_marker": "X1"},
        regions={"endo": {"parameters": {"tau_out": 3.0}}},
    )
    model = build_cell_model(conf.cell, tmp_path / "c", MPI.COMM_WORLD)
    idx = model.module["parameter_index"]("tau_out")
    assert model.region_ids == {"mid": 0, "endo": 1, "epi": 2}
    assert model.parameters[1][idx] == pytest.approx(3.0)
    assert model.parameters[0][idx] == pytest.approx(6.0)


def test_transmural_markers_on_rectangle(tmp_path):
    conf = conf_with(
        tmp_path,
        layers={"method": "transmural", "endo_markers": ["X0"], "epi_marker": "X1"},
    )
    geo = build_geometry(conf.geometry, MPI.COMM_WORLD)
    V = dolfinx.fem.functionspace(geo.mesh, ("P", 1))
    markers = build_region_markers(conf.cell, geo, V)
    values = set(np.unique(markers.x.array).astype(int))
    all_values = set().union(*MPI.COMM_WORLD.allgather(values))
    assert all_values == {0, 1, 2}


@pytest.mark.skip_in_parallel
def test_steady_state_is_cached(tmp_path):
    conf = conf_with(tmp_path, steady_state={"num_beats": 1, "BCL": "2 ms", "dt": "0.1 ms"})
    build_cell_model(conf.cell, tmp_path / "c", MPI.COMM_WORLD)
    cached = list((tmp_path / "c" / "init_states").glob("tissue_*.npy"))
    assert len(cached) == 1


def test_steady_state_cache_key_includes_scheme(tmp_path):
    """The steady state is computed with the scheme-specific stepper (`cell.fun`), so changing
    `cell.scheme` (with everything else, including resolved parameters, identical) must not
    reuse a steady state cached under a different scheme."""
    ss = {"num_beats": 1, "BCL": "2 ms", "dt": "0.1 ms"}
    conf_default = conf_with(tmp_path, steady_state=ss)
    build_cell_model(conf_default.cell, tmp_path / "c", MPI.COMM_WORLD)
    files_default = set((tmp_path / "c" / "init_states").glob("tissue_*.npy"))
    assert len(files_default) == 1

    conf_other = conf_with(tmp_path, scheme="explicit_euler", steady_state=ss)
    build_cell_model(conf_other.cell, tmp_path / "c", MPI.COMM_WORLD)
    files_after = set((tmp_path / "c" / "init_states").glob("tissue_*.npy"))

    assert files_after != files_default
    assert len(files_after) == 2


def test_malformed_ode_file_raises_config_error(tmp_path):
    """A gotranx parse failure on rank 0 (during codegen) must surface as a `ConfigError` on
    every rank, not a raw traceback (and, under MPI, not a hang on the other ranks)."""
    conf = conf_with(tmp_path)
    bad_ode = conf.cell.ode_file.parent / "bad.ode"
    if MPI.COMM_WORLD.rank == 0:
        bad_ode.write_text("this is not valid gotran syntax !!! ===")
    MPI.COMM_WORLD.barrier()
    conf = conf_with(tmp_path, ode_file=str(bad_ode))
    with pytest.raises(ConfigError):
        build_cell_model(conf.cell, tmp_path / "c_bad", MPI.COMM_WORLD)


def _two_region_geometry(tag: str = "all"):
    """Unit square with cell markers A (x < 0.5, value 1) and B (x >= 0.5, value 2).

    ``tag`` selects which of this rank's cells are tagged: ``"all"``, only ``"A"`` (so the B
    cells are uncovered) or ``"none"`` -- per rank, if the caller varies it by rank.
    """
    from beat.cli.geometry import CLIGeometry

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    tdim = mesh.topology.dim
    imap = mesh.topology.index_map(tdim)
    cells = np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)  # incl. ghost cells
    mid = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
    values = np.where(mid[:, 0] < 0.5, 1, 2).astype(np.int32)
    keep = {"all": values > 0, "A": values == 1, "none": values < 0}[tag]
    cells, values = cells[keep], values[keep]
    cfun = dolfinx.mesh.meshtags(mesh, tdim, cells, values)
    return CLIGeometry(mesh=mesh, cfun=cfun, markers={"A": (1, tdim), "B": (2, tdim)})


def _cell_marker_conf(tmp_path):
    layers = {"method": "cell_markers", "map": {"left": "A", "right": "B"}}
    return conf_with(tmp_path, layers=layers)


def test_cell_markers_map_regions(tmp_path):
    conf = _cell_marker_conf(tmp_path)
    geo = _two_region_geometry("all")
    V = dolfinx.fem.functionspace(geo.mesh, ("P", 1))
    markers = build_region_markers(conf.cell, geo, V)
    values = set(np.unique(markers.x.array).astype(int))
    assert set().union(*MPI.COMM_WORLD.allgather(values)) == {0, 1}


def test_cell_markers_uncovered_cells_error_on_every_rank(tmp_path):
    """Uncovered cells on only *some* ranks must still raise on every rank (a rank-local check
    would make only those ranks raise, and the rest deadlock in the next collective)."""
    conf = _cell_marker_conf(tmp_path)
    comm = MPI.COMM_WORLD
    # Serial: B untagged. Parallel: every rank but the last is fully covered; the last rank
    # tags none of its cells.
    if comm.size == 1:
        tag = "A"
    else:
        tag = "all" if comm.rank < comm.size - 1 else "none"
    geo = _two_region_geometry(tag)
    V = dolfinx.fem.functionspace(geo.mesh, ("P", 1))
    with pytest.raises(ConfigError, match="does not cover every mesh cell"):
        build_region_markers(conf.cell, geo, V)


def test_steady_state_cache_check_is_decided_on_rank0(tmp_path, monkeypatch):
    """Whether the steady-state cache exists must be decided once (rank 0) and broadcast: a
    rank that (e.g. via NFS lag, or a concurrent job) sees a different answer must not take a
    different path around the collective broadcast."""
    from pathlib import Path

    conf = conf_with(tmp_path, steady_state={"num_beats": 1, "BCL": "2 ms", "dt": "0.1 ms"})
    first = build_cell_model(conf.cell, tmp_path / "c", MPI.COMM_WORLD)
    if MPI.COMM_WORLD.rank != 0:
        real = Path.is_file
        monkeypatch.setattr(
            Path,
            "is_file",
            lambda self: False if self.suffix == ".npy" else real(self),
        )
    again = build_cell_model(conf.cell, tmp_path / "c", MPI.COMM_WORLD)
    assert np.allclose(first.init_states[0], again.init_states[0])
