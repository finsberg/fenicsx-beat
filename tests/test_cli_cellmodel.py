from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
from cli_helpers import minimal_config_dict

from beat.cli.cellmodel import build_cell_model, build_region_markers, state_index
from beat.cli.config import Config, ConfigError
from beat.cli.geometry import build_geometry


def conf_with(tmp_path, **cell):
    data = minimal_config_dict(tmp_path)
    data["cell"].update(cell)
    return Config.model_validate(data)


# Every test that calls `build_cell_model` exercises the on-disk, rank-0-only codegen cache
# (`load_module`), which assumes `cache_dir` is a *shared* path across ranks. Under
# `mpirun -n 2 pytest`, `tmp_path` is a per-process pytest fixture and is NOT rank-synchronized
# (verified: two ranks get e.g. `/tmp/pytest-of-root/pytest-50/test_tmp0` and `.../pytest-51/...`),
# so only rank 0 (which did the writing into its own local tmp_path) would find the file; other
# ranks would hit `FileNotFoundError`. Same issue and fix as the generated-geometry cache tests
# in `tests/test_cli_geometry.py`.


@pytest.mark.skip_in_parallel
def test_codegen_is_cached_by_content(tmp_path):
    conf = conf_with(tmp_path)
    m1 = build_cell_model(conf.cell, tmp_path / "cache", MPI.COMM_WORLD)
    files = sorted((tmp_path / "cache").glob("cell_model_*.py"))
    assert len(files) == 1
    build_cell_model(conf.cell, tmp_path / "cache", MPI.COMM_WORLD)
    assert sorted((tmp_path / "cache").glob("cell_model_*.py")) == files
    assert m1.state_names == ["v", "h"]
    assert m1.v_index == 0


@pytest.mark.skip_in_parallel
def test_parameter_overrides_and_unknown_parameter(tmp_path):
    conf = conf_with(tmp_path, parameters={"tau_in": 0.5})
    model = build_cell_model(conf.cell, tmp_path / "cache", MPI.COMM_WORLD)
    idx = model.module["parameter_index"]("tau_in")
    assert model.parameters[0][idx] == pytest.approx(0.5)
    bad = conf_with(tmp_path, parameters={"tau_typo": 0.5})
    with pytest.raises(ConfigError, match="tau_typo"):
        build_cell_model(bad.cell, tmp_path / "cache", MPI.COMM_WORLD)


@pytest.mark.skip_in_parallel
def test_unknown_v_name(tmp_path):
    with pytest.raises(ConfigError, match="V"):
        build_cell_model(conf_with(tmp_path, v_name="V").cell, tmp_path / "c", MPI.COMM_WORLD)


@pytest.mark.skip_in_parallel
def test_state_index_unknown(tmp_path):
    model = build_cell_model(conf_with(tmp_path).cell, tmp_path / "c", MPI.COMM_WORLD)
    with pytest.raises(ConfigError, match="cai"):
        state_index(model, "cai")


@pytest.mark.skip_in_parallel
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
