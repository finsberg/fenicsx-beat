import json
import logging

from mpi4py import MPI

import dolfinx
import io4dolfinx
import numpy as np
import pytest
import toml
from cli_helpers import minimal_config_dict

from beat.cli.config import ConfigError
from beat.cli.geometry import build_geometry
from beat.cli.log import MPIFileHandler
from beat.cli.overrides import load_config
from beat.cli.runner import RESULTS, read_result_times, run


@pytest.fixture
def tmp_path(tmp_path):
    """Share rank 0's ``tmp_path`` with every rank (see tests/test_cli_cellmodel.py).

    Under ``mpirun -n 2 pytest`` each rank is an independent pytest process with its own
    ``tmp_path``; the runner assumes one shared output folder, so every rank must use rank 0's.
    """
    return MPI.COMM_WORLD.bcast(tmp_path, root=0)


def write_cfg(tmp_path, **over):
    path = tmp_path / "config.toml"
    if MPI.COMM_WORLD.rank == 0:
        path.write_text(toml.dumps(minimal_config_dict(tmp_path, **over)))
    MPI.COMM_WORLD.barrier()
    return path


def test_run_writes_results_and_metadata(tmp_path):
    cfg = write_cfg(tmp_path, output={"fields": ["h"]})
    conf = load_config(cfg, environ={})
    out = run(conf)
    assert (out / "config.resolved.toml").is_file()
    meta = json.loads((out / "run.json").read_text())
    assert meta["status"] == "finished"
    assert meta["n_ranks"] == MPI.COMM_WORLD.size
    times = read_result_times(out / RESULTS, MPI.COMM_WORLD)
    assert np.allclose(times, [0.0, 0.1, 0.2, 0.3])
    assert len(read_result_times(out / RESULTS, MPI.COMM_WORLD, name="h")) == 4
    assert json.loads((out / "restart.json").read_text())["step"] == 3


def test_existing_output_requires_overwrite(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    run(conf)
    with pytest.raises(ConfigError, match="--overwrite"):
        run(conf)
    run(conf, overwrite=True)


def test_restart_without_checkpoint_errors(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    with pytest.raises(ConfigError, match="restart"):
        run(conf, restart=True)


def _final_v(conf):
    geo = build_geometry(conf.geometry, MPI.COMM_WORLD)
    V = dolfinx.fem.functionspace(geo.mesh, ("P", 1))
    v = dolfinx.fem.Function(V)
    times = read_result_times(conf.output.folder / RESULTS, MPI.COMM_WORLD)
    io4dolfinx.read_function(conf.output.folder / RESULTS, v, time=times[-1], name="v")
    return v


@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_restart_matches_continuous_run(tmp_path, theta):
    full = load_config(
        write_cfg(tmp_path / "a", solver={"dt": "0.1 ms", "end_time": "0.6 ms", "theta": theta}),
        environ={},
    )
    run(full)
    part = load_config(
        write_cfg(tmp_path / "b", solver={"dt": "0.1 ms", "end_time": "0.3 ms", "theta": theta}),
        environ={},
    )
    run(part)
    ext = load_config(
        part_path := (tmp_path / "b" / "config.toml"),
        environ={},
        sets=['solver.end_time="0.6 ms"'],
    )
    run(ext, restart=True)
    times = read_result_times(ext.output.folder / RESULTS, MPI.COMM_WORLD)
    assert np.allclose(times, np.arange(7) * 0.1)
    # No duplicate timestamps were appended (io4dolfinx would return the first, stale one).
    raw = io4dolfinx.read_timestamps(
        filename=ext.output.folder / RESULTS,
        comm=MPI.COMM_WORLD,
        function_name="v",
    )
    assert len(raw) == 7
    assert np.allclose(_final_v(full).x.array, _final_v(ext).x.array, atol=1e-10)
    assert part_path.is_file()
    meta = json.loads((ext.output.folder / "run.json").read_text())
    assert meta["restart"] is True
    assert meta["status"] == "finished"


def test_restart_rejects_changed_physics(tmp_path):
    cfg = write_cfg(tmp_path)
    run(load_config(cfg, environ={}))
    changed = load_config(cfg, environ={}, sets=['solver.dt="0.05 ms"'])
    with pytest.raises(ConfigError, match="physics"):
        run(changed, restart=True)


def test_checkpoint_every_writes_intermediate_restart(tmp_path):
    cfg = write_cfg(tmp_path, output={"checkpoint_every": "0.1 ms"})
    out = run(load_config(cfg, environ={}))
    times = read_result_times(out / "restart.bp", MPI.COMM_WORLD)
    assert np.allclose(times, [0.1, 0.2, 0.3])


def test_repeated_runs_do_not_accumulate_log_handlers(tmp_path):
    conf = load_config(write_cfg(tmp_path), environ={})
    run(conf)
    run(conf, overwrite=True)
    handlers = [h for h in logging.getLogger().handlers if isinstance(h, MPIFileHandler)]
    assert handlers == []
    assert (conf.output.folder / "output.log").is_file()


def test_performance_summary_is_saved(tmp_path):
    out = run(load_config(write_cfg(tmp_path, output={"performance": True}), environ={}))
    data = json.loads((out / "performance.json").read_text())
    assert data["total_steps"] > 0
