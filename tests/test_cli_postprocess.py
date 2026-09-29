import csv
import json

from mpi4py import MPI

import pytest
import toml
from cli_helpers import minimal_config_dict

from beat.cli.overrides import load_config
from beat.cli.postprocess import run_ecg, run_post
from beat.cli.runner import run


@pytest.fixture
def finished(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    data = minimal_config_dict(
        tmp_path,
        postprocess={
            "points": {"P1": [0.5, 0.5], "far": [10.0, 10.0]},
            "activation_threshold": 0.5,
        },
    )
    path = tmp_path / "config.toml"
    if MPI.COMM_WORLD.rank == 0:
        path.write_text(toml.dumps(data))
    MPI.COMM_WORLD.barrier()
    conf = load_config(path, environ={})
    run(conf)
    return conf


def test_post_writes_vtx_and_activation(finished):
    run_post(finished, MPI.COMM_WORLD)
    post = finished.output.folder / "post"
    assert (post / "v.bp").exists()
    assert (post / "activation_time.bp").exists()
    data = json.loads((post / "activation_times.json").read_text())
    assert data["far"] is None
    assert "P1" in data


def test_post_without_vtx(finished):
    finished.postprocess.vtx = False
    run_post(finished, MPI.COMM_WORLD)
    assert not (finished.output.folder / "post" / "v.bp").exists()


def test_ecg_csv(finished):
    run_ecg(finished, MPI.COMM_WORLD)
    rows = list(csv.reader((finished.output.folder / "post" / "ecg.csv").open()))
    assert rows[0] == ["time", "P1", "far"]
    assert len(rows) == 1 + 4  # 4 unique saved times


def test_ecg_requires_points(finished):
    from beat.cli.config import ConfigError

    finished.postprocess.points = {}
    with pytest.raises(ConfigError, match="points"):
        run_ecg(finished, MPI.COMM_WORLD)


def test_post_requires_prior_run(tmp_path):
    from beat.cli.config import Config, ConfigError

    conf = Config.model_validate(minimal_config_dict(tmp_path))
    with pytest.raises(ConfigError, match="beat run"):
        run_post(conf, MPI.COMM_WORLD)
