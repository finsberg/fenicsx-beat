import logging

from mpi4py import MPI

import numpy as np
import pytest

import beat
import beat.cli
from beat.config import Config

MITCHELL_SCHAEFFER_ODE = """
parameters(
    tau_in = 0.3,
    tau_out = 6.0,
    tau_open = 120.0,
    tau_close = 150.0,
    v_gate = 0.13
)
states(
    v = 0.0,
    h = 1.0
)

h_inf = Conditional(Lt(v, v_gate), 1.0, 0.0)
tau_h = Conditional(Lt(v, v_gate), tau_open, tau_close)

dv_dt = h * (v**2 * (1.0 - v)) / tau_in - v / tau_out
dh_dt = (h_inf - h) / tau_h
"""


def test_version(caplog):
    caplog.set_level(logging.INFO)
    ret = beat.cli.main(["version"])
    assert ret == 0
    assert caplog.records[0].msg == f"fenicsx-beat: {beat.__version__}"


@pytest.fixture(scope="module")
def slab_geometry_folder(tmp_path_factory):
    cg = pytest.importorskip("cardiac_geometries")
    outdir = tmp_path_factory.mktemp("mesh")
    cg.mesh.slab(
        outdir=outdir,
        lx=1.0,
        ly=0.3,
        lz=0.3,
        dx=0.15,
        create_fibers=True,
        use_dolfinx=True,
        comm=MPI.COMM_WORLD,
    )
    return outdir


@pytest.fixture
def ode_file(tmp_path):
    path = tmp_path / "mitchell_schaeffer.ode"
    path.write_text(MITCHELL_SCHAEFFER_ODE)
    return path


@pytest.fixture
def run_config(tmp_path, slab_geometry_folder, ode_file):
    # A deliberately tiny/short simulation: a handful of tets, a couple of PDE timesteps,
    # and a single beat, just to exercise the full CLI wiring end-to-end quickly.
    # P1 = [0.5, 0.15, 0.15] sits inside the slab (lx=1.0, ly=lz=0.3), for `beat ecg`/`beat post`.
    return Config(
        mesh={"unit": "mm", "folder": str(slab_geometry_folder)},
        cell={
            "ode_file": str(ode_file),
            "num_beats": 1,
            "BCL": "5 ms",
            "dt": "0.1 ms",
            "module_name": str(tmp_path / "mitchell_schaeffer.py"),
            "v_name": "v",
            "track_indices": ["v", "h"],
        },
        simulation={
            "num_beats": 1,
            "BCL": "5 ms",
            "dt": "0.1 ms",
            "save_every_ms": 1.0,
            "output_folder": str(tmp_path / "output"),
        },
        stimulus={"marker": "X0", "amplitude": 5000.0, "duration": "1.0 ms"},
        postprocess={
            # Mitchell-Schaeffer's v is a normalized (roughly 0-1, not mV) action potential, and
            # its resting value is exactly 0.0 - a threshold of 0.0 would count every point as
            # "activated" at t=0 by definition. 0.5 requires an actual propagating upstroke.
            "activation_threshold": 0.5,
            "points": {"P1": [0.5, 0.15, 0.15]},
        },
    )


@pytest.mark.skip_in_parallel
def test_run_end_to_end(run_config, tmp_path):
    config_path = tmp_path / "config.toml"
    run_config.dump_toml(config_path)

    ret = beat.cli.main(["run", str(config_path)])
    assert ret == 0

    output_folder = run_config.simulation.output_folder
    assert (output_folder / "result.bp").exists()
    assert (output_folder / "output.log").exists()
    assert any((output_folder / "init_states").glob("steady_states_*.npy"))


@pytest.mark.skip_in_parallel
def test_run_unknown_stimulus_marker(run_config, tmp_path, caplog):
    run_config.stimulus.marker = "NOT_A_MARKER"
    config_path = tmp_path / "config.toml"
    run_config.dump_toml(config_path)

    caplog.set_level(logging.ERROR)
    ret = beat.cli.main(["run", str(config_path)])
    assert ret == 1
    assert "NOT_A_MARKER" in caplog.text


@pytest.mark.skip_in_parallel
def test_ecg_end_to_end(run_config, tmp_path):
    config_path = tmp_path / "config.toml"
    run_config.dump_toml(config_path)
    assert beat.cli.main(["run", str(config_path)]) == 0

    ret = beat.cli.main(["ecg", str(config_path)])
    assert ret == 0

    output_folder = run_config.simulation.output_folder
    csv_path = output_folder / "ecg.csv"
    assert csv_path.is_file()

    import csv

    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["t", "P1"]
    # one row per saved timestep (6: t=0..5 ms at save_every_ms=1.0) plus the header
    assert len(rows) == 7
    assert all(len(row) == 2 for row in rows[1:])
    # values must be finite numbers, not blow up
    assert all(np.isfinite(float(row[1])) for row in rows[1:])


@pytest.mark.skip_in_parallel
def test_ecg_requires_points(run_config, tmp_path, caplog):
    run_config.postprocess.points = {}
    config_path = tmp_path / "config.toml"
    run_config.dump_toml(config_path)
    assert beat.cli.main(["run", str(config_path)]) == 0

    caplog.set_level(logging.ERROR)
    ret = beat.cli.main(["ecg", str(config_path)])
    assert ret == 1
    assert "points" in caplog.text


def test_ecg_requires_prior_run(tmp_path, caplog):
    # Doesn't need a real mesh/ode file: the checkpoint-exists check runs before geometry is
    # ever loaded, so a bare default Config is enough (and keeps this test MPI-safe).
    config_path = tmp_path / "config.toml"
    Config(
        simulation={"output_folder": str(tmp_path / "output")},
        postprocess={"points": {"P1": [0.0, 0.0, 0.0]}},
    ).dump_toml(config_path)

    caplog.set_level(logging.ERROR)
    ret = beat.cli.main(["ecg", str(config_path)])
    assert ret == 1
    assert "beat run" in caplog.text


@pytest.mark.skip_in_parallel
def test_post_end_to_end(run_config, tmp_path):
    config_path = tmp_path / "config.toml"
    run_config.dump_toml(config_path)
    assert beat.cli.main(["run", str(config_path)]) == 0

    ret = beat.cli.main(["post", str(config_path)])
    assert ret == 0

    output_folder = run_config.simulation.output_folder
    assert (output_folder / "activation_time.xdmf").is_file()
    assert (output_folder / "activation_time.h5").is_file()

    import json

    result = json.loads((output_folder / "activation_times.json").read_text())
    assert result["threshold_mV"] == 0.5
    # P1 is close to the stimulated face (X0) and the stimulus is strong (5000), so it should
    # genuinely activate partway through the 5 ms window - not stay at the -1.0 "never
    # activated" sentinel, and not be None (reserved for points outside the mesh, see
    # test_post_point_outside_mesh_is_null below).
    assert result["P1"] is not None
    assert 0.0 < result["P1"] <= 5.0


@pytest.mark.skip_in_parallel
def test_post_point_outside_mesh_is_null(run_config, tmp_path, caplog):
    run_config.postprocess.points = {"Outside": [100.0, 100.0, 100.0]}
    config_path = tmp_path / "config.toml"
    run_config.dump_toml(config_path)
    assert beat.cli.main(["run", str(config_path)]) == 0

    caplog.set_level(logging.WARNING)
    ret = beat.cli.main(["post", str(config_path)])
    assert ret == 0
    assert "outside the mesh" in caplog.text

    import json

    output_folder = run_config.simulation.output_folder
    result = json.loads((output_folder / "activation_times.json").read_text())
    assert result["Outside"] is None


def test_post_requires_prior_run(tmp_path, caplog):
    config_path = tmp_path / "config.toml"
    Config(simulation={"output_folder": str(tmp_path / "output")}).dump_toml(config_path)

    caplog.set_level(logging.ERROR)
    ret = beat.cli.main(["post", str(config_path)])
    assert ret == 1
    assert "beat run" in caplog.text


def test_run_missing_config(tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    ret = beat.cli.main(["run", str(tmp_path / "does-not-exist.toml")])
    assert ret == 1
    assert "does not exist" in caplog.text


def test_validate_config_valid(tmp_path, caplog):
    # validate-config only checks the schema, not that referenced files/folders exist,
    # so this doesn't need a real mesh or ode file (and stays MPI-safe, unlike `run_config`).
    config_path = tmp_path / "config.toml"
    Config().dump_toml(config_path)

    caplog.set_level(logging.INFO)
    ret = beat.cli.main(["validate-config", str(config_path)])
    assert ret == 0
    assert "is valid" in caplog.text


def test_validate_config_missing_file(tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    ret = beat.cli.main(["validate-config", str(tmp_path / "does-not-exist.toml")])
    assert ret == 1
    assert "does not exist" in caplog.text


def test_validate_config_invalid_content(tmp_path, caplog):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[ep]\nchi = "5 ms"\n')  # wrong dimensionality

    caplog.set_level(logging.ERROR)
    ret = beat.cli.main(["validate-config", str(config_path)])
    assert ret == 1


def test_init_creates_default_config(tmp_path):
    config_path = tmp_path / "config.toml"
    ret = beat.cli.main(["init", str(config_path)])
    assert ret == 0
    assert config_path.is_file()

    # Round-trips through Config without error.
    Config.parse_toml(config_path)


def test_init_defaults_to_config_toml_in_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ret = beat.cli.main(["init"])
    assert ret == 0
    assert (tmp_path / "config.toml").is_file()


def test_init_refuses_to_overwrite_without_force(tmp_path, caplog):
    config_path = tmp_path / "config.toml"
    config_path.write_text("existing content")

    caplog.set_level(logging.ERROR)
    ret = beat.cli.main(["init", str(config_path)])
    assert ret == 1
    assert config_path.read_text() == "existing content"


def test_init_overwrites_with_force(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("existing content")

    ret = beat.cli.main(["init", str(config_path), "--force"])
    assert ret == 0
    assert config_path.read_text() != "existing content"


def test_dry_run_does_not_execute(tmp_path):
    ret = beat.cli.main(["--dry-run", "run", str(tmp_path / "does-not-exist.toml")])
    assert ret == 0
