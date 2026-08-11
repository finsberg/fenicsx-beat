import logging

from mpi4py import MPI

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
        stimulus={"marker": "X0", "amplitude": 1.0, "duration": "1.0 ms"},
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
