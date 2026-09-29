import logging
import signal
from contextlib import contextmanager

from mpi4py import MPI

import pytest
import toml
from cli_helpers import minimal_config_dict

import beat
from beat.cli import main


@contextmanager
def _timeout(seconds: int):
    """Fail (instead of hanging forever) if the body doesn't return within ``seconds``.

    Guards MPI-deadlock regressions: if a rank-0-only failure ever again skipped the
    broadcast in ``beat.cli``'s ``init`` dispatch, the other ranks under ``mpirun`` would block
    on ``comm.bcast``/``comm.barrier`` forever instead of raising -- turning that hang into a
    clear ``TimeoutError`` instead of a stuck CI job.
    """

    def _handler(signum, frame):
        raise TimeoutError(f"timed out after {seconds}s (possible MPI deadlock)")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@pytest.fixture
def tmp_path(tmp_path):
    """Share rank 0's ``tmp_path`` with every rank (see tests/test_cli_runner.py)."""
    return MPI.COMM_WORLD.bcast(tmp_path, root=0)


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "config.toml"
    if MPI.COMM_WORLD.rank == 0:
        path.write_text(toml.dumps(minimal_config_dict(tmp_path)))
    MPI.COMM_WORLD.barrier()
    return path


def test_version(caplog):
    caplog.set_level(logging.INFO)
    assert main(["version"]) == 0
    assert f"fenicsx-beat: {beat.__version__}" in caplog.text


@pytest.mark.skip_in_parallel  # capsys is per-rank; only rank 0 prints the resolved config
def test_validate_config_ok_and_prints_resolved(cfg, caplog, capsys):
    caplog.set_level(logging.INFO)
    assert main(["validate-config", str(cfg), "--set", 'solver.dt="0.05 ms"']) == 0
    assert "0.05 millisecond" in capsys.readouterr().out


def test_validate_config_invalid_exit_1(cfg, caplog):
    assert main(["validate-config", str(cfg), "--set", "solver.dt=1"]) == 1
    assert "solver.dt" in caplog.text


def test_missing_config_exit_1(tmp_path, caplog):
    assert main(["run", str(tmp_path / "nope.toml")]) == 1
    assert "does not exist" in caplog.text


def test_run_and_restart(cfg, tmp_path):
    assert main(["run", str(cfg)]) == 0
    assert (tmp_path / "output" / "results.bp").exists()
    assert main(["run", str(cfg)]) == 1  # refuses without --overwrite
    assert main(["run", str(cfg), "--overwrite"]) == 0
    assert main(["run", str(cfg), "--restart", "--set", 'solver.end_time="0.5 ms"']) == 0


def test_output_folder_flag(cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(cfg), "--output-folder", "run_07"]) == 0
    assert (tmp_path / "run_07" / "results.bp").exists()


def test_env_override(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("BEAT_OUTPUT__FOLDER", str(tmp_path / "from_env"))
    assert main(["run", str(cfg)]) == 0
    assert (tmp_path / "from_env" / "results.bp").exists()


def test_solver_failure_exit_2(cfg, monkeypatch):
    import beat.cli.runner as runner

    def boom(*a, **k):
        raise runner.SolverFailure("diverged")

    monkeypatch.setattr(runner, "_time_loop", boom)
    assert main(["run", str(cfg)]) == 2


@pytest.mark.xfail(reason="templates added in Task 12", strict=False)
def test_init_template(tmp_path):
    target = tmp_path / "case" / "config.toml"
    assert main(["init", str(target), "--template", "niederer_benchmark"]) == 0
    assert target.is_file()
    assert main(["validate-config", str(target)]) == 0
    assert main(["init", str(target)]) == 1  # exists
    assert main(["init", str(target), "--force"]) == 0


def test_init_filesystem_error_becomes_config_error(tmp_path, monkeypatch, caplog):
    """A rank-0-only OSError (e.g. from mkdir/copyfile) during ``init`` must become a
    ConfigError raised on every rank (exit 1), never skip the broadcast and hang the others.
    """
    import beat.cli as cli

    if MPI.COMM_WORLD.rank == 0:

        def boom(target, template, force):
            raise OSError("disk full")

        monkeypatch.setattr(cli, "_init", boom)

    with _timeout(30):
        code = main(["init", str(tmp_path / "config.toml")])
    assert code == 1
    assert "disk full" in caplog.text


# NOTE (deviation from the literal brief, flagged for the controller): with no templates
# shipped yet (Task 12 adds them; Task 12's own RED step expects _available_templates() == set()
# beforehand), there is no "slab" template to list here. Marked xfail like test_init_template
# until Task 12 lands; remove this mark there too.
@pytest.mark.xfail(reason="templates added in Task 12", strict=False)
def test_init_unknown_template(tmp_path, caplog):
    assert main(["init", str(tmp_path / "c.toml"), "--template", "nope"]) == 1
    assert "slab" in caplog.text  # lists available templates


def test_geometry_command(cfg, tmp_path):
    assert main(["geometry", str(cfg)]) == 0


def test_dry_run(cfg, tmp_path):
    assert main(["--dry-run", "run", str(cfg)]) == 0
    assert not (tmp_path / "output").exists()
