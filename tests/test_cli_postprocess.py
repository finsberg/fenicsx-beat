import csv
import json
import signal

from mpi4py import MPI

import pytest
import toml
from cli_helpers import minimal_config_dict

from beat.cli.overrides import load_config
from beat.cli.postprocess import run_ecg, run_post
from beat.cli.runner import run


class _TimedOut(Exception):
    pass


def _run_with_timeout(fn, *args, seconds=60, **kwargs):
    """Run ``fn`` under a hard wall-clock timeout.

    A real MPI deadlock (one rank waiting forever in a collective the others never enter)
    would otherwise hang the whole test run; this turns it into a clear per-rank failure
    instead. POSIX-only (``SIGALRM``), which is fine for the Linux CI/test environment.
    """

    def _on_alarm(signum, frame):
        raise _TimedOut(f"{fn.__name__} did not return within {seconds}s (possible MPI deadlock)")

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


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


@pytest.mark.postprocess
def test_post_writes_vtx_and_activation(finished):
    run_post(finished, MPI.COMM_WORLD)
    post = finished.output.folder / "post"
    assert (post / "v.bp").exists()
    assert (post / "activation_time.bp").exists()
    data = json.loads((post / "activation_times.json").read_text())
    assert data["far"] is None
    assert "P1" in data


@pytest.mark.postprocess
def test_post_without_vtx(finished):
    finished.postprocess.vtx = False
    run_post(finished, MPI.COMM_WORLD)
    assert not (finished.output.folder / "post" / "v.bp").exists()


@pytest.mark.postprocess
def test_ecg_csv(finished):
    run_ecg(finished, MPI.COMM_WORLD)
    rows = list(csv.reader((finished.output.folder / "post" / "ecg.csv").open()))
    assert rows[0] == ["time", "P1", "far"]
    assert len(rows) == 1 + 4  # 4 unique saved times


@pytest.mark.postprocess
def test_ecg_requires_points(finished):
    from beat.cli.config import ConfigError

    finished.postprocess.points = {}
    with pytest.raises(ConfigError, match="points"):
        run_ecg(finished, MPI.COMM_WORLD)


@pytest.mark.postprocess
def test_post_gif_render_failure_does_not_hang(finished, monkeypatch):
    """A rendering failure on rank 0 (e.g. off-screen rendering on a headless node) must not
    leave rank 0 skipping a collective ``io4dolfinx.read_function`` call that the other ranks
    still enter - see ``beat.cli.postprocess._rank0_guarded``. Simulated here by making
    ``pyvista.Plotter.screenshot`` raise on rank 0 only; every rank must still return from
    ``run_post`` (no hang, no propagated exception), and the non-visualization outputs must
    still be written.
    """
    pyvista = pytest.importorskip("pyvista")

    finished.postprocess.make_gif = True
    comm = MPI.COMM_WORLD

    def boom(self, *args, **kwargs):
        if comm.rank == 0:
            raise RuntimeError("simulated headless rendering failure")
        return None  # pragma: no cover - never reached, screenshot is only called on rank 0

    monkeypatch.setattr(pyvista.Plotter, "screenshot", boom)

    _run_with_timeout(run_post, finished, comm, seconds=60)

    post = finished.output.folder / "post"
    assert (post / "activation_time.bp").exists()
    assert (post / "activation_times.json").exists()
    assert (post / "v.bp").exists()
    assert not (post / "voltage_final.png").exists()
    assert not (post / "voltage.gif").exists()


@pytest.mark.postprocess
def test_post_requires_prior_run(tmp_path):
    from beat.cli.config import Config, ConfigError

    conf = Config.model_validate(minimal_config_dict(tmp_path))
    with pytest.raises(ConfigError, match="beat run"):
        run_post(conf, MPI.COMM_WORLD)


@pytest.mark.postprocess
@pytest.mark.parametrize("fn", [run_post, run_ecg])
def test_post_rejects_config_changed_since_run(finished, fn):
    """beat post/ecg must refuse a config whose physics differ from the run that wrote
    results.bp (e.g. an edited geometry), instead of crashing or silently mixing them."""
    from beat.cli.config import ConfigError

    changed = load_config(
        finished.output.folder.parent / "config.toml",
        environ={},
        sets=["geometry.dx=0.5"],
    )
    with pytest.raises(ConfigError, match="config.resolved.toml"):
        fn(changed, MPI.COMM_WORLD)
    # Without restart.json (run killed before its first checkpoint) the check falls back to
    # config.resolved.toml.
    if MPI.COMM_WORLD.rank == 0:
        (finished.output.folder / "restart.json").unlink()
    MPI.COMM_WORLD.barrier()
    with pytest.raises(ConfigError, match="config.resolved.toml"):
        fn(changed, MPI.COMM_WORLD)
    fn(finished, MPI.COMM_WORLD)  # the unchanged config still works


@pytest.mark.postprocess
def test_visualize_warns_previews_are_rank0_partition_only(finished, caplog):
    pytest.importorskip("pyvista")
    import dolfinx

    from beat.cli.geometry import build_geometry
    from beat.cli.postprocess import _visualize
    from beat.cli.runner import RESULTS, read_result_times

    class _TwoRanks:  # pretend to run on 2 ranks: only size is consulted for the warning
        size = 2
        rank = MPI.COMM_WORLD.rank

        def bcast(self, obj, root=0):
            return MPI.COMM_WORLD.bcast(obj, root=root)

    geo = build_geometry(finished.geometry, MPI.COMM_WORLD)
    V = dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 1))
    v, tact = dolfinx.fem.Function(V), dolfinx.fem.Function(V)
    path = finished.output.folder / RESULTS
    post = finished.output.folder / "post"
    if MPI.COMM_WORLD.rank == 0:
        post.mkdir(exist_ok=True)
    MPI.COMM_WORLD.barrier()
    times = read_result_times(path, MPI.COMM_WORLD)
    _visualize(finished, v=v, tact=tact, path=path, times=times, post=post, comm=_TwoRanks())
    if MPI.COMM_WORLD.rank == 0:
        assert "only rank 0's" in caplog.text


def test_post_reads_flat_restart_json(finished):
    """A restart.json written by beat 0.7.1/0.7.2 (flat) still passes the physics check."""
    if MPI.COMM_WORLD.rank == 0:
        path = finished.output.folder / "restart.json"
        flat = dict(json.loads(path.read_text())["ep"])
        flat.pop("functions")
        path.write_text(json.dumps(flat))
    MPI.COMM_WORLD.barrier()
    run_post(finished, MPI.COMM_WORLD)
