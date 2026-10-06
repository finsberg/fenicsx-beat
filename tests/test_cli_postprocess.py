import csv
import json
import logging
import signal

from mpi4py import MPI

import numpy as np
import pytest
import toml
from cli_helpers import minimal_config_dict

from beat.cli.config import ConfigError, ECGConfig
from beat.cli.overrides import load_config
from beat.cli.postprocess import run_post
from beat.cli.runner import run
from beat.ecg import twelve_lead
from beat.units import ureg


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


# --- the pseudo-ECG ([postprocess.ecg]) ----------------------------------------------------
# The fixture's run is the unit square in mm. E1, E2 and RL lie outside it.
_ELECTRODES = {"E1": [2.0, 0.5], "E2": [-1.0, 0.5], "RL": [0.5, 3.0]}
_TWELVE_LEAD_NAMES = ("LA", "RA", "LL", "RL", *(f"V{i}" for i in range(1, 7)))


def _circle(radius: float = 3.0, centre=(0.5, 0.5)) -> dict[str, list[float]]:
    """Ten electrodes on a circle about the tissue, named for the twelve leads (plus RL)."""
    angles = 2 * np.pi * np.arange(len(_TWELVE_LEAD_NAMES)) / len(_TWELVE_LEAD_NAMES)
    return {
        name: [float(centre[0] + radius * np.cos(a)), float(centre[1] + radius * np.sin(a))]
        for name, a in zip(_TWELVE_LEAD_NAMES, angles)
    }


def _read_csv(path) -> tuple[list[str], list[list[float]]]:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], [[float(x) for x in row] for row in rows[1:]]


def _post_ecg(conf, **ecg) -> tuple[list[str], list[list[float]]]:
    conf.postprocess.ecg = ECGConfig(**ecg)
    run_post(conf, MPI.COMM_WORLD)
    return _read_csv(conf.output.folder / "post" / "ecg.csv")


def _expected_potentials(conf, electrodes) -> list[list[float]]:
    """ElectrodePotentials on the saved v, set up by hand (gate E5)."""
    import dolfinx
    import io4dolfinx

    from beat.cli.geometry import build_conductivity, build_geometry
    from beat.cli.runner import RESULTS, read_result_times
    from beat.ecg import ECGRecovery, ElectrodePotentials

    geo = build_geometry(conf.geometry, MPI.COMM_WORLD)
    v = dolfinx.fem.Function(dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 1)), name="v")
    recovery = ECGRecovery(
        v=v,
        sigma_b=1.0,
        C_m=conf.ep.C_m.to(f"uF/{conf.geometry.unit}**2").magnitude,
        M=build_conductivity(conf.ep, geo, conf.geometry.fibers),
    )
    probe = ElectrodePotentials(recovery, electrodes)
    path = conf.output.folder / RESULTS
    rows = []
    for t in read_result_times(path, MPI.COMM_WORLD):
        io4dolfinx.read_function(path, v, time=t, name="v")
        rows.append([float(t), *probe().values()])
    return rows


@pytest.mark.postprocess
def test_post_without_ecg_writes_no_ecg_files(finished):
    run_post(finished, MPI.COMM_WORLD)
    post = finished.output.folder / "post"
    assert (post / "activation_time.bp").exists()
    assert not list(post.glob("ecg*.csv"))
    assert not list(post.glob("ecg*.png"))


@pytest.mark.postprocess
def test_post_ecg_csv(finished, monkeypatch):
    import beat.cli.postprocess as postprocess

    writes: list[str] = []
    write_csv = postprocess._write_csv
    monkeypatch.setattr(
        postprocess,
        "_write_csv",
        lambda path, *args: (writes.append(path.name), write_csv(path, *args)),
    )

    header, rows = _post_ecg(finished, electrodes=_ELECTRODES)
    post = finished.output.folder / "post"
    assert header == ["time", "E1", "E2", "RL"]
    assert len(rows) == 4  # 4 unique saved times
    assert np.isfinite(rows).all()
    assert rows == _expected_potentials(finished, _ELECTRODES)
    assert not (post / "ecg_leads.csv").exists()
    # Written once, on one rank.
    assert sum(MPI.COMM_WORLD.allgather(writes), []) == ["ecg.csv"]


@pytest.mark.postprocess
def test_post_ecg_unit(finished):
    """The same electrodes given in cm give the same ecg.csv as in mm (the geometry's unit)."""
    post = finished.output.folder / "post"
    _post_ecg(finished, electrodes=_ELECTRODES)
    in_mm = (post / "ecg.csv").read_bytes()

    in_cm = {name: [x / 10 for x in pos] for name, pos in _ELECTRODES.items()}
    assert all(y * 10 == x for name in in_cm for x, y in zip(_ELECTRODES[name], in_cm[name]))
    _post_ecg(finished, electrodes=in_cm, unit="cm")
    if ureg.Quantity(1, "cm").to("mm").magnitude == 10.0:
        assert (post / "ecg.csv").read_bytes() == in_mm
    else:  # pragma: no cover - pint's factor not exactly 10: compare to round-off instead
        np.testing.assert_allclose(
            _read_csv(post / "ecg.csv")[1],
            [[float(x) for x in row.split(b",")] for row in in_mm.splitlines()[1:]],
            rtol=1e-14,
        )


@pytest.mark.postprocess
def test_post_twelve_lead_potential(finished):
    electrodes = _circle()
    header, rows = _post_ecg(finished, electrodes=electrodes, leads="twelve-lead")
    assert header == ["time", *electrodes]  # RL too, though no lead uses it (Review Focus 3)
    assert "RL" in header

    system = twelve_lead()
    lead_header, lead_rows = _read_csv(finished.output.folder / "post" / "ecg_leads.csv")
    assert lead_header == ["time", *system.names]
    assert len(lead_rows) == len(rows)
    for row, lead_row in zip(rows, lead_rows):
        assert lead_row[0] == row[0]
        leads = system.leads(dict(zip(header[1:], row[1:])))
        assert lead_row[1:] == [leads[name] for name in system.names]


@pytest.mark.postprocess
def test_post_twelve_lead_position(finished):
    post = finished.output.folder / "post"
    electrodes = _circle()
    _post_ecg(finished, electrodes=electrodes, leads="twelve-lead")
    names, potential = _read_csv(post / "ecg_leads.csv")
    header, _ = _post_ecg(
        finished,
        electrodes=electrodes,
        leads="twelve-lead",
        reference="position",
    )
    assert header == ["time", *electrodes]  # the given electrodes only, no derived points
    position_names, position = _read_csv(post / "ecg_leads.csv")
    assert position_names == names

    a, b = np.asarray(potential), np.asarray(position)
    assert (a[:, 0] == b[:, 0]).all()
    for i, name in enumerate(names[1:], start=1):
        if name in ("I", "II", "III"):
            assert (a[:, i] == b[:, i]).all(), name
        else:
            assert np.isfinite(b[:, i]).all(), name
            assert (a[:, i] != b[:, i]).any(), name


@pytest.mark.postprocess
def test_post_ecg_wrong_dimension(finished):
    finished.postprocess.ecg = ECGConfig(electrodes={**_ELECTRODES, "Z3": [2.0, 0.5, 0.0]})
    with pytest.raises(ConfigError, match="Z3"):
        run_post(finished, MPI.COMM_WORLD)
    assert not (finished.output.folder / "post" / "ecg.csv").exists()


@pytest.mark.postprocess
def test_post_twelve_lead_missing_electrode(finished):
    electrodes = _circle()
    del electrodes["V6"]
    finished.postprocess.ecg = ECGConfig(electrodes=electrodes, leads="twelve-lead")
    with pytest.raises(ConfigError, match="V6"):
        run_post(finished, MPI.COMM_WORLD)
    assert not (finished.output.folder / "post" / "ecg.csv").exists()


@pytest.mark.postprocess
def test_post_ecg_electrode_named_like_a_derived_point(finished):
    """Under reference = "position", WCT_pt is a derived point; an electrode of that name
    would be overwritten by it, and ecg.csv would hold the wrong potential under its name."""
    electrodes = {**_circle(), "WCT_pt": [2.0, 0.5]}
    finished.postprocess.ecg = ECGConfig(
        electrodes=electrodes,
        leads="twelve-lead",
        reference="position",
    )
    with pytest.raises(ConfigError, match="WCT_pt"):
        run_post(finished, MPI.COMM_WORLD)


@pytest.mark.postprocess
def test_post_warns_electrode_inside_mesh(finished, caplog):
    """An electrode inside the tissue gives a finite near-field value, with a warning naming
    it, logged once (on rank 0), not once per rank (Review Focus 1)."""
    caplog.set_level(logging.WARNING, logger="beat.cli.postprocess")
    # On two ranks, each corner lies in one rank's cells only, so rank 0 can name both only
    # through the reduction over ranks.
    inside = {"E_centre": [0.5, 0.5], "E_corner_a": [0.05, 0.95], "E_corner_b": [0.95, 0.05]}
    header, rows = _post_ecg(finished, electrodes={**inside, **_ELECTRODES})
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "E_centre" in r.getMessage()
    ]
    assert len(warnings) == (1 if MPI.COMM_WORLD.rank == 0 else 0)
    assert all(name in w for w in warnings for name in inside)
    assert all(name not in w for w in warnings for name in _ELECTRODES)
    for name in inside:
        column = header.index(name)
        assert np.isfinite([row[column] for row in rows]).all(), name


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
@pytest.mark.parametrize("fn", [run_post])
def test_post_rejects_config_changed_since_run(finished, fn):
    """beat post must refuse a config whose physics differ from the run that wrote
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
