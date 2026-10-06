"""Postprocess a `beat run` output folder: pseudo-ECG, activation times, VTX conversion.

Reads ``<output.folder>/results.bp`` (the io4dolfinx checkpoint written by `beat run`) and
writes everything this module produces into ``<output.folder>/post/``, so that a plain
``beat run --overwrite`` (which deletes ``post/`` along with its own artifacts, see
``beat.cli.runner._ARTIFACT_NAMES``) never touches the user's config/geometry/ode files.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable

from mpi4py import MPI

import dolfinx
import dolfinx.geometry
import io4dolfinx
import numpy as np
import scifem

from ..ecg import ECGRecovery, ElectrodePotentials, LeadSystem, twelve_lead
from ..units import ureg
from .config import Config, ConfigError
from .geometry import CLIGeometry, build_conductivity, build_geometry
from .overrides import load_config, physics_hash
from .runner import RESTART_META, RESULTS, _on_rank0, read_result_times

logger = logging.getLogger(__name__)


def _check_matches_run(conf: Config, comm) -> None:
    """Refuse a config whose physics differ from the run that wrote ``results.bp``.

    ``beat post`` rebuilds the geometry (and, for the ECG, the conductivity) from the
    config, so e.g. an edited ``geometry.dx`` would otherwise crash reading ``results.bp``
    or, on a same-topology mesh, silently produce wrong results. Only what ``physics_hash``
    excludes (``[output]``, ``[postprocess]``, the run length) may differ. The recorded hash is
    taken from ``restart.json`` or, if the run was killed before its first checkpoint, recomputed
    from ``config.resolved.toml``; if neither exists the results are refused as unverifiable.
    Checked on rank 0 and broadcast, so every rank raises together.
    """
    folder = conf.output.folder
    resolved = folder / "config.resolved.toml"

    def check() -> None:
        meta = folder / RESTART_META
        if meta.is_file():
            recorded = json.loads(meta.read_text())["physics_hash"]
        elif resolved.is_file():
            recorded = physics_hash(load_config(resolved, environ={}))
        else:
            raise ConfigError(
                f"Cannot verify that {folder / RESULTS} was written with this config: neither "
                f"{meta} nor {resolved} exists",
            )
        if recorded != physics_hash(conf):
            raise ConfigError(
                f"The config's physics settings differ from the run that wrote "
                f"{folder / RESULTS} (only [output], [postprocess] and the run length may "
                f"change for beat post). Compare with {resolved}",
            )

    _on_rank0(comm, ConfigError, check)


def _open_results(conf: Config, comm):
    """Load the geometry (via ``build_geometry``) and a ``v`` Function on *that same mesh
    object*, ready to be filled from the ``results.bp`` written by `beat run`.

    ``results.bp`` is written with ``io4dolfinx.write_function_on_input_mesh``, which pairs with
    a Function built directly on the mesh from ``build_geometry`` (rather than a freshly
    ``read_mesh``'d one) - required so that ``v`` can be combined in one UFL form with M (built
    from that same geometry's fiber field) for the ECG recovery.
    """
    path = conf.output.folder / RESULTS
    if not path.exists():
        raise ConfigError(f"No results found at {path}. Run `beat run <config>` first.")
    _check_matches_run(conf, comm)
    geo = build_geometry(conf.geometry, comm)
    V = dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 1))
    v = dolfinx.fem.Function(V, name="v")
    times = read_result_times(path, comm)  # unique + sorted (restart-safe)
    post = conf.output.folder / "post"
    _on_rank0(comm, OSError, lambda: post.mkdir(parents=True, exist_ok=True))
    return path, geo, v, times, post


def _ecg_setup(
    conf: Config,
    geo: CLIGeometry,
    v: dolfinx.fem.Function,
) -> tuple[ElectrodePotentials, LeadSystem | None, list[str]] | None:
    """Set up the pseudo-ECG of ``[postprocess.ecg]``, or return ``None`` without it.

    Everything that can refuse the section happens here, before any saved time is read: a
    position whose length is not the mesh's dimension, and a lead system's missing electrode.
    Returns the probe (one recovery solve per call, at the given electrodes and the lead
    system's derived points), the lead system (``None`` for ``leads = "none"``), and the given
    electrode names in order, which are ``ecg.csv``'s columns.
    """
    ecg = conf.postprocess.ecg
    if ecg is None:
        return None
    unit = conf.geometry.unit
    scale = float(ureg.Quantity(1, ecg.unit or unit).to(unit).magnitude)
    gdim = geo.mesh.geometry.dim
    wrong = [name for name, pos in ecg.electrodes.items() if len(pos) != gdim]
    if wrong:
        raise ConfigError(
            f"postprocess.ecg.electrodes: the mesh is {gdim}-dimensional, but "
            + ", ".join(f"{name} has {len(ecg.electrodes[name])} coordinates" for name in wrong),
        )
    electrodes = {
        name: np.asarray(pos, dtype=float) * scale for name, pos in ecg.electrodes.items()
    }

    system: LeadSystem | None = None
    points: dict[str, np.ndarray] = electrodes
    if ecg.leads == "twelve-lead":
        try:
            system = twelve_lead(ecg.reference)
            points = system.points(electrodes)
        except ValueError as e:
            raise ConfigError(f"postprocess.ecg (leads = {ecg.leads!r}): {e}") from e
        clash = [name for name in system.derived if name in electrodes]
        if clash:
            raise ConfigError(
                f"postprocess.ecg.electrodes: {', '.join(clash)} would be replaced by the "
                f"derived point of that name under reference = {ecg.reference!r}; rename it",
            )

    inside = _inside_mesh(geo.mesh, electrodes)
    if inside and geo.mesh.comm.rank == 0:
        logger.warning(
            f"postprocess.ecg: electrode(s) {', '.join(repr(n) for n in inside)} lie inside "
            "the mesh; the potential there is a near-field value, not a body-surface one.",
        )

    recovery = ECGRecovery(
        v=v,
        sigma_b=ecg.sigma_b,
        C_m=conf.ep.C_m.to(f"uF/{unit}**2").magnitude,
        M=build_conductivity(conf.ep, geo, conf.geometry.fibers),
    )
    return ElectrodePotentials(recovery, points), system, list(electrodes)


def _inside_mesh(mesh: dolfinx.mesh.Mesh, points: dict[str, np.ndarray]) -> list[str]:
    """The names of the points inside some cell of ``mesh``, the same on every rank."""
    names = list(points)
    x = np.zeros((len(names), 3), dtype=mesh.geometry.x.dtype)
    for i, name in enumerate(names):
        x[i, : len(points[name])] = points[name]
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, x)
    cells = dolfinx.geometry.compute_colliding_cells(mesh, candidates, x)
    local = np.array([cells.links(i).size > 0 for i in range(len(names))], dtype=np.int32)
    found = np.zeros_like(local)
    mesh.comm.Allreduce(local, found, op=MPI.MAX)
    return [name for name, f in zip(names, found) if f]


def _write_csv(path: Path, header: list[str], times, rows: list[list[float]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", *header])
        for t, row in zip(times, rows):
            writer.writerow([t, *row])


def _write_ecg(
    potentials: list[dict[str, float]],
    system: LeadSystem | None,
    names: list[str],
    times,
    post: Path,
    comm,
) -> None:
    """Write ``ecg.csv``/``ecg.png`` (the given electrodes) and, with a lead system,
    ``ecg_leads.csv``/``ecg_leads.png``, on rank 0. Without a lead system, an earlier run's
    ``ecg_leads.*`` are deleted."""
    rows = [[row[n] for n in names] for row in potentials]
    lead_names = list(system.names) if system is not None else []
    lead_rows: list[list[float]] = []
    if system is not None:
        for row in potentials:
            leads = system.leads(row)
            lead_rows.append([float(leads[n]) for n in lead_names])

    def columns(header: list[str], table: list[list[float]]) -> dict[str, list[float]]:
        return {n: [r[i] for r in table] for i, n in enumerate(header)}

    def write_outputs() -> None:
        csv_path = post / "ecg.csv"
        _write_csv(csv_path, names, times, rows)
        logger.info(f"ECG values saved to {csv_path}")
        _plot_ecg(times, columns(names, rows), post / "ecg.png")
        if system is None:
            # An earlier run's leads were computed from other electrodes: don't leave them
            # beside this ecg.csv.
            for stale in ("ecg_leads.csv", "ecg_leads.png"):
                (post / stale).unlink(missing_ok=True)
            return
        leads_path = post / "ecg_leads.csv"
        _write_csv(leads_path, lead_names, times, lead_rows)
        logger.info(f"ECG leads saved to {leads_path}")
        _plot_leads(
            times,
            columns(lead_names, lead_rows),
            system.layout or tuple((n,) for n in lead_names),
            post / "ecg_leads.png",
        )

    _on_rank0(comm, OSError, write_outputs)


def _plot_ecg(times, values: dict[str, list[float]], png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed, skipping the ECG plot")
        return

    fig, ax = plt.subplots()
    for name, y in values.items():
        ax.plot(times, y, label=name)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"$\phi_e$")
    ax.legend()
    fig.savefig(png_path)
    plt.close(fig)
    logger.info(f"ECG plot saved to {png_path}")


def _plot_leads(
    times,
    leads: dict[str, list[float]],
    layout: tuple[tuple[str, ...], ...],
    png_path: Path,
) -> None:
    """One panel per lead, on the lead system's grid, each titled by the lead's name."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed, skipping the ECG leads plot")
        return

    nrows, ncols = len(layout), max(len(row) for row in layout)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharex=True,
        figsize=(3 * ncols, 2 * nrows),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for i, row in enumerate(layout):
        for j, name in enumerate(row):
            ax = axes[i, j]
            ax.set_visible(True)
            ax.plot(times, leads[name])
            ax.set_title(name)
            if i == nrows - 1:
                ax.set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    logger.info(f"ECG leads plot saved to {png_path}")


def run_post(conf: Config, comm=MPI.COMM_WORLD) -> Path:
    """Compute a full-mesh local activation time map (and, at ``postprocess.points``, activation
    times as scalars) from a previously saved `beat run` output, the pseudo-ECG (if
    ``[postprocess.ecg]`` is set), convert ``v`` to VTX for ParaView (if ``postprocess.vtx``),
    and (if pyvista is installed) render PNG/GIF visualizations. Everything is written into
    ``post/``.
    """
    path, geo, v, times, post = _open_results(conf, comm)
    ecg = _ecg_setup(conf, geo, v)  # every [postprocess.ecg] refusal, before any time is read

    threshold = conf.postprocess.activation_threshold
    tact = dolfinx.fem.Function(v.function_space, name="activation_time")
    tact.x.array[:] = -1.0
    potentials: list[dict[str, float]] = []
    for t in times:
        io4dolfinx.read_function(path, v, time=t, name="v")
        pending = tact.x.array < 0.0
        tact.x.array[pending & (v.x.array >= threshold)] = t
        if ecg is not None:
            probe, _, _ = ecg
            potentials.append(probe())

    activation_path = post / "activation_time.bp"
    with dolfinx.io.VTXWriter(comm, activation_path, [tact], engine="BP4") as vtx:
        vtx.write(0.0)
    logger.info(f"Activation time map saved to {activation_path}")

    point_results: dict[str, float | None] = {"threshold_mV": threshold}
    if conf.postprocess.points:
        names = list(conf.postprocess.points)
        pts = [conf.postprocess.points[name] for name in names]
        vals = np.asarray(scifem.evaluate_function(tact, pts)).reshape(-1)
        for name, val in zip(names, vals):
            if np.isfinite(val):
                point_results[name] = float(val)
            else:
                # scifem returns a non-finite value for points outside the mesh - null rather
                # than -1.0 (our "not yet activated" sentinel for points inside the mesh) since
                # it's not a meaningful activation time at all, and -inf/nan aren't valid JSON.
                point_results[name] = None
                logger.warning(
                    f"Point {name!r} ({conf.postprocess.points[name]}) lies outside the mesh "
                    "domain; activation time is undefined there (recorded as null). "
                    "Electrodes for the pseudo-ECG go in [postprocess.ecg.electrodes].",
                )

    json_path = post / "activation_times.json"
    _on_rank0(comm, OSError, lambda: json_path.write_text(json.dumps(point_results, indent=2)))
    logger.info(f"Activation times at points saved to {json_path}")

    if ecg is not None:
        _, system, names = ecg
        _write_ecg(potentials, system, names, times, post, comm)

    if conf.postprocess.vtx:
        _convert_to_vtx(path, v, times, post, comm)

    _visualize(conf, v=v, tact=tact, path=path, times=times, post=post, comm=comm)

    return activation_path


def _convert_to_vtx(path: Path, v: dolfinx.fem.Function, times, post: Path, comm) -> None:
    out = post / "v.bp"
    with dolfinx.io.VTXWriter(comm, out, [v], engine="BP4") as vtx:
        for t in times:
            io4dolfinx.read_function(path, v, time=t, name="v")
            vtx.write(float(t))
    logger.info(f"VTX output for ParaView written to {out}")


def _rank0_guarded(comm, ok: bool, step_name: str, fn: Callable[[], None]) -> bool:
    """Run ``fn`` on rank 0 only, if no earlier step already failed, and make the resulting
    "did visualization succeed so far" flag identical on every rank.

    This is the crux of keeping ``_visualize`` MPI-safe: rank 0's pyvista rendering
    (off-screen rendering can fail on a headless cluster node, disk errors, etc.) must never
    raise past this point while the other ranks carry on into a later collective
    ``io4dolfinx.read_function`` call that rank 0 then never reaches - that would deadlock
    every other rank waiting on rank 0 forever. Catching the exception here and broadcasting
    ``ok`` (always rank 0's value, via ``root=0``) means every rank agrees, after this call,
    on whether to keep going - so any subsequent collective call is either entered by every
    rank or skipped by every rank together.
    """
    if ok and comm.rank == 0:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - any rendering failure is recoverable here
            ok = False
            logger.warning(f"Visualization failed at {step_name!r}, skipping it: {e!r}")
    return comm.bcast(ok, root=0)


def _visualize(
    conf: Config,
    v: dolfinx.fem.Function,
    tact: dolfinx.fem.Function,
    path: Path,
    times,
    post: Path,
    comm,
) -> None:
    """Render PNG/GIF previews (if pyvista is installed) into ``post``.

    ``io4dolfinx.read_function`` is collective over ``v``'s mesh communicator (it does
    parallel dof/cell-ownership exchanges internally), so it must be called on *every* rank,
    outside of any rank-0-only block, regardless of whether pyvista is installed there or
    whether rendering has failed. Only the actual pyvista grid/plotter/screenshot/gif calls
    (which would otherwise race writing the same file from every rank, and use rank 0's local
    mesh partition) are restricted to rank 0, each wrapped by :func:`_rank0_guarded` so a
    rendering failure there can never leave rank 0 out of step with the other ranks.
    """
    available = True
    if comm.rank == 0:
        try:
            import pyvista  # noqa: F401
        except ImportError:
            available = False
    available = comm.bcast(available, root=0)
    if not available:
        if comm.rank == 0:
            logger.warning(
                "pyvista is not installed, skipping visualization. Install it with "
                "'pip install pyvista' (or the 'docs' extra) to get PNG/GIF output from "
                "`beat post`.",
            )
        return

    if comm.size > 1 and comm.rank == 0:
        logger.warning(
            "beat post is running on more than one rank: the PNG/GIF previews show only rank "
            "0's partition of the mesh (the VTX output is complete). Run `beat post` on a "
            "single rank for full previews.",
        )

    import dolfinx.plot
    import pyvista

    pyvista.OFF_SCREEN = True

    io4dolfinx.read_function(path, v, time=times[-1], name="v")

    # Rank-0-only pyvista state (grid/plotter), threaded between the guarded steps below via
    # this dict rather than plain local variables/asserts - `ok` becoming False mid-way means
    # a later step is simply never attempted, so these never need to exist for mypy's sake.
    state: dict[str, Any] = {}

    def render_snapshots() -> None:
        grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(v.function_space))
        state["grid"] = grid
        grid.point_data["v"] = v.x.array
        plotter = pyvista.Plotter(off_screen=True)
        plotter.add_mesh(
            grid,
            scalars="v",
            show_edges=True,
            lighting=False,
            cmap="viridis",
            clim=[-90.0, 40.0],
        )
        voltage_png = post / "voltage_final.png"
        plotter.screenshot(voltage_png)
        plotter.close()
        logger.info(f"Final voltage snapshot saved to {voltage_png}")

        grid.point_data["activation_time"] = tact.x.array
        plotter = pyvista.Plotter(off_screen=True)
        plotter.add_mesh(
            grid,
            scalars="activation_time",
            show_edges=True,
            lighting=False,
            cmap="viridis",
        )
        activation_png = post / "activation_time_map.png"
        plotter.screenshot(activation_png)
        plotter.close()
        logger.info(f"Activation time map snapshot saved to {activation_png}")

    ok = _rank0_guarded(comm, True, "PNG snapshots", render_snapshots)
    if not ok or not conf.postprocess.make_gif:
        return

    io4dolfinx.read_function(path, v, time=times[0], name="v")
    gif_path = post / "voltage.gif"

    def start_gif() -> None:
        grid = state["grid"]
        grid.point_data["v"] = v.x.array
        plotter = pyvista.Plotter(off_screen=True)
        plotter.add_mesh(
            grid,
            scalars="v",
            show_edges=True,
            lighting=False,
            cmap="viridis",
            clim=[-90.0, 40.0],
        )
        plotter.open_gif(gif_path.as_posix())
        state["plotter"] = plotter

    ok = _rank0_guarded(comm, ok, "GIF setup", start_gif)

    for t in times:
        if not ok:
            break
        io4dolfinx.read_function(path, v, time=t, name="v")

        def write_frame(t=t) -> None:
            state["grid"].point_data["v"] = v.x.array
            state["plotter"].write_frame()

        ok = _rank0_guarded(comm, ok, f"GIF frame at t={t}", write_frame)

    if comm.rank == 0 and "plotter" in state:
        try:
            state["plotter"].close()
        except Exception as e:  # noqa: BLE001 - closing is best-effort, never fatal
            logger.warning(f"Failed to close the GIF plotter cleanly: {e!r}")
    if ok and comm.rank == 0:
        logger.info(f"Voltage animation saved to {gif_path}")
