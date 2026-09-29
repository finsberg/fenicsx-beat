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

from mpi4py import MPI

import dolfinx
import io4dolfinx
import numpy as np
import scifem

from ..ecg import ECGRecovery
from .config import Config, ConfigError
from .geometry import build_conductivity, build_geometry
from .runner import RESULTS, _on_rank0, read_result_times

logger = logging.getLogger(__name__)


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
    geo = build_geometry(conf.geometry, comm)
    V = dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 1))
    v = dolfinx.fem.Function(V, name="v")
    times = read_result_times(path, comm)  # unique + sorted (restart-safe)
    post = conf.output.folder / "post"
    _on_rank0(comm, OSError, lambda: post.mkdir(parents=True, exist_ok=True))
    return path, geo, v, times, post


def _points_or_raise(conf: Config) -> dict[str, list[float]]:
    if not conf.postprocess.points:
        raise ConfigError(
            "No points configured. Add a [postprocess.points] section to the config, e.g. "
            "`points = {P1 = [0.0, 0.0, 0.0]}` (coordinates in geometry.unit).",
        )
    return conf.postprocess.points


def run_ecg(conf: Config, comm=MPI.COMM_WORLD) -> Path:
    """Recover the extracellular potential (pseudo-ECG) at ``postprocess.points`` from a
    previously saved `beat run` output, and save the resulting time series to
    ``post/ecg.csv`` (and ``post/ecg.png``, if matplotlib is available).
    """
    points = _points_or_raise(conf)
    path, geo, v, times, post = _open_results(conf, comm)

    M = build_conductivity(conf.ep, geo, conf.geometry.fibers)
    C_m = conf.ep.C_m.to(f"uF/{conf.geometry.unit}**2").magnitude

    ecg = ECGRecovery(v=v, sigma_b=conf.postprocess.sigma_b, C_m=C_m, M=M)
    names = list(points)
    forms = {name: ecg.eval(points[name]) for name in names}

    values: dict[str, list[float]] = {name: [] for name in names}
    for t in times:
        io4dolfinx.read_function(path, v, time=t, name="v")
        ecg.solve()
        for name in names:
            values[name].append(
                geo.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(forms[name]), op=MPI.SUM),
            )

    csv_path = post / "ecg.csv"

    def write_outputs() -> None:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", *names])
            for i, t in enumerate(times):
                writer.writerow([t, *(values[name][i] for name in names)])
        logger.info(f"ECG values saved to {csv_path}")
        _plot_ecg(times, values, post / "ecg.png")

    _on_rank0(comm, OSError, write_outputs)
    return csv_path


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


def run_post(conf: Config, comm=MPI.COMM_WORLD) -> Path:
    """Compute a full-mesh local activation time map (and, at ``postprocess.points``, activation
    times as scalars) from a previously saved `beat run` output, convert ``v`` to VTX for
    ParaView (if ``postprocess.vtx``), and (if pyvista is installed) render PNG/GIF
    visualizations. Everything is written into ``post/``.
    """
    path, geo, v, times, post = _open_results(conf, comm)

    threshold = conf.postprocess.activation_threshold
    tact = dolfinx.fem.Function(v.function_space, name="activation_time")
    tact.x.array[:] = -1.0
    for t in times:
        io4dolfinx.read_function(path, v, time=t, name="v")
        pending = tact.x.array < 0.0
        tact.x.array[pending & (v.x.array >= threshold)] = t

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
                # scifem returns a non-finite value for points outside the mesh (e.g. a
                # far-field point meant only for `beat ecg`) - null rather than -1.0 (our
                # "not yet activated" sentinel for points inside the mesh) since it's not a
                # meaningful activation time at all, and -inf/nan aren't valid JSON.
                point_results[name] = None
                logger.warning(
                    f"Point {name!r} ({conf.postprocess.points[name]}) lies outside the mesh "
                    "domain; activation time is undefined there (recorded as null).",
                )

    json_path = post / "activation_times.json"
    _on_rank0(comm, OSError, lambda: json_path.write_text(json.dumps(point_results, indent=2)))
    logger.info(f"Activation times at points saved to {json_path}")

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
    parallel dof/cell-ownership exchanges internally), so it must be called on *every* rank
    regardless of whether pyvista is installed there - only the actual pyvista
    grid/plotter/screenshot calls (which would otherwise race writing the same file from every
    rank) are restricted to rank 0, using rank 0's local mesh partition.
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

    import dolfinx.plot
    import pyvista

    pyvista.OFF_SCREEN = True

    io4dolfinx.read_function(path, v, time=times[-1], name="v")
    grid = None
    if comm.rank == 0:
        grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(v.function_space))
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

    if conf.postprocess.make_gif:
        io4dolfinx.read_function(path, v, time=times[0], name="v")
        gif_path = post / "voltage.gif"
        plotter = None
        if comm.rank == 0:
            assert grid is not None
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
        for t in times:
            io4dolfinx.read_function(path, v, time=t, name="v")
            if comm.rank == 0:
                assert grid is not None and plotter is not None
                grid.point_data["v"] = v.x.array
                plotter.write_frame()
        if comm.rank == 0:
            assert plotter is not None
            plotter.close()
            logger.info(f"Voltage animation saved to {gif_path}")
