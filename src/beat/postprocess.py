import csv
import json
import logging
from pathlib import Path

from mpi4py import MPI

import dolfinx
import io4dolfinx
import numpy as np
import scifem

from .config import Config
from .ecg import ECGRecovery
from .runner import build_conductivity_tensor, checkpoint_path, load_geometry

logger = logging.getLogger(__name__)


def _read_checkpoint(conf: Config, comm):
    """Load the geometry (from ``conf.mesh.folder``) and a ``v`` Function on *that same mesh
    object*, ready to be filled from the checkpoint written by `beat run`.

    The checkpoint is written with ``io4dolfinx.write_function_on_input_mesh``, which pairs with
    a Function built directly on the mesh loaded from ``conf.mesh.folder`` (rather than a
    freshly-``read_mesh``'d one) - required so that ``v`` can be combined in one UFL form with M
    (built from that same geometry's fiber field) for the ECG recovery.
    """
    ckpt_path = checkpoint_path(conf.simulation.output_folder)
    if not ckpt_path.exists():
        raise ValueError(
            f"No checkpoint found at {ckpt_path}. Run `beat run <config>` first.",
        )
    geo = load_geometry(conf, comm=comm)
    V = dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 1))
    v = dolfinx.fem.Function(V, name="v")
    times = io4dolfinx.read_timestamps(comm=comm, filename=ckpt_path, function_name="v")
    if len(times) == 0:
        raise ValueError(f"No saved timesteps found in checkpoint {ckpt_path}")
    return ckpt_path, geo, v, times


def _points_or_raise(conf: Config) -> dict[str, list[float]]:
    if not conf.postprocess.points:
        raise ValueError(
            "No points configured. Add a [postprocess.points] section to the config, e.g. "
            "`points = {P1 = [0.0, 0.0, 0.0]}` (coordinates in mesh.unit).",
        )
    return conf.postprocess.points


def run_ecg_file(config: Path, comm=MPI.COMM_WORLD) -> Path:
    conf = Config.parse_toml(config)
    return run_ecg(conf, comm=comm)


def run_ecg(conf: Config, comm=MPI.COMM_WORLD) -> Path:
    """Recover the extracellular potential (pseudo-ECG) at ``postprocess.points`` from a
    previously saved `beat run` checkpoint, and save the resulting time series to CSV (and a
    PNG plot, if matplotlib is available).
    """
    points = _points_or_raise(conf)
    output_folder = conf.simulation.output_folder
    ckpt_path, geo, v, times = _read_checkpoint(conf, comm)

    M = build_conductivity_tensor(conf, geo)
    C_m = conf.ep.C_m.to(f"uF/{conf.mesh.unit}**2").magnitude

    ecg = ECGRecovery(v=v, sigma_b=conf.postprocess.sigma_b, C_m=C_m, M=M)
    names = list(points)
    forms = {name: ecg.eval(points[name]) for name in names}

    values: dict[str, list[float]] = {name: [] for name in names}
    for t in times:
        io4dolfinx.read_function(ckpt_path, v, time=t, name="v")
        ecg.solve()
        for name in names:
            values[name].append(
                geo.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(forms[name]), op=MPI.SUM),
            )

    csv_path = output_folder / "ecg.csv"
    if comm.rank == 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", *names])
            for i, t in enumerate(times):
                writer.writerow([t, *(values[name][i] for name in names)])
        logger.info(f"ECG values saved to {csv_path}")
        _plot_ecg(times, values, output_folder / "ecg.png")
    comm.barrier()
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


def run_post_file(config: Path, comm=MPI.COMM_WORLD) -> Path:
    conf = Config.parse_toml(config)
    return run_post(conf, comm=comm)


def run_post(conf: Config, comm=MPI.COMM_WORLD) -> Path:
    """Compute a full-mesh local activation time map (and, at ``postprocess.points``, activation
    times as scalars) from a previously saved `beat run` checkpoint, and (if pyvista is
    installed) render PNG/GIF visualizations.
    """
    output_folder = conf.simulation.output_folder
    ckpt_path, geo, v, times = _read_checkpoint(conf, comm)

    threshold = conf.postprocess.activation_threshold
    tact = dolfinx.fem.Function(v.function_space, name="activation_time")
    tact.x.array[:] = -1.0
    for t in times:
        io4dolfinx.read_function(ckpt_path, v, time=t, name="v")
        pending = tact.x.array < 0.0
        tact.x.array[pending & (v.x.array >= threshold)] = t

    xdmf_path = output_folder / "activation_time.xdmf"
    with dolfinx.io.XDMFFile(comm, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(geo.mesh)
        xdmf.write_function(tact)
    logger.info(f"Activation time map saved to {xdmf_path}")

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

    json_path = output_folder / "activation_times.json"
    if comm.rank == 0:
        json_path.write_text(json.dumps(point_results, indent=2))
        logger.info(f"Activation times at points saved to {json_path}")
    comm.barrier()

    _visualize(conf, v=v, tact=tact, ckpt_path=ckpt_path, times=times)

    return xdmf_path


def _visualize(conf: Config, v: dolfinx.fem.Function, tact: dolfinx.fem.Function, ckpt_path, times):
    try:
        import pyvista
    except ImportError:
        logger.warning(
            "pyvista is not installed, skipping visualization. Install it with "
            "'pip install pyvista' (or the 'docs' extra) to get PNG/GIF output from `beat post`.",
        )
        return
    import dolfinx.plot

    output_folder = conf.simulation.output_folder
    pyvista.OFF_SCREEN = True

    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(v.function_space))

    io4dolfinx.read_function(ckpt_path, v, time=times[-1], name="v")
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
    voltage_png = output_folder / "voltage_final.png"
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
    activation_png = output_folder / "activation_time_map.png"
    plotter.screenshot(activation_png)
    plotter.close()
    logger.info(f"Activation time map snapshot saved to {activation_png}")

    if conf.postprocess.make_gif:
        io4dolfinx.read_function(ckpt_path, v, time=times[0], name="v")
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
        gif_path = output_folder / "voltage.gif"
        plotter.open_gif(gif_path.as_posix())
        for t in times:
            io4dolfinx.read_function(ckpt_path, v, time=t, name="v")
            grid.point_data["v"] = v.x.array
            plotter.write_frame()
        plotter.close()
        logger.info(f"Voltage animation saved to {gif_path}")
