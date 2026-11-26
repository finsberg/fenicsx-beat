"""One full beat of the Durrer activation demo, for the movie and the ECG in the docs.

This is the setup of ``demos/durrer_activation.py`` (the same ellipsoid, activation sequence and
calibration of the conductivities) on a finer mesh of about 1 mm, with transmural cell types
(endo/mid/epi ToR-ORd, paced to steady state), run over a whole beat. It writes, into
``--static`` (``docs/_static``):

* ``durrer_activation.gif``: v over the beat, on the epicardium and on a long-axis cut;
* ``durrer_activation_ecg.png``: the 12-lead ECG, at the electrodes of ``demos/biv_endocardial.py``;

and into ``--outdir`` the geometry, the steady states and ``results.npz`` (time, v at every node
every ``--frame-every`` ms, the electrode potentials and leads every ms). It is not run when the
docs are built. Run it from the root of the repository, in serial (about 25 minutes):

    python3 scripts/durrer_activation_full.py

``--figures-only`` redraws the movie and the ECG from an earlier ``results.npz``.
"""

import argparse
import json
import logging
import time as wallclock
from pathlib import Path

from mpi4py import MPI

import cardiac_geometries
import dolfinx
import gotranx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista

import beat
from beat.activation_sequence import EndocardialActivation

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("durrer_activation_full")

# The electrodes of demos/biv_endocardial.py, on a torso around the same ellipsoid (cm)
ELECTRODES = dict(
    RA=(-15.0, 0.0, -10.0),
    LA=(4.0, -12.0, -7.0),
    RL=(0.0, 20.0, 3.0),
    LL=(17.0, 11.0, 7.0),
    V1=(-3.0, 4.0, -9.0),
    V2=(0.0, 2.0, -8.0),
    V3=(3.0, 1.0, -8.0),
    V4=(6.0, 1.0, -6.0),
    V5=(10.0, 2.0, 0.0),
    V6=(10.0, -6.0, 2.0),
)
# Layer markers and the ToR-ORd celltype parameter of each layer
LAYERS = {"mid": (0, 2), "endo": (1, 0), "epi": (2, 1)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=Path("results-durrer-activation-full"))
    parser.add_argument("--static", type=Path, default=ROOT / "docs" / "_static")
    parser.add_argument(
        "--char-length",
        type=float,
        default=0.1,
        help="mesh size (cm); the demo uses 0.3",
    )
    parser.add_argument(
        "--cv",
        type=float,
        nargs=2,
        default=(0.6, 0.3),
        metavar=("ALONG", "ACROSS"),
        help="conduction velocities (mm/ms) along and across the fibres that the Bishop "
        "conductivities are calibrated to, as in the demo",
    )
    parser.add_argument("--dt", type=float, default=0.05, help="ms")
    parser.add_argument("--T", type=float, default=500.0, help="ms")
    parser.add_argument("--nbeats", type=int, default=200, help="single-cell beats to steady state")
    parser.add_argument("--frame-every", type=float, default=2.0, help="ms between saved frames")
    parser.add_argument(
        "--movie-fast-after",
        type=float,
        default=100.0,
        help="the movie has every saved frame until then (ms), and every --movie-every ms after",
    )
    parser.add_argument(
        "--movie-every",
        type=float,
        default=6.0,
        help="ms, after --movie-fast-after",
    )
    parser.add_argument("--movie-end", type=float, default=360.0, help="ms")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="redraw the movie and the ECG from the results.npz of an earlier run",
    )
    return parser.parse_args(argv)


def cell_model(outdir: Path):
    """The `rush_larsen_2` code of the endocardial ToR-ORd model (all cell types)"""
    path = outdir / "ToRORd_dynCl_endo.py"
    if not path.is_file() or "def rush_larsen_2" not in path.read_text():
        ode = gotranx.load_ode(ROOT / "odes" / "torord" / "ToRORd_dynCl_endo.ode")
        scheme = [gotranx.schemes.Scheme.rush_larsen_2]
        path.write_text(gotranx.cli.gotran2py.get_code(ode, scheme=scheme))
    model: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), model)
    return model


def make_movie(path, grid, frames, times, frame_obj):
    """Two views: the epicardium from the RV side and a cut along the long axis, through both
    cavities"""
    grid = grid.copy()
    grid.point_data["ids"] = np.arange(grid.n_points)
    grid.point_data["v"] = frames[0]
    center = frame_obj.base_center - 0.5 * frame_obj.length * frame_obj.long_axis
    surface = grid.extract_surface()

    def clip():
        # The posterior half (clip keeps the side opposite to the normal), seen from the front.
        # The clip interpolates v to the points it creates on the cut plane.
        return grid.clip(normal=frame_obj.anterior, origin=center).extract_surface()

    cut = clip()
    meshes = {"epicardium, from the RV side": surface, "cut along the long axis": cut}
    cameras = {
        "epicardium, from the RV side": center + 16.0 * frame_obj.septal,
        "cut along the long axis": center + 16.0 * frame_obj.anterior,
    }
    plotter = pyvista.Plotter(shape=(1, 2), window_size=(1000, 520), off_screen=True)
    for column, (title, mesh) in enumerate(meshes.items()):
        plotter.subplot(0, column)
        if title.startswith("epicardium"):
            mesh.point_data["v"] = frames[0][mesh.point_data["ids"]]
        plotter.add_mesh(
            mesh,
            scalars="v",
            cmap="viridis",
            clim=[-90.0, 40.0],
            smooth_shading=True,
            scalar_bar_args=dict(title="v (mV)", color="black"),
        )
        plotter.camera.focal_point = center
        plotter.camera.position = cameras[title]
        plotter.camera.up = frame_obj.long_axis
        plotter.add_text(title, color="black", font_size=10)
    plotter.open_gif(str(path))
    for values, t in zip(frames, times):
        grid.point_data["v"] = values
        surface.point_data["v"] = values[surface.point_data["ids"]]
        cut.point_data["v"] = clip().point_data["v"]
        for column in range(2):
            plotter.subplot(0, column)
            plotter.add_text(
                f"t = {t:3.0f} ms",
                name="time",
                position="lower_left",
                color="black",
                font_size=10,
            )
        plotter.write_frame()
    plotter.close()


def plot_ecg(path, times, leads, layout, t_activation):
    fig, ax = plt.subplots(3, 4, sharex=True, figsize=(12, 7))
    for i, row in enumerate(layout):
        for j, name in enumerate(row):
            axi = ax[i, j]
            axi.plot(times, leads[name], color="k", lw=1.0)
            axi.axvspan(0.0, t_activation, color="tab:orange", alpha=0.15, lw=0)
            axi.axhline(0.0, color="0.7", lw=0.5)
            axi.set_title(name)
            if i == len(layout) - 1:
                axi.set_xlabel("time (ms)")
    fig.suptitle("12-lead ECG (shaded: ventricular activation)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def simulate(args, geo, activation, mesh_unit):
    """Run the beat; save and return the ECG, the frames and the activation times"""
    ureg = beat.units.ureg
    # --- Cells: endo/mid/epi layers, paced to steady state -------------------------------------
    model = cell_model(args.outdir)
    V = dolfinx.fem.functionspace(geo.mesh, ("P", 1))
    layers = beat.utils.expand_layer_biv(
        V=V,
        ft=geo.ffun,
        endo_lv_marker=geo.markers["LV"][0],
        endo_rv_marker=geo.markers["RV"][0],
        epi_marker=geo.markers["EPI"][0],
        endo_size=0.3,
        epi_size=0.3,
        output_mid_marker=LAYERS["mid"][0],
        output_endo_marker=LAYERS["endo"][0],
        output_epi_marker=LAYERS["epi"][0],
    )
    fun = model["rush_larsen_2"]
    v_index = model["state_index"]("v")
    init_states, parameters = {}, {}
    for name, (marker, celltype) in LAYERS.items():
        logger.info(f"Steady state of the {name} cells ({args.nbeats} beats)")
        init_states[marker] = beat.single_cell.get_steady_state(
            fun=fun,
            init_states=model["init_state_values"](),
            parameters=model["init_parameter_values"](celltype=celltype),
            outdir=args.outdir / "steady_states" / name,
            BCL=1000,
            nbeats=args.nbeats,
            dt=args.dt,
            track_indices=[v_index, model["state_index"]("cai")],
        )
        parameters[marker] = model["init_parameter_values"](i_Stim_Amplitude=0.0, celltype=celltype)

    # --- Tissue ----------------------------------------------------------------------------------
    # Calibrated for this mesh size and time step, with the endocardial cells at steady state
    endo = LAYERS["endo"][0]
    calibration = beat.conduction_velocity.calibrate_conductivities(
        ureg.Quantity(args.cv[0], "mm/ms"),
        ureg.Quantity(args.cv[1], "mm/ms"),
        conductivities=beat.conductivities.default_conductivities("Bishop"),
        fun=fun,
        init_states=init_states[endo],
        parameters=parameters[endo],
        v_index=v_index,
        dx=ureg.Quantity(args.char_length, "cm"),
        dt=args.dt,
    )
    logger.info(f"Calibrated: scales {calibration.scales}, velocities {calibration.velocities}")
    conductivities = calibration.conductivities
    C_m = ureg.Quantity(1.0, "uF/cm**2").to(f"uF/{mesh_unit}**2").magnitude
    M = beat.conductivities.define_conductivity_tensor(f0=geo.f0, **conductivities)
    time = dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0))
    pde = beat.MonodomainModel(
        time=time,
        mesh=geo.mesh,
        M=M,
        I_s=activation.stimulus(time, args.dt),
        C_m=C_m,
    )
    ode = beat.odesolver.DolfinMultiODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=pde.state,
        markers=layers,
        num_states={k: len(s) for k, s in init_states.items()},
        fun={k: fun for k in init_states},
        init_states=init_states,
        parameters=parameters,
        v_index={k: v_index for k in init_states},
    )
    # Strang splitting (the default) without merged half-steps: we read `v` after every step
    # (activation times), which merged half-steps would delay by up to half a time step.
    solver = beat.SplittingSolver(pde=pde, ode=ode)

    # --- ECG: electrode potentials every ms -----------------------------------------------------
    ecg = beat.ecg.ECGRecovery(v=solver.v, sigma_b=1.0, C_m=C_m, M=M)
    system = beat.ecg.twelve_lead()
    probe = beat.ecg.ElectrodePotentials(ecg, system.points(ELECTRODES))
    tracker = beat.activation.ActivationTracker(solver.v, method="threshold", threshold=-30.0)

    steps = round(args.T / args.dt)
    ecg_every = round(1.0 / args.dt)
    frame_every = round(args.frame_every / args.dt)
    ecg_times, potentials = [0.0], {k: [v] for k, v in probe().items()}
    frame_times, frames = [0.0], [solver.v.x.array.copy()]
    start = wallclock.perf_counter()
    t = 0.0
    for step in range(1, steps + 1):
        solver.step((t, t + args.dt))
        t = step * args.dt
        tracker.update(t)
        if step % ecg_every == 0:
            ecg_times.append(t)
            for k, value in probe().items():
                potentials[k].append(value)
        if step % frame_every == 0:
            frame_times.append(t)
            frames.append(solver.v.x.array.copy())
        if step % round(50.0 / args.dt) == 0:
            v = solver.v.x.array
            logger.info(f"t = {t:5.0f} ms, v in [{v.min():6.1f}, {v.max():5.1f}] mV")
    elapsed = wallclock.perf_counter() - start

    t_a = tracker.t_a.x.array
    t_activation = float(np.nanmax(t_a))
    leads = system.leads({k: np.asarray(v) for k, v in potentials.items()})
    summary = {
        "num_vertices": int(V.dofmap.index_map.size_global),
        "dt": args.dt,
        "T": args.T,
        "conduction_velocity_targets": list(args.cv),
        "calibration": {"scales": calibration.scales, "velocities": calibration.velocities},
        "last_activation_ms": t_activation,
        "not_activated": int(np.isnan(t_a).sum()),
        "simulation_seconds": elapsed,
        "activation": activation.summary(),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.outdir / "results.npz",
        ecg_times=np.asarray(ecg_times),
        frame_times=np.asarray(frame_times),
        frames=np.asarray(frames, dtype=np.float32),
        activation_time=t_a,
        **{f"electrode_{k}": np.asarray(v) for k, v in potentials.items()},
        **{f"lead_{k}": np.asarray(v) for k, v in leads.items()},
    )
    logger.info(
        f"Simulated {args.T} ms in {elapsed:.0f} s; last activation at {t_activation:.1f} ms",
    )
    return dict(
        ecg_times=np.asarray(ecg_times),
        leads=leads,
        frame_times=np.asarray(frame_times),
        frames=np.asarray(frames),
        t_activation=t_activation,
    )


def load_results(path):
    """What ``simulate`` returns, from its results.npz"""
    data = np.load(path)
    return dict(
        ecg_times=data["ecg_times"],
        leads={k[len("lead_") :]: data[k] for k in data.files if k.startswith("lead_")},
        frame_times=data["frame_times"],
        frames=data["frames"],
        t_activation=float(np.nanmax(data["activation_time"])),
    )


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)
    comm = MPI.COMM_WORLD
    if comm.size > 1:
        raise SystemExit("Run this script in serial: the movie needs v at every node on one rank")
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.static.mkdir(parents=True, exist_ok=True)
    mesh_unit = "cm"
    ureg = beat.units.ureg

    # --- Geometry and activation sequence, as in the demo -------------------------------------
    geodir = args.outdir / f"geo_h{args.char_length:g}"
    if not geodir.exists():
        cardiac_geometries.mesh.biv_ellipsoid(
            comm=comm,
            outdir=geodir,
            char_length=args.char_length,
            create_fibers=True,
        )
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)
    activation = EndocardialActivation.from_geometry(
        geo,
        mesh_unit=mesh_unit,
        conduction_velocity=ureg.Quantity(2.0, "mm/ms"),
    )

    if args.figures_only:
        results = load_results(args.outdir / "results.npz")
    else:
        results = simulate(args, geo, activation, mesh_unit)

    # --- Figures for the docs --------------------------------------------------------------------
    system = beat.ecg.twelve_lead()
    assert system.layout is not None
    plot_ecg(
        args.static / "durrer_activation_ecg.png",
        results["ecg_times"],
        results["leads"],
        system.layout,
        results["t_activation"],
    )
    # Every saved frame during the activation, fewer during the plateau and repolarisation
    t = results["frame_times"]
    phase = np.round(t - args.movie_fast_after, 6) % args.movie_every
    keep = (t <= args.movie_fast_after) | ((t <= args.movie_end) & np.isclose(phase, 0.0))
    V = dolfinx.fem.functionspace(geo.mesh, ("P", 1))
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V))
    make_movie(
        args.static / "durrer_activation.gif",
        grid,
        results["frames"][keep],
        t[keep],
        activation.frame,
    )
    logger.info(f"Wrote the movie ({keep.sum()} frames) and the ECG to {args.static}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
