"""The wedge benchmark of Neic et al. 2017 (Figs. 2, 4, 5) for one model and resolution.

    mpirun -n 16 python scripts/reaction_eikonal_wedge.py --model rd --dx 0.1 --end-time 500

Models: "rd" (monodomain reaction-diffusion), "re+" (reaction-eikonal with diffusion), "re-"
(reaction-eikonal without diffusion). Writes <outdir>/<model>_dx<dx>.npz on rank 0.
"""

import argparse
import json
import time as walltime
from pathlib import Path

from mpi4py import MPI

import dolfinx
import gotranx
import numpy as np
import scifem

import beat
from beat.activation import ActivationTracker
from beat.eikonal import Source, solve_eikonal
from beat.reaction_eikonal import ExponentialFoot, FootCurrentODESolver, ODEOnlySolver

ROOT = Path(__file__).resolve().parents[1]
ODE_DIR = ROOT / "odes" / "tentusscher_panfilov_2006"
CV_FILE = Path(__file__).resolve().parent / "reaction_eikonal_cv.json"
T_R50 = 305.0  # ms, 50% repolarisation instant of the paper's Fig. 5
GRID_N = 101
NO_ACTIVATION = 1e9  # sentinel for dofs that never activated
PROBES = {"P_E": (25.0, 12.5, 2.5), "endo": (21.25, 12.5, 2.5), "M": (11.875, 12.5, 2.5)}
PETSC = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "ksp_rtol": 1e-8,
}
CELL_TYPES = {"EPI": "epi", "M": "M", "ENDO": "endo"}


def load_cell_models():
    models = {}
    for layer, name in CELL_TYPES.items():
        ode = gotranx.load_ode(ODE_DIR / f"tentusscher_panfilov_2006_{name}_cell.ode")
        code = gotranx.cli.gotran2py.get_code(
            ode,
            scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
        )
        ns: dict = {}
        exec(code, ns)
        models[beat.geometry.WEDGE_LAYERS[layer]] = ns
    return models


def cell_solver(V, v_pde, markers):
    models = load_cell_models()
    init = {k: m["init_state_values"]() for k, m in models.items()}
    return beat.odesolver.DolfinMultiODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=v_pde,
        markers=markers,
        init_states=init,
        parameters={k: m["init_parameter_values"](stim_amplitude=0.0) for k, m in models.items()},
        fun={k: m["generalized_rush_larsen"] for k, m in models.items()},
        num_states={k: len(v) for k, v in init.items()},
        v_index={k: m["state_index"]("V") for k, m in models.items()},
    )


def build(model, dx, cv, comm):
    """The solver, the potential to record, and the eikonal activation time (or None)"""
    wedge = beat.geometry.get_3D_wedge_geometry(comm, dx=dx)
    geo = wedge.geometry
    mesh = geo.mesh
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    markers = beat.geometry.wedge_layer_markers(V)
    g = beat.conductivities.default_conductivities("Neic2017_wedge")
    C_m = beat.units.ureg.Quantity(1.0, "uF/cm**2").to("uF/mm**2").magnitude
    time = dolfinx.fem.Constant(mesh, 0.0)
    t_a = None
    if model != "rd":
        sources = [Source((wedge.cell_tags, tag), t0) for tag, t0 in wedge.stimuli.values()]
        speeds = (cv["longitudinal"], cv["transverse"], cv["transverse"])
        t_a = solve_eikonal(mesh, speeds, (geo.f0, geo.s0, geo.n0), sources)
    if model == "re-":
        ode = FootCurrentODESolver(
            cell_solver(V, dolfinx.fem.Function(V), markers),
            t_a,
            ExponentialFoot(),
        )
        solver = ODEOnlySolver(ode)
        return solver, solver.v, t_a, wedge
    M = beat.conductivities.define_conductivity_tensor(f0=geo.f0, **g)
    if model == "rd":
        I_s = [
            beat.stimulation.define_stimulus(
                mesh=mesh,
                chi=g["chi"],
                time=time,
                subdomain_data=wedge.cell_tags,
                marker=tag,
                mesh_unit="mm",
                amplitude=50_000.0,
                start=t0,
                duration=2.0,
            )
            for tag, t0 in wedge.stimuli.values()
        ]
        pde = beat.MonodomainModel(
            time=time,
            mesh=mesh,
            M=M,
            I_s=I_s,
            C_m=C_m,
            dx=I_s[0].dZ,
            params={"petsc_options": PETSC},
        )
        ode = cell_solver(V, pde.state, markers)
    else:
        pde = beat.MonodomainModel(
            time=time,
            mesh=mesh,
            M=M,
            C_m=C_m,
            params={"petsc_options": PETSC},
        )
        ode = FootCurrentODESolver(cell_solver(V, pde.state, markers), t_a, ExponentialFoot())
    return beat.MonodomainSplittingSolver(pde=pde, ode=ode), pde.v, t_a, wedge


class PointEvaluator:
    """Evaluate a Function at fixed points; the point location is done once"""

    def __init__(self, mesh, points):
        self.comm = mesh.comm
        self.num_points = len(points)
        tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
        cand = dolfinx.geometry.compute_collisions_points(tree, points)
        adj = dolfinx.geometry.compute_colliding_cells(mesh, cand, points)
        self.indices = np.flatnonzero(adj.offsets[1:] - adj.offsets[:-1])
        self.cells = adj.array[adj.offsets[self.indices]]
        self.points = points[self.indices]

    def __call__(self, u):
        u.x.scatter_forward()
        out = np.full(self.num_points, -np.inf)
        out[self.indices] = u.eval(self.points, self.cells).ravel()
        self.comm.Allreduce(MPI.IN_PLACE, out, op=MPI.MAX)
        return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", choices=["rd", "re+", "re-"], required=True)
    p.add_argument("--dx", type=float, default=0.4)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--end-time", type=float, default=500.0)
    p.add_argument("--cv", choices=["measured", "nominal"], default="measured")
    p.add_argument("--outdir", type=Path, default=Path("results-reaction-eikonal"))
    p.add_argument("--vtx", action="store_true", help="also write v every 5 ms to a .bp file")
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    cv = (
        json.loads(CV_FILE.read_text())
        if args.cv == "measured"
        else {"longitudinal": 0.6, "transverse": 0.3}
    )
    start = walltime.perf_counter()
    solver, v, t_a, wedge = build(args.model, args.dx, cv, comm)

    gx, gy = np.meshgrid(np.linspace(0, 25, GRID_N), np.linspace(0, 25, GRID_N))
    grid = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 2.5)])
    line_x = np.linspace(0, 25, 501)
    line = np.column_stack([line_x, np.full_like(line_x, 12.5), np.full_like(line_x, 2.5)])
    probes = np.array(list(PROBES.values()))

    tracker = ActivationTracker(v, method="max_dvdt", threshold=-30.0, t0=0.0)
    out = args.outdir
    if comm.rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    name = f"{args.model}_dx{args.dx:g}"
    vtx = None
    if args.vtx:
        vtx = dolfinx.io.VTXWriter(comm, out / f"{name}.bp", [v], engine="BP4")
    probe_eval = PointEvaluator(wedge.geometry.mesh, probes)
    trace_t, traces = [0.0], [probe_eval(v)]
    v_r50_grid = np.full(gx.size, np.nan)
    v_r50_line = np.full(line_x.size, np.nan)
    t, step, dt = 0.0, 0, args.dt
    sample = max(1, round(0.5 / dt))  # every 0.5 ms after the upstrokes
    while t < args.end_time - 1e-9:
        solver.step((t, t + dt))
        t += dt
        step += 1
        tracker.update(t)
        if t <= 60.0 or step % sample == 0:
            trace_t.append(t)
            traces.append(probe_eval(v))
        if np.isnan(v_r50_line).all() and t >= T_R50 - 1e-9:
            v_r50_grid = scifem.evaluate_function(v, grid).ravel()
            v_r50_line = scifem.evaluate_function(v, line).ravel()
        if vtx is not None and step % round(5.0 / dt) == 0:
            vtx.write(t)
        if comm.rank == 0 and step % round(10.0 / dt) == 0:
            print(f"{name}: t = {t:.0f} ms ({walltime.perf_counter() - start:.0f} s)", flush=True)
    if vtx is not None:
        vtx.close()

    # NaN (never activated) must not enter the MPI max reduction in evaluate_function (the result is
    # then -inf or NaN depending on the rank layout): evaluate with a large sentinel instead.
    t_a_track = tracker.t_a
    t_a_track.x.array[np.isnan(t_a_track.x.array)] = NO_ACTIVATION
    activation = scifem.evaluate_function(t_a_track, grid).ravel()
    activation[~(activation < 1e3)] = np.nan  # also mixed (interpolated) cells
    eikonal = (
        scifem.evaluate_function(t_a, grid).ravel() if t_a is not None else np.full(gx.size, np.nan)
    )
    num_vertices = v.function_space.dofmap.index_map.size_global
    if comm.rank == 0:
        np.savez(
            out / f"{name}.npz",
            grid_x=gx,
            grid_y=gy,
            t_a=activation.reshape(gx.shape),
            t_a_eikonal=eikonal.reshape(gx.shape),
            line_x=line_x,
            v_r50_line=v_r50_line,
            v_r50_grid=v_r50_grid.reshape(gx.shape),
            trace_t=np.array(trace_t),
            traces=np.array(traces),
            probe_names=np.array(list(PROBES)),
            cv=np.array([cv["longitudinal"], cv["transverse"]]),
            dx=args.dx,
            dt=dt,
            wall_time=walltime.perf_counter() - start,
            num_vertices=num_vertices,
            ranks=comm.size,
        )
        print(f"wrote {out / f'{name}.npz'}")


if __name__ == "__main__":
    main()
