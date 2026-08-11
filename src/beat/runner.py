import logging
import shutil
from pathlib import Path

from mpi4py import MPI

import cardiac_geometries as cg
import dolfinx
import gotranx

from . import single_cell
from .conductivities import define_conductivity_tensor
from .config import Config
from .log import add_logfile_handler
from .monodomain_model import MonodomainModel
from .monodomain_solver import MonodomainSplittingSolver
from .odesolver import DolfinODESolver
from .stimulation import define_stimulus

logger = logging.getLogger(__name__)


def run_file(config: Path, comm=MPI.COMM_WORLD) -> Path:
    conf = Config.parse_toml(config)
    return run(conf, comm=comm)


def run(conf: Config, comm=MPI.COMM_WORLD) -> Path:
    # Creating output folder if it does not exist
    output_folder = conf.simulation.output_folder
    if output_folder.exists():
        logger.info(f"Output folder already exists: {output_folder}. Deleting old files.")
        shutil.rmtree(output_folder, ignore_errors=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Add file handlers to the logger
    add_logfile_handler(output_folder, comm=comm)
    logger.info(f"Output folder created: {output_folder}")

    mesh_unit = conf.mesh.unit

    module_path = conf.cell.module_name
    if not module_path.is_file():
        logger.info(f"Generating cell model code from {conf.cell.ode_file}")
        ode = gotranx.load_ode(conf.cell.ode_file)
        code = gotranx.cli.gotran2py.get_code(
            ode,
            scheme=[gotranx.schemes.Scheme[conf.cell.scheme]],
        )
        if comm.rank == 0:
            module_path.write_text(code)
    comm.barrier()

    cell_model: dict = {}
    exec(module_path.read_text(), cell_model)

    fun = cell_model[conf.cell.scheme]
    init_states = cell_model["init_state_values"]()
    parameters = cell_model["init_parameter_values"]()
    v_index = cell_model["state_index"](conf.cell.v_name)
    track_indices = [cell_model["state_index"](name) for name in conf.cell.track_indices]

    logger.info(f"Computing steady state initial conditions ({conf.cell.num_beats} beats)")
    init_states = single_cell.get_steady_state(
        fun=fun,
        init_states=init_states,
        parameters=parameters,
        outdir=output_folder / "init_states",
        BCL=conf.cell.BCL.to("ms").magnitude,
        nbeats=conf.cell.num_beats,
        track_indices=track_indices,
        dt=conf.cell.dt.to("ms").magnitude,
    )

    logger.info(f"Reading geometry from {conf.mesh.folder}")
    geo = cg.geometry.Geometry.from_folder(comm=comm, folder=conf.mesh.folder)
    if geo.f0 is None:
        raise ValueError(f"No fiber field found in geometry loaded from {conf.mesh.folder}")

    marker_name = conf.stimulus.marker
    if marker_name not in geo.markers:
        raise ValueError(
            f"Stimulus marker {marker_name!r} not found in geometry markers "
            f"{sorted(geo.markers)}",
        )
    marker_id, marker_dim = geo.markers[marker_name]
    facet_dim = geo.mesh.topology.dim - 1
    if marker_dim != facet_dim or geo.ffun is None:
        raise ValueError(
            f"Stimulus marker {marker_name!r} must refer to a facet marker "
            f"(dimension {facet_dim}), got dimension {marker_dim}",
        )

    time = dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0))
    I_s = define_stimulus(
        mesh=geo.mesh,
        chi=conf.ep.chi,
        time=time,
        subdomain_data=geo.ffun,
        marker=marker_id,
        mesh_unit=mesh_unit,
        amplitude=conf.stimulus.amplitude,
        duration=conf.stimulus.duration.to("ms").magnitude,
        start=conf.stimulus.start.to("ms").magnitude,
    )

    M = define_conductivity_tensor(
        chi=conf.ep.chi,
        f0=geo.f0,
        g_il=conf.ep.conductivity.sigma_il,
        g_it=conf.ep.conductivity.sigma_it,
        g_el=conf.ep.conductivity.sigma_el,
        g_et=conf.ep.conductivity.sigma_et,
    )

    C_m = conf.ep.C_m.to(f"uF/{mesh_unit}**2").magnitude
    pde = MonodomainModel(time=time, mesh=geo.mesh, M=M, I_s=I_s, C_m=C_m)

    V_ode = dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 1))
    ode = DolfinODESolver(
        v_ode=dolfinx.fem.Function(V_ode),
        v_pde=pde.state,
        fun=fun,
        init_states=init_states,
        parameters=parameters,
        num_states=len(init_states),
        v_index=v_index,
    )

    solver = MonodomainSplittingSolver(pde=pde, ode=ode, theta=conf.simulation.theta)

    dt = conf.simulation.dt.to("ms").magnitude
    BCL = conf.simulation.BCL.to("ms").magnitude
    end_time = conf.simulation.num_beats * BCL
    save_freq = max(1, round(conf.simulation.save_every_ms / dt))

    result_path = output_folder / "result.bp"
    shutil.rmtree(result_path, ignore_errors=True)

    t = 0.0
    i = 0
    with dolfinx.io.VTXWriter(comm, result_path, [solver.pde.state], engine="BP4") as vtx:
        while t < end_time - 1e-10:
            if i % save_freq == 0:
                logger.info(f"Solving for t={t:.3f} ms")
                vtx.write(t)
            solver.step((t, t + dt))
            t += dt
            i += 1
        vtx.write(t)

    logger.info(f"Simulation finished. Results saved to {result_path}")
    return output_folder
