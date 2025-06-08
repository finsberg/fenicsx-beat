import logging
import shutil
from pathlib import Path

from mpi4py import MPI

import cardiac_geometries as cg
import gotranx

from . import single_cell
from .config import Config
from .log import add_logfile_handler

logger = logging.getLogger(__name__)


def run_file(config: Path, comm=MPI.COMM_WORLD):
    conf = Config.parse_toml(config)
    return run(conf, comm=comm)


def run(conf: Config, comm=MPI.COMM_WORLD):
    # Creating output folder if it does not exist
    output_folder = conf.simulation.output_folder
    if output_folder.exists():
        logging.info(f"Output folder already exists: {output_folder}. Deleting old files.")
        shutil.rmtree(output_folder, ignore_errors=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Add file handlers to the logger
    add_logfile_handler(output_folder, comm=comm)
    logging.info(f"Output folder created: {output_folder}")

    module_path = conf.cell.module_name
    if not module_path.is_file():
        ode = gotranx.load_ode(conf.cell.ode_file)
        code = gotranx.cli.gotran2py.get_code(
            ode, scheme=[gotranx.schemes.Scheme[conf.cell.scheme]],
        )
        if comm.rank == 0:
            module_path.write_text(code)

    comm.barrier()

    cell_model = {}
    exec(module_path.read_text(), cell_model)
    breakpoint()
    init_states = single_cell.get_steady_state(
        fun=cell_model[conf.cell.scheme],
        init_states=cell_model["init_state_values"](),
        parameters=cell_model["init_parameter_values"](),
        outdir=output_folder / "states_0D",
        BCL=conf.cell.BCL.magnitude,
        nbeats=conf.cell.num_beats,
        track_indices=[
            cell_model["state_index"]("v"),
            cell_model["state_index"]("cai"),
        ],
        dt=conf.cell.dt.magnitude,
    )

    geo = cg.geometry.Geometry.from_folder(
        comm=comm,
        folder=conf.mesh.folder,
    )
    logger.info(geo)
