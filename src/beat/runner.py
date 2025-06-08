import logging
import shutil
from pathlib import Path

from mpi4py import MPI

import cardiac_geometries as cg

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

    geo = cg.geometry.Geometry.from_folder(
        comm=comm,
        folder=conf.mesh.folder,
    )
    logger.info(geo)
