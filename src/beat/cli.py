import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from mpi4py import MPI

from .log import setup_logging

logger = logging.getLogger(__name__)


def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Root parser
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just print the command and do not run it",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print more information",
    )
    parser.add_argument(
        "--log-all-cpus",
        action="store_true",
        help="Log on all CPUs",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Version parser
    subparsers.add_parser("version", help="Display version information")

    # Run simulation parser
    run_parser = subparsers.add_parser("run", help="Run simulations")
    run_parser.add_argument(
        "config",
        type=Path,
        default="config.toml",
        help="Path to the configuration file",
    )

    validate_config_parser = subparsers.add_parser(
        "validate-config",
        help="Validate the configuration file",
    )
    validate_config_parser.add_argument(
        "config",
        type=Path,
        default="config.toml",
        help="Path to the configuration file to validate",
    )

    # ECG parser
    subparsers.add_parser("ecg", help="Compute ECG signals")

    # Postprocessing parser
    subparsers.add_parser("post", help="Postprocessing")

    return parser


def display_version_info():
    from petsc4py import PETSc

    import dolfinx

    from . import __version__

    logger.info(f"fenicsx-beat: {__version__}")
    logger.info(f"dolfinx: {dolfinx.__version__}")
    logger.info(f"mpi4py: {MPI.Get_version()}")
    logger.info(f"petsc4py: {PETSc.Sys.getVersion()}")


def dispatch(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None) -> int:
    args = vars(parser.parse_args(argv))
    level = logging.DEBUG if args.pop("verbose") else logging.INFO
    log_all_cpus = args.pop("log_all_cpus")
    comm = MPI.COMM_WORLD
    setup_logging(level=level, log_all_cpus=log_all_cpus, comm=comm)

    dry_run = args.pop("dry_run")
    command = args.pop("command")

    if dry_run:
        logger.info("Dry run: %s", command)
        logger.info("Arguments: %s", args)
        return 0

    try:
        if command == "version":
            display_version_info()
        elif command == "run":
            from .runner import run_file

            run_file(**args, comm=comm)

        elif command == "validate-config":
            from .config import Config

            config_path = args.pop("config")
            if not config_path.exists():
                raise ValueError(f"Configuration file {config_path} does not exist.")
            Config.parse_toml(config_path)
            logger.info(f"Configuration file {config_path} is valid.")
        elif command == "ecg":
            return NotImplemented
        elif command == "post":
            return NotImplemented
        else:
            logger.error(f"Unknown command {command}")
            parser.print_help()
            return 1
    except ValueError as e:
        logger.error(e)
        return 1

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = setup_parser()
    return dispatch(parser, argv)
