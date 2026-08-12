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

    # Init config parser
    init_parser = subparsers.add_parser(
        "init",
        help="Create a configuration file with default values",
    )
    init_parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path("config.toml"),
        help="Path to the configuration file to create",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the configuration file if it already exists",
    )

    # ECG parser
    ecg_parser = subparsers.add_parser(
        "ecg",
        help="Recover the pseudo-ECG at the points in [postprocess.points] from a previous run",
    )
    ecg_parser.add_argument(
        "config",
        type=Path,
        default="config.toml",
        help="Path to the configuration file",
    )

    # Postprocessing parser
    post_parser = subparsers.add_parser(
        "post",
        help="Compute activation times (and visualizations) from a previous run",
    )
    post_parser.add_argument(
        "config",
        type=Path,
        default="config.toml",
        help="Path to the configuration file",
    )

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

            if not args["config"].exists():
                raise ValueError(f"Configuration file {args['config']} does not exist.")
            run_file(**args, comm=comm)

        elif command == "validate-config":
            from .config import Config

            config_path = args.pop("config")
            if not config_path.exists():
                raise ValueError(f"Configuration file {config_path} does not exist.")
            Config.parse_toml(config_path)
            logger.info(f"Configuration file {config_path} is valid.")
        elif command == "init":
            from .config import Config

            config_path = args.pop("config")
            force = args.pop("force")
            if config_path.exists() and not force:
                raise ValueError(
                    f"Configuration file {config_path} already exists. Use --force to overwrite.",
                )
            Config().dump_toml(config_path)
        elif command == "ecg":
            from .postprocess import run_ecg_file

            if not args["config"].exists():
                raise ValueError(f"Configuration file {args['config']} does not exist.")
            run_ecg_file(**args, comm=comm)
        elif command == "post":
            from .postprocess import run_post_file

            if not args["config"].exists():
                raise ValueError(f"Configuration file {args['config']} does not exist.")
            run_post_file(**args, comm=comm)
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
