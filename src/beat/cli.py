import argparse
import logging
from typing import Optional, Sequence

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
    subparsers = parser.add_subparsers(dest="command")

    # Version parser
    subparsers.add_parser("version", help="Display version information")

    # Run simulation parser
    subparsers.add_parser("run", help="Run simulations")

    # ECG parser
    subparsers.add_parser("ecg", help="Compute ECG signals")

    # Postprocessing parser
    subparsers.add_parser("post", help="Postprocessing")

    return parser


def _disable_loggers():
    for name in ["matplotlib"]:
        logging.getLogger(name).setLevel(logging.WARNING)


def display_version_info():
    from mpi4py import MPI
    from petsc4py import PETSc

    import dolfinx

    from . import __version__

    logger.info(f"fenicsx-beat: {__version__}")
    logger.info(f"dolfinx: {dolfinx.__version__}")
    logger.info(f"mpi4py: {MPI.Get_version()}")
    logger.info(f"petsc4py: {PETSc.Sys.getVersion()}")


def dispatch(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None) -> int:
    args = vars(parser.parse_args(argv))
    logging.basicConfig(level=logging.DEBUG if args.pop("verbose") else logging.INFO)
    _disable_loggers()

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
            return NotImplemented
        elif command == "ecg":
            return NotImplemented
        elif command == "post":
            return NotImplemented
        else:
            logger.error(f"Unknown command {command}")
            parser.print_help()
    except ValueError as e:
        logger.error(e)
        parser.print_help()

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = setup_parser()
    return dispatch(parser, argv)
