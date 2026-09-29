"""Command line interface for fenicsx-beat (``beat``).

Requires the ``cli`` extra: ``pip install "fenicsx-beat[cli]"``.
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional, Sequence

from mpi4py import MPI

from .log import setup_logging

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
EXIT_OK, EXIT_CONFIG, EXIT_RUNTIME = 0, 1, 2


def _available_templates() -> list[str]:
    if not TEMPLATES_DIR.is_dir():
        return []
    return sorted(p.name for p in TEMPLATES_DIR.iterdir() if (p / "config.toml").is_file())


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", type=Path, help="Path to the configuration file")
    p.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, VALUE parsed as TOML, e.g. --set 'solver.dt=\"0.02 ms\"'. "
        "Repeatable. Precedence: file < BEAT_* env vars < --set < flags",
    )


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command, do not run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print more information")
    parser.add_argument("--log-all-cpus", action="store_true", help="Log on all ranks")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Display version information")

    init = sub.add_parser("init", help="Write a starter config from a template")
    init.add_argument("config", type=Path, nargs="?", default=Path("config.toml"))
    init.add_argument(
        "--template",
        default="slab",
        help="Template name (src/beat/cli/templates/<name>)",
    )
    init.add_argument("--force", action="store_true", help="Overwrite existing files")

    validate = sub.add_parser("validate-config", help="Validate and print the resolved config")
    _add_config_args(validate)

    geometry = sub.add_parser("geometry", help="Only generate/load the geometry")
    _add_config_args(geometry)

    run = sub.add_parser("run", help="Run a simulation")
    _add_config_args(run)
    run.add_argument("--restart", action="store_true", help="Continue from the last checkpoint")
    run.add_argument("--overwrite", action="store_true", help="Replace existing results")
    run.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Override output.folder (relative to the current directory)",
    )
    run.add_argument(
        "--petsc-options",
        default=None,
        help='PETSc options for the PDE solve, e.g. "-ksp_type cg -pc_type hypre"',
    )

    for name, help_ in (
        ("ecg", "Recover the pseudo-ECG at [postprocess.points]"),
        ("post", "Activation times, VTX conversion and visualizations"),
    ):
        p = sub.add_parser(name, help=help_)
        _add_config_args(p)
        p.add_argument("--output-folder", type=Path, default=None)
    return parser


def display_version_info() -> None:
    from petsc4py import PETSc

    import dolfinx

    from .. import __version__

    logger.info(f"fenicsx-beat: {__version__}")
    logger.info(f"dolfinx: {dolfinx.__version__}")
    logger.info(f"mpi4py: {MPI.Get_version()}")
    logger.info(f"petsc4py: {PETSc.Sys.getVersion()}")


def _init(target: Path, template: str, force: bool) -> None:
    from .config import ConfigError

    available = _available_templates()
    if template not in available:
        raise ConfigError(f"Unknown template {template!r}; available: {', '.join(available)}")
    if target.exists() and not force:
        raise ConfigError(f"{target} already exists. Use --force to overwrite.")
    src = TEMPLATES_DIR / template
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "config.toml", target)
    for extra in src.iterdir():  # companion files, e.g. the .ode model
        if extra.name != "config.toml" and extra.is_file():
            dest = target.parent / extra.name
            if dest.exists() and not force:
                raise ConfigError(f"{dest} already exists. Use --force to overwrite.")
            shutil.copyfile(extra, dest)
    logger.info(f"Wrote {target} from template {template!r}")


def _dispatch(args: dict, comm) -> None:
    from .config import ConfigError
    from .overrides import load_config

    command = args["command"]
    if command == "version":
        display_version_info()
        return
    if command == "init":
        from .runner import _on_rank0

        # _on_rank0 runs _init on rank 0 only and re-raises *any* exception (not just
        # ConfigError -- an OSError/PermissionError from mkdir/copyfile must not skip the
        # broadcast either) as a ConfigError on every rank, so a rank-0-only failure here can
        # never leave the other ranks waiting forever on a barrier/bcast that rank 0 never
        # reaches.
        _on_rank0(comm, ConfigError, lambda: _init(args["config"], args["template"], args["force"]))
        return

    conf = load_config(
        args["config"],
        sets=args["sets"],
        output_folder=args.get("output_folder"),
        petsc_options=args.get("petsc_options"),
    )
    if command == "validate-config":
        # Rank-0-only, and needs no barrier: load_config above already ran (and would have
        # raised ConfigError) identically on every rank, so every rank reaches this point only
        # on success, printing is not collective, and nothing after this depends on it.
        if comm.rank == 0:
            import toml

            print(toml.dumps(conf.model_dump(mode="json", exclude_none=True)))
        logger.info(f"Configuration file {args['config']} is valid.")
    elif command == "geometry":
        from .geometry import build_geometry

        geo = build_geometry(conf.geometry, comm)
        n_cells = geo.mesh.topology.index_map(geo.mesh.topology.dim).size_global
        logger.info(f"Geometry ready: {n_cells} cells, markers {sorted(geo.markers)}")
    elif command == "run":
        from .runner import run

        run(conf, comm=comm, restart=args["restart"], overwrite=args["overwrite"])
    elif command == "ecg":
        from .postprocess import run_ecg

        run_ecg(conf, comm=comm)
    elif command == "post":
        from .postprocess import run_post

        run_post(conf, comm=comm)
    else:  # pragma: no cover - argparse restricts choices
        raise ConfigError(f"Unknown command {command}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = setup_parser()
    args = vars(parser.parse_args(argv))
    comm = MPI.COMM_WORLD
    setup_logging(
        level=logging.DEBUG if args.pop("verbose") else logging.INFO,
        log_all_cpus=args.pop("log_all_cpus"),
        comm=comm,
    )
    if args.pop("dry_run"):
        logger.info("Dry run: %s %s", args["command"], args)
        return EXIT_OK

    from .config import ConfigError
    from .runner import SolverFailure

    try:
        _dispatch(args, comm)
    except ConfigError as e:
        logger.error(str(e))
        return EXIT_CONFIG
    except SolverFailure as e:
        logger.error(f"Simulation failed: {e}")
        return EXIT_RUNTIME
    return EXIT_OK


__all__ = ["main"]
