"""Wire the builders together, run the splitting time loop, write results and restarts.

Output folder layout (``conf.output.folder``)::

    config.resolved.toml   the fully resolved configuration of the (latest) run
    run.json               run metadata (versions, ranks, status: running/finished/failed)
    results.bp             io4dolfinx: ``v`` (+ ``output.fields``) every ``output.save_every``
    restart.bp             io4dolfinx: ``v`` and every ODE state (``state_<name>``)
    restart.json           the latest checkpoint's time/step and the run's physics hash
    output.log             log file (``output_all_cpus.log`` too when running on >1 rank)
    performance.json       timing summary (only with ``output.performance = true``)

io4dolfinx *appends* a function written at an already existing timestamp, and ``read_function``
returns the *first* match. So a restarted run never writes a ``results.bp`` timestamp ``<=`` the
last one already there, and readers deduplicate with ``np.unique`` (:func:`read_result_times`).
"""

import datetime
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from mpi4py import MPI

import dolfinx
import io4dolfinx
import numpy as np

from ..monodomain_solver import MonodomainSplittingSolver
from ..telemetry import NullMonitor, PerformanceMonitor
from .cellmodel import CellModel, build_cell_model, build_region_markers, state_index
from .config import Config, ConfigError, ms
from .geometry import CLIGeometry, build_conductivity, build_geometry
from .log import add_logfile_handler, remove_logfile_handlers
from .overrides import dump_config, physics_hash
from .solvers import build_ode, build_pde, check_backends_available, ode_space
from .stimulus import StimulusSet, build_stimuli

logger = logging.getLogger(__name__)

RESULTS = "results.bp"
RESTART = "restart.bp"
RESTART_META = "restart.json"
RUN_META = "run.json"
PERFORMANCE = "performance.json"


class SolverFailure(RuntimeError):
    """The simulation failed at runtime (exit code 2)."""


@dataclass
class Simulation:
    conf: Config
    geo: CLIGeometry
    time: dolfinx.fem.Constant
    pde: Any
    ode: Any
    solver: MonodomainSplittingSolver
    cell: CellModel
    stimuli: StimulusSet
    V_ode: Any


# Everything beat itself writes into an output folder (``beat run``: results, restarts,
# metadata, logs; ``beat post``: ``post/``). --overwrite deletes exactly these and nothing else,
# since the output folder may well be the config's own directory, holding the user's
# config.toml/.ode files.
_ARTIFACT_NAMES = (
    RESULTS,
    RESTART,
    RESTART_META,
    RUN_META,
    "config.resolved.toml",
    PERFORMANCE,
    "output.log",
    "output_all_cpus.log",
    "post",
)
# The content-hashed cell-model caches (``cell_model_<hash>.py``, ``init_states/``) are beat's
# too but are *kept* by --overwrite: they are always valid for their hash, and they are built
# (as part of validating the config) before the old results are wiped.
_CACHE_NAMES = ("init_states",)
_CACHE_GLOBS = ("cell_model_*.py",)


def _on_rank0(comm, error_type: type[Exception], fn: Callable[[], Any]) -> None:
    """Run ``fn`` on rank 0 only, and make any failure raise on *every* rank.

    Rank-0-only filesystem work followed by a barrier/collective otherwise deadlocks when rank 0
    raises. A :class:`ConfigError` stays a ConfigError; anything else is raised as
    ``error_type`` with the same message (chained to the original on rank 0).
    """
    original: Exception | None = None
    error: tuple[bool, str] | None = None
    if comm.rank == 0:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - re-raised on every rank below
            original = e
            error = (isinstance(e, ConfigError), str(e) or repr(e))
    error = comm.bcast(error, root=0)
    if error is None:
        return
    is_config, message = error
    exc: Exception = ConfigError(message) if is_config else error_type(message)
    raise exc from original


def _write_json(path: Path, data: dict, comm) -> None:
    def write() -> None:
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)

    _on_rank0(comm, OSError, write)


def _remove_artifacts(folder: Path) -> None:
    paths = [folder / name for name in _ARTIFACT_NAMES]
    for p in paths:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        elif p.exists() or p.is_symlink():
            p.unlink()


def read_result_times(path: Path, comm, name: str = "v") -> np.ndarray:
    """Sorted, unique timestamps of ``name`` in the io4dolfinx file ``path``."""
    return np.unique(io4dolfinx.read_timestamps(filename=path, comm=comm, function_name=name))


def _output_decision(conf: Config, restart: bool, overwrite: bool) -> tuple[str, str]:
    """Decide (on one rank, from the filesystem) what :func:`prepare_output` must do.

    Returns ``("error", message)``, ``("restart", "")``, ``("wipe", "")`` or ``("create", "")``.
    """
    folder = conf.output.folder
    no_checkpoint = (
        f"no restart checkpoint exists yet in {folder} (no {RESTART_META}; the run stopped "
        "before its first checkpoint); use --overwrite to start over"
    )
    if restart:
        if not (folder / RESTART_META).is_file():
            return "error", f"Cannot restart: {no_checkpoint}"
        meta = json.loads((folder / RESTART_META).read_text())
        try:
            current = physics_hash(conf)
        except ConfigError as e:
            return "error", str(e)
        if meta.get("physics_hash") != current:
            return "error", (
                "Cannot restart: the physics settings differ from the original run "
                "(only solver.end_time/num_beats/BCL, [output] and [postprocess] may change). "
                f"Compare with {folder / 'config.resolved.toml'}"
            )
        return "restart", ""
    if (folder / RESULTS).exists() or (folder / RESTART_META).exists():
        if not overwrite:
            if not (folder / RESTART_META).exists():
                return (
                    "error",
                    f"Output folder {folder} already contains results, but {no_checkpoint}",
                )
            return "error", (
                f"Output folder {folder} already contains results. Use --overwrite to replace "
                "them or --restart to continue the run."
            )
        return "wipe", ""
    return "create", ""


def decide_output(conf: Config, restart: bool, overwrite: bool, comm) -> str:
    """Decide what to do with ``conf.output.folder`` without touching it.

    The decision is made on rank 0 only and broadcast, so that every rank raises the same
    :class:`ConfigError` together (never one rank raising while the others wait in a barrier).
    Returns ``"restart"``, ``"wipe"`` or ``"create"`` for :func:`apply_output`.
    """
    decision = None
    if comm.rank == 0:
        try:
            decision = _output_decision(conf, restart, overwrite)
        except Exception as e:  # e.g. a corrupt restart.json: must not raise on rank 0 alone
            decision = ("error", f"Cannot prepare output folder {conf.output.folder}: {e!r}")
    action, message = comm.bcast(decision, root=0)
    if action == "error":
        raise ConfigError(message)
    return action


def apply_output(conf: Config, action: str, comm) -> None:
    """Carry out :func:`decide_output`'s ``action``: create the folder, and for ``"wipe"``
    delete only beat's own artifacts (see ``_ARTIFACT_NAMES``), never other files."""
    if action == "restart":
        return
    folder = conf.output.folder

    def prepare() -> None:
        if action == "wipe":
            _remove_artifacts(folder)
        folder.mkdir(parents=True, exist_ok=True)

    try:
        _on_rank0(comm, ConfigError, prepare)
    except ConfigError as e:
        raise ConfigError(f"Cannot prepare output folder {folder}: {e}") from e


def prepare_output(conf: Config, restart: bool, overwrite: bool, comm) -> None:
    """Validate/prepare ``conf.output.folder`` for a fresh run, an overwrite or a restart."""
    apply_output(conf, decide_output(conf, restart, overwrite, comm), comm)


def build_simulation(conf: Config, comm=MPI.COMM_WORLD, monitor=None) -> Simulation:
    """Build every model/solver the run needs.

    ``check_backends_available`` must run before *any* UFL form is processed: Irksome has to be
    imported before UFL forms are built (import-order requirement), and a missing optional
    backend should surface as a ConfigError before any expensive set-up.
    """
    check_backends_available(conf.solver)
    cell = build_cell_model(conf.cell, conf.output.folder, comm)
    for name in conf.output.fields:
        state_index(cell, name)
    geo = build_geometry(conf.geometry, comm)
    unit = conf.geometry.unit
    time = dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0))
    stimuli = build_stimuli(conf.stimulus, geo, conf.ep, time, unit, conf.solver.t_end_ms())
    M = build_conductivity(conf.ep, geo, conf.geometry.fibers)
    C_m = conf.ep.C_m.to(f"uF/{unit}**2").magnitude
    pde: Any = build_pde(conf.solver, geo.mesh, time, M, stimuli, C_m=C_m, monitor=monitor)
    V_ode = ode_space(geo.mesh)
    markers = build_region_markers(conf.cell, geo, V_ode)
    ode = build_ode(conf.solver, cell, markers, pde, time, monitor=monitor)
    solver = MonodomainSplittingSolver(
        pde=pde,
        ode=ode,
        theta=conf.solver.theta,
        monitor=monitor or NullMonitor(),
    )
    return Simulation(conf, geo, time, pde, ode, solver, cell, stimuli, V_ode)


def _save_results(sim: Simulation, t: float) -> None:
    path = sim.conf.output.folder / RESULTS
    io4dolfinx.write_function_on_input_mesh(path, sim.pde.state, time=t, name="v")
    if sim.conf.output.fields:
        funcs = sim.ode.states_to_dolfin(sim.cell.state_names)
        for name in sim.conf.output.fields:
            f = funcs[sim.cell.state_names.index(name)]
            io4dolfinx.write_function_on_input_mesh(path, f, time=t, name=name)


def _write_restart(sim: Simulation, t: float, step: int, write_data: bool = True) -> None:
    """Write a checkpoint at ``t`` and point restart.json at it.

    ``write_data=False`` only updates restart.json: used when restart.bp already holds a
    checkpoint at ``t`` (a run killed between writing restart.bp and restart.json, then
    restarted from the earlier checkpoint). io4dolfinx would append a duplicate timestamp and
    ``read_function`` returns the *first* one, i.e. the existing data -- which is the same state,
    recomputed from the same checkpoint with the same physics.
    """
    folder = sim.conf.output.folder
    if write_data:
        path = folder / RESTART
        io4dolfinx.write_function_on_input_mesh(path, sim.pde.state, time=t, name="v")
        states = sim.ode.states_to_dolfin(sim.cell.state_names)
        for name, f in zip(sim.cell.state_names, states):
            io4dolfinx.write_function_on_input_mesh(path, f, time=t, name=f"state_{name}")
    # restart.json is written last (atomically): it only ever points at a complete checkpoint.
    _write_json(
        folder / RESTART_META,
        {
            "t": t,
            "step": step,
            "physics_hash": physics_hash(sim.conf),
            "state_names": sim.cell.state_names,
        },
        sim.geo.mesh.comm,
    )


def _load_restart(sim: Simulation) -> tuple[int, float, np.ndarray]:
    """Restore the PDE and ODE state from the latest checkpoint.

    This is all the state the splitting scheme carries between steps: every step starts from
    the ODE states (``ode.step``), then overwrites the PDE state with the ODE's ``v``
    (``ode_to_pde``) and the PDE's previous value with that (``pde.assign_previous``). The PDE
    state is still read (and ``assign_previous`` called, as ``MonodomainSplittingSolver`` does
    on construction) so the restored model is consistent before the first step too.
    """
    folder = sim.conf.output.folder
    comm = sim.geo.mesh.comm
    meta = json.loads((folder / RESTART_META).read_text())
    if meta["state_names"] != sim.cell.state_names:
        raise ConfigError(
            f"Cannot restart: checkpoint states {meta['state_names']} differ from the cell "
            f"model's {sim.cell.state_names}",
        )
    t = float(meta["t"])
    # restart.bp may hold several checkpoints (checkpoint_every); pick the one restart.json
    # names by its exact stored timestamp (avoids float-equality surprises), but refuse to
    # silently fall back to a *different* checkpoint. Every rank reads the same file, so every
    # rank raises together.
    stored = np.asarray(
        io4dolfinx.read_timestamps(filename=folder / RESTART, comm=comm, function_name="v"),
        dtype=float,
    )
    tol = 1e-9 * ms(sim.conf.solver.dt)
    if stored.size == 0 or np.min(np.abs(stored - t)) >= tol:
        raise ConfigError(
            f"Cannot restart: {folder / RESTART} has no checkpoint at t={t} ms named by "
            f"{folder / RESTART_META} (stored: {sorted(set(stored.tolist()))})",
        )
    t_file = float(stored[np.argmin(np.abs(stored - t))])
    io4dolfinx.read_function(folder / RESTART, sim.pde.state, time=t_file, name="v")
    sim.pde.state.x.scatter_forward()
    funcs = sim.ode.states_to_dolfin(sim.cell.state_names)
    for name, f in zip(sim.cell.state_names, funcs):
        io4dolfinx.read_function(folder / RESTART, f, time=t_file, name=f"state_{name}")
        f.x.scatter_forward()
    sim.ode.load_all_states(funcs)
    sim.pde.assign_previous()
    logger.info(f"Restarting from t={t} ms (step {meta['step']})")
    # Checkpoint times whose data is *complete* in restart.bp (a kill mid-checkpoint can leave
    # ``v`` written but not every state): only these may be reused instead of rewritten.
    complete = stored
    for name in sim.cell.state_names:
        times = io4dolfinx.read_timestamps(
            filename=folder / RESTART,
            comm=comm,
            function_name=f"state_{name}",
        )
        times = np.asarray(times, dtype=float)
        complete = complete[[bool(np.any(np.abs(times - c) < tol)) for c in complete]]
    return int(meta["step"]), t, complete


def run(
    conf: Config,
    comm=MPI.COMM_WORLD,
    restart: bool = False,
    overwrite: bool = False,
) -> Path:
    """Run the simulation described by ``conf``; return the output folder.

    Raises :class:`ConfigError` (exit code 1) for configuration problems and
    :class:`SolverFailure` (exit code 2) for anything failing at runtime.

    The whole simulation (cell model, geometry, markers, stimuli, solvers) is built -- i.e. the
    config is fully validated -- *before* ``--overwrite`` deletes anything, so an invalid config
    never costs the previous run's results. Setup errors leave ``run.json`` untouched.
    """
    action = decide_output(conf, restart=restart, overwrite=overwrite, comm=comm)
    folder = conf.output.folder
    monitor = PerformanceMonitor(comm=comm) if conf.output.performance else None
    try:
        sim = build_simulation(conf, comm, monitor=monitor)
    except (ConfigError, SolverFailure):
        raise
    except Exception as e:
        raise SolverFailure(f"Setting up the simulation failed: {e!r}") from e
    apply_output(conf, action, comm)
    add_logfile_handler(folder, comm=comm)
    try:
        return _run(sim, comm, restart, monitor)
    finally:
        remove_logfile_handlers()


def _run(sim: Simulation, comm, restart: bool, monitor) -> Path:
    conf = sim.conf
    folder = conf.output.folder
    _on_rank0(comm, OSError, lambda: dump_config(conf, folder / "config.resolved.toml"))
    from .. import __version__

    record: dict[str, Any] = {
        "beat": __version__,
        "dolfinx": dolfinx.__version__,
        "n_ranks": comm.size,
        "start": datetime.datetime.now().isoformat(),
        "restart": restart,
        "status": "running",
    }
    _write_json(folder / RUN_META, record, comm)

    try:
        _time_loop(sim, restart)
    except Exception as e:
        record["status"] = "failed"
        record["error"] = str(e) if isinstance(e, ConfigError) else repr(e)
        _write_json(folder / RUN_META, record, comm)
        if isinstance(e, (ConfigError, SolverFailure)):
            raise
        raise SolverFailure(str(e)) from e

    record["status"] = "finished"
    record["end"] = datetime.datetime.now().isoformat()
    _write_json(folder / RUN_META, record, comm)
    if monitor is not None:
        monitor.display_summary()
        monitor.save_summary(folder / PERFORMANCE)
    logger.info(f"Simulation finished. Results in {folder / RESULTS}")
    return folder


def _time_loop(sim: Simulation, restart: bool) -> None:
    conf = sim.conf
    dt = ms(conf.solver.dt)
    n_steps = round(conf.solver.t_end_ms() / dt)
    save_stride = max(1, round(ms(conf.output.save_every) / dt))
    ckpt_every = ms(conf.output.checkpoint_every)
    ckpt_stride = round(ckpt_every / dt) if ckpt_every > 0 else 0
    comm = sim.geo.mesh.comm

    step0, last_saved, checkpoints = 0, -np.inf, np.zeros(0)
    if restart:
        step0, _, checkpoints = _load_restart(sim)
        results = conf.output.folder / RESULTS
        if results.exists():
            last_saved = float(read_result_times(results, comm).max())
        if step0 >= n_steps:
            logger.info("Nothing to do: the restart point is at or beyond the end time")
            return

    def maybe_save(step: int) -> None:
        nonlocal last_saved
        t = step * dt
        # Never write a timestamp already in results.bp: io4dolfinx would append a duplicate
        # and read_function returns the *first* (stale) one.
        if t > last_saved + 1e-9 * dt:
            _save_results(sim, t)
            last_saved = t

    def checkpoint(step: int) -> None:
        nonlocal checkpoints
        t = step * dt
        # Never append a timestamp restart.bp already holds (see _write_restart). Membership,
        # not "<= last": checkpoint_every may differ from the original run's, and restart.json
        # must only ever name a timestamp that is actually in restart.bp.
        exists = bool(np.any(np.abs(checkpoints - t) < 1e-9 * dt))
        _write_restart(sim, t, step, write_data=not exists)
        checkpoints = np.append(checkpoints, t)

    tic = perf_counter()
    for step in range(step0, n_steps):
        if step % save_stride == 0:
            maybe_save(step)
        t = step * dt
        sim.time.value = t  # type: ignore[assignment]
        for update in sim.stimuli.updates:
            update()
        sim.solver.step((t, t + dt))
        # Collective check: every rank must agree, so no rank raises alone.
        finite = bool(np.all(np.isfinite(sim.pde.state.x.array)))
        if not comm.allreduce(finite, op=MPI.LAND):
            raise SolverFailure(f"Non-finite transmembrane potential at t={t + dt} ms")
        if ckpt_stride and (step + 1) % ckpt_stride == 0 and step + 1 < n_steps:
            checkpoint(step + 1)
        if (step + 1) % conf.output.log_every == 0:
            elapsed = perf_counter() - tic
            rate = (step + 1 - step0) / elapsed
            logger.info(
                f"t={(step + 1) * dt:.3f} ms  step {step + 1}/{n_steps}  "
                f"{rate:.1f} steps/s  ETA {(n_steps - step - 1) / rate:.0f} s",
            )
    maybe_save(n_steps)
    checkpoint(n_steps)
