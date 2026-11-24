import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from mpi4py import MPI

import dolfinx
import numpy as np

from .telemetry import BaseMonitor, NullMonitor
from .utils import time_steps

logger = logging.getLogger(__name__)


class ODESolver(Protocol):
    """What the splitting driver asks of an ODE (cell model) solver.

    The solver keeps the cell states at its own points (the degrees of freedom of ``v_ode``'s
    space) and moves the potential between them and the PDE's space. :class:`SplittingSolver`
    never copies between ``v_pde`` and the PDE model's ``v``: they must be the same Function (pass
    ``v_pde=pde.state``). Without a PDE model (``pde=None``) ``v_pde`` is not used.

    Implemented by every :class:`beat.odesolver.BaseDolfinODESolver` subclass and by
    :class:`beat.reaction_eikonal.FootCurrentODESolver`.
    """

    @property
    def v_ode(self) -> dolfinx.fem.Function:
        """The transmembrane potential at the ODE points."""

    @property
    def v_pde(self) -> dolfinx.fem.Function:
        """The transmembrane potential in the PDE space: the PDE model's ``v``."""

    def to_dolfin(self) -> None:
        """Copy the potential from the cell states into ``v_ode``."""

    def from_dolfin(self) -> None:
        """Copy ``v_ode`` back into the cell states."""

    def ode_to_pde(self) -> None:
        """Project ``v_ode`` into ``v_pde``."""

    def pde_to_ode(self) -> None:
        """Project ``v_pde`` into ``v_ode``."""

    def step(self, t0: float, dt: float) -> None:
        """Advance the cell states from ``t0`` by ``dt``."""

    def states_to_dolfin(self, names: list[str] | None = None) -> list[dolfinx.fem.Function]:
        """New Functions on ``v_ode``'s space holding every state, named ``names``."""

    def load_all_states(self, functions: list[dolfinx.fem.Function]) -> None:
        """Set every state from ``functions`` (as returned by :meth:`states_to_dolfin`)."""


class PDEModel(Protocol):
    """What the splitting driver asks of a PDE model.

    It only ever advances the model and asks it to keep its history; the transmembrane
    potential is shared with the ODE solver directly. So any model offering these is
    drivable here, whatever else it solves for alongside.
    """

    @property
    def v(self) -> dolfinx.fem.Function:
        """The transmembrane potential."""

    def assign_previous(self) -> None: ...

    def step(self, interval: tuple[float, float]) -> None: ...


# Fraction of the time step the first ODE sub-step covers, per splitting scheme.
_SPLITTING_FRACTION: dict[str, float] = {"godunov": 1.0, "strang": 0.5}


@dataclass
class SplittingSolver:
    """Advance a PDE model and an ODE solver by operator splitting.

    ``splitting="godunov"`` (the default) takes a full ODE step and then a full PDE step: first
    order in time. ``splitting="strang"`` takes half an ODE step, a full PDE step and another half
    ODE step: second order in time, provided both sub-steps are second order themselves (PDE
    ``params["theta"] = 0.5`` and a second-order cell scheme such as gotranx's
    ``generalized_rush_larsen_2``). ``theta`` is the deprecated way of choosing the splitting (the
    fraction of the step the first ODE sub-step covers: 1.0 Godunov, 0.5 Strang); after
    construction it always holds that fraction.

    With ``pde=None`` only the cell model is advanced and the splitting is ignored. Wrapping the
    ODE solver in :class:`beat.reaction_eikonal.FootCurrentODESolver` gives the R-E+ model (with a
    PDE) or the R-E- model (``pde=None``). ``monitor=None`` (or leaving it out) records nothing
    (a :class:`beat.telemetry.NullMonitor`).
    """

    pde: PDEModel | None
    ode: ODESolver
    theta: float | None = None  # deprecated; kept third for positional callers
    monitor: BaseMonitor = field(default_factory=NullMonitor)
    splitting: Literal["godunov", "strang"] | None = None

    def __post_init__(self) -> None:
        self._resolve_splitting()
        if self.monitor is None:
            self.monitor = NullMonitor()
        self.ode.to_dolfin()  # numpy array (ODE solver) -> dolfin function
        if self.pde is not None:
            self.ode.ode_to_pde()  # dolfin function in ODE space (quad?) -> CG1 dolfin function
            self.pde.assign_previous()

    def _resolve_splitting(self) -> None:
        if self.theta is None:
            splitting = "godunov" if self.splitting is None else self.splitting
            if splitting not in _SPLITTING_FRACTION:
                raise ValueError(f"splitting must be 'godunov' or 'strang', got {splitting!r}")
            self.splitting = splitting  # type: ignore[assignment]
            self.theta = _SPLITTING_FRACTION[splitting]
            return
        if self.splitting is not None:
            raise TypeError("Give splitting=, not the deprecated theta= as well")
        theta = float(self.theta)
        if not 0.0 <= theta <= 1.0:
            raise ValueError(f"theta must be in [0, 1], got {theta}")
        name = next((n for n, f in _SPLITTING_FRACTION.items() if np.isclose(theta, f)), None)
        if name is None:
            hint = (
                "only splitting='godunov' (theta=1) and splitting='strang' (theta=0.5) are "
                f"supported; theta={theta} still runs but is first order"
            )
        else:
            hint = f"use splitting={name!r}"
        warnings.warn(
            f"SplittingSolver(theta=...) is deprecated; {hint}",
            DeprecationWarning,
            stacklevel=4,  # _resolve_splitting -> __post_init__ -> dataclass __init__ -> caller
        )
        self.splitting = name  # type: ignore[assignment]
        self.theta = theta

    @property
    def v(self) -> dolfinx.fem.Function:
        """The transmembrane potential: ``pde.v``, or ``ode.v_ode`` without a PDE."""
        return self.ode.v_ode if self.pde is None else self.pde.v

    def restart_functions(
        self,
        state_names: Sequence[str],
    ) -> list[tuple[str, dolfinx.fem.Function]]:
        """The Functions that hold everything the splitting scheme carries between steps.

        These are ``("v", pde.v)`` followed by ``("state_<name>", f)`` for each ODE state, named
        by ``state_names`` (one per state). Write them to save a checkpoint. On a fresh solver,
        read into them and pass them to :meth:`load_restart`. Without a PDE, ``"v"`` is
        ``ode.v_ode``.
        """
        states = self.ode.states_to_dolfin(list(state_names))
        return [("v", self.v)] + [(f"state_{name}", f) for name, f in zip(state_names, states)]

    def restart_metadata(self) -> dict[str, Any]:
        """State beyond the Functions, to be stored with a checkpoint (JSON-serializable).

        Pass it back to :meth:`load_restart`. ``"has_pde"`` records whether this solver has a PDE
        model, so a checkpoint is not loaded into the other kind; ``"pde"`` holds the PDE model's
        own metadata, if it has any.
        """
        metadata: dict[str, Any] = {"has_pde": self.pde is not None}
        if self.pde is not None and hasattr(self.pde, "restart_metadata"):
            metadata["pde"] = self.pde.restart_metadata()
        return metadata

    def load_restart(
        self,
        functions: Sequence[tuple[str, dolfinx.fem.Function]],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Restore the solver from ``functions`` as returned by :meth:`restart_functions`.

        Overwrites ``pde.v`` and every ODE state, restores the PDE's metadata if given (as
        returned by :meth:`restart_metadata`), then sets the PDE's previous value, exactly as
        construction does.

        Without a PDE, ``v`` is one of the ODE states, so the states are authoritative: the
        checkpoint's ``"v"`` is overwritten by the loaded states (``ode.to_dolfin()``).

        Raises ``ValueError``, before anything is modified and on every rank, if the checkpoint
        was saved by a solver with a PDE and this one has none or the reverse (``metadata``'s
        ``"has_pde"``; checkpoints without it are not checked), or if its ``"v"`` does not fit
        this solver's ``v``.
        """
        names = [name for name, _ in functions]
        if not names or names[0] != "v" or not all(n.startswith("state_") for n in names[1:]):
            raise ValueError(
                f"Restart functions must be 'v' followed by 'state_<name>' ones, got {names}",
            )
        has_pde = self.pde is not None
        if metadata is not None:
            saved_with_pde = metadata.get("has_pde", "pde" in metadata or has_pde)
            if bool(saved_with_pde) != has_pde:
                saved = "with a PDE" if saved_with_pde else "without a PDE"
                this = "has one" if has_pde else "has no PDE"
                raise ValueError(
                    f"The checkpoint was saved by a solver {saved}, but this solver {this}",
                )
        v = functions[0][1]
        target = self.v
        comm = target.function_space.mesh.comm
        if comm.allreduce(int(v.x.array.size != target.x.array.size), op=MPI.MAX):
            raise ValueError(
                "The checkpoint's 'v' does not fit this solver's v (different function space "
                "or mesh)",
            )
        if v is not target:
            target.x.array[:] = v.x.array
        target.x.scatter_forward()
        self.ode.load_all_states([f for _, f in functions[1:]])
        if self.pde is None:
            self.ode.to_dolfin()
            return
        if metadata is not None and "pde" in metadata:
            self.pde.load_restart_metadata(metadata["pde"])  # type: ignore[attr-defined]
        self.pde.assign_previous()

    def solve(self, interval: tuple[float, float], dt: float | None) -> None:
        """Step over ``interval`` with time step ``dt`` (see :func:`beat.utils.time_steps`)."""
        for t0, t1 in time_steps(interval, dt):
            logger.debug(f"Solving on t = ({t0:.2f}, {t1:.2f})")
            self.step((t0, t1))

    def step(self, interval):
        if self.pde is None:
            t0, t1 = interval
            with self.monitor.track_time("total_step"):
                with self.monitor.track_time("ode_step"):
                    self.ode.step(t0, t1 - t0)
                with self.monitor.track_time("ode_to_dolfin"):
                    self.ode.to_dolfin()
            self.monitor.advance_step(t0, t1)
            return

        # Extract some parameters for readability
        theta = self.theta
        assert theta is not None

        # Extract time domain
        t0, t1 = interval
        logger.debug(
            f"Stepping from {t0} to {t1} with {self.splitting or f'theta={theta}'} splitting",
        )

        dt = t1 - t0
        t = t0 + theta * dt

        logger.debug(f"Tentative ODE step with t0={t0:.5f} dt={theta * dt:.5f}")

        with self.monitor.track_time("total_step"):
            with self.monitor.track_time("ode_step"):
                self.ode.step(t0=t0, dt=theta * dt)

            with self.monitor.track_time("ode_to_dolfin"):
                # numpy array (ODE solver) -> dolfin function
                self.ode.to_dolfin()

            with self.monitor.track_time("ode_to_pde"):
                # dolfin function in ODE space (quad?) -> CG1 dolfin function
                self.ode.ode_to_pde()

            with self.monitor.track_time("pde_assign_previous_before"):
                self.pde.assign_previous()

            logger.debug("PDE step")

            with self.monitor.track_time("pde_step"):
                self.pde.step((t0, t1))

            with self.monitor.track_time("pde_to_ode"):
                # CG1 dolfin function -> dolfin function in ODE space (quad?)
                self.ode.pde_to_ode()

            with self.monitor.track_time("ode_from_dolfin"):
                self.ode.from_dolfin()

            # If first order splitting, we are done. Otherwise, we need to do a corrective ODE step.
            if np.isclose(theta, 1.0):
                with self.monitor.track_time("pde_assign_previous_after"):
                    # But first update previous value in PDE
                    self.pde.assign_previous()
            else:
                logger.debug(f"Corrective ODE step with t0={t:5f} and dt={(1.0 - theta) * dt:.5f}")

                with self.monitor.track_time("corrective_ode_step"):
                    self.ode.step(t, (1.0 - theta) * dt)

                with self.monitor.track_time("corrective_ode_to_dolfin"):
                    # numpy array (ODE solver) -> dolfin function
                    self.ode.to_dolfin()

                with self.monitor.track_time("corrective_ode_to_pde"):
                    # dolfin function in ODE space (quad?) -> CG1 dolfin function
                    self.ode.ode_to_pde()

                with self.monitor.track_time("corrective_pde_assign_previous"):
                    self.pde.assign_previous()

        # Alert the monitor that the step is finished so it can handle logging/aggregation
        self.monitor.advance_step(t0, t1)


class MonodomainSplittingSolver(SplittingSolver):
    """Deprecated name of :class:`SplittingSolver` (same arguments and behaviour)."""

    def __post_init__(self) -> None:
        warnings.warn(
            "MonodomainSplittingSolver is deprecated; use SplittingSolver",
            DeprecationWarning,
            stacklevel=3,  # past __post_init__ and the dataclass __init__, to the caller
        )
        super().__post_init__()
