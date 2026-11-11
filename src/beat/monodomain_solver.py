import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import dolfinx
import numpy as np

from .telemetry import BaseMonitor, NullMonitor

logger = logging.getLogger(__name__)
EPS = 1e-12


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


@dataclass
class SplittingSolver:
    """Advance a PDE model and an ODE solver by operator splitting (Godunov or Strang, ``theta``)

    With ``pde=None`` only the cell model is advanced and ``theta`` is ignored. Wrapping the ODE
    solver in :class:`beat.reaction_eikonal.FootCurrentODESolver` gives the R-E+ model (with a PDE)
    or the R-E- model (``pde=None``).
    """

    pde: PDEModel | None
    ode: ODESolver
    theta: float = 1.0
    monitor: BaseMonitor = field(default_factory=NullMonitor)

    def __post_init__(self) -> None:
        self.ode.to_dolfin()  # numpy array (ODE solver) -> dolfin function
        if self.pde is not None:
            self.ode.ode_to_pde()  # dolfin function in ODE space (quad?) -> CG1 dolfin function
            self.pde.assign_previous()

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

        Pass it back to :meth:`load_restart`. It is empty if the PDE model has none.
        """
        if self.pde is not None and hasattr(self.pde, "restart_metadata"):
            return {"pde": self.pde.restart_metadata()}
        return {}

    def load_restart(
        self,
        functions: Sequence[tuple[str, dolfinx.fem.Function]],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Restore the solver from ``functions`` as returned by :meth:`restart_functions`.

        Overwrites ``pde.v`` and every ODE state, restores the PDE's metadata if given (as
        returned by :meth:`restart_metadata`), then sets the PDE's previous value, exactly as
        construction does. Without a PDE, the states are loaded and copied into ``v``; metadata with
        PDE state is refused.
        """
        names = [name for name, _ in functions]
        if not names or names[0] != "v" or not all(n.startswith("state_") for n in names[1:]):
            raise ValueError(
                f"Restart functions must be 'v' followed by 'state_<name>' ones, got {names}",
            )
        if self.pde is None and metadata is not None and "pde" in metadata:
            raise ValueError("The restart metadata has PDE state, but this solver has no PDE")
        v = functions[0][1]
        target = self.v
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

    def solve(self, interval, dt):
        T0, T = interval
        if dt is None:
            dt = T - T0
        t0 = T0
        t1 = T0 + dt

        while t1 < T + EPS:
            logger.debug(f"Solving on t = ({t0:.2f}, {t1:.2f})")
            self.step((t0, t1))

            t0 = t1
            t1 = t0 + dt

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

        # Extract time domain
        t0, t1 = interval
        logger.debug(f"Stepping from {t0} to {t1} using theta = {theta}")

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
