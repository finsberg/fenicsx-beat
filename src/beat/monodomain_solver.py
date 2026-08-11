import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .monodomain_model import MonodomainModel
from .telemetry import BaseMonitor, NullMonitor

logger = logging.getLogger(__name__)
EPS = 1e-12


class ODESolver(Protocol):
    def to_dolfin(self) -> None: ...

    def from_dolfin(self) -> None: ...

    def ode_to_pde(self) -> None: ...

    def pde_to_ode(self) -> None: ...

    def step(self, t0: float, dt: float) -> None: ...


@dataclass
class MonodomainSplittingSolver:
    pde: MonodomainModel
    ode: ODESolver
    theta: float = 1.0
    monitor: BaseMonitor = field(default_factory=NullMonitor)

    def __post_init__(self) -> None:
        # assert np.isclose(self.theta, 1.0), "Only first order splitting is implemented"
        self.ode.to_dolfin()  # numpy array (ODE solver) -> dolfin function
        self.ode.ode_to_pde()  # dolfin function in ODE space (quad?) -> CG1 dolfin function
        self.pde.assign_previous()

    def solve(self, interval, dt):
        T0, T = interval
        if dt is None:
            dt = T - T0
        t0 = T0
        t1 = T0 + dt

        while t1 < T + EPS:
            logger.debug(f"Solving on t = ({t0:.2f}, {t0:.2f})")
            self.step((t0, t1))

            t0 = t1
            t1 = t0 + dt

    def step(self, interval):
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
