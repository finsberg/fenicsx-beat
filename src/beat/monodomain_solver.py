import logging
import numpy as np
from typing import Protocol
from dataclasses import dataclass
from .monodomain_model import MonodomainModel

logger = logging.getLogger(__name__)
EPS = 1e-12


class ODESolver(Protocol):
    def to_dolfin(self) -> None:
        ...

    def from_dolfin(self) -> None:
        ...

    def ode_to_pde(self) -> None:
        ...

    def pde_to_ode(self) -> None:
        ...

    def step(self, t0: float, t1: float) -> None:
        ...


@dataclass
class MonodomainSplittingSolver:
    pde: MonodomainModel
    ode: ODESolver
    theta: float = 0.5

    def __post_init__(self) -> None:
        self.ode.to_dolfin()  # numpy array (ODE solver) -> dolfin function
        self.ode.ode_to_pde()  # dolfin function in ODE space (quad?) -> CG1 dolfin function
        self.pde.assign_previous()

    def solve(self, interval, dt):
        (T0, T) = interval
        if dt is None:
            dt = T - T0
        t0 = T0
        t1 = T0 + dt

        while t1 < T + EPS:
            logger.info(f"Solving on t = ({t0:.2f}, {t0:.2f})")
            self.step((t0, t1))

            t0 = t1
            t1 = t0 + dt

    def step(self, interval):
        # Extract some parameters for readability
        theta = self.theta

        # Extract time domain
        (t0, t1) = interval
        logger.info(f"Stepping from {t0} to {t1} using theta = {theta}")
        dt = t1 - t0
        t = t0 + theta * dt

        logger.info(f"Tentative ODE step with t0={t0:.5f} dt={theta * dt:.5f}")

        # Solve ODE
        self.ode.step(t0=t0, dt=theta * dt)
        # Move voltage to FEniCS
        self.ode.to_dolfin()  # numpy array (ODE solver) -> dolfin function
        self.ode.ode_to_pde()  # dolfin function in ODE space (quad?) -> CG1 dolfin function
        self.pde.assign_previous()

        logger.info("PDE step")
        # Solve PDE
        self.pde.step((t0, t1))

        self.ode.pde_to_ode()  # CG1 dolfin function -> dolfin function in ODE space (quad?)
        # Copy voltage from PDE to ODE
        self.ode.from_dolfin()

        # If first order splitting, we are done.
        if np.isclose(theta, 1.0):
            # But first update previous value in PDE
            self.pde.assign_previous()
            return

        # Otherwise, we do another ode_step:
        logger.info(
            f"Corrective ODE step with t0={t:5f} and dt={(1.0 - theta) * dt:.5f}"
        )

        # To the correction step
        self.ode.step(t, (1.0 - theta) * dt)
        # And copy the solution back to FEniCS
        self.ode.to_dolfin()  # numpy array (ODE solver) -> dolfin function
        self.ode.ode_to_pde()  # dolfin function in ODE space (quad?) -> CG1 dolfin function
        self.pde.assign_previous()
