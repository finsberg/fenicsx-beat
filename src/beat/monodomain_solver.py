import logging
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .monodomain_model import MonodomainModel

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
    log_timings: bool = False
    timing_log_frequency: int = 1

    def __post_init__(self) -> None:
        # assert np.isclose(self.theta, 1.0), "Only first order splitting is implemented"
        self._step_counter = 0
        self._timing_totals: dict[str, float] = {}  # timing summary.

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
            logger.debug(f"Solving on t = ({t0:.2f}, {t0:.2f})")
            self.step((t0, t1))

            t0 = t1
            t1 = t0 + dt

    def step(self, interval):
        step_start = time.perf_counter()
        timings: dict[str, float] = {}

        # Extract some parameters for readability
        theta = self.theta

        # Extract time domain
        (t0, t1) = interval
        logger.debug(f"Stepping from {t0} to {t1} using theta = {theta}")

        dt = t1 - t0
        t = t0 + theta * dt

        logger.debug(f"Tentative ODE step with t0={t0:.5f} dt={theta * dt:.5f}")

        # Solve ODE
        tic = time.perf_counter()
        self.ode.step(t0=t0, dt=theta * dt)
        timings["ode_step"] = time.perf_counter() - tic

        # Move voltage to FEniCS
        tic = time.perf_counter()
        self.ode.to_dolfin()  # numpy array (ODE solver) -> dolfin function
        timings["ode_to_dolfin"] = time.perf_counter() - tic

        tic = time.perf_counter()
        self.ode.ode_to_pde()  # dolfin function in ODE space (quad?) -> CG1 dolfin function
        timings["ode_to_pde"] = time.perf_counter() - tic

        tic = time.perf_counter()
        self.pde.assign_previous()
        timings["pde_assign_previous_before"] = time.perf_counter() - tic

        logger.debug("PDE step")

        # Solve PDE
        tic = time.perf_counter()
        self.pde.step((t0, t1))
        timings["pde_step"] = time.perf_counter() - tic

        tic = time.perf_counter()
        self.ode.pde_to_ode()  # CG1 dolfin function -> dolfin function in ODE space (quad?)
        timings["pde_to_ode"] = time.perf_counter() - tic

        # Copy voltage from PDE to ODE
        tic = time.perf_counter()
        self.ode.from_dolfin()
        timings["ode_from_dolfin"] = time.perf_counter() - tic

        # If first order splitting, we are done. Otherwise, we need to do a corrective ODE step.
        if np.isclose(theta, 1.0):
            tic = time.perf_counter()
            self.pde.assign_previous()  # But first update previous value in PDE
            timings["pde_assign_previous_after"] = time.perf_counter() - tic

            timings["total_step"] = time.perf_counter() - step_start
            self._log_step_timings(t0, t1, timings)
            self._step_counter += 1
            return

        # Otherwise, we do another ode_step:
        logger.debug(f"Corrective ODE step with t0={t:5f} and dt={(1.0 - theta) * dt:.5f}")

        # To the correction step
        tic = time.perf_counter()
        self.ode.step(t, (1.0 - theta) * dt)
        timings["corrective_ode_step"] = time.perf_counter() - tic

        # And copy the solution back to FEniCS
        tic = time.perf_counter()
        self.ode.to_dolfin()  # numpy array (ODE solver) -> dolfin function
        timings["corrective_ode_to_dolfin"] = time.perf_counter() - tic

        tic = time.perf_counter()
        self.ode.ode_to_pde()  # dolfin function in ODE space (quad?) -> CG1 dolfin function
        timings["corrective_ode_to_pde"] = time.perf_counter() - tic

        tic = time.perf_counter()
        self.pde.assign_previous()
        timings["corrective_pde_assign_previous"] = time.perf_counter() - tic

        timings["total_step"] = time.perf_counter() - step_start
        self._log_step_timings(t0, t1, timings)
        self._step_counter += 1

    def _log_step_timings(
        self,
        t0: float,
        t1: float,
        timings: dict[str, float],
    ) -> None:
        """Accumulate and optionally log timing information for one splitting step."""
        for name, value in timings.items():
            self._timing_totals[name] = self._timing_totals.get(name, 0.0) + value

        if not self.log_timings:
            return

        if self.timing_log_frequency <= 0:
            return

        if self._step_counter % self.timing_log_frequency != 0:
            return

        timing_text = ", ".join(f"{name}={value:.6f}s" for name, value in timings.items())

        logger.info(
            "Monodomain step timing "
            f"step={self._step_counter}, "
            f"t=({t0:.5f}, {t1:.5f}), "
            f"{timing_text}",
        )

    def timing_summary(self) -> dict[str, float]:
        """Return accumulated timing information for all completed steps."""
        return dict(self._timing_totals)
