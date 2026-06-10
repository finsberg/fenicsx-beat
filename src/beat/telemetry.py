import abc
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Union

from mpi4py import MPI
from petsc4py import PETSc

logger = logging.getLogger(__name__)


class BaseMonitor(abc.ABC):
    @abc.abstractmethod
    @contextmanager
    def track_time(self, name: str):
        yield

    @abc.abstractmethod
    def record_ksp(self, ksp: PETSc.KSP) -> None:
        pass

    @abc.abstractmethod
    def advance_step(self, t0: float, t1: float) -> None:
        pass


class NullMonitor(BaseMonitor):
    @contextmanager
    def track_time(self, name: str):
        yield

    def record_ksp(self, ksp: PETSc.KSP) -> None:
        pass

    def advance_step(self, t0: float, t1: float) -> None:
        pass


class PerformanceMonitor(BaseMonitor):
    """A monitor that accumulates timings and KSP stats, logging them periodically,
    and can save/display a final summary."""

    def __init__(self, log_frequency: int = 1, comm: MPI.Intracomm = MPI.COMM_WORLD):
        self.log_frequency = log_frequency
        self.comm = comm
        self.step_counter = 0
        self.timings: Dict[str, float] = {}

        self.ksp_total_iterations = 0
        self.ksp_max_iterations = 0
        self.ksp_last_iterations = 0
        self.ksp_last_residual_norm = 0.0
        self.ksp_last_converged_reason = 0

    @contextmanager
    def track_time(self, name: str):
        tic = time.perf_counter()
        try:
            yield
        finally:
            toc = time.perf_counter()
            self.timings[name] = self.timings.get(name, 0.0) + (toc - tic)

    def record_ksp(self, ksp: PETSc.KSP) -> None:
        try:
            iterations = int(ksp.getIterationNumber())
            self.ksp_last_iterations = iterations
            self.ksp_total_iterations += iterations
            self.ksp_max_iterations = max(self.ksp_max_iterations, iterations)
            self.ksp_last_residual_norm = float(ksp.getResidualNorm())
            self.ksp_last_converged_reason = int(ksp.getConvergedReason())
        except PETSc.Error:
            pass

    def advance_step(self, t0: float, t1: float) -> None:
        self.step_counter += 1

        if self.log_frequency <= 0 or self.step_counter % self.log_frequency != 0:
            return

        timing_text = ", ".join(f"{name}={value:.6f}s" for name, value in self.timings.items())
        logger.info(
            f"PDE step timing step={self.step_counter}, "
            f"t=({t0:.5f}, {t1:.5f}), "
            f"ksp_iterations={self.ksp_last_iterations}, "
            f"ksp_residual_norm={self.ksp_last_residual_norm:.6e}, "
            f"ksp_converged_reason={self.ksp_last_converged_reason}, "
            f"{timing_text}",
        )

    def display_summary(self) -> None:
        """Logs a nicely formatted summary of the accumulated timings and metrics."""
        if self.comm.rank != 0:
            return

        summary = ["\n" + "=" * 50]
        summary.append(f"{'PERFORMANCE SUMMARY':^50}")
        summary.append("=" * 50)
        summary.append(f"Total Steps:           {self.step_counter}")
        summary.append(f"KSP Total Iterations:  {self.ksp_total_iterations}")
        summary.append(f"KSP Max Iterations:    {self.ksp_max_iterations}")
        summary.append("-" * 50)
        summary.append(f"{'Metric':<35} | {'Time (s)':>10}")
        summary.append("-" * 50)

        # Sort timings by duration (descending)
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        for name, duration in sorted_timings:
            summary.append(f"{name:<35} | {duration:>10.4f}")
        summary.append("=" * 50 + "\n")

        logger.info("\n".join(summary))

    def save_summary(self, filepath: Union[str, Path]) -> None:
        """Saves the performance metrics to a JSON file."""
        if self.comm.rank != 0:
            return

        data = {
            "total_steps": self.step_counter,
            "ksp": {
                "total_iterations": self.ksp_total_iterations,
                "max_iterations": self.ksp_max_iterations,
            },
            "timings": self.timings,
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Performance summary saved to {filepath}")
