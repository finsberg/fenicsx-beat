from __future__ import annotations

import abc
import logging
import time
from enum import Enum, auto
from typing import Any, Literal, NamedTuple, Sequence

from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import ufl
from packaging.version import Version
from ufl.core.expr import Expr

from .stimulation import Stimulus

logger = logging.getLogger(__name__)
_dolfinx_version = Version(dolfinx.__version__)


class Status(str, Enum):
    OK = auto()
    NOT_CONVERGING = auto()


class Results(NamedTuple):
    state: dolfinx.fem.Function
    status: Status


def _transform_I_s(
    I_s: Stimulus | Sequence[Stimulus] | ufl.Coefficient | None,
    dZ: ufl.Measure,
) -> list[Stimulus]:
    if I_s is None:
        return [Stimulus(expr=ufl.zero(), dZ=dZ)]
    if isinstance(I_s, Stimulus):
        return [I_s]
    if isinstance(I_s, ufl.core.expr.Expr):
        return [Stimulus(expr=I_s, dZ=dZ)]

    # FIXME: Might need more checks here
    return list(I_s)


class BaseModel:
    """
    Base class for models.

    Parameters
    ----------
    time : dolfinx.fem.Constant
        The current time
    mesh : dolfinx.mesh.Mesh
        The mesh
    dx : ufl.Measure, optional
        The measure for the spatial domain, by default None
    params : dict, optional
        Parameters for the model, by default None
    I_s : Stimulus | Sequence[Stimulus] | ufl.Coefficient, optional
        The stimulus, by default None
    jit_options : dict, optional
        JIT options, by default None
    form_compiler_options : dict, optional
        Form compiler options, by default None
    petsc_options : dict, optional
        PETSc options, by default None

    """

    def __init__(
        self,
        time: dolfinx.fem.Constant,
        mesh: dolfinx.mesh.Mesh,
        dx: ufl.Measure | None = None,
        params: dict[str, Any] | None = None,
        I_s: Stimulus | Sequence[Stimulus] | ufl.Coefficient | None = None,
        **kwargs: Any,
    ) -> None:
        # Warn about unused kwargs
        if kwargs:
            logger.warning(
                "Unused keyword arguments: %s",
                ", ".join(f"{k}={v}" for k, v in kwargs.items()),
            )

        self._mesh = mesh
        self.time = time
        self.dx = dx or ufl.dx(domain=mesh)

        self.parameters = type(self).default_parameters()
        if params is not None:
            self.parameters.update(params)

        self._step_counter = 0
        self._timing_totals: dict[str, float] = {}
        self._ksp_total_iterations = 0
        self._ksp_max_iterations = 0
        self._ksp_last_iterations = 0
        self._ksp_last_residual_norm = 0.0
        self._ksp_last_converged_reason = 0

        form_compiler_options = self.parameters["form_compiler_options"]
        jit_options = self.parameters["jit_options"]
        petsc_options = self.parameters["petsc_options"]

        self._I_s = _transform_I_s(I_s, dZ=self.dx)

        self._setup_state_space()

        self._timestep = dolfinx.fem.Constant(mesh, self.parameters["default_timestep"])
        a, L = self.variational_forms(self._timestep)

        kwargs = {}
        if _dolfinx_version >= Version("0.10"):
            kwargs["petsc_options_prefix"] = "beat_base_model_"

        self._solver = dolfinx.fem.petsc.LinearProblem(
            a,
            L,
            u=self.state,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            petsc_options=petsc_options,
            **kwargs,
        )
        dolfinx.fem.petsc.assemble_matrix(self._solver.A, self._solver.a)  # type: ignore
        self._solver.A.assemble()

    @abc.abstractmethod
    def _setup_state_space(self) -> None: ...

    @property
    @abc.abstractmethod
    def state(self) -> dolfinx.fem.Function: ...

    @abc.abstractmethod
    def assign_previous(self) -> None: ...

    @staticmethod
    def default_parameters(
        solver_type: Literal["iterative", "direct"] = "direct",
    ) -> dict[str, Any]:
        if solver_type == "iterative":
            petsc_options = {
                "ksp_type": "cg",
                "pc_type": "hypre",
                # "pc_type": "petsc_amg",
                "pc_hypre_type": "boomeramg",
                # "ksp_norm_type": "unpreconditioned",
                # "ksp_atol": 1e-15,
                # "ksp_rtol": 1e-10,
                # "ksp_max_it": 10_000,
                # "ksp_error_if_not_converged": False,
            }
        else:
            petsc_options = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        return {
            "theta": 0.5,
            "degree": 1,
            "family": "Lagrange",
            "default_timestep": 1.0,
            "jit_options": {},
            "form_compiler_options": {},
            "petsc_options": petsc_options,
            "log_timings": False,
            "timing_log_frequency": 1,
        }

    @abc.abstractmethod
    def variational_forms(self, dt: Expr | float) -> tuple[ufl.Form, ufl.Form]:
        """Create the variational forms corresponding to the given
        discretization of the given system of equations.

        Parameters
        ----------
        dt : Expr | float
            The time step

        Returns
        -------
        tuple[ufl.Form, ufl.Form]
            The variational form and the precondition

        """
        ...

    def _update_matrices(self):
        """
        Re-assemble matrix.
        """
        self._solver.A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self._solver.A, self._solver.a)  # type: ignore
        self._solver.A.assemble()

    def _update_rhs(self):
        """
        Re-assemble RHS vector
        """
        with self._solver.b.localForm() as b_loc:
            b_loc.set(0)
        dolfinx.fem.petsc.assemble_vector(self._solver.b, self._solver.L)
        self._solver.b.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )

    def step(self, interval):
        """Perform a single time step.

        Parameters
        ----------
        interval : tuple[float, float]
            The time interval (T0, T)
        """
        step_start = time.perf_counter()
        timings: dict[str, float] = {}

        # Extract interval and time step.
        t0, t1 = interval
        dt = t1 - t0
        theta = self.parameters["theta"]
        t = t0 + theta * dt

        tic = time.perf_counter()
        self.time.value = t
        timings["pde_set_time"] = time.perf_counter() - tic

        # Update matrix only when the time step changes.
        timings["pde_update_matrices"] = 0.0

        # Update matrix and linear solvers etc as needed
        timestep_unchanged = abs(dt - float(self._timestep)) < 1.0e-12

        if not timestep_unchanged:
            tic = time.perf_counter()
            self._timestep.value = dt
            self._update_matrices()
            timings["pde_update_matrices"] = time.perf_counter() - tic

        tic = time.perf_counter()
        self._update_rhs()
        timings["pde_update_rhs"] = time.perf_counter() - tic

        # Solve linear system and update ghost values in the solution.
        tic = time.perf_counter()
        self._solver.solver.solve(self._solver.b, self.state.x.petsc_vec)
        timings["pde_linear_solve"] = time.perf_counter() - tic

        tic = time.perf_counter()
        self.state.x.scatter_forward()
        timings["pde_scatter_forward"] = time.perf_counter() - tic

        timings["pde_total_step"] = time.perf_counter() - step_start

        self._update_ksp_summary()
        self._log_step_timings(t0, t1, timings)
        self._step_counter += 1

    def _update_ksp_summary(self) -> None:
        """Update accumulated PETSc/KSP solver information."""
        ksp = self._solver.solver

        iterations = int(ksp.getIterationNumber())
        self._ksp_last_iterations = iterations
        self._ksp_total_iterations += iterations
        self._ksp_max_iterations = max(self._ksp_max_iterations, iterations)

        try:
            self._ksp_last_residual_norm = float(ksp.getResidualNorm())
        except PETSc.Error:
            self._ksp_last_residual_norm = float("nan")

        try:
            self._ksp_last_converged_reason = int(ksp.getConvergedReason())
        except PETSc.Error:
            self._ksp_last_converged_reason = 0

    def _log_step_timings(
        self,
        t0: float,
        t1: float,
        timings: dict[str, float],
    ) -> None:
        """Accumulate and optionally log PDE timing information."""
        for name, value in timings.items():
            self._timing_totals[name] = self._timing_totals.get(name, 0.0) + value

        if not self.parameters.get("log_timings", False):
            return

        timing_log_frequency = int(self.parameters.get("timing_log_frequency", 1))
        if timing_log_frequency <= 0:
            return

        if self._step_counter % timing_log_frequency != 0:
            return

        timing_text = ", ".join(f"{name}={value:.6f}s" for name, value in timings.items())

        logger.info(
            "PDE step timing "
            f"step={self._step_counter}, "
            f"t=({t0:.5f}, {t1:.5f}), "
            f"ksp_iterations={self._ksp_last_iterations}, "
            f"ksp_residual_norm={self._ksp_last_residual_norm:.6e}, "
            f"ksp_converged_reason={self._ksp_last_converged_reason}, "
            f"{timing_text}",
        )

    def timing_summary(self) -> dict[str, float]:
        """Return accumulated PDE timing information."""
        return dict(self._timing_totals)

    def ksp_summary(self) -> dict[str, float]:
        """Return accumulated PETSc/KSP solver information."""
        average_iterations = (
            self._ksp_total_iterations / self._step_counter if self._step_counter > 0 else 0.0
        )

        return {
            "ksp_total_iterations": float(self._ksp_total_iterations),
            "ksp_average_iterations": float(average_iterations),
            "ksp_max_iterations": float(self._ksp_max_iterations),
            "ksp_last_iterations": float(self._ksp_last_iterations),
            "ksp_last_residual_norm": float(self._ksp_last_residual_norm),
            "ksp_last_converged_reason": float(self._ksp_last_converged_reason),
        }

    def _G_stim(self, w):
        return sum([i.expr * w * i.dz for i in self._I_s])

    def solve(
        self,
        interval: tuple[float, float],
        dt: float | None = None,
    ) -> Results:
        """
        Solve on the given time interval.

        Parameters
        ----------
        interval : tuple[float, float]
            The time interval (T0, T)
        dt : float, optional
            The time step, by default None

        Returns
        -------
        Results
            The results of the solution

        """

        # Initial set-up
        # Solve on entire interval if no interval is given.
        (T0, T) = interval
        if dt is None:
            dt = T - T0
        t0 = T0
        t1 = T0 + dt

        # Step through time steps until at end time
        while True:
            logger.info("Solving on t = (%g, %g)" % (t0, t1))
            self.step((t0, t1))

            # Yield solutions
            # yield (t0, t1), self.solution_fields()

            # Break if this is the last step
            if (t1 + dt) > (T + 1e-12):
                break

            self.assign_previous()

            t0 = t1
            t1 = t0 + dt

        return Results(state=self.state, status=Status.OK)
