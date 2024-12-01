from __future__ import annotations

import abc
import logging
from enum import Enum, auto
from typing import Any, Literal, NamedTuple, Sequence

from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import ufl
from ufl.core.expr import Expr

from .stimulation import Stimulus

logger = logging.getLogger(__name__)


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
        jit_options: dict[str, Any] | None = None,
        form_compiler_options: dict[str, Any] | None = None,
        petsc_options: dict[str, Any] | None = None,
    ) -> None:
        self._mesh = mesh
        self.time = time
        self.dx = dx or ufl.dx(domain=mesh)

        self.parameters = type(self).default_parameters()
        if params is not None:
            self.parameters.update(params)

        self._I_s = _transform_I_s(I_s, dZ=self.dx)

        self._setup_state_space()

        self._timestep = dolfinx.fem.Constant(mesh, self.parameters["default_timestep"])
        a, L = self.variational_forms(self._timestep)
        self._solver = dolfinx.fem.petsc.LinearProblem(
            a,
            L,
            u=self.state,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            petsc_options=petsc_options,
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
                # "pc_type": "hypre",
                "pc_type": "petsc_amg",
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
        """
        Perform a single time step.

        Parameters
        ----------
        interval : tuple[float, float]
            The time interval (T0, T)

        """

        # timer = dolfin.Timer("PDE Step")

        # Extract interval and thus time-step
        (t0, t1) = interval
        dt = t1 - t0
        theta = self.parameters["theta"]
        t = t0 + theta * dt
        self.time.value = t

        # Update matrix and linear solvers etc as needed
        timestep_unchanged = abs(dt - float(self._timestep)) < 1.0e-12
        if not timestep_unchanged:
            self._timestep.value = dt
            self._update_matrices()

        self._update_rhs()
        # Solve linear system and update ghost values in the solution

        self._solver.solver.solve(self._solver.b, self.state.x.petsc_vec)
        self.state.x.scatter_forward()

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
