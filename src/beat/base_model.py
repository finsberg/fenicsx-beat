from __future__ import annotations
from typing import Any, NamedTuple
import abc
import logging
from enum import Enum, auto


from petsc4py import PETSc
import dolfinx
import dolfinx.fem.petsc
import ufl
from ufl.core.expr import Expr


logger = logging.getLogger(__name__)


class Status(str, Enum):
    OK = auto()
    NOT_CONVERGING = auto()


class Results(NamedTuple):
    state: dolfinx.fem.Function
    status: Status


class Stimulus(NamedTuple):
    dz: ufl.Measure
    expr: dolfinx.fem.Expression


class BaseModel:
    def __init__(
        self,
        time: dolfinx.fem.Constant,
        mesh: dolfinx.mesh.Mesh,
        params: dict[str, Any] | None = None,
        I_s: Stimulus | ufl.Coefficient | None = None,
        jit_options: dict[str, Any] | None = None,
        form_compiler_options: dict[str, Any] | None = None,
        petsc_options: dict[str, Any] | None = None,
    ) -> None:
        self._mesh = mesh
        self.time = time

        self.parameters = type(self).default_parameters()
        if params is not None:
            self.parameters.update(params)

        if I_s is None:
            I_s = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
        if not isinstance(I_s, Stimulus):
            I_s = Stimulus(expr=I_s, dz=ufl.dx)
        self._I_s = I_s

        self._setup_state_space()

        self._timestep = dolfinx.fem.Constant(mesh, self.parameters["default_timestep"])
        (self._G, self._prec) = self.variational_forms(self._timestep)

        a, L = ufl.system(self._G)
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
    def _setup_state_space(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def state(self) -> dolfinx.fem.Function:
        ...

    @abc.abstractmethod
    def assign_previous(self) -> None:
        ...

    @staticmethod
    def default_parameters():

        return {
            "theta": 0.5,
            "degree": 1,
            "family": "Lagrange",
            "default_timestep": 1.0,
            "jit_options": {},
            "form_compiler_options": {},
            "petsc_options": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        }

    @abc.abstractmethod
    def variational_forms(self, k_n: Expr | float) -> tuple[ufl.Form, ufl.Form]:
        """Create the variational forms corresponding to the given
        discretization of the given system of equations.

        Parameters
        ----------
        k_n : Expr | float
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
        Solve on the given time step (t0, t1).

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval (t0, t1) for the step

        *Invariants*
          Assuming that v_ is in the correct state for t0, gives
          self.v in correct state at t1.
        """

        # timer = dolfin.Timer("PDE Step")

        # Extract interval and thus time-step
        (t0, t1) = interval
        dt = t1 - t0
        theta = self.parameters["theta"]
        t = t0 + theta * dt
        # breakpoint()
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

    def solve(
        self,
        interval: tuple[float, float],
        dt: float | None = None,
    ):
        """
        Solve the discretization on a given time interval (t0, t1)
        with a given timestep dt and return generator for a tuple of
        the interval and the current solution.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int, optional)
            The timestep for the solve. Defaults to length of interval

        *Returns*
          (timestep, solution_field) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          solutions = solver.solve((0.0, 1.0), 0.1)

          # Iterate over generator (computes solutions as you go)
          for (interval, solution_fields) in solutions:
            (t0, t1) = interval
            v_, v = solution_fields
            # do something with the solutions
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
