from __future__ import annotations

import abc
import logging
from enum import Enum, auto
from typing import Any, Callable, Literal, NamedTuple, Optional, Sequence, Union, cast

from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import ufl
from packaging.version import Version
from ufl.core.expr import Expr

from .stimulation import Stimulus
from .telemetry import BaseMonitor, NullMonitor

logger = logging.getLogger(__name__)
_dolfinx_version = Version(dolfinx.__version__)


# A single-field model assembles one form per side; a model with several coupled fields
# assembles them blocked, which is the shape dolfinx' blocked assembly takes.
_BilinearForm = Union[ufl.Form, Sequence[Sequence[Optional[ufl.Form]]]]
_LinearForm = Union[ufl.Form, Sequence[ufl.Form]]


class Status(str, Enum):
    OK = auto()
    NOT_CONVERGING = auto()


class Results(NamedTuple):
    state: dolfinx.fem.Function | Sequence[dolfinx.fem.Function]
    status: Status


def _transform_I_s(
    I_s: Stimulus | Sequence[Stimulus] | ufl.core.expr.Expr | None,
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
    I_s : Stimulus | Sequence[Stimulus] | ufl.core.expr.Expr, optional
        The stimulus, by default None
    bcs : Callable[[BaseModel], Sequence[dolfinx.fem.DirichletBC]], optional
        Factory returning the Dirichlet boundary conditions, by default None (natural
        boundary conditions). It is called once the state space exists and is handed the
        model, because a boundary condition is only honoured on the very function space
        object the model built -- one constructed separately on the same mesh and element
        is silently ignored during assembly.
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
        I_s: Stimulus | Sequence[Stimulus] | ufl.core.expr.Expr | None = None,
        bcs: Callable[[BaseModel], Sequence[dolfinx.fem.DirichletBC]] | None = None,
        monitor: BaseMonitor | None = None,
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
        self.monitor = monitor or NullMonitor()

        if bcs is not None and not callable(bcs):
            raise TypeError(
                "'bcs' must be a callable taking the model and returning the boundary "
                "conditions. A DirichletBC only applies to the function space object it "
                "was built on, and the model builds its own, so the conditions cannot be "
                "constructed before the model exists.",
            )
        self._bcs_factory = bcs

        self.parameters = type(self).default_parameters()
        if params is not None:
            # Keys that no default covers are read by nothing, so a caller that passes one
            # believes it configures something and it does not. Warn as we do for kwargs.
            unknown = set(params) - set(self.parameters)
            if unknown:
                logger.warning(
                    "Unknown parameters: %s",
                    ", ".join(f"{k}={params[k]}" for k in sorted(unknown)),
                )
            self.parameters.update(params)

        self._I_s = _transform_I_s(I_s, dZ=self.dx)

        self._setup_state_space()
        self.bcs = list(self._bcs_factory(self)) if self._bcs_factory is not None else []

        self._timestep = dolfinx.fem.Constant(mesh, self.parameters["default_timestep"])

        self._setup_solver()
        self._assemble_matrix()

    @property
    def mesh(self) -> dolfinx.mesh.Mesh:
        """The mesh the model is discretized on."""
        return self._mesh

    @abc.abstractmethod
    def _setup_state_space(self) -> None: ...

    @property
    @abc.abstractmethod
    def state(self) -> dolfinx.fem.Function | Sequence[dolfinx.fem.Function]: ...

    @abc.abstractmethod
    def assign_previous(self) -> None: ...

    def _setup_solver(self) -> None:
        """Build the linear problem the time stepping reuses every step.

        Subclasses whose :meth:`variational_forms` returns something other than a pair of
        scalar forms -- nested lists for a blocked system, say -- override this.
        """
        a, L = self.variational_forms(self._timestep)

        kwargs: dict[str, Any] = {}
        if _dolfinx_version >= Version("0.10"):
            kwargs["petsc_options_prefix"] = "beat_base_model_"

        # Blocked forms may contain ``None`` blocks, which dolfinx's annotation does not admit.
        self._solver = dolfinx.fem.petsc.LinearProblem(
            cast(Any, a),
            L,
            u=self.state,
            bcs=self.bcs,
            form_compiler_options=self.parameters["form_compiler_options"],
            jit_options=self.parameters["jit_options"],
            petsc_options=self.parameters["petsc_options"],
            **kwargs,
        )

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
    def variational_forms(
        self,
        dt: Expr | float,
    ) -> tuple[_BilinearForm, _LinearForm]:
        """Create the variational forms corresponding to the given
        discretization of the given system of equations.

        Parameters
        ----------
        dt : Expr | float
            The time step

        Returns
        -------
        tuple[_BilinearForm, _LinearForm]
            The bilinear and linear form. A model with several coupled fields returns
            these blocked, as a nested list of forms and a list of forms respectively.

        """
        ...

    def _assemble_matrix(self) -> None:
        """(Re-)assemble the system matrix."""
        # The compiled form spans both the single-field and the blocked shape, which the
        # assembly overloads cannot be resolved against statically.
        A, a = self._solver.A, cast(Any, self._solver.a)
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, a, bcs=self.bcs)  # type: ignore[arg-type, misc]
        A.assemble()

    def _assemble_rhs(self) -> None:
        """(Re-)assemble the right-hand side vector."""
        b, a = self._solver.b, cast(Any, self._solver.a)
        with b.localForm() as b_loc:
            b_loc.set(0)
        dolfinx.fem.petsc.assemble_vector(b, self._solver.L)  # type: ignore[arg-type]
        # Move the Dirichlet columns of the matrix over to the right-hand side before the
        # ghost contributions are summed, then overwrite the constrained rows afterwards.
        dolfinx.fem.petsc.apply_lifting(b, [a], bcs=[self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
        dolfinx.fem.petsc.set_bc(b, self.bcs)

    def _solve_linear_system(self) -> None:
        """Solve the assembled system into the state vector."""
        state = self.state
        assert isinstance(
            state,
            dolfinx.fem.Function,
        ), "A model whose state is several fields must override _solve_linear_system"
        self._solver.solver.solve(self._solver.b, state.x.petsc_vec)

    def _scatter_forward(self) -> None:
        """Update the ghost values of every field the state is made of."""
        for function in self._state_functions():
            function.x.scatter_forward()

    def _state_functions(self) -> Sequence[dolfinx.fem.Function]:
        """The state as a sequence, whether it is one field or several."""
        state = self.state
        if isinstance(state, dolfinx.fem.Function):
            return (state,)
        return state

    def step(self, interval):
        """Perform a single time step.

        Parameters
        ----------
        interval : tuple[float, float]
            The time interval (T0, T)
        """
        t0, t1 = interval
        dt = t1 - t0
        theta = self.parameters["theta"]
        t = t0 + theta * dt

        with self.monitor.track_time("pde_total_step"):
            with self.monitor.track_time("pde_set_time"):
                self.time.value = t

            timestep_unchanged = abs(dt - float(self._timestep)) < 1.0e-12

            if not timestep_unchanged:
                self._timestep.value = dt
                with self.monitor.track_time("pde_update_matrices"):
                    self._assemble_matrix()

            with self.monitor.track_time("pde_update_rhs"):
                self._assemble_rhs()

            with self.monitor.track_time("pde_linear_solve"):
                self._solve_linear_system()

            # Record solver metrics
            self.monitor.record_ksp(self._solver.solver)

            with self.monitor.track_time("pde_scatter_forward"):
                self._scatter_forward()

        # Trigger logging/end-of-step aggregation
        self.monitor.advance_step(t0, t1)

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
        T0, T = interval
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
