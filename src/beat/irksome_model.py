import logging
from typing import Sequence

import dolfinx
import ufl

from .monodomain_model import MonodomainModel
from .stimulation import Stimulus

logger = logging.getLogger(__name__)


class IrksomeMonodomainModel(MonodomainModel):
    r"""The monodomain model advanced by an Irksome Runge-Kutta stepper.

    Built like :class:`beat.MonodomainModel` (same parameters, ``monitor``, ``bcs`` factory and
    restart support); the theta-rule linear problem is replaced by an Irksome
    ``StageDerivativeTimeStepper`` for the given ``butcher_tableau``.

    The ``bcs`` factory must return Irksome-compatible conditions, built with
    ``irksome.backends.dolfinx.dirichletbc`` (plain ``dolfinx.fem.dirichletbc`` objects are rejected
    by Irksome). A restart reproduces the run to round-off rather than bit for bit: the stepper
    starts each solve from the previous step's stages, which a checkpoint does not hold.
    """

    def __init__(
        self,
        time: dolfinx.fem.Constant,
        mesh: dolfinx.mesh.Mesh,
        M: ufl.Coefficient | float,
        butcher_tableau,
        I_s: Stimulus | Sequence[Stimulus] | ufl.core.expr.Expr | None = None,
        params=None,
        C_m: float = 1.0,
        dx: ufl.Measure | None = None,
        **kwargs,
    ):
        try:
            import irksome
        except ImportError:
            raise ImportError(
                "The 'irksome' package is required for IrksomeMonodomainModel. "
                "Install it with pip: 'pip install irksome[dolfinx]'.",
            )
        self._irksome = irksome
        self.butcher_tableau = butcher_tableau
        super().__init__(
            time=time,
            mesh=mesh,
            M=M,
            I_s=I_s,
            params=params,
            C_m=C_m,
            dx=dx,
            **kwargs,
        )

    def _setup_solver(self) -> None:
        """Build the Irksome stepper for ``C_m dv/dt = div(M grad v) + I_stim``."""
        irksome = self._irksome
        v = self._state
        w = ufl.TestFunction(self.V)
        F = (self.C_m * irksome.Dt(v) * w + ufl.inner(self._M * ufl.grad(v), ufl.grad(w))) * self.dx
        F -= self._G_stim(w)
        self.stepper = irksome.stage_derivative.StageDerivativeTimeStepper(
            F,
            self.butcher_tableau,
            self.time,
            self._timestep,
            self._state,
            bcs=self.bcs,
            solver_parameters=self.parameters["petsc_options"],
            backend="dolfinx",
        )

    def _assemble_matrix(self) -> None:
        """Nothing to do: Irksome assembles its systems itself in :meth:`step`."""

    def assign_previous(self) -> None:
        """Nothing to do: Irksome keeps its own history within ``advance()``."""

    def step(self, interval) -> None:
        t0, t1 = interval
        dt = t1 - t0
        with self.monitor.track_time("pde_total_step"):
            self._timestep.value = dt
            self.time.value = t0
            self.stepper.advance()
            self.time.value = float(self.time) + dt
        self.monitor.advance_step(t0, t1)
