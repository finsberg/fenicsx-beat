import logging
from typing import Sequence

import basix
import dolfinx
import ufl

from .base_model import Results, Status, _transform_I_s
from .monodomain_model import MonodomainModel
from .stimulation import Stimulus

logger = logging.getLogger(__name__)


class IrksomeMonodomainModel(MonodomainModel):
    r"""Solve Monodomain model using Irksome for Runge-Kutta time stepping."""

    def __init__(
        self,
        time: dolfinx.fem.Constant,
        mesh: dolfinx.mesh.Mesh,
        M: ufl.Coefficient | float,
        butcher_tableau,
        I_s: Stimulus | Sequence[Stimulus] | ufl.Coefficient | None = None,
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

        self._mesh = mesh
        self.time = time
        self.dx = dx or ufl.dx(domain=mesh)
        self._M = M
        self.C_m = dolfinx.fem.Constant(mesh, C_m)
        self._I_s = _transform_I_s(I_s, dZ=self.dx)
        self.butcher_tableau = butcher_tableau

        self.parameters = MonodomainModel.default_parameters()
        if params is not None:
            self.parameters.update(params)

        self._setup_state_space()

        self._timestep = dolfinx.fem.Constant(mesh, self.parameters["default_timestep"])

        # Define the continuous weak form for Irksome
        v = self._state
        w = ufl.TestFunction(self.V)

        # F = C_m * Dt(v) * w + M * grad(v) * grad(w) - I_stim * w
        F = (self.C_m * irksome.Dt(v) * w + ufl.inner(self._M * ufl.grad(v), ufl.grad(w))) * self.dx
        F -= self._G_stim(w)

        # Setup Irksome stepper
        self.stepper = irksome.stage_derivative.StageDerivativeTimeStepper(
            F,
            self.butcher_tableau,
            self.time,
            self._timestep,
            self._state,
            bcs=[],
            solver_parameters=self.parameters["petsc_options"],
            backend="dolfinx",
        )

    def _setup_state_space(self) -> None:
        k = self.parameters["degree"]
        family = self.parameters["family"]
        element = basix.ufl.element(family=family, cell=self._mesh.basix_cell(), degree=k)
        self.V = dolfinx.fem.functionspace(self._mesh, element)
        self._state = dolfinx.fem.Function(self.V, name="v")

    @property
    def state(self) -> dolfinx.fem.Function:
        return self._state

    def assign_previous(self):
        # Irksome inherently updates the states within `advance()`,
        # so manual history assignment isn't required here.
        pass

    def _G_stim(self, w):
        return sum([i.expr * w * i.dz for i in self._I_s])

    def step(self, interval):
        t0, t1 = interval
        dt = t1 - t0

        self._timestep.value = dt
        self.time.value = t0

        # Take the Runge-Kutta step
        self.stepper.advance()

        # Advance the constant time attribute manually
        self.time.value = float(self.time) + dt

    def solve(self, interval, dt=None) -> Results:
        T0, T = interval
        if dt is None:
            dt = T - T0
        t0 = T0
        t1 = T0 + dt

        while t1 < T + 1e-12:
            self.step((t0, t1))
            t0 = t1
            t1 = t0 + dt

        return Results(state=self.state, status=Status.OK)
