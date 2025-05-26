from __future__ import annotations

import logging
from typing import Sequence

import basix
import dolfinx
import ufl
from ufl.core.expr import Expr

from .base_model import BaseModel
from .stimulation import Stimulus

logger = logging.getLogger(__name__)


class MonodomainModel(BaseModel):
    r"""Solve

    .. math::

        \frac{\partial V}{\partial t} -
        \nabla \cdot \left( M \nabla V \right) - I_{\mathrm{stim}} = 0

    """

    def __init__(
        self,
        time: dolfinx.fem.Constant,
        mesh: dolfinx.mesh.Mesh,
        M: ufl.Coefficient | float,
        I_s: Stimulus | Sequence[Stimulus] | ufl.Coefficient | None = None,
        params=None,
        C_m: float = 1.0,
        dx: ufl.Measure | None = None,
        **kwargs,
    ) -> None:
        self._M = M
        self.C_m = dolfinx.fem.Constant(mesh, C_m)
        super().__init__(mesh=mesh, time=time, params=params, I_s=I_s, dx=dx, **kwargs)

    def _setup_state_space(self) -> None:
        # Set-up function spaces
        k = self.parameters["degree"]
        family = self.parameters["family"]

        element = basix.ufl.element(family=family, cell=self._mesh.basix_cell(), degree=k)

        self.V = dolfinx.fem.functionspace(self._mesh, element)

        # Set-up solution fields:
        self.v_ = dolfinx.fem.Function(self.V, name="v_")
        self._state = dolfinx.fem.Function(self.V, name="v")

    @property
    def state(self) -> dolfinx.fem.Function:
        return self._state

    def assign_previous(self):
        self.v_.x.array[:] = self.state.x.array[:]

    @staticmethod
    def default_parameters():
        params = super(MonodomainModel, MonodomainModel).default_parameters()
        params["use_custom_preconditioner"] = True
        return params

    def variational_forms(self, dt: Expr | float) -> tuple[ufl.Form, ufl.Form]:
        """Create the variational forms corresponding to the given
        discretization of the given system of equations.

        Parameters
        ----------
        dt : float
            Time step size

        Returns
        -------
        tuple[ufl.Form, ufl.Form]
            The variational form and the precondition

        """
        theta = self.parameters["theta"]

        # Define variational formulation
        v = ufl.TrialFunction(self.V)
        w = ufl.TestFunction(self.V)

        # # Set-up variational problem
        Dt_v_dt = v - self.v_
        v_mid = theta * v + (1.0 - theta) * self.v_

        theta_parabolic = ufl.inner(self._M * ufl.grad(v_mid), ufl.grad(w))

        G = (self.C_m * Dt_v_dt * w + dt * theta_parabolic) * self.dx - dt * self._G_stim(w)
        a, L = ufl.system(G)

        return a, L
