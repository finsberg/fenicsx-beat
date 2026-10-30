from __future__ import annotations

import logging
from typing import Any, Literal, Sequence, cast

from petsc4py import PETSc

import basix
import dolfinx
import dolfinx.fem.petsc
import ufl
from ufl.algorithms.analysis import extract_coefficients, extract_constants
from ufl.core.expr import Expr

from .base_model import BaseModel
from .stimulation import Stimulus

logger = logging.getLogger(__name__)


class MonodomainModel(BaseModel):
    r"""Solve

    .. math::

        \frac{\partial V}{\partial t} -
        \nabla \cdot \left( M \nabla V \right) - I_{\mathrm{stim}} = 0

    The right-hand side is affine in the previous potential, :math:`L = B v_{n} + L_0`,
    where :math:`B` holds the mass and the explicit part of the diffusion and :math:`L_0`
    the stimulus. With the parameter ``rhs_matrix`` (default True), :math:`B` is assembled
    together with the system matrix, i.e. only when the time step changes, and every step
    assembles only :math:`L_0` and adds the product :math:`B v_{n}`. This assumes, like the
    system matrix already does, that the conductivity and the capacitance do not change in
    time. Set ``rhs_matrix`` to False to assemble the whole right-hand side every step;
    this is also done, with a warning, when the right-hand side is not affine in
    :math:`v_{n}` or :math:`B` depends on data the system matrix does not (e.g. a stimulus
    linear in :math:`v_{n}` that varies in time).

    """

    def __init__(
        self,
        time: dolfinx.fem.Constant,
        mesh: dolfinx.mesh.Mesh,
        M: ufl.Coefficient | float,
        I_s: Stimulus | Sequence[Stimulus] | ufl.core.expr.Expr | None = None,
        params=None,
        C_m: float = 1.0,
        dx: ufl.Measure | None = None,
        **kwargs,
    ) -> None:
        self._M = M
        self.C_m = dolfinx.fem.Constant(mesh, C_m)
        super().__init__(mesh=mesh, time=time, params=params, I_s=I_s, dx=dx, **kwargs)

    @staticmethod
    def default_parameters(
        solver_type: Literal["iterative", "direct"] = "direct",
    ) -> dict[str, Any]:
        parameters = BaseModel.default_parameters(solver_type)
        parameters["rhs_matrix"] = True
        return parameters

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
    def v(self) -> dolfinx.fem.Function:
        """The transmembrane potential."""
        return self._state

    @property
    def state(self) -> dolfinx.fem.Function:
        """The transmembrane potential.

        Kept for compatibility; :attr:`v` says the same thing by name and is preferred.
        """
        return self._state

    def assign_previous(self):
        self.v_.x.array[:] = self.state.x.array[:]

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

    def _setup_solver(self) -> None:
        super()._setup_solver()
        self._rhs_B: tuple[dolfinx.fem.Form, PETSc.Mat] | None = None
        self._rhs_L0: dolfinx.fem.Form | None = None
        if not self.parameters["rhs_matrix"]:
            return

        a, L = self.variational_forms(self._timestep)
        B = ufl.algorithms.expand_derivatives(
            ufl.derivative(L, self.v_, ufl.TrialFunction(self.V)),
        )
        if not ufl.algorithms.expand_derivatives(ufl.derivative(B, self.v_)).empty():
            logger.warning("The right-hand side is not affine in v_; assembling it every step")
            return
        # B is only re-assembled with the system matrix, so it may depend on nothing the
        # system matrix does not. A stimulus linear in v_ ends up in B, and may depend on
        # the time or on a field that changes between steps.
        if not (
            set(extract_coefficients(B)) <= set(extract_coefficients(a))
            and set(extract_constants(B)) <= set(extract_constants(a))
        ):
            logger.warning(
                "The right-hand side matrix depends on data the system matrix does not; "
                "assembling the right-hand side every step",
            )
            return
        L0 = ufl.replace(L, {self.v_: ufl.zero()})

        options = {
            "form_compiler_options": self.parameters["form_compiler_options"],
            "jit_options": self.parameters["jit_options"],
        }
        B_form = dolfinx.fem.form(B, **options)
        self._rhs_B = (B_form, dolfinx.fem.petsc.create_matrix(B_form))
        # Without a stimulus nothing is left of the right-hand side but B v_
        if not L0.empty():
            self._rhs_L0 = dolfinx.fem.form(L0, **options)

    def _assemble_matrix(self) -> None:
        super()._assemble_matrix()
        if self._rhs_B is not None:
            # Without boundary conditions: the constrained rows are overwritten in the
            # right-hand side, and the constrained columns act on v_ as in the assembled form
            B_form, B = self._rhs_B
            B.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(B, B_form)  # type: ignore[arg-type]
            B.assemble()

    def _assemble_rhs(self) -> None:
        if self._rhs_B is None:
            super()._assemble_rhs()
            return
        b, a = self._solver.b, cast(Any, self._solver.a)
        with b.localForm() as b_loc:
            b_loc.set(0)
        if self._rhs_L0 is not None:
            dolfinx.fem.petsc.assemble_vector(b, self._rhs_L0)  # type: ignore[arg-type]
        dolfinx.fem.petsc.apply_lifting(b, [a], bcs=[self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
        # b += B v_, on the owned entries, once the ghost contributions are summed
        self._rhs_B[1].multAdd(self.v_.x.petsc_vec, b, b)
        dolfinx.fem.petsc.set_bc(b, self.bcs)
