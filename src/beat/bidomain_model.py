from __future__ import annotations

import logging
from typing import Any, NamedTuple, Sequence, cast

from petsc4py import PETSc

import basix
import dolfinx
import dolfinx.fem.petsc
import ufl
from ufl.core.expr import Expr

from .base_model import BaseModel, _BilinearForm, _LinearForm
from .stimulation import Stimulus
from .utils import cpp_function_space

logger = logging.getLogger(__name__)


class BidomainState(NamedTuple):
    """The pair of fields a bidomain model solves for."""

    v: dolfinx.fem.Function
    u_e: dolfinx.fem.Function


class BidomainModel(BaseModel):
    r"""Solve the bidomain model

    .. math::

        C_m \frac{\partial v}{\partial t}
            - \nabla \cdot (M_i \nabla v) - \nabla \cdot (M_i \nabla u_e)
            - I_{\mathrm{stim}} &= 0 \\
        \nabla \cdot (M_i \nabla v) + \nabla \cdot ((M_i + M_e) \nabla u_e) &= 0

    for the transmembrane potential :math:`v` and the extracellular potential
    :math:`u_e`, with insulating boundary conditions on both.

    The two fields live in separate Lagrange spaces and the system is assembled blocked
    and solved monolithically. Keeping ``v`` in a space of its own -- rather than making
    the pair one mixed element -- is what lets the ODE solvers consume it unchanged.

    :math:`u_e` is only determined up to a constant. By default it is normalised to zero
    mean with a Lagrange multiplier in a real-valued space; supplying ``bcs`` that
    constrain :math:`u_e` grounds it instead and the multiplier is dropped, since the two
    together would over-determine the problem.

    Parameters
    ----------
    time : dolfinx.fem.Constant
        The current time
    mesh : dolfinx.mesh.Mesh
        The mesh
    M_i : ufl.Coefficient | float
        The intracellular conductivity tensor
    M_e : ufl.Coefficient | float
        The extracellular conductivity tensor
    I_s : Stimulus | Sequence[Stimulus] | ufl.core.expr.Expr, optional
        The stimulus, by default None
    params : dict, optional
        Parameters for the model, by default None
    C_m : float, optional
        The membrane capacitance, by default 1.0
    dx : ufl.Measure, optional
        The measure for the spatial domain, by default None

    """

    def __init__(
        self,
        time: dolfinx.fem.Constant,
        mesh: dolfinx.mesh.Mesh,
        M_i: ufl.Coefficient | float,
        M_e: ufl.Coefficient | float,
        I_s: Stimulus | Sequence[Stimulus] | ufl.core.expr.Expr | None = None,
        params: dict[str, Any] | None = None,
        C_m: float = 1.0,
        dx: ufl.Measure | None = None,
        **kwargs,
    ) -> None:
        self._M_i = M_i
        self._M_e = M_e
        self.C_m = dolfinx.fem.Constant(mesh, C_m)
        # The extracellular potential at the previous time is only meaningful once an
        # initial transmembrane potential has arrived, which happens after construction.
        self._previous_u_e_is_stale = True
        super().__init__(mesh=mesh, time=time, params=params, I_s=I_s, dx=dx, **kwargs)

    def _setup_state_space(self) -> None:
        k = self.parameters["degree"]
        family = self.parameters["family"]
        element = basix.ufl.element(family=family, cell=self._mesh.basix_cell(), degree=k)

        self.V = dolfinx.fem.functionspace(self._mesh, element)
        self.V_ue = dolfinx.fem.functionspace(self._mesh, element)

        self.v_ = dolfinx.fem.Function(self.V, name="v_")
        self.ue_ = dolfinx.fem.Function(self.V_ue, name="u_e_")
        self._v = dolfinx.fem.Function(self.V, name="v")
        self._u_e = dolfinx.fem.Function(self.V_ue, name="u_e")

    @property
    def v(self) -> dolfinx.fem.Function:
        """The transmembrane potential."""
        return self._v

    @property
    def u_e(self) -> dolfinx.fem.Function:
        """The extracellular potential."""
        return self._u_e

    @property
    def state(self) -> BidomainState:
        """Both solved-for fields, in the order the blocked system holds them."""
        return BidomainState(v=self._v, u_e=self._u_e)

    def assign_previous(self) -> None:
        self.v_.x.array[:] = self._v.x.array[:]
        self.ue_.x.array[:] = self._u_e.x.array[:]

    @property
    def _u_e_is_grounded_by_bc(self) -> bool:
        space = self.V_ue._cpp_object
        return any(space.contains(cpp_function_space(bc.function_space)) for bc in self.bcs)

    def _setup_solver(self) -> None:
        self._multiplier = None
        if not self._u_e_is_grounded_by_bc:
            # One global degree of freedom carrying the constant that enforces zero mean.
            real = basix.ufl.real_element(self._mesh.basix_cell(), value_shape=())
            self.R = dolfinx.fem.functionspace(self._mesh, real)
            self._multiplier = dolfinx.fem.Function(self.R, name="u_e_mean")

        self._unknowns = [self._v, self._u_e]
        if self._multiplier is not None:
            self._unknowns.append(self._multiplier)

        a, L = self.variational_forms(self._timestep)

        # The forms' declared types also span the single-field shape, and the blocked forms
        # contain ``None`` blocks, which dolfinx's annotation does not admit.
        self._solver = dolfinx.fem.petsc.LinearProblem(
            cast(Any, a),
            cast(Any, L),
            u=self._unknowns,
            bcs=self.bcs,
            kind="mpi",
            form_compiler_options=self.parameters["form_compiler_options"],
            jit_options=self.parameters["jit_options"],
            petsc_options=self.parameters["petsc_options"],
            petsc_options_prefix="beat_bidomain_model_",
        )

    def variational_forms(self, dt: Expr | float) -> tuple[_BilinearForm, _LinearForm]:
        """Blocked theta-rule forms for the coupled system.

        Both rows are written in terms of the midpoint pair. The elliptic row carries no
        time derivative, but it is linear and homogeneous, so imposing it on the midpoint
        is equivalent to imposing it at the new time given that it held at the old one --
        and it is the only choice under which the collapse to the monodomain model at
        equal anisotropy ratio stays exact away from theta = 1.

        Parameters
        ----------
        dt : Expr | float
            The time step

        Returns
        -------
        tuple[_BilinearForm, _LinearForm]
            The blocked bilinear and linear forms

        """
        theta = self.parameters["theta"]
        M_i = self._M_i
        M_total = self._M_i + self._M_e

        v = ufl.TrialFunction(self.V)
        w = ufl.TestFunction(self.V)
        u_e = ufl.TrialFunction(self.V_ue)
        q = ufl.TestFunction(self.V_ue)

        def diffusion(M, trial, test):
            return ufl.inner(M * ufl.grad(trial), ufl.grad(test)) * self.dx

        # The elliptic row has no source, and at theta = 1 no history either, so without
        # this its right-hand side collapses to a form carrying no test function at all.
        zero = dolfinx.fem.Constant(self._mesh, dolfinx.default_scalar_type(0.0))

        a00 = self.C_m * v * w * self.dx + dt * theta * diffusion(M_i, v, w)
        a01 = dt * theta * diffusion(M_i, u_e, w)
        L0 = (
            self.C_m * self.v_ * w * self.dx
            - dt * (1.0 - theta) * diffusion(M_i, self.v_, w)
            - dt * (1.0 - theta) * diffusion(M_i, self.ue_, w)
            + dt * self._G_stim(w)
        )

        a10 = theta * diffusion(M_i, v, q)
        a11 = theta * diffusion(M_total, u_e, q)
        L1 = (
            zero * q * self.dx
            - (1.0 - theta) * diffusion(M_i, self.v_, q)
            - (1.0 - theta) * diffusion(M_total, self.ue_, q)
        )

        if self._multiplier is None:
            return [[a00, a01], [a10, a11]], [L0, L1]

        c = ufl.TrialFunction(self.R)
        r = ufl.TestFunction(self.R)

        return (
            [[a00, a01, None], [a10, a11, c * q * self.dx], [None, u_e * r * self.dx, None]],
            [L0, L1, zero * r * self.dx],
        )

    def _assemble_rhs(self) -> None:
        # The compiled blocked form's declared type also spans the single-field shape.
        a = cast(Any, self._solver.a)
        b = self._solver.b
        with b.localForm() as b_loc:
            b_loc.set(0)
        dolfinx.fem.petsc.assemble_vector(b, self._solver.L)  # type: ignore[arg-type]
        if self.bcs:
            dolfinx.fem.petsc.apply_lifting(
                b,
                a,
                bcs=dolfinx.fem.bcs_by_block(
                    dolfinx.fem.extract_function_spaces(a, 1),  # type: ignore[arg-type]
                    self.bcs,
                ),
            )
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
        if self.bcs:
            dolfinx.fem.petsc.set_bc(
                b,
                dolfinx.fem.bcs_by_block(
                    dolfinx.fem.extract_function_spaces(self._solver.L),  # type: ignore[arg-type]
                    self.bcs,
                ),
            )

    def _solve_linear_system(self) -> None:
        # The blocked solution vector is distributed field by field. Its ghost entries are
        # left alone here: the step brings the state's ghosts up to date immediately
        # afterwards, from the owning ranks.
        x = self._solver.x
        self._solver.solver.solve(self._solver.b, x)
        dolfinx.fem.petsc.assign(x, self._unknowns)  # type: ignore[arg-type]

    def step(self, interval) -> None:
        if self._previous_u_e_is_stale:
            self._solve_for_previous_extracellular_potential()
            self._previous_u_e_is_stale = False
        super().step(interval)

    def _solve_for_previous_extracellular_potential(self) -> None:
        """Bring ``ue_`` into agreement with ``v_`` before the first step.

        The midpoint elliptic row presumes that the elliptic equation already holds at the
        old time. It cannot be established in ``__init__``, because the initial
        transmembrane potential is handed over by the ODE solver, which is wired up after
        the model is built. At theta = 1 the old extracellular potential never enters the
        forms, so there is nothing to establish.
        """
        if self.parameters["theta"] >= 1.0:
            return

        M_total = self._M_i + self._M_e
        u_e = ufl.TrialFunction(self.V_ue)
        q = ufl.TestFunction(self.V_ue)

        def diffusion(M, trial, test):
            return ufl.inner(M * ufl.grad(trial), ufl.grad(test)) * self.dx

        a: list[list[ufl.Form | None]] = [[diffusion(M_total, u_e, q)]]
        L: list[ufl.Form] = [-diffusion(self._M_i, self.v_, q)]
        unknowns = [self.ue_]

        if self._multiplier is not None:
            c = ufl.TrialFunction(self.R)
            r = ufl.TestFunction(self.R)
            zero = dolfinx.fem.Constant(self._mesh, dolfinx.default_scalar_type(0.0))
            a = [[a[0][0], c * q * self.dx], [u_e * r * self.dx, None]]
            L = [L[0], zero * r * self.dx]
            unknowns = [self.ue_, dolfinx.fem.Function(self.R)]

        problem = dolfinx.fem.petsc.LinearProblem(
            cast(Any, a),
            L,
            u=unknowns,
            bcs=self.bcs,
            kind="mpi",
            form_compiler_options=self.parameters["form_compiler_options"],
            jit_options=self.parameters["jit_options"],
            petsc_options=self.parameters["petsc_options"],
            petsc_options_prefix="beat_bidomain_initial_",
        )
        problem.solve()
        self.ue_.x.scatter_forward()
