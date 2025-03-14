import logging
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from petsc4py import PETSc

import dolfinx
import numpy as np
import ufl

logger = logging.getLogger(__name__)


@dataclass
class ECGRecovery:
    v: dolfinx.fem.Function
    sigma_b: float | dolfinx.fem.Constant = 1.0
    C_m: float | dolfinx.fem.Constant = 1.0
    dx: ufl.Measure | None = None
    M: float = 1.0
    petsc_options: dict[str, Any] = field(
        default_factory=lambda: {
            "ksp_type": "cg",
            "pc_type": "sor",
            # "ksp_monitor": None,
            "ksp_rtol": 1.0e-8,
            "ksp_atol": 1.0e-8,
            # "ksp_error_if_not_converged": True,
        },
    )

    def __post_init__(self):
        if self.dx is None:
            self.dx = ufl.dx(domain=self.mesh, metadata={"quadrature_degree": 4})
        self.sol = dolfinx.fem.Function(self.V)

        w = ufl.TestFunction(self.V)
        Im = ufl.TrialFunction(self.V)

        self.sol = dolfinx.fem.Function(self.V)

        self._lhs = -self.C_m * Im * w * self.dx
        self._rhs = ufl.inner(self.M * ufl.grad(self.v), ufl.grad(w)) * self.dx

        self.solver = dolfinx.fem.petsc.LinearProblem(
            self._lhs,
            self._rhs,
            u=self.sol,
            petsc_options=self.petsc_options,
        )
        dolfinx.fem.petsc.assemble_matrix(self.solver.A, self.solver.a)
        self.solver.A.assemble()

    @property
    def V(self) -> dolfinx.fem.FunctionSpace:
        return self.v.function_space

    @property
    def mesh(self) -> dolfinx.mesh.Mesh:
        return self.v.function_space.mesh

    def solve(self):
        logger.debug("Solving ECG recovery")
        with self.solver.b.localForm() as b_loc:
            b_loc.set(0)
        dolfinx.fem.petsc.assemble_vector(self.solver.b, self.solver.L)
        self.solver.b.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )

        self.solver.solver.solve(self.solver.b, self.sol.x.petsc_vec)
        self.sol.x.scatter_forward()

    def eval(self, point) -> dolfinx.fem.forms.Form:
        r = ufl.SpatialCoordinate(self.mesh) - dolfinx.fem.Constant(self.mesh, point)
        dist = ufl.sqrt((r**2))
        return dolfinx.fem.form((1 / (4 * ufl.pi * self.sigma_b)) * (self.sol / dist) * self.dx)


def _check_attr(attr: np.ndarray | None):
    if attr is None:
        raise AttributeError(f"Missing attribute {attr}")


# Taken from https://en.wikipedia.org/wiki/Electrocardiography
class Leads12(NamedTuple):
    RA: np.ndarray
    LA: np.ndarray
    LL: np.ndarray
    RL: np.ndarray | None = None  # Do we really need this?
    V1: np.ndarray | None = None
    V2: np.ndarray | None = None
    V3: np.ndarray | None = None
    V4: np.ndarray | None = None
    V5: np.ndarray | None = None
    V6: np.ndarray | None = None

    @property
    def I(self) -> np.ndarray:
        """Voltage between the (positive) left arm (LA)
        electrode and right arm (RA) electrode"""
        return self.LA - self.RA

    @property
    def II(self) -> np.ndarray:
        """Voltage between the (positive) left leg (LL)
        electrode and the right arm (RA) electrode
        """
        return self.LL - self.RA

    @property
    def III(self) -> np.ndarray:
        """Voltage between the (positive) left leg (LL)
        electrode and the left arm (LA) electrode
        """
        return self.LL - self.LA

    @property
    def Vw(self) -> np.ndarray:
        """Wilson's central terminal"""
        return (1 / 3) * (self.RA + self.LA + self.LL)

    @property
    def aVR(self) -> np.ndarray:
        """Lead augmented vector right (aVR) has the positive
        electrode on the right arm. The negative pole is a
        combination of the left arm electrode and the left leg electrode
        """
        return (3 / 2) * (self.RA - self.Vw)

    @property
    def aVL(self) -> np.ndarray:
        """Lead augmented vector left (aVL) has the positive electrode
        on the left arm. The negative pole is a combination of the right
        arm electrode and the left leg electrode
        """
        return (3 / 2) * (self.LA - self.Vw)

    @property
    def aVF(self) -> np.ndarray:
        """Lead augmented vector foot (aVF) has the positive electrode on the
        left leg. The negative pole is a combination of the right arm
        electrode and the left arm electrode
        """
        return (3 / 2) * (self.LL - self.Vw)

    @property
    def V1_(self) -> np.ndarray:
        _check_attr(self.V1)
        return self.V1 - self.Vw

    @property
    def V2_(self) -> np.ndarray:
        _check_attr(self.V2)
        return self.V2 - self.Vw

    @property
    def V3_(self) -> np.ndarray:
        _check_attr(self.V3)
        return self.V3 - self.Vw

    @property
    def V4_(self) -> np.ndarray:
        _check_attr(self.V4)
        return self.V4 - self.Vw

    @property
    def V5_(self) -> np.ndarray:
        _check_attr(self.V5)
        return self.V5 - self.Vw

    @property
    def V6_(self) -> np.ndarray:
        _check_attr(self.V6)
        return self.V6 - self.Vw
