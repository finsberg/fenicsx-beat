from dataclasses import dataclass
from typing import NamedTuple, Sequence

import dolfinx
import numpy as np
import ufl


@dataclass
class ECGRecovery:
    v: dolfinx.fem.Function
    mesh: dolfinx.mesh.Mesh
    sigma_b: dolfinx.fem.Constant | float
    dx: ufl.Measure | None = None
    point: Sequence[float] | None = None
    r: dolfinx.fem.Function | None = None

    def __post_init__(self):
        if isinstance(self.sigma_b, float):
            self._sigma_b = dolfinx.fem.Constant(self.mesh, self.sigma_b)
        else:
            self._sigma_b = self.sigma_b

        if self.dx is None:
            # breakpoint()
            self.dx = ufl.dx(domain=self.mesh, metadata={"quadrature_degree": 4})

        if self.r is None:
            assert self.point is not None, "Both r and point cannot be None"
            r = ufl.SpatialCoordinate(self.mesh) - dolfinx.fem.Constant(self.mesh, self.point)
            self._r = r
        else:
            self._r = self.r

        r3 = ufl.sqrt((self._r**2)) ** 3
        # https://carp.medunigraz.at/knowledge-base/tissue-scale-ep.html
        # return (1 / (4 * ufl.pi * self.sigma_b)) * dolfinx.fem.assemble(
        #     (ufl.inner(ufl.grad(self.v), self.r) / r3) * self.dx
        # )
        self.form = dolfinx.fem.form(
            (1 / (4 * ufl.pi * self.sigma_b))
            * (ufl.inner(ufl.grad(self.v), self._r) / r3)
            * self.dx,
        )

    def assemble(self):
        return dolfinx.fem.assemble_scalar(self.form)


# def ecg_recovery(
#     *,
#     v: dolfin.Function,
#     mesh: dolfin.Mesh,
#     sigma_b: dolfin.Constant,
#     dx: dolfin.Measure | None = None,
#     point: np.ndarray | None = None,
#     r: dolfin.Function | None = None,
# ):
#     if dx is None:
#         # breakpoint()
#         dx = ufl.dx(domain=mesh, metadata={"quadrature_degree": 4})
#     if r is None:
#         r = ufl.SpatialCoordinate(mesh) - dolfin.Constant(point)

#     r3 = ufl.sqrt((r**2)) ** 3
#     # https://carp.medunigraz.at/knowledge-base/tissue-scale-ep.html
#     return (1 / (4 * ufl.pi * sigma_b)) * dolfinx.fem.assemble(
#         (ufl.inner(ufl.grad(v), r) / r3) * dx
#     )


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
