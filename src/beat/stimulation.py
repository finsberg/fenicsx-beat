from typing import NamedTuple

import dolfinx
import numpy as np
import ufl

from .units import ureg


class Stimulus(NamedTuple):
    expr: ufl.core.expr.Expr
    dZ: ufl.Measure
    marker: int | None = None

    @property
    def dz(self):
        return self.dZ(self.marker)

    def assign(self, amp: float):
        self.expr.amplitude = amp


def define_stimulus(
    mesh: dolfinx.mesh.Mesh,
    chi: float,
    time: dolfinx.fem.Constant,
    subdomain_data: dolfinx.mesh.MeshTags,
    marker: int,
    mesh_unit: str = "cm",
    duration: float = 2.0,
    amplitude: float = 500.0,
    start: float = 0.0,
    PCL: float | dolfinx.fem.Constant = 1000.0,
):
    # breakpoint()
    dim = subdomain_data.dim

    # breakpoint()

    if isinstance(amplitude, ureg.Quantity):
        A = amplitude
    else:
        if dim <= 1:
            A = amplitude * ureg("uA / cm")
        elif dim == 2:
            A = amplitude * ureg("uA / cm**2")
        elif dim == 3:
            A = amplitude * ureg("uA / cm**3")

    if dim == 0:
        unit = "uA"
    else:
        unit = f"uA/{mesh_unit}**{dim - 1}"

    amp = (A / chi).to(unit).magnitude
    I_s = ufl.conditional(ufl.And(ufl.ge(time, start), ufl.le(time, start + duration)), amp, 0.0)
    # I_s = dolfin.Expression(
    #     "std::fmod(time,PCL) >= start "
    #     "? (std::fmod(time,PCL) <= (duration + start) ? amplitude : 0.0)"
    #     " : 0.0",
    #     time=time,
    #     start=start,
    #     duration=duration,
    #     amplitude=amp,
    #     degree=0,
    #     PCL=PCL,
    # )

    if dim == mesh.topology.dim - 2:
        dZ = ufl.Measure("dP", domain=mesh, subdomain_data=subdomain_data)
    elif dim == mesh.topology.dim - 1:
        dZ = ufl.Measure("ds", domain=mesh, subdomain_data=subdomain_data)
    elif dim == mesh.topology.dim:
        dZ = ufl.Measure("dx", domain=mesh, subdomain_data=subdomain_data)
    return Stimulus(dZ=dZ, marker=marker, expr=I_s)


def near(a: ufl.Coefficient, b: ufl.Coefficient, tol: float = 1e-12) -> ufl.core.expr.Expr:
    return ufl.And(ufl.ge(a, b - tol), ufl.le(a, b + tol))


def generate_random_activation(
    mesh: dolfinx.mesh.Mesh,
    time: dolfinx.fem.Constant,
    points: np.ndarray,
    delays: np.ndarray,
    stim_start: float = 0.0,
    stim_duration: float = 2.0,
    stim_amplitude: float = 1.0,
    tol: float = 1e-12,
):
    assert len(points) == len(delays), "Points and delays must have the same length"
    X = ufl.SpatialCoordinate(mesh)

    stim_expr = ufl.as_ufl(0.0)
    X = ufl.SpatialCoordinate(mesh)
    for i, point in enumerate(points):
        stim_expr += ufl.conditional(
            ufl.And(
                ufl.And(
                    ufl.And(near(X[0], point[0], tol=tol), near(X[1], point[1], tol=tol)),
                    ufl.And(
                        near(X[2], point[2], tol=tol),
                        ufl.ge(time, stim_start + delays[i]),
                    ),
                ),
                ufl.le(time, stim_start + stim_duration + delays[i]),
            ),
            stim_amplitude,
            0.0,
        )
    return stim_expr
