import logging
from typing import NamedTuple

import dolfinx
import numpy as np
import pint
import ufl

from .units import ureg

logger = logging.getLogger(__name__)


class Stimulus(NamedTuple):
    expr: ufl.core.expr.Expr
    dZ: ufl.Measure
    marker: int | None = None

    @property
    def dz(self):
        return self.dZ(self.marker)

    def assign(self, amp: float):
        self.expr.amplitude = amp


def compute_effective_dim(mesh: dolfinx.mesh.Mesh, subdomain_data: dolfinx.mesh.MeshTags) -> int:
    """
    Compute the effective dimension of the stimulus domain based on the subdomain data and the mesh.
    The effective dimension is the dimension used
    to compute the correct unit of the stimulus.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh object.
    subdomain_data : dolfinx.mesh.MeshTags
        The subdomain data object.

    Returns
    -------
    int
        The effective dimension of the stimulus domain.

    Raises
    ------
    ValueError
        If the mesh topology dimension is not 1, 2 or 3.

    """
    dim = subdomain_data.dim
    # view subdomains of surfaces and line as slices of 3D
    if mesh.topology.dim == 3:
        return dim
    elif mesh.topology.dim == 2:
        return dim + 1
    elif mesh.topology.dim == 1:
        return dim + 2

    raise ValueError("Invalid mesh topology dimension")


def get_dZ(mesh: dolfinx.mesh.Mesh, subdomain_data: dolfinx.mesh.MeshTags) -> ufl.Measure:
    """
    Get the measure for the subdomain data based on the mesh topology dimension.
    The measure is used to define the integral over the subdomain where
    the stimulus is applied.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh object.
    subdomain_data : dolfinx.mesh.MeshTags
        The subdomain data object.

    Returns
    -------
    ufl.Measure
        The measure for the subdomain data.

    Raises
    ------
    ValueError
        If the mesh topology dimension is not 1, 2 or 3,
        or if the subdomain data dimension is not compatible with
        the mesh topology dimension.
    """

    dim = subdomain_data.dim
    # We do not support other cases than facets and cells for now
    # if dim == mesh.topology.dim - 2:
    #     # If the subdomain data has two dimensions lower than the mesh,
    #     # then the only valid case is that the mesh is 3D and the subdomain data is 1D.
    #     if mesh.topology.dim != 3:
    #         raise ValueError("Invalid mesh topology dimension")

    #     return ufl.Measure("dP", domain=mesh, subdomain_data=subdomain_data)

    if dim == mesh.topology.dim - 1:
        # If the subdomain data has one dimension lower than the mesh,
        # then the only valid case is that the mesh is 2D and the subdomain data is 1D,
        # or that the mesh is 3D and the subdomain data is 2D.
        if mesh.topology.dim <= 1:
            raise ValueError("Invalid mesh topology dimension")

        return ufl.Measure("ds", domain=mesh, subdomain_data=subdomain_data)

    elif dim == mesh.topology.dim:
        return ufl.Measure("dx", domain=mesh, subdomain_data=subdomain_data)

    raise ValueError("Invalid subdomain data dimension")


def convert_amplitude(effective_dim: int, amplitude: float | pint.Quantity) -> pint.Quantity:
    """
    Convert the amplitude to the appropriate unit based on the effective dimension.
    The function checks if the amplitude is already a pint.Quantity.
    If it is, it returns it as is. Otherwise, it assumes the amplitude is in the
    appropriate unit based on the effective dimension and converts it to a pint.Quantity.

    Parameters
    ----------
    effective_dim : int
        The effective dimension of the stimulus domain.
    amplitude : float | pint.Quantity
        The amplitude value to be converted.

    Returns
    -------
    pint.Quantity
        The converted amplitude value in the appropriate unit.

    Raises
    ------
    ValueError
        If the effective dimension is negative or greater than 3.
    """
    if isinstance(amplitude, ureg.Quantity):
        return amplitude

    if effective_dim <= 1:
        unit = ureg("uA / cm")
    elif effective_dim == 2:
        unit = ureg("uA / cm**2")
    elif effective_dim == 3:
        unit = ureg("uA / cm**3")
    else:
        raise ValueError(f"Invalid effective dimension {effective_dim}. Must be 0, 1, 2 or 3.")
    logger.debug(f"Assuming amplitude is in {unit}")
    return amplitude * unit


def compute_stimulus_unit(effective_dim: int, mesh_unit: str) -> str:
    """
    Compute the unit of the stimulus based on the effective dimension and mesh unit.

    Parameters
    ----------
    effective_dim : int
        The effective dimension of the stimulus.
    mesh_unit : str
        The unit of the mesh.

    Returns
    -------
    str
        The unit of the stimulus.

    Raises
    ------
    ValueError
        If the effective dimension is negative or greater than 3.

    """
    if effective_dim < 0:
        raise ValueError("Effective dimension must be non-negative")
    if effective_dim > 3:
        raise ValueError("Effective dimension must be less than or equal to 3")

    if effective_dim == 0:
        return ureg("uA")
    else:
        return ureg(f"uA/{mesh_unit}**{effective_dim - 1}")


def convert_chi(chi: float, mesh_unit: str) -> pint.Quantity:
    """
    Convert the surface to volume ratio to the appropriate unit based on the mesh unit.

    Parameters
    ----------
    chi : float
        The surface to volume ratio.
    mesh_unit : str
        The unit of the mesh.

    Returns
    -------
    pint.Quantity
        The converted conductivity value.

    """
    if isinstance(chi, ureg.Quantity):
        return chi

    logger.debug(f"Assuming chi is in {mesh_unit}^-1")
    return chi * ureg(f"{mesh_unit}**-1")


def define_stimulus(
    mesh: dolfinx.mesh.Mesh,
    chi: float | pint.Quantity,
    time: dolfinx.fem.Constant,
    subdomain_data: dolfinx.mesh.MeshTags,
    marker: int,
    mesh_unit: str = "cm",
    duration: float = 2.0,
    amplitude: float = 500.0,
    start: float = 0.0,
) -> Stimulus:
    """
    Define a stimulus for the given mesh and subdomain data.
    The function computes the effective dimension of the stimulus domain,
    converts the amplitude and surface to volume ratio to the appropriate units,
    and defines the stimulus expression based on the time variable.
    The stimulus is defined as a conditional expression that is active
    within the specified duration and start time.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh object.
    chi : float | pint.Quantity
        The surface to volume ratio.
    time : dolfinx.fem.Constant
        The time variable.
    subdomain_data : dolfinx.mesh.MeshTags
        The subdomain data object.
    marker : int
        The marker for the subdomain where the stimulus is applied.
    mesh_unit : str, optional
        Unit of the mesh, by default "cm"
    duration : float, optional
        Duration of the stimulus, by default 2.0
    amplitude : float, optional
        Amplitude of the stimulus, by default 500.0
    start : float, optional
        Start time of the stimulus, by default 0.0

    Returns
    -------
    Stimulus
        The stimulus object containing the expression, measure, and marker.

    Raises
    ------
    ValueError
        If the mesh topology dimension is not 1, 2 or 3,
        or if the subdomain data dimension is not compatible with
        the mesh topology dimension.
        If the effective dimension is negative or greater than 3.
        If the mesh unit is not valid.
    """
    effective_dim = compute_effective_dim(mesh, subdomain_data)
    chi = convert_chi(chi, mesh_unit)
    A = convert_amplitude(effective_dim, amplitude)
    dZ = get_dZ(mesh, subdomain_data)
    unit = compute_stimulus_unit(effective_dim, mesh_unit)
    amp = (A / chi).to(unit).magnitude
    I_s = ufl.conditional(ufl.And(ufl.ge(time, start), ufl.le(time, start + duration)), amp, 0.0)

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
    """
    Generate a UFL expression for a random spatio-temporal activation pattern.

    This function creates a sum of UFL conditionals, where each conditional
    represents a localized stimulus that triggers at a specific spatial point
    and a specific time (defined by a base start time plus a point-specific delay).
    It builds a balanced UFL tree to avoid hitting Python's maximum recursion
    depth during JIT compilation of large numbers of points.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The computational mesh over which the spatial coordinates are defined.
    time : dolfinx.fem.Constant
        The UFL time variable used to evaluate the temporal conditions.
    points : np.ndarray
        An array of shape (N, d) containing the spatial coordinates of the
        N activation points, where d is the geometric dimension.
    delays : np.ndarray
        An array of shape (N,) containing the temporal delay for each
        activation point.
    stim_start : float, optional
        The global start time for the stimulation, by default 0.0.
    stim_duration : float, optional
        The duration that the stimulus remains active at each point,
        by default 2.0.
    stim_amplitude : float, optional
        The amplitude (strength) of the stimulus, by default 1.0.
    tol : float, optional
        The spatial tolerance used to match the continuous spatial coordinate
        X to the discrete activation `points`, by default 1e-12.

    Returns
    -------
    ufl.core.expr.Expr
        A balanced UFL expression representing the total stimulus.

    Raises
    ------
    AssertionError
        If the lengths of the `points` and `delays` arrays do not match.
    """
    assert len(points) == len(delays), "Points and delays must have the same length"
    X = ufl.SpatialCoordinate(mesh)

    terms = []
    for i, point in enumerate(points):
        terms.append(
            ufl.conditional(
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
            ),
        )

    if not terms:
        return ufl.as_ufl(0.0)

    # Tree reduction (balanced sum) to avoid maximum recursion depth during compilation
    while len(terms) > 1:
        if len(terms) % 2 != 0:
            terms = [terms[i] + terms[i + 1] for i in range(0, len(terms) - 1, 2)] + [terms[-1]]
        else:
            terms = [terms[i] + terms[i + 1] for i in range(0, len(terms), 2)]

    return terms[0]
