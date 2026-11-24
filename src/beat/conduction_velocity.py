"""Measure the conduction velocity of a monodomain model, and calibrate its conductivities

A finite element mesh that is too coarse to resolve the upstroke of the action potential slows the
wave down: the conduction velocity depends on the mesh size and the time step as well as on the
conductivities and the cell model. The usual remedy (e.g. ``tuneCV`` in openCARP, Costa et al.
2013) is to choose the conductivities so that a plane wave on a slab, discretised as the
simulation will be, travels at the measured conduction velocities of the tissue.

:func:`measure_conduction_velocity` runs such a plane wave along or across the fibres of a thin
slab and fits the velocity to the activation times. :func:`calibrate_conductivities` scales the
conductivities in each direction until the velocities match target values.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, cast

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt
import pint

from .activation import ActivationTracker
from .conductivities import define_conductivity_tensor
from .geometry import get_3D_slab_mesh
from .monodomain_model import MonodomainModel
from .monodomain_solver import SplittingSolver
from .odesolver import DolfinODESolver
from .stimulation import define_stimulus
from .units import ureg

logger = logging.getLogger(__name__)

Direction = Literal["longitudinal", "transverse"]
#: The conductivities that each direction scales: the longitudinal velocity depends only on the
#: conductivities along the fibres, the transverse velocity only on those across them
SCALED_KEYS: dict[str, tuple[str, str]] = {
    "longitudinal": ("g_il", "g_el"),
    "transverse": ("g_it", "g_et"),
}


def _magnitude(value: float | pint.Quantity, unit: str) -> float:
    if isinstance(value, pint.Quantity):
        return float(value.to(unit).magnitude)
    return float(value)


def measure_conduction_velocity(
    *,
    fun: Callable[..., npt.NDArray[np.float64]],
    init_states: npt.NDArray[np.float64],
    parameters: npt.NDArray[np.float64],
    v_index: int,
    conductivities: dict[str, Any],
    dx: float | pint.Quantity,
    dt: float = 0.05,
    direction: Direction = "longitudinal",
    C_m: float | pint.Quantity = ureg.Quantity(1.0, "uF/cm**2"),
    amplitude: float | pint.Quantity = ureg.Quantity(50.0, "uA/cm**2"),
    threshold: float = -30.0,
    length: float | pint.Quantity | None = None,
    t_max: float | None = None,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> float:
    """Conduction velocity (mm/ms) of a plane wave along or across the fibres of a slab

    The slab is a box of cubes of side ``dx``, each split into six tetrahedra
    (:func:`beat.geometry.get_3D_slab_mesh`), of length ``length`` and cross-section
    ``2 dx x 2 dx``, with the fibres along its length
    (``"longitudinal"``) or across it (``"transverse"``). The cells within ``max(1 mm, dx)`` of
    one end are stimulated for 2 ms, and the velocity is the slope of a least-squares fit of
    position against activation time (``threshold`` crossing) of the nodes between 40 % and
    90 % of the length.

    Parameters
    ----------
    fun, init_states, parameters, v_index
        The cell model, as for :class:`beat.odesolver.DolfinODESolver` (``init_states`` of one
        cell, shape ``(num_states,)``)
    conductivities
        ``g_il``, ``g_it``, ``g_el``, ``g_et`` and ``chi``, as returned by
        :func:`beat.conductivities.default_conductivities`
    dx
        Mesh size, a float in mm or a pint quantity. Use the mesh size of the simulation.
    dt
        Time step (ms). Use the time step of the simulation.
    direction : {"longitudinal", "transverse"}
        Whether the wave travels along or across the fibres
    C_m
        Membrane capacitance, a float in uF/cm² or a pint quantity
    amplitude
        Stimulus current per membrane area, a float in uA/cm² or a pint quantity
    threshold
        The potential (mV) whose upward crossing is the activation time
    length
        Length of the slab, a float in mm or a pint quantity; by default ``max(10 mm, 10 dx)``
    t_max
        Simulated time at most (ms); by default the time a wave at 0.05 mm/ms needs to cross
        the slab. The simulation also stops, as a conduction block, when no node has been
        activated for as long as such a wave needs to cross two elements (at least 10 ms).
    comm
        The MPI communicator

    Returns
    -------
    float
        The conduction velocity in mm/ms, or 0 if the wave did not cross the measured part of
        the slab within ``t_max`` (conduction block)
    """
    if direction not in SCALED_KEYS:
        raise ValueError(f"direction must be 'longitudinal' or 'transverse', got {direction!r}")
    h = _magnitude(dx, "mm")
    L = max(10.0, 10.0 * h) if length is None else _magnitude(length, "mm")
    t_end = L / 0.05 if t_max is None else t_max
    domain = get_3D_slab_mesh(comm, h, L, 2.0 * h, 2.0 * h)
    fibre = [1.0, 0.0, 0.0] if direction == "longitudinal" else [0.0, 1.0, 0.0]
    f0 = dolfinx.fem.Constant(domain, np.array(fibre))
    M = define_conductivity_tensor(f0=f0, **conductivities)

    chi = conductivities["chi"]
    if not isinstance(chi, pint.Quantity):
        chi = ureg.Quantity(chi, "1/cm")
    if not isinstance(amplitude, pint.Quantity):
        amplitude = ureg.Quantity(amplitude, "uA/cm**2")
    if not isinstance(C_m, pint.Quantity):
        C_m = ureg.Quantity(C_m, "uF/cm**2")
    x_stim = max(1.0, h) + 1e-8
    cells = dolfinx.mesh.locate_entities(domain, 3, lambda x: x[0] <= x_stim)
    tags = dolfinx.mesh.meshtags(domain, 3, cells, np.full(len(cells), 1, dtype=np.int32))
    time = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0.0))
    I_s = define_stimulus(
        mesh=domain,
        chi=chi,
        time=time,
        subdomain_data=tags,
        marker=1,
        mesh_unit="mm",
        amplitude=(amplitude * chi).to("uA/cm**3"),
    )
    pde = MonodomainModel(
        time=time,
        mesh=domain,
        M=M,
        I_s=I_s,
        C_m=float(C_m.to("uF/mm**2").magnitude),
        dx=I_s.dZ,
    )
    V = dolfinx.fem.functionspace(domain, ("P", 1))
    num_points = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
    y0 = np.asarray(init_states, dtype=np.float64)
    ode = DolfinODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=pde.state,
        fun=fun,
        init_states=np.repeat(y0[:, None], num_points, axis=1),
        parameters=parameters,
        num_states=len(y0),
        v_index=v_index,
    )
    solver = SplittingSolver(pde=pde, ode=ode)
    tracker = ActivationTracker(solver.v, method="threshold", threshold=threshold)

    n_owned = V.dofmap.index_map.size_local
    x = V.tabulate_dof_coordinates()[:n_owned, 0]
    measured = (x >= 0.4 * L) & (x <= 0.9 * L)
    check_every = max(1, round(1.0 / dt))
    stall = max(10.0, 2.0 * h / 0.05)
    num_activated, last_change = 0, 0.0
    t = 0.0
    step = 0
    done = False
    while t < t_end - 1e-9:
        solver.step((t, t + dt))
        step += 1
        t = step * dt
        tracker.update(t)
        if step % check_every == 0:
            missing = int(np.isnan(tracker.t_a.x.array[:n_owned][measured]).sum())
            if comm.allreduce(missing, op=MPI.SUM) == 0:
                done = True
                break
            activated = comm.allreduce(
                int(np.isfinite(tracker.t_a.x.array[:n_owned]).sum()),
                op=MPI.SUM,
            )
            if activated > num_activated:
                num_activated, last_change = activated, t
            elif t - last_change > stall:
                logger.info(f"Conduction block: no node activated in the last {stall:g} ms")
                break
    if not done:
        return 0.0
    xs = np.concatenate(comm.allgather(x[measured]))
    ts = np.concatenate(comm.allgather(tracker.t_a.x.array[:n_owned][measured]))
    return float(np.polyfit(ts, xs, 1)[0])


def scale_conductivities(
    conductivities: dict[str, Any],
    longitudinal: float = 1.0,
    transverse: float = 1.0,
) -> dict[str, Any]:
    """A copy of ``conductivities`` with the intra- and extracellular conductivities along the
    fibres multiplied by ``longitudinal`` and those across them by ``transverse``

    Scaling the intra- and extracellular conductivity of a direction by the same factor scales
    their harmonic mean, the monodomain conductivity, by that factor too.
    """
    scaled = dict(conductivities)
    for direction, factor in (("longitudinal", longitudinal), ("transverse", transverse)):
        for key in SCALED_KEYS[direction]:
            scaled[key] = factor * conductivities[key]
    return scaled


@dataclass
class CalibrationResult:
    """The result of :func:`calibrate_conductivities`

    Attributes
    ----------
    conductivities
        The calibrated conductivities, to pass to
        :func:`beat.conductivities.define_conductivity_tensor`
    scales
        The factors the conductivities along (``"longitudinal"``) and across
        (``"transverse"``) the fibres were multiplied by
    velocities
        The conduction velocities (mm/ms) with the calibrated conductivities
    history
        Every iteration: the scales and the velocities they gave
    """

    conductivities: dict[str, Any]
    scales: dict[str, float]
    velocities: dict[str, float]
    history: list[tuple[dict[str, float], dict[str, float]]] = field(default_factory=list)


def calibrate_conductivities(
    target_longitudinal: float | pint.Quantity,
    target_transverse: float | pint.Quantity,
    *,
    conductivities: dict[str, Any],
    rtol: float = 0.02,
    max_iterations: int = 10,
    **kwargs,
) -> CalibrationResult:
    """Scale the conductivities so that plane waves along and across the fibres travel at the
    target velocities

    Each iteration measures both velocities with :func:`measure_conduction_velocity` and updates
    the scale ``k`` of each direction's conductivities by ``k <- k (c_target / c)^(1/p)``, until
    both velocities ``c`` are within ``rtol`` of their targets. With a resolved upstroke the
    velocity is proportional to the square root of the conductivity, so ``p`` starts at 1/2;
    after two iterations it is the slope of ``log c`` against ``log k`` between the last two,
    which is smaller on a coarse mesh. Both directions are measured with both scales applied:
    on a coarse mesh of tetrahedra a plane wave along one direction also feels the
    conductivity across it.

    Parameters
    ----------
    target_longitudinal, target_transverse
        Target velocities along and across the fibres, floats in mm/ms or pint quantities
    conductivities
        The conductivities to scale, as returned by
        :func:`beat.conductivities.default_conductivities`; they are also the first guess
    rtol
        Relative tolerance on the velocities, by default 2 %
    max_iterations
        Iterations at most (two simulations each); if the tolerance is not met the iteration
        with the smallest relative error is used and a ``RuntimeWarning`` issued
    kwargs
        Passed to :func:`measure_conduction_velocity`: the cell model (``fun``,
        ``init_states``, ``parameters``, ``v_index``), the mesh size ``dx`` and the time step
        ``dt`` (both those of the simulation), and the others

    Returns
    -------
    CalibrationResult
    """
    targets: dict[str, float] = {
        "longitudinal": _magnitude(target_longitudinal, "mm/ms"),
        "transverse": _magnitude(target_transverse, "mm/ms"),
    }
    scales = {direction: 1.0 for direction in targets}
    previous: dict[str, tuple[float, float]] = {}
    history: list[tuple[dict[str, float], dict[str, float]]] = []

    def error(velocities: dict[str, float]) -> float:
        return max(abs(velocities[d] / targets[d] - 1.0) for d in targets)

    for _ in range(max_iterations):
        scaled = scale_conductivities(conductivities, **scales)
        velocities = {
            d: measure_conduction_velocity(
                conductivities=scaled,
                direction=cast(Direction, d),
                **kwargs,
            )
            for d in targets
        }
        history.append((dict(scales), velocities))
        logger.info(
            "scales "
            + ", ".join(f"{d} {scales[d]:.4g}" for d in targets)
            + " give "
            + ", ".join(f"{d} {velocities[d]:.4g} mm/ms" for d in targets),
        )
        if error(velocities) <= rtol:
            break
        for d, target in targets.items():
            k, c = scales[d], velocities[d]
            if c <= 0:
                scales[d] = 4.0 * k  # conduction block: more conductivity
                continue
            p = 0.5
            if d in previous:
                kp, cp = previous[d]
                if cp > 0 and kp != k and cp != c:
                    p = float(np.clip(np.log(c / cp) / np.log(k / kp), 0.1, 1.0))
            previous[d] = (k, c)
            scales[d] = k * (target / c) ** (1.0 / p)
    else:
        warnings.warn(
            f"The velocities did not reach their targets within {rtol:.0%} in "
            f"{max_iterations} iterations; using the closest",
            RuntimeWarning,
            stacklevel=2,
        )
    best_scales, best_velocities = min(history, key=lambda h: error(h[1]))
    return CalibrationResult(
        conductivities=scale_conductivities(conductivities, **best_scales),
        scales=best_scales,
        velocities=best_velocities,
        history=history,
    )
