"""PDE stimuli from ``[[stimulus]]`` entries (marker / box / random_endocardial)."""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from mpi4py import MPI

import dolfinx
import numpy as np
import pint
import ufl

from ..stimulation import (
    Stimulus,
    compute_stimulus_unit,
    convert_chi,
    generate_random_activation,
    get_dZ,
)
from ..units import ureg
from ..utils import interpolation_points
from .config import ConfigError, EPConfig, StimulusConfig, ms
from .geometry import CLIGeometry, get_marker

logger = logging.getLogger(__name__)


@dataclass
class StimulusSet:
    stimuli: list[Stimulus] = field(default_factory=list)
    updates: list[Callable[[], None]] = field(default_factory=list)


def _scaled_amplitude(
    conf: Any,
    ep: EPConfig,
    mesh: dolfinx.mesh.Mesh,
    entity_dim: int,
    mesh_unit: str,
) -> float:
    """Same scaling as beat.stimulation.define_stimulus: amplitude / chi in mesh units."""
    effective_dim = entity_dim + (3 - mesh.topology.dim)
    A = ureg.Quantity(conf.amplitude)
    unit = compute_stimulus_unit(effective_dim, mesh_unit)
    try:
        return float((A / convert_chi(ep.chi, mesh_unit)).to(unit).magnitude)
    except pint.DimensionalityError as e:
        expected = {1: "uA/cm", 2: "uA/cm**2", 3: "uA/cm**3"}[effective_dim]
        raise ConfigError(
            f"stimulus amplitude {conf.amplitude!r} has the wrong dimension for a "
            f"{effective_dim}D stimulus domain; expected e.g. {expected}",
        ) from e


def _pulses(time: dolfinx.fem.Constant, starts: list[float], duration: float, amp: float) -> Any:
    terms = [
        ufl.conditional(ufl.And(ufl.ge(time, s), ufl.le(time, s + duration)), amp, 0.0)
        for s in starts
    ]
    return sum(terms[1:], terms[0])


def _marker(
    conf: Any,
    geo: CLIGeometry,
    ep: EPConfig,
    time: dolfinx.fem.Constant,
    mesh_unit: str,
    t_end: float,
) -> Stimulus:
    value, dim = get_marker(geo, conf.marker)
    tdim = geo.mesh.topology.dim
    tags = geo.ffun if dim == tdim - 1 else geo.cfun if dim == tdim else None
    if tags is None:
        raise ConfigError(f"Marker {conf.marker!r} (dim {dim}) must be a facet or cell marker")
    amp = _scaled_amplitude(conf, ep, geo.mesh, dim, mesh_unit)
    expr = _pulses(time, conf.pulse_starts_ms(t_end), ms(conf.duration), amp)
    return Stimulus(expr=expr, dZ=get_dZ(geo.mesh, tags), marker=value)


def _box(
    conf: Any,
    geo: CLIGeometry,
    ep: EPConfig,
    time: dolfinx.fem.Constant,
    mesh_unit: str,
    t_end: float,
) -> Stimulus:
    mesh = geo.mesh
    gdim, tdim = mesh.geometry.dim, mesh.topology.dim
    if len(conf.min) != gdim:
        raise ConfigError(f"box stimulus needs {gdim} coordinates for a {gdim}D mesh")
    lo, hi = np.asarray(conf.min), np.asarray(conf.max)

    def inside(x: np.ndarray) -> np.ndarray:
        return np.all((x[:gdim].T >= lo - 1e-12) & (x[:gdim].T <= hi + 1e-12), axis=1)

    cells = dolfinx.mesh.locate_entities(mesh, tdim, inside)
    if mesh.comm.allreduce(len(cells), op=MPI.SUM) == 0:
        raise ConfigError(f"box stimulus [{conf.min}, {conf.max}] contains no cells of the mesh")
    tags = dolfinx.mesh.meshtags(
        mesh,
        tdim,
        cells.astype(np.int32),
        np.ones(len(cells), dtype=np.int32),
    )
    amp = _scaled_amplitude(conf, ep, mesh, tdim, mesh_unit)
    expr = _pulses(time, conf.pulse_starts_ms(t_end), ms(conf.duration), amp)
    return Stimulus(expr=expr, dZ=get_dZ(mesh, tags), marker=1)


def _random(
    conf: Any,
    geo: CLIGeometry,
    ep: EPConfig,
    time: dolfinx.fem.Constant,
    mesh_unit: str,
    t_end: float,
    out: StimulusSet,
) -> None:
    mesh = geo.mesh
    fdim = mesh.topology.dim - 1
    if geo.ffun is None:
        raise ConfigError("random_endocardial stimulus requires facet markers")
    facets = np.concatenate([geo.ffun.find(get_marker(geo, m)[0]) for m in conf.markers])
    num_owned = mesh.topology.index_map(fdim).size_local
    facets = facets[facets < num_owned]
    local = dolfinx.mesh.compute_midpoints(mesh, fdim, facets.astype(np.int32))
    all_mid = np.concatenate(mesh.comm.allgather(local))
    if len(all_mid) < conf.num_points:
        raise ConfigError(
            f"random_endocardial: only {len(all_mid)} facets on {conf.markers}, "
            f"fewer than num_points={conf.num_points}",
        )
    # Choose on rank 0 with a fixed seed and broadcast -> identical on any number of ranks.
    if mesh.comm.rank == 0:
        rng = np.random.default_rng(conf.seed)
        idx = rng.choice(len(all_mid), size=conf.num_points, replace=False)
        lo, hi = (ms(d) for d in conf.delay_range)
        payload = (all_mid[idx], rng.uniform(lo, hi, conf.num_points))
    else:
        payload = None
    points, delays = mesh.comm.bcast(payload, root=0)
    amp = _scaled_amplitude(conf, ep, mesh, mesh.topology.dim, mesh_unit)
    exprs = [
        generate_random_activation(
            mesh=mesh,
            time=time,
            points=points[:, : mesh.geometry.dim],
            delays=delays,
            stim_start=s,
            stim_duration=ms(conf.duration),
            stim_amplitude=amp,
            tol=conf.tol,
        )
        for s in conf.pulse_starts_ms(t_end)
    ]
    # As in demos/ukb_atlas.py: interpolate the (large) expression into DG0 each step instead
    # of assembling it inside the PDE form.
    W = dolfinx.fem.functionspace(mesh, ("DG", 0))
    stim = dolfinx.fem.Function(W, name="I_s")
    expression = dolfinx.fem.Expression(sum(exprs[1:], exprs[0]), interpolation_points(W))
    out.updates.append(lambda: stim.interpolate(expression))
    out.stimuli.append(Stimulus(expr=stim, dZ=ufl.dx(domain=mesh), marker=None))


def build_stimuli(
    confs: list[StimulusConfig],
    geo: CLIGeometry,
    ep: EPConfig,
    time: dolfinx.fem.Constant,
    mesh_unit: str,
    t_end_ms: float,
) -> StimulusSet:
    out = StimulusSet()
    for conf in confs:
        if conf.type == "marker":
            out.stimuli.append(_marker(conf, geo, ep, time, mesh_unit, t_end_ms))
        elif conf.type == "box":
            out.stimuli.append(_box(conf, geo, ep, time, mesh_unit, t_end_ms))
        else:
            _random(conf, geo, ep, time, mesh_unit, t_end_ms, out)
    if not out.stimuli:
        logger.warning("No [[stimulus]] configured: the tissue will not be stimulated by the PDE")
    return out
