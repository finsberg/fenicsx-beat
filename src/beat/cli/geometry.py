"""Build meshes, markers, fibers and the conductivity tensor from ``[geometry]``/``[ep]``."""

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mpi4py import MPI

import dolfinx
import numpy as np

from .. import geometry as beat_geometry
from ..conductivities import define_conductivity_tensor, get_harmonic_mean_conductivity
from .config import GENERATED_GEOMETRY_TYPES, ConfigError, EPConfig, FibersConfig, GeometryConfig

logger = logging.getLogger(__name__)

META_FILE = "beat_geometry.json"


@dataclass
class CLIGeometry:
    mesh: dolfinx.mesh.Mesh
    ffun: dolfinx.mesh.MeshTags | None = None
    cfun: dolfinx.mesh.MeshTags | None = None
    markers: dict[str, tuple[int, int]] = field(default_factory=dict)
    f0: Any = None
    s0: Any = None
    n0: Any = None


def get_marker(geo: CLIGeometry, name: str) -> tuple[int, int]:
    if name not in geo.markers:
        raise ConfigError(f"Marker {name!r} not found in geometry markers {sorted(geo.markers)}")
    return geo.markers[name]


def _axis_facet_tags(
    mesh: dolfinx.mesh.Mesh,
    lengths: list[float],
) -> tuple[dolfinx.mesh.MeshTags, dict[str, tuple[int, int]]]:
    """Tag the faces x_i = 0 / x_i = L_i as X0/X1, Y0/Y1, Z0/Z1 (values 1..6)."""
    fdim = mesh.topology.dim - 1
    names: dict[str, tuple[int, int]] = {}
    entities = []
    values = []
    for axis, (letter, length) in enumerate(zip("XYZ", lengths)):
        for side, coord in ((0, 0.0), (1, length)):
            value = 2 * axis + side + 1
            facets = dolfinx.mesh.locate_entities_boundary(
                mesh,
                fdim,
                lambda x, a=axis, c=coord: np.isclose(x[a], c),
            )
            names[f"{letter}{side}"] = (value, fdim)
            entities.append(facets)
            values.append(np.full(len(facets), value, dtype=np.int32))
    ent = np.concatenate(entities).astype(np.int32)
    val = np.concatenate(values)
    order = np.argsort(ent)
    ffun = dolfinx.mesh.meshtags(mesh, fdim, ent[order], val[order])
    return ffun, names


def _geometry_hash(conf: GeometryConfig) -> str:
    # `unit` doesn't affect the generated mesh (only how the CLI later interprets its
    # coordinates), so excluding it avoids needless regeneration; `folder` is the cache
    # location itself and would make the hash location-dependent; `fibers` stays IN the hash
    # since it drives `create_fibers` below.
    blob = json.dumps(conf.model_dump(mode="json", exclude={"folder", "unit"}), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _needs_regeneration(meta: Path, current_hash: str) -> tuple[bool, str | None]:
    """Whether the cached geometry at ``meta`` is stale.

    Called on rank 0 only, before the result is broadcast to the other ranks: must never raise,
    or the other ranks would deadlock waiting on the broadcast. A missing, unreadable or corrupt
    metadata file (e.g. left behind by a job killed mid-write) just means "regenerate".
    """
    if not meta.is_file():
        return True, None
    try:
        cached_hash = json.loads(meta.read_text()).get("hash")
    except (OSError, ValueError) as e:
        # ValueError also catches json.JSONDecodeError and UnicodeDecodeError.
        return True, f"Ignoring unreadable/corrupt {meta} ({e}); regenerating geometry"
    return cached_hash != current_hash, None


def _write_meta_atomically(meta: Path, geometry_type: str, h: str) -> None:
    """Write the cache metadata file atomically.

    Called on rank 0 only, before ``comm.barrier()``: must never raise, or the other ranks
    would deadlock waiting on the barrier. Writes to a temp file and ``os.replace``s it into
    place so a job killed mid-write can never leave a partially-written ``meta`` behind (the
    next run would then see a corrupt file, harmlessly handled by ``_needs_regeneration``).
    """
    try:
        tmp = meta.with_name(f".{meta.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps({"type": geometry_type, "hash": h}, indent=2))
        os.replace(tmp, meta)
    except OSError as e:
        logger.warning(f"Could not write geometry cache metadata {meta}: {e}")


def ensure_generated(conf: GeometryConfig, comm: MPI.Intracomm) -> Path:
    """Generate a cardiac-geometriesx mesh into ``conf.folder`` unless it is up to date."""
    import cardiac_geometries as cg

    folder = Path(conf.folder)
    meta = folder / META_FILE
    h = _geometry_hash(conf)
    need = True
    if comm.rank == 0:
        need, warning = _needs_regeneration(meta, h)
        if warning:
            logger.warning(warning)
    need = comm.bcast(need, root=0)
    if not need:
        logger.info(f"Reusing cached {conf.type} geometry in {folder}")
        return folder

    logger.info(f"Generating {conf.type} geometry in {folder}")
    if comm.rank == 0:
        shutil.rmtree(folder, ignore_errors=True)
    comm.barrier()
    generator = getattr(cg.mesh, conf.type)
    kwargs = conf.generator_kwargs()
    kwargs["create_fibers"] = conf.fibers.type == "from_geometry"
    generator(outdir=folder, comm=comm, **kwargs)
    if comm.rank == 0:
        _write_meta_atomically(meta, conf.type, h)
    comm.barrier()
    return folder


def _from_folder(folder: Path, comm: MPI.Intracomm) -> CLIGeometry:
    import cardiac_geometries as cg

    if not Path(folder).is_dir():
        raise ConfigError(f"Geometry folder {folder} does not exist")
    g = cg.geometry.Geometry.from_folder(comm=comm, folder=folder)
    return CLIGeometry(
        mesh=g.mesh,
        ffun=g.ffun,
        cfun=g.cfun,
        markers=dict(g.markers or {}),
        f0=g.f0,
        s0=g.s0,
        n0=g.n0,
    )


def build_geometry(conf: GeometryConfig, comm: MPI.Intracomm = MPI.COMM_WORLD) -> CLIGeometry:
    if conf.type == "folder":
        return _from_folder(conf.folder, comm)
    if conf.type in GENERATED_GEOMETRY_TYPES:
        return _from_folder(ensure_generated(conf, comm), comm)
    if conf.type == "interval":
        n = max(1, round(conf.length / conf.dx))
        mesh = dolfinx.mesh.create_interval(comm, n, [0.0, conf.length])
        ffun, markers = _axis_facet_tags(mesh, [conf.length])
        return CLIGeometry(mesh=mesh, ffun=ffun, markers=markers)
    if conf.type == "rectangle":
        n_xy = [max(1, round(conf.lx / conf.dx)), max(1, round(conf.ly / conf.dx))]
        mesh = dolfinx.mesh.create_rectangle(comm, [[0.0, 0.0], [conf.lx, conf.ly]], n_xy)
        ffun, markers = _axis_facet_tags(mesh, [conf.lx, conf.ly])
        return CLIGeometry(mesh=mesh, ffun=ffun, markers=markers)
    if conf.type == "box_slab":
        g = beat_geometry.get_3D_slab_geometry(comm, conf.dx, conf.lx, conf.ly, conf.lz)
        ffun, markers = _axis_facet_tags(g.mesh, [conf.lx, conf.ly, conf.lz])
        return CLIGeometry(mesh=g.mesh, ffun=ffun, markers=markers, f0=g.f0, s0=g.s0, n0=g.n0)
    raise ConfigError(f"Unsupported geometry type {conf.type!r}")  # pragma: no cover


def build_fibers(geo: CLIGeometry, fibers: FibersConfig) -> Any:
    if fibers.type == "isotropic":
        return None
    if fibers.type == "from_geometry":
        if geo.f0 is None:
            raise ConfigError(
                "geometry.fibers.type = 'from_geometry' but the geometry has no fiber field; "
                "use type = 'axis' or 'isotropic', or a mesh with fibers",
            )
        return geo.f0
    gdim = geo.mesh.geometry.dim
    axis = "xyz".index(fibers.direction)
    if axis >= gdim:
        raise ConfigError(f"Fiber direction {fibers.direction!r} not valid for a {gdim}D mesh")
    vec = np.zeros(gdim, dtype=dolfinx.default_scalar_type)
    vec[axis] = 1.0
    return dolfinx.fem.Constant(geo.mesh, vec)


def build_conductivity(ep: EPConfig, geo: CLIGeometry, fibers: FibersConfig) -> Any:
    g = ep.conductivity.resolved()
    f0 = build_fibers(geo, fibers)
    if f0 is None:
        s_l, _ = get_harmonic_mean_conductivity(ep.chi, **g)
        return float(s_l)
    return define_conductivity_tensor(chi=ep.chi, f0=f0, **g)
