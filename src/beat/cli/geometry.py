"""Build meshes, markers, fibers and the conductivity tensor from ``[geometry]``/``[ep]``."""

import hashlib
import json
import logging
import os
import shutil
import uuid
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
    except (OSError, ValueError, AttributeError) as e:
        # ValueError also catches json.JSONDecodeError and UnicodeDecodeError.
        return True, f"Ignoring unreadable/corrupt {meta} ({e}); regenerating geometry"
    return cached_hash != current_hash, None


def cache_folder(conf: GeometryConfig) -> Path:
    """The folder a generated geometry is cached in: ``geometry.folder/<hash16>/``.

    Keyed by the geometry hash so that different geometry parameters (e.g. an array-job sweep
    over ``geometry.dx``) never share, overwrite or delete each other's mesh, and so that
    ``geometry.folder`` itself -- possibly the config's own directory -- is never deleted.
    """
    return Path(conf.folder) / _geometry_hash(conf)[:16]


def _install_generated(tmp: Path, target: Path, geometry_type: str, h: str) -> None:
    """Move the freshly generated ``tmp`` folder into place as ``target`` (rank 0 only).

    ``target`` is a hash-keyed subfolder beat itself owns, so a stale/corrupt one may be
    replaced. If a concurrent job already installed a complete entry for the same hash, it is
    reused and ``tmp`` discarded. The metadata file is written into ``tmp`` *before* the
    (atomic) rename, so ``target`` only ever appears complete.
    """
    (tmp / META_FILE).write_text(json.dumps({"type": geometry_type, "hash": h}, indent=2))
    for _ in range(3):
        if not _needs_regeneration(target / META_FILE, h)[0]:
            shutil.rmtree(tmp, ignore_errors=True)  # another job finished first: reuse its
            return
        if target.exists():
            # Stale or partial entry beat created: move it aside first (atomic), then delete.
            trash = target.with_name(f".trash-{target.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}")
            try:
                os.rename(target, trash)
            except FileNotFoundError:
                pass  # another job removed/replaced it meanwhile; re-check
            else:
                shutil.rmtree(trash, ignore_errors=True)
        try:
            os.rename(tmp, target)
            return
        except OSError:
            continue  # another job installed ``target`` in between; re-check
    raise OSError(f"Could not install the generated geometry into {target}")


def ensure_generated(conf: GeometryConfig, comm: MPI.Intracomm) -> Path:
    """Generate a cardiac-geometriesx mesh into :func:`cache_folder` unless it is up to date.

    Generation writes into a private temporary sibling folder, which is then atomically
    renamed into place, so concurrent jobs generating the same geometry never see (or delete)
    each other's half-written mesh. Nothing else in ``geometry.folder`` is ever touched.
    """
    import cardiac_geometries as cg

    from .runner import _on_rank0

    h = _geometry_hash(conf)
    target = cache_folder(conf)
    meta = target / META_FILE
    need = True
    if comm.rank == 0:
        need, warning = _needs_regeneration(meta, h)
        if warning:
            logger.warning(warning)
    need = comm.bcast(need, root=0)
    if not need:
        logger.info(f"Reusing cached {conf.type} geometry in {target}")
        return target

    logger.info(f"Generating {conf.type} geometry in {target}")
    name = f".tmp-{target.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}" if comm.rank == 0 else None
    tmp: Path = target.with_name(comm.bcast(name, root=0))
    _on_rank0(comm, OSError, lambda: target.parent.mkdir(parents=True, exist_ok=True))
    generator = getattr(cg.mesh, conf.type)
    kwargs = conf.generator_kwargs()
    kwargs["create_fibers"] = conf.fibers.type == "from_geometry"
    try:
        generator(outdir=tmp, comm=comm, **kwargs)
    except BaseException as e:
        # No collective here (ranks may fail independently): best-effort cleanup only.
        if comm.rank == 0:
            shutil.rmtree(tmp, ignore_errors=True)
        if isinstance(e, ImportError):
            # e.g. BiV/UKB fibers need fenicsx-ldrb; every rank hits the same import.
            raise ConfigError(
                f"Generating a {conf.type!r} geometry needs an optional package: {e}",
            ) from e
        raise
    _on_rank0(comm, OSError, lambda: _install_generated(tmp, target, conf.type, h))
    return target


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
