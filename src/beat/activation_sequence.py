"""Endocardial activation sequences: stimuli that mimic the Purkinje system

The Purkinje network activates the ventricles from a few early sites on the endocardium.
Durrer et al. (1970) mapped them in isolated human hearts: three areas in the LV (the anterior
paraseptal wall, the centre of the left septal surface and the posterior paraseptal wall, between
0 and 5 ms) and, about 5 ms later, two in the RV (the septal surface and the free wall near the
insertion of the anterior papillary muscle). From there the activation spreads over the
endocardium much faster than through the wall, and the base, which has few Purkinje fibres, is
activated by the tissue.

:class:`EndocardialActivation` builds such a sequence on a bi-ventricular mesh:

1. a :class:`HeartFrame` (apex, long axis, septal and anterior directions) from the marked
   surfaces;
2. the root sites (:class:`RootSite`, by default :data:`DURRER_ROOT_SITES`) placed on the
   endocardium by height along the long axis and angle around the LV axis;
3. the time at which an endocardial wave, starting at the root sites after their delays and
   travelling over the endocardial surface at a Purkinje-like speed, reaches each endocardial
   vertex, from the eikonal equation on the surface (:class:`beat.eikonal.EikonalSolver`);
4. a stimulus (:meth:`EndocardialActivation.stimulus`) that switches on every cell with a facet on
   the endocardium below ``max_height`` when this wave reaches it.

Reference: D. Durrer, R. T. van Dam, G. E. Freud, M. J. Janse, F. L. Meijler and
R. C. Arzbaecher, Total excitation of the isolated human heart, Circulation 41 (1970) 899-912.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import numpy as np
import numpy.typing as npt
import pint
import ufl

from .eikonal import FAR, EikonalSolver, Source
from .stimulation import Stimulus
from .units import ureg

logger = logging.getLogger(__name__)

#: Names of the facet markers taken as the base (valve planes) by
#: :meth:`EndocardialActivation.from_geometry`, in the order they are looked for
BASE_MARKER_NAMES = ("BASE", "MV", "AV", "TV", "PV")


@dataclass(frozen=True)
class RootSite:
    """An early activation site on the endocardium

    Parameters
    ----------
    surface : {"LV", "RV"}
        The endocardial surface the site is on
    height : float
        Height above the apex, as a fraction of the apex-base distance (0 at the apex, 1 at the
        base)
    position : float | {"septal", "free"}
        A float is the angle (degrees) around the LV axis, measured from the septum (0) towards
        the anterior wall (90); the posterior wall is at -90. ``"septal"`` and ``"free"`` take the
        vertex of the surface at that height closest to or furthest from the LV axis (the septal
        surface or the free wall of the RV).
    delay : float
        Time (ms) at which the site activates, before ``delay_scale``
    """

    surface: Literal["LV", "RV"]
    height: float
    position: float | Literal["septal", "free"]
    delay: float = 0.0

    def __post_init__(self) -> None:
        if self.surface not in ("LV", "RV"):
            raise ValueError(f"surface must be 'LV' or 'RV', got {self.surface!r}")
        if isinstance(self.position, str) and self.position not in ("septal", "free"):
            raise ValueError(
                f"position must be an angle, 'septal' or 'free', got {self.position!r}"
            )
        if not 0.0 <= self.height <= 1.0:
            raise ValueError(f"height must be between 0 and 1, got {self.height}")


#: The root sites of Durrer et al. (1970): three LV sites at 0 ms and two RV sites 5 ms later
#: (delays of a human heart; scale them with ``delay_scale`` for a smaller or larger heart)
DURRER_ROOT_SITES: dict[str, RootSite] = {
    "LV anterior paraseptal": RootSite("LV", 0.70, 60.0, 0.0),
    "LV mid-septal": RootSite("LV", 0.50, 0.0, 0.0),
    "LV posterior paraseptal": RootSite("LV", 0.33, -60.0, 0.0),
    "RV septal": RootSite("RV", 0.50, "septal", 5.0),
    "RV free wall": RootSite("RV", 0.40, "free", 5.0),
}


def _as_values(values: int | Sequence[int]) -> list[int]:
    if isinstance(values, (int, np.integer)):
        return [int(values)]
    return [int(v) for v in values]


def _vertex_to_dof(V: dolfinx.fem.FunctionSpace) -> npt.NDArray[np.int32]:
    """Local P1 dof of every local vertex (P1 dofs follow the reference vertex order of a cell)"""
    mesh = V.mesh
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)
    dofs = V.dofmap.list
    imap = mesh.topology.index_map(0)
    v2d = np.full(imap.size_local + imap.num_ghosts, -1, dtype=np.int32)
    v2d[c2v.array.reshape(dofs.shape)] = dofs
    return v2d


def _surface_dofs(
    V: dolfinx.fem.FunctionSpace,
    facet_tags: dolfinx.mesh.MeshTags,
    values: Sequence[int],
) -> npt.NDArray[np.int32]:
    """Owned P1 dofs on the facets tagged with any of ``values``"""
    mesh = V.mesh
    mesh.topology.create_connectivity(facet_tags.dim, mesh.topology.dim)
    facets = np.concatenate([facet_tags.find(v) for v in values]).astype(np.int32)
    dofs = dolfinx.fem.locate_dofs_topological(V, facet_tags.dim, facets)
    return dofs[dofs < V.dofmap.index_map.size_local]


def _global_mean(comm: MPI.Comm, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    total = comm.allreduce(x.sum(axis=0), op=MPI.SUM)
    count = comm.allreduce(len(x), op=MPI.SUM)
    if count == 0:
        raise ValueError("Empty surface: no vertex has the given marker")
    return np.asarray(total) / count


def _global_plane_normal(
    comm: MPI.Comm, x: npt.NDArray[np.float64], center: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Unit normal of the least-squares plane through the points ``x`` (on all ranks) and
    ``center``, their global mean"""
    r = x - center
    cov = np.asarray(comm.allreduce(r.T @ r, op=MPI.SUM))
    return np.linalg.eigh(cov)[1][:, 0]


def _global_argmin(comm: MPI.Comm, score: npt.NDArray[np.float64], x: npt.NDArray[np.float64]):
    """The point of ``x`` with the smallest ``score`` over all ranks (lowest rank on ties)"""
    best = (np.inf, None) if len(score) == 0 else (float(score.min()), x[np.argmin(score)])
    candidates = comm.allgather(best)
    k = min(range(len(candidates)), key=lambda i: candidates[i][0])
    if candidates[k][1] is None:
        raise ValueError("No candidate vertex on any rank")
    return np.asarray(candidates[k][1], dtype=np.float64)


@dataclass(frozen=True)
class HeartFrame:
    """A coordinate system of a bi-ventricular geometry

    ``long_axis`` points from the apex towards the base: by default along the normal of the base
    plane, so that the base is at height 1 all around (see ``axis`` in :meth:`from_facet_tags`).
    ``septal`` points from the LV towards the RV (perpendicular to the long axis) and
    ``anterior = long_axis x septal``, flipped if needed to point towards ``anterior_point`` (for
    instance the pulmonary valve). The height of a point is its distance from the apex along the
    long axis, divided by that of the centre of the base.
    """

    apex: npt.NDArray[np.float64]
    base_center: npt.NDArray[np.float64]
    lv_center: npt.NDArray[np.float64]
    rv_center: npt.NDArray[np.float64]
    long_axis: npt.NDArray[np.float64]
    septal: npt.NDArray[np.float64]
    anterior: npt.NDArray[np.float64]
    length: float

    @classmethod
    def from_facet_tags(
        cls,
        mesh: dolfinx.mesh.Mesh,
        facet_tags: dolfinx.mesh.MeshTags,
        *,
        lv: int | Sequence[int],
        rv: int | Sequence[int],
        epi: int | Sequence[int],
        base: int | Sequence[int],
        anterior: int | Sequence[int] | None = None,
        anterior_point: npt.ArrayLike | None = None,
        axis: Literal["base_normal", "apex_to_base"] = "base_normal",
    ) -> HeartFrame:
        """Build the frame from the marked surfaces (collective)

        Parameters
        ----------
        mesh, facet_tags
            The mesh and its facet tags
        lv, rv, epi, base
            Marker values of the LV and RV endocardium, the epicardium and the base (all valve
            planes)
        anterior
            Marker value(s) of an anterior surface (the pulmonary valve) that fixes the sign of
            the anterior direction
        anterior_point
            A point on the anterior side, used instead of ``anterior``. Without either the sign
            of the anterior direction is arbitrary (``long_axis x septal``).
        axis : {"base_normal", "apex_to_base"}
            The direction of the long axis. ``"base_normal"`` (the default) is the normal of the
            plane fitted to the base. ``"apex_to_base"`` points from the apex to the centre of the
            base; when the RV pulls that centre to one side the axis is tilted, and heights on the
            base then differ between the ventricles (in the bi-ventricular ellipsoid the base is at
            height 0.9 on the LV free wall and 1.1 on the RV free wall).
        """
        if axis not in ("base_normal", "apex_to_base"):
            raise ValueError(f"axis must be 'base_normal' or 'apex_to_base', got {axis!r}")
        comm = mesh.comm
        V = dolfinx.fem.functionspace(mesh, ("P", 1))
        x = V.tabulate_dof_coordinates()

        def coords(values):
            return x[_surface_dofs(V, facet_tags, _as_values(values))]

        base_center = _global_mean(comm, coords(base))
        x_epi = coords(epi)
        apex = _global_argmin(comm, -np.linalg.norm(x_epi - base_center, axis=1), x_epi)
        ez = (base_center - apex) / np.linalg.norm(base_center - apex)
        if axis == "base_normal":
            normal = _global_plane_normal(comm, coords(base), base_center)
            ez = normal if normal @ ez > 0 else -normal
        length = float((base_center - apex) @ ez)
        lv_center = _global_mean(comm, coords(lv))
        rv_center = _global_mean(comm, coords(rv))
        d = rv_center - lv_center
        d = d - (d @ ez) * ez
        e_septal = d / np.linalg.norm(d)
        e_anterior = np.cross(ez, e_septal)
        if anterior is not None and anterior_point is None:
            anterior_point = _global_mean(comm, coords(anterior))
        if anterior_point is not None:
            if (np.asarray(anterior_point, dtype=float) - lv_center) @ e_anterior < 0:
                e_anterior = -e_anterior
        return cls(
            apex=apex,
            base_center=base_center,
            lv_center=lv_center,
            rv_center=rv_center,
            long_axis=ez,
            septal=e_septal,
            anterior=e_anterior,
            length=length,
        )

    def coordinates(
        self,
        x: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Height fraction, angle (degrees, 0 = septum, 90 = anterior) and distance from the LV
        axis of the points ``x`` (shape ``(n, 3)``)"""
        r = np.atleast_2d(np.asarray(x, dtype=float)) - self.lv_center
        z = (r + self.lv_center - self.apex) @ self.long_axis / self.length
        angle = np.degrees(np.arctan2(r @ self.anterior, r @ self.septal))
        r_normal = r - np.outer(r @ self.long_axis, self.long_axis)
        return z, angle, np.linalg.norm(r_normal, axis=1)

    def point(self, height: float, angle: float, radius: float) -> npt.NDArray[np.float64]:
        """The point at ``height`` (fraction), ``angle`` (degrees) and ``radius`` from the LV
        axis (the inverse of :meth:`coordinates`), e.g. to place electrodes"""
        on_axis = self.apex + height * self.length * self.long_axis
        on_axis = (
            on_axis
            + (self.lv_center - on_axis)
            - ((self.lv_center - on_axis) @ self.long_axis) * self.long_axis
        )
        a = np.radians(angle)
        return on_axis + radius * (np.cos(a) * self.septal + np.sin(a) * self.anterior)


def place_root_site(
    frame: HeartFrame,
    comm: MPI.Comm,
    x: npt.NDArray[np.float64],
    site: RootSite,
    tol: float = 0.03,
    angle_window: float = 30.0,
) -> npt.NDArray[np.float64]:
    """The point of ``x`` (the owned vertices of the site's surface on this rank) that best
    matches ``site``

    Among the vertices within ``tol`` (doubled until there are some) of its height: for an angle,
    the vertex closest to that angle; for ``"septal"``/``"free"``, the vertices within
    ``angle_window`` degrees (doubled until there are some) of the septal direction are split by
    their distance from the LV axis into the septal surface (the nearer half of the range) and
    the free wall (the further half), and the vertex of that side closest to the septal direction
    is taken. This keeps the septal site off the junctions of the RV with the LV, which are about
    as close to the LV axis as the septal surface itself.
    """

    def first_nonempty(select, width, limit):
        while True:
            mask = select(width)
            if comm.allreduce(int(mask.sum()), op=MPI.SUM) > 0:
                return mask
            if width > limit:
                raise ValueError(f"No vertex found for root site {site}")
            width *= 2

    z, angle, radius = frame.coordinates(np.reshape(x, (-1, 3)))
    near = first_nonempty(lambda w: np.abs(z - site.height) < w, tol, 1.0)
    if isinstance(site.position, str):
        near = first_nonempty(lambda w: near & (np.abs(angle) <= w), angle_window, 180.0)
        r_min = comm.allreduce(float(radius[near].min(initial=np.inf)), op=MPI.MIN)
        r_max = comm.allreduce(float(radius[near].max(initial=-np.inf)), op=MPI.MAX)
        r_mid = 0.5 * (r_min + r_max)
        side = radius <= r_mid if site.position == "septal" else radius >= r_mid
        near = near & side
        score = np.abs(angle[near])
    else:
        score = np.abs((angle[near] - float(site.position) + 180.0) % 360.0 - 180.0)
    return _global_argmin(comm, score, x[near])


def _speed(value: Any, mesh_unit: str) -> float:
    if isinstance(value, pint.Quantity):
        value = value.to(f"{mesh_unit}/ms").magnitude
    value = float(value)
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"The conduction velocity must be positive, got {value}")
    return value


def _amplitude(
    amplitude: float | pint.Quantity,
    mesh_unit: str,
    chi: float | pint.Quantity | None,
) -> float:
    """A membrane current density (uA per membrane area, in mesh units), as ``I_s`` with ``dx``"""
    if not isinstance(amplitude, pint.Quantity):
        amplitude = amplitude * ureg("uA/cm**2")
    if amplitude.check("[current] / [length] ** 3"):
        if chi is None:
            raise ValueError("A volumetric stimulus amplitude needs chi")
        if not isinstance(chi, pint.Quantity):
            chi = chi * ureg(f"{mesh_unit}**-1")
        amplitude = amplitude / chi
    return float(amplitude.to(f"uA/{mesh_unit}**2").magnitude)


@dataclass(eq=False)
class EndocardialActivation:
    """An endocardial activation sequence from root sites (by default those of Durrer et al.)

    The activation times are computed when the object is created (collective). Use
    :meth:`stimulus` to get the stimulus current for :class:`beat.MonodomainModel`.

    Parameters
    ----------
    mesh, facet_tags
        A bi-ventricular mesh and its facet tags
    lv, rv, epi, base
        Marker values of the LV and RV endocardium, the epicardium and the base (all the valve
        planes)
    anterior
        Marker value(s) of an anterior surface (the pulmonary valve), which fixes the sign of the
        anterior direction; see :meth:`HeartFrame.from_facet_tags`
    anterior_point
        A point on the anterior side, instead of ``anterior``
    root_sites
        The root sites, by default :data:`DURRER_ROOT_SITES`
    conduction_velocity
        Speed of the endocardial wave, a float in mesh units per ms or a pint quantity, by
        default 2 mm/ms (a Purkinje-like speed)
    axis : {"base_normal", "apex_to_base"}
        The long axis of the heart frame, see :meth:`HeartFrame.from_facet_tags`
    max_height
        No stimulus above this fraction of the apex-base distance (the base has few Purkinje
        fibres and is activated by the tissue), by default 0.85
    delay_scale
        Multiplies the delays of the root sites (e.g. the size of the heart relative to a human
        heart), by default 1
    mesh_unit
        Unit of the mesh coordinates, by default "mm"
    stimulated_cells : {"facet", "vertex"}
        Which cells are stimulated: those with a facet on the endocardium (the default), or also
        those that only touch it at a vertex or an edge. Each cell is stimulated at the earliest
        time of its endocardial vertices. With ``"facet"``, nodes on a sharp concave edge of a
        cavity (where most of the surrounding tissue touches the endocardium only at the edge) get
        too little stimulus on a coarse mesh and are activated late, by the tissue; ``"vertex"``
        stimulates the whole fan of cells around them.
    eikonal_kwargs
        Passed to :class:`beat.eikonal.EikonalSolver` (``openmp``, ``cc``, ...)

    Attributes
    ----------
    frame : HeartFrame
        The coordinate system of the heart
    roots : dict[str, dict]
        Per root site: ``"x"``, the vertex it was placed on, and ``"delay"`` (ms, scaled)
    activation_times : dolfinx.fem.Function
        P1 time (ms) at which the endocardial wave reaches each vertex of the endocardium below
        ``max_height``; ``inf`` elsewhere
    cell_activation_times : dolfinx.fem.Function
        DG0 time (ms) at which each stimulated cell (see ``stimulated_cells``) is stimulated;
        ``inf`` for the other cells
    """

    mesh: dolfinx.mesh.Mesh
    facet_tags: dolfinx.mesh.MeshTags
    lv: int | Sequence[int]
    rv: int | Sequence[int]
    epi: int | Sequence[int]
    base: int | Sequence[int]
    anterior: int | Sequence[int] | None = None
    anterior_point: npt.ArrayLike | None = None
    root_sites: Mapping[str, RootSite] = field(default_factory=lambda: dict(DURRER_ROOT_SITES))
    conduction_velocity: float | pint.Quantity = field(
        default_factory=lambda: ureg.Quantity(2.0, "mm/ms"),
    )
    axis: Literal["base_normal", "apex_to_base"] = "base_normal"
    max_height: float = 0.85
    delay_scale: float = 1.0
    mesh_unit: str = "mm"
    stimulated_cells: Literal["facet", "vertex"] = "facet"
    eikonal_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.root_sites:
            raise ValueError("Give at least one root site")
        if self.stimulated_cells not in ("facet", "vertex"):
            raise ValueError(
                f"stimulated_cells must be 'facet' or 'vertex', got {self.stimulated_cells!r}"
            )
        self.frame = HeartFrame.from_facet_tags(
            self.mesh,
            self.facet_tags,
            lv=self.lv,
            rv=self.rv,
            epi=self.epi,
            base=self.base,
            anterior=self.anterior,
            anterior_point=self.anterior_point,
            axis=self.axis,
        )
        self._V = dolfinx.fem.functionspace(self.mesh, ("P", 1))
        self.roots = self._place_roots()
        self.activation_times = self._endocardial_times()
        self.cell_activation_times = self._cell_times()

    @classmethod
    def from_geometry(cls, geometry: Any, **kwargs: Any) -> EndocardialActivation:
        """Build from a geometry with ``mesh``, ``ffun`` and ``markers`` (name -> (value, dim)),
        such as a ``cardiac_geometries.geometry.Geometry`` or a :class:`beat.geometry.Geometry`.
        The markers ``LV``, ``RV`` and ``EPI`` are required; the base is every marker in
        :data:`BASE_MARKER_NAMES` that is present, and ``PV``, if present, is the anterior surface.
        The other keyword arguments go to the class."""
        markers = geometry.markers
        missing = [k for k in ("LV", "RV", "EPI") if k not in markers]
        base = [markers[k][0] for k in BASE_MARKER_NAMES if k in markers]
        if missing or not base:
            raise ValueError(
                f"The geometry needs the markers LV, RV, EPI and one of {BASE_MARKER_NAMES}; "
                f"missing {missing or 'a base marker'}, has {sorted(markers)}",
            )
        defaults: dict[str, Any] = {
            "lv": markers["LV"][0],
            "rv": markers["RV"][0],
            "epi": markers["EPI"][0],
            "base": base,
        }
        if "PV" in markers and "anterior_point" not in kwargs:
            defaults["anterior"] = markers["PV"][0]
        defaults.update(kwargs)
        return cls(mesh=geometry.mesh, facet_tags=geometry.ffun, **defaults)

    @property
    def _endocardial_values(self) -> list[int]:
        return _as_values(self.lv) + _as_values(self.rv)

    def _place_roots(self) -> dict[str, dict[str, Any]]:
        x = self._V.tabulate_dof_coordinates()
        surfaces = {
            "LV": x[_surface_dofs(self._V, self.facet_tags, _as_values(self.lv))],
            "RV": x[_surface_dofs(self._V, self.facet_tags, _as_values(self.rv))],
        }
        roots = {}
        for name, site in self.root_sites.items():
            p = place_root_site(self.frame, self.mesh.comm, surfaces[site.surface], site)
            roots[name] = {"x": p, "delay": site.delay * self.delay_scale}
            logger.debug(f"Root site {name!r} at {p}, delay {roots[name]['delay']} ms")
        return roots

    def _endocardial_times(self) -> dolfinx.fem.Function:
        mesh = self.mesh
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facets = np.unique(
            np.concatenate([self.facet_tags.find(v) for v in self._endocardial_values]),
        ).astype(np.int32)
        surface, _, vertex_map, _ = dolfinx.mesh.create_submesh(mesh, tdim - 1, facets)

        # Isotropic spread over the surface: any three orthonormal directions give V = v^2 I
        speed = _speed(self.conduction_velocity, self.mesh_unit)
        directions = [dolfinx.fem.Constant(surface, e) for e in np.eye(3)]
        solver = EikonalSolver(
            surface,
            [speed] * 3,
            directions,
            mesh_unit=self.mesh_unit,
            **self.eikonal_kwargs,
        )
        atol = 1e-9 * self.frame.length

        def at(p):
            return lambda x: np.linalg.norm(x.T - p, axis=1) < atol

        sources = [Source(at(r["x"]), t0=r["delay"]) for r in self.roots.values()]
        # A surface without a root site (only LV sites, say) is not reached and not stimulated
        t_surface = solver.solve(sources, allow_unreached=True)
        self.eikonal_stats = solver.stats

        # From the vertices of the surface to the P1 dofs of the mesh
        n_sub = surface.topology.index_map(0).size_local + surface.topology.index_map(0).num_ghosts
        parent_vertex = vertex_map.sub_topology_to_topology(np.arange(n_sub, dtype=np.int32), False)
        t_values = t_surface.x.array[_vertex_to_dof(t_surface.function_space)]
        t = dolfinx.fem.Function(self._V, name="endocardial_activation_time")
        t.x.array[:] = FAR
        t.x.array[_vertex_to_dof(self._V)[parent_vertex]] = np.where(
            np.isfinite(t_values), t_values, FAR
        )
        # A vertex that is ghosted on a rank where its endocardial facets are not local gets its
        # value from the ranks where they are: the minimum over all copies
        t.x.array[:] = -t.x.array
        t.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.MAX_VALUES, mode=PETSc.ScatterMode.REVERSE)
        t.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        t.x.array[:] = -t.x.array

        height, _, _ = self.frame.coordinates(self._V.tabulate_dof_coordinates())
        t.x.array[(t.x.array >= FAR) | (height > self.max_height)] = np.inf
        return t

    def _cell_times(self) -> dolfinx.fem.Function:
        mesh = self.mesh
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        mesh.topology.create_connectivity(tdim - 1, tdim)
        f2v = mesh.topology.connectivity(tdim - 1, 0)
        f2c = mesh.topology.connectivity(tdim - 1, tdim)
        v2d = _vertex_to_dof(self._V)
        t_vertex = self.activation_times.x.array
        num_cells = (
            mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
        )
        if self.stimulated_cells == "vertex":
            # inf off the endocardium, so this is the earliest endocardial vertex of each cell
            t_cell = t_vertex[self._V.dofmap.list].min(axis=1)
        else:
            t_cell = np.full(num_cells, np.inf)
            for value in self._endocardial_values:
                for f in self.facet_tags.find(value):
                    t_face = t_vertex[v2d[f2v.links(f)]].min()
                    cells = f2c.links(f)
                    t_cell[cells] = np.minimum(t_cell[cells], t_face)
        W = dolfinx.fem.functionspace(mesh, ("DG", 0))
        t = dolfinx.fem.Function(W, name="stimulus_time")
        t.x.array[W.dofmap.list[:, 0]] = t_cell
        t.x.scatter_forward()
        return t

    def stimulus(
        self,
        time: dolfinx.fem.Constant,
        dt: float | dolfinx.fem.Constant,
        *,
        amplitude: float | pint.Quantity = 40.0,
        duration: float = 2.0,
        start: float = 0.0,
        chi: float | pint.Quantity | None = None,
    ) -> Stimulus:
        """The stimulus current, to pass as ``I_s`` to :class:`beat.MonodomainModel`

        Every cell with a facet on the endocardium gets ``amplitude`` for ``duration`` ms from
        ``start`` plus its activation time. The pulse is averaged over a window of length ``dt``
        centred on ``time`` (the time at which the PDE evaluates the stimulus), so every cell
        receives the same charge whatever the time step and wherever its pulse starts within a
        step.

        Parameters
        ----------
        time
            The time Constant of the PDE model
        dt
            The time step (a float, or a Constant to change it later)
        amplitude
            A membrane current density, a float in uA/cm^2 or a pint quantity in current per
            area, or in current per volume together with ``chi``; by default 40 uA/cm^2
        duration
            Length of the pulse (ms), by default 2
        start
            Offset added to all the activation times (ms), e.g. the start of a beat
        chi
            The surface-to-volume ratio, only for a volumetric ``amplitude`` (float in 1 /
            mesh_unit or pint quantity)
        """
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        amp = _amplitude(amplitude, self.mesh_unit, chi)
        dt_c = (
            dt
            if isinstance(dt, dolfinx.fem.Constant)
            else dolfinx.fem.Constant(self.mesh, float(dt))
        )
        t_on = self.cell_activation_times + start
        overlap = ufl.min_value(time + dt_c / 2, t_on + duration) - ufl.max_value(
            time - dt_c / 2, t_on
        )
        expr = amp * ufl.max_value(overlap, 0.0) / dt_c
        return Stimulus(expr=expr, dZ=ufl.Measure("dx", domain=self.mesh))

    def summary(self) -> dict[str, Any]:
        """Root sites, frame and the range of the stimulus times (collective), e.g. for logging"""
        comm = self.mesh.comm
        n_owned = self.cell_activation_times.function_space.dofmap.index_map.size_local
        t = self.cell_activation_times.x.array[:n_owned]
        finite = t[np.isfinite(t)]
        return {
            "roots": {
                k: {"x": v["x"].tolist(), "delay": v["delay"]} for k, v in self.roots.items()
            },
            "num_stimulated_cells": int(comm.allreduce(len(finite), op=MPI.SUM)),
            "first_stimulus": float(
                comm.allreduce(finite.min() if len(finite) else np.inf, op=MPI.MIN)
            ),
            "last_stimulus": float(
                comm.allreduce(finite.max() if len(finite) else -np.inf, op=MPI.MAX)
            ),
            "frame": {
                "apex": self.frame.apex.tolist(),
                "base_center": self.frame.base_center.tolist(),
                "long_axis": self.frame.long_axis.tolist(),
                "septal": self.frame.septal.tolist(),
                "anterior": self.frame.anterior.tolist(),
                "length": self.frame.length,
            },
        }
