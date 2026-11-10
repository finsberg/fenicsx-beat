"""Bath loading and the pseudo-bidomain model

Neic et al. (2017, Sec. 2.2) compute extracellular potentials from a monodomain model in two steps:

* **Augmented monodomain.** Near a tissue-bath interface the extracellular current returns
  through the bath, which speeds up conduction along the surface ("bath loading"). Bishop & Plank
  (2011) reproduce this in a monodomain model by replacing, in a thin layer of tissue next to the
  interface, the harmonic mean of the intra- and extracellular conductivities by the harmonic mean
  of the intracellular and bath conductivities (Neic Eqs. 14 and 15):
  :func:`interface_layer` and :func:`define_augmented_conductivity_tensor`.
* **Pseudo-bidomain.** Given :math:`v`, the extracellular potential in tissue and bath solves the
  elliptic part of the bidomain model (Neic Eq. 11): :class:`VolumeConductor`. Electrograms at a few
  points are cheaper through lead fields: :class:`LeadFieldElectrodes`.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import numpy.typing as npt
import pint
import scifem
import ufl

from .conductivities import conductivity_tensor, get_harmonic_mean_conductivity


def _ghost_max(
    work: dolfinx.fem.Function,
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Every rank's copy of a dof becomes the maximum over all ranks"""
    work.x.array[:] = values
    work.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.MAX_VALUES, mode=PETSc.ScatterMode.REVERSE)
    work.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return work.x.array.copy()


def interface_layer(
    mesh: dolfinx.mesh.Mesh,
    num_layers: int = 3,
    facets: npt.NDArray[np.int32] | None = None,
) -> dolfinx.fem.Function:
    """The cells within ``num_layers`` element rings of the tissue-bath interface

    Ring 1 is every cell with a vertex on one of ``facets`` (by default all exterior facets, i.e. a
    tissue mesh fully immersed in bath); ring :math:`k + 1` is every cell sharing a vertex with ring
    :math:`k`. Bishop & Plank (2011) use three rings on a 100 µm mesh. Returns a DG0 function that
    is 1 in the layer and 0 elsewhere; the result does not depend on the mesh partition.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be at least 1, got {num_layers}")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    if facets is None:
        facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    work = dolfinx.fem.Function(V)
    marked = np.zeros(work.x.array.size)
    interface_dofs = dolfinx.fem.locate_dofs_topological(
        V,
        tdim - 1,
        np.asarray(facets, dtype=np.int32),
    )
    marked[interface_dofs] = 1.0
    marked = _ghost_max(work, marked)
    imap = mesh.topology.index_map(tdim)
    num_cells = imap.size_local + imap.num_ghosts
    cell_dofs = V.dofmap.list[:num_cells]
    in_layer = marked[cell_dofs].max(axis=1) > 0.5
    for _ in range(num_layers - 1):
        marked = np.zeros_like(marked)
        marked[np.unique(cell_dofs[in_layer])] = 1.0
        marked = _ghost_max(work, marked)
        in_layer = marked[cell_dofs].max(axis=1) > 0.5
    Q = dolfinx.fem.functionspace(mesh, ("DG", 0))
    layer = dolfinx.fem.Function(Q, name="augmentation_layer")
    layer.x.array[Q.dofmap.list[:num_cells].ravel()] = in_layer.astype(np.float64)
    layer.x.scatter_forward()
    return layer


def define_augmented_conductivity_tensor(
    chi: pint.Quantity,
    f0: dolfinx.fem.Function | dolfinx.fem.Constant,
    layer: dolfinx.fem.Function,
    g_il: float | pint.Quantity = 0.17,
    g_it: float | pint.Quantity = 0.019,
    g_el: float | pint.Quantity = 0.62,
    g_et: float | pint.Quantity = 0.24,
    g_b: float | pint.Quantity = 1.0,
) -> Any:
    """The monodomain conductivity tensor with bath loading (Neic et al. 2017, Eqs. 14-15)

    In the bulk each direction gets the harmonic mean of the intra- and extracellular
    conductivities, as in :func:`beat.conductivities.define_conductivity_tensor`; where ``layer`` is
    1 (see :func:`interface_layer`) the extracellular conductivity is replaced by the bath
    conductivity ``g_b``. Floats are in S/m; the result is scaled by :math:`1/\\chi` like
    ``define_conductivity_tensor``, so it goes straight into :class:`beat.MonodomainModel`.
    """
    bulk_l, bulk_t = get_harmonic_mean_conductivity(chi, g_il, g_it, g_el, g_et)
    edge_l, edge_t = get_harmonic_mean_conductivity(chi, g_il, g_it, g_b, g_b)
    s_l = bulk_l + layer * (edge_l - bulk_l)
    s_t = bulk_t + layer * (edge_t - bulk_t)
    return conductivity_tensor(s_l, s_t, f0)


#: CG with BoomerAMG, aggressive coarsening (about twice as fast as the default BoomerAMG for the
#: wedge in its bath, at the same memory)
DEFAULT_PETSC_OPTIONS: dict[str, Any] = {
    "ksp_type": "cg",
    "ksp_rtol": 1e-10,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_agg_nl": 1,
    "pc_hypre_boomeramg_strong_threshold": 0.5,
    "pc_hypre_boomeramg_coarsen_type": "HMIS",
    "pc_hypre_boomeramg_interp_type": "ext+i",
    "pc_hypre_boomeramg_P_max": 4,
}
_prefixes = itertools.count()


def _markers(markers: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(markers, (int, np.integer)):
        return (int(markers),)
    return tuple(int(m) for m in markers)


def _all_cells(mesh: dolfinx.mesh.Mesh) -> npt.NDArray[np.int32]:
    imap = mesh.topology.index_map(mesh.topology.dim)
    return np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)


class VolumeConductor:
    """The extracellular potential of the pseudo-bidomain model (Neic et al. 2017, Eq. 11)

    Solves, on a mesh of tissue (cells tagged ``tissue_markers``) and bath (``bath_markers``),

    .. math::

        \\int_\\Omega \\sigma \\nabla\\phi_e \\cdot \\nabla w \\, dx
            = -\\int_{\\Omega_T} \\sigma_i \\nabla v \\cdot \\nabla w \\, dx,
        \\qquad \\sigma = \\sigma_i + \\sigma_e \\text{ in tissue}, \\; \\sigma_b \\text{ in bath},

    with no flux through the outer boundary and :math:`\\phi_e` normalised to zero mean.
    ``sigma_i``, ``sigma_e`` (tensors or floats, on ``mesh``) and ``sigma_b`` are plain
    conductivities, not scaled by :math:`\\chi`; only their ratios matter, so use one unit for all
    three. ``bath_markers`` may be empty for tissue without a bath. Every cell must be tagged.

    The matrix is assembled and the preconditioner built once, so :meth:`solve` and
    :meth:`lead_field` cost one Krylov solve each. ``petsc_options`` replaces
    :data:`DEFAULT_PETSC_OPTIONS`.
    """

    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        cell_tags: dolfinx.mesh.MeshTags,
        tissue_markers: int | Sequence[int],
        bath_markers: int | Sequence[int],
        sigma_i: Any,
        sigma_e: Any,
        sigma_b: Any = 1.0,
        petsc_options: dict[str, Any] | None = None,
    ) -> None:
        comm = mesh.comm
        self.mesh = mesh
        self.tissue_markers = _markers(tissue_markers)
        self.bath_markers = _markers(bath_markers)
        if not self.tissue_markers:
            raise ValueError("At least one tissue marker is needed")
        tdim = mesh.topology.dim
        n_owned = mesh.topology.index_map(tdim).size_local
        tagged = np.zeros(n_owned, dtype=bool)
        for name, markers in (("tissue", self.tissue_markers), ("bath", self.bath_markers)):
            for marker in markers:
                cells = cell_tags.find(marker)
                if comm.allreduce(len(cells), op=MPI.SUM) == 0:
                    raise ValueError(f"No {name} cells are tagged {marker}")
                tagged[cells[cells < n_owned]] = True
        if comm.allreduce(int((~tagged).sum()), op=MPI.SUM) > 0:
            raise ValueError("Some cells are tagged neither tissue nor bath")
        self._tissue_cells = np.unique(
            np.concatenate([cell_tags.find(m) for m in self.tissue_markers]),
        ).astype(np.int32)

        self.V = dolfinx.fem.functionspace(mesh, ("P", 1))
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
        dx_tissue = dx(self.tissue_markers)
        u, w = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        a = ufl.inner((sigma_i + sigma_e) * ufl.grad(u), ufl.grad(w)) * dx_tissue
        if self.bath_markers:
            a += sigma_b * ufl.inner(ufl.grad(u), ufl.grad(w)) * dx(self.bath_markers)
        self.A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
        self.A.assemble()
        self._nullspace = PETSc.NullSpace().create(constant=True, comm=comm)
        self.A.setNullSpace(self._nullspace)
        self.ksp = PETSc.KSP().create(comm)
        self.ksp.setOperators(self.A)
        prefix = f"beat_volume_conductor_{next(_prefixes)}_"
        self.ksp.setOptionsPrefix(prefix)
        options = PETSc.Options()
        for key, value in (
            DEFAULT_PETSC_OPTIONS if petsc_options is None else petsc_options
        ).items():
            options[prefix + key] = value  # type: ignore[index]
        self.ksp.setFromOptions()

        self._v = dolfinx.fem.Function(self.V, name="v")
        self._rhs = dolfinx.fem.form(
            -ufl.inner(sigma_i * ufl.grad(self._v), ufl.grad(w)) * dx_tissue,
        )
        self._test_integral = dolfinx.fem.form(w * ufl.dx(domain=mesh))
        self._work = dolfinx.fem.Function(self.V)
        self._mean = dolfinx.fem.form(self._work * ufl.dx(domain=mesh))
        one = dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * ufl.dx(domain=mesh))
        self.volume = comm.allreduce(dolfinx.fem.assemble_scalar(one), op=MPI.SUM)
        self._interpolation: list[tuple[dolfinx.fem.FunctionSpace, Any]] = []

    def _interpolation_data(self, V_from: dolfinx.fem.FunctionSpace) -> Any:
        for V, data in self._interpolation:
            if V is V_from:
                return data
        data = dolfinx.fem.create_interpolation_data(
            self.V,
            V_from,
            self._tissue_cells,
            padding=1e-8,
        )
        self._interpolation.append((V_from, data))
        return data

    def _set_v(self, v: dolfinx.fem.Function) -> None:
        V_from = v.function_space
        if V_from.mesh.geometry.dim != self.mesh.geometry.dim:
            raise ValueError(
                f"v has geometric dimension {V_from.mesh.geometry.dim}, the volume conductor "
                f"{self.mesh.geometry.dim}",
            )
        if V_from.mesh is self.mesh:
            self._v.interpolate(v)
        else:
            self._v.x.array[:] = 0.0
            self._v.interpolate_nonmatching(v, self._tissue_cells, self._interpolation_data(V_from))
        self._v.x.scatter_forward()

    def _solve(self, b: dolfinx.fem.Function, name: str) -> dolfinx.fem.Function:
        """Solve with the local (not yet accumulated) right-hand side ``b``"""
        b.x.scatter_reverse(dolfinx.la.InsertMode.add)
        vec = b.x.petsc_vec
        self._nullspace.remove(vec)
        out = dolfinx.fem.Function(self.V, name=name)
        self.ksp.solve(vec, out.x.petsc_vec)
        reason: int = self.ksp.getConvergedReason()  # type: ignore[assignment]
        if reason < 0:
            raise RuntimeError(f"The volume-conductor solve did not converge (reason {reason})")
        out.x.scatter_forward()
        self._work.x.array[:] = out.x.array
        mean = self.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(self._mean), op=MPI.SUM)
        out.x.array[:] -= mean / self.volume
        return out

    def solve(self, v: dolfinx.fem.Function) -> dolfinx.fem.Function:
        """:math:`\\phi_e` for the transmembrane potential ``v``

        ``v`` is a function on this mesh or on a separate tissue mesh; in the latter case it is
        interpolated onto the tissue cells (exact when the tissue cells coincide).
        """
        self._set_v(v)
        b = dolfinx.fem.Function(self.V)
        b.x.array[:] = dolfinx.fem.assemble_vector(self._rhs).array
        return self._solve(b, "phi_e")

    def lead_field(self, point: npt.ArrayLike) -> dolfinx.fem.Function:
        """The lead field :math:`Z_P` of an electrode at ``point``

        :math:`\\int_\\Omega \\sigma \\nabla Z_P \\cdot \\nabla w\\,dx = w(P) - |\\Omega|^{-1}
        \\int_\\Omega w\\,dx`, zero mean. Then :math:`\\phi_e(P) = -\\int_{\\Omega_T}
        \\sigma_i \\nabla v \\cdot \\nabla Z_P\\,dx` for every :math:`v` (reciprocity).
        """
        comm = self.mesh.comm
        gdim = self.mesh.geometry.dim
        p = np.asarray(point, dtype=np.float64).ravel()
        if p.size != gdim:
            raise ValueError(f"The electrode {list(p)} needs {gdim} coordinates")
        points = np.zeros((1 if comm.rank == 0 else 0, 3))
        if comm.rank == 0:
            points[0, :gdim] = p
        ownership = dolfinx.geometry.determine_point_ownership(self.mesh, points, 1e-10)
        inside = comm.bcast(comm.rank != 0 or bool(ownership.src_owner[0] >= 0), root=0)
        if not inside:
            raise ValueError(f"The electrode {list(p)} is outside the volume-conductor mesh")
        b = dolfinx.fem.Function(self.V)
        scifem.PointSource(self.V, points, magnitude=1.0).apply_to_vector(b)
        b.x.array[:] -= dolfinx.fem.assemble_vector(self._test_integral).array / self.volume
        return self._solve(b, "lead_field")


class LeadFieldElectrodes:
    """:math:`\\phi_e` at a few electrodes from lead fields, cheap enough for every time step

    One lead field per electrode is solved once (:meth:`VolumeConductor.lead_field`) and
    interpolated onto ``v``'s space; each call then assembles
    :math:`-\\int \\sigma_i \\nabla v \\cdot \\nabla Z_P \\, dx` over ``v``'s mesh, which must be
    the tissue (dynamics) mesh and nothing else. ``sigma_i`` is the intracellular conductivity on
    that mesh, in the units given to the :class:`VolumeConductor`. The values equal point
    evaluation of :meth:`VolumeConductor.solve` exactly when the tissue cells coincide, and up to
    interpolation error otherwise. Calling returns ``{name: value}``, the same on every rank.
    """

    def __init__(
        self,
        conductor: VolumeConductor,
        v: dolfinx.fem.Function,
        sigma_i: Any,
        electrodes: Mapping[str, npt.ArrayLike],
    ) -> None:
        V = v.function_space
        if V.mesh.geometry.dim != conductor.mesh.geometry.dim:
            raise ValueError(
                f"v has geometric dimension {V.mesh.geometry.dim}, the volume conductor "
                f"{conductor.mesh.geometry.dim}",
            )
        cells = _all_cells(V.mesh)
        data = None
        lead_fields = {}
        for name, point in electrodes.items():
            Z_conductor = conductor.lead_field(point)
            if data is None:
                data = dolfinx.fem.create_interpolation_data(V, conductor.V, cells, padding=1e-8)
            Z = dolfinx.fem.Function(V, name=f"lead_field_{name}")
            Z.interpolate_nonmatching(Z_conductor, cells, data)
            Z.x.scatter_forward()
            lead_fields[name] = Z
        self._setup(v, sigma_i, lead_fields)

    @classmethod
    def from_lead_fields(
        cls,
        v: dolfinx.fem.Function,
        sigma_i: Any,
        lead_fields: Mapping[str, dolfinx.fem.Function],
    ) -> LeadFieldElectrodes:
        """The recorder for lead fields already on ``v``'s mesh (e.g. loaded from disk)"""
        self = cls.__new__(cls)
        self._setup(v, sigma_i, lead_fields)
        return self

    def _setup(
        self,
        v: dolfinx.fem.Function,
        sigma_i: Any,
        lead_fields: Mapping[str, dolfinx.fem.Function],
    ) -> None:
        self.v = v
        self.lead_fields = dict(lead_fields)
        self.names = tuple(self.lead_fields)
        self._comm = v.function_space.mesh.comm
        dx = ufl.dx(domain=v.function_space.mesh)
        self._forms = [
            dolfinx.fem.form(-ufl.inner(sigma_i * ufl.grad(v), ufl.grad(Z)) * dx)
            for Z in self.lead_fields.values()
        ]

    def __call__(self) -> dict[str, float]:
        self.v.x.scatter_forward()
        values = np.array(
            [dolfinx.fem.assemble_scalar(form) for form in self._forms],
            dtype=np.float64,
        )
        self._comm.Allreduce(MPI.IN_PLACE, values, op=MPI.SUM)
        return dict(zip(self.names, values.tolist()))
