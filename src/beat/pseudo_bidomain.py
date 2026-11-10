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

from typing import Any

from petsc4py import PETSc

import dolfinx
import numpy as np
import numpy.typing as npt
import pint

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
