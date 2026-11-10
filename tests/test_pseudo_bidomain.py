"""Tests for beat.pseudo_bidomain (all run serially and under mpirun -n 2)."""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import beat
from beat.pseudo_bidomain import define_augmented_conductivity_tensor, interface_layer


def num_marked(layer):
    n_owned = layer.function_space.dofmap.index_map.size_local
    return layer.function_space.mesh.comm.allreduce(
        int(round(layer.x.array[:n_owned].sum())),
        op=MPI.SUM,
    )


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize(
    "cell_type, per_hex",
    [(dolfinx.mesh.CellType.tetrahedron, 6), (dolfinx.mesh.CellType.hexahedron, 1)],
)
def test_interface_layer_counts(k, cell_type, per_hex):
    n = 10
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, cell_type=cell_type)
    layer = interface_layer(mesh, num_layers=k)
    assert num_marked(layer) == per_hex * (n**3 - (n - 2 * k) ** 3)


def test_interface_layer_2d():
    n, k = 10, 2
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    assert num_marked(interface_layer(mesh, num_layers=k)) == 2 * (n**2 - (n - 2 * k) ** 2)


def test_interface_layer_restricted_to_facets():
    n, k = 10, 2
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    facets = dolfinx.mesh.locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[0], 0.0))
    layer = interface_layer(mesh, num_layers=k, facets=facets)
    assert num_marked(layer) == 6 * k * n * n


def test_interface_layer_needs_a_layer():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    with pytest.raises(ValueError, match="num_layers"):
        interface_layer(mesh, num_layers=0)


G = dict(g_il=0.16069, g_it=0.04258, g_el=0.625, g_et=0.236)


def tensor_values(expr, mesh):
    X = np.array([[0.25, 0.25, 0.25]])
    cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)
    return dolfinx.fem.Expression(expr, X).eval(mesh, cells).reshape(len(cells), -1)


def test_augmented_tensor_bulk_and_layer_values():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 6, 6, 6)
    layer = interface_layer(mesh, num_layers=1)
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    chi = 1400.0 * beat.units.ureg("cm**-1")
    M = define_augmented_conductivity_tensor(chi, f0, layer, g_b=1.0, **G)
    vals = tensor_values(M, mesh).reshape(-1, 3, 3)
    in_layer = tensor_values(layer, mesh).ravel() > 0.5
    assert mesh.comm.allreduce(int(in_layer.sum())) > 0
    assert mesh.comm.allreduce(int((~in_layer).sum())) > 0

    def hm(a, b):
        return a * b / (a + b)

    scale = (1.0 * beat.units.ureg("S/m") / chi).to("uA/mV").magnitude
    s_l = np.where(in_layer, hm(G["g_il"], 1.0), hm(G["g_il"], G["g_el"])) * scale
    s_t = np.where(in_layer, hm(G["g_it"], 1.0), hm(G["g_it"], G["g_et"])) * scale
    assert np.allclose(vals[:, 0, 0], s_l)
    assert np.allclose(vals[:, 1, 1], s_t)
    assert np.allclose(vals[:, 2, 2], s_t)
    assert np.allclose(vals[:, 0, 1], 0.0)


def test_augmented_tensor_without_layer_is_the_plain_tensor():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    layer = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", 0)))
    f0 = dolfinx.fem.Constant(mesh, np.array([0.0, 0.6, 0.8]))
    chi = 1400.0 * beat.units.ureg("cm**-1")
    augmented = define_augmented_conductivity_tensor(chi, f0, layer, g_b=1.0, **G)
    plain = beat.conductivities.define_conductivity_tensor(chi, f0, **G)
    assert np.allclose(tensor_values(augmented, mesh), tensor_values(plain, mesh))


def test_wedge_bath_conductivities():
    g = beat.conductivities.default_conductivities("Neic2017_wedge_bath")
    plain = beat.conductivities.default_conductivities("Neic2017_wedge")
    assert g["g_b"].to("S/m").magnitude == 1.0
    assert {k: v for k, v in g.items() if k != "g_b"} == plain
