from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import beat
from beat.geometry import WEDGE_LAYERS, get_3D_wedge_geometry, wedge_layer_markers


@pytest.fixture(scope="module")
def wedge():
    return get_3D_wedge_geometry(MPI.COMM_WORLD, dx=1.0, fiber_space=("P", 1))


def test_layer_volume_fractions(wedge):
    mesh = wedge.geometry.mesh
    DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    layers = wedge_layer_markers(DG0)
    expected = {"EPI": 0.25, "M": 0.45, "ENDO": 0.30}
    total = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ufl.dx(domain=mesh))),
    )
    for name, value in WEDGE_LAYERS.items():
        indicator = dolfinx.fem.Function(DG0)
        indicator.x.array[:] = layers.x.array == value
        vol = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(indicator * ufl.dx)))
        assert abs(vol / total - expected[name]) <= 1.0 / 25.0 + 1e-12, name


def test_fibers(wedge):
    f0, s0, n0 = wedge.geometry.f0, wedge.geometry.s0, wedge.geometry.n0
    V = f0.function_space
    x = V.tabulate_dof_coordinates()
    f = f0.x.array.reshape(-1, 3)
    s = s0.x.array.reshape(-1, 3)
    n = n0.x.array.reshape(-1, 3)
    angle = np.rad2deg(np.arctan2(f[:, 2], f[:, 1]))
    np.testing.assert_allclose(angle, -60.0 + 120.0 * x[:, 0] / 25.0, atol=1e-10)
    for a, b in ((f, s), (f, n), (s, n)):
        np.testing.assert_allclose(np.einsum("ij,ij->i", a, b), 0.0, atol=1e-12)
    for a in (f, s, n):
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-12)


def test_stimuli_and_electrode(wedge):
    assert wedge.stimuli == {"S1": (1, 0.0), "S2": (2, 5.0), "S3": (3, 0.0)}
    comm = wedge.geometry.mesh.comm
    for tag, _ in wedge.stimuli.values():
        assert comm.allreduce(len(wedge.cell_tags.find(tag))) > 0
    np.testing.assert_allclose(wedge.electrode, [25.0, 12.5, 2.5])


def test_too_coarse_for_stimulus_raises():
    with pytest.raises(ValueError, match="stimulus"):
        get_3D_wedge_geometry(MPI.COMM_WORLD, dx=2.5)


def test_hexahedral_wedge():
    w = get_3D_wedge_geometry(MPI.COMM_WORLD, dx=1.0, cell_type=dolfinx.mesh.CellType.hexahedron)
    assert w.geometry.mesh.topology.cell_type == dolfinx.mesh.CellType.hexahedron


def test_neic_conductivities():
    g = beat.conductivities.default_conductivities("Neic2017_wedge")
    assert g["g_il"].to("S/m").magnitude == pytest.approx(0.16069)
    assert g["g_el"].to("S/m").magnitude == pytest.approx(0.625)
    assert g["g_it"].to("S/m").magnitude == pytest.approx(0.04258)
    assert g["g_et"].to("S/m").magnitude == pytest.approx(0.236)
    assert g["chi"].to("1/cm").magnitude == pytest.approx(1400.0)
