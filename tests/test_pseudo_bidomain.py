"""Tests for beat.pseudo_bidomain (all run serially and under mpirun -n 2)."""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import scifem
import ufl

import beat
from beat.ecg import ElectrodeRecorder, Lead, LeadSystem
from beat.pseudo_bidomain import (
    LeadFieldElectrodes,
    VolumeConductor,
    define_augmented_conductivity_tensor,
    interface_layer,
)


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


SIGMA_I = ufl.as_matrix([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
SIGMA_E = ufl.as_matrix([[3.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]])
ELECTRODES = {"tissue": [0.5, 0.4, 0.3], "bath": [1.1, 0.5, -0.1], "interface": [1.0, 0.5, 0.5]}


def all_cells(mesh):
    imap = mesh.topology.index_map(mesh.topology.dim)
    return np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)


def box_with_bath(comm, n=12, pad=3, split=False):
    """Tissue [0, 1]^3 (tag 1; with split, x > 1/2 is tag 3) inside a bath (tag 2), spacing 1/n"""
    h = 1.0 / n
    mesh = dolfinx.mesh.create_box(comm, [[-pad * h] * 3, [1.0 + pad * h] * 3], [n + 2 * pad] * 3)
    cells = all_cells(mesh)
    mid = dolfinx.mesh.compute_midpoints(mesh, 3, cells)
    values = np.where(np.all((mid > 0.0) & (mid < 1.0), axis=1), 1, 2)
    if split:
        values[(values == 1) & (mid[:, 0] > 0.5)] = 3
    return mesh, dolfinx.mesh.meshtags(mesh, 3, cells, values.astype(np.int32))


def smooth(x):
    return np.sin(2.0 * x[0]) * np.cos(3.0 * x[1]) + x[2] ** 2


def integral(expr, mesh):
    form = dolfinx.fem.form(expr * ufl.dx(domain=mesh))
    return mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)


def test_no_bath_matches_analytic_solution():
    errors = []
    for n in (8, 16):
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
        cells = all_cells(mesh)
        tags = dolfinx.mesh.meshtags(mesh, 3, cells, np.ones(len(cells), dtype=np.int32))
        conductor = VolumeConductor(mesh, tags, 1, (), 1.0, 2.0)
        v = dolfinx.fem.Function(conductor.V)
        v.interpolate(lambda x: np.cos(np.pi * x[0]))
        phi = conductor.solve(v)
        x = ufl.SpatialCoordinate(mesh)
        exact = -ufl.cos(ufl.pi * x[0]) / 3.0
        errors.append(np.sqrt(integral((phi - exact) ** 2, mesh)))
    assert np.log2(errors[0] / errors[1]) > 1.8


def test_solve_is_repeatable_and_shift_invariant():
    mesh, tags = box_with_bath(MPI.COMM_WORLD)
    conductor = VolumeConductor(mesh, tags, 1, 2, SIGMA_I, SIGMA_E, 1.0)
    v = dolfinx.fem.Function(conductor.V)
    v.interpolate(smooth)
    phi = conductor.solve(v)
    assert abs(integral(phi, mesh)) < 1e-10
    again = conductor.solve(v)
    assert np.allclose(phi.x.array, again.x.array, atol=1e-12)
    v.x.array[:] += 5.0
    shifted = conductor.solve(v)
    scale = mesh.comm.allreduce(np.abs(phi.x.array).max(), op=MPI.MAX)
    assert np.allclose(phi.x.array, shifted.x.array, atol=1e-7 * scale)


def test_bath_with_tissue_conductivity_equals_no_bath():
    comm = MPI.COMM_WORLD
    mesh, tags = box_with_bath(comm)
    cells = all_cells(mesh)
    everything = dolfinx.mesh.meshtags(mesh, 3, cells, np.ones(len(cells), dtype=np.int32))

    def bump(x):
        r2 = (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 + (x[2] - 0.5) ** 2
        return np.maximum(0.0, 0.04 - r2) ** 2

    results = []
    for conductor in (
        VolumeConductor(mesh, tags, 1, 2, 1.0, 2.0, 3.0),
        VolumeConductor(mesh, everything, 1, (), 1.0, 2.0),
    ):
        v = dolfinx.fem.Function(conductor.V)
        v.interpolate(bump)
        results.append(conductor.solve(v).x.array.copy())
    scale = comm.allreduce(np.abs(results[1]).max(), op=MPI.MAX)
    assert scale > 0
    assert np.allclose(results[0], results[1], atol=1e-7 * scale)


def test_multiple_tissue_markers():
    comm = MPI.COMM_WORLD
    mesh, tags = box_with_bath(comm)
    _, split = box_with_bath(comm, split=True)
    out = []
    for markers, cell_tags in ((1, tags), ((1, 3), split)):
        conductor = VolumeConductor(mesh, cell_tags, markers, 2, SIGMA_I, SIGMA_E, 1.0)
        v = dolfinx.fem.Function(conductor.V)
        v.interpolate(smooth)
        out.append(conductor.solve(v).x.array.copy())
    scale = comm.allreduce(np.abs(out[0]).max(), op=MPI.MAX)
    assert np.allclose(out[0], out[1], atol=1e-7 * scale)


@pytest.mark.parametrize("n_tissue", [12, 9])
def test_lead_field_reciprocity(n_tissue):
    """Lead fields equal point evaluation of the full solve; exact on matching meshes (ruling R1)"""
    comm = MPI.COMM_WORLD
    mesh, tags = box_with_bath(comm, n=12)
    conductor = VolumeConductor(mesh, tags, 1, 2, SIGMA_I, SIGMA_E, 1.0)
    tissue = dolfinx.mesh.create_unit_cube(comm, n_tissue, n_tissue, n_tissue)
    v = dolfinx.fem.Function(dolfinx.fem.functionspace(tissue, ("P", 1)))
    v.interpolate(smooth)
    recorded = LeadFieldElectrodes(conductor, v, SIGMA_I, ELECTRODES)()
    assert comm.allgather(recorded) == [recorded] * comm.size
    phi = conductor.solve(v)
    direct = scifem.evaluate_function(phi, np.array(list(ELECTRODES.values()))).ravel()
    for i, name in enumerate(ELECTRODES):
        if n_tissue == 12:
            assert recorded[name] == pytest.approx(direct[i], rel=1e-8, abs=1e-12)
        elif name == "bath":
            assert recorded[name] == pytest.approx(direct[i], rel=2e-2)


def test_from_lead_fields_matches_constructor():
    comm = MPI.COMM_WORLD
    mesh, tags = box_with_bath(comm, n=12)
    conductor = VolumeConductor(mesh, tags, 1, 2, SIGMA_I, SIGMA_E, 1.0)
    tissue = dolfinx.mesh.create_unit_cube(comm, 12, 12, 12)
    v = dolfinx.fem.Function(dolfinx.fem.functionspace(tissue, ("P", 1)))
    v.interpolate(smooth)
    full = LeadFieldElectrodes(conductor, v, SIGMA_I, ELECTRODES)
    again = LeadFieldElectrodes.from_lead_fields(v, SIGMA_I, full.lead_fields)
    assert again.names == full.names
    assert again() == pytest.approx(full())


def test_errors_are_collective():
    comm = MPI.COMM_WORLD
    mesh, tags = box_with_bath(comm, n=6, pad=2)
    with pytest.raises(ValueError, match="tagged 7"):
        VolumeConductor(mesh, tags, 7, 2, 1.0, 1.0)
    with pytest.raises(ValueError, match="tagged 9"):
        VolumeConductor(mesh, tags, 1, 9, 1.0, 1.0)
    tissue_only = dolfinx.mesh.meshtags(
        mesh,
        3,
        tags.find(1),
        np.ones(len(tags.find(1)), dtype=np.int32),
    )
    with pytest.raises(ValueError, match="neither tissue nor bath"):
        VolumeConductor(mesh, tissue_only, 1, (), 1.0, 1.0)
    conductor = VolumeConductor(mesh, tags, 1, 2, 1.0, 1.0)
    with pytest.raises(ValueError, match="outside"):
        conductor.lead_field([5.0, 5.0, 5.0])
    with pytest.raises(ValueError, match="coordinates"):
        conductor.lead_field([0.5, 0.5])
    square = dolfinx.mesh.create_unit_square(comm, 4, 4)
    v2d = dolfinx.fem.Function(dolfinx.fem.functionspace(square, ("P", 1)))
    with pytest.raises(ValueError, match="dimension"):
        conductor.solve(v2d)


def test_lead_field_accepts_point_just_outside_boundary():
    mesh, tags = box_with_bath(MPI.COMM_WORLD, n=6, pad=2)
    conductor = VolumeConductor(mesh, tags, 1, 2, 1.0, 1.0)
    top = 1.0 + 2.0 / 6.0
    Z = conductor.lead_field([0.5, 0.5, top + 1e-12])
    assert np.all(np.isfinite(Z.x.array))


def test_uncovered_interpolation_raises_collectively():
    comm = MPI.COMM_WORLD
    mesh, tags = box_with_bath(comm, n=6, pad=2)
    conductor = VolumeConductor(mesh, tags, 1, 2, 1.0, 1.0)
    small = dolfinx.mesh.create_box(comm, [[0.0] * 3, [0.8] * 3], [4] * 3)
    v_small = dolfinx.fem.Function(dolfinx.fem.functionspace(small, ("P", 1)))
    with pytest.raises(ValueError, match="not covered"):
        conductor.solve(v_small)
    big = dolfinx.mesh.create_box(comm, [[0.0] * 3, [1.5] * 3], [6] * 3)
    v_big = dolfinx.fem.Function(dolfinx.fem.functionspace(big, ("P", 1)))
    with pytest.raises(ValueError, match="not inside"):
        LeadFieldElectrodes(conductor, v_big, 1.0, {"a": [0.5, 0.5, 0.5]})


# Annotated on purpose: mypy then checks the body, i.e. that LeadFieldElectrodes satisfies the
# ElectrodeRecorder protocol. Keep the annotation.
def test_lead_system_reads_a_lead_field_recording() -> None:
    comm = MPI.COMM_WORLD
    mesh, tags = box_with_bath(comm, n=12)
    conductor = VolumeConductor(mesh, tags, 1, 2, SIGMA_I, SIGMA_E, 1.0)
    tissue = dolfinx.mesh.create_unit_cube(comm, 12, 12, 12)
    v = dolfinx.fem.Function(dolfinx.fem.functionspace(tissue, ("P", 1)))
    v.interpolate(smooth)
    recorder: ElectrodeRecorder = LeadFieldElectrodes(conductor, v, SIGMA_I, ELECTRODES)
    potentials = recorder()
    leads = LeadSystem((Lead("bipolar", {"tissue": 1.0, "bath": -1.0}),)).leads(potentials)
    assert leads["bipolar"] == pytest.approx(potentials["tissue"] - potentials["bath"])
