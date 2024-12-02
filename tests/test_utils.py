from mpi4py import MPI

import dolfinx
import numpy as np
import scifem

import beat


def test_expand_layer_single():
    N = 50
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)

    endo_marker = 1
    epi_marker = 2
    tol = 1.0e-8

    fdim = mesh.topology.dim - 1
    facets_endo = dolfinx.mesh.locate_entities_boundary(
        mesh,
        fdim,
        lambda x: x[0] <= tol,
    )

    facets_epi = dolfinx.mesh.locate_entities_boundary(
        mesh,
        fdim,
        lambda x: x[0] >= 1 - tol,
    )
    marked_facets = np.hstack([facets_endo, facets_epi])
    marked_values = np.hstack(
        [np.full(len(facets_endo), endo_marker), np.full(len(facets_epi), epi_marker)],
    )
    sorted_facets = np.argsort(marked_facets)

    ft = dolfinx.mesh.meshtags(
        mesh,
        fdim,
        marked_facets[sorted_facets],
        marked_values[sorted_facets],
    )

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    output_mid_marker = 4
    output_endo_marker = 3
    output_epi_marker = 1
    markers = beat.utils.expand_layer(
        V=V,
        ft=ft,
        endo_marker=endo_marker,
        epi_marker=epi_marker,
        endo_size=0.3,
        epi_size=0.3,
        output_mid_marker=output_mid_marker,
        output_endo_marker=output_endo_marker,
        output_epi_marker=output_epi_marker,
    )

    points = np.array([(x, y) for x in [0.0, 0.1, 0.2] for y in [0.0, 0.5, 1.0]])

    endo = scifem.evaluate_function(markers, points)
    assert np.allclose(endo, output_endo_marker)

    mid = scifem.evaluate_function(markers, points + np.array([0.4, 0.0]))
    assert np.allclose(mid, output_mid_marker)

    epi = scifem.evaluate_function(markers, points + np.array([0.8, 0.0]))
    assert np.allclose(epi, output_epi_marker)


def test_expand_layer_biv():
    N = 50
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)

    endo_lv_marker = 1
    endo_rv_marker = 2
    epi_marker = 3
    tol = 1.0e-10

    fdim = mesh.topology.dim - 1
    facets_lv_endo = dolfinx.mesh.locate_entities_boundary(
        mesh,
        fdim,
        lambda x: np.logical_and(x[1] <= tol, x[0] <= 0.5 + tol),
    )
    facets_rv_endo = dolfinx.mesh.locate_entities_boundary(
        mesh,
        fdim,
        lambda x: np.logical_and(x[1] >= 1 - tol, x[0] <= 0.5 + tol),
    )

    facets_epi = dolfinx.mesh.locate_entities_boundary(
        mesh,
        fdim,
        lambda x: x[0] >= 1 - tol,
    )
    marked_facets = np.hstack([facets_lv_endo, facets_rv_endo, facets_epi])
    marked_values = np.hstack(
        [
            np.full(len(facets_lv_endo), endo_lv_marker),
            np.full(len(facets_rv_endo), endo_rv_marker),
            np.full(len(facets_epi), epi_marker),
        ],
    )
    sorted_facets = np.argsort(marked_facets)

    ft = dolfinx.mesh.meshtags(
        mesh,
        fdim,
        marked_facets[sorted_facets],
        marked_values[sorted_facets],
    )

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    output_mid_marker = 4
    output_endo_marker = 3
    output_epi_marker = 1
    markers = beat.utils.expand_layer_biv(
        V=V,
        ft=ft,
        endo_lv_marker=endo_lv_marker,
        endo_rv_marker=endo_rv_marker,
        epi_marker=epi_marker,
        endo_size=0.3,
        epi_size=0.3,
        output_mid_marker=output_mid_marker,
        output_endo_marker=output_endo_marker,
        output_epi_marker=output_epi_marker,
    )

    endo_points = np.array([(0.0, 0.0), (0.0, 1.0), (0.2, 0.2), (0.2, 0.8)])
    mid_points = np.array(
        [(0.5 + i, 0.5 + j) for i in [-0.1, 0.0, 0.1] for j in [-0.1, 0.0, 0.1]] + [(0.0, 0.5)],
    )
    epi_points = np.array([(1.0, 0.0), (1.0, 1.0), (0.8, 0.2), (0.8, 0.8)])

    endo = scifem.evaluate_function(markers, endo_points)
    assert np.allclose(endo, output_endo_marker)

    mid = scifem.evaluate_function(markers, mid_points)
    assert np.allclose(mid, output_mid_marker)

    epi = scifem.evaluate_function(markers, epi_points)
    assert np.allclose(epi, output_epi_marker)
