import gc

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

from beat.irksome_model import IrksomeMonodomainModel

irksome = pytest.importorskip("irksome")


@pytest.mark.parametrize(
    "M, ac_str, exact, err",
    (
        (
            0.0,
            lambda x, t: ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.cos(t),
            lambda x, t: ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t),
            1e-4,
        ),
        (
            1.0,
            lambda x, t: ufl.cos(2 * ufl.pi * x[0])
            * ufl.cos(2 * ufl.pi * x[1])
            * (ufl.cos(t) + 8 * pow(ufl.pi, 2) * ufl.sin(t)),
            lambda x, t: ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t),
            2e-4,
        ),
        (
            2.0,
            lambda x, t: ufl.cos(2 * ufl.pi * x[0])
            * ufl.cos(2 * ufl.pi * x[1])
            * (ufl.cos(t) + 16 * pow(ufl.pi, 2) * ufl.sin(t)),
            lambda x, t: ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t),
            2e-4,
        ),
    ),
)
def test_irksome_monodomain_analytic(M, ac_str, exact, err):
    N = 15

    dt = 0.001
    T = 10 * dt

    params = dict(petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    x = ufl.SpatialCoordinate(mesh)
    t_var = ufl.variable(time)
    I_s = ac_str(x, t_var)

    # Use a 2-stage Radau IIA tableau
    tableau = irksome.RadauIIA(2)

    model = IrksomeMonodomainModel(
        time=time,
        mesh=mesh,
        M=M,
        butcher_tableau=tableau,
        I_s=I_s,
        params=params,
    )
    res = model.solve((0, T), dt=dt)

    v_exact = ufl.replace(exact(x, t_var), {t_var: T})
    diff = res.state - v_exact
    metadata = {"quadrature_degree": 8}
    error = dolfinx.fem.form(
        ufl.inner(diff, diff) * ufl.dx(domain=mesh, metadata=metadata),
    )
    v_error = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error), op=MPI.SUM))

    assert v_error < err

    # Ensure all processes have finished before cleaning up the solver
    mesh.comm.Barrier()
    del model
    mesh.comm.Barrier()
    gc.collect()
    mesh.comm.Barrier()


def test_irksome_monodomain_spatial_convergence():
    Ns = [2**level for level in (2, 3, 4, 5)]

    errors = []
    comm = MPI.COMM_WORLD
    dt = 0.001
    T = 10 * dt
    metadata = {"quadrature_degree": 8}
    params = dict(petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    for N in Ns:
        mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)
        time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
        x = ufl.SpatialCoordinate(mesh)
        t = ufl.variable(time)
        I_s = (
            ufl.cos(2 * ufl.pi * x[0])
            * ufl.cos(2 * ufl.pi * x[1])
            * (ufl.cos(t) + 8 * pow(ufl.pi, 2) * ufl.sin(t))
        )
        tableau = irksome.RadauIIA(2)
        model = IrksomeMonodomainModel(
            time=time,
            mesh=mesh,
            M=1.0,
            butcher_tableau=tableau,
            I_s=I_s,
            params=params,
        )

        v_exact = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t)
        v_exact = ufl.replace(v_exact, {t: T})

        res = model.solve((0, T), dt=dt)

        diff = res.state - v_exact

        error = dolfinx.fem.form(
            ufl.inner(diff, diff) * ufl.dx(domain=mesh, metadata=metadata),
        )
        v_error = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error), op=MPI.SUM)

        errors.append(np.sqrt(v_error))

        # IMPORTANT: Synchronize and force garbage collection before
        # moving to the next iteration to avoid MPI deadlocks in PETSc
        mesh.comm.Barrier()
        del model
        mesh.comm.Barrier()
        gc.collect()
        mesh.comm.Barrier()

    rates = [np.log(e1 / e2) / np.log(2) for e1, e2 in zip(errors[:-1], errors[1:])]
    assert all(rate >= 2.0 for rate in rates)


# @pytest.mark.skip_in_parallel
@pytest.mark.parametrize(
    "num_stages, expected_rate",
    (
        (1, 1.9),  # GaussLegendre(1) is 2nd order (Implicit Midpoint)
        (2, 3.8),  # GaussLegendre(2) is 4th order
    ),
)
def test_irksome_monodomain_higher_order_temporal_convergence(num_stages, expected_rate):
    comm = MPI.COMM_WORLD
    T = 1.0

    # We use a coarser mesh but a higher polynomial degree (degree=3)
    # to ensure spatial error doesn't bottleneck the 4th-order temporal convergence.
    N = 25
    metadata = {"quadrature_degree": 8}

    # Pass degree=3 to use higher-order spatial elements
    params = dict(degree=3, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)
    x = ufl.SpatialCoordinate(mesh)

    v_exact = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(T)

    errors = []
    dts = [1.0 / (2**level) for level in (0, 1, 2, 3)]

    for dt in dts:
        time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
        t = ufl.variable(time)

        I_s = (
            ufl.cos(2 * ufl.pi * x[0])
            * ufl.cos(2 * ufl.pi * x[1])
            * (ufl.cos(t) + 8 * pow(ufl.pi, 2) * ufl.sin(t))
        )

        # Select the higher-order Gauss-Legendre method
        tableau = irksome.GaussLegendre(num_stages)

        model = IrksomeMonodomainModel(
            time=time,
            mesh=mesh,
            M=1.0,
            butcher_tableau=tableau,
            I_s=I_s,
            params=params,
        )

        res = model.solve((0, T), dt=dt)

        diff = res.state - v_exact

        error = dolfinx.fem.form(
            ufl.inner(diff, diff) * ufl.dx(domain=mesh, metadata=metadata),
        )
        v_error = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error), op=MPI.SUM)

        errors.append(np.sqrt(v_error))

        # Explicit garbage collection to prevent MPI deadlocks
        mesh.comm.Barrier()
        del model
        mesh.comm.Barrier()
        gc.collect()
        mesh.comm.Barrier()

    # Calculate convergence rates between consecutive dt refinements
    rates = [np.log(e1 / e2) / np.log(2) for e1, e2 in zip(errors[:-1], errors[1:])]
    # Assert that the final computed rate (asymptotic regime) meets the expected theoretical order
    assert rates[-1] >= expected_rate, f"Failed for {num_stages} stages: rates {rates}"
