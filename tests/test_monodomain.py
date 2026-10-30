from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import beat


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
            lambda x, t: (
                ufl.cos(2 * ufl.pi * x[0])
                * ufl.cos(2 * ufl.pi * x[1])
                * (ufl.cos(t) + 8 * pow(ufl.pi, 2) * ufl.sin(t))
            ),
            lambda x, t: ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t),
            2e-4,
        ),
        (
            2.0,
            lambda x, t: (
                ufl.cos(2 * ufl.pi * x[0])
                * ufl.cos(2 * ufl.pi * x[1])
                * (ufl.cos(t) + 16 * pow(ufl.pi, 2) * ufl.sin(t))
            ),
            lambda x, t: ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t),
            2e-4,
        ),
    ),
)
def test_monodomain_analytic(M, ac_str, exact, err):
    N = 15

    theta = 0.5
    dt = 0.001
    T = 10 * dt

    params = dict(theta=theta)
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    x = ufl.SpatialCoordinate(mesh)
    t_var = ufl.variable(time)
    I_s = ac_str(x, t_var)

    model = beat.MonodomainModel(time=time, mesh=mesh, M=M, I_s=I_s, params=params)
    res = model.solve((0, T), dt=dt)

    v_exact = ufl.replace(exact(x, t_var), {t_var: T})
    diff = res.state - v_exact
    metadata = {"quadrature_degree": 8}
    error = dolfinx.fem.form(
        ufl.inner(diff, diff) * ufl.dx(domain=mesh, metadata=metadata),
    )
    v_error = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error), op=MPI.SUM))

    assert v_error < err


def test_monodomain_spatial_convergence():
    Ns = [2**level for level in (2, 3, 4, 5)]

    errors = []
    comm = MPI.COMM_WORLD
    theta = 0.5
    dt = 0.001
    T = 10 * dt
    metadata = {"quadrature_degree": 8}
    params = dict(theta=theta)  # , default_timestep=dt)
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
        model = beat.MonodomainModel(time=time, mesh=mesh, M=1.0, I_s=I_s, params=params)

        v_exact = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t)
        v_exact = ufl.replace(v_exact, {t: T})

        res = model.solve((0, T), dt=dt)

        diff = res.state - v_exact

        error = dolfinx.fem.form(
            ufl.inner(diff, diff) * ufl.dx(domain=mesh, metadata=metadata),
        )
        v_error = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error), op=MPI.SUM)

        errors.append(np.sqrt(v_error))

    rates = [np.log(e1 / e2) / np.log(2) for e1, e2 in zip(errors[:-1], errors[1:])]
    assert all(rate >= 2.0 for rate in rates)


@pytest.mark.skip_in_parallel
def test_monodomain_temporal_convergence():
    errors = []
    comm = MPI.COMM_WORLD
    theta = 0.5
    T = 1.0
    N = 100

    metadata = {"quadrature_degree": 8}
    params = dict(theta=theta)  # , default_timestep=dt)
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
        model = beat.MonodomainModel(time=time, mesh=mesh, M=1.0, I_s=I_s, params=params)

        res = model.solve((0, T), dt=dt)

        diff = res.state - v_exact

        error = dolfinx.fem.form(
            ufl.inner(diff, diff) * ufl.dx(domain=mesh, metadata=metadata),
        )
        v_error = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error), op=MPI.SUM)

        errors.append(np.sqrt(v_error))

    rates = [np.log(e1 / e2) / np.log(2) for e1, e2 in zip(errors[:-1], errors[1:])]
    assert all(rate >= 2.0 for rate in rates)


@pytest.mark.parametrize("theta", (0.5, 1.0))
@pytest.mark.parametrize("stimulus", (True, False))
def test_monodomain_rhs_matrix_matches_assembled_rhs(theta, stimulus):
    """B v_ + L_0 is the assembled right-hand side, also after the time step changes"""
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 12, 12, dolfinx.cpp.mesh.CellType.triangle)
    x = ufl.SpatialCoordinate(mesh)
    # Anisotropic and varying in space, as a conductivity tensor built from fibres is
    M = ufl.as_matrix(((2.0 + x[0], 0.3 * x[1]), (0.3 * x[1], 0.5)))

    def run(rhs_matrix):
        time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
        I_s = ufl.sin(3 * x[0]) * ufl.cos(time) if stimulus else None
        model = beat.MonodomainModel(
            time=time,
            mesh=mesh,
            M=M,
            I_s=I_s,
            C_m=1.5,
            params=dict(theta=theta, rhs_matrix=rhs_matrix),
        )
        model.v_.interpolate(lambda x: np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]))
        model.solve((0.0, 0.05), dt=0.01)
        model.assign_previous()
        model.solve((0.05, 0.1), dt=0.025)
        return model.v.x.array.copy()

    v_matrix = run(rhs_matrix=True)
    v_assembled = run(rhs_matrix=False)
    assert np.linalg.norm(v_assembled) > 0.1
    assert np.allclose(v_matrix, v_assembled, rtol=1e-12, atol=1e-12)
