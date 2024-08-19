import dolfinx
import numpy as np
import pytest
from mpi4py import MPI
import ufl

import beat


def v_exact_func(x, t):
    return ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t)


def s_exact_func(x, t):
    return -ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.cos(t)


def ac_func(x, t):
    return (
        8
        * ufl.pi**2
        * ufl.cos(2 * ufl.pi * x[0])
        * ufl.cos(2 * ufl.pi * x[1])
        * ufl.sin(t)
    )


def simple_ode_forward_euler(states, t, dt, parameters):
    v, s = states
    values = np.zeros_like(states)
    values[0] = v - s * dt
    values[1] = s + v * dt
    return values


@pytest.mark.parametrize(
    "odespace",
    [
        "P_1",
        "P_2",
        "DG_0",
        "DG_1",
        "Quadrature_2",
        # "Quadrature_4",  # Currently not supported
    ],
)
def test_monodomain_splitting_analytic(odespace):
    N = 50

    M = 1.0
    # T = 4.0
    dt = 0.01
    T = dt
    t0 = 0.0

    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(
        comm, N, N, dolfinx.cpp.mesh.CellType.triangle
    )
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    x = ufl.SpatialCoordinate(mesh)

    t_var = ufl.variable(time)

    s_exact = ufl.replace(s_exact_func(x, t_var), {t_var: T})
    I_s = ac_func(x, t_var)

    pde = beat.MonodomainModel(time=time, mesh=mesh, M=M, I_s=I_s)

    V_ode = beat.utils.space_from_string(odespace, mesh, dim=1)
    v_ode = dolfinx.fem.Function(V_ode)

    s = dolfinx.fem.Function(V_ode)
    s.interpolate(dolfinx.fem.Expression(s_exact, V_ode.element.interpolation_points()))

    s_arr = s.x.array
    init_states = np.zeros((2, s_arr.size))
    init_states[1, :] = s_arr

    ode = beat.odesolver.DolfinODESolver(
        v_ode=v_ode,
        v_pde=pde.state,
        fun=simple_ode_forward_euler,
        init_states=init_states,
        parameters=None,
        num_states=2,
        v_index=0,
    )
    solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)
    solver.solve((t0, T), dt=dt)

    v_exact = dolfinx.fem.Function(V_ode)
    v_exact.interpolate(
        dolfinx.fem.Expression(
            ufl.replace(v_exact_func(x, t_var), {t_var: T}),
            V_ode.element.interpolation_points(),
        )
    )

    error = dolfinx.fem.form((pde.state - v_exact) ** 2 * ufl.dx)
    E = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(error), MPI.SUM))
    assert E < 0.001


@pytest.mark.parametrize(
    "odespace",
    [
        "CG_1",
        "CG_2",
        "DG_0",
        "DG_1",
        "Quadrature_2",
    ],
)
def test_monodomain_splitting_spatial_convergence(odespace, caplog):

    # caplog.set_level(logging.INFO)
    # family = "Lagrange"
    # degree = 1

    M = 1.0
    dt = 0.01
    T = 1.0
    t0 = 0.0

    comm = MPI.COMM_WORLD
    errors = []
    Ns = [2**level for level in (3, 4, 5)]

    for N in Ns:
        mesh = dolfinx.mesh.create_unit_square(
            comm, N, N, dolfinx.cpp.mesh.CellType.triangle
        )
        time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
        x = ufl.SpatialCoordinate(mesh)

        t_var = ufl.variable(time)

        s_exact = ufl.replace(s_exact_func(x, t_var), {t_var: T})
        I_s = ac_func(x, t_var)

        pde = beat.MonodomainModel(time=time, mesh=mesh, M=M, I_s=I_s)

        V_ode = beat.utils.space_from_string(odespace, mesh, dim=1)
        v_ode = dolfinx.fem.Function(V_ode)

        s = dolfinx.fem.Function(V_ode)
        s.interpolate(
            dolfinx.fem.Expression(s_exact, V_ode.element.interpolation_points())
        )

        s_arr = s.x.array
        init_states = np.zeros((2, s_arr.size))
        init_states[1, :] = s_arr

        ode = beat.odesolver.DolfinODESolver(
            v_ode=v_ode,
            v_pde=pde.state,
            fun=simple_ode_forward_euler,
            init_states=init_states,
            parameters=None,
            num_states=2,
            v_index=0,
        )
        solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=0.5)
        solver.solve((t0, T), dt=dt)

        v_exact = dolfinx.fem.Function(V_ode)
        v_exact.interpolate(
            dolfinx.fem.Expression(
                ufl.replace(v_exact_func(x, t_var), {t_var: T}),
                V_ode.element.interpolation_points(),
            )
        )

        error = dolfinx.fem.form((pde.state - v_exact) ** 2 * ufl.dx)
        E = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(error), MPI.SUM))
        errors.append(E)

    rates = [np.log(e1 / e2) / np.log(2) for e1, e2 in zip(errors[:-1], errors[1:])]
    cvg_rate = sum(rates) / len(rates)

    assert np.isclose(cvg_rate, 2, rtol=0.15)


@pytest.mark.parametrize(
    "odespace",
    [
        "CG_1",
        "CG_2",
        "DG_0",
        "DG_1",
        "Quadrature_2",
        "Quadrature_4",
    ],
)
def test_monodomain_splitting_temporal_convergence(odespace, caplog):

    M = 1.0
    T = 1.0
    t0 = 0.0

    comm = MPI.COMM_WORLD

    N = 150
    mesh = dolfinx.mesh.create_unit_square(
        comm, N, N, dolfinx.cpp.mesh.CellType.triangle
    )
    V_ode = beat.utils.space_from_string(odespace, mesh, dim=1)
    v_ode = dolfinx.fem.Function(V_ode)

    errors = []

    dts = [1.0 / (2**level) for level in (2, 3, 4)]
    for dt in dts:

        time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
        x = ufl.SpatialCoordinate(mesh)

        t_var = ufl.variable(time)

        s_exact = ufl.replace(s_exact_func(x, t_var), {t_var: T})
        I_s = ac_func(x, t_var)

        pde = beat.MonodomainModel(time=time, mesh=mesh, M=M, I_s=I_s)

        s = dolfinx.fem.Function(V_ode)
        s.interpolate(
            dolfinx.fem.Expression(s_exact, V_ode.element.interpolation_points())
        )

        s_arr = s.x.array
        init_states = np.zeros((2, s_arr.size))
        init_states[1, :] = s_arr

        ode = beat.odesolver.DolfinODESolver(
            v_ode=v_ode,
            v_pde=pde.state,
            fun=simple_ode_forward_euler,
            init_states=init_states,
            parameters=None,
            num_states=2,
            v_index=0,
        )
        solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=0.5)
        solver.solve((t0, T), dt=dt)

        v_exact = dolfinx.fem.Function(V_ode)
        v_exact.interpolate(
            dolfinx.fem.Expression(
                ufl.replace(v_exact_func(x, t_var), {t_var: T}),
                V_ode.element.interpolation_points(),
            )
        )

        error = dolfinx.fem.form((pde.state - v_exact) ** 2 * ufl.dx)
        E = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(error), MPI.SUM))
        errors.append(E)

    rates = [np.log(e1 / e2) / np.log(2) for e1, e2 in zip(errors[:-1], errors[1:])]
    cvg_rate = sum(rates) / len(rates)

    assert np.isclose(cvg_rate, 2, rtol=0.15)
