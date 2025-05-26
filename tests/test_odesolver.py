from mpi4py import MPI

import dolfinx
import numpy as np
import ufl

from beat.odesolver import DolfinMultiODESolver, DolfinODESolver, ODESystemSolver


def simple_ode_forward_euler(states, t, dt, parameters):
    v, s = states
    a, b = parameters
    values = np.zeros_like(states)
    values[0] = v - a * s * dt
    values[1] = s + b * v * dt
    return values


def test_simple_ode_odesystemsolver():
    num_points = 1
    parameters = np.array([1, 1])
    t_bound = 1.0
    t0 = 0.0
    x = np.arange(0.1, t_bound + 0.1, 0.1)
    y = np.zeros((len(x), 2))
    sol = np.vstack((np.cos(x), np.sin(x))).T

    errors = []
    for dt in [0.1, 0.01, 0.001, 0.0001]:
        states = np.zeros((2, num_points))
        states.T[:] = [1, 0]
        ode = ODESystemSolver(
            fun=simple_ode_forward_euler,
            states=states,
            parameters=parameters,
        )
        j = 0
        t = 0.0
        for _ in range(int((t_bound - t0) / dt)):
            ode.step(t, dt)
            t += dt
            if np.isclose(t, x[j]):
                print(t, j)
                y[j, :] = ode.states[:, 0]
                j += 1
        errors.append(np.linalg.norm(sol - y))
    rates = [np.log(e1 / e2) / np.log(10) for e1, e2 in zip(errors[:-1], errors[1:])]
    assert np.allclose(rates, 1, atol=0.01)


def test_DolfinODESolver():
    comm = MPI.COMM_WORLD
    N = 5
    mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)

    V_pde = dolfinx.fem.functionspace(mesh, ("P", 1))
    v_pde = dolfinx.fem.Function(V_pde)

    V_ode = dolfinx.fem.functionspace(mesh, ("P", 1))
    v_ode = dolfinx.fem.Function(V_ode)

    N_ode = V_ode.dofmap.index_map.size_local + V_ode.dofmap.index_map.num_ghosts

    v0 = 1.0
    s0 = 2.0
    init_states = np.array([v0, s0])
    parameters = np.array([1, 1])
    ode = DolfinODESolver(
        v_ode=v_ode,
        v_pde=v_pde,
        init_states=init_states,
        parameters=parameters,
        fun=simple_ode_forward_euler,
        num_states=2,
        v_index=0,  # The is the index in the state vector that corresponds to PDE variable
    )

    assert ode.full_values.shape == (2, N_ode)
    assert ode.values.shape == (2, N_ode)
    # First the values should be the initial state
    assert np.allclose(ode.values[0, :], v0)
    assert np.allclose(ode.values[1, :], s0)

    # Now we step the odes
    dt = 0.1
    ode.step(0.0, dt)
    # And the values should be updated
    assert np.allclose(ode.values[0, :], v0 - s0 * dt)
    assert np.allclose(ode.values[1, :], s0 + v0 * dt)
    # However, the dolfin function should not be updated
    assert np.allclose(v_ode.x.array, 0.0)
    # We can assign the values to the dolfin function
    ode.to_dolfin()
    # And we check that the values are updated
    assert np.allclose(v_ode.x.array, v0 - s0 * dt)
    # However, not for the PDE function
    assert np.allclose(v_pde.x.array, 0.0)
    # Which is updated by the next method
    ode.ode_to_pde()
    assert np.allclose(v_pde.x.array, v0 - s0 * dt)
    # Now let us update the pde
    v_pde.x.array[:] = 1.0
    # And transfer the values back
    ode.pde_to_ode()
    assert np.allclose(v_ode.x.array, 1.0)
    # Finally let us transfer the values back to the ODE values
    ode.from_dolfin()
    assert np.allclose(ode.values[0, :], 1.0)
    # The s value should be the same as before
    assert np.allclose(ode.values[1, :], s0 + v0 * dt)

    # Finally test states_to dolfin
    states = ode.states_to_dolfin()
    assert len(states) == 2
    np.allclose(states[0].x.array, 1.0)
    np.allclose(states[1].x.array, v0 - s0 * dt)


def test_DolfinMultiODESolver():
    comm = MPI.COMM_WORLD
    N = 5
    mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)

    V_pde = dolfinx.fem.functionspace(mesh, ("P", 1))
    v_pde = dolfinx.fem.Function(V_pde)

    V_ode = dolfinx.fem.functionspace(mesh, ("P", 1))
    v_ode = dolfinx.fem.Function(V_ode)

    markers = dolfinx.fem.Function(V_ode)
    X = ufl.SpatialCoordinate(mesh)
    expr = ufl.conditional(ufl.lt(X[0], 0.5), 1, 2)
    markers.interpolate(dolfinx.fem.Expression(expr, V_ode.element.interpolation_points()))

    first_v0 = 1.0
    first_s0 = 2.0
    second_v0 = 3.0
    second_s0 = 4.0
    init_states = {1: np.array([first_v0, first_s0]), 2: np.array([second_v0, second_s0])}

    first_p0 = 1
    second_p0 = 2
    parameters = {1: np.array([first_p0, first_p0]), 2: np.array([second_p0, second_p0])}

    N_ode = V_ode.dofmap.index_map.size_local + V_ode.dofmap.index_map.num_ghosts

    ode = DolfinMultiODESolver(
        v_ode=v_ode,
        v_pde=v_pde,
        markers=markers,
        init_states=init_states,
        parameters=parameters,
        fun={1: simple_ode_forward_euler, 2: simple_ode_forward_euler},
        num_states={i: len(s) for i, s in init_states.items()},
        v_index={i: 0 for i in init_states.keys()},
    )

    assert ode.full_values.shape == (2, N_ode)
    assert ode.values(1).shape == (2, (markers.x.array == 1).sum())
    assert ode.values(2).shape == (2, (markers.x.array == 2).sum())
    # First the values should be the initial state
    assert np.allclose(ode.values(1)[0, :], first_v0)
    assert np.allclose(ode.values(1)[1, :], first_s0)
    assert np.allclose(ode.values(2)[0, :], second_v0)
    assert np.allclose(ode.values(2)[1, :], second_s0)

    # Now we step the odes
    dt = 0.1
    ode.step(0.0, dt)
    # And the values should be updated
    assert np.allclose(ode.values(1)[0, :], first_v0 - first_p0 * first_s0 * dt)
    assert np.allclose(ode.values(1)[1, :], first_s0 + first_p0 * first_v0 * dt)
    assert np.allclose(ode.values(2)[0, :], second_v0 - second_p0 * second_s0 * dt)
    assert np.allclose(ode.values(2)[1, :], second_s0 + second_p0 * second_v0 * dt)

    # However, the dolfin function should not be updated
    assert np.allclose(v_ode.x.array, 0.0)
    # We can assign the values to the dolfin function
    ode.to_dolfin()
    # And we check that the values are updated
    assert np.allclose(v_ode.x.array[markers.x.array == 1], first_v0 - first_p0 * first_s0 * dt)
    assert np.allclose(v_ode.x.array[markers.x.array == 2], second_v0 - second_p0 * second_s0 * dt)
    # However, not for the PDE function
    assert np.allclose(v_pde.x.array, 0.0)
    # Which is updated by the next method
    ode.ode_to_pde()
    assert np.allclose(v_pde.x.array[markers.x.array == 1], first_v0 - first_p0 * first_s0 * dt)
    assert np.allclose(v_pde.x.array[markers.x.array == 2], second_v0 - second_p0 * second_s0 * dt)
    # Now let us update the pde
    v_pde.x.array[:] = 1.0
    # And transfer the values back
    ode.pde_to_ode()
    assert np.allclose(v_ode.x.array, 1.0)
    # Finally let us transfer the values back to the ODE values
    ode.from_dolfin()
    assert np.allclose(ode.values(1)[0, :], 1.0)
    assert np.allclose(ode.values(2)[0, :], 1.0)
    # The s value should be the same as before
    assert np.allclose(ode.values(1)[1, :], first_s0 + first_p0 * first_v0 * dt)
    assert np.allclose(ode.values(2)[1, :], second_s0 + second_p0 * second_v0 * dt)

    # Finally test states_to dolfin
    states = ode.states_to_dolfin()
    assert len(states) == 2
    np.allclose(states[0].x.array[markers.x.array == 1], 1.0)
    np.allclose(states[0].x.array[markers.x.array == 2], 1.0)
    np.allclose(states[1].x.array[markers.x.array == 1], first_v0 - first_p0 * first_s0 * dt)
    np.allclose(states[1].x.array[markers.x.array == 2], second_v0 - second_p0 * second_s0 * dt)
