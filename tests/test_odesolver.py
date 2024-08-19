import numpy as np

# import dolfinx
# from mpi4py import MPI

# import beat
from beat.odesolver import ODESystemSolver  # , DolfinODESolver, DolfinMultiODESolver


def test_simple_ode_odesystemsolver():
    def simple_ode_forward_euler(states, t, dt, parameters):
        v, s = states
        values = np.zeros_like(states)
        values[0] = v - s * dt
        values[1] = s + v * dt
        return values

    num_points = 1

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
            parameters=None,
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


# def test_beeler_reuter_odesystemsolver():
#     model = beat.cellmodels.beeler_reuter_1977
#     num_points = 10
#     init_states = model.init_state_values()
#     parameters = model.init_parameter_values()
#     parameters[model.parameter_indices("IstimAmplitude")] = 1.0
#     num_states = len(init_states)
#     states = np.zeros((num_states, num_points))
#     states.T[:] = init_states
#     dt = 0.1
#     t0 = 0.0
#     old_states = np.copy(states)

#     ode = ODESystemSolver(
#         fun=beat.cellmodels.beeler_reuter_1977.forward_generalized_rush_larsen,
#         states=states,
#         parameters=parameters,
#     )
#     assert np.allclose(ode.states, old_states)

#     ode.step(t0, dt)

#     assert not np.allclose(ode.states, old_states)


# def test_beeler_reuter_unit_square():
#     model = beat.cellmodels.beeler_reuter_1977
#     init_states = model.init_state_values()
#     parameters = model.init_parameter_values()
#     parameters[model.parameter_indices("IstimAmplitude")] = 1.0

#     comm = MPI.COMM_WORLD
#     mesh = mesh = dolfinx.mesh.create_unit_square(
#         comm, 5, 5, dolfinx.cpp.mesh.CellType.triangle
#     )
#     V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
#     s = dolfinx.fem.Function(V)
#     dt = 0.1
#     t0 = 0.0

#     v_index = beat.cellmodels.beeler_reuter_1977.state_indices("V")

#     dolfin_ode = DolfinODESolver(
#         v_ode=dolfinx.fem.Function(V),
#         v_pde=s,
#         num_states=len(init_states),
#         fun=beat.cellmodels.beeler_reuter_1977.forward_generalized_rush_larsen,
#         init_states=init_states,
#         parameters=parameters,
#         v_index=v_index,
#     )
#     assert np.allclose(dolfin_ode.v_ode.x.array, 0.0)
#     dolfin_ode.to_dolfin()
#     assert np.allclose(dolfin_ode.v_ode.x.array, init_states[v_index])
#     dolfin_ode.ode_to_pde()

#     # Just check that values have been updated
#     assert np.allclose(dolfin_ode.v_pde.x.array, init_states[v_index])

#     N = 10
#     t = t0
#     for _ in range(N):
#         dolfin_ode.step(t, dt)
#         t += dt

#     dolfin_ode.to_dolfin()
#     dolfin_ode.ode_to_pde()
#     # Just check that values have been updated
#     assert not np.allclose(dolfin_ode.v_pde.x.array, init_states[v_index])


# def test_assignment_ode():
#     model = beat.cellmodels.beeler_reuter_1977
#     init_states = model.init_state_values()
#     parameters = model.init_parameter_values()
#     parameters[model.parameter_indices("IstimAmplitude")] = 1.0
#     v_index = model.state_indices("V")

#     comm = MPI.COMM_WORLD
#     mesh = mesh = dolfinx.mesh.create_unit_square(
#         comm, 5, 5, dolfinx.cpp.mesh.CellType.triangle
#     )
#     V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
#     s = dolfinx.fem.Function(V)

#     ode = DolfinODESolver(
#         v_ode=dolfinx.fem.Function(V),
#         v_pde=s,
#         num_states=len(init_states),
#         fun=beat.cellmodels.beeler_reuter_1977.forward_generalized_rush_larsen,
#         init_states=init_states,
#         parameters=parameters,
#         v_index=v_index,
#     )

#     assert np.allclose(ode.v_ode.x.array, 0)
#     assert np.allclose(ode.values[:, 0], init_states)

#     ode.to_dolfin()
#     ode.ode_to_pde()
#     assert np.allclose(ode.v_pde.x.array, init_states[v_index])

#     # Now update values for v
#     ode.values[v_index, :] = 42.0
#     assert np.allclose(ode.v_ode.x.array, init_states[v_index])
#     ode.to_dolfin()
#     ode.ode_to_pde()
#     assert np.allclose(ode.v_pde.x.array, 42.0)

#     # Now update dolfin function for v
#     ode.v_pde.x.array[:] = 13.0

#     ode.pde_to_ode()
#     ode.from_dolfin()

#     assert np.allclose(ode.values[v_index, :], 13.0)
#     assert np.allclose(ode.full_values[v_index, :], 13.0)
