import gc

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import beat.utils
from beat.external_operator_odesolver import (
    ExternalOperatorMultiODESolver,
    ExternalOperatorODESolver,
)

# Skip all tests in this file if dolfinx-external-operator is not installed
pytest.importorskip("dolfinx_external_operator")


def simple_ode_forward_euler(t, states, dt, parameters):
    """
    Forward Euler step of the simple harmonic oscillator:
    v' = -a * s
    s' = b * v

    Deliberately uses a *different* positional parameter order than
    beat.odesolver.ODESystemSolver's own `fun(states, t, parameters, dt)` docstring
    convention, to prove that ExternalOperatorODESolver/ExternalOperatorMultiODESolver
    call `fun` by keyword (states=, t=, parameters=, dt=) exactly like
    beat.odesolver.ODESystemSolver does, rather than relying on a fixed argument order.
    """
    v, s = states
    a, b = parameters
    values = np.zeros_like(states)
    values[0] = v - a * s * dt
    values[1] = s + b * v * dt
    return values


def test_external_operator_odesolver_mappings():
    """Test that data maps correctly between the ODE state function and the PDEs."""
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
    init_states = np.zeros((2, N_ode))
    init_states[0, :] = v0
    init_states[1, :] = s0

    ode = ExternalOperatorODESolver(
        v_ode=v_ode,
        v_pde=v_pde,
        fun=simple_ode_forward_euler,
        init_states=init_states,
        num_states=2,
        v_index=0,
        parameters=np.array([1.0, 1.0]),
    )

    # 1. Initial state assignment verification
    assert ode.full_values.shape == (2, N_ode)
    assert np.allclose(ode.full_values[0, :], v0)
    assert np.allclose(ode.full_values[1, :], s0)

    # 2. Step the ODEs using forward Euler
    dt = 0.1
    ode.step(0.0, dt)
    v1_exact = v0 - s0 * dt
    s1_exact = s0 + v0 * dt
    assert np.allclose(ode.full_values[0, :], v1_exact)
    assert np.allclose(ode.full_values[1, :], s1_exact)

    # 3. Check mapping: the standard dolfin Function should not be updated automatically
    assert np.allclose(v_ode.x.array, 0.0)

    # Check to_dolfin() pushes the ODE state -> v_ode
    ode.to_dolfin()
    assert np.allclose(v_ode.x.array, v1_exact)
    assert np.allclose(v_pde.x.array, 0.0)

    # Check ode_to_pde() local projection
    ode.ode_to_pde()
    assert np.allclose(v_pde.x.array, v1_exact)

    # 4. Check reverse mapping: modifying PDE and pulling back to the ODE state
    v_pde.x.array[:] = 5.0
    ode.pde_to_ode()
    assert np.allclose(v_ode.x.array, 5.0)

    ode.from_dolfin()
    assert np.allclose(ode.full_values[0, :], 5.0)
    assert np.allclose(ode.full_values[1, :], s1_exact)  # s state remains untouched

    # 5. Extract all states to separate functions
    states = ode.states_to_dolfin()
    assert len(states) == 2
    assert np.allclose(states[0].x.array, 5.0)
    assert np.allclose(states[1].x.array, s1_exact)

    # Prevent MPI deadlocks during destruction
    mesh.comm.Barrier()
    del ode
    mesh.comm.Barrier()
    gc.collect()
    mesh.comm.Barrier()


def test_external_operator_odesolver_temporal_convergence():
    """Test that ExternalOperatorODESolver converges at the expected (1st, forward Euler)
    rate, and matches beat's own ODESystemSolver."""
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 1, 1, dolfinx.cpp.mesh.CellType.triangle)

    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    v_ode = dolfinx.fem.Function(V)
    v_pde = dolfinx.fem.Function(V)
    N_ode = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts

    T = 1.0
    errors = []
    dts = [0.05, 0.025, 0.0125]

    for dt in dts:
        init_states = np.zeros((2, N_ode))
        init_states[0, :] = 1.0  # v(0) = 1.0
        init_states[1, :] = 0.0  # s(0) = 0.0

        ode = ExternalOperatorODESolver(
            v_ode=v_ode,
            v_pde=v_pde,
            fun=simple_ode_forward_euler,
            init_states=init_states,
            num_states=2,
            v_index=0,
            parameters=np.array([1.0, 1.0]),
        )

        t = 0.0
        while t < T - 1e-8:
            ode.step(t, dt)
            t += dt

        # The exact solution for this system is v(t) = cos(t), s(t) = sin(t)
        v_exact = np.cos(T)
        error = np.max(np.abs(ode.full_values[0, :] - v_exact))
        global_error = comm.allreduce(error, op=MPI.MAX)
        errors.append(global_error)

        mesh.comm.Barrier()
        del ode
        mesh.comm.Barrier()
        gc.collect()
        mesh.comm.Barrier()

    rates = [
        np.log(e1 / e2) / np.log(dts[i] / dts[i + 1])
        for i, (e1, e2) in enumerate(zip(errors[:-1], errors[1:]))
    ]
    assert all(0.9 < rate < 1.15 for rate in rates), f"Expected 1st order convergence, got {rates}"


def test_external_operator_multi_odesolver_mappings():
    """Test that data maps correctly for a two-region ExternalOperatorMultiODESolver,
    mirroring test_external_operator_odesolver_mappings (single region) and
    test_DolfinMultiODESolver (plain-numpy multi region)."""
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
    markers.interpolate(dolfinx.fem.Expression(expr, beat.utils.interpolation_points(V_ode)))

    N_ode = V_ode.dofmap.index_map.size_local + V_ode.dofmap.index_map.num_ghosts

    first_v0, first_s0 = 1.0, 2.0
    second_v0, second_s0 = 3.0, 4.0
    init_states = {
        1: np.array([first_v0, first_s0]),
        2: np.array([second_v0, second_s0]),
    }
    first_p0, second_p0 = 1.0, 2.0
    parameters = {
        1: np.array([first_p0, first_p0]),
        2: np.array([second_p0, second_p0]),
    }

    ode = ExternalOperatorMultiODESolver(
        v_ode=v_ode,
        v_pde=v_pde,
        markers=markers,
        fun={1: simple_ode_forward_euler, 2: simple_ode_forward_euler},
        init_states=init_states,
        num_states={1: 2, 2: 2},
        v_index={1: 0, 2: 0},
        parameters=parameters,
    )

    # 1. Initial state assignment verification
    assert ode.full_values.shape == (2, N_ode)
    assert ode.values(1).shape == (2, int((markers.x.array == 1).sum()))
    assert ode.values(2).shape == (2, int((markers.x.array == 2).sum()))
    assert np.allclose(ode.values(1)[0, :], first_v0)
    assert np.allclose(ode.values(1)[1, :], first_s0)
    assert np.allclose(ode.values(2)[0, :], second_v0)
    assert np.allclose(ode.values(2)[1, :], second_s0)

    # 2. Step the ODEs using forward Euler
    dt = 0.1
    ode.step(0.0, dt)

    v1_first = first_v0 - first_p0 * first_s0 * dt
    s1_first = first_s0 + first_p0 * first_v0 * dt
    v1_second = second_v0 - second_p0 * second_s0 * dt
    s1_second = second_s0 + second_p0 * second_v0 * dt

    assert np.allclose(ode.values(1)[0, :], v1_first)
    assert np.allclose(ode.values(1)[1, :], s1_first)
    assert np.allclose(ode.values(2)[0, :], v1_second)
    assert np.allclose(ode.values(2)[1, :], s1_second)

    # 3. Check mapping: the standard dolfin Function should not be updated automatically
    assert np.allclose(v_ode.x.array, 0.0)

    ode.to_dolfin()
    assert np.allclose(v_ode.x.array[markers.x.array == 1], v1_first)
    assert np.allclose(v_ode.x.array[markers.x.array == 2], v1_second)
    assert np.allclose(v_pde.x.array, 0.0)

    ode.ode_to_pde()
    assert np.allclose(v_pde.x.array[markers.x.array == 1], v1_first)
    assert np.allclose(v_pde.x.array[markers.x.array == 2], v1_second)

    # 4. Check reverse mapping: modifying PDE and pulling back to the ODE spaces
    v_pde.x.array[:] = 7.0
    ode.pde_to_ode()
    assert np.allclose(v_ode.x.array, 7.0)

    ode.from_dolfin()
    assert np.allclose(ode.values(1)[0, :], 7.0)
    assert np.allclose(ode.values(2)[0, :], 7.0)
    assert np.allclose(ode.values(1)[1, :], s1_first)
    assert np.allclose(ode.values(2)[1, :], s1_second)

    # 5. Extract all states to separate functions
    states = ode.states_to_dolfin()
    assert len(states) == 2
    assert np.allclose(states[0].x.array[markers.x.array == 1], 7.0)
    assert np.allclose(states[0].x.array[markers.x.array == 2], 7.0)
    assert np.allclose(states[1].x.array[markers.x.array == 1], s1_first)
    assert np.allclose(states[1].x.array[markers.x.array == 2], s1_second)

    # Prevent MPI deadlocks during destruction
    mesh.comm.Barrier()
    del ode
    mesh.comm.Barrier()
    gc.collect()
    mesh.comm.Barrier()
