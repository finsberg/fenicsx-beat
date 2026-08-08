import gc

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import beat.utils
from beat.irksome_odesolver import IrksomeMultiODESolver, IrksomeODESolver

# Skip all tests in this file if irksome is not installed
irksome = pytest.importorskip("irksome")


def simple_ode_ufl(states, t, parameters):
    """
    UFL formulation of the simple harmonic oscillator:
    v' = -a * s
    s' = b * v
    """
    v, s = states[0], states[1]
    a, b = parameters[0], parameters[1]

    dv = -a * s
    ds = b * v

    return [dv, ds]


def test_irksome_odesolver_temporal_convergence():
    """Test that IrksomeODESolver converges at the expected rate."""
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 1, 1, dolfinx.cpp.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, 0.0)

    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    v_ode = dolfinx.fem.Function(V)
    v_pde = dolfinx.fem.Function(V)

    N_ode = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
    init_states = np.zeros((2, N_ode))
    init_states[0, :] = 1.0  # v(0) = 1.0
    init_states[1, :] = 0.0  # s(0) = 0.0

    # Use 2nd-order Implicit Midpoint (Gauss-Legendre 1-stage)
    tableau = irksome.GaussLegendre(1)

    T = 1.0
    errors = []
    dts = [0.1, 0.05, 0.025]

    for dt in dts:
        time.value = 0.0
        ode = IrksomeODESolver(
            v_ode=v_ode,
            v_pde=v_pde,
            fun=simple_ode_ufl,
            init_states=init_states.copy(),
            butcher_tableau=tableau,
            time=time,
            num_states=2,
            v_index=0,
            parameters=[1.0, 1.0],  # a=1.0, b=1.0
        )

        t = 0.0
        while t < T - 1e-8:
            ode.step(t, dt)
            t += dt

        # The exact solution for this system is v(t) = cos(t), s(t) = sin(t)
        v_exact = np.cos(T)

        # Calculate max error on the current mesh
        vals = ode.full_values
        error = np.max(np.abs(vals[0, :] - v_exact))

        # Aggregate error across all MPI ranks
        global_error = comm.allreduce(error, op=MPI.MAX)
        errors.append(global_error)

        # Prevent MPI deadlocks during destruction
        mesh.comm.Barrier()
        del ode
        mesh.comm.Barrier()
        gc.collect()
        mesh.comm.Barrier()

    # Verify that the method achieves ~2nd order convergence
    rates = [
        np.log(e1 / e2) / np.log(dts[i] / dts[i + 1])
        for i, (e1, e2) in enumerate(zip(errors[:-1], errors[1:]))
    ]
    assert all(rate > 1.9 for rate in rates), f"Expected 2nd order convergence, got {rates}"


def test_irksome_odesolver_mappings():
    """Test that data maps correctly between the Mixed Element ODE space and the PDEs."""
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

    time = dolfinx.fem.Constant(mesh, 0.0)
    tableau = irksome.BackwardEuler()

    ode = IrksomeODESolver(
        v_ode=v_ode,
        v_pde=v_pde,
        fun=simple_ode_ufl,
        init_states=init_states,
        butcher_tableau=tableau,
        time=time,
        num_states=2,
        v_index=0,
        parameters=[1.0, 1.0],
    )

    # 1. Initial State Assignment Verification
    assert ode.full_values.shape == (2, N_ode)
    assert np.allclose(ode.full_values[0, :], v0)
    assert np.allclose(ode.full_values[1, :], s0)

    # 2. Step the ODEs using Backward Euler
    dt = 0.1
    ode.step(0.0, dt)

    # Exact discrete step for fully implicit backward euler:
    # v1 = v0 - dt * s1  =>  v1 + dt * s1 = v0
    # s1 = s0 + dt * v1  => -dt * v1 + s1 = s0
    v1_exact = (v0 - dt * s0) / (1 + dt**2)
    s1_exact = s0 + dt * v1_exact

    assert np.allclose(ode.full_values[0, :], v1_exact)
    assert np.allclose(ode.full_values[1, :], s1_exact)

    # 3. Check Mapping: The standard dolfin Function should NOT be updated automatically
    assert np.allclose(v_ode.x.array, 0.0)

    # Check to_dolfin() pushes ODE mixed space -> v_ode
    ode.to_dolfin()
    assert np.allclose(v_ode.x.array, v1_exact)
    assert np.allclose(v_pde.x.array, 0.0)

    # Check ode_to_pde() local projection
    ode.ode_to_pde()
    assert np.allclose(v_pde.x.array, v1_exact)

    # 4. Check Reverse Mapping: Modifying PDE and pulling back to ODE mixed space
    v_pde.x.array[:] = 5.0
    ode.pde_to_ode()
    assert np.allclose(v_ode.x.array, 5.0)

    ode.from_dolfin()
    assert np.allclose(ode.full_values[0, :], 5.0)
    assert np.allclose(ode.full_values[1, :], s1_exact)  # S state remains untouched

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


def test_irksome_multi_odesolver_mappings():
    """Test that data maps correctly for a two-region IrksomeMultiODESolver, mirroring
    test_irksome_odesolver_mappings (single region) and test_DolfinMultiODESolver
    (plain-numpy multi region)."""
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

    time = dolfinx.fem.Constant(mesh, 0.0)
    tableau = irksome.BackwardEuler()

    ode = IrksomeMultiODESolver(
        v_ode=v_ode,
        v_pde=v_pde,
        markers=markers,
        fun={1: simple_ode_ufl, 2: simple_ode_ufl},
        init_states=init_states,
        butcher_tableau=tableau,
        time=time,
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

    # 2. Step the ODEs using Backward Euler
    dt = 0.1
    ode.step(0.0, dt)

    # Exact discrete step for fully implicit backward Euler, per region:
    # v1 = v0 - dt*a*s1, s1 = s0 + dt*a*v1
    # => v1 (1 + dt^2 a^2) = v0 - dt*a*s0
    def backward_euler_step(v0, s0, a, dt):
        v1 = (v0 - dt * a * s0) / (1 + dt**2 * a * a)
        s1 = s0 + dt * a * v1
        return v1, s1

    v1_first, s1_first = backward_euler_step(first_v0, first_s0, first_p0, dt)
    v1_second, s1_second = backward_euler_step(second_v0, second_s0, second_p0, dt)

    assert np.allclose(ode.values(1)[0, :], v1_first)
    assert np.allclose(ode.values(1)[1, :], s1_first)
    assert np.allclose(ode.values(2)[0, :], v1_second)
    assert np.allclose(ode.values(2)[1, :], s1_second)

    # 3. Check mapping: the standard dolfin Function should not be updated automatically
    assert np.allclose(v_ode.x.array, 0.0)

    # Check to_dolfin() pushes each region's mixed ODE space -> v_ode
    ode.to_dolfin()
    assert np.allclose(v_ode.x.array[markers.x.array == 1], v1_first)
    assert np.allclose(v_ode.x.array[markers.x.array == 2], v1_second)
    assert np.allclose(v_pde.x.array, 0.0)

    # Check ode_to_pde() local projection
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
    # The s state in each region should be untouched
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


def test_irksome_multi_odesolver_temporal_convergence():
    """Test that IrksomeMultiODESolver converges at the expected rate in each region,
    with a different oscillation frequency (parameter `a`) per region."""
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 2, 2, dolfinx.cpp.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, 0.0)

    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    v_ode = dolfinx.fem.Function(V)
    v_pde = dolfinx.fem.Function(V)

    markers = dolfinx.fem.Function(V)
    X = ufl.SpatialCoordinate(mesh)
    expr = ufl.conditional(ufl.lt(X[0], 0.5), 1, 2)
    markers.interpolate(dolfinx.fem.Expression(expr, beat.utils.interpolation_points(V)))

    a_first, a_second = 1.0, 2.0
    parameters = {1: [a_first, a_first], 2: [a_second, a_second]}

    # Use 2nd-order Implicit Midpoint (Gauss-Legendre 1-stage)
    tableau = irksome.GaussLegendre(1)

    T = 1.0
    errors = {1: [], 2: []}
    dts = [0.1, 0.05, 0.025]

    for dt in dts:
        time.value = 0.0
        init_states = {
            1: np.zeros((2, int((markers.x.array == 1).sum()))),
            2: np.zeros((2, int((markers.x.array == 2).sum()))),
        }
        init_states[1][0, :] = 1.0  # v(0) = 1.0
        init_states[2][0, :] = 1.0

        ode = IrksomeMultiODESolver(
            v_ode=v_ode,
            v_pde=v_pde,
            markers=markers,
            fun={1: simple_ode_ufl, 2: simple_ode_ufl},
            init_states=init_states,
            butcher_tableau=tableau,
            time=time,
            num_states={1: 2, 2: 2},
            v_index={1: 0, 2: 0},
            parameters=parameters,
        )

        t = 0.0
        while t < T - 1e-8:
            ode.step(t, dt)
            t += dt

        # The exact solution for v' = -a*s, s' = a*v, v(0)=1, s(0)=0 is v(t) = cos(a*t)
        for marker, a in ((1, a_first), (2, a_second)):
            v_exact = np.cos(a * T)
            error = np.max(np.abs(ode.values(marker)[0, :] - v_exact))
            errors[marker].append(comm.allreduce(error, op=MPI.MAX))

        # Prevent MPI deadlocks during destruction
        mesh.comm.Barrier()
        del ode
        mesh.comm.Barrier()
        gc.collect()
        mesh.comm.Barrier()

    for marker, marker_errors in errors.items():
        rates = [
            np.log(e1 / e2) / np.log(dts[i] / dts[i + 1])
            for i, (e1, e2) in enumerate(zip(marker_errors[:-1], marker_errors[1:]))
        ]
        assert all(
            rate > 1.9 for rate in rates
        ), f"Expected 2nd order convergence in region {marker}, got {rates}"
