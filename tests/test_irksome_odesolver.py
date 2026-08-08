import gc

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

from beat.irksome_odesolver import IrksomeODESolver

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
