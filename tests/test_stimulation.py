from mpi4py import MPI

import dolfinx
import numpy as np
import ufl

import beat


def test_single_stimulation():
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_interval(comm, 10)
    value = 2.0
    end = 1.0
    start = 0.5
    dt = 0.01
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))

    expr = ufl.conditional(ufl.And(ufl.ge(time, start), ufl.le(time, end)), value, 0.0)
    I_s = beat.stimulation.Stimulus(dZ=ufl.dx(domain=mesh), expr=expr)

    pde = beat.MonodomainModel(
        time=time,
        mesh=mesh,
        M=dolfinx.fem.Constant(mesh, 0.0),
        I_s=I_s,
    )

    pde.step((0.0, 0.4))
    assert np.allclose(pde.state.x.array, 0.0)

    t0 = 0.9
    stim_t0 = value * (t0 - start)
    pde.solve((0.4, t0), dt=dt)

    # At time dt the stimulus should be value and since M is zero the state should be value * dt
    assert np.allclose(pde.state.x.array, stim_t0)

    pde.solve((t0, end + dt), dt=dt)

    # At end the stimulus should be zero and since M is zero the state should be zero
    assert np.allclose(pde.state.x.array, (end - start - dt) * value)

    # Solving for longer time should not change the state
    pde.solve((end + dt, 2 * end), dt=dt)
    assert np.allclose(pde.state.x.array, (end - start - dt) * value)


def test_double_stimulation():
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_interval(comm, 10)
    dt = 0.01
    value1 = 2.0
    value2 = 3.0
    start1 = 0.5
    end1 = 1.0
    start2 = 0.9
    end2 = 1.5

    time = dolfinx.fem.Constant(mesh, 0.0)
    expr1 = ufl.conditional(ufl.And(ufl.ge(time, start1), ufl.le(time, end1)), value1, 0.0)
    expr2 = ufl.conditional(ufl.And(ufl.ge(time, start2), ufl.le(time, end2)), value2, 0.0)
    dx = ufl.dx(domain=mesh)
    I_s = [
        beat.stimulation.Stimulus(dZ=dx, expr=expr1),
        beat.stimulation.Stimulus(dZ=dx, expr=expr2),
    ]

    pde = beat.MonodomainModel(
        time=time,
        mesh=mesh,
        M=dolfinx.fem.Constant(mesh, 0.0),
        I_s=I_s,
    )

    pde.step((0.0, 0.4))
    assert np.allclose(pde.state.x.array, 0.0)

    # Solve up to the second stimulus starts
    t0 = 0.9
    stim_t0 = value1 * (t0 - start1)
    pde.solve((0.4, t0), dt=dt)
    # At time dt the stimulus should be value and since M is zero the state should be value * dt
    assert np.allclose(pde.state.x.array, stim_t0)

    # Solve up to the end of the first stimulus
    pde.solve((t0, end1 + dt), dt=dt)
    assert np.allclose(
        pde.state.x.array,
        (end1 - start1 - dt) * value1 + (end1 + dt - start2) * value2,
    )

    # Solve up to the end of the second stimulus
    pde.solve((end1 + dt, end2 + dt), dt=dt)
    assert np.allclose(
        pde.state.x.array,
        (end1 - start1 - dt) * value1 + (end2 - start2 - dt) * value2,
    )

    # Solving for longer time should not change the state
    pde.solve((end2 + dt, 2 * end2), dt=dt)
    assert np.allclose(
        pde.state.x.array,
        (end1 - start1 - dt) * value1 + (end2 - start2 - dt) * value2,
    )
