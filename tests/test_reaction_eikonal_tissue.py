"""R-E- and R-E+ on a 2D slab with a cheap bistable cell model (in mV)."""

from mpi4py import MPI

import dolfinx
import numpy as np

import beat
from beat.activation import ActivationTracker
from beat.conductivities import conductivity_tensor
from beat.eikonal import Source, solve_eikonal
from beat.odesolver import DolfinODESolver
from beat.pseudo_bidomain import LeadFieldElectrodes, VolumeConductor
from beat.reaction_eikonal import ExponentialFoot, FootCurrentODESolver, ODEOnlySolver

V_REST, V_C, V_PEAK, K = -85.0, -60.0, 30.0, 1e-4  # dv/dt = -K (v-V_REST)(v-V_C)(v-V_PEAK)
VL, VT = 0.6, 0.3
L = 10.0


def bistable(states, t, parameters, dt):
    v = states[0]
    return (v - dt * K * (v - V_REST) * (v - V_C) * (v - V_PEAK))[None, :]


def setup():
    mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, 2.0]], [40, 8])
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0]))
    t_a = solve_eikonal(mesh, (VL, VT), (f0, s0), [Source(lambda x: np.isclose(x[0], 0.0))])
    V = t_a.function_space
    return mesh, V, t_a


def cell_model(V, v_pde):
    return DolfinODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=v_pde,
        init_states=np.array([V_REST]),
        parameters=np.array([]),
        fun=bistable,
        num_states=1,
        v_index=0,
    )


def run(solver, v, end=20.0, dt=0.01):
    tracker = ActivationTracker(v, method="threshold", threshold=-30.0)
    t = 0.0
    while t < end - 1e-9:
        solver.step((t, t + dt))
        t += dt
        tracker.update(t)
    return tracker.t_a


def owned(u):
    return u.x.array[: u.function_space.dofmap.index_map.size_local]


def test_re_minus_activates_after_t_a():
    mesh, V, t_a = setup()
    ode = FootCurrentODESolver(cell_model(V, dolfinx.fem.Function(V)), t_a, ExponentialFoot())
    solver = ODEOnlySolver(ode)
    act = run(solver, solver.v)
    delay = owned(act) - owned(t_a)
    comm = mesh.comm
    assert comm.allreduce(int(np.isnan(owned(act)).sum())) == 0
    assert comm.allreduce(delay.min(), op=MPI.MIN) >= 0.0
    assert comm.allreduce(delay.max(), op=MPI.MAX) <= 1.5


def test_re_plus_follows_t_a():
    mesh, V, t_a = setup()
    time = dolfinx.fem.Constant(mesh, 0.0)
    pde = beat.MonodomainModel(time=time, mesh=mesh, M=0.02)
    ode = FootCurrentODESolver(cell_model(V, pde.state), t_a, ExponentialFoot())
    solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)
    act = run(solver, pde.v)
    delay = owned(act) - owned(t_a)
    comm = mesh.comm
    assert comm.allreduce(int(np.isnan(owned(act)).sum())) == 0
    assert comm.allreduce(delay.min(), op=MPI.MIN) >= -1.0
    assert comm.allreduce(delay.max(), op=MPI.MAX) <= 1.5


def slab_electrogram(comm, end=20.0, dt=0.02):
    """phi_e above the middle of a 2D slab in a bath while an R-E- front crosses it"""
    tissue = dolfinx.mesh.create_rectangle(comm, [[0.0, 0.0], [L, 2.0]], [40, 8])
    bath = dolfinx.mesh.create_rectangle(comm, [[-1.0, -1.0], [L + 1.0, 3.0]], [48, 16])
    imap = bath.topology.index_map(2)
    cells = np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)
    mid = dolfinx.mesh.compute_midpoints(bath, 2, cells)
    inside = (mid[:, 0] > 0) & (mid[:, 0] < L) & (mid[:, 1] > 0) & (mid[:, 1] < 2.0)
    tags = dolfinx.mesh.meshtags(bath, 2, cells, np.where(inside, 1, 2).astype(np.int32))

    f0 = dolfinx.fem.Constant(tissue, np.array([1.0, 0.0]))
    s0 = dolfinx.fem.Constant(tissue, np.array([0.0, 1.0]))
    t_a = solve_eikonal(tissue, (VL, VT), (f0, s0), [Source(lambda x: np.isclose(x[0], 0.0))])
    V = t_a.function_space
    solver = ODEOnlySolver(
        FootCurrentODESolver(cell_model(V, dolfinx.fem.Function(V)), t_a, ExponentialFoot()),
    )

    fb = dolfinx.fem.Constant(bath, np.array([1.0, 0.0]))
    conductor = VolumeConductor(
        bath,
        tags,
        1,
        2,
        conductivity_tensor(0.16, 0.04, fb),
        conductivity_tensor(0.62, 0.24, fb),
        1.0,
    )
    record = LeadFieldElectrodes(
        conductor,
        solver.v,
        conductivity_tensor(0.16, 0.04, f0),
        {"top": [L / 2, 2.5]},
    )
    trace, t = [], 0.0
    while t < end - 1e-9:
        solver.step((t, t + dt))
        t += dt
        trace.append(record()["top"])
    return np.array(trace)


def test_electrogram_in_a_bath_is_partition_independent():
    parallel = slab_electrogram(MPI.COMM_WORLD)
    serial = slab_electrogram(MPI.COMM_SELF)
    assert np.all(np.isfinite(parallel))
    assert np.abs(parallel).max() > 1e-6
    assert np.allclose(parallel, serial, rtol=1e-7, atol=1e-10)
