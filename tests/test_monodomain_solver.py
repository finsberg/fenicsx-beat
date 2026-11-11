import contextlib
import gc
import logging
import warnings

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import beat


def v_exact_func(x, t):
    return ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t)


def s_exact_func(x, t):
    return -ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.cos(t)


def ac_func(x, t):
    return 8 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t)


def simple_ode_forward_euler(states, t, dt, parameters):
    v, s = states
    values = np.zeros_like(states)
    values[0] = v - s * dt
    values[1] = s + v * dt
    return values


class NullODESolver:
    """An ODE solver that does nothing, for exercising the splitting driver on its own."""

    def to_dolfin(self) -> None: ...

    def from_dolfin(self) -> None: ...

    def ode_to_pde(self) -> None: ...

    def pde_to_ode(self) -> None: ...

    def step(self, t0: float, dt: float) -> None: ...

    def states_to_dolfin(self, names=None) -> list[dolfinx.fem.Function]:
        return []

    def load_all_states(self, functions) -> None: ...


def test_splitting_solver_logs_both_ends_of_the_interval(caplog):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 2, 2, dolfinx.cpp.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    pde = beat.MonodomainModel(time=time, mesh=mesh, M=1.0)
    solver = beat.SplittingSolver(pde=pde, ode=NullODESolver())

    with caplog.at_level(logging.DEBUG, logger="beat.monodomain_solver"):
        solver.solve((0.0, 1.0), dt=0.5)

    messages = [r.getMessage() for r in caplog.records if r.name == "beat.monodomain_solver"]

    assert "Solving on t = (0.00, 0.50)" in messages, messages
    assert "Solving on t = (0.50, 1.00)" in messages, messages


@pytest.mark.parametrize(
    "odespace",
    [
        "P_1",
        "P_2",
        "DG_1",
    ],
)
def test_monodomain_splitting_analytic(odespace):
    N = 50

    M = 1.0
    dt = 0.01
    T = 1.0
    t0 = 0.0

    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    x = ufl.SpatialCoordinate(mesh)

    I_s = ac_func(x, time)

    pde = beat.MonodomainModel(time=time, mesh=mesh, M=M, I_s=I_s)

    V_ode = beat.utils.space_from_string(odespace, mesh, dim=1)
    v_ode = dolfinx.fem.Function(V_ode)

    s = dolfinx.fem.Function(V_ode)

    s.interpolate(
        dolfinx.fem.Expression(s_exact_func(x, time), beat.utils.interpolation_points(V_ode)),
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
    solver = beat.SplittingSolver(pde=pde, ode=ode)
    solver.solve((t0, T), dt=dt)

    v_exact = v_exact_func(x, time)
    error = dolfinx.fem.form((pde.state - v_exact) ** 2 * ufl.dx)
    E = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(error), MPI.SUM))
    print("Error: ", E, odespace)
    assert E < 0.002


@pytest.mark.parametrize(
    "odespace",
    [
        "CG_1",
        "CG_2",
        "DG_1",
    ],
)
def test_monodomain_splitting_spatial_convergence(odespace):
    M = 1.0
    dt = 0.001
    T = 1.0
    t0 = 0.0

    comm = MPI.COMM_WORLD
    errors = []
    Ns = [2**level for level in range(3, 6)]

    for N in Ns:
        mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)
        time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
        x = ufl.SpatialCoordinate(mesh)

        I_s = ac_func(x, time)

        pde = beat.MonodomainModel(time=time, mesh=mesh, M=M, I_s=I_s)

        V_ode = beat.utils.space_from_string(odespace, mesh, dim=1)
        v_ode = dolfinx.fem.Function(V_ode)

        s = dolfinx.fem.Function(V_ode)
        s.interpolate(
            dolfinx.fem.Expression(s_exact_func(x, time), beat.utils.interpolation_points(V_ode)),
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
        solver = beat.SplittingSolver(pde=pde, ode=ode, theta=1.0)
        solver.solve((t0, T), dt=dt)

        v_exact = v_exact_func(x, time)
        error = dolfinx.fem.form((pde.state - v_exact) ** 2 * ufl.dx)
        E = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(error), MPI.SUM))
        errors.append(E)

    rates = [np.log(e1 / e2) / np.log(2) for e1, e2 in zip(errors[:-1], errors[1:])]
    cvg_rate = sum(rates) / len(rates)
    # Should be 2
    assert cvg_rate > 1.85


@pytest.mark.parametrize("theta", [1.0])  # Should also be implemented fro theta=0.5
@pytest.mark.parametrize(
    "odespace",
    [
        "CG_1",
        "CG_2",
        "DG_1",
    ],
)
def test_monodomain_splitting_temporal_convergence(theta, odespace):
    M = 1.0
    T = 1.0
    t0 = 0.0

    comm = MPI.COMM_WORLD

    N = 150
    mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)
    V_ode = beat.utils.space_from_string(odespace, mesh, dim=1)
    v_ode = dolfinx.fem.Function(V_ode)

    errors = []

    dts = [1.0 / (2**level) for level in range(3, 6)]
    for dt in dts:
        time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
        x = ufl.SpatialCoordinate(mesh)

        I_s = ac_func(x, time)

        pde = beat.MonodomainModel(time=time, mesh=mesh, M=M, I_s=I_s)

        s = dolfinx.fem.Function(V_ode)
        s.interpolate(
            dolfinx.fem.Expression(s_exact_func(x, time), beat.utils.interpolation_points(V_ode)),
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
        solver = beat.SplittingSolver(pde=pde, ode=ode, theta=theta)
        solver.solve((t0, T), dt=dt)
        v_exact = v_exact_func(x, time)

        error = dolfinx.fem.form((pde.state - v_exact) ** 2 * ufl.dx)
        E = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(error), MPI.SUM))
        errors.append(E)

    rates = [np.log2(e1 / e2) for e1, e2 in zip(errors[:-1], errors[1:])]
    cvg_rate = sum(rates) / len(rates)

    print(rates, odespace, theta)

    # Should be 1
    assert cvg_rate > 1.0


@pytest.mark.parametrize(
    "odespace",
    [
        "P_1",
        "P_2",
        "DG_1",
    ],
)
def test_irksome_monodomain_splitting_analytic(odespace):
    irksome = pytest.importorskip("irksome")
    from beat.irksome_model import IrksomeMonodomainModel

    N = 50

    M = 1.0
    dt = 0.01
    T = 1.0
    t0 = 0.0

    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    x = ufl.SpatialCoordinate(mesh)

    I_s = ac_func(x, time)

    # Use Backward Euler to match the first-order Godunov splitting
    tableau = irksome.BackwardEuler()
    params = dict(petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    pde = IrksomeMonodomainModel(
        time=time,
        mesh=mesh,
        M=M,
        butcher_tableau=tableau,
        I_s=I_s,
        params=params,
    )

    V_ode = beat.utils.space_from_string(odespace, mesh, dim=1)
    v_ode = dolfinx.fem.Function(V_ode)

    s = dolfinx.fem.Function(V_ode)

    s.interpolate(
        dolfinx.fem.Expression(s_exact_func(x, time), beat.utils.interpolation_points(V_ode)),
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

    solver = beat.SplittingSolver(pde=pde, ode=ode)
    solver.solve((t0, T), dt=dt)

    v_exact = v_exact_func(x, time)
    error = dolfinx.fem.form((pde.state - v_exact) ** 2 * ufl.dx)
    E = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(error), MPI.SUM))

    print("Error: ", E, odespace)

    # Clean up to prevent MPI hangs
    mesh.comm.Barrier()
    del solver
    del pde
    del ode
    mesh.comm.Barrier()
    gc.collect()
    mesh.comm.Barrier()

    assert E < 0.002


@pytest.mark.parametrize(
    "odespace",
    [
        "P_1",
        "P_2",
        "DG_1",
    ],
)
def test_external_operator_monodomain_splitting_analytic(odespace):
    """Same setup as test_monodomain_splitting_analytic, but with the ODE step handled by
    ExternalOperatorODESolver instead of DolfinODESolver, combined with the *unmodified*
    beat.MonodomainModel PDE step -- i.e. the "swap in a new ODE solver, keep everything
    else the same" story that beat.external_operator_odesolver is designed for."""
    pytest.importorskip("dolfinx_external_operator")
    from beat.external_operator_odesolver import ExternalOperatorODESolver

    N = 50

    M = 1.0
    dt = 0.01
    T = 1.0
    t0 = 0.0

    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    x = ufl.SpatialCoordinate(mesh)

    I_s = ac_func(x, time)

    pde = beat.MonodomainModel(time=time, mesh=mesh, M=M, I_s=I_s)

    V_ode = beat.utils.space_from_string(odespace, mesh, dim=1)
    v_ode = dolfinx.fem.Function(V_ode)

    s = dolfinx.fem.Function(V_ode)
    s.interpolate(
        dolfinx.fem.Expression(s_exact_func(x, time), beat.utils.interpolation_points(V_ode)),
    )

    s_arr = s.x.array
    init_states = np.zeros((2, s_arr.size))
    init_states[1, :] = s_arr

    ode = ExternalOperatorODESolver(
        v_ode=v_ode,
        v_pde=pde.state,
        fun=simple_ode_forward_euler,
        init_states=init_states,
        num_states=2,
        v_index=0,
    )

    solver = beat.SplittingSolver(pde=pde, ode=ode)
    solver.solve((t0, T), dt=dt)

    v_exact = v_exact_func(x, time)
    error = dolfinx.fem.form((pde.state - v_exact) ** 2 * ufl.dx)
    E = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(error), MPI.SUM))

    print("Error: ", E, odespace)

    # Clean up to prevent MPI hangs
    mesh.comm.Barrier()
    del solver
    del pde
    del ode
    mesh.comm.Barrier()
    gc.collect()
    mesh.comm.Barrier()

    assert E < 0.002


RESTART_DT = 0.01


def build_restart_solver(theta):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, dolfinx.cpp.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    x = ufl.SpatialCoordinate(mesh)
    pde = beat.MonodomainModel(time=time, mesh=mesh, M=1.0)

    V_ode = beat.utils.space_from_string("P_1", mesh, dim=1)
    s = dolfinx.fem.Function(V_ode)
    s.interpolate(
        dolfinx.fem.Expression(s_exact_func(x, time), beat.utils.interpolation_points(V_ode)),
    )
    init_states = np.zeros((2, s.x.array.size))
    init_states[1, :] = s.x.array

    ode = beat.odesolver.DolfinODESolver(
        v_ode=dolfinx.fem.Function(V_ode),
        v_pde=pde.state,
        fun=simple_ode_forward_euler,
        init_states=init_states,
        parameters=None,
        num_states=2,
        v_index=0,
    )
    return beat.SplittingSolver(pde=pde, ode=ode, theta=theta)


def run_steps(solver, first, last):
    for i in range(first, last):
        solver.step((i * RESTART_DT, (i + 1) * RESTART_DT))


@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_restart_functions_restore_the_solver_bit_for_bit(theta):
    a = build_restart_solver(theta)
    run_steps(a, 0, 3)
    saved = [(n, f.x.array.copy()) for n, f in a.restart_functions(["v", "s"])]
    assert np.any(saved[0][1] != 0.0)

    b = build_restart_solver(theta)
    functions = b.restart_functions(["v", "s"])
    for (_, f), (_, arr) in zip(functions, saved):
        f.x.array[:] = arr
    b.load_restart(functions, a.restart_metadata())

    run_steps(a, 3, 6)
    run_steps(b, 3, 6)

    assert np.array_equal(a.pde.v.x.array, b.pde.v.x.array)
    assert np.array_equal(a.ode.values, b.ode.values)


def test_restart_function_names():
    solver = build_restart_solver(1.0)
    names = [n for n, _ in solver.restart_functions(["v", "s"])]
    assert names == ["v", "state_v", "state_s"]


def test_restart_functions_hand_out_the_pde_potential_itself():
    solver = build_restart_solver(1.0)
    assert solver.restart_functions(["v", "s"])[0][1] is solver.pde.v


def test_load_restart_refuses_other_names():
    solver = build_restart_solver(1.0)
    functions = solver.restart_functions(["v", "s"])
    with pytest.raises(ValueError, match="v"):
        solver.load_restart([("u", functions[0][1]), *functions[1:]])
    with pytest.raises(ValueError, match="state_"):
        solver.load_restart([functions[0], ("w", functions[1][1]), functions[2]])


def test_restart_metadata_round_trips_the_pde_timestep():
    a = build_restart_solver(1.0)
    run_steps(a, 3, 5)  # starts at 0.03: its dt is not the float 0.01
    metadata = a.restart_metadata()
    assert metadata["pde"]["timestep"] == float(a.pde._timestep.value)

    b = build_restart_solver(1.0)
    assert float(b.pde._timestep.value) != metadata["pde"]["timestep"]
    b.load_restart(b.restart_functions(["v", "s"]), metadata)
    assert float(b.pde._timestep.value) == metadata["pde"]["timestep"]


def test_restart_metadata_is_empty_for_a_pde_without_it():
    class Bare:
        v = None

        def assign_previous(self) -> None: ...

    solver = beat.SplittingSolver(pde=Bare(), ode=NullODESolver())  # type: ignore[arg-type]
    assert solver.restart_metadata() == {}


def _small_pde():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2, dolfinx.cpp.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    return beat.MonodomainModel(time=time, mesh=mesh, M=1.0)


def test_old_name_warns_at_the_call_site_and_still_works():
    pde = _small_pde()
    with pytest.warns(DeprecationWarning, match="use SplittingSolver") as record:
        solver = beat.MonodomainSplittingSolver(pde=pde, ode=NullODESolver())
    assert len(record) == 1
    assert record[0].filename == __file__
    assert isinstance(solver, beat.SplittingSolver)
    solver.solve((0.0, 1.0), dt=0.5)


def test_new_name_does_not_warn():
    pde = _small_pde()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        beat.SplittingSolver(pde=pde, ode=NullODESolver())


class RecordingMonitor(beat.BaseMonitor):
    def __init__(self):
        self.names, self.steps = [], []

    @contextlib.contextmanager
    def track_time(self, name):
        self.names.append(name)
        yield

    def record_ksp(self, ksp):
        pass

    def advance_step(self, t0, t1):
        self.steps.append((t0, t1))


ODE_ONLY_DT = 0.125  # a power of two, so t1 - t0 == dt exactly


def build_ode_only(theta=1.0, monitor=None):
    """A pde=None solver over a 2-state forward-Euler ODE on P1, and its initial states"""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, dolfinx.cpp.mesh.CellType.triangle)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    x = V.tabulate_dof_coordinates()
    init_states = np.zeros((2, x.shape[0]))
    init_states[0] = np.cos(2 * np.pi * x[:, 0])
    init_states[1] = np.sin(2 * np.pi * x[:, 1])
    ode = beat.odesolver.DolfinODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=dolfinx.fem.Function(V),
        fun=simple_ode_forward_euler,
        init_states=init_states.copy(),
        parameters=None,
        num_states=2,
        v_index=0,
    )
    kwargs = {} if monitor is None else {"monitor": monitor}
    return beat.SplittingSolver(pde=None, ode=ode, theta=theta, **kwargs), init_states


def test_splitting_solver_v_is_the_pde_potential():
    solver = build_restart_solver(1.0)
    assert solver.v is solver.pde.v


def test_ode_only_v_is_the_ode_potential_and_starts_at_the_initial_state():
    solver, init = build_ode_only()
    assert solver.v is solver.ode.v_ode
    assert np.array_equal(solver.v.x.array, init[0])


def test_ode_only_solve_matches_stepping_the_ode_alone():
    solver, init = build_ode_only()
    solver.solve((0.0, 5 * ODE_ONLY_DT), ODE_ONLY_DT)

    reference, _ = build_ode_only()
    ode = reference.ode
    t = 0.0
    for _ in range(5):
        ode.step(t, ODE_ONLY_DT)
        t += ODE_ONLY_DT
    ode.to_dolfin()
    assert np.array_equal(solver.v.x.array, ode.v_ode.x.array)
    assert np.array_equal(solver.ode.values, ode.values)
    assert not np.array_equal(solver.v.x.array, init[0])


def test_ode_only_ignores_theta():
    a, _ = build_ode_only(theta=1.0)
    b, _ = build_ode_only(theta=0.5)
    a.solve((0.0, 3 * ODE_ONLY_DT), ODE_ONLY_DT)
    b.solve((0.0, 3 * ODE_ONLY_DT), ODE_ONLY_DT)
    assert np.array_equal(a.ode.values, b.ode.values)


def test_ode_only_telemetry():
    monitor = RecordingMonitor()
    solver, _ = build_ode_only(monitor=monitor)
    solver.solve((0.0, 2 * ODE_ONLY_DT), ODE_ONLY_DT)
    assert monitor.names == ["total_step", "ode_step", "ode_to_dolfin"] * 2
    assert monitor.steps == [(0.0, ODE_ONLY_DT), (ODE_ONLY_DT, 2 * ODE_ONLY_DT)]


def test_ode_only_restart_function_names():
    solver, _ = build_ode_only()
    functions = solver.restart_functions(["v", "s"])
    assert [n for n, _ in functions] == ["v", "state_v", "state_s"]
    assert functions[0][1] is solver.ode.v_ode
    assert solver.restart_metadata() == {}


def test_ode_only_restart_is_bit_for_bit():
    a, _ = build_ode_only()
    a.solve((0.0, 3 * ODE_ONLY_DT), ODE_ONLY_DT)
    saved = [(n, f.x.array.copy()) for n, f in a.restart_functions(["v", "s"])]

    b, _ = build_ode_only()
    functions = b.restart_functions(["v", "s"])
    for (_, f), (_, arr) in zip(functions, saved):
        f.x.array[:] = arr
    b.load_restart(functions, a.restart_metadata())

    a.solve((3 * ODE_ONLY_DT, 6 * ODE_ONLY_DT), ODE_ONLY_DT)
    b.solve((3 * ODE_ONLY_DT, 6 * ODE_ONLY_DT), ODE_ONLY_DT)
    assert np.array_equal(a.v.x.array, b.v.x.array)
    assert np.array_equal(a.ode.values, b.ode.values)


def test_ode_only_refuses_pde_metadata_without_touching_anything():
    solver, init = build_ode_only()
    functions = solver.restart_functions(["v", "s"])
    other = [(n, f.copy()) for n, f in functions]
    for _, f in other:
        f.x.array[:] = 7.0
    with pytest.raises(ValueError, match="no PDE"):
        solver.load_restart(other, {"pde": {"timestep": 0.1}})
    assert np.array_equal(solver.v.x.array, init[0])
    assert np.array_equal(solver.ode.values, init)
