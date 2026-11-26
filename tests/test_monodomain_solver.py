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


def simple_ode_rotation(states, t, dt, parameters):
    """Exact flow of dv/dt = -s, ds/dt = v: two half-steps compose to exactly one full step."""
    v, s = states
    c, sn = np.cos(dt), np.sin(dt)
    return np.array([v * c - s * sn, v * sn + s * c])


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


def test_merged_strang_splitting_is_second_order(strang_mms_module):
    """Merging the half-steps keeps Strang + generalized_rush_larsen_2 second order."""
    module = strang_mms_module

    def final_v(dt):
        return _strang_mms_final_v(
            module,
            "generalized_rush_larsen_2",
            "strang",
            dt,
            merge_half_steps=True,
        )

    ref = final_v(2.0**-10)
    dts = [2.0**-k for k in range(2, 7)]
    errors = [_rms_difference(final_v(dt), ref) for dt in dts]
    rate = np.mean(np.log2(np.array(errors[:-1]) / np.array(errors[1:])))
    print("merged", errors, rate)
    assert rate >= 1.9


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
        solver = beat.SplittingSolver(pde=pde, ode=ode, splitting="godunov")
        solver.solve((t0, T), dt=dt)

        v_exact = v_exact_func(x, time)
        error = dolfinx.fem.form((pde.state - v_exact) ** 2 * ufl.dx)
        E = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(error), MPI.SUM))
        errors.append(E)

    rates = [np.log(e1 / e2) / np.log(2) for e1, e2 in zip(errors[:-1], errors[1:])]
    cvg_rate = sum(rates) / len(rates)
    # Should be 2
    assert cvg_rate > 1.85


# Strang is checked by test_strang_splitting_second_order_with_gotranx_schemes
@pytest.mark.parametrize("splitting", ["godunov"])
@pytest.mark.parametrize(
    "odespace",
    [
        "CG_1",
        "CG_2",
        "DG_1",
    ],
)
def test_monodomain_splitting_temporal_convergence(splitting, odespace):
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
        solver = beat.SplittingSolver(pde=pde, ode=ode, splitting=splitting)
        solver.solve((t0, T), dt=dt)
        v_exact = v_exact_func(x, time)

        error = dolfinx.fem.form((pde.state - v_exact) ** 2 * ufl.dx)
        E = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(error), MPI.SUM))
        errors.append(E)

    rates = [np.log2(e1 / e2) for e1, e2 in zip(errors[:-1], errors[1:])]
    cvg_rate = sum(rates) / len(rates)

    print(rates, odespace, splitting)

    # Should be 1
    assert cvg_rate > 1.0


STRANG_MMS_ODE = """
states(v=0, s=-0.5)
dv_dt = -v - s
ds_dt = v - s
"""
STRANG_MMS_SCHEMES = ["explicit_euler", "generalized_rush_larsen", "generalized_rush_larsen_2"]


@pytest.fixture(scope="module")
def strang_mms_module():
    gotranx = pytest.importorskip("gotranx")
    ode = gotranx.load.ode_from_string(STRANG_MMS_ODE, name="strang_mms")
    code = gotranx.cli.gotran2py.get_code(
        ode,
        scheme=[gotranx.schemes.Scheme[s] for s in STRANG_MMS_SCHEMES],
    )
    module: dict = {}
    exec(code, module)
    return module


def _strang_mms_final_v(module, scheme, splitting, dt, N=16, M=0.01, T=1.0, merge_half_steps=False):
    """v at T for the manufactured solution v = phi sin t, s = phi (sin t - cos t) / 2.

    s solves ds/dt = v - s exactly; the stimulus closes the v equation
    dv/dt = M lap v + I_s - v - s.
    """
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    x = ufl.SpatialCoordinate(mesh)
    phi = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    s_t = (ufl.sin(time) - ufl.cos(time)) / 2
    I_s = phi * (ufl.cos(time) + ufl.sin(time) + s_t + 8 * ufl.pi**2 * M * ufl.sin(time))
    pde = beat.MonodomainModel(time=time, mesh=mesh, M=M, I_s=I_s, params={"theta": 0.5})

    V_ode = beat.utils.space_from_string("P_1", mesh, dim=1)
    s0 = dolfinx.fem.Function(V_ode)
    s0.interpolate(dolfinx.fem.Expression(phi * s_t, beat.utils.interpolation_points(V_ode)))
    # gotranx orders the states alphabetically (s before v)
    init_states = np.zeros((2, s0.x.array.size))
    init_states[module["state_index"]("s")] = s0.x.array
    ode = beat.odesolver.DolfinODESolver(
        v_ode=dolfinx.fem.Function(V_ode),
        v_pde=pde.state,
        fun=module[scheme],
        init_states=init_states,
        parameters=module["init_parameter_values"](),
        num_states=2,
        v_index=module["state_index"]("v"),
    )
    beat.SplittingSolver(
        pde=pde,
        ode=ode,
        splitting=splitting,
        merge_half_steps=merge_half_steps,
    ).solve((0.0, T), dt=dt)
    return pde.state


def _rms_difference(a, b):
    """Root-mean-square nodal difference over the owned dofs of all ranks."""
    n = a.function_space.dofmap.index_map.size_local * a.function_space.dofmap.index_map_bs
    d = a.x.array[:n] - b.x.array[:n]
    comm = a.function_space.mesh.comm
    return np.sqrt(comm.allreduce(np.sum(d**2), MPI.SUM) / comm.allreduce(n, MPI.SUM))


@pytest.mark.parametrize(
    ("scheme", "splitting", "second_order"),
    [
        ("generalized_rush_larsen_2", "strang", True),
        ("generalized_rush_larsen", "strang", False),
        ("explicit_euler", "strang", False),
        ("generalized_rush_larsen_2", "godunov", False),
    ],
)
def test_strang_splitting_second_order_with_gotranx_schemes(
    strang_mms_module,
    scheme,
    splitting,
    second_order,
):
    """Strang splitting is second order only with a second-order cell scheme.

    The diffusion is weak (M = 0.01) on purpose: with M = 1 the PDE dominates the error and even
    first-order cell schemes look second order. Errors are measured against a fine-dt reference
    on the same mesh, so the spatial error cancels and a coarse mesh suffices.
    """
    ref = _strang_mms_final_v(strang_mms_module, scheme, splitting, dt=2.0**-10)
    dts = [2.0**-k for k in range(2, 7)]
    errors = [
        _rms_difference(_strang_mms_final_v(strang_mms_module, scheme, splitting, dt), ref)
        for dt in dts
    ]
    rate = np.mean(np.log2(np.array(errors[:-1]) / np.array(errors[1:])))
    print(scheme, splitting, errors, rate)
    if second_order:
        assert rate >= 1.9
    else:
        assert rate < 1.5


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


def build_restart_solver(
    splitting,
    merge_half_steps=False,
    fun=simple_ode_forward_euler,
    monitor=None,
):
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
        fun=fun,
        init_states=init_states,
        parameters=None,
        num_states=2,
        v_index=0,
    )
    kwargs = {} if monitor is None else {"monitor": monitor}
    return beat.SplittingSolver(
        pde=pde,
        ode=ode,
        splitting=splitting,
        merge_half_steps=merge_half_steps,
        **kwargs,
    )


def run_steps(solver, first, last):
    for i in range(first, last):
        solver.step((i * RESTART_DT, (i + 1) * RESTART_DT))


@pytest.mark.parametrize("splitting", ["godunov", "strang"])
def test_restart_functions_restore_the_solver_bit_for_bit(splitting):
    a = build_restart_solver(splitting)
    run_steps(a, 0, 3)
    saved = [(n, f.x.array.copy()) for n, f in a.restart_functions(["v", "s"])]
    assert np.any(saved[0][1] != 0.0)

    b = build_restart_solver(splitting)
    functions = b.restart_functions(["v", "s"])
    for (_, f), (_, arr) in zip(functions, saved):
        f.x.array[:] = arr
    b.load_restart(functions, a.restart_metadata())

    run_steps(a, 3, 6)
    run_steps(b, 3, 6)

    assert np.array_equal(a.pde.v.x.array, b.pde.v.x.array)
    assert np.array_equal(a.ode.values, b.ode.values)


def test_restart_function_names():
    solver = build_restart_solver("godunov")
    names = [n for n, _ in solver.restart_functions(["v", "s"])]
    assert names == ["v", "state_v", "state_s"]


def test_merged_strang_matches_plain_strang_for_an_exact_ode_flow():
    plain_monitor, merged_monitor = RecordingMonitor(), RecordingMonitor()
    plain = build_restart_solver("strang", fun=simple_ode_rotation, monitor=plain_monitor)
    merged = build_restart_solver(
        "strang",
        merge_half_steps=True,
        fun=simple_ode_rotation,
        monitor=merged_monitor,
    )
    plain.solve((0.0, 10 * RESTART_DT), RESTART_DT)
    merged.solve((0.0, 10 * RESTART_DT), RESTART_DT)
    assert np.allclose(merged.pde.v.x.array, plain.pde.v.x.array, rtol=0, atol=1e-12)
    assert np.allclose(merged.ode.values, plain.ode.values, rtol=0, atol=1e-12)
    # merging really happened: one closing half-step (from solve's final synchronize) instead of 10
    assert plain_monitor.names.count("corrective_ode_step") == 10
    assert merged_monitor.names.count("corrective_ode_step") == 1
    assert merged_monitor.names.count("ode_step") == 10
    assert len(merged_monitor.steps) == 10


def test_merged_strang_handles_varying_time_steps():
    plain = build_restart_solver("strang", fun=simple_ode_rotation)
    merged = build_restart_solver("strang", merge_half_steps=True, fun=simple_ode_rotation)
    for interval in [(0.0, 0.01), (0.01, 0.015), (0.015, 0.035), (0.035, 0.045)]:
        plain.step(interval)
        merged.step(interval)
    merged.synchronize()
    assert np.allclose(merged.pde.v.x.array, plain.pde.v.x.array, rtol=0, atol=1e-12)
    assert np.allclose(merged.ode.values, plain.ode.values, rtol=0, atol=1e-12)


def test_turning_merging_off_synchronizes_the_stale_half_step():
    plain = build_restart_solver("strang", fun=simple_ode_rotation)
    merged = build_restart_solver("strang", merge_half_steps=True, fun=simple_ode_rotation)
    run_steps(plain, 0, 2)
    run_steps(merged, 0, 2)
    assert not merged.synchronized
    merged.merge_half_steps = False
    run_steps(plain, 2, 4)
    run_steps(merged, 2, 4)
    assert merged.synchronized
    assert np.allclose(merged.pde.v.x.array, plain.pde.v.x.array, rtol=0, atol=1e-12)
    assert np.allclose(merged.ode.values, plain.ode.values, rtol=0, atol=1e-12)


def test_merged_strang_passes_the_right_times_to_the_cell_model():
    calls = []

    def recording_rotation(states, t, dt, parameters):
        calls.append((t, dt))
        return simple_ode_rotation(states, t, dt, parameters)

    merged = build_restart_solver("strang", merge_half_steps=True, fun=recording_rotation)
    for interval in [(0.0, 0.01), (0.01, 0.03), (0.03, 0.04)]:
        merged.step(interval)
    merged.synchronize()
    expected = [(0.0, 0.005), (0.005, 0.015), (0.02, 0.015), (0.035, 0.005)]
    assert len(calls) == len(expected)
    assert np.allclose(calls, expected, rtol=0, atol=1e-12)


def test_synchronize_runs_the_pending_half_step_once():
    solver = build_restart_solver("strang", merge_half_steps=True)
    assert solver.synchronized
    solver.synchronize()  # nothing pending: no-op
    solver.step((0.0, RESTART_DT))
    assert not solver.synchronized
    before = solver.ode.values.copy()
    solver.synchronize()
    assert solver.synchronized
    assert not np.array_equal(solver.ode.values, before)
    after = solver.ode.values.copy()
    solver.synchronize()  # again: no-op
    assert np.array_equal(solver.ode.values, after)


@pytest.mark.parametrize("splitting", ["godunov", "strang"])
def test_synchronize_is_a_no_op_without_merging(splitting):
    solver = build_restart_solver(splitting)
    solver.step((0.0, RESTART_DT))
    assert solver.synchronized
    before = solver.ode.values.copy()
    solver.synchronize()
    assert np.array_equal(solver.ode.values, before)


def test_solve_ends_synchronized():
    solver = build_restart_solver("strang", merge_half_steps=True)
    solver.solve((0.0, 3 * RESTART_DT), RESTART_DT)
    assert solver.synchronized


def test_merge_half_steps_needs_strang():
    with pytest.raises(ValueError, match="splitting='strang'"):
        build_restart_solver("godunov", merge_half_steps=True)


def test_merge_half_steps_with_deprecated_theta():
    pde = _small_pde()
    with pytest.warns(DeprecationWarning):
        solver = beat.SplittingSolver(
            pde=pde,
            ode=NullODESolver(),
            theta=0.5,
            merge_half_steps=True,
        )
    assert solver.splitting == "strang"
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="splitting='strang'"):
        beat.SplittingSolver(
            pde=_small_pde(),
            ode=NullODESolver(),
            theta=0.3,
            merge_half_steps=True,
        )


def test_ode_only_ignores_merge_half_steps():
    a, _ = build_ode_only(splitting="strang")
    b, _ = build_ode_only(splitting="strang", merge_half_steps=True)
    a.solve((0.0, 3 * ODE_ONLY_DT), ODE_ONLY_DT)
    for i in range(3):
        b.step((i * ODE_ONLY_DT, (i + 1) * ODE_ONLY_DT))
        assert b.synchronized
    assert np.array_equal(a.ode.values, b.ode.values)


def test_merged_restart_is_bit_for_bit():
    a = build_restart_solver("strang", merge_half_steps=True)
    run_steps(a, 0, 3)
    assert not a.synchronized
    saved = [(n, f.x.array.copy()) for n, f in a.restart_functions(["v", "s"])]
    assert a.synchronized  # restart_functions synchronized first

    b = build_restart_solver("strang", merge_half_steps=True)
    functions = b.restart_functions(["v", "s"])
    for (_, f), (_, arr) in zip(functions, saved):
        f.x.array[:] = arr
    b.load_restart(functions, a.restart_metadata())
    assert b.synchronized

    run_steps(a, 3, 6)
    run_steps(b, 3, 6)
    a.synchronize()
    b.synchronize()
    assert np.array_equal(a.pde.v.x.array, b.pde.v.x.array)
    assert np.array_equal(a.ode.values, b.ode.values)


def test_restart_functions_hand_out_the_pde_potential_itself():
    solver = build_restart_solver("godunov")
    assert solver.restart_functions(["v", "s"])[0][1] is solver.pde.v


def test_load_restart_refuses_other_names():
    solver = build_restart_solver("godunov")
    functions = solver.restart_functions(["v", "s"])
    with pytest.raises(ValueError, match="v"):
        solver.load_restart([("u", functions[0][1]), *functions[1:]])
    with pytest.raises(ValueError, match="state_"):
        solver.load_restart([functions[0], ("w", functions[1][1]), functions[2]])


def test_restart_metadata_round_trips_the_pde_timestep():
    a = build_restart_solver("godunov")
    run_steps(a, 3, 5)  # starts at 0.03: its dt is not the float 0.01
    metadata = a.restart_metadata()
    assert metadata["pde"]["timestep"] == float(a.pde._timestep.value)

    b = build_restart_solver("godunov")
    assert float(b.pde._timestep.value) != metadata["pde"]["timestep"]
    b.load_restart(b.restart_functions(["v", "s"]), metadata)
    assert float(b.pde._timestep.value) == metadata["pde"]["timestep"]


def test_restart_metadata_only_records_the_pde_for_a_pde_without_it():
    class Bare:
        v = None

        def assign_previous(self) -> None: ...

    solver = beat.SplittingSolver(pde=Bare(), ode=NullODESolver())  # type: ignore[arg-type]
    assert solver.restart_metadata() == {"has_pde": True}


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


def test_old_name_theta_warning_points_at_the_call_site():
    with pytest.warns(DeprecationWarning) as record:
        beat.MonodomainSplittingSolver(pde=_small_pde(), ode=NullODESolver(), theta=0.5)
    assert len(record) == 2
    assert all(w.filename == __file__ for w in record)


def test_new_name_does_not_warn():
    pde = _small_pde()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        beat.SplittingSolver(pde=pde, ode=NullODESolver())


def test_default_splitting_is_strang_without_merging():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        solver = beat.SplittingSolver(pde=_small_pde(), ode=NullODESolver())
    assert solver.splitting == "strang"
    assert solver.theta == 0.5
    assert solver.merge_half_steps is False


class RecordingODESolver(NullODESolver):
    def __init__(self):
        self.calls = []
        self.v_ode = None

    def step(self, t0: float, dt: float) -> None:
        self.calls.append((t0, dt))


def test_ode_only_mode_takes_one_full_ode_step_with_the_default_splitting():
    ode = RecordingODESolver()
    solver = beat.SplittingSolver(pde=None, ode=ode)
    solver.step((0.0, 0.1))
    assert ode.calls == [(0.0, 0.1)]


def test_strang_splitting_takes_half_an_ode_step_first():
    solver = beat.SplittingSolver(pde=_small_pde(), ode=NullODESolver(), splitting="strang")
    assert solver.splitting == "strang"
    assert solver.theta == 0.5


def test_unknown_splitting_name_is_rejected():
    with pytest.raises(ValueError, match="'godunov' or 'strang'"):
        beat.SplittingSolver(pde=_small_pde(), ode=NullODESolver(), splitting="lie")


@pytest.mark.parametrize(("theta", "name"), [(1.0, "godunov"), (0.5, "strang")])
def test_deprecated_theta_maps_to_a_splitting_name(theta, name):
    with pytest.warns(DeprecationWarning, match=f"splitting={name!r}") as record:
        solver = beat.SplittingSolver(pde=_small_pde(), ode=NullODESolver(), theta=theta)
    assert len(record) == 1
    assert record[0].filename == __file__
    assert solver.splitting == name
    assert solver.theta == theta


@pytest.mark.parametrize(("theta", "fraction"), [(1.0 - 1e-9, 1.0), (0.5 + 1e-9, 0.5)])
def test_deprecated_theta_close_to_a_splitting_stores_its_exact_fraction(theta, fraction):
    with pytest.warns(DeprecationWarning):
        solver = beat.SplittingSolver(pde=_small_pde(), ode=NullODESolver(), theta=theta)
    assert solver.theta == fraction


def test_deprecated_positional_theta_still_works():
    with pytest.warns(DeprecationWarning, match="splitting='strang'"):
        solver = beat.SplittingSolver(_small_pde(), NullODESolver(), 0.5)
    assert solver.splitting == "strang"


def test_deprecated_custom_theta_runs_and_warns_it_is_first_order():
    with pytest.warns(DeprecationWarning, match="first order"):
        solver = beat.SplittingSolver(pde=_small_pde(), ode=NullODESolver(), theta=0.3)
    assert solver.splitting is None
    assert solver.theta == 0.3
    solver.solve((0.0, 1.0), dt=0.5)


def test_theta_and_splitting_together_is_an_error():
    with pytest.raises(TypeError, match="not the deprecated theta"):
        beat.SplittingSolver(pde=_small_pde(), ode=NullODESolver(), theta=0.5, splitting="strang")


@pytest.mark.parametrize("theta", [-0.1, 1.5])
def test_theta_outside_unit_interval_is_rejected(theta):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        beat.SplittingSolver(pde=_small_pde(), ode=NullODESolver(), theta=theta)


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


def build_ode_only(splitting="godunov", monitor=None, merge_half_steps=False):
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
    solver = beat.SplittingSolver(
        pde=None,
        ode=ode,
        splitting=splitting,
        merge_half_steps=merge_half_steps,
        **kwargs,
    )
    return solver, init_states


def test_splitting_solver_v_is_the_pde_potential():
    solver = build_restart_solver("godunov")
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


def test_ode_only_ignores_splitting():
    a, _ = build_ode_only(splitting="godunov")
    b, _ = build_ode_only(splitting="strang")
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
    assert solver.restart_metadata() == {"has_pde": False}


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


def test_restart_refuses_a_checkpoint_saved_without_a_pde_without_touching_anything():
    saved_by, _ = build_ode_only()
    metadata = saved_by.restart_metadata()
    solver = build_restart_solver("godunov")
    v_before = solver.v.x.array.copy()
    values_before = solver.ode.values.copy()
    # Same mesh and P1 spaces, so every Function fits: only the metadata can tell them apart
    functions = [(n, f.copy()) for n, f in saved_by.restart_functions(["v", "s"])]
    with pytest.raises(ValueError, match="without a PDE"):
        solver.load_restart(functions, metadata)
    assert np.array_equal(solver.v.x.array, v_before)
    assert np.array_equal(solver.ode.values, values_before)


def test_ode_only_refuses_a_checkpoint_saved_with_a_pde():
    solver, init = build_ode_only()
    functions = solver.restart_functions(["v", "s"])
    with pytest.raises(ValueError, match="with a PDE"):
        solver.load_restart(functions, {"has_pde": True})
    assert np.array_equal(solver.ode.values, init)


def test_load_restart_accepts_metadata_without_has_pde():
    """Checkpoints written before ``has_pde`` existed only carry the PDE's metadata"""
    a = build_restart_solver("godunov")
    run_steps(a, 0, 2)
    b = build_restart_solver("godunov")
    b.load_restart(b.restart_functions(["v", "s"]), {"pde": a.restart_metadata()["pde"]})


def test_load_restart_refuses_a_v_of_another_size_on_every_rank():
    solver = build_restart_solver("godunov")
    functions = solver.restart_functions(["v", "s"])
    coarse = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        3,
        3,
        dolfinx.cpp.mesh.CellType.triangle,
    )
    v_other = dolfinx.fem.Function(dolfinx.fem.functionspace(coarse, ("P", 1)))
    v_before = solver.v.x.array.copy()
    with pytest.raises(ValueError, match="does not fit"):
        solver.load_restart([("v", v_other), *functions[1:]], solver.restart_metadata())
    assert np.array_equal(solver.v.x.array, v_before)


def test_monitor_none_means_no_monitoring():
    solver, _ = build_ode_only(monitor=None)
    assert isinstance(solver.monitor, beat.NullMonitor)
    explicit = beat.SplittingSolver(pde=None, ode=solver.ode, monitor=None)
    assert isinstance(explicit.monitor, beat.NullMonitor)
    explicit.solve((0.0, 2 * ODE_ONLY_DT), ODE_ONLY_DT)


def test_solve_with_a_dt_longer_than_the_interval_raises():
    solver = build_restart_solver("godunov")
    with pytest.raises(ValueError, match="longer than"):
        solver.solve((0.0, 0.01), 0.02)
    with pytest.raises(ValueError, match="longer than"):
        solver.pde.solve((0.0, 0.01), 0.02)


def test_base_model_solve_steps_match_time_steps():
    solver = build_restart_solver("godunov")
    seen = []
    step = solver.pde.step
    solver.pde.step = lambda interval: (seen.append(interval), step(interval))[1]
    solver.pde.solve((0.0, 0.05), 0.01)
    assert seen == list(beat.utils.time_steps((0.0, 0.05), 0.01))


def _shared_monitor_solver(monitor):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    pde = beat.MonodomainModel(time=time, mesh=mesh, M=1.0, monitor=monitor)
    V_ode = beat.utils.space_from_string("P_1", mesh, dim=1)
    n = dolfinx.fem.Function(V_ode).x.array.size
    ode = beat.odesolver.DolfinODESolver(
        v_ode=dolfinx.fem.Function(V_ode),
        v_pde=pde.state,
        fun=simple_ode_forward_euler,
        init_states=np.zeros((2, n)),
        parameters=None,
        num_states=2,
        v_index=0,
    )
    return beat.SplittingSolver(pde=pde, ode=ode, monitor=monitor)


def test_splitting_solver_and_pde_sharing_a_monitor_count_each_step_once():
    monitor = RecordingMonitor()
    _shared_monitor_solver(monitor).solve((0.0, 0.3), dt=0.1)
    assert len(monitor.steps) == 3


def test_pde_solve_counts_each_step_once():
    monitor = RecordingMonitor()
    _shared_monitor_solver(monitor).pde.solve((0.0, 0.3), dt=0.1)
    assert len(monitor.steps) == 3
