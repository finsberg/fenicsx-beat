import contextlib

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pytest

import beat
from beat.odesolver import DolfinMultiODESolver, DolfinODESolver
from beat.reaction_eikonal import (
    ExponentialFoot,
    FootCurrentODESolver,
    exponential_foot_rate,
)


def identity(states, t, parameters, dt):
    return states


def constant(V, value):
    f = dolfinx.fem.Function(V)
    f.x.array[:] = value
    return f


def make_ode(V, v0=-85.0, v_pde=None):
    return DolfinODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=v_pde if v_pde is not None else dolfinx.fem.Function(V),
        init_states=np.array([v0]),
        parameters=np.array([]),
        fun=identity,
        num_states=1,
        v_index=0,
    )


@pytest.fixture
def V():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    return dolfinx.fem.functionspace(mesh, ("P", 1))


def test_rate_window_and_threshold():
    foot = ExponentialFoot()
    dT = np.array([-0.1, 0.0, 1.0, 5.0, 5.1, 1.0, np.inf, -np.inf, np.nan])
    v = np.array([-85, -85, -85, -85, -85, -30, -85, -85, -85.0])
    expected = np.zeros_like(dT)
    for i in (1, 2, 3):
        expected[i] = 0.91 / 0.25 * np.exp(dT[i] / 0.25)
    np.testing.assert_allclose(foot.rate(dT, v), expected)
    np.testing.assert_allclose(
        exponential_foot_rate(1.0, -85.0, 2.0, 0.5, -30.0, 5.0),
        2.0 / 0.5 * np.exp(2.0),
    )


def test_step_is_explicit_euler_of_the_foot(V):
    t_a = constant(V, 0.5)
    ode = FootCurrentODESolver(make_ode(V), t_a)
    v = np.full(V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts, -85.0)
    t, dt = 0.0, 0.01
    for _ in range(300):
        ode.step(t, dt)
        v = v + dt * ExponentialFoot().rate(t + dt - 0.5, v)
        t += dt
        np.testing.assert_allclose(ode.v_ode.x.array, v, rtol=0, atol=1e-12)
    assert (v >= -30.0).all() and (v < -30.0 + 10.0).all()  # stopped once above V_th


def test_strang_corrective_step_also_applies_the_foot(V):
    t_a = 0.2037
    time = dolfinx.fem.Constant(V.mesh, 0.0)
    pde = beat.MonodomainModel(time=time, mesh=V.mesh, M=0.0)
    ode = FootCurrentODESolver(make_ode(V, v_pde=pde.state), constant(V, t_a))
    solver = beat.SplittingSolver(pde=pde, ode=ode, theta=0.5)
    v = np.full(V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts, -85.0)
    t, dt = 0.0, 0.02
    rate = ExponentialFoot().rate
    for _ in range(40):
        solver.step((t, t + dt))
        v = v + 0.5 * dt * rate(t + 0.5 * dt - t_a, v)
        v = v + 0.5 * dt * rate(t + dt - t_a, v)
        t += dt
    np.testing.assert_allclose(pde.v.x.array, v, rtol=0, atol=1e-8)


def test_per_region_feet(V):
    markers = dolfinx.fem.Function(V)
    markers.interpolate(lambda x: np.where(x[0] < 0.5, 1.0, 2.0))
    inner = DolfinMultiODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=dolfinx.fem.Function(V),
        markers=markers,
        init_states={1: np.array([-85.0]), 2: np.array([-85.0])},
        parameters={1: np.array([]), 2: np.array([])},
        fun={1: identity, 2: identity},
        num_states={1: 1, 2: 1},
        v_index={1: 0, 2: 0},
    )
    feet = {1: ExponentialFoot(A=0.91), 2: ExponentialFoot(A=2.0)}
    ode = FootCurrentODESolver(inner, constant(V, 0.0), feet, markers=markers)
    ode.step(0.0, 0.01)
    v = ode.v_ode.x.array
    one = markers.x.array == 1
    np.testing.assert_allclose(v[one], -85.0 + 0.01 * 0.91 / 0.25 * np.exp(0.01 / 0.25))
    np.testing.assert_allclose(v[~one], -85.0 + 0.01 * 2.0 / 0.25 * np.exp(0.01 / 0.25))


def test_missing_marker_raises(V):
    markers = constant(V, 3.0)
    with pytest.raises(ValueError, match="marker 3"):
        FootCurrentODESolver(make_ode(V), constant(V, 0.0), {1: ExponentialFoot()}, markers=markers)


def test_dict_foot_needs_markers(V):
    with pytest.raises(TypeError, match="markers"):
        FootCurrentODESolver(make_ode(V), constant(V, 0.0), {1: ExponentialFoot()})


def test_unreached_t_a(V):
    t_a = constant(V, 0.0)
    x = V.tabulate_dof_coordinates()
    t_a.x.array[x[:, 0] > 0.5] = np.inf
    with pytest.raises(ValueError, match="not reached"):
        FootCurrentODESolver(make_ode(V), t_a)
    ode = FootCurrentODESolver(make_ode(V), t_a, allow_unreached=True)
    ode.step(0.0, 0.01)
    assert (ode.v_ode.x.array[x[:, 0] > 0.5] == -85.0).all()
    assert (ode.v_ode.x.array[x[:, 0] < 0.4] > -85.0).all()


def test_unreached_t_a_into_quadrature_space(V):
    t_a = constant(V, 0.0)
    x = V.tabulate_dof_coordinates()
    t_a.x.array[x[:, 0] > 0.5] = np.inf
    element = basix.ufl.quadrature_element(V.mesh.basix_cell(), value_shape=(), degree=2)
    Q = dolfinx.fem.functionspace(V.mesh, element)
    ode = FootCurrentODESolver(make_ode(Q), t_a, allow_unreached=True)
    ode.step(0.0, 0.01)
    v = ode.v_ode.x.array
    assert np.isfinite(v).all()
    assert ((v == -85.0) | (v > -85.0)).all() and (v > -85.0).any() and (v == -85.0).any()


def test_unreached_vertex_into_quadrature_space_raises(V):
    t_a = constant(V, 0.0)
    x = V.tabulate_dof_coordinates()
    # exactly one unreached vertex (owned by one rank only)
    t_a.x.array[np.isclose(x[:, 0], 0.5) & np.isclose(x[:, 1], 0.5)] = np.inf
    element = basix.ufl.quadrature_element(V.mesh.basix_cell(), value_shape=(), degree=2)
    Q = dolfinx.fem.functionspace(V.mesh, element)
    with pytest.raises(ValueError, match="not reached"):
        FootCurrentODESolver(make_ode(Q), t_a)


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


# Annotated on purpose: mypy then checks the body, i.e. that FootCurrentODESolver satisfies
# the ODESolver protocol (v_ode/v_pde included). Keep the annotations.
def test_re_minus_solver(V: dolfinx.fem.FunctionSpace) -> None:
    monitor = RecordingMonitor()
    ode = FootCurrentODESolver(make_ode(V), constant(V, 0.0))
    solver = beat.SplittingSolver(pde=None, ode=ode, monitor=monitor)
    assert solver.v is ode.v_ode
    solver.solve((0.0, 2.0), dt=0.1)
    assert len(monitor.steps) == 20
    assert "ode_step" in monitor.names
    assert (solver.v.x.array >= -30.0).all()


def test_re_minus_restart_is_bit_for_bit(V):
    def build():
        return beat.SplittingSolver(
            pde=None,
            ode=FootCurrentODESolver(make_ode(V), constant(V, 0.0)),
        )

    a = build()
    a.solve((0.0, 0.5), dt=0.125)
    saved = [f.x.array.copy() for _, f in a.restart_functions(["v"])]
    b = build()
    functions = b.restart_functions(["v"])
    for (_, f), arr in zip(functions, saved):
        f.x.array[:] = arr
    b.load_restart(functions, a.restart_metadata())
    a.solve((0.5, 1.0), dt=0.125)
    b.solve((0.5, 1.0), dt=0.125)
    assert np.array_equal(a.v.x.array, b.v.x.array)
    assert not np.array_equal(a.v.x.array, constant(V, -85.0).x.array)


def test_non_integer_marker_raises(V):
    markers = constant(V, 1.5)
    with pytest.raises(ValueError, match="integer key"):
        FootCurrentODESolver(make_ode(V), constant(V, 0.0), {1: ExponentialFoot()}, markers=markers)


def test_wrapper_exposes_the_wrapped_solver_but_forwards_nothing_else(V):
    inner = make_ode(V)
    ode = FootCurrentODESolver(inner, constant(V, 0.0))
    assert ode.ode is inner
    assert ode.v_ode is inner.v_ode
    assert ode.v_pde is inner.v_pde
    with pytest.raises(AttributeError):
        ode.full_values


def test_period_none_is_unchanged(V):
    a = FootCurrentODESolver(make_ode(V), constant(V, 0.5))
    b = FootCurrentODESolver(make_ode(V), constant(V, 0.5), period=None)
    t, dt = 0.0, 0.01
    for _ in range(300):
        a.step(t, dt)
        b.step(t, dt)
        t += dt
    assert np.array_equal(a.v_ode.x.array, b.v_ode.x.array)


def test_foot_fires_every_period(V):
    t_a, P = 0.5, 10.0
    ode = FootCurrentODESolver(make_ode(V), constant(V, t_a), period=P)
    n = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
    v = np.full(n, -85.0)
    dt = 0.01
    seen = {}
    for i in range(2500):
        t = i * dt
        if i in (800, 1800):
            ode.v_ode.x.array[:] = -85.0
            ode.from_dolfin()
            v[:] = -85.0
        ode.step(t, dt)
        v = v + dt * ExponentialFoot().rate(np.mod(t + dt - t_a, P), v)
        np.testing.assert_allclose(ode.v_ode.x.array, v, rtol=0, atol=1e-12)
        for probe in (600, 1600, 2499):
            if i == probe:
                seen[probe] = ode.v_ode.x.array.copy()
    for probe in (600, 1600, 2499):
        assert (seen[probe] >= -30.0).all()


def test_never_fires_before_t_a(V):
    ode = FootCurrentODESolver(
        make_ode(V),
        constant(V, 5.0),
        ExponentialFoot(duration=1.0),
        period=2.0,
    )
    t, dt = 0.0, 0.01
    while t + dt < 5.0 - 1e-9:
        ode.step(t, dt)
        t += dt
        assert (ode.v_ode.x.array == -85.0).all()
    for _ in range(200):
        ode.step(t, dt)
        t += dt
    assert (ode.v_ode.x.array > -85.0).all()


def test_unreached_never_fires_with_period(V):
    t_a = constant(V, 0.0)
    x = V.tabulate_dof_coordinates()
    t_a.x.array[x[:, 0] > 0.5] = np.inf
    ode = FootCurrentODESolver(make_ode(V), t_a, allow_unreached=True, period=10.0)
    t, dt = 0.0, 0.01
    for _ in range(1500):
        ode.step(t, dt)
        t += dt
    assert (ode.v_ode.x.array[x[:, 0] > 0.5] == -85.0).all()
    assert (ode.v_ode.x.array[x[:, 0] < 0.4] > -85.0).all()


def test_period_must_exceed_duration(V):
    with pytest.raises(ValueError, match="period"):
        FootCurrentODESolver(make_ode(V), constant(V, 0.0), period=5.0)
    with pytest.raises(ValueError, match="period"):
        FootCurrentODESolver(make_ode(V), constant(V, 0.0), period=0.0)
    markers = constant(V, 1.0)
    with pytest.raises(ValueError, match="period"):
        FootCurrentODESolver(
            make_ode(V),
            constant(V, 0.0),
            {1: ExponentialFoot(duration=1.0), 2: ExponentialFoot(duration=4.0)},
            markers,
            period=3.0,
        )


def test_re_minus_restart_with_period_is_bit_for_bit(V):
    def build():
        return beat.SplittingSolver(
            pde=None,
            ode=FootCurrentODESolver(make_ode(V), constant(V, 0.0), period=10.0),
        )

    a = build()
    a.solve((0.0, 0.5), dt=0.125)
    saved = [f.x.array.copy() for _, f in a.restart_functions(["v"])]
    b = build()
    functions = b.restart_functions(["v"])
    for (_, f), arr in zip(functions, saved):
        f.x.array[:] = arr
    b.load_restart(functions, a.restart_metadata())
    a.solve((0.5, 1.0), dt=0.125)
    b.solve((0.5, 1.0), dt=0.125)
    assert np.array_equal(a.v.x.array, b.v.x.array)
    assert not np.array_equal(a.v.x.array, constant(V, -85.0).x.array)
