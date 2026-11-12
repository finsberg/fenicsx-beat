"""Shared behaviour of every ODE backend (characterization tests for the base-class refactor)."""

import gc

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import beat
import beat.utils
from beat.odesolver import DolfinMultiODESolver, DolfinODESolver

BACKENDS = [
    "dolfin",
    "dolfin_multi",
    "irksome",
    "irksome_multi",
    "external_operator",
    "external_operator_multi",
]


def numpy_fun(states, t, parameters, dt):
    return states


def ufl_fun(states, t, parameters):
    return [0 * states[0], 0 * states[1]]


def _spaces():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    return mesh, V


def _markers(V):
    markers = dolfinx.fem.Function(V)
    X = ufl.SpatialCoordinate(V.mesh)
    markers.interpolate(
        dolfinx.fem.Expression(
            ufl.conditional(ufl.lt(X[0], 0.5), 1, 2),
            beat.utils.interpolation_points(V),
        ),
    )
    return markers


def build(name):
    mesh, V = _spaces()
    v_ode, v_pde = dolfinx.fem.Function(V), dolfinx.fem.Function(V)
    x = V.tabulate_dof_coordinates()
    single = np.vstack([1.0 + x[:, 0], 2.0 + x[:, 1]])  # distinct per state and per point
    multi = {1: np.array([1.0, 2.0]), 2: np.array([3.0, 4.0])}
    if name == "dolfin":
        return DolfinODESolver(
            v_ode=v_ode,
            v_pde=v_pde,
            init_states=single.copy(),
            parameters=np.array([]),
            fun=numpy_fun,
            num_states=2,
        ), "single"
    if name == "dolfin_multi":
        return DolfinMultiODESolver(
            v_ode=v_ode,
            v_pde=v_pde,
            markers=_markers(V),
            init_states=multi,
            parameters={1: np.array([]), 2: np.array([0.0])},
            fun={1: numpy_fun, 2: numpy_fun},
            num_states={1: 2, 2: 2},
            v_index={1: 0, 2: 0},
        ), "multi"
    if name in ("irksome", "irksome_multi"):
        irksome = pytest.importorskip("irksome")
        from beat.irksome_odesolver import IrksomeMultiODESolver, IrksomeODESolver

        time = dolfinx.fem.Constant(mesh, 0.0)
        if name == "irksome":
            return IrksomeODESolver(
                v_ode=v_ode,
                v_pde=v_pde,
                fun=ufl_fun,
                init_states=single.copy(),
                butcher_tableau=irksome.BackwardEuler(),
                time=time,
                num_states=2,
            ), "single"
        return IrksomeMultiODESolver(
            v_ode=v_ode,
            v_pde=v_pde,
            markers=_markers(V),
            fun={1: ufl_fun, 2: ufl_fun},
            init_states=multi,
            butcher_tableau=irksome.BackwardEuler(),
            time=time,
            num_states={1: 2, 2: 2},
            v_index={1: 0, 2: 0},
            parameters={1: np.array([]), 2: np.array([0.0])},
        ), "multi"
    pytest.importorskip("dolfinx_external_operator")
    from beat.external_operator_odesolver import (
        ExternalOperatorMultiODESolver,
        ExternalOperatorODESolver,
    )

    if name == "external_operator":
        return ExternalOperatorODESolver(
            v_ode=v_ode,
            v_pde=v_pde,
            fun=numpy_fun,
            init_states=single.copy(),
            num_states=2,
        ), "single"
    return ExternalOperatorMultiODESolver(
        v_ode=v_ode,
        v_pde=v_pde,
        markers=_markers(V),
        fun={1: numpy_fun, 2: numpy_fun},
        init_states=multi,
        num_states={1: 2, 2: 2},
        v_index={1: 0, 2: 0},
        parameters={1: np.array([]), 2: np.array([0.0])},
    ), "multi"


@pytest.fixture(params=BACKENDS)
def backend(request):
    ode, kind = build(request.param)
    yield ode, kind
    comm = ode.v_ode.function_space.mesh.comm
    comm.Barrier()
    del ode
    comm.Barrier()
    gc.collect()
    comm.Barrier()


def expected(ode, kind):
    """The initial states as a (2, n) array on v_ode's dofs"""
    V = ode.v_ode.function_space
    if kind == "single":
        x = V.tabulate_dof_coordinates()
        return np.vstack([1.0 + x[:, 0], 2.0 + x[:, 1]])
    m = ode.markers.x.array
    out = np.zeros((2, m.size))
    out[:, m == 1] = np.array([[1.0], [2.0]])
    out[:, m == 2] = np.array([[3.0], [4.0]])
    return out


def test_full_values(backend):
    ode, kind = backend
    assert np.allclose(ode.full_values, expected(ode, kind))


def test_to_and_from_dolfin(backend):
    ode, kind = backend
    ode.to_dolfin()
    assert np.allclose(ode.v_ode.x.array, expected(ode, kind)[0])
    ode.v_ode.x.array[:] = 7.0
    ode.from_dolfin()
    assert np.allclose(ode.full_values[0], 7.0)
    assert np.allclose(ode.full_values[1], expected(ode, kind)[1])


def test_states_round_trip(backend):
    ode, kind = backend
    functions = ode.states_to_dolfin(["a", "b"])
    assert [f.name for f in functions] == ["a", "b"]
    for f, row in zip(functions, expected(ode, kind)):
        assert np.allclose(f.x.array, row)
    for f in functions:
        f.x.array[:] += 10.0
    ode.load_all_states(functions)
    assert np.allclose(ode.full_values, expected(ode, kind) + 10.0)
    assert np.allclose(ode.v_ode.x.array, expected(ode, kind)[0] + 10.0)  # load ends with to_dolfin
    assert [f.name for f in ode.states_to_dolfin()] == ["state_0", "state_1"]


def test_assign_all_states(backend):
    ode, kind = backend
    V = ode.v_ode.function_space
    functions = [dolfinx.fem.Function(V), dolfinx.fem.Function(V)]
    ode.assign_all_states(functions)
    for f, row in zip(functions, expected(ode, kind)):
        assert np.allclose(f.x.array, row)


def test_region_accessors(backend):
    ode, kind = backend
    if kind == "single":
        pytest.skip("region accessors exist on multi-region backends only")
    m = ode.markers.x.array
    for marker, (v0, s0) in ((1, (1.0, 2.0)), (2, (3.0, 4.0))):
        n = int((m == marker).sum())
        assert ode.num_points(marker) == n
        assert ode.shape(marker) == (2, n)
        assert np.allclose(ode.values(marker), np.array([[v0], [s0]]))
    assert ode.num_parameters(1) == 0 and ode.num_parameters(2) == 1


def test_wrong_number_of_names_or_functions_raises(backend):
    ode, _ = backend
    V = ode.v_ode.function_space
    with pytest.raises(ValueError, match="must match number of states"):
        ode.states_to_dolfin(["only_one"])
    with pytest.raises(ValueError, match="must match number of states"):
        ode.assign_all_states([dolfinx.fem.Function(V)])
    with pytest.raises(ValueError, match="must match number of states"):
        ode.load_all_states([dolfinx.fem.Function(V)])


def test_monitor_none_is_a_null_monitor():
    mesh, V = _spaces()
    ode = DolfinODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=dolfinx.fem.Function(V),
        init_states=np.zeros(2),
        parameters=np.array([]),
        fun=numpy_fun,
        num_states=2,
        monitor=None,
    )
    assert isinstance(ode.monitor, beat.NullMonitor)
    ode.step(0.0, 0.1)


def test_subclass_overriding_the_old_methods_still_works():
    """A user backend written against the old abstract methods needs no primitives"""
    from beat.odesolver import BaseDolfinODESolver

    class OldStyle(BaseDolfinODESolver):
        def __init__(self, v):
            self.v_ode = self.v_pde = v
            self._values = np.ones((1, v.x.array.size))

        def to_dolfin(self):
            self.v_ode.x.array[:] = self._values[0]

        def from_dolfin(self):
            self._values[0] = self.v_ode.x.array

        def step(self, t0, dt):
            self._values *= 2.0

        @property
        def full_values(self):
            return self._values

        def assign_all_states(self, functions):
            functions[0].x.array[:] = self._values[0]

        def load_all_states(self, functions):
            self._values[0] = functions[0].x.array

        def states_to_dolfin(self, names=None):
            f = dolfinx.fem.Function(self.v_ode.function_space)
            self.assign_all_states([f])
            return [f]

    mesh, V = _spaces()
    ode = OldStyle(dolfinx.fem.Function(V))
    ode.step(0.0, 1.0)
    ode.to_dolfin()
    assert np.allclose(ode.v_ode.x.array, 2.0)


@pytest.mark.parametrize("name", ["dolfin_multi", "irksome_multi", "external_operator_multi"])
def test_dofs_outside_every_region_are_left_alone(name):
    ode, _ = build(name)
    comm = ode.v_ode.function_space.mesh.comm
    m = ode.markers.x.array
    outside = m == 2
    # Drop region 2 from the bookkeeping: its dofs are now covered by no region
    ode._marker_values = (1,)
    ode.v_ode.x.array[:] = -5.0
    ode.to_dolfin()
    assert np.allclose(ode.v_ode.x.array[outside], -5.0)
    assert np.allclose(ode.v_ode.x.array[m == 1], 1.0)
    assert np.allclose(ode.full_values[:, outside], 0.0)
    comm.Barrier()
    del ode
    comm.Barrier()
    gc.collect()
    comm.Barrier()


def test_dolfin_multi_values_is_the_live_array():
    ode, _ = build("dolfin_multi")
    ode.values(1)[0, :] = 9.0
    assert np.allclose(ode.full_values[0, ode.markers.x.array == 1], 9.0)
