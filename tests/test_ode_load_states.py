from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import beat


def rhs(states, t, parameters, dt):
    return states


def make_single(V, v_pde):
    return beat.odesolver.DolfinODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=v_pde,
        fun=rhs,
        init_states=np.array([0.0, 1.0]),
        parameters=np.array([]),
        num_states=2,
        v_index=0,
    )


def make_multi(V, v_pde):
    markers = dolfinx.fem.Function(V)
    markers.x.array[: markers.x.array.size // 2] = 1
    markers.x.scatter_forward()
    return beat.odesolver.DolfinMultiODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=v_pde,
        markers=markers,
        init_states={0: np.array([0.0, 1.0]), 1: np.array([0.0, 1.0])},
        parameters={0: np.array([]), 1: np.array([])},
        fun={0: rhs, 1: rhs},
        num_states={0: 2, 1: 2},
        v_index={0: 0, 1: 0},
    )


@pytest.mark.parametrize("factory", [make_single, make_multi])
def test_load_all_states_roundtrip(factory):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    ode = factory(V, dolfinx.fem.Function(V))
    funcs = ode.states_to_dolfin(["v", "h"])
    rng = np.random.default_rng(0)
    for f in funcs:
        f.x.array[:] = rng.random(f.x.array.size)
    ode.load_all_states(funcs)
    again = ode.states_to_dolfin(["v", "h"])
    for a, b in zip(funcs, again):
        assert np.allclose(a.x.array, b.x.array)
    assert np.allclose(ode.v_ode.x.array, funcs[0].x.array)


@pytest.mark.parametrize("factory", [make_single, make_multi])
def test_load_all_states_then_step_starts_from_loaded_values(factory):
    """After load_all_states, ode.step must operate on the loaded values, not on
    whatever was there before. Uses an identity RHS (fun returns the states
    unchanged) so a step should leave the just-loaded values untouched."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    ode = factory(V, dolfinx.fem.Function(V))
    funcs = ode.states_to_dolfin(["v", "h"])
    rng = np.random.default_rng(1)
    for f in funcs:
        f.x.array[:] = rng.random(f.x.array.size)
        # Ensure ghost dofs are consistent with their owner (as they would be for
        # values loaded from an actual checkpoint), so the loaded values are
        # well-defined across ranks under MPI.
        f.x.scatter_forward()
    loaded = [np.copy(f.x.array) for f in funcs]
    ode.load_all_states(funcs)
    ode.step(0.0, 0.1)
    after = ode.states_to_dolfin(["v", "h"])
    for a, vals in zip(after, loaded):
        assert np.allclose(a.x.array, vals)


def zero_ode_ufl(states, t, parameters):
    """UFL RHS that is identically zero, so states stay constant under a step,
    regardless of the Butcher tableau used."""
    return [0 * s for s in states]


def make_irksome_single(V, v_pde, time, tableau):
    n_ode = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
    init_states = np.zeros((2, n_ode))
    init_states[0, :] = 0.0
    init_states[1, :] = 1.0
    return beat.IrksomeODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=v_pde,
        fun=zero_ode_ufl,
        init_states=init_states,
        butcher_tableau=tableau,
        time=time,
        num_states=2,
        v_index=0,
        parameters=np.array([]),
    )


def make_irksome_multi(V, v_pde, time, tableau):
    markers = dolfinx.fem.Function(V)
    markers.x.array[: markers.x.array.size // 2] = 1
    markers.x.scatter_forward()
    return beat.IrksomeMultiODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=v_pde,
        markers=markers,
        fun={0: zero_ode_ufl, 1: zero_ode_ufl},
        init_states={0: np.array([0.0, 1.0]), 1: np.array([0.0, 1.0])},
        butcher_tableau=tableau,
        time=time,
        num_states={0: 2, 1: 2},
        v_index={0: 0, 1: 0},
        parameters={0: np.array([]), 1: np.array([])},
    )


@pytest.mark.parametrize("factory", [make_irksome_single, make_irksome_multi])
def test_load_all_states_roundtrip_irksome(factory):
    irksome = pytest.importorskip("irksome")
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    time = dolfinx.fem.Constant(mesh, 0.0)
    ode = factory(V, dolfinx.fem.Function(V), time, irksome.RadauIIA(1))
    funcs = ode.states_to_dolfin(["v", "h"])
    rng = np.random.default_rng(2)
    for f in funcs:
        f.x.array[:] = rng.random(f.x.array.size)
    ode.load_all_states(funcs)
    again = ode.states_to_dolfin(["v", "h"])
    for a, b in zip(funcs, again):
        assert np.allclose(a.x.array, b.x.array)
    assert np.allclose(ode.v_ode.x.array, funcs[0].x.array)


@pytest.mark.parametrize("factory", [make_irksome_single, make_irksome_multi])
def test_load_all_states_then_step_starts_from_loaded_values_irksome(factory):
    irksome = pytest.importorskip("irksome")
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    time = dolfinx.fem.Constant(mesh, 0.0)
    ode = factory(V, dolfinx.fem.Function(V), time, irksome.RadauIIA(1))
    funcs = ode.states_to_dolfin(["v", "h"])
    rng = np.random.default_rng(3)
    for f in funcs:
        f.x.array[:] = rng.random(f.x.array.size)
        f.x.scatter_forward()
    loaded = [np.copy(f.x.array) for f in funcs]
    ode.load_all_states(funcs)
    ode.step(0.0, 0.1)
    after = ode.states_to_dolfin(["v", "h"])
    for a, vals in zip(after, loaded):
        assert np.allclose(a.x.array, vals)


def make_extop_single(V, v_pde):
    n_ode = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
    init_states = np.zeros((2, n_ode))
    init_states[0, :] = 0.0
    init_states[1, :] = 1.0
    return beat.ExternalOperatorODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=v_pde,
        fun=rhs,
        init_states=init_states,
        num_states=2,
        v_index=0,
        parameters=np.array([]),
    )


def make_extop_multi(V, v_pde):
    markers = dolfinx.fem.Function(V)
    markers.x.array[: markers.x.array.size // 2] = 1
    markers.x.scatter_forward()
    return beat.ExternalOperatorMultiODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=v_pde,
        markers=markers,
        fun={0: rhs, 1: rhs},
        init_states={0: np.array([0.0, 1.0]), 1: np.array([0.0, 1.0])},
        num_states={0: 2, 1: 2},
        v_index={0: 0, 1: 0},
        parameters={0: np.array([]), 1: np.array([])},
    )


@pytest.mark.parametrize("factory", [make_extop_single, make_extop_multi])
def test_load_all_states_roundtrip_external_operator(factory):
    pytest.importorskip("dolfinx_external_operator")
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    ode = factory(V, dolfinx.fem.Function(V))
    funcs = ode.states_to_dolfin(["v", "h"])
    rng = np.random.default_rng(4)
    for f in funcs:
        f.x.array[:] = rng.random(f.x.array.size)
    ode.load_all_states(funcs)
    again = ode.states_to_dolfin(["v", "h"])
    for a, b in zip(funcs, again):
        assert np.allclose(a.x.array, b.x.array)
    assert np.allclose(ode.v_ode.x.array, funcs[0].x.array)


@pytest.mark.parametrize("factory", [make_extop_single, make_extop_multi])
def test_load_all_states_then_step_starts_from_loaded_values_external_operator(factory):
    pytest.importorskip("dolfinx_external_operator")
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    ode = factory(V, dolfinx.fem.Function(V))
    funcs = ode.states_to_dolfin(["v", "h"])
    rng = np.random.default_rng(5)
    for f in funcs:
        f.x.array[:] = rng.random(f.x.array.size)
        f.x.scatter_forward()
    loaded = [np.copy(f.x.array) for f in funcs]
    ode.load_all_states(funcs)
    ode.step(0.0, 0.1)
    after = ode.states_to_dolfin(["v", "h"])
    for a, vals in zip(after, loaded):
        assert np.allclose(a.x.array, vals)
