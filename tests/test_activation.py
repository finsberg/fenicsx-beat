from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

from beat.activation import ActivationTracker

C, TAU, V_REST, V_AMP = 0.5, 0.5, -85.0, 115.0


def wave(x, t):
    return V_REST + V_AMP * 0.5 * (1.0 + np.tanh((t - x / C) / TAU))


def simulate(method, dt=0.05, end=12.0):
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 40, [0.0, 4.0])
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    v = dolfinx.fem.Function(V)
    x = V.tabulate_dof_coordinates()[:, 0]
    v.x.array[:] = wave(x, 0.0)
    tracker = ActivationTracker(v, method=method, threshold=-30.0, t0=0.0)
    t = 0.0
    while t < end - 1e-12:
        t += dt
        v.x.array[:] = wave(x, t)
        tracker.update(t)
    return x, tracker.t_a, dt


def test_max_dvdt_recovers_the_wave():
    x, t_a, dt = simulate("max_dvdt")
    inside = x > 0.5  # skip the start, where the upstroke began before t = 0
    assert np.abs(t_a.x.array[inside] - x[inside] / C).max() <= dt
    assert t_a.name == "activation_time"


def test_threshold_crossing_recovers_the_wave():
    x, t_a, dt = simulate("threshold")
    shift = TAU * np.arctanh(2 * (-30.0 - V_REST) / V_AMP - 1.0)
    inside = x > 0.5
    assert np.abs(t_a.x.array[inside] - (x[inside] / C + shift)).max() <= 0.1 * dt


@pytest.mark.parametrize("method", ["max_dvdt", "threshold"])
def test_never_activated_is_nan(method):
    x, t_a, _ = simulate(method, end=4.0)  # the front has reached x ~ 2 at t = 4
    assert np.isnan(t_a.x.array[x > 3.0]).all()
    assert np.isfinite(t_a.x.array[(x > 0.5) & (x < 1.5)]).all()


def test_invalid_input():
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    v = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1)))
    with pytest.raises(ValueError, match="method"):
        ActivationTracker(v, method="peak")
    tracker = ActivationTracker(v, t0=1.0)
    with pytest.raises(ValueError, match="increase"):
        tracker.update(1.0)


@pytest.mark.parametrize("method", ["max_dvdt", "threshold"])
def test_ghosts_equal_owners(method):
    _, t_a, _ = simulate(method)
    before = t_a.x.array.copy()
    t_a.x.scatter_forward()
    np.testing.assert_array_equal(before, t_a.x.array)
