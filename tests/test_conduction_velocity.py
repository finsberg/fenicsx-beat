import numpy as np
import pytest

import beat
from beat.conduction_velocity import (
    calibrate_conductivities,
    measure_conduction_velocity,
    scale_conductivities,
)
from beat.conductivities import get_harmonic_mean_conductivity

# A bistable (Nagumo) cell, dv/dt = -K (v - V_REST)(v - V_C)(v - V_PEAK), whose travelling wave
# has the speed sqrt(K D / 2) (V_PEAK + V_REST - 2 V_C) for the diffusivity D = s / C_m
V_REST, V_C, V_PEAK, K = -85.0, -60.0, 30.0, 1e-4
C_M = 0.01  # 1 uF/cm^2 in uF/mm^2


def bistable(states, t, parameters, dt):
    v = states[0]
    return (v - dt * K * (v - V_REST) * (v - V_C) * (v - V_PEAK))[None, :]


CELL = dict(fun=bistable, init_states=np.array([V_REST]), parameters=np.array([]), v_index=0)


def exact_speed(conductivities, direction):
    s_l, s_t = get_harmonic_mean_conductivity(**conductivities)
    s = s_l if direction == "longitudinal" else s_t
    return np.sqrt(K * (s / C_M) / 2) * (V_PEAK + V_REST - 2 * V_C)


@pytest.fixture
def bishop():
    return beat.conductivities.default_conductivities("Bishop")


@pytest.mark.parametrize("direction", ["longitudinal", "transverse"])
def test_measured_velocity_converges_to_exact(bishop, direction):
    exact = exact_speed(bishop, direction)
    errors = [
        abs(
            measure_conduction_velocity(
                conductivities=bishop, dx=dx, dt=0.01, direction=direction, **CELL
            )
            / exact
            - 1.0
        )
        for dx in (0.25, 0.1)
    ]
    assert errors[1] < errors[0]
    assert errors[1] < 0.01


def test_scale_conductivities(bishop):
    scaled = scale_conductivities(bishop, longitudinal=2.0, transverse=3.0)
    for key, factor in dict(g_il=2.0, g_el=2.0, g_it=3.0, g_et=3.0, chi=1.0).items():
        assert np.isclose((scaled[key] / bishop[key]).to("").magnitude, factor)
    s_l, s_t = get_harmonic_mean_conductivity(**bishop)
    s_l2, s_t2 = get_harmonic_mean_conductivity(**scaled)
    assert np.isclose(s_l2, 2.0 * s_l) and np.isclose(s_t2, 3.0 * s_t)


def test_calibrate_reaches_the_targets(bishop):
    ureg = beat.units.ureg
    targets = {"longitudinal": 0.15, "transverse": 0.05}
    result = calibrate_conductivities(
        ureg.Quantity(0.15, "mm/ms"),
        targets["transverse"],
        conductivities=bishop,
        dx=0.1,
        dt=0.01,
        rtol=0.01,
        **CELL,
    )
    for direction, target in targets.items():
        assert abs(result.velocities[direction] / target - 1.0) <= 0.01
        # The wave is resolved, so the velocity goes as the square root of the conductivity
        expected = (target / exact_speed(bishop, direction)) ** 2
        assert np.isclose(result.scales[direction], expected, rtol=0.05)
    assert result.history[-1] == (result.scales, result.velocities)
    assert np.isclose(exact_speed(result.conductivities, "longitudinal"), 0.15, rtol=0.05)


def test_block_gives_zero(bishop):
    # No stimulus, no wave: the measurement stops early and reports a block
    c = measure_conduction_velocity(conductivities=bishop, dx=0.25, dt=0.01, amplitude=0.0, **CELL)
    assert c == 0.0


def test_calibrate_warns_when_not_converged(bishop):
    with pytest.warns(RuntimeWarning, match="did not reach"):
        result = calibrate_conductivities(
            0.15, 0.05, conductivities=bishop, dx=0.25, dt=0.01, max_iterations=1, **CELL
        )
    assert result.scales == {"longitudinal": 1.0, "transverse": 1.0}


def test_invalid_direction(bishop):
    with pytest.raises(ValueError, match="direction"):
        measure_conduction_velocity(conductivities=bishop, dx=0.25, direction="sheet", **CELL)
