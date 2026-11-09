from pathlib import Path

import numpy as np
import pytest

gotranx = pytest.importorskip("gotranx")

ODE_DIR = Path(__file__).parents[1] / "odes" / "tentusscher_panfilov_2006"


def apd90(name, dt=0.02, end=1000.0):
    ode = gotranx.load_ode(ODE_DIR / f"tentusscher_panfilov_2006_{name}_cell.ode")
    code = gotranx.cli.gotran2py.get_code(
        ode,
        scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
    )
    model: dict = {}
    exec(code, model)
    y = model["init_state_values"]()
    p = model["init_parameter_values"]()  # default pacing: 1 ms stimulus at t = 10 ms
    iv = model["state_index"]("V")
    step = model["generalized_rush_larsen"]
    n = int(end / dt)
    V = np.empty(n)
    t = 0.0
    for i in range(n):
        y = step(states=y, t=t, parameters=p, dt=dt)
        t += dt
        V[i] = y[iv]
    v90 = V.max() - 0.9 * (V.max() - V[: int(5 / dt)].min())
    up = np.argmax(V > v90)
    down = up + np.argmax(V[up:] < v90)
    return (down - up) * dt


def test_apd_ordering():
    epi, endo, m = apd90("epi"), apd90("endo"), apd90("M")
    assert m > 1.15 * epi and m > 1.15 * endo, (epi, endo, m)
    assert abs(endo - epi) < 0.1 * epi, (epi, endo, m)
