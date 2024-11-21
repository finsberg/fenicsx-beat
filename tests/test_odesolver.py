import numpy as np

from beat.odesolver import ODESystemSolver


def test_simple_ode_odesystemsolver():
    def simple_ode_forward_euler(states, t, dt, parameters):
        v, s = states
        values = np.zeros_like(states)
        values[0] = v - s * dt
        values[1] = s + v * dt
        return values

    num_points = 1

    t_bound = 1.0
    t0 = 0.0
    x = np.arange(0.1, t_bound + 0.1, 0.1)
    y = np.zeros((len(x), 2))
    sol = np.vstack((np.cos(x), np.sin(x))).T

    errors = []
    for dt in [0.1, 0.01, 0.001, 0.0001]:
        states = np.zeros((2, num_points))
        states.T[:] = [1, 0]
        ode = ODESystemSolver(
            fun=simple_ode_forward_euler,
            states=states,
            parameters=None,
        )
        j = 0
        t = 0.0
        for _ in range(int((t_bound - t0) / dt)):
            ode.step(t, dt)
            t += dt
            if np.isclose(t, x[j]):
                print(t, j)
                y[j, :] = ode.states[:, 0]
                j += 1
        errors.append(np.linalg.norm(sol - y))
    rates = [np.log(e1 / e2) / np.log(10) for e1, e2 in zip(errors[:-1], errors[1:])]
    assert np.allclose(rates, 1, atol=0.01)
