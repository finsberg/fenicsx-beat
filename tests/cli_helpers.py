"""Shared fixtures/data for the CLI tests."""

from typing import Any

MITCHELL_SCHAEFFER_ODE = """
parameters(
    tau_in = 0.3,
    tau_out = 6.0,
    tau_open = 120.0,
    tau_close = 150.0,
    v_gate = 0.13
)
states(
    v = 0.0,
    h = 1.0
)

h_inf = Conditional(Lt(v, v_gate), 1.0, 0.0)
tau_h = Conditional(Lt(v, v_gate), tau_open, tau_close)

dv_dt = h * (v**2 * (1.0 - v)) / tau_in - v / tau_out
dh_dt = (h_inf - h) / tau_h
"""


def minimal_config_dict(tmp_path, **overrides: Any) -> dict[str, Any]:
    """A tiny valid config (2D unit square, Mitchell-Schaeffer, 3 steps)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    ode = tmp_path / "ms.ode"
    ode.write_text(MITCHELL_SCHAEFFER_ODE)
    data: dict[str, Any] = {
        "geometry": {"type": "rectangle", "lx": 1.0, "ly": 1.0, "dx": 0.25, "unit": "mm"},
        "cell": {"ode_file": str(ode), "v_name": "v"},
        "stimulus": [
            {
                "type": "box",
                "min": [0.0, 0.0],
                "max": [0.3, 0.3],
                "amplitude": "50000 uA/cm**3",
                "duration": "1 ms",
            },
        ],
        "solver": {"dt": "0.1 ms", "end_time": "0.3 ms"},
        "output": {"folder": str(tmp_path / "output"), "save_every": "0.1 ms"},
    }
    for key, value in overrides.items():
        data[key] = {**data.get(key, {}), **value} if isinstance(value, dict) else value
    return data
