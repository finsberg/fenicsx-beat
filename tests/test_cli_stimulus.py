from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl  # noqa: F401
from cli_helpers import minimal_config_dict

from beat.cli.config import Config, ConfigError
from beat.cli.geometry import build_geometry
from beat.cli.stimulus import build_stimuli


def setup(tmp_path, stimuli, geometry=None):
    data = minimal_config_dict(tmp_path, stimulus=stimuli)
    if geometry:
        data["geometry"] = geometry
    conf = Config.model_validate(data)
    geo = build_geometry(conf.geometry, MPI.COMM_WORLD)
    time = dolfinx.fem.Constant(geo.mesh, 0.0)
    s = build_stimuli(conf.stimulus, geo, conf.ep, time, conf.geometry.unit, conf.solver.t_end_ms())
    return s, time, geo


def integral(stim_set, time, t):
    time.value = t
    for u in stim_set.updates:
        u()
    total = 0.0
    for s in stim_set.stimuli:
        form = dolfinx.fem.form(s.expr * s.dz)
        total += dolfinx.fem.assemble_scalar(form)
    return MPI.COMM_WORLD.allreduce(total, op=MPI.SUM)


def test_marker_stimulus_single_pulse(tmp_path):
    s, time, _ = setup(
        tmp_path,
        [
            {
                "type": "marker",
                "marker": "X0",
                "amplitude": "1 uA/cm**2",
                "start": "1 ms",
                "duration": "1 ms",
            },
        ],
    )
    assert integral(s, time, 0.5) == 0.0
    assert integral(s, time, 1.5) > 0.0
    assert integral(s, time, 2.5) == 0.0


def test_pulse_train(tmp_path):
    s, time, _ = setup(
        tmp_path,
        [
            {
                "type": "box",
                "min": [0, 0],
                "max": [0.5, 0.5],
                "amplitude": "1 uA/cm**3",
                "duration": "1 ms",
                "period": "10 ms",
                "num_pulses": 2,
            },
        ],
    )
    assert integral(s, time, 0.5) > 0
    assert integral(s, time, 10.5) > 0
    assert integral(s, time, 20.5) == 0.0  # only 2 pulses


def test_box_stimulus_outside_mesh_errors(tmp_path):
    with pytest.raises(ConfigError, match="no cells"):
        setup(tmp_path, [{"type": "box", "min": [5, 5], "max": [6, 6], "amplitude": "1 uA/cm**3"}])


def test_box_dimension_mismatch(tmp_path):
    with pytest.raises(ConfigError, match="coordinates"):
        setup(tmp_path, [{"type": "box", "min": [0], "max": [1], "amplitude": "1 uA/cm**3"}])


def test_unknown_marker_errors(tmp_path):
    with pytest.raises(ConfigError, match="ENDO"):
        setup(tmp_path, [{"type": "marker", "marker": "ENDO", "amplitude": "1 uA/cm**2"}])


def test_wrong_amplitude_dimension_for_marker(tmp_path):
    # X0 is a facet of a 2D mesh -> effective dim 2 -> needs uA/cm**2
    with pytest.raises(ConfigError, match="amplitude"):
        setup(tmp_path, [{"type": "marker", "marker": "X0", "amplitude": "1 uA/cm**3"}])


def test_random_endocardial_is_rank_independent(tmp_path):
    s, time, _ = setup(
        tmp_path,
        [
            {
                "type": "random_endocardial",
                "markers": ["X0"],
                "num_points": 3,
                "seed": 1,
                "tol": 0.2,
                "amplitude": "1 uA/cm**3",
                "delay_range": ["0 ms", "1 ms"],
            },
        ],
    )
    val = integral(s, time, 1.5)
    assert val > 0.0
    assert np.isclose(MPI.COMM_WORLD.bcast(val, root=0), val)
