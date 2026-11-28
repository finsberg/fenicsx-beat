from pathlib import Path

import numpy as np
import pytest

from beat import single_cell
from beat.single_cell import get_steady_state


def _linear_step(
    states: np.ndarray,
    t: float,
    parameters: np.ndarray,
    dt: float,
) -> np.ndarray:
    return states + parameters * dt


@pytest.mark.skip_in_parallel
def test_get_steady_state_single_tracked_index(tmp_path: Path):
    init_states = np.array([0.0, 1.0], dtype=float)
    parameters = np.array([1.0, -0.5], dtype=float)

    y = get_steady_state(
        fun=_linear_step,
        init_states=init_states.copy(),
        parameters=parameters,
        outdir=tmp_path,
        nbeats=2,
        BCL=10,
        save_every_ms=1.0,
        dt=0.5,
        track_indices=[0],
    )

    np.testing.assert_allclose(y, [20.0, -9.0])
    assert len(list(tmp_path.glob("tracked_values_*.npy"))) == 1
    assert len(list(tmp_path.glob("tracked_values_*.png"))) == 1


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize(
    ("bcl", "dt", "save_every_ms", "nbeats"),
    [
        (10, 0.05, 0.33, 2),
        (10, 0.5, 20.0, 3),
    ],
)
def test_get_steady_state_last_beat_plot_alignment(
    tmp_path: Path,
    bcl: int,
    dt: float,
    save_every_ms: float,
    nbeats: int,
):
    init_states = np.array([0.0, 1.0], dtype=float)
    parameters = np.array([1.0, 0.5], dtype=float)

    y = get_steady_state(
        fun=_linear_step,
        init_states=init_states.copy(),
        parameters=parameters,
        outdir=tmp_path,
        nbeats=nbeats,
        BCL=bcl,
        save_every_ms=save_every_ms,
        dt=dt,
        track_indices=[0, 1],
    )

    np.testing.assert_allclose(y, init_states + parameters * (bcl * nbeats))
    tracked_files = list(tmp_path.glob("tracked_values_*.npy"))
    assert len(tracked_files) == 1
    tracked = np.load(tracked_files[0])
    save_freq = int(np.ceil(save_every_ms / dt))
    expected_per_beat = len(np.arange(0.0, bcl, dt)[::save_freq])
    assert tracked.shape == (expected_per_beat * nbeats, 2)
    assert len(list(tmp_path.glob("tracked_values_*.png"))) == 1


@pytest.mark.skip_in_parallel
def test_get_steady_state_without_matplotlib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(single_cell, "plt", None)
    init_states = np.array([0.0, 1.0], dtype=float)
    parameters = np.array([1.0, -0.5], dtype=float)

    y = get_steady_state(
        fun=_linear_step,
        init_states=init_states.copy(),
        parameters=parameters,
        outdir=tmp_path,
        nbeats=2,
        BCL=10,
        save_every_ms=1.0,
        dt=0.5,
        track_indices=[0],
    )

    np.testing.assert_allclose(y, [20.0, -9.0])
    assert len(list(tmp_path.glob("tracked_values_*.npy"))) == 1
    assert len(list(tmp_path.glob("tracked_values_*.png"))) == 0
