from pathlib import Path

from mpi4py import MPI

import numpy as np
import pytest

import beat.c_backend

comm = MPI.COMM_WORLD

# A two-state forward Euler step with a time-dependent term, in the gotranx per-cell signature
TOY_C = r"""
#include <math.h>
void toy(const double *__restrict states, const double t, const double dt,
         const double *__restrict parameters, double *values)
{
    values[0] = states[0] - parameters[0] * states[1] * dt;
    values[1] = states[1] + parameters[1] * states[0] * dt + t * dt;
}
"""


def toy_numpy(states, t, parameters, dt):
    values = np.empty_like(states)
    values[0] = states[0] - parameters[0] * states[1] * dt
    values[1] = states[1] + parameters[1] * states[0] * dt + t * dt
    return values


@pytest.fixture
def cache_dir(tmp_path):
    # All ranks must use rank 0's directory
    return Path(comm.bcast(str(tmp_path), root=0))


@pytest.fixture
def toy(cache_dir):
    return beat.c_backend.compile_scheme(
        TOY_C,
        "toy",
        num_states=2,
        num_parameters=2,
        cache_dir=cache_dir,
    )


def random_states(n_points, seed=1):
    return np.random.default_rng(seed).uniform(-1, 1, size=(2, n_points))


def test_matches_numpy_2d(toy):
    states = random_states(37)
    parameters = np.array([1.5, 0.5])
    result = toy(states=states, t=0.3, parameters=parameters, dt=0.1)
    assert result.shape == states.shape
    assert result is not states
    np.testing.assert_allclose(result, toy_numpy(states, 0.3, parameters, 0.1), rtol=1e-14)


def test_matches_numpy_1d_states(toy):
    states = random_states(1)[:, 0].copy()
    parameters = np.array([1.5, 0.5])
    result = toy(states=states, t=0.3, parameters=parameters, dt=0.1)
    np.testing.assert_allclose(result, toy_numpy(states, 0.3, parameters, 0.1), rtol=1e-14)


def test_spatially_varying_parameters(toy):
    states = random_states(11)
    parameters = np.random.default_rng(2).uniform(0, 2, size=(2, 11))
    result = toy(states=states, t=0.0, parameters=parameters, dt=0.1)
    np.testing.assert_allclose(result, toy_numpy(states, 0.0, parameters, 0.1), rtol=1e-14)


def test_out_argument(toy):
    states = random_states(5)
    parameters = np.array([1.0, 1.0])
    out = np.zeros_like(states)
    result = toy(states=states, t=0.0, parameters=parameters, dt=0.1, out=out)
    assert result is out
    np.testing.assert_allclose(out, toy_numpy(states, 0.0, parameters, 0.1), rtol=1e-14)


def test_out_aliasing_states(toy):
    states = random_states(9)
    expected = toy_numpy(states, 0.2, np.array([1.0, 2.0]), 0.1)
    toy(states=states, t=0.2, parameters=np.array([1.0, 2.0]), dt=0.1, out=states)
    np.testing.assert_allclose(states, expected, rtol=1e-14)


@pytest.mark.parametrize(
    "states, parameters, error",
    [
        (np.zeros((2, 4), dtype=np.float32), np.ones(2), TypeError),
        (np.zeros((4, 2)).T, np.ones(2), ValueError),  # not C-contiguous
        (np.zeros((2, 8))[:, ::2], np.ones(2), ValueError),  # strided
        (np.zeros((2, 4)), np.array([1, 1]), TypeError),  # int parameters
        (np.zeros((2, 4, 1)), np.ones(2), ValueError),  # 3D
        (np.zeros((2, 4)), np.ones((2, 3)), ValueError),  # parameter points mismatch
        ([[0.0], [0.0]], np.ones(2), TypeError),  # not an ndarray
    ],
)
def test_rejects_bad_arrays(toy, states, parameters, error):
    with pytest.raises(error):
        toy(states=states, t=0.0, parameters=parameters, dt=0.1)


def test_rejects_bad_out(toy):
    with pytest.raises(ValueError):
        toy(states=np.zeros((2, 4)), t=0.0, parameters=np.ones(2), dt=0.1, out=np.zeros((2, 5)))


def test_wrong_number_of_states_or_parameters(toy):
    with pytest.raises(ValueError, match="states"):
        toy(states=np.zeros((3, 4)), t=0.0, parameters=np.ones(2), dt=0.1)
    with pytest.raises(ValueError, match="parameters"):
        toy(states=np.zeros((2, 4)), t=0.0, parameters=np.ones(5), dt=0.1)


def test_library_in_cache_dir(toy, cache_dir):
    assert toy.library_path.is_file()
    assert toy.library_path.parent.parent == cache_dir.resolve()
    assert toy.scheme == "toy"


def test_rank0_os_error_raises_on_all_ranks(cache_dir):
    # A regular file where a directory is needed makes mkdir fail on rank 0 (chmod does not
    # work as root). Every rank must raise instead of the others hanging in the broadcast.
    blocker = cache_dir / "blocker"
    if comm.rank == 0:
        blocker.write_text("")
    comm.barrier()
    with pytest.raises(RuntimeError, match="Error"):
        beat.c_backend.compile_scheme(
            TOY_C,
            "toy",
            num_states=2,
            num_parameters=2,
            cache_dir=blocker / "sub",
        )
