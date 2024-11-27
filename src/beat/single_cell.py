import hashlib
import logging
import warnings
from pathlib import Path
from typing import Callable

import numpy as np


class NumbaWarning(UserWarning):
    pass


class PlottingWarning(UserWarning):
    pass


try:
    from numba import jit
except ImportError:

    def jit(*args, **kwargs):
        return lambda x: x

    warnings.warn(
        "Numba not installed, falling back to slower execution.",
        category=NumbaWarning,
    )

try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn(
        "Matplotlib not installed, plotting not available.",
        category=PlottingWarning,
    )
    plt = None  # type: ignore

logger = logging.getLogger(__name__)


@jit(nopython=True)
def solve_with_save(fun, nbeats, times, y, p, dt, save_freq, track_values, track_indices):
    k = 0
    for _ in range(nbeats):
        j = 0
        for t in times:
            if j % save_freq == 0:
                i = 0
                for index in track_indices:
                    track_values[k, i] = y[index]
                    i += 1
                k += 1

            y[:] = fun(states=y, t=t, parameters=p, dt=dt)
            j += 1
    return y, track_values


@jit(nopython=True)
def solve_without_save(fun, nbeats, times, y, p, dt):
    for _ in range(nbeats):
        for t in times:
            y[:] = fun(states=y, t=t, parameters=p, dt=dt)
    return y


def compute_hash(
    fun: Callable[[np.ndarray, float, np.ndarray, float], np.ndarray],
    init_states: np.ndarray,
    parameters: np.ndarray,
    nbeats: int = 200,
    BCL: float = 1000.0,
    dt: float = 0.05,
):
    hash_input = hashlib.md5()
    hash_input.update(fun.__code__.co_code)
    hash_input.update(str(init_states).encode())
    hash_input.update(str(parameters).encode())
    hash_input.update(str(nbeats).encode())
    hash_input.update(str(BCL).encode())
    hash_input.update(str(dt).encode())
    return hash_input.hexdigest()


def get_steady_state(
    fun: Callable[[np.ndarray, float, np.ndarray, float], np.ndarray],
    init_states: np.ndarray,
    parameters: np.ndarray,
    outdir: Path,
    nbeats: int = 200,
    BCL: int = 1000,
    save_every_ms: float = 1.0,
    dt: float = 0.05,
    track_indices: list[int] | None = None,
):
    # Compute hash of input arguments
    hash_input = compute_hash(
        fun=fun,
        init_states=init_states,
        parameters=parameters,
        nbeats=nbeats,
        BCL=BCL,
        dt=dt,
    )

    fname = outdir / f"steady_states_{hash_input}.npy"

    if fname.is_file():
        return np.load(fname)
    outdir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Computing steady state with {nbeats} beats.")

    times = np.arange(0.0, BCL, dt)

    kwargs = {
        "fun": jit(nopython=True)(fun),
        "nbeats": nbeats,
        "times": times,
        "y": init_states,
        "p": parameters,
        "dt": dt,
    }

    if track_indices is not None:
        save_freq = int(np.ceil(save_every_ms / dt))
        M = int(np.ceil(len(times) / save_freq) * nbeats)
        N = len(track_indices)
        track_values = np.zeros((M, N))

        indices = np.array(track_indices).astype(np.int32)
        kwargs.update(
            {
                "track_values": track_values,
                "track_indices": indices,
                "save_freq": save_freq,
            },
        )
        y, track_values = solve_with_save(**kwargs)
        np.save(outdir / f"tracked_values_{hash_input}.npy", track_values)
        fig, ax = plt.subplots(N, 2, sharex="col", sharey="row")
        for i in range(N):
            ax[i, 0].plot(np.linspace(0, BCL * nbeats, M), track_values[:, i])
            ax[i, 1].plot(
                times[::save_freq][-int(np.ceil(BCL // save_every_ms)) :],
                track_values[-int(np.ceil(BCL // save_every_ms)) :, i],
            )
        fig.tight_layout()
        fig.savefig(outdir / f"tracked_values_{hash_input}.png")
        plt.close()
    else:
        y = solve_without_save(**kwargs)

    np.save(fname, y)
    return y
