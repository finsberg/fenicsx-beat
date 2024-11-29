import typing
from pathlib import Path

from mpi4py import MPI

import adios4dolfinx
import adios4dolfinx.adios2_helpers
import numpy as np
import numpy.typing as npt

adios2 = adios4dolfinx.adios2_helpers.adios2


def read_timestamps(
    comm: MPI.Intracomm,
    filename: typing.Union[Path, str],
    function_name: str,
    engine="BP4",
) -> npt.NDArray[np.float64]:
    """
    Read time-stamps from a checkpoint file.

    Args:
        comm: MPI communicator
        filename: Path to file
        function_name: Name of the function to read time-stamps for
        engine: ADIOS2 engine
    Returns:
        The time-stamps
    """

    adios = adios2.ADIOS(comm)

    with adios4dolfinx.adios2_helpers.ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios2.Mode.Read,
        engine=engine,
        io_name="TimestepReader",
    ) as adios_file:
        time_name = f"{function_name}_time"
        time_stamps = []
        for i in range(adios_file.file.Steps()):
            adios_file.file.BeginStep()
            if time_name in adios_file.io.AvailableVariables().keys():
                arr = adios_file.io.InquireVariable(time_name)
                time_shape = arr.Shape()
                arr.SetSelection([[0], [time_shape[0]]])
                times = np.empty(
                    time_shape[0],
                    dtype=adios4dolfinx.adios2_helpers.adios_to_numpy_dtype[arr.Type()],
                )
                adios_file.file.Get(arr, times, adios2.Mode.Sync)
                time_stamps.append(times[0])
            adios_file.file.EndStep()

    return np.array(time_stamps)
