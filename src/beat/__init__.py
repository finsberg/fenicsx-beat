from importlib.metadata import metadata

from . import (
    base_model,
    conductivities,
    ecg,
    geometry,
    monodomain_model,
    monodomain_solver,
    odesolver,
    single_cell,
    stimulation,
    utils,
)
from .ecg import ECGRecovery
from .monodomain_model import MonodomainModel
from .monodomain_solver import MonodomainSplittingSolver
from .stimulation import Stimulus

meta = metadata("fenicsx-beat")
__version__ = meta["Version"]
__author__ = meta["Author-email"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = [
    "monodomain_model",
    "odesolver",
    "base_model",
    "MonodomainModel",
    "monodomain_solver",
    "MonodomainSplittingSolver",
    "utils",
    "conductivities",
    "stimulation",
    "geometry",
    "single_cell",
    "stimulation",
    "ecg",
    "Stimulus",
    "ECGRecovery",
]
