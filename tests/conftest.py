import gc
import sys
from pathlib import Path

from mpi4py import MPI

import pytest

sys.path.insert(0, str(Path(__file__).parent))

# irksome adds a UFL type, so it must be imported before UFL processes any form (it raises
# IrksomeImportOrderException otherwise). Tests import it lazily, after other tests have built
# forms, so import it here, before any test runs, when it is installed.
try:
    import irksome  # noqa: F401
except ImportError:
    pass


@pytest.fixture(autouse=True)
def _collective_garbage_collection():
    """Collect garbage on all ranks together, never at a rank-local moment.

    Some objects (e.g. dolfinx's ``NonlinearProblem``, which Irksome steppers hold) end up in
    reference cycles and destroy their PETSc objects, which is collective, in ``__del__``. Left to
    the automatic collector that happens at a different moment on each rank, e.g. while one rank
    JIT-compiles a form and the other waits for it, and the two ranks deadlock.
    """
    gc.disable()
    try:
        yield
    finally:
        MPI.COMM_WORLD.Barrier()
        gc.collect()
        MPI.COMM_WORLD.Barrier()
        gc.enable()
