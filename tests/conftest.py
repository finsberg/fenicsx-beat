import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# irksome adds a UFL type, so it must be imported before UFL processes any form (it raises
# IrksomeImportOrderException otherwise). Tests import it lazily, after other tests have built
# forms, so import it here, before any test runs, when it is installed.
try:
    import irksome  # noqa: F401
except ImportError:
    pass
