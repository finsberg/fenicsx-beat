import os
import subprocess
import sys
import textwrap
from pathlib import Path

from mpi4py import MPI

import pytest

from beat.log import MPIFileHandler


def test_mpi_file_handler_uses_a_concrete_encoding(tmp_path):
    # encoding must never be left as None: FileHandler.__init__ then resolves it via
    # io.text_encoding(None), which - outside UTF-8 mode - returns the literal string
    # "locale" (a sentinel meant for open(), not a real codec name). MPIFileHandler.emit()
    # calls msg.encode(self.encoding) directly, so a bare "locale" blows up with
    # `LookupError: unknown encoding: locale`. See test_emit_under_non_utf8_locale below
    # for a full reproduction of that failure mode.
    handler = MPIFileHandler(tmp_path / "out.log", comm=MPI.COMM_WORLD, delay=True)
    try:
        assert handler.encoding == "utf-8"
        "".encode(handler.encoding)  # raises LookupError if not a real codec
    finally:
        handler.delay = True  # nothing was opened (delay=True), nothing to close


@pytest.mark.skip_in_parallel
def test_emit_under_non_utf8_locale(tmp_path):
    # Full regression test for the bug: `-X utf8=0` deterministically disables Python's
    # UTF-8 mode (regardless of the host's actual locale), which is exactly the condition
    # that made io.text_encoding(None) return the literal string "locale" and broke emit().
    # skip_in_parallel: the subprocess below calls MPI.COMM_WORLD itself, and under mpirun it
    # inherits the parent job's PMI environment variables and fails to join it (unrelated to
    # the encoding bug this test targets).
    script = textwrap.dedent(
        """
        import logging
        from pathlib import Path
        from mpi4py import MPI
        from beat.log import add_logfile_handler

        logging.getLogger().setLevel(logging.INFO)
        outdir = Path({outdir!r})
        outdir.mkdir(exist_ok=True)
        add_logfile_handler(outdir, comm=MPI.COMM_WORLD)
        logging.getLogger("regression").info("hello from a non-utf8 locale")
        """,
    ).format(outdir=str(tmp_path))

    result = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", script],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    assert result.returncode == 0, result.stderr
    assert "unknown encoding" not in result.stderr

    log_file = Path(tmp_path) / "output.log"
    assert log_file.is_file()
    assert "hello from a non-utf8 locale" in log_file.read_text()
