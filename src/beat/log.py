import logging
from pathlib import Path

from mpi4py import MPI


def add_logfile_handler(output_folder: Path, comm=MPI.COMM_WORLD):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    FORMAT_ALL = (
        "%(asctime)s %(rank)s%(name)s - %(levelname)s - " "%(message)s (%(filename)s:%(lineno)d)"
    )
    FORMAT = "%(asctime)s %(name)s - %(levelname)s - " "%(message)s (%(filename)s:%(lineno)d)"

    class Formatter(logging.Formatter):
        def format(self, record):
            record.rank = f"CPU {rank}: " if size > 1 else ""
            return super().format(record)

    class MPIFilter(logging.Filter):
        def filter(self, record):
            if rank == 0:
                return 1
            else:
                return 0

    if size > 1:
        file_handler_all = MPIFileHandler(output_folder / "output_all_cpus.log", comm=comm)
        file_handler_all.setLevel(logging.INFO)
        file_handler_all.setFormatter(Formatter(FORMAT_ALL))
        logging.getLogger().addHandler(file_handler_all)

    file_handler = MPIFileHandler(output_folder / "output.log", comm=comm)
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(MPIFilter())
    file_handler.setFormatter(logging.Formatter(FORMAT))
    logging.getLogger().addHandler(file_handler)


def setup_logging(level: int = logging.INFO, log_all_cpus: bool = False, comm=MPI.COMM_WORLD):
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.theme import Theme

    rank = comm.rank
    size = comm.size

    FORMAT = (
        "%(asctime)s %(rank)s%(name)s - %(levelname)s - " "%(message)s (%(filename)s:%(lineno)d)"
    )

    class Formatter(logging.Formatter):
        def format(self, record):
            record.rank = f"CPU {rank}: " if size > 1 else ""
            return super().format(record)

    class MPIFilter(logging.Filter):
        def filter(self, record):
            if rank == 0:
                return 1
            else:
                return 0

    console = Console(theme=Theme({"logging.level.custom": "green"}), width=140)
    handler = RichHandler(level=level, console=console)

    handler.setFormatter(Formatter(FORMAT))
    if not log_all_cpus:
        handler.addFilter(MPIFilter())

    logging.basicConfig(
        level="NOTSET",
        format=FORMAT,
        handlers=[handler],
    )

    _disable_loggers()


def _disable_loggers():
    for name in ["matplotlib"]:
        logging.getLogger(name).setLevel(logging.WARNING)


mode2mpi_mode = {
    "w": MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_EXCL,
    "a": MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_APPEND,
}


class MPIFileHandler(logging.FileHandler):
    def __init__(
        self,
        filename: Path,
        mode: str = "a",
        comm=MPI.COMM_WORLD,
        delay: bool = False,
    ):
        self.comm = comm
        self.mpi_mode = mode2mpi_mode[mode]
        super().__init__(filename=filename, mode=mode, delay=delay)

    def _open(self):
        stream = MPI.File.Open(self.comm, self.baseFilename, self.mpi_mode)
        stream.Set_atomicity(True)
        return stream

    def emit(self, record):
        msg = self.format(record)
        self.stream.Write_shared((msg + self.terminator).encode(self.encoding))

    def close(self):
        self.stream.Sync()
        self.stream.Close()
