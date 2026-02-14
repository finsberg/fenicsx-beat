import logging
from pathlib import Path
from typing import Annotated

try:
    from pydantic_pint import PydanticPintQuantity
except ImportError:
    msg = (
        "pydantic_pint is not installed. "
        "Install it with 'pip install pydantic-pint' to use PydanticPintQuantity.",
    )
    logging.warning(msg)
    import sys

    sys.exit(1)

from pint import Quantity
from pydantic import AfterValidator, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Conductivity(BaseSettings):
    sigma_el: Annotated[Quantity, PydanticPintQuantity("S/m")] = Field(
        "0.62 S/m",
        description="Extracellular conductivity in the longitudinal direction (S/m)",
    )
    sigma_et: Annotated[Quantity, PydanticPintQuantity("S/m")] = Field(
        "0.24 S/m",
        description="Extracellular conductivity in the transverse direction (S/m)",
    )
    sigma_il: Annotated[Quantity, PydanticPintQuantity("S/m")] = Field(
        "0.17 S/m",
        description="Intracellular conductivity in the longitudinal direction (S/m)",
    )
    sigma_it: Annotated[Quantity, PydanticPintQuantity("S/m")] = Field(
        "0.019 S/m",
        description="Intracellular conductivity in the transverse direction (S/m)",
    )


class EPConfig(BaseSettings):
    conductivity: Conductivity = Conductivity()
    chi: Annotated[Quantity, PydanticPintQuantity("cm**-1")] = Field(
        "1400 cm**-1",
        description="Surface to volume ratio (cm^-1)",
    )
    C_m: Annotated[Quantity, PydanticPintQuantity("uF/mm**2")] = Field(
        "0.01 uF/mm**2",
        description="Membrane capacitance (uF/mm^2)",
    )


class MeshConfig(BaseSettings):
    unit: str = Field("mm", description="Unit of the mesh")
    folder: Path = Field("mesh", description="Folder containing the mesh files")


def check_file_exists(file_path: Path) -> Path:
    """
    Ensure the specified file exists.

    Parameters
    ----------
    file_path : Path
        The path to the file to check.

    Returns
    -------
    Path
        The validated file path.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path


class CellConfig(BaseSettings):
    filename: Annotated[Path, AfterValidator(check_file_exists)] = Field(
        "model.ode",
        description="Path to the .ode model",
    )
    num_beats_prepace: int = Field(10, description="Number of beats to prepace the cell model")


def check_output_folder(output_folder: Path) -> Path:
    """
    Ensure the output folder exists, creating it if necessary.

    Parameters
    ----------
    output_folder : Path
        The path to the output folder.

    Returns
    -------
    Path
        The validated output folder path.
    """
    output_folder = output_folder.resolve()
    if not output_folder.exists():
        logger.info(
            f"Output folder {output_folder!r} does not exist, and will be created when running.",
        )
    else:
        logger.info(f"Output folder already exists: {output_folder}")

    return output_folder


class SimulationConfig(BaseSettings):
    num_beats: int = Field(10, description="Number of beats to simulate")
    dt: Annotated[Quantity, PydanticPintQuantity("ms")] = Field(
        "0.01 ms",
        description="Time step for the simulation (ms)",
    )
    BCL: Annotated[Quantity, PydanticPintQuantity("ms")] = Field(
        "1000 ms",
        description="Basic cycle length for the simulation (ms)",
    )
    output_folder: Annotated[Path, AfterValidator(check_output_folder)] = Field(
        "output",
        description="Folder to save the simulation output",
    )


class Config(BaseSettings):
    ep: EPConfig = Field(default_factory=EPConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    cell: CellConfig = Field(default_factory=CellConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

    def dump_toml(self, path: Path) -> None:
        """
        Dump the configuration to a TOML file.

        Parameters
        ----------
        path : Path
            The path to the TOML file where the configuration will be saved.
        """
        import json

        import toml

        Path(path).write_text(toml.dumps(json.loads(self.json())))
        logger.info(f"Configuration dumped to {path}")

    @classmethod
    def parse_toml(cls, path: Path) -> "Config":
        """
        Parse a TOML file into a Config object.

        Parameters
        ----------
        path : Path
            The path to the TOML file to parse.

        Returns
        -------
        Config
            The parsed configuration object.
        """

        import toml

        config_data = toml.loads(path.read_text())
        return cls.parse_obj(config_data)
