import logging
from pathlib import Path
from typing import Annotated

try:
    from pydantic_pint import PydanticPintQuantity, set_registry
except ImportError as e:
    raise ImportError(
        "pydantic_pint is not installed. "
        "Install it with 'pip install pydantic-pint' to use PydanticPintQuantity.",
    ) from e

from pint import Quantity
from pydantic import AfterValidator, Field
from pydantic_settings import BaseSettings

from .units import ureg

logger = logging.getLogger(__name__)

# Make pydantic_pint validate/serialize quantities using beat's own unit registry, so that
# Config quantities can be freely combined (arithmetic, comparisons) with quantities created
# elsewhere in beat (e.g. in conductivities.py/stimulation.py, which use `beat.units.ureg`).
# Pint raises if you mix Quantity objects created by different UnitRegistry instances.
set_registry(ureg)


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
        logger.warning(f"File not found: {file_path}")
    return file_path


class CellConfig(BaseSettings):
    ode_file: Annotated[Path, AfterValidator(check_file_exists)] = Field(
        "model.ode",
        description="Path to the .ode model",
    )
    num_beats: int = Field(10, description="Number of beats for the single cell model")
    BCL: Annotated[Quantity, PydanticPintQuantity("ms")] = Field(
        "1000 ms",
        description="Basic cycle length for the single cell simulation (ms)",
    )
    dt: Annotated[Quantity, PydanticPintQuantity("ms")] = Field(
        "0.01 ms",
        description="Time step for single cell simulations (ms)",
    )
    module_name: Path = Field(
        "cell_model.py",
        description="Python module containing the cell model",
    )
    scheme: str = Field("generalized_rush_larsen", description="Scheme for the cell model")
    v_name: str = Field(
        "v",
        description="Name of the state variable representing the transmembrane potential",
    )
    track_indices: list[str] = Field(
        default_factory=lambda: ["v", "cai"],
        description="List of state names to track while computing the steady state",
    )


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
        "0.05 ms",
        description="Time step for the simulation (ms)",
    )
    BCL: Annotated[Quantity, PydanticPintQuantity("ms")] = Field(
        "1000 ms",
        description="Basic cycle length for the simulation (ms)",
    )
    theta: float = Field(
        1.0,
        description="Splitting scheme parameter (1.0 = Godunov, 0.5 = Strang)",
    )
    save_every_ms: float = Field(1.0, description="Save the simulation state every N ms")
    output_folder: Annotated[Path, AfterValidator(check_output_folder)] = Field(
        "output",
        description="Folder to save the simulation output",
    )


class StimulusConfig(BaseSettings):
    marker: str = Field(
        "ENDO",
        description="Name of the facet marker (from the mesh markers) where the stimulus "
        "is applied",
    )
    amplitude: float = Field(
        5000.0,
        description="Amplitude of the stimulus current, in the unit implied by the marker's "
        "dimension and the mesh unit (see beat.stimulation.define_stimulus)",
    )
    duration: Annotated[Quantity, PydanticPintQuantity("ms")] = Field(
        "2.0 ms",
        description="Duration of the stimulus (ms)",
    )
    start: Annotated[Quantity, PydanticPintQuantity("ms")] = Field(
        "0.0 ms",
        description="Start time of the stimulus (ms)",
    )


class Config(BaseSettings):
    ep: EPConfig = Field(default_factory=EPConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    cell: CellConfig = Field(default_factory=CellConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    stimulus: StimulusConfig = Field(default_factory=StimulusConfig)

    def dump_toml(self, path: Path) -> None:
        """
        Dump the configuration to a TOML file.

        Parameters
        ----------
        path : Path
            The path to the TOML file where the configuration will be saved.
        """
        import toml

        Path(path).write_text(toml.dumps(self.model_dump(mode="json")))
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
        return cls.model_validate(config_data)
