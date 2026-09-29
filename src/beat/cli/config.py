"""Pydantic models for the ``beat`` CLI configuration file.

This module must not import dolfinx: it is used by ``beat validate-config`` and by the fast unit
tests. Everything with a physical unit is a pint quantity (``"<value> <unit>"`` in TOML).
"""

import math
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Union

import pint
from pint import Quantity
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_pint import PydanticPintQuantity, set_registry

from ..units import ureg

# Share beat's registry so config quantities combine with quantities created elsewhere in beat.
set_registry(ureg)

Time = Annotated[Quantity, PydanticPintQuantity("ms")]
SigmaQ = Annotated[Quantity, PydanticPintQuantity("S/m")]
InvLength = Annotated[Quantity, PydanticPintQuantity("cm**-1")]
Capacitance = Annotated[Quantity, PydanticPintQuantity("uF/cm**2")]


class ConfigError(ValueError):
    """Invalid configuration (exit code 1)."""


def ms(q: Quantity) -> float:
    return float(q.to("ms").magnitude)


def _q(default: str) -> Quantity:
    """Type-only cast for a pint-quantity field's string-literal default.

    Every ``Time``/``SigmaQ``/``InvLength``/``Capacitance`` field is statically typed as
    ``pint.Quantity`` (via ``PydanticPintQuantity``), but its convenient default is a plain
    ``"<value> <unit>"`` string; pydantic-pint parses that string into a real ``Quantity`` at
    validation time because every model here sets ``validate_default=True``. mypy can't see
    through that runtime conversion, so this helper centralizes the one necessary lie ("this
    string is already a Quantity") in a single documented place instead of scattering
    ``# type: ignore`` comments across every field declaration. Always use it through
    ``default_factory=lambda: _q(...)`` (not ``default=_q(...)``) so each model instance gets
    its own validated value.
    """
    return default  # type: ignore[return-value]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


# --- geometry ---------------------------------------------------------------------------


class FromGeometryFibers(_Base):
    type: Literal["from_geometry"] = "from_geometry"


class AxisFibers(_Base):
    type: Literal["axis"] = "axis"
    direction: Literal["x", "y", "z"] = "x"


class IsotropicFibers(_Base):
    type: Literal["isotropic"] = "isotropic"


FibersConfig = Annotated[
    Union[FromGeometryFibers, AxisFibers, IsotropicFibers],
    Field(discriminator="type"),
]


class _GeometryBase(_Base):
    unit: str = Field(default="mm", description="Length unit of the mesh coordinates")
    folder: Path = Field(
        default=Path("geometry"),
        description="type=folder: folder to read. Generated types (slab, lv_ellipsoid, "
        "biv_ellipsoid, ukb): cache root, each mesh is cached in its own <hash>/ subfolder",
    )

    @field_validator("unit")
    @classmethod
    def _is_length(cls, v: str) -> str:
        try:
            dim = ureg.Quantity(1, v).dimensionality
        except pint.errors.PintError as e:
            raise ValueError(f"geometry.unit must be a valid unit, got {v!r}: {e}") from e
        if dim != ureg.Quantity(1, "m").dimensionality:
            raise ValueError(f"geometry.unit must be a length unit, got {v!r}")
        return v

    def generator_kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude={"type", "unit", "folder", "fibers"})


class FolderGeometry(_GeometryBase):
    type: Literal["folder"] = "folder"
    fibers: FibersConfig = Field(default_factory=FromGeometryFibers)


class IntervalGeometry(_GeometryBase):
    type: Literal["interval"] = "interval"
    length: float = Field(default=10.0, gt=0, description="Cable length (geometry.unit)")
    dx: float = Field(default=0.1, gt=0, description="Element size (geometry.unit)")
    fibers: FibersConfig = Field(default_factory=IsotropicFibers)


class RectangleGeometry(_GeometryBase):
    type: Literal["rectangle"] = "rectangle"
    lx: float = Field(default=1.0, gt=0)
    ly: float = Field(default=1.0, gt=0)
    dx: float = Field(default=0.05, gt=0)
    fibers: FibersConfig = Field(default_factory=IsotropicFibers)


class BoxSlabGeometry(_GeometryBase):
    """Structured tetrahedral box (beat.geometry.get_3D_slab_mesh); no gmsh needed."""

    type: Literal["box_slab"] = "box_slab"
    lx: float = Field(default=20.0, gt=0)
    ly: float = Field(default=7.0, gt=0)
    lz: float = Field(default=3.0, gt=0)
    dx: float = Field(default=0.5, gt=0)
    fibers: FibersConfig = Field(default_factory=AxisFibers)


class _FiberAngles(_GeometryBase):
    fiber_angle_endo: float = 60.0
    fiber_angle_epi: float = -60.0
    fiber_space: str = "P_1"
    fibers: FibersConfig = Field(default_factory=FromGeometryFibers)


class SlabGeometry(_FiberAngles):
    type: Literal["slab"] = "slab"
    lx: float = 20.0
    ly: float = 7.0
    lz: float = 3.0
    dx: float = 1.0


class LVEllipsoidGeometry(_FiberAngles):
    type: Literal["lv_ellipsoid"] = "lv_ellipsoid"
    r_short_endo: float = 7.0
    r_short_epi: float = 10.0
    r_long_endo: float = 17.0
    r_long_epi: float = 20.0
    psize_ref: float = 3.0
    mu_apex_endo: float = -math.pi
    mu_base_endo: float = -1.2722641256100204
    mu_apex_epi: float = -math.pi
    mu_base_epi: float = -1.318116071652818


class BiVEllipsoidGeometry(_FiberAngles):
    type: Literal["biv_ellipsoid"] = "biv_ellipsoid"
    char_length: float = 0.5


class UKBGeometry(_FiberAngles):
    type: Literal["ukb"] = "ukb"
    mode: int = -1
    std: float = 1.5
    case: Literal["ED", "ES"] = "ED"
    char_length_max: float = 5.0
    char_length_min: float = 5.0
    clipped: bool = False


GENERATED_GEOMETRY_TYPES = ("slab", "lv_ellipsoid", "biv_ellipsoid", "ukb")

GeometryConfig = Annotated[
    Union[
        FolderGeometry,
        IntervalGeometry,
        RectangleGeometry,
        BoxSlabGeometry,
        SlabGeometry,
        LVEllipsoidGeometry,
        BiVEllipsoidGeometry,
        UKBGeometry,
    ],
    Field(discriminator="type"),
]

# --- electrophysiology ------------------------------------------------------------------


class ConductivityConfig(_Base):
    preset: Literal["Niederer", "Bishop"] | None = Field(
        default=None,
        description="Literature set from beat.conductivities.default_conductivities; "
        "explicit sigma_* override it field by field",
    )
    sigma_il: SigmaQ | None = None
    sigma_it: SigmaQ | None = None
    sigma_el: SigmaQ | None = None
    sigma_et: SigmaQ | None = None

    def resolved(self) -> dict[str, Quantity]:
        from ..conductivities import default_conductivities

        values = default_conductivities(self.preset or "Niederer")
        out = {k: values[k] for k in ("g_il", "g_it", "g_el", "g_et")}
        for short in ("il", "it", "el", "et"):
            explicit = getattr(self, f"sigma_{short}")
            if explicit is not None:
                out[f"g_{short}"] = explicit
        return out


class EPConfig(_Base):
    chi: InvLength = Field(
        default_factory=lambda: _q("1400 cm**-1"),
        description="Surface to volume ratio",
    )
    C_m: Capacitance = Field(
        default_factory=lambda: _q("1 uF/cm**2"),
        description="Membrane capacitance",
    )
    conductivity: ConductivityConfig = Field(default_factory=ConductivityConfig)


# --- cell model -------------------------------------------------------------------------


class SteadyStateConfig(_Base):
    num_beats: int = Field(default=20, ge=1)
    BCL: Time = Field(default_factory=lambda: _q("1000 ms"))
    dt: Time = Field(default_factory=lambda: _q("0.05 ms"))
    track: list[str] = Field(default_factory=list, description="States to record")


class NoLayers(_Base):
    method: Literal["none"] = "none"


class TransmuralLayers(_Base):
    method: Literal["transmural"] = "transmural"
    endo_size: float = Field(default=0.3, ge=0, le=1)
    epi_size: float = Field(default=0.3, ge=0, le=1)
    endo_markers: list[str] | None = Field(
        default=None,
        description='Facet markers of the endocardium. Default: ["ENDO"] if present, '
        'else ["LV", "RV"] (BiV)',
    )
    epi_marker: str = "EPI"


class CellMarkerLayers(_Base):
    method: Literal["cell_markers"] = "cell_markers"
    map: dict[str, str] = Field(description="region name -> geometry cell marker name")


LayersConfig = Annotated[
    Union[NoLayers, TransmuralLayers, CellMarkerLayers],
    Field(discriminator="method"),
]

TRANSMURAL_REGIONS = ("mid", "endo", "epi")  # marker values 0, 1, 2 (expand_layer defaults)

# Clinical (endo -> mid -> epi) order used only when *displaying* allowed region names in
# error messages; `TRANSMURAL_REGIONS`/`region_names()` keep marker-value order (0, 1, 2) for
# downstream consumers that iterate regions by marker index.
_REGION_DISPLAY_ORDER = {"endo": 0, "mid": 1, "epi": 2}


def _display_regions(names: "list[str]") -> "list[str]":
    return sorted(names, key=lambda n: (_REGION_DISPLAY_ORDER.get(n, 99), n))


class RegionConfig(_Base):
    parameters: dict[str, float] = Field(default_factory=dict)


class CellConfig(_Base):
    ode_file: Path = Field(description="Path to the gotranx .ode cell model")
    scheme: str = "generalized_rush_larsen"
    v_name: str = Field(default="v", description="Name of the transmembrane potential state")
    parameters: dict[str, float] = Field(default_factory=dict)
    steady_state: SteadyStateConfig | None = Field(
        default=None,
        description="If set, pre-pace a single cell to steady state before the tissue run",
    )
    layers: LayersConfig = Field(default_factory=NoLayers)
    regions: dict[str, RegionConfig] = Field(default_factory=dict)

    def region_names(self) -> list[str]:
        if self.layers.method == "none":
            return ["tissue"]
        if self.layers.method == "transmural":
            return list(TRANSMURAL_REGIONS)
        return list(self.layers.map)

    @model_validator(mode="after")
    def _check_regions(self) -> "CellConfig":
        if self.layers.method == "none":
            if self.regions:
                raise ValueError("cell.regions requires cell.layers.method != 'none'")
            return self
        allowed = self.region_names()
        unknown = sorted(set(self.regions) - set(allowed))
        if unknown:
            raise ValueError(
                f"Unknown cell.regions {unknown}; allowed for layers.method="
                f"{self.layers.method!r}: {', '.join(_display_regions(allowed))}",
            )
        return self


# --- stimulus ---------------------------------------------------------------------------


def _check_current_density(v: str, dims: tuple[int, ...] = (2, 3)) -> str:
    """Check ``v`` is a current per length**k for some k in ``dims``.

    The effective stimulus dimension (see ``beat.stimulation.compute_effective_dim``) is only
    ever 2 (a facet marker) or 3 (a cell marker, box or random_endocardial), so a current per
    length (k = 1) is never valid.
    """
    try:
        q = ureg.Quantity(v)
    except pint.errors.PintError as e:
        raise ValueError(f"amplitude must be a valid quantity, got {v!r}: {e}") from e
    current = ureg.Quantity(1, "uA").dimensionality
    for k in dims:
        if q.dimensionality == current / ureg.Quantity(1, "cm").dimensionality ** k:
            return v
    expected = " or ".join(f"uA/cm**{k}" for k in dims)
    raise ValueError(f"amplitude must be a current density (e.g. {expected}), got {v!r}")


class _StimulusBase(_Base):
    # Allowed length exponents of the amplitude's current density (see _check_current_density).
    _amplitude_dims: ClassVar[tuple[int, ...]] = (2, 3)

    amplitude: str = Field(
        description="Current density: uA/cm**2 for a marker stimulus on a facet marker, "
        "uA/cm**3 for a marker stimulus on a cell marker and for box/random_endocardial "
        "stimuli (any mesh dimension); see beat.stimulation.define_stimulus",
    )
    duration: Time = Field(default_factory=lambda: _q("2 ms"))
    start: Time = Field(default_factory=lambda: _q("0 ms"))
    period: Time | None = Field(default=None, description="If set, repeat the pulse every period")
    num_pulses: int | None = Field(
        default=None,
        ge=1,
        description="Number of pulses (requires period). Default: until the end time",
    )

    @field_validator("amplitude")
    @classmethod
    def _amp(cls, v: str) -> str:
        return _check_current_density(v, cls._amplitude_dims)

    @model_validator(mode="after")
    def _check_train(self) -> "_StimulusBase":
        if self.num_pulses is not None and self.period is None:
            raise ValueError("num_pulses requires period")
        if self.period is not None and self.period <= self.duration:
            raise ValueError("period must be larger than duration")
        return self

    def pulse_starts_ms(self, t_end_ms: float) -> list[float]:
        start = ms(self.start)
        if self.period is None:
            return [start]
        period = ms(self.period)
        n = self.num_pulses or max(1, math.ceil((t_end_ms - start) / period))
        return [start + k * period for k in range(n)]


class MarkerStimulus(_StimulusBase):
    type: Literal["marker"] = "marker"
    marker: str = Field(description="Facet or cell marker name from the geometry")


class BoxStimulus(_StimulusBase):
    _amplitude_dims: ClassVar[tuple[int, ...]] = (3,)  # always volumetric

    type: Literal["box"] = "box"
    min: list[float]
    max: list[float]

    @model_validator(mode="after")
    def _check_box(self) -> "BoxStimulus":
        if len(self.min) != len(self.max):
            raise ValueError("box min and max must have the same length")
        if any(a >= b for a, b in zip(self.min, self.max)):
            raise ValueError("box requires min < max in every coordinate")
        return self


class RandomEndocardialStimulus(_StimulusBase):
    _amplitude_dims: ClassVar[tuple[int, ...]] = (3,)  # always volumetric

    type: Literal["random_endocardial"] = "random_endocardial"
    markers: list[str] = Field(default_factory=lambda: ["LV", "RV"], min_length=1)
    num_points: int = Field(default=200, ge=1)
    delay_range: tuple[Time, Time] = Field(
        default_factory=lambda: (_q("0 ms"), _q("4 ms")),
    )
    seed: int = 0
    tol: float = Field(default=1.0, gt=0, description="Radius around each point (geometry.unit)")


StimulusConfig = Annotated[
    Union[MarkerStimulus, BoxStimulus, RandomEndocardialStimulus],
    Field(discriminator="type"),
]

# --- solver -----------------------------------------------------------------------------


class ThetaPDE(_Base):
    type: Literal["theta"] = "theta"
    theta: float = Field(default=0.5, ge=0, le=1, description="PDE theta-scheme parameter")
    linear_solver: Literal["direct", "iterative"] = "direct"


class IrksomePDE(_Base):
    type: Literal["irksome"] = "irksome"
    tableau: str = Field(default="RadauIIA", description="Irksome Butcher tableau class name")
    stages: int = Field(default=1, ge=1)
    linear_solver: Literal["direct", "iterative"] = "direct"


PDEConfig = Annotated[Union[ThetaPDE, IrksomePDE], Field(discriminator="type")]


class DolfinODE(_Base):
    type: Literal["dolfin"] = "dolfin"


class IrksomeODE(_Base):
    type: Literal["irksome"] = "irksome"
    tableau: str = "RadauIIA"
    stages: int = Field(default=1, ge=1)


class ExternalOperatorODE(_Base):
    type: Literal["external_operator"] = "external_operator"


ODEBackendConfig = Annotated[
    Union[DolfinODE, IrksomeODE, ExternalOperatorODE],
    Field(discriminator="type"),
]


class SolverConfig(_Base):
    dt: Time = Field(default_factory=lambda: _q("0.05 ms"))
    theta: float = Field(default=1.0, ge=0, le=1, description="Splitting: 1.0 Godunov, 0.5 Strang")
    end_time: Time | None = Field(
        default=None,
        description="Simulated end time. Give either this or num_beats and BCL",
    )
    num_beats: int | None = Field(
        default=None,
        ge=1,
        description="Run length in beats: end time = num_beats x BCL",
    )
    BCL: Time | None = Field(
        default=None,
        description="Basic cycle length, only used for the run length (num_beats x BCL). It "
        "does not pace anything: set a [[stimulus]] period for that",
    )
    pde: PDEConfig = Field(default_factory=ThetaPDE)
    ode: ODEBackendConfig = Field(default_factory=DolfinODE)
    petsc_options: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_end(self) -> "SolverConfig":
        by_time = self.end_time is not None
        by_beats = self.num_beats is not None or self.BCL is not None
        if by_time == by_beats or (by_beats and (self.num_beats is None or self.BCL is None)):
            raise ValueError(
                "solver: give exactly one of `end_time` or (`num_beats` and `BCL`)",
            )
        return self

    def t_end_ms(self) -> float:
        if self.end_time is not None:
            return ms(self.end_time)
        assert self.num_beats is not None and self.BCL is not None
        return self.num_beats * ms(self.BCL)


# --- output / postprocess ---------------------------------------------------------------


class OutputConfig(_Base):
    folder: Path = Path("output")
    save_every: Time = Field(default_factory=lambda: _q("1 ms"))
    fields: list[str] = Field(default_factory=list, description="Extra ODE states to save")
    checkpoint_every: Time = Field(
        default_factory=lambda: _q("0 ms"),
        description="Restart checkpoint interval; 0 = end only",
    )
    performance: bool = False
    log_every: int = Field(default=100, ge=1)


class PostprocessConfig(_Base):
    points: dict[str, list[float]] = Field(default_factory=dict)
    activation_threshold: float = 0.0
    sigma_b: float = 1.0
    vtx: bool = True
    make_gif: bool = False


class Config(_Base):
    geometry: GeometryConfig
    ep: EPConfig = Field(default_factory=EPConfig)
    cell: CellConfig
    stimulus: list[StimulusConfig] = Field(default_factory=list)
    solver: SolverConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)

    @model_validator(mode="after")
    def _check_times(self) -> "Config":
        if self.output.save_every < self.solver.dt:
            raise ValueError("output.save_every must be >= solver.dt")
        if 0 < ms(self.output.checkpoint_every) < ms(self.solver.dt):
            raise ValueError("output.checkpoint_every must be 0 or >= solver.dt")
        return self


ALL_MODELS: tuple[type[BaseModel], ...] = tuple(
    obj
    for obj in list(globals().values())
    if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
)
