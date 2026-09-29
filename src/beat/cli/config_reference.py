"""Render the config reference (``docs/cli_reference.md``) from the pydantic models.

Kept in ``beat.cli`` (not ``docs/``) so it can import ``beat.cli.config`` directly and so
``tests/test_cli_reference.py`` can import it without needing dolfinx (``config.py`` has no
dolfinx import either). :func:`render_reference` must be **deterministic**: it is compared
byte-for-byte against the checked-in ``docs/cli_reference.md`` by that test, so nothing here may
depend on object identity/memory addresses (e.g. the default ``repr()`` of a ``pathlib.Path``
differs between platforms, and a naive ``str()`` of an ``Annotated[...]`` type embeds the
``PydanticPintQuantity`` marker's ``repr()``, which includes its memory address) or on iteration
order that isn't already fixed by the model definitions.
"""

import types
import typing
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from . import config as c

SECTIONS: list[tuple[str, list[type[BaseModel]]]] = [
    (
        "[geometry]",
        [
            c.FolderGeometry,
            c.IntervalGeometry,
            c.RectangleGeometry,
            c.BoxSlabGeometry,
            c.SlabGeometry,
            c.LVEllipsoidGeometry,
            c.BiVEllipsoidGeometry,
            c.UKBGeometry,
        ],
    ),
    ("[geometry.fibers]", [c.FromGeometryFibers, c.AxisFibers, c.IsotropicFibers]),
    ("[ep]", [c.EPConfig, c.ConductivityConfig]),
    ("[cell]", [c.CellConfig, c.SteadyStateConfig, c.RegionConfig]),
    ("[cell.layers]", [c.NoLayers, c.TransmuralLayers, c.CellMarkerLayers]),
    ("[[stimulus]]", [c.MarkerStimulus, c.BoxStimulus, c.RandomEndocardialStimulus]),
    ("[solver]", [c.SolverConfig]),
    ("[solver.pde]", [c.ThetaPDE, c.IrksomePDE]),
    ("[solver.ode]", [c.DolfinODE, c.IrksomeODE, c.ExternalOperatorODE]),
    ("[output]", [c.OutputConfig]),
    ("[postprocess]", [c.PostprocessConfig]),
]


def _type_name(annotation: Any) -> str:
    """A short, deterministic type name for a table cell.

    Recurses through ``typing`` generics (``Union``/``Optional``, ``Annotated``, ``list``,
    ``dict``, ``tuple``, ``Literal``) instead of stringifying the annotation directly, because
    ``str()`` of an ``Annotated[Quantity, PydanticPintQuantity(...)]`` field embeds that marker
    object's default ``repr()`` (``<pydantic_pint.quantity.PydanticPintQuantity object at
    0x...>``), which is a memory address and differs on every run. Model fields spell "optional"
    both ways (``Optional[X]`` == ``typing.Union[X, None]``, and the newer ``X | None`` ==
    ``types.UnionType``), so both are handled here.
    """
    origin = typing.get_origin(annotation)

    if origin is None:
        if annotation is type(None):
            return "None"
        if annotation is Path:
            return "path"
        if isinstance(annotation, type):
            if issubclass(annotation, BaseModel):
                return annotation.__name__
            return annotation.__name__
        return str(annotation)

    if origin is typing.Union or origin is types.UnionType:
        args = typing.get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        names = list(dict.fromkeys(_type_name(a) for a in non_none))
        text = " \\| ".join(names)
        if len(non_none) != len(args):
            text += " (optional)"
        return text

    if origin is typing.Annotated:
        # Annotated[actual_type, *metadata]: the metadata (e.g. PydanticPintQuantity) is not
        # part of the user-facing type.
        return _type_name(typing.get_args(annotation)[0])

    if origin is typing.Literal:
        return " \\| ".join(repr(a) for a in typing.get_args(annotation))

    if origin is list:
        (item,) = typing.get_args(annotation)
        return f"list[{_type_name(item)}]"

    if origin is dict:
        key, value = typing.get_args(annotation)
        return f"dict[{_type_name(key)}, {_type_name(value)}]"

    if origin is tuple:
        args = typing.get_args(annotation)
        return f"tuple[{', '.join(_type_name(a) for a in args)}]"

    return str(annotation)


def _format_value(value: Any) -> str:
    """A deterministic, TOML-flavoured repr of a default value.

    Quantities are strings already (see ``beat.cli.config._q``): a default like ``"0.05 ms"``
    round-trips through ``default_factory()`` as that same plain string, so ``repr()`` alone
    already renders it the way a user would write it in ``config.toml``. ``Path`` is the one
    default whose stock ``repr()`` is platform-dependent (``PosixPath(...)`` vs
    ``WindowsPath(...)``), so it's special-cased to its ``str()`` instead.
    """
    if isinstance(value, Path):
        return repr(str(value))
    return repr(value)


def _default(field: FieldInfo) -> str:
    if field.is_required():
        return "**required**"
    # pydantic's default_factory type is a union of a 0-arg and a 1-arg (validated-data) callable;
    # every default_factory in beat.cli.config is the 0-arg kind (a lambda or a bare class/type).
    value = (
        field.default_factory()  # type: ignore[call-arg]
        if field.default_factory is not None
        else field.default
    )
    if value is None:
        return "–"
    return f"`{_format_value(value)}`"


def render_reference() -> str:
    lines = [
        "# CLI configuration reference",
        "",
        "Generated from `beat.cli.config` by `scripts/gen_cli_reference.py` -- do not edit.",
        'Quantities are strings with units, e.g. `"0.05 ms"`.',
        "",
    ]
    for section, models in SECTIONS:
        lines += [f"## `{section}`", ""]
        for model in models:
            type_field = model.model_fields.get("type") or model.model_fields.get("method")
            tag = f" (`{type_field.default}`)" if type_field is not None else ""
            lines += [f"### {model.__name__}{tag}", ""]
            if model.__doc__:
                lines += [typing.cast(str, model.__doc__).strip(), ""]
            lines += ["| Field | Type | Default | Description |", "|---|---|---|---|"]
            for name, field in model.model_fields.items():
                lines.append(
                    f"| `{name}` | {_type_name(field.annotation)} | {_default(field)} | "
                    f"{field.description or ''} |",
                )
            lines.append("")
    return "\n".join(lines)
