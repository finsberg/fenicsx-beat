"""Merge a TOML config with environment variables, ``--set`` overrides and CLI flags.

Precedence (lowest to highest): TOML file < ``BEAT_*`` env vars < ``--set`` < flags.
"""

import hashlib
import json
import os
import shlex
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import toml
from pint import Quantity
from pydantic import ValidationError

from .config import ALL_MODELS, Config, ConfigError

ENV_PREFIX = "BEAT_"

# Canonical spelling of every config field name, keyed by its lower-case form, so that
# upper-case env vars (BEAT_EP__C_M) and sloppy --set keys map onto e.g. `C_m` / `BCL`.
_KEY_CASE: dict[str, str] = {
    name.lower(): name for model in ALL_MODELS for name in model.model_fields
}


def _canonical(key: str) -> str:
    return _KEY_CASE.get(key.lower(), key)


def parse_value(raw: str) -> Any:
    """Parse ``raw`` as a TOML literal, falling back to the plain string."""
    try:
        return toml.loads(f"v = {raw}")["v"]
    except toml.TomlDecodeError:
        return raw


def apply_override(data: dict, dotted: str, value: Any) -> None:
    parts = [_canonical(p) for p in dotted.split(".")]
    node: Any = data
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if isinstance(node, list):
            if not part.isdigit() or int(part) >= len(node):
                raise ConfigError(
                    f"{dotted}: list index {part!r} out of range (length {len(node)})",
                )
            if last:
                node[int(part)] = value
            else:
                node = node[int(part)]
            continue
        if not isinstance(node, dict):
            raise ConfigError(
                f"{dotted}: {'.'.join(parts[:i])!r} is not a table, cannot set {part!r}",
            )
        if last:
            node[part] = value
        else:
            node = node.setdefault(part, {})


def env_overrides(environ: Mapping[str, str]) -> list[tuple[str, Any]]:
    out = []
    for key, raw in sorted(environ.items()):
        if not key.startswith(ENV_PREFIX) or len(key) == len(ENV_PREFIX):
            continue
        dotted = ".".join(_canonical(p) for p in key[len(ENV_PREFIX) :].split("__"))
        out.append((dotted, parse_value(raw)))
    return out


def _parse_set(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
    key, raw = item.split("=", 1)
    return key.strip(), parse_value(raw.strip())


def _looks_like_value(token: str) -> bool:
    """True if ``token`` is a PETSc option *value*, not the next ``-option`` flag.

    A token that doesn't start with ``-`` is always a value. One that does start with ``-`` is
    normally the next flag (``-ksp_type cg -pc_type hypre``), but PETSc options also take
    negative numbers as values (``-ksp_rtol -1e-6``); those must not be mistaken for a flag.
    """
    if not token.startswith("-"):
        return True
    try:
        float(token)
    except ValueError:
        return False
    return True


def parse_petsc_options(s: str) -> dict[str, str | bool]:
    tokens = shlex.split(s)
    out: dict[str, str | bool] = {}
    i = 0
    while i < len(tokens):
        key = tokens[i].lstrip("-")
        if i + 1 < len(tokens) and _looks_like_value(tokens[i + 1]):
            out[key] = tokens[i + 1]
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def _format_validation_error(err: ValidationError) -> str:
    lines = ["Invalid configuration:"]
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"])
        lines.append(f"  {loc}: {e['msg']}")
    return "\n".join(lines)


def _resolve(base: Path, p: Path) -> Path:
    return p if p.is_absolute() else (base / p).resolve()


def load_config(
    path: Path,
    sets: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
    output_folder: Path | None = None,
    petsc_options: str | None = None,
) -> Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file {path} does not exist.")
    try:
        data = toml.loads(path.read_text())
    except toml.TomlDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from e

    for dotted, value in env_overrides(os.environ if environ is None else environ):
        apply_override(data, dotted, value)
    for item in sets:
        apply_override(data, *_parse_set(item))
    if petsc_options:
        solver = data.setdefault("solver", {})
        solver["petsc_options"] = {
            **solver.get("petsc_options", {}),
            **parse_petsc_options(petsc_options),
        }

    try:
        conf = Config.model_validate(data)
    except ValidationError as e:
        raise ConfigError(_format_validation_error(e)) from e

    base = path.parent.resolve()
    conf.cell.ode_file = _resolve(base, conf.cell.ode_file)
    conf.geometry.folder = _resolve(base, conf.geometry.folder)
    conf.output.folder = _resolve(base, conf.output.folder)
    if output_folder is not None:
        conf.output.folder = Path(output_folder).resolve()
    return conf


def _format_quantity(q: Quantity) -> str:
    """Format ``q`` so re-parsing it round-trips the magnitude's type exactly.

    ``str(quantity)`` uses ``"/"`` for units with a negative exponent (e.g. ``"1400 /
    centimeter"``), and pint's parser evaluates that as a division, silently promoting an
    integer magnitude to a float. Spelling every unit with an explicit ``**`` exponent instead
    (``"1400 centimeter**-1"``) avoids the division and round-trips losslessly, which matters
    because ``physics_hash`` must not change just because a config was dumped and reloaded.
    """
    magnitude, unit_exponents = q.to_tuple()
    parts = [f"{name}**{exp}" if exp != 1 else name for name, exp in unit_exponents]
    return f"{magnitude} {' * '.join(parts)}".strip()


def _toml_safe(value: Any) -> Any:
    if isinstance(value, Quantity):
        return _format_quantity(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _toml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_toml_safe(v) for v in value]
    return value


def dump_config(conf: Config, path: Path) -> None:
    # mode="python" hands back raw pint.Quantity objects so we can format them ourselves
    # (see _format_quantity); pydantic-pint's core schema declares a str return type for this
    # field regardless of mode, so pydantic warns about the "unexpected" Quantity value even
    # though returning it is exactly what mode="python" is documented to do. Harmless; silenced
    # so it doesn't drown out real warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
        dumped = conf.model_dump(mode="python", exclude_none=True)
    data = _toml_safe(dumped)
    Path(path).write_text(toml.dumps(data))


def file_hash(path: Path) -> str:
    path = Path(path)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError as e:
        raise ConfigError(f"{path}: file not found") from e


def physics_hash(conf: Config) -> str:
    """Hash of everything that must not change across a restart.

    Excludes run length (solver.end_time/num_beats/BCL -- BCL only sets the run length, it
    paces nothing), output and postprocess settings, and
    replaces cell.ode_file's path by its contents' hash (so a moved run can still restart).
    geometry.folder is kept only for geometry.type == "folder", where it *is* the mesh being
    simulated; for every other (generated) geometry type it's just a cache location, so it's
    dropped like any other non-physics path.
    """
    data = conf.model_dump(
        mode="json",
        exclude={"output": True, "postprocess": True, "solver": {"end_time", "num_beats", "BCL"}},
    )
    data["cell"]["ode_file"] = file_hash(conf.cell.ode_file)
    if data["geometry"].get("type") != "folder":
        data["geometry"].pop("folder", None)
    blob = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()
