"""gotranx code generation (cached), per-region parameters and steady-state initial states."""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mpi4py import MPI

import dolfinx
import numpy as np

from ..utils import expand_layer, expand_layer_biv
from .config import CellConfig, ConfigError, ms
from .geometry import CLIGeometry, get_marker

logger = logging.getLogger(__name__)


@dataclass
class CellModel:
    module: dict
    fun: Callable
    state_names: list[str]
    v_index: int
    init_states: dict[int, np.ndarray]
    parameters: dict[int, np.ndarray]
    region_ids: dict[str, int]


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def load_module(cell: CellConfig, cache_dir: Path, comm: MPI.Intracomm = MPI.COMM_WORLD) -> dict:
    """Generate (or reuse a cached) gotranx python module for ``cell.ode_file``.

    The codegen itself only runs on rank 0 (and only when the cache is missing); any failure
    there is broadcast to every rank so they all raise ``ConfigError`` together instead of the
    other ranks deadlocking at a barrier/read that rank 0 never reaches.
    """
    import gotranx

    if not cell.ode_file.is_file():
        raise ConfigError(f"cell.ode_file {cell.ode_file} does not exist")
    try:
        scheme = gotranx.schemes.Scheme[cell.scheme]
    except KeyError as e:
        options = ", ".join(s.name for s in gotranx.schemes.Scheme)
        raise ConfigError(f"Unknown cell.scheme {cell.scheme!r}; options: {options}") from e

    h = hashlib.sha256(cell.ode_file.read_bytes() + cell.scheme.encode()).hexdigest()[:16]
    module_path = Path(cache_dir) / f"cell_model_{h}.py"

    error: str | None = None
    if comm.rank == 0 and not module_path.is_file():
        try:
            logger.info(f"Generating cell model code from {cell.ode_file}")
            module_path.parent.mkdir(parents=True, exist_ok=True)
            ode = gotranx.load_ode(cell.ode_file)
            code = gotranx.cli.gotran2py.get_code(ode, scheme=[scheme])
            _atomic_write(module_path, code)
        except Exception as e:  # noqa: BLE001 - reported as ConfigError on every rank below
            error = f"Failed to generate cell model code from {cell.ode_file}: {e}"
    error = comm.bcast(error, root=0)
    if error is not None:
        raise ConfigError(error)

    module: dict = {}
    exec(compile(module_path.read_text(), str(module_path), "exec"), module)
    return module


def _index(module: dict, kind: str, name: str) -> int:
    try:
        return module[f"{kind}_index"](name)
    except KeyError as e:
        raise ConfigError(f"Cell model has no {kind} named {name!r}") from e


def state_index(model: CellModel, name: str) -> int:
    if name not in model.state_names:
        raise ConfigError(f"Cell model has no state named {name!r}; states: {model.state_names}")
    return model.state_names.index(name)


def _state_names(module: dict) -> list[str]:
    n = len(module["init_state_values"]())
    names: list[str | None] = [None] * n
    for name, i in module["state"].items():
        names[i] = name
    return names  # type: ignore[return-value]


def _steady_state(
    cell: CellConfig,
    module: dict,
    fun: Callable,
    states: np.ndarray,
    params: np.ndarray,
    name: str,
    cache_dir: Path,
    comm: MPI.Intracomm,
) -> np.ndarray:
    ss = cell.steady_state
    assert ss is not None
    key = json.dumps(
        {
            "ode": hashlib.sha256(cell.ode_file.read_bytes()).hexdigest(),
            "scheme": cell.scheme,
            "params": params.tolist(),
            "ss": ss.model_dump(mode="json"),
        },
        sort_keys=True,
    )
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = Path(cache_dir) / "init_states" / f"{name}_{h}.npy"

    if not path.is_file():
        from ..single_cell import get_steady_state

        error: str | None = None
        if comm.rank == 0:
            try:
                logger.info(f"Computing steady state for region {name!r} ({ss.num_beats} beats)")
                path.parent.mkdir(parents=True, exist_ok=True)
                track_indices = [_index(module, "state", s) for s in ss.track] or None
                result = get_steady_state(
                    fun=fun,
                    init_states=states,
                    parameters=params,
                    outdir=path.parent / name,
                    # get_steady_state's BCL is annotated int but only ever used as a plain
                    # number (np.arange/multiplication), so a ms-converted float is safe here.
                    BCL=ms(ss.BCL),  # type: ignore[arg-type]
                    nbeats=ss.num_beats,
                    dt=ms(ss.dt),
                    track_indices=track_indices,
                )
                tmp = path.with_suffix(f".tmp{os.getpid()}.npy")
                np.save(tmp, result)
                os.replace(tmp, path)
            except Exception as e:  # noqa: BLE001 - reported as ConfigError on every rank below
                error = f"Failed to compute steady state for region {name!r}: {e}"
        error = comm.bcast(error, root=0)
        if error is not None:
            raise ConfigError(error)

    return np.load(path)


def build_cell_model(
    cell: CellConfig,
    cache_dir: Path,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> CellModel:
    module = load_module(cell, cache_dir, comm)
    if cell.scheme not in module:
        raise ConfigError(f"Generated module has no scheme function {cell.scheme!r}")
    fun = module[cell.scheme]
    names = _state_names(module)
    if cell.v_name not in names:
        raise ConfigError(f"cell.v_name {cell.v_name!r} is not a state; states: {names}")

    region_ids = {name: i for i, name in enumerate(cell.region_names())}
    init_states: dict[int, np.ndarray] = {}
    parameters: dict[int, np.ndarray] = {}
    for name, rid in region_ids.items():
        overrides = {
            **cell.parameters,
            **(cell.regions[name].parameters if name in cell.regions else {}),
        }
        for p in overrides:
            _index(module, "parameter", p)  # raises ConfigError for unknown names
        params = module["init_parameter_values"](**overrides)
        states = module["init_state_values"]()
        if cell.steady_state is not None:
            states = _steady_state(cell, module, fun, states, params, name, cache_dir, comm)
        init_states[rid] = states
        parameters[rid] = params

    return CellModel(
        module=module,
        fun=fun,
        state_names=names,
        v_index=names.index(cell.v_name),
        init_states=init_states,
        parameters=parameters,
        region_ids=region_ids,
    )


def build_region_markers(
    cell: CellConfig,
    geo: CLIGeometry,
    V: dolfinx.fem.FunctionSpace,
) -> dolfinx.fem.Function | None:
    layers = cell.layers
    if layers.method == "none":
        return None

    if layers.method == "transmural":
        if geo.ffun is None:
            raise ConfigError("cell.layers.method = 'transmural' requires facet markers")
        endo = layers.endo_markers or (["ENDO"] if "ENDO" in geo.markers else ["LV", "RV"])
        epi = get_marker(geo, layers.epi_marker)[0]
        ids = [get_marker(geo, m)[0] for m in endo]
        if len(ids) == 1:
            return expand_layer(
                V=V,
                ft=geo.ffun,
                endo_marker=ids[0],
                epi_marker=epi,
                endo_size=layers.endo_size,
                epi_size=layers.epi_size,
                output_mid_marker=0,
                output_endo_marker=1,
                output_epi_marker=2,
            )
        if len(ids) == 2:
            return expand_layer_biv(
                V=V,
                ft=geo.ffun,
                endo_lv_marker=ids[0],
                endo_rv_marker=ids[1],
                epi_marker=epi,
                endo_size=layers.endo_size,
                epi_size=layers.epi_size,
                output_mid_marker=0,
                output_endo_marker=1,
                output_epi_marker=2,
            )
        raise ConfigError("cell.layers.endo_markers must have 1 (LV) or 2 (LV, RV) entries")

    # cell_markers: region i <- geometry cell marker layers.map[region]
    if geo.cfun is None:
        raise ConfigError("cell.layers.method = 'cell_markers' requires cell markers")
    markers = dolfinx.fem.Function(V)
    markers.x.array[:] = -1
    tdim = geo.mesh.topology.dim
    for rid, (_region, marker_name) in enumerate(layers.map.items()):
        value, dim = get_marker(geo, marker_name)
        if dim != tdim:
            raise ConfigError(f"Marker {marker_name!r} is not a cell marker (dim {dim})")
        cells = geo.cfun.find(value)
        dofs = dolfinx.fem.locate_dofs_topological(V, tdim, cells)
        markers.x.array[dofs] = rid
    if np.any(markers.x.array < 0):
        raise ConfigError("cell.layers.map does not cover every mesh cell")
    return markers
