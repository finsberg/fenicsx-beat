"""Build the PDE model and ODE solver for the configured backends."""

import importlib
import logging
from typing import Any

import dolfinx
import numpy as np

from ..base_model import BaseModel
from ..monodomain_model import MonodomainModel
from ..odesolver import BaseDolfinODESolver, DolfinMultiODESolver, DolfinODESolver
from ..telemetry import NullMonitor
from .cellmodel import CellModel
from .config import ConfigError, IrksomeODE, SolverConfig
from .stimulus import StimulusSet

logger = logging.getLogger(__name__)

_HINTS = {
    "irksome": (
        "The irksome backend needs Irksome. The PyPI release is broken for dolfinx; install "
        'with: pip install "irksome[dolfinx] @ git+https://github.com/firedrakeproject/Irksome.git"'
    ),
    "dolfinx_external_operator": (
        "The external_operator backend needs dolfinx-external-operator: "
        'pip install "fenicsx-beat[external_operator]"'
    ),
}


def _require(module: str):
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ConfigError(_HINTS[module]) from e


def check_backends_available(solver: SolverConfig) -> None:
    if solver.pde.type == "irksome" or solver.ode.type == "irksome":
        _require("irksome")
    if solver.ode.type == "external_operator":
        _require("dolfinx_external_operator")


def _tableau(tableau: str, stages: int):
    irksome = _require("irksome")
    if not hasattr(irksome, tableau):
        raise ConfigError(f"Unknown Irksome tableau {tableau!r}")
    return getattr(irksome, tableau)(stages)


def _irksome_fun(cell: CellModel):
    """Build the continuous-time, UFL-valued right-hand side the Irksome ODE backend needs.

    ``cell.fun`` (from ``beat.cli.cellmodel.build_cell_model``) is a discrete-time stepping
    scheme -- e.g. generalized Rush-Larsen -- called as ``fun(states, t, parameters, dt) ->
    new_states``, generated with gotranx's NumPy backend (``numpy.where`` for conditionals,
    etc.); ``IrksomeODESolver``/``IrksomeMultiODESolver`` instead evaluate ``F`` (a UFL form)
    once per stage and need ``fun(states, t, parameters) -> list[ufl.core.expr.Expr]``, the
    *continuous* ODE right-hand side, on UFL operands (``ufl.conditional`` instead of
    ``numpy.where``). Neither is a drop-in for the other, so this regenerates the right-hand
    side from ``cell.ode_file`` with gotranx's UFL backend instead of reusing ``cell.fun``.
    Not cached: this is a few milliseconds, done once at CLI start-up (model construction),
    never per time step.
    """
    import gotranx

    ode = gotranx.load_ode(cell.ode_file)
    code = gotranx.cli.gotran2ufl.get_code(ode)
    module: dict = {}
    exec(compile(code, str(cell.ode_file), "exec"), module)
    rhs = module["rhs"]

    def fun(states, t, parameters):
        return rhs(t, states, parameters)

    return fun


def ode_space(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.FunctionSpace:
    return dolfinx.fem.functionspace(mesh, ("Lagrange", 1))


def _broadcast_states(states: np.ndarray, num_states: int, num_points: int) -> np.ndarray:
    """Broadcast a per-state array (shape ``(num_states,)``, one scalar per state, the shape
    ``build_cell_model`` produces) to per-point shape ``(num_states, num_points)``.

    ``DolfinODESolver``/``DolfinMultiODESolver`` and the ``*Multi*`` Irksome/external-operator
    solvers all do this broadcast themselves; the single-region ``IrksomeODESolver`` and
    ``ExternalOperatorODESolver`` do not (they index ``init_states[i, :]``/``init_states[i, :]``
    assuming it is already per-point), so it is done here before constructing them.
    """
    shape = (num_states, num_points)
    if np.shape(states) == shape:
        return np.copy(states)
    values = np.zeros(shape)
    values.T[:] = states
    return values


def build_pde(
    solver: SolverConfig,
    mesh: dolfinx.mesh.Mesh,
    time: dolfinx.fem.Constant,
    M,
    stimuli: StimulusSet,
    C_m: float,
    monitor,
) -> BaseModel:
    petsc = {
        **BaseModel.default_parameters(solver.pde.linear_solver)["petsc_options"],
        **solver.petsc_options,
    }
    params: dict[str, Any] = {"petsc_options": petsc}
    kwargs = dict(time=time, mesh=mesh, M=M, I_s=stimuli.stimuli or None, C_m=C_m)
    if solver.pde.type == "theta":
        params["theta"] = solver.pde.theta
        return MonodomainModel(params=params, monitor=monitor or NullMonitor(), **kwargs)

    from ..irksome_model import IrksomeMonodomainModel

    # IrksomeMonodomainModel's __init__ builds the Irksome stepper directly (it does not call
    # BaseModel.__init__) and does not use a ``monitor`` -- it is silently discarded via
    # **kwargs if passed, so we don't pass it at all.
    return IrksomeMonodomainModel(
        butcher_tableau=_tableau(solver.pde.tableau, solver.pde.stages),
        params=params,
        **kwargs,
    )


def build_ode(
    solver: SolverConfig,
    cell: CellModel,
    markers: dolfinx.fem.Function | None,
    pde: BaseModel,
    time: dolfinx.fem.Constant,
    monitor,
) -> BaseDolfinODESolver:
    V = ode_space(pde.mesh)
    v_ode = dolfinx.fem.Function(V)
    n = len(cell.state_names)
    multi = markers is not None
    monitor = monitor or NullMonitor()
    kind = solver.ode.type

    if not multi:
        states, params = cell.init_states[0], cell.parameters[0]
        common = dict(
            v_ode=v_ode,
            v_pde=pde.state,
            num_states=n,
            v_index=cell.v_index,
            parameters=params,
            monitor=monitor,
        )
        if kind == "dolfin":
            # DolfinODESolver broadcasts a per-state (num_states,) array itself.
            return DolfinODESolver(fun=cell.fun, init_states=states, **common)

        # IrksomeODESolver/ExternalOperatorODESolver index init_states[i, :] directly and do
        # not broadcast a per-state scalar array themselves (unlike DolfinODESolver and the
        # *Multi* variants of all three backends), so do it here.
        init_states = _broadcast_states(states, n, v_ode.x.array.size)
        if kind == "irksome":
            from ..irksome_odesolver import IrksomeODESolver

            assert isinstance(solver.ode, IrksomeODE)
            return IrksomeODESolver(
                fun=_irksome_fun(cell),
                butcher_tableau=_tableau(solver.ode.tableau, solver.ode.stages),
                time=time,
                init_states=init_states,
                **common,
            )
        from ..external_operator_odesolver import ExternalOperatorODESolver

        return ExternalOperatorODESolver(fun=cell.fun, init_states=init_states, **common)

    ids = list(cell.init_states)
    common = dict(
        v_ode=v_ode,
        v_pde=pde.state,
        markers=markers,
        init_states=cell.init_states,
        num_states={i: n for i in ids},
        v_index={i: cell.v_index for i in ids},
        parameters=cell.parameters,
        monitor=monitor,
    )
    if kind == "dolfin":
        return DolfinMultiODESolver(fun={i: cell.fun for i in ids}, **common)
    if kind == "irksome":
        from ..irksome_odesolver import IrksomeMultiODESolver

        assert isinstance(solver.ode, IrksomeODE)
        irksome_fun = _irksome_fun(cell)
        return IrksomeMultiODESolver(
            fun={i: irksome_fun for i in ids},
            butcher_tableau=_tableau(solver.ode.tableau, solver.ode.stages),
            time=time,
            **common,
        )
    from ..external_operator_odesolver import ExternalOperatorMultiODESolver

    return ExternalOperatorMultiODESolver(fun={i: cell.fun for i in ids}, **common)
