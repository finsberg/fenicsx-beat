"""The reaction-eikonal (R-E) model of Neic et al. (2017)

An activation map :math:`t_a` (e.g. from :func:`beat.eikonal.solve_eikonal`) drives the cell
model through a foot current that starts at :math:`t_a`:

.. math::

    \\frac{dv}{dt} = \\frac{A}{\\tau_F} e^{(t - t_a)/\\tau_F}, \\qquad
    0 \\le t - t_a \\le T_{\\mathrm{foot}}, \\; v < V_{th}.

``A`` is in mV (paper Eqs. 45-46), so this already is :math:`dv/dt` and no :math:`C_m` enters.
The window follows openCARP: the paper's Eq. 27 states :math:`[t_a - T_{foot}, t_a]`, which
contradicts its own appendix and the openCARP implementation.

* R-E+ (with diffusion): ``MonodomainSplittingSolver(pde, FootCurrentODESolver(...))`` with a
  PDE model without stimulus.
* R-E- (without diffusion): ``ODEOnlySolver(FootCurrentODESolver(...))``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import dolfinx
import numpy as np
import numpy.typing as npt

from .monodomain_solver import EPS, ODESolver
from .telemetry import BaseMonitor, NullMonitor
from .utils import interpolation_points

logger = logging.getLogger(__name__)

# Unreached activation times are replaced by this before interpolation (inf * 0 = nan)
_LARGE = 1e30


def exponential_foot_rate(
    dT: npt.ArrayLike,
    v: npt.ArrayLike,
    A: npt.ArrayLike,
    tau_F: npt.ArrayLike,
    V_th: npt.ArrayLike,
    duration: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """:math:`dv/dt = (A/\\tau_F) e^{dT/\\tau_F}` where ``0 <= dT <= duration`` and ``v < V_th``,
    zero elsewhere (arguments broadcast; ``dT = t - t_a``)"""
    arrays = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (dT, v, A, tau_F, V_th, duration)),
    )
    dT_, v_, A_, tau_, Vth_, dur_ = arrays
    rate = np.zeros(dT_.shape)
    active = (dT_ >= 0.0) & (dT_ <= dur_) & (v_ < Vth_)
    rate[active] = A_[active] / tau_[active] * np.exp(dT_[active] / tau_[active])
    return rate


@dataclass(frozen=True)
class ExponentialFoot:
    """The exponential foot current of Neic et al. (2017)

    Parameters
    ----------
    A : float
        Amplitude (mV), by default 0.91
    tau_F : float
        Time constant (ms), by default 0.25
    V_th : float
        The foot stops once :math:`v` reaches this (mV), by default -30.0
    duration : float
        The foot is active for :math:`0 \\le t - t_a \\le` ``duration`` (ms), by default 5.0
    """

    A: float = 0.91
    tau_F: float = 0.25
    V_th: float = -30.0
    duration: float = 5.0

    def rate(self, dT: npt.ArrayLike, v: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """:math:`dv/dt` of the foot at ``dT = t - t_a``"""
        return exponential_foot_rate(dT, v, self.A, self.tau_F, self.V_th, self.duration)


class FootCurrentODESolver:
    """Wrap an ODE solver so that every step adds the foot current at the activation time

    After the wrapped solver's step from ``t0`` to ``t1 = t0 + dt``:
    ``v += dt * foot.rate(t1 - t_a, v)``. Evaluating at ``t1`` means the step that contains
    :math:`t_a` already gets the foot. All other ``ODESolver`` methods are the wrapped solver's.

    Parameters
    ----------
    ode : ODESolver
        The cell model solver (e.g. :class:`beat.odesolver.DolfinODESolver` or
        :class:`beat.odesolver.DolfinMultiODESolver`), with ``v_ode`` and ``v_pde``
    t_a : dolfinx.fem.Function
        The activation time; evaluated once into ``v_ode``'s space
    foot : ExponentialFoot | Mapping[int, ExponentialFoot], optional
        The foot, or one per marker value in ``markers``
    markers : dolfinx.fem.Function | None, optional
        Region markers on ``v_ode``'s space (required with a mapping ``foot``)
    allow_unreached : bool, optional
        Allow non-finite ``t_a`` (no foot there) instead of raising, by default False
    """

    def __init__(
        self,
        ode: ODESolver,
        t_a: dolfinx.fem.Function,
        foot: ExponentialFoot | Mapping[int, ExponentialFoot] = ExponentialFoot(),
        markers: dolfinx.fem.Function | None = None,
        allow_unreached: bool = False,
    ) -> None:
        self.ode = ode
        v_ode: dolfinx.fem.Function = ode.v_ode  # type: ignore[attr-defined]
        V = v_ode.function_space
        source = dolfinx.fem.Function(t_a.function_space)
        source.x.array[:] = np.where(np.isfinite(t_a.x.array), t_a.x.array, _LARGE)
        target = dolfinx.fem.Function(V)
        target.interpolate(dolfinx.fem.Expression(source, interpolation_points(V)))
        values = target.x.array.copy()
        unreached = ~np.isfinite(values) | (values > 0.5 * _LARGE)
        comm = V.mesh.comm
        if comm.allreduce(int(unreached.sum())) and not allow_unreached:
            raise ValueError("The activation time is not reached (not finite) everywhere")
        values[unreached] = np.inf
        self._t_a = values
        n = values.size
        if isinstance(foot, ExponentialFoot):
            params = [np.full(n, getattr(foot, k)) for k in ("A", "tau_F", "V_th", "duration")]
        else:
            if markers is None:
                raise TypeError("A foot per region needs the markers Function")
            m = markers.x.array
            if m.size != n:
                raise ValueError("markers must live on the same space as the ODE solver's v_ode")
            present = np.unique(np.concatenate(comm.allgather(np.unique(m))))
            missing = [int(k) for k in present if int(k) not in foot]
            if missing:
                raise ValueError(f"No foot given for marker {', '.join(map(str, missing))}")
            params = [np.empty(n) for _ in range(4)]
            for key, f in foot.items():
                where = m == key
                for array, name in zip(params, ("A", "tau_F", "V_th", "duration")):
                    array[where] = getattr(f, name)
        self._A, self._tau_F, self._V_th, self._duration = params

    @property
    def v_ode(self) -> dolfinx.fem.Function:
        return self.ode.v_ode  # type: ignore[attr-defined]

    @property
    def v_pde(self) -> dolfinx.fem.Function:
        return self.ode.v_pde  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes not found on the wrapper itself
        return getattr(self.ode, name)

    def to_dolfin(self) -> None:
        self.ode.to_dolfin()

    def from_dolfin(self) -> None:
        self.ode.from_dolfin()

    def ode_to_pde(self) -> None:
        self.ode.ode_to_pde()

    def pde_to_ode(self) -> None:
        self.ode.pde_to_ode()

    def states_to_dolfin(self, names: list[str] | None = None) -> list[dolfinx.fem.Function]:
        return self.ode.states_to_dolfin(names)

    def load_all_states(self, functions: list[dolfinx.fem.Function]) -> None:
        self.ode.load_all_states(functions)

    def step(self, t0: float, dt: float) -> None:
        self.ode.step(t0, dt)
        self.ode.to_dolfin()
        v = self.v_ode.x.array
        v += dt * exponential_foot_rate(
            t0 + dt - self._t_a,
            v,
            self._A,
            self._tau_F,
            self._V_th,
            self._duration,
        )
        self.ode.from_dolfin()


@dataclass
class ODEOnlySolver:
    """Advance only the cell model (with a :class:`FootCurrentODESolver`: the R-E- model)

    Same ``solve``/``step`` interface and telemetry as
    :class:`beat.monodomain_solver.MonodomainSplittingSolver`; :attr:`v` is the ODE solver's
    ``v_ode``.
    """

    ode: ODESolver
    monitor: BaseMonitor = field(default_factory=NullMonitor)

    def __post_init__(self) -> None:
        self.ode.to_dolfin()

    @property
    def v(self) -> dolfinx.fem.Function:
        return self.ode.v_ode  # type: ignore[attr-defined]

    def solve(self, interval: tuple[float, float], dt: float | None) -> None:
        T0, T = interval
        if dt is None:
            dt = T - T0
        t0, t1 = T0, T0 + dt
        while t1 < T + EPS:
            self.step((t0, t1))
            t0, t1 = t1, t1 + dt

    def step(self, interval: tuple[float, float]) -> None:
        t0, t1 = interval
        with self.monitor.track_time("total_step"):
            with self.monitor.track_time("ode_step"):
                self.ode.step(t0, t1 - t0)
            with self.monitor.track_time("ode_to_dolfin"):
                self.ode.to_dolfin()
        self.monitor.advance_step(t0, t1)
