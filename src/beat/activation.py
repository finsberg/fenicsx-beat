"""Record local activation times during a simulation"""

from __future__ import annotations

from typing import Literal

import dolfinx
import numpy as np


class ActivationTracker:
    """Track the activation time of every dof of ``v``

    Call :meth:`update` after every time step. With ``method="max_dvdt"`` the activation time
    is the midpoint of the step with the largest :math:`dv/dt` (the marker used by Neic et al.
    2017), counted only where :math:`v` reached ``threshold``; with ``method="threshold"`` it is
    the first upward crossing of ``threshold``, interpolated linearly within the step.

    Parameters
    ----------
    v : dolfinx.fem.Function
        The transmembrane potential (any space; the result lives in the same space)
    method : {"max_dvdt", "threshold"}, optional
        How to define activation, by default "max_dvdt"
    threshold : float, optional
        The threshold (mV), by default -30.0
    t0 : float, optional
        The time of the current values of ``v``, by default 0.0

    Notes
    -----
    * NaN marks dofs that have never activated. ``scifem.evaluate_function`` combines ranks
      with a MAX over a ``-inf`` fill and mishandles NaN under MPI, so replace NaN by a
      sentinel value before point evaluation.
    * With ``max_dvdt`` the time is that of the steepest upstroke over the whole run. For
      multi-beat runs use one tracker per beat, or use ``threshold``.
    * Dofs already at or above ``threshold`` at ``t0`` never get a ``threshold`` crossing
      (they stay NaN), but they count as reached for ``max_dvdt``.
    * The ghost values of ``v`` must be current when :meth:`update` is called (true for the
      solvers in beat).
    """

    def __init__(
        self,
        v: dolfinx.fem.Function,
        method: Literal["max_dvdt", "threshold"] = "max_dvdt",
        threshold: float = -30.0,
        t0: float = 0.0,
    ) -> None:
        if method not in ("max_dvdt", "threshold"):
            raise ValueError(f"Unknown method {method!r}; use 'max_dvdt' or 'threshold'")
        self.v = v
        self.method = method
        self.threshold = threshold
        self._t_prev = float(t0)
        self._v_prev = v.x.array.copy()
        n = self._v_prev.size
        self._t_a = np.full(n, np.nan)
        self._max_rate = np.full(n, -np.inf)
        self._reached = self._v_prev >= threshold

    def update(self, t: float) -> None:
        """Account for the step from the previous time to ``t``"""
        dt = t - self._t_prev
        if dt <= 0:
            raise ValueError(f"Time must increase between updates ({self._t_prev} -> {t})")
        v = self.v.x.array
        if self.method == "max_dvdt":
            rate = (v - self._v_prev) / dt
            larger = rate > self._max_rate
            self._max_rate[larger] = rate[larger]
            self._t_a[larger] = self._t_prev + 0.5 * dt
            self._reached |= v >= self.threshold
        else:
            crossed = np.isnan(self._t_a) & (self._v_prev < self.threshold) & (v >= self.threshold)
            v_before = self._v_prev[crossed]
            fraction = (self.threshold - v_before) / (v[crossed] - v_before)
            self._t_a[crossed] = self._t_prev + fraction * dt
        self._v_prev[:] = v
        self._t_prev = float(t)

    @property
    def t_a(self) -> dolfinx.fem.Function:
        """The activation times so far (NaN where not activated)"""
        values = self._t_a.copy()
        if self.method == "max_dvdt":
            values[~self._reached] = np.nan
        t_a = dolfinx.fem.Function(self.v.function_space, name="activation_time")
        t_a.x.array[:] = values
        t_a.x.scatter_forward()
        return t_a
