from __future__ import annotations

import abc
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

import dolfinx
import numpy as np
import numpy.typing as npt

from .telemetry import BaseMonitor, NullMonitor
from .utils import local_project

EPS = 1e-12
logger = logging.getLogger(__name__)


class ODEResults(NamedTuple):
    y: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]


def solve(
    fun: Callable,
    t_bound: float,
    states: npt.NDArray,
    V: npt.NDArray,
    V_index: int,
    dt: float,
    parameters: npt.NDArray,
    t0: float = 0.0,
    extra: dict[str, float | npt.NDArray] | None = None,
):
    if extra is None:
        extra = {}
    i = 0
    t = t0
    while t + dt < t_bound:
        fun(states=states, t=t, parameters=parameters, dt=dt, **extra)
        V[i, :] = states[V_index, :]
        i += 1
        t += dt


@dataclass
class ODESystemSolver:
    fun: Callable
    states: npt.NDArray
    parameters: npt.NDArray
    missing_variables: npt.NDArray | None = None
    _kwargs: dict[str, npt.NDArray] = field(default_factory=dict)
    monitor: BaseMonitor = field(default_factory=NullMonitor)

    def __post_init__(self):
        if self.missing_variables is not None:
            self._kwargs["missing_variables"] = self.missing_variables

    @property
    def num_points(self) -> int:
        return self.states.shape[1]

    @property
    def num_states(self) -> int:
        return self.states.shape[0]

    def step(self, t0: float, dt: float) -> None:
        with self.monitor.track_time("ode_total_step"):
            with self.monitor.track_time("ode_function_call"):
                updated_states = self.fun(
                    states=self.states,
                    t=t0,
                    parameters=self.parameters,
                    dt=dt,
                    **self._kwargs,
                )

            with self.monitor.track_time("ode_state_update"):
                self.states[:] = updated_states


class BaseDolfinODESolver(abc.ABC):
    """Base class of the ODE solvers that keep the cell states on a dolfinx function space

    ``v_ode`` is the potential on the ODE space (P1, DG, quadrature, ...) and ``v_pde`` the PDE
    model's potential; :meth:`ode_to_pde` and :meth:`pde_to_ode` project between them.
    Implements :class:`beat.monodomain_solver.ODESolver`.

    A backend stores its states however it likes and implements :meth:`step` and the two storage
    primitives :meth:`_get_state` and :meth:`_set_state`; everything that moves states between
    that storage and dolfinx Functions is built from them here. Multi-region backends derive from
    :class:`BaseDolfinMultiODESolver` instead.
    """

    v_ode: dolfinx.fem.Function
    v_pde: dolfinx.fem.Function
    num_states: Any
    v_index: Any
    _metadata: dict[str, Any] | None = None

    def _initialize_metadata(self):
        # basix.ufl names quadrature elements "quadrature" (lowercase), and
        # `degree` is a property, not a method.
        element = self.v_ode.ufl_element()
        if element.family_name == "quadrature":
            self._metadata = {"quadrature_degree": element.degree}
        else:
            self._metadata = None

    def _get_state(self, index: int) -> npt.NDArray:
        """State ``index`` at every point of ``v_ode``'s space (callers do not write to it)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _get_state")

    def _set_state(self, index: int, values: npt.ArrayLike) -> None:
        """Overwrite state ``index`` at every point of ``v_ode``'s space."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _set_state")

    def _states_per_point(self) -> int:
        """How many states each point carries: one Function per state in the methods below."""
        return int(self.num_states)

    def _check_count(self, items: Sequence[Any], what: str) -> None:
        num_states = self._states_per_point()
        if len(items) != num_states:
            raise ValueError(
                f"Number of {what} must match number of states, got {len(items)} {what}, "
                f"but number of states is {num_states}",
            )

    def to_dolfin(self) -> None:
        """Copy the potential from the cell states into ``v_ode``."""
        self.v_ode.x.array[:] = self._get_state(self.v_index)

    def from_dolfin(self) -> None:
        """Copy ``v_ode`` back into the cell states."""
        self._set_state(self.v_index, self.v_ode.x.array)

    def ode_to_pde(self) -> None:
        """Projects v_ode (DG0, quadrature space, ...) into v_pde (CG1)"""
        local_project(
            self.v_ode,
            self.v_pde.function_space,
            self.v_pde,
        )

    def pde_to_ode(self) -> None:
        """Projects v_pde (CG1) into v_ode (DG0, quadrature space, ...)"""
        local_project(
            self.v_pde,
            self.v_ode.function_space,
            self.v_ode,
        )

    @abc.abstractmethod
    def step(self, t0: float, dt: float) -> None:
        pass

    @property
    def full_values(self) -> npt.NDArray:
        """Every state at every point of ``v_ode``'s space, shape ``(num_states, n)``."""
        return np.stack([self._get_state(i) for i in range(self._states_per_point())])

    def assign_all_states(self, functions: list[dolfinx.fem.Function]) -> None:
        """Copy state ``i`` into ``functions[i]`` (one per state, on ``v_ode``'s space)."""
        self._check_count(functions, "functions")
        for index, f in enumerate(functions):
            f.x.array[:] = self._get_state(index)

    def load_all_states(self, functions: list[dolfinx.fem.Function]) -> None:
        """Inverse of :meth:`assign_all_states`: overwrite every ODE state from ``functions``
        (one per state, on ``v_ode``'s function space), e.g. when restarting from a checkpoint.
        Ends with :meth:`to_dolfin`.
        """
        self._check_count(functions, "functions")
        for index, f in enumerate(functions):
            self._set_state(index, f.x.array)
        self.to_dolfin()

    def states_to_dolfin(self, names: list[str] | None = None) -> list[dolfinx.fem.Function]:
        """New Functions on ``v_ode``'s space holding every state, named ``names`` (default
        ``state_<i>``)."""
        if names is None:
            names = [f"state_{i}" for i in range(self._states_per_point())]
        else:
            self._check_count(names, "names")
        V = self.v_ode.function_space
        functions = [dolfinx.fem.Function(V, name=name) for name in names]
        self.assign_all_states(functions)
        return functions


@dataclass
class DolfinODESolver(BaseDolfinODESolver):
    v_ode: dolfinx.fem.Function
    v_pde: dolfinx.fem.Function
    init_states: npt.NDArray
    parameters: npt.NDArray
    fun: Callable
    num_states: int
    v_index: int = 0
    missing_variables: npt.NDArray | None = None
    num_missing_variables: int = 0
    monitor: BaseMonitor = field(default_factory=NullMonitor)

    def __post_init__(self):
        if self.monitor is None:
            self.monitor = NullMonitor()
        if np.shape(self.init_states) == self.shape:
            self._values = np.copy(self.init_states)
        else:
            self._values = np.zeros(self.shape)
            self._values.T[:] = self.init_states

        self._ode = ODESystemSolver(
            fun=self.fun,
            states=self._values,
            parameters=self.parameters,
            missing_variables=self.missing_variables,
            monitor=self.monitor,
        )
        self._initialize_metadata()

    def _get_state(self, index: int) -> npt.NDArray:
        return self._values[index, :]

    def _set_state(self, index: int, values: npt.ArrayLike) -> None:
        self._values[index, :] = values

    @property
    def values(self):
        return self._values

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_states, self.num_points)

    @property
    def shape_missing_values(self) -> tuple[int, int]:
        return (self.num_missing_variables, self.num_points)

    @property
    def num_points(self) -> int:
        return self.v_ode.x.array.size

    def step(self, t0: float, dt: float):
        self._ode.step(t0=t0, dt=dt)

    @property
    def full_values(self):
        """The live state array, shape ``(num_states, num_points)``."""
        return self._values


@dataclass
class DolfinMultiODESolver(BaseDolfinODESolver):
    v_ode: dolfinx.fem.Function
    v_pde: dolfinx.fem.Function
    markers: dolfinx.fem.Function
    init_states: dict[int, npt.NDArray]
    parameters: dict[int, npt.NDArray]
    fun: dict[int, Callable]
    num_states: dict[int, int]
    v_index: dict[int, int]
    monitor: BaseMonitor = field(default_factory=NullMonitor)

    def __post_init__(self):
        if self.v_ode.x.array.size != self.markers.x.array.size:
            raise RuntimeError("Marker and voltage need to be in the same function space")

        self._marker_values = tuple(self.init_states.keys())
        self._num_points = {}
        self._odes = {}
        self._values = {}
        self._inds = {}

        self._initialize_full_values()

        for marker in self._marker_values:
            where = self.markers.x.array == marker
            self._num_points[marker] = where.sum()
            self._inds[marker] = where

            if np.shape(self.init_states[marker]) == self.shape(marker):
                self._values[marker] = np.copy(self.init_states[marker])
            else:
                self._values[marker] = np.zeros(self.shape(marker))
                self._values[marker].T[:] = self.init_states[marker]

            self._odes[marker] = ODESystemSolver(
                fun=self.fun[marker],
                states=self._values[marker],
                parameters=self.parameters[marker],
                monitor=self.monitor,
            )
        self._initialize_metadata()

    def _initialize_full_values(self):
        self._all_states_equal_size = (
            np.array(tuple(self.num_states.values())) == tuple(self.num_states.values())[0]
        ).all()
        if self._all_states_equal_size:
            self._full_values = np.zeros(
                (next(iter(self.num_states.values())), self.markers.x.array.size),
            )

    def to_dolfin(self) -> None:
        """Assign values from numpy array to dolfinx function"""
        arr = self.v_ode.x.array.copy()
        for marker in self._marker_values:
            arr[self._inds[marker]] = self._values[marker][self.v_index[marker], :]

        self.v_ode.x.array[:] = arr

    def from_dolfin(self) -> None:
        """Assign values from dolfinx function to numpy array"""
        arr = self.v_ode.x.array
        for marker in self._marker_values:
            self._values[marker][self.v_index[marker], :] = arr[self._inds[marker]]

    def values(self, marker: int) -> npt.NDArray:
        return self._values[marker]

    def num_parameters(self, marker: int) -> int:
        return len(self.parameters[marker])

    def shape(self, marker: int) -> tuple[int, int]:
        return (self.num_states[marker], self._num_points[marker])

    def num_points(self, marker: int) -> int:
        return self._num_points[marker]

    def step(self, t0: float, dt: float):
        with self.monitor.track_time("total_ode_step"):
            for marker, ode in self._odes.items():
                with self.monitor.track_time(f"marker_{marker}_ode_step"):
                    ode.step(t0=t0, dt=dt)

    def assign_all_states(self, functions: list[dolfinx.fem.Function]) -> None:
        num_states = self._values[self._marker_values[0]].shape[0]
        assert len(functions) == num_states, "Number of functions must match number of states"
        for index, f in enumerate(functions):
            for marker in self._marker_values:
                f.x.array[self._inds[marker]] = self._values[marker][index, :]

    def load_all_states(self, functions: list[dolfinx.fem.Function]) -> None:
        num_states = self._values[self._marker_values[0]].shape[0]
        assert len(functions) == num_states, "Number of functions must match number of states"
        for index, f in enumerate(functions):
            for marker in self._marker_values:
                self._values[marker][index, :] = f.x.array[self._inds[marker]]
        self.to_dolfin()

    def states_to_dolfin(self, names: list[str] | None = None) -> list[dolfinx.fem.Function]:
        V = self.v_ode.function_space
        functions = []
        num_states = self._values[self._marker_values[0]].shape[0]

        if names is not None:
            msg = (
                "Number of names must match number of states, got "
                f"{len(names)} names, but number of states is {num_states}"
            )
            assert len(names) == num_states, msg
        else:
            names = [f"state_{i}" for i in range(num_states)]

        for name in names:
            f = dolfinx.fem.Function(V, name=name)
            functions.append(f)

        self.assign_all_states(functions)

        return functions

    @property
    def full_values(self):
        if not self._all_states_equal_size:
            msg = (
                "Cannot get full values size states are not of equal size. "
                f"Have {self.num_states=}, use .values(marker) instead"
            )
            raise RuntimeError(msg)

        for marker in self._marker_values:
            where = self.markers.x.array == marker
            self._full_values[:, where] = self._values[marker]

        return self._full_values
