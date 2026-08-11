from typing import Callable

import basix
import dolfinx
import numpy as np

from beat.odesolver import BaseDolfinODESolver
from beat.telemetry import BaseMonitor, NullMonitor


def _vector_element(scalar_element, num_states: int, mesh: dolfinx.mesh.Mesh):
    """Build a ``num_states``-component version of a scalar ``basix.ufl`` element,
    preserving whether it is a quadrature element or a standard (Lagrange/DG) one."""
    if scalar_element.element_family is None:
        # Quadrature elements are identified by family_name == "quadrature" and have no
        # ElementFamily; rebuild with the same points via value_shape instead of shape.
        return basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            degree=scalar_element.degree,
            value_shape=(num_states,),
        )
    return basix.ufl.element(
        scalar_element.family_name,
        mesh.basix_cell(),
        scalar_element.degree,
        shape=(num_states,),
        discontinuous=scalar_element.discontinuous,
    )


class ExternalOperatorODESolver(BaseDolfinODESolver):
    """An ODE solver that steps a cell model via
    `dolfinx-external-operator <https://github.com/a-latyshev/dolfinx-external-operator>`_'s
    ``FEMExternalOperator``, instead of the plain-numpy loop used by
    :class:`beat.odesolver.DolfinODESolver`.

    Unlike :class:`beat.irksome_odesolver.IrksomeODESolver`, ``fun`` keeps the exact same
    signature already used by :class:`beat.odesolver.DolfinODESolver`,
    ``fun(states, t, parameters, dt) -> new_states`` (plain NumPy or Numba-jitted arrays) —
    no UFL-returning variant of the cell model is needed. All ``num_states`` state
    variables are stored in a single vector-valued (blocked) ``dolfinx.fem.Function``,
    rather than a mixed-element one, since ``FEMExternalOperator`` operates on raw
    ``(num_cells, num_quadrature_points, num_states)`` arrays rather than UFL expressions.
    """

    def __init__(
        self,
        v_ode: dolfinx.fem.Function,
        v_pde: dolfinx.fem.Function,
        fun: Callable,
        init_states: np.ndarray,
        num_states: int,
        v_index: int = 0,
        parameters: np.ndarray | None = None,
        monitor: BaseMonitor | None = None,
    ):
        from dolfinx_external_operator import (
            FEMExternalOperator,
            evaluate_external_operators,
            evaluate_operands,
        )

        self._evaluate_operands = evaluate_operands
        self._evaluate_external_operators = evaluate_external_operators

        self.v_ode = v_ode
        self.v_pde = v_pde
        self.fun = fun
        self.num_states = num_states
        self.v_index = v_index
        self.parameters = parameters if parameters is not None else np.array([])
        self.monitor = monitor or NullMonitor()

        # Initialize base class properties
        self._initialize_metadata()

        # A single vector-valued (blocked) function to hold all ODE state variables. Its
        # dof layout is component-interleaved, i.e. states.x.array[i::num_states] is
        # state i at every point.
        mesh = v_ode.function_space.mesh
        el = v_ode.function_space.ufl_element()
        vec_el = _vector_element(el, num_states, mesh)
        self.S = dolfinx.fem.functionspace(mesh, vec_el)
        self.states = dolfinx.fem.Function(self.S, name="ode_states")

        if init_states is not None:
            for i in range(num_states):
                self.states.x.array[i::num_states] = init_states[i, :]

        # Current time/timestep, closed over by f_impl below and updated in step().
        self._t0 = 0.0
        self._dt = 0.0

        def f_impl(states_flat: np.ndarray) -> np.ndarray:
            # states_flat is packed as (num_cells * num_points, num_states); reshape to
            # (num_states, N) to match beat's usual `fun(states, t, parameters, dt)`.
            states = states_flat.reshape(-1, num_states).T
            new_states = self.fun(
                states=states,
                t=self._t0,
                parameters=self.parameters,
                dt=self._dt,
            )
            return np.asarray(new_states).T.flatten()

        def f_external(derivatives):
            if derivatives == (0,):
                return f_impl
            return NotImplementedError

        # The operator's only operand is the state function itself, and `coefficient=`
        # aliases its output back onto the same function: each step reads the previous
        # values from self.states and evaluate_external_operators() overwrites them with
        # the new ones in place (handling the dof-vs-quadrature-point layout correctly,
        # including the unrolled dofmap scatter needed for continuous Lagrange spaces).
        self._operators = [
            FEMExternalOperator(
                self.states,
                function_space=self.S,
                external_function=f_external,
                coefficient=self.states,
            ),
        ]

    def step(self, t0: float, dt: float) -> None:
        with self.monitor.track_time("ode_total_step"):
            self._t0 = t0
            self._dt = dt
            with self.monitor.track_time("ode_function_call"):
                coefficients = self._evaluate_operands(self._operators)
                self._evaluate_external_operators(self._operators, coefficients)

    def to_dolfin(self) -> None:
        """Move the voltage variable from the ODE state function into v_ode."""
        self.v_ode.x.array[:] = self.states.x.array[self.v_index :: self.num_states]

    def from_dolfin(self) -> None:
        """Move the voltage variable from v_ode back into the ODE state function."""
        self.states.x.array[self.v_index :: self.num_states] = self.v_ode.x.array

    @property
    def full_values(self):
        vals = np.zeros((self.num_states, self.v_ode.x.array.size))
        for i in range(self.num_states):
            vals[i, :] = self.states.x.array[i :: self.num_states]
        return vals

    def assign_all_states(self, functions) -> None:
        for i, f in enumerate(functions):
            f.x.array[:] = self.states.x.array[i :: self.num_states]

    def states_to_dolfin(self, names=None):
        functions = []
        for i in range(self.num_states):
            name = names[i] if names else f"state_{i}"
            f = dolfinx.fem.Function(self.v_ode.function_space, name=name)
            functions.append(f)
        self.assign_all_states(functions)
        return functions


class ExternalOperatorMultiODESolver(BaseDolfinODESolver):
    """Multi-region counterpart to :class:`ExternalOperatorODESolver`, analogous to how
    :class:`beat.odesolver.DolfinMultiODESolver` extends
    :class:`beat.odesolver.DolfinODESolver` and :class:`beat.irksome_odesolver.
    IrksomeMultiODESolver` extends :class:`ExternalOperatorODESolver`'s Irksome sibling:
    a different cell model (``fun``, ``init_states``, ``parameters``, ``num_states``) can
    be used in each region of the mesh, as identified by an integer-valued ``markers``
    function.

    As in :class:`beat.irksome_odesolver.IrksomeMultiODESolver`, one full-mesh external
    operator is built *per marker*, and only the degrees of freedom that actually belong
    to that marker are read back out; the cost of a step therefore scales with the number
    of regions. Prefer :class:`beat.odesolver.DolfinMultiODESolver` for problems with many
    regions or very large meshes.
    """

    def __init__(
        self,
        v_ode: dolfinx.fem.Function,
        v_pde: dolfinx.fem.Function,
        markers: dolfinx.fem.Function,
        fun: dict[int, Callable],
        init_states: dict[int, np.ndarray],
        num_states: dict[int, int],
        v_index: dict[int, int],
        parameters: dict[int, np.ndarray] | None = None,
        monitor: BaseMonitor | None = None,
    ):
        from dolfinx_external_operator import (
            FEMExternalOperator,
            evaluate_external_operators,
            evaluate_operands,
        )

        self._evaluate_operands = evaluate_operands
        self._evaluate_external_operators = evaluate_external_operators

        if v_ode.x.array.size != markers.x.array.size:
            raise RuntimeError("Marker and voltage need to be in the same function space")

        self.v_ode = v_ode
        self.v_pde = v_pde
        self.markers = markers
        self.fun = fun
        self.num_states = num_states
        self.v_index = v_index
        self.parameters = parameters if parameters is not None else {}
        self.monitor = monitor or NullMonitor()

        # Initialize base class properties
        self._initialize_metadata()

        self._marker_values = tuple(init_states.keys())
        mesh = v_ode.function_space.mesh
        el = v_ode.function_space.ufl_element()

        self._inds: dict[int, np.ndarray] = {}
        self._states: dict[int, dolfinx.fem.Function] = {}
        self._operators: dict[int, list] = {}
        self._t0 = 0.0
        self._dt = 0.0

        for marker in self._marker_values:
            where = markers.x.array == marker
            self._inds[marker] = where
            n = num_states[marker]

            values = self._broadcast_init_states(init_states[marker], n, int(where.sum()))

            vec_el = _vector_element(el, n, mesh)
            S = dolfinx.fem.functionspace(mesh, vec_el)
            states = dolfinx.fem.Function(S, name=f"ode_states_{marker}")
            for i in range(n):
                full = np.zeros(where.shape)
                full[where] = values[i, :]
                states.x.array[i::n] = full
            self._states[marker] = states

            marker_parameters = self.parameters.get(marker, np.array([]))

            def f_impl(states_flat, marker=marker, n=n, marker_parameters=marker_parameters):
                states = states_flat.reshape(-1, n).T
                new_states = fun[marker](
                    states=states,
                    t=self._t0,
                    parameters=marker_parameters,
                    dt=self._dt,
                )
                return np.asarray(new_states).T.flatten()

            def f_external(derivatives, f_impl=f_impl):
                if derivatives == (0,):
                    return f_impl
                return NotImplementedError

            # `coefficient=states` aliases the operator's output back onto the same
            # function that is also its operand (see ExternalOperatorODESolver).
            self._operators[marker] = [
                FEMExternalOperator(
                    states,
                    function_space=S,
                    external_function=f_external,
                    coefficient=states,
                ),
            ]

    @staticmethod
    def _broadcast_init_states(init_states: np.ndarray, num_states: int, num_points: int):
        """Broadcast ``init_states`` to shape ``(num_states, num_points)``, mirroring
        :class:`beat.odesolver.DolfinMultiODESolver`."""
        shape = (num_states, num_points)
        if np.shape(init_states) == shape:
            return np.copy(init_states)
        values = np.zeros(shape)
        values.T[:] = init_states
        return values

    def step(self, t0: float, dt: float) -> None:
        with self.monitor.track_time("ode_total_step"):
            self._t0 = t0
            self._dt = dt
            for marker in self._marker_values:
                with self.monitor.track_time(f"marker_{marker}_ode_step"):
                    operators = self._operators[marker]
                    coefficients = self._evaluate_operands(operators)
                    self._evaluate_external_operators(operators, coefficients)

    def to_dolfin(self) -> None:
        """Move the voltage variable from the per-marker state functions into v_ode."""
        arr = self.v_ode.x.array.copy()
        for marker in self._marker_values:
            where = self._inds[marker]
            n = self.num_states[marker]
            arr[where] = self._states[marker].x.array[self.v_index[marker] :: n][where]
        self.v_ode.x.array[:] = arr

    def from_dolfin(self) -> None:
        """Move the voltage variable from v_ode back into the per-marker state functions."""
        for marker in self._marker_values:
            where = self._inds[marker]
            n = self.num_states[marker]
            full = self._states[marker].x.array[self.v_index[marker] :: n]
            full[where] = self.v_ode.x.array[where]
            self._states[marker].x.array[self.v_index[marker] :: n] = full

    def values(self, marker: int) -> np.ndarray:
        where = self._inds[marker]
        n = self.num_states[marker]
        out = np.zeros((n, int(where.sum())))
        for i in range(n):
            out[i, :] = self._states[marker].x.array[i::n][where]
        return out

    def num_parameters(self, marker: int) -> int:
        return len(self.parameters[marker])

    def num_points(self, marker: int) -> int:
        return int(self._inds[marker].sum())

    def shape(self, marker: int) -> tuple[int, int]:
        return (self.num_states[marker], self.num_points(marker))

    @property
    def full_values(self):
        num_states_values = tuple(self.num_states.values())
        if any(n != num_states_values[0] for n in num_states_values):
            msg = (
                "Cannot get full values size states are not of equal size. "
                f"Have {self.num_states=}, use .values(marker) instead"
            )
            raise RuntimeError(msg)

        n = num_states_values[0]
        vals = np.zeros((n, self.v_ode.x.array.size))
        for marker in self._marker_values:
            where = self._inds[marker]
            for i in range(n):
                vals[i, where] = self._states[marker].x.array[i::n][where]
        return vals

    def assign_all_states(self, functions) -> None:
        num_states = self.num_states[self._marker_values[0]]
        assert len(functions) == num_states, "Number of functions must match number of states"
        for index, f in enumerate(functions):
            for marker in self._marker_values:
                where = self._inds[marker]
                n = self.num_states[marker]
                f.x.array[where] = self._states[marker].x.array[index::n][where]

    def states_to_dolfin(self, names=None):
        functions = []
        num_states = self.num_states[self._marker_values[0]]
        for i in range(num_states):
            name = names[i] if names else f"state_{i}"
            f = dolfinx.fem.Function(self.v_ode.function_space, name=name)
            functions.append(f)
        self.assign_all_states(functions)
        return functions
