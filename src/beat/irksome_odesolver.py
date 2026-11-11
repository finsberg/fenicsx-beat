from typing import Callable

import basix
import dolfinx
import numpy as np
import ufl

from beat.odesolver import BaseDolfinMultiODESolver, BaseDolfinODESolver
from beat.telemetry import BaseMonitor, NullMonitor


class IrksomeODESolver(BaseDolfinODESolver):
    """An ODE Solver mapping the DolfinODESolver interface to an Irksome stepper."""

    def __init__(
        self,
        v_ode: dolfinx.fem.Function,
        v_pde: dolfinx.fem.Function,
        fun: Callable,
        init_states: np.ndarray,
        butcher_tableau,
        time: dolfinx.fem.Constant,
        num_states: int,
        v_index: int = 0,
        parameters: dict | None = None,
        monitor: BaseMonitor | None = None,
    ):
        import irksome

        self.v_ode = v_ode
        self.v_pde = v_pde
        self.fun = fun
        self.num_states = num_states
        self.v_index = v_index
        self.time = time
        self.parameters = parameters if parameters is not None else np.array([])
        self.monitor = monitor or NullMonitor()

        # Initialize base class properties
        self._initialize_metadata()

        # Create a Mixed Function Space to hold all ODE state variables
        mesh = v_ode.function_space.mesh
        el = v_ode.function_space.ufl_element()
        mixed_el = basix.ufl.mixed_element([el] * num_states)
        self.W = dolfinx.fem.functionspace(mesh, mixed_el)
        self.states = dolfinx.fem.Function(self.W, name="ode_states")

        # Map and initialize the starting states from the underlying arrays
        self._maps: list[np.ndarray] = []
        for i in range(num_states):
            _, map_i = self.W.sub(i).collapse()
            # dolfinx's collapse() returns the dof map wrapped in a length-1 list; unwrap it
            # to a flat array so downstream indexing (including boolean masks) works as expected.
            dofs = np.asarray(map_i).reshape(-1)
            self._maps.append(dofs)
            if init_states is not None:
                self.states.x.array[dofs] = init_states[i, :]

        # Setup the UFL weak form for the ODE system
        w = ufl.TestFunctions(self.W)
        y = ufl.split(self.states)

        self.dt = dolfinx.fem.Constant(mesh, 0.0)

        # The user-provided `fun` must now return a tuple/list of UFL expressions
        rhs = self.fun(y, self.time, self.parameters)

        F = 0
        for i in range(num_states):
            F += (irksome.Dt(y[i]) * w[i] - rhs[i] * w[i]) * ufl.dx

        # Block Jacobi with CG is typically highly efficient for pure ODE mass matrices
        petsc_options = {
            "ksp_type": "cg",
            "pc_type": "bjacobi",
            "ksp_rtol": 1e-6,
        }

        self.stepper = irksome.stage_derivative.StageDerivativeTimeStepper(
            F,
            butcher_tableau,
            self.time,
            self.dt,
            self.states,
            solver_parameters=petsc_options,
            backend="dolfinx",
        )

    def step(self, t0: float, dt: float) -> None:
        with self.monitor.track_time("ode_total_step"):
            self.time.value = t0  # type: ignore[assignment]
            self.dt.value = dt  # type: ignore[assignment]
            self.stepper.advance()

    def _get_state(self, index: int) -> np.ndarray:
        return self.states.x.array[self._maps[index]]

    def _set_state(self, index: int, values) -> None:
        self.states.x.array[self._maps[index]] = values


class IrksomeMultiODESolver(BaseDolfinMultiODESolver):
    """Multi-region counterpart to :class:`IrksomeODESolver`, analogous to how
    :class:`beat.odesolver.DolfinMultiODESolver` extends
    :class:`beat.odesolver.DolfinODESolver`: a different cell model (``fun``,
    ``init_states``, ``parameters``, ``num_states``) can be used in each region of
    the mesh, as identified by an integer-valued ``markers`` function.

    All regions share the same ``butcher_tableau`` and ``time``/``dt`` stepping.
    Internally, one full-mesh mixed-element Irksome stepper is built *per marker*
    (as in :class:`IrksomeODESolver`), and only the degrees of freedom that
    actually belong to that marker are read back out; the values computed at the
    other degrees of freedom are discarded. This keeps the implementation simple
    and correct, but means the cost of a step scales with the number of regions
    (each region's stepper solves over the whole mesh), unlike
    :class:`beat.odesolver.DolfinMultiODESolver`, whose plain-numpy ODE systems
    are only ever sized to the points that need them. Prefer
    :class:`beat.odesolver.DolfinMultiODESolver` for problems with many regions
    or very large meshes.
    """

    def __init__(
        self,
        v_ode: dolfinx.fem.Function,
        v_pde: dolfinx.fem.Function,
        markers: dolfinx.fem.Function,
        fun: dict[int, Callable],
        init_states: dict[int, np.ndarray],
        butcher_tableau,
        time: dolfinx.fem.Constant,
        num_states: dict[int, int],
        v_index: dict[int, int],
        parameters: dict[int, np.ndarray] | None = None,
        monitor: BaseMonitor | None = None,
    ):
        import irksome

        self.v_ode = v_ode
        self.v_pde = v_pde
        self.markers = markers
        self.fun = fun
        self.num_states = num_states
        self.v_index = v_index
        self.time = time
        self.parameters = parameters if parameters is not None else {}
        self.monitor = monitor or NullMonitor()

        # Initialize base class properties
        self._initialize_metadata()

        self._setup_regions(init_states.keys())
        mesh = v_ode.function_space.mesh
        el = v_ode.function_space.ufl_element()

        self._maps: dict[int, list[np.ndarray]] = {}
        self._states: dict[int, dolfinx.fem.Function] = {}
        self._dt: dict[int, dolfinx.fem.Constant] = {}
        self._steppers: dict = {}

        for marker in self._marker_values:
            where = self._inds[marker]
            n = num_states[marker]

            values = self._broadcast_init_states(init_states[marker], n, int(where.sum()))

            mixed_el = basix.ufl.mixed_element([el] * n)
            W = dolfinx.fem.functionspace(mesh, mixed_el)
            states = dolfinx.fem.Function(W, name=f"ode_states_{marker}")

            maps: list[np.ndarray] = []
            for i in range(n):
                _, map_i = W.sub(i).collapse()
                # collapse() wraps the dof map in a length-1 list; unwrap to a flat array so
                # downstream indexing (including boolean masks) works as expected.
                dofs = np.asarray(map_i).reshape(-1)
                maps.append(dofs)
                full = np.zeros(where.shape)
                full[where] = values[i, :]
                states.x.array[dofs] = full
            self._maps[marker] = maps
            self._states[marker] = states

            w = ufl.TestFunctions(W)
            y = ufl.split(states)
            dt = dolfinx.fem.Constant(mesh, 0.0)
            self._dt[marker] = dt

            marker_parameters = self.parameters.get(marker, np.array([]))
            rhs = fun[marker](y, self.time, marker_parameters)

            F = 0
            for i in range(n):
                F += (irksome.Dt(y[i]) * w[i] - rhs[i] * w[i]) * ufl.dx

            # Block Jacobi with CG is typically highly efficient for pure ODE mass matrices
            petsc_options = {
                "ksp_type": "cg",
                "pc_type": "bjacobi",
                "ksp_rtol": 1e-6,
            }

            self._steppers[marker] = irksome.stage_derivative.StageDerivativeTimeStepper(
                F,
                butcher_tableau,
                self.time,
                dt,
                states,
                solver_parameters=petsc_options,
                backend="dolfinx",
            )

    def step(self, t0: float, dt: float) -> None:
        with self.monitor.track_time("ode_total_step"):
            self.time.value = t0  # type: ignore[assignment]
            for marker in self._marker_values:
                with self.monitor.track_time(f"marker_{marker}_ode_step"):
                    self._dt[marker].value = dt  # type: ignore[assignment]
                    self._steppers[marker].advance()

    def _get_region_state(self, marker: int, index: int) -> np.ndarray:
        return self._states[marker].x.array[self._maps[marker][index]][self._inds[marker]]

    def _set_region_state(self, marker: int, index: int, values) -> None:
        # Index the dof map itself so the write lands on x.array (fancy indexing x.array by the
        # map first would write into a copy).
        dofs = np.asarray(self._maps[marker][index])
        self._states[marker].x.array[dofs[self._inds[marker]]] = values
