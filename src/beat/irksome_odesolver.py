from typing import Callable

import basix
import dolfinx
import numpy as np
import ufl

from beat.odesolver import BaseDolfinODESolver
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
        self._maps = []
        for i in range(num_states):
            _, map_i = self.W.sub(i).collapse()
            self._maps.append(map_i)
            if init_states is not None:
                self.states.x.array[map_i] = init_states[i, :]

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
            self.time.value = t0
            self.dt.value = dt
            self.stepper.advance()

    def to_dolfin(self) -> None:
        """Move the voltage variable from the mixed ODE space into v_ode."""
        self.v_ode.x.array[:] = self.states.x.array[self._maps[self.v_index]]

    def from_dolfin(self) -> None:
        """Move the voltage variable from v_ode back into the mixed ODE space."""
        self.states.x.array[self._maps[self.v_index]] = self.v_ode.x.array

    @property
    def full_values(self):
        # FIX: Rely on the size of the target discrete array instead of the FEniCSx map object
        vals = np.zeros((self.num_states, self.v_ode.x.array.size))
        for i in range(self.num_states):
            vals[i, :] = self.states.x.array[self._maps[i]]
        return vals

    def assign_all_states(self, functions) -> None:
        for i, f in enumerate(functions):
            f.x.array[:] = self.states.x.array[self._maps[i]]

    def states_to_dolfin(self, names=None):
        functions = []
        for i in range(self.num_states):
            name = names[i] if names else f"state_{i}"
            f = dolfinx.fem.Function(self.v_ode.function_space, name=name)
            functions.append(f)
        self.assign_all_states(functions)
        return functions
