![_](https://raw.githubusercontent.com/finsberg/fenicsx-beat/refs/heads/main/docs/_static/logo.png)

# fenicsx-beat
Cardiac electrophysiology simulator in FEniCSx

- Source code: https://github.com/finsberg/fenicsx-beat
- Documentation: https://finsberg.github.io/fenicsx-beat


## Install
You can install the library with pip
```
python3 -m pip install fenicsx-beat
```


## Getting started

```python
from mpi4py import MPI
import dolfinx
import ufl
import beat
import numpy as np


comm = MPI.COMM_WORLD
mesh = dolfinx.mesh.create_unit_square(comm, 10, 10, dolfinx.cpp.mesh.CellType.triangle)
time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
# Create stimulus
stim_expr = ufl.conditional(ufl.And(ufl.ge(time, 0.0), ufl.le(time, 0.5)), 200.0, 0.0)
stim_marker = 1
cells = dolfinx.mesh.locate_entities(
    mesh, mesh.topology.dim, lambda x: np.logical_and(x[0] <= 0.5, x[1] <= 0.5)
)
stim_tags = dolfinx.mesh.meshtags(
    mesh,
    mesh.topology.dim,
    cells,
    np.full(len(cells), stim_marker, dtype=np.int32),
)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=stim_tags)
I_s = beat.Stimulus(expr=stim_expr, dZ=dx, marker=stim_marker)
# Create PDE model
pde = beat.MonodomainModel(time=time, mesh=mesh, M=0.01, I_s=I_s, dx=dx)


# Define scheme to solve ODE
def fun(t, states, parameters, dt):
    v, s = states
    a, b = parameters
    values = np.zeros_like(states)
    values[0] = v - a * s * dt
    values[1] = s + b * v * dt
    return values


# Define ODE solver
ode_space = dolfinx.fem.functionspace(mesh, ("P", 1))
parameters = np.array([1.0, 1.0])
init_states = np.array([0.0, 0.0])
ode = beat.odesolver.DolfinODESolver(
    v_ode=dolfinx.fem.Function(ode_space),
    v_pde=pde.state,
    fun=fun,
    init_states=init_states,
    parameters=parameters,
    num_states=2,
    v_index=1,
)
# Combine PDE and ODE solver
solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)
# Solve
T = 5.0
t = 0.0
dt = 0.01
while t < T:
    v = solver.pde.state.x.array
    solver.step((t, t + dt))
    t += dt

```
![_](https://raw.githubusercontent.com/finsberg/fenicsx-beat/refs/heads/main/docs/_static/simple.gif)

See more examples in the [documentation](https://finsberg.github.io/fenicsx-beat)

## License
MIT

## Need help or having issues
Please submit an [issue](https://github.com/finsberg/fenicsx-beat/issues)
