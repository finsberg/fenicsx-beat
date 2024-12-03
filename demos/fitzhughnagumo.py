# # A simple example of excitable tissue: The FitzHugh–Nagumo model
#
# The FitzHugh–Nagumo model (FHN), named after Richard FitzHugh (1922–2007) who suggested the system in 1961 and J. Nagumo et al. who created the equivalent circuit the following year, describing a prototype of an excitable system (e.g., a neuron).
#
# The FHN Model is an example of a relaxation oscillator because, if the external stimulus exceeds a certain threshold value, the system will exhibit a characteristic excursion in phase space, before the variables and relax back to their rest values.
#
# This behaviour is typical for spike generations (a short, nonlinear elevation of membrane voltage, diminished over time by a slower, linear recovery variable) in a neuron after stimulation by an external input current.
#
# The equations for this dynamical system read
#
# $$
# V_{amp} = V_{peak} - V_{rest} \\
# I_{Stim} =
#     \begin{cases}
#       stim\_amplitude & \text{if } t \geq stim\_start \text{ and } t \leq stim\_start + stim\_duration \\
#       0 & \text{otherwise}
#     \end{cases} \\
# \frac{ds}{dt} = b \cdot (-c_3 \cdot s + (V - V_{rest})) \\
# V_{th} = V_{amp} \cdot a + V_{rest} \\
# I = -s \cdot \left(\frac{c_2}{V_{amp}}\right) \cdot (V - V_{rest}) + \left(\frac{c_1}{V_{amp}^2}\right) \cdot (V - V_{rest}) \cdot (V - V_{th}) \cdot (-V + V_{peak}) \\
# \frac{dV}{dt} = I + i_{Stim} \\
# $$
#
# where $V$ is the membrane potential, $s$ is the gating variable, $V_{rest}$ is the resting potential, $V_{peak}$ is the peak potential, $V_{amp}$ is the amplitude of the action potential, $V_{th}$ is the threshold potential, $I$ is the current, $I_{Stim}$ is the external stimulus current, $a$, $b$, $c_1$, $c_2$, $c_3$ are parameters, and $t$ is time.
#
# The FitzHugh–Nagumo model is a simplified version of the Hodgkin–Huxley model, which is a more complex model of the biophysics of the action potential in neurons.
#
# We can formulate the right hand side of the FHN model as a Python function

# +
import numpy as np
from pathlib import Path


def rhs(t, states, parameters):

    # Assign states
    s = states[0]
    V = states[1]

    # Assign parameters
    V_peak = parameters[0]
    V_rest = parameters[1]
    a = parameters[2]
    b = parameters[3]
    c_1 = parameters[4]
    c_2 = parameters[5]
    c_3 = parameters[6]
    stim_amplitude = parameters[7]
    stim_duration = parameters[8]
    stim_start = parameters[9]

    # Assign expressions
    values = np.zeros_like(states, dtype=np.float64)
    V_amp = V_peak - V_rest
    i_Stim = np.where(
        t >= stim_start and t <= stim_start + stim_duration,
        stim_amplitude,
        0,
    )
    ds_dt = b * (-c_3 * s + (V - V_rest))
    values[0] = ds_dt
    V_th = V_amp * a + V_rest
    I = -s * (c_2 / V_amp) * (V - V_rest) + (
        ((c_1 / V_amp**2) * (V - V_rest)) * (V - V_th)
    ) * (-V + V_peak)
    dV_dt = I + i_Stim
    values[1] = dV_dt

    return values


# -


# Now, let us pick some parameters and initial conditions

V_peak = 40.0
V_rest = -85.0
a = 0.13
b = 0.013
c_1 = 0.26
c_2 = 0.1
c_3 = 1.0
stim_amplitude = 80.0
stim_duration = 1
stim_start = 1
parameters = np.array(
    [
        V_peak,
        V_rest,
        a,
        b,
        c_1,
        c_2,
        c_3,
        stim_amplitude,
        stim_duration,
        stim_start,
    ],
)
states = np.array([0.0, -85.0])

# We can now solve the ODE using a simple explicit Euler method

# +
dt = 0.01
times = np.arange(0, 1000, dt)
all_states = np.zeros((len(times), len(states)))
for i, t in enumerate(times):
    all_states[i, :] = states
    states += rhs(t, states, parameters) * dt

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(times, all_states[:, 0])
ax[0].set_ylabel("s")
ax[1].plot(times, all_states[:, 1])
ax[1].set_ylabel("V")
ax[1].set_xlabel("Time")
fig.savefig(Path("fitzhughnagumo_0D.png"))
# -

# Now, let us solve the FHN model in 2D using FEniCSx and FEniCSx-beat.
# Let us first create a unit square mesh of size $10 \times 10$ elements
#

from mpi4py import MPI
import dolfinx
import ufl
import beat

#
#

comm = MPI.COMM_WORLD
N = 10
mesh = dolfinx.mesh.create_unit_square(
    comm,
    N,
    N,
    dolfinx.cpp.mesh.CellType.triangle,
)

# We will also create a variables for the time
time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))

# We can use this time variable to define the external stimulus current which will now be applied to a specific region of the mesh

stim_expr = ufl.conditional(
    ufl.And(ufl.ge(time, stim_start), ufl.le(time, stim_start + stim_duration)),
    stim_amplitude,
    0.0,
)


# We will apply the simulus to the lower left corner of the mesh


# +
def stim_region(x):
    return np.logical_and(x[0] <= 0.5, x[1] <= 0.5)


stim_marker = 1
cells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, stim_region)
stim_tags = dolfinx.mesh.meshtags(
    mesh,
    mesh.topology.dim,
    cells,
    np.full(len(cells), stim_marker, dtype=np.int32),
)
# -

# Now we can create a `ufl.Measure` which can be used to integrate over the stimulus region

dx = ufl.Measure("dx", domain=mesh, subdomain_data=stim_tags)

# We can now define the stimulus current

I_s = beat.Stimulus(expr=stim_expr, dZ=dx, marker=stim_marker)

# We also need to make sure to turn off the stimulus current in the ODE solver

parameters[7] = 0.0

# We can now define the PDE model.

pde = beat.MonodomainModel(
    time=time,
    mesh=mesh,
    M=0.01,
    I_s=I_s,
    dx=dx,
)

# For the ODE model, we first need to define a function that will be used to solve the ODEs. We will use the same function as before in an explicit Euler scheme


def fun(t, states, parameters, dt):
    return states + dt * rhs(t, states, parameters)


# We also need to specify a function space for the ODE's. We will use a simple piecewise linear function space which will solve one ODE per node in the mesh.

ode_space = dolfinx.fem.functionspace(mesh, ("P", 1))
ode = beat.odesolver.DolfinODESolver(
    v_ode=dolfinx.fem.Function(ode_space),
    v_pde=pde.state,
    fun=fun,
    init_states=states,
    parameters=parameters,
    num_states=2,
    v_index=1,
)

# Here, we also need to specify the index of the state variable corresponding to the membrane potential, which in our case is the second state variable, with index 1.
# We combine the PDE and ODE models into a single solver using a splitting scheme

solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)

# Let us set up some way to save the solution to a file. We will use the VTXWriter from FEniCSx where you can visualize the solution using ParaView.

# +
import shutil

shutil.rmtree("fitzhughnagumo.bp", ignore_errors=True)
vtx = dolfinx.io.VTXWriter(
    comm,
    "fitzhughnagumo.bp",
    [solver.pde.state],
    engine="BP4",
)

# +
# We also visualize the solution using PyVista, if available


try:

    import pyvista

except ImportError:

    pyvista = None

else:

    pyvista.start_xvfb()
    plotter = pyvista.Plotter()
    viridis = plt.get_cmap("viridis")
    grid = pyvista.UnstructuredGrid(
        *dolfinx.plot.vtk_mesh(solver.pde.state.function_space.mesh),
    )
    grid.point_data["V"] = solver.pde.state.x.array
    grid.set_active_scalars("V")
    renderer = plotter.add_mesh(
        grid,
        show_edges=True,
        lighting=False,
        cmap=viridis,
        clim=[-90.0, 20.0],
    )
    gif_file = Path("fitzhughnagumo.gif")
    gif_file.unlink(missing_ok=True)
    plotter.view_xy()
    plotter.open_gif(gif_file.as_posix())

# We can now solve the model for a given time interval, say 1000 ms


T = 50
t = 0.0
i = 0
while t < T:
    v = solver.pde.state.x.array
    if i % 20 == 0:
        print(f"Solve for {t=:.2f}, {v.max() =}, {v.min() =}")
        vtx.write(t)
        if pyvista:
            grid.point_data["V"] = solver.pde.state.x.array
            plotter.write_frame()

    solver.step((t, t + dt))
    i += 1
    t += dt

vtx.close()
if pyvista:
    plotter.close()
# -
#


# ![_](fitzhughnagumo.gif)
