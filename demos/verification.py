# %% [markdown]
# # Verifying Second-Order Temporal Convergence
#
# This demo details the process, challenges, and final mathematical formulation used to verify the second-order temporal convergence ($\mathcal{O}(\Delta t^2)$) of the operator splitting scheme in the `fenicsx-beat` cardiac electrophysiology library.
#
# ## Background
#
# The `fenicsx-beat` library solves the Monodomain model coupled with cellular ODEs. The continuous problem is given by:
#
# $$
# \frac{\partial v}{\partial t} = \nabla \cdot (M \nabla v) + I_{\text{stim}} - I_{\text{ion}}(v, s)
# $$
#
# $$
# \frac{\partial s}{\partial t} = f(v, s)
# $$
#
# To solve this efficiently, `fenicsx-beat` employs an **operator splitting scheme**. To achieve second-order accuracy in time, two conditions must be met:
#
# 1. **Strang Splitting:** The temporal integration must follow a half-step ODE, full-step PDE, half-step ODE sequence ($\theta = 0.5$).
#
# 2. **Crank-Nicolson:** The PDE step must use a $\theta$-rule with $\theta = 0.5$ for the diffusion operator.
#
# ## 2. The Challenges of Verification
#
# Verifying a second-order scheme using the Method of Manufactured Solutions (MMS) is not straight forward. We encountered several pitfalls that initially masked the true convergence behavior:
#
# ### Pitfall 1: The ODE Solver Bottleneck
#
# The default MMS tests used a 1st-order Forward Euler method for the ODE step. Because the total error is bounded by the lowest-order method in the coupled system, the Forward Euler solver dragged the entire scheme down to $\mathcal{O}(\Delta t)$.
# **Solution:** We implemented an *exact* analytical solver for the specific ODE used in the manufactured solution.
#
# ### Pitfall 2: The Midpoint Time Evaluation Bug
#
# Because $\theta = 0.5$, the UFL `time` variable evaluates at $t = T - \Delta t / 2$ during the final PDE step. When measuring the $L_2$ error at the end of the simulation, comparing the numerical solution against `v_exact_func(x, time)` accidentally introduced an artificial $\mathcal{O}(\Delta t)$ error.
# **Solution:** Manually force `time.value = T` immediately before the error evaluation.
#
# ### Pitfall 3: The Spatial Error Floor
#
# Using $P_1$ (Linear) spatial elements on a $150 \times 150$ grid resulted in a spatial error of roughly $2 \times 10^{-4}$. When $\Delta t$ became small, the temporal error vanished beneath this spatial error floor, causing the convergence rate to flatline at $0.0$.
# **Solution:** Upgraded the finite element space to `CG_2` (Quadratic elements), plunging the spatial error floor to $\sim 10^{-7}$.
#
# ### Pitfall 4: Commuting Operators and Exponential Blowup
#
# * **Attempt 1 (**$v \propto \sin(t)$**):** The ODE and PDE operators perfectly decoupled, meaning the splitting error was zero. The scheme accidentally became exact in time.
#
# * **Attempt 2 (**$v \propto e^t$**):** The exponential growth caused massive truncation errors at large time steps ($\Delta t = 0.5$). As $\Delta t$ halved, the error plummeted so violently that it registered fake "super-convergent" rates (e.g., 3.5, 14.4).
#
# * **Solution:** We used a "Goldilocks" solution—a damped oscillator ($v \propto \sin(t)e^t$)—which forces the ODE and PDE to interact without causing massive initial truncation errors.
#
# ## 3. The Final Mathematical Formulation
#
# We designed the following manufactured solution to test the coupled system:
#
# **Exact Solutions:**
#
# $$
# v(x,y,t) = \phi(x,y) \sin(t) e^t
# $$
#
# $$
# s(x,y,t) = \frac{1}{2} \phi(x,y) e^t (\sin(t) - \cos(t))
# $$
#
#
# Where the spatial basis is $\phi(x,y) = \cos(2\pi x)\cos(2\pi y)$.
#
# **ODE System:**
# We define the local dynamics such that:
#
# $$
# \frac{\partial v}{\partial t}\bigg|_{\text{ODE}} = -v
# $$
#
# $$
# \frac{\partial s}{\partial t}\bigg|_{\text{ODE}} = v
# $$
#
# **PDE System:**
# The total derivative of $v$ is $\frac{\partial v}{\partial t} = \phi(x,y) e^t (\sin(t) + \cos(t))$.
# Since the ODE handles $-v$, the PDE must handle the remainder:
#
# $$
# \frac{\partial v}{\partial t}\bigg|_{\text{PDE}} = \frac{\partial v}{\partial t}\bigg|_{\text{Total}} - \frac{\partial v}{\partial t}\bigg|_{\text{ODE}} = \phi(x,y) e^t (2\sin(t) + \cos(t))
# $$
#
# Setting this equal to the diffusion operator $\nabla \cdot \nabla v + I_{\text{stim}}$ (where $\nabla \cdot \nabla \phi = -8\pi^2 \phi$), we solve for the required stimulus:
#
# $$
# I_{\text{stim}}(x,y,t) = \phi(x,y) e^t [ (2 + 8\pi^2)\sin(t) + \cos(t) ]
# $$
#
# ## 4. The Verification Script
#
# The following script executes the convergence test using the formulations described above.

# %%
import numpy as np
from mpi4py import MPI
import dolfinx
import ufl
import beat


# Manufactured Solution: v = sin(t)*e^t
def v_exact_func(x, t):
    phi = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    return phi * ufl.sin(t) * ufl.exp(t)


def s_exact_func(x, t):
    phi = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    return 0.5 * phi * ufl.exp(t) * (ufl.sin(t) - ufl.cos(t))


def ac_func(x, t):
    phi = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    return phi * ufl.exp(t) * ((2.0 + 8.0 * ufl.pi**2) * ufl.sin(t) + ufl.cos(t))


def simple_ode_exact(states, t, dt, parameters):
    v, s = states
    values = np.zeros_like(states)
    values[0] = v * np.exp(-dt)
    values[1] = s + v * (1.0 - np.exp(-dt))
    return values

comm = MPI.COMM_WORLD

M = 1.0
T = 1.0
t0 = 0.0
theta = 0.5  # Strang Splitting & Crank-Nicolson
odespace = "CG_2"  # P2 elements to drop the spatial error floor

N = 150
mesh = dolfinx.mesh.create_unit_square(
    comm, N, N, dolfinx.cpp.mesh.CellType.triangle,
)
V_ode = beat.utils.space_from_string(odespace, mesh, dim=1)

errors = []
dts = [1.0 / (2**level) for level in range(1, 5)]

print(f"{'dt':<10} | {'L2 Error':<15} | {'Rate':<10}")
print("-" * 40)

for i, dt in enumerate(dts):
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    x = ufl.SpatialCoordinate(mesh)

    I_s = ac_func(x, time)

    pde = beat.MonodomainModel(
        time=time, mesh=mesh, M=M, I_s=I_s, params={"theta": theta, "degree": 2},
    )

    s = dolfinx.fem.Function(V_ode)
    s.interpolate(
        dolfinx.fem.Expression(
            s_exact_func(x, time), beat.utils.interpolation_points(V_ode),
        ),
    )

    v_ode = dolfinx.fem.Function(V_ode)
    v_ode.interpolate(
        dolfinx.fem.Expression(
            v_exact_func(x, time), beat.utils.interpolation_points(V_ode),
        ),
    )

    init_states = np.zeros((2, s.x.array.size))
    init_states[0, :] = v_ode.x.array
    init_states[1, :] = s.x.array

    ode = beat.odesolver.DolfinODESolver(
        v_ode=v_ode,
        v_pde=pde.state,
        fun=simple_ode_exact,
        init_states=init_states,
        parameters=None,
        num_states=2,
        v_index=0,
    )

    solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=theta)
    solver.solve((t0, T), dt=dt)

    # Advance time.value to the final endpoint T before calculating error
    time.value = T

    v_exact = v_exact_func(x, time)
    error_form = dolfinx.fem.form((pde.state - v_exact) ** 2 * ufl.dx)
    L2_error = np.sqrt(
        comm.allreduce(dolfinx.fem.assemble_scalar(error_form), MPI.SUM),
    )
    errors.append(L2_error)

    if i == 0:
        print(f"{dt:<10.5f} | {L2_error:<15.5e} | {'-':<10}")
    else:
        rate = np.log2(errors[i - 1] / errors[i])
        print(f"{dt:<10.5f} | {L2_error:<15.5e} | {rate:<10.4f}")
