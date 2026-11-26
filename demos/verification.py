# %% [markdown]
# # Verifying Second-Order Temporal Convergence
#
# This demo details the process, challenges, and final mathematical formulation used to verify the second-order temporal convergence ($\mathcal{O}(\Delta t^2)$) of the operator splitting scheme in the `fenicsx-beat` cardiac electrophysiology library.
#
# See the [mathematical background](../docs/math_background.md) page for a general derivation of the equations and the splitting scheme summarized below, and the [convergence demo](monodomain_convergence.py) for the companion check of the (first-order) spatial and Godunov-splitting convergence rates.
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
# 1. **Strang Splitting:** The temporal integration must follow a half-step ODE, full-step PDE, half-step ODE sequence (`splitting="strang"`).
#
# 2. **Crank-Nicolson:** The PDE step must use a $\theta$-rule with $\theta = 0.5$ for the diffusion operator.
#
# ## 2. The Challenges of Verification
#
# Verifying a second-order scheme using the Method of Manufactured Solutions (MMS) is not straight forward. We encountered several pitfalls that initially masked the true convergence behavior:
#
# ### Pitfall 1: The ODE Solver Bottleneck
#
# The total error is bounded by the lowest-order method in the coupled system, so a first-order
# cell scheme (such as Forward Euler) eventually drags Strang splitting down to $\mathcal{O}(\Delta t)$.
# How soon depends on the problem: with this demo's strong diffusion ($M = 1$, eigenvalue
# $8\pi^2$) the PDE error dominates and a first-order cell scheme can still *look* second order.
# **Solution:** The study below uses an *exact* analytical solver for the specific ODE of the
# manufactured solution, which isolates the PDE step and the splitting. Sections 5 and 6 then use
# gotranx-generated cell schemes with weak diffusion, where the cell scheme's order shows.
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
theta = 0.5  # Crank-Nicolson PDE step (the splitting is chosen below)
odespace = "CG_2"  # P2 elements to drop the spatial error floor

N = 150
mesh = dolfinx.mesh.create_unit_square(
    comm, N, N, dolfinx.mesh.CellType.triangle,
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
        parameters=np.array([]),
        num_states=2,
        v_index=0,
    )

    solver = beat.SplittingSolver(pde=pde, ode=ode, splitting="strang")
    solver.solve((t0, T), dt=dt)

    # Advance time.value to the final endpoint T before calculating error
    time.value = T  # type: ignore[assignment]

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

# %% [markdown]
# ## 5. Cell schemes from gotranx
#
# Real cell models are stepped by a scheme generated with
# [gotranx](https://github.com/finsberg/gotranx). The Rush-Larsen family treats each state's
# own (diagonal) dependence exponentially and the rest explicitly: `generalized_rush_larsen` is
# first order, `generalized_rush_larsen_2` second order. (On this model every state is affine in
# itself, so `rush_larsen` and `generalized_rush_larsen` coincide; section 6 uses `rush_larsen`.) To see the difference we use a cell model
# whose states are coupled,
#
# $$
# \frac{dv}{dt} = -v - s, \qquad \frac{ds}{dt} = v - s,
# $$
#
# with the manufactured solution $v = \phi \sin t$, $s = \phi(\sin t - \cos t)/2$,
# $\phi = \cos 2\pi x \cos 2\pi y$, and *weak* diffusion $M = 0.01$. We measure the error at
# $T = 1$ against a run with a much smaller time step on the same mesh, so the spatial error
# cancels and a $16 \times 16$ mesh is enough.

# %%
import gotranx
import matplotlib.pyplot as plt

MMS_ODE = """
states(v=0, s=-0.5)
dv_dt = -v - s
ds_dt = v - s
"""
schemes = ["explicit_euler", "generalized_rush_larsen", "generalized_rush_larsen_2"]
code = gotranx.cli.gotran2py.get_code(
    gotranx.load.ode_from_string(MMS_ODE, name="strang_mms"),
    scheme=[gotranx.schemes.Scheme[s] for s in schemes],
)
mms_model: dict = {}
exec(code, mms_model)


def rms_difference(a, b):
    n = a.function_space.dofmap.index_map.size_local * a.function_space.dofmap.index_map_bs
    d = a.x.array[:n] - b.x.array[:n]
    return np.sqrt(comm.allreduce(np.sum(d**2), MPI.SUM) / comm.allreduce(n, MPI.SUM))


def mms_final_v(scheme, splitting, dt, M, N=16, T=1.0):
    mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.mesh.CellType.triangle)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    x = ufl.SpatialCoordinate(mesh)
    phi = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    s_t = (ufl.sin(time) - ufl.cos(time)) / 2
    I_s = phi * (ufl.cos(time) + ufl.sin(time) + s_t + 8 * ufl.pi**2 * M * ufl.sin(time))
    pde = beat.MonodomainModel(time=time, mesh=mesh, M=M, I_s=I_s, params={"theta": 0.5})
    V_ode = beat.utils.space_from_string("P_1", mesh, dim=1)
    s0 = dolfinx.fem.Function(V_ode)
    s0.interpolate(dolfinx.fem.Expression(phi * s_t, beat.utils.interpolation_points(V_ode)))
    init_states = np.zeros((2, s0.x.array.size))
    init_states[mms_model["state_index"]("s")] = s0.x.array  # gotranx orders states s, v
    ode = beat.odesolver.DolfinODESolver(
        v_ode=dolfinx.fem.Function(V_ode),
        v_pde=pde.state,
        fun=mms_model[scheme],
        init_states=init_states,
        parameters=mms_model["init_parameter_values"](),
        num_states=2,
        v_index=mms_model["state_index"]("v"),
    )
    beat.SplittingSolver(pde=pde, ode=ode, splitting=splitting).solve((0.0, T), dt=dt)
    return pde.state


def self_convergence(run, dts, dt_ref):
    ref = run(dt_ref)
    errors = np.array([rms_difference(run(dt), ref) for dt in dts])
    return errors, np.log2(errors[:-1] / errors[1:])


mms_dts = [2.0**-k for k in range(2, 7)]
cases = [(s, "strang") for s in schemes] + [("generalized_rush_larsen_2", "godunov")]
fig, ax = plt.subplots()
for scheme, splitting in cases:
    errors, rates = self_convergence(
        lambda dt, scheme=scheme, splitting=splitting: mms_final_v(scheme, splitting, dt, M=0.01), mms_dts, 2.0**-10,
    )
    print(f"{splitting:8s} {scheme:26s} rates {np.round(rates, 2)}")
    ax.loglog(mms_dts, errors, "o-", label=f"{splitting}, {scheme}")
ax.loglog(mms_dts, [0.1 * dt for dt in mms_dts], "k--", label=r"$\mathcal{O}(\Delta t)$")
ax.loglog(mms_dts, [0.1 * dt**2 for dt in mms_dts], "k:", label=r"$\mathcal{O}(\Delta t^2)$")
ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel("RMS difference to reference")
ax.legend()
fig.savefig("verification_gotranx_schemes.png")

# %% [markdown]
# Only Strang splitting with `generalized_rush_larsen_2` converges at second order. Godunov
# splitting stays first order even with `generalized_rush_larsen_2`: the splitting error dominates.
# The rates of the first-order schemes settle towards 1 from both sides: those of `explicit_euler`
# fall towards it and those of `generalized_rush_larsen` rise towards it. With strong diffusion
# the picture changes:

# %%
for scheme in ["generalized_rush_larsen", "generalized_rush_larsen_2"]:
    _, rates = self_convergence(
        lambda dt, scheme=scheme: mms_final_v(scheme, "strang", dt, M=1.0), mms_dts, 2.0**-10,
    )
    print(f"M = 1, strang {scheme:26s} rates {np.round(rates, 2)}")

# %% [markdown]
# With $M = 1$ even the first-order `generalized_rush_larsen` looks second order: the diffusion
# error hides it. Convergence studies of the cell scheme need weak diffusion.

# %% [markdown]
# ## 6. A real cell model
#
# The same holds for the ten Tusscher-Panfilov 2006 epicardial cell model on a
# $10 \times 2$ mm slab. A wave is started from the initial condition ($V = 0$ mV for
# $x < 1$ mm, the cell's own stimulus switched off), which avoids the time discontinuity of a
# stimulus pulse. We compare $V$ at $T = 10$ ms with a reference run with
# $\Delta t = 0.0015625$ ms. The model has piecewise rate functions (for example, the time
# constants of the sodium gates $h$ and $j$ jump at $V = -40$ mV). A jump like that can pull any
# fixed-step method back to first order, but it does not show at these step sizes: Strang
# splitting with `rush_larsen_2` approaches second order (rates 1.35, 1.8, 1.94
# below). A separate run with time steps down to $\Delta t = 0.003125$ ms (against a
# $\Delta t = 0.000390625$ ms reference, too slow to include here) gave rates 1.79, 1.92, 1.96,
# 2.01 (measured with `rush_larsen_2`), so the rate reaches 2.

# %%
from pathlib import Path

tp06_file = (
    Path.cwd() / ".." / "odes" / "tentusscher_panfilov_2006"
    / "tentusscher_panfilov_2006_epi_cell.ode"
)
tp06_code = gotranx.cli.gotran2py.get_code(
    gotranx.load_ode(tp06_file),
    scheme=[
        gotranx.schemes.Scheme.rush_larsen,
        gotranx.schemes.Scheme.rush_larsen_2,
    ],
)
tp06: dict = {}
exec(tp06_code, tp06)
slab = dolfinx.mesh.create_rectangle(comm, [[0.0, 0.0], [10.0, 2.0]], [40, 8])


def tp06_final_v(scheme, splitting, dt, T=10.0, merge_half_steps=False):
    time = dolfinx.fem.Constant(slab, dolfinx.default_scalar_type(0.0))
    pde = beat.MonodomainModel(time=time, mesh=slab, M=0.1, params={"theta": 0.5})  # mm^2/ms
    V_ode = beat.utils.space_from_string("P_1", slab, dim=1)
    x = V_ode.tabulate_dof_coordinates()[:, 0]
    iv = tp06["state_index"]("V")
    init_states = np.tile(tp06["init_state_values"]()[:, None], (1, x.size))
    init_states[iv, x < 1.0] = 0.0
    pde.state.x.array[:] = init_states[iv]
    ode = beat.odesolver.DolfinODESolver(
        v_ode=dolfinx.fem.Function(V_ode),
        v_pde=pde.state,
        fun=tp06[scheme],
        init_states=init_states,
        parameters=tp06["init_parameter_values"](stim_amplitude=0.0),
        num_states=init_states.shape[0],
        v_index=iv,
    )
    beat.SplittingSolver(
        pde=pde, ode=ode, splitting=splitting, merge_half_steps=merge_half_steps,
    ).solve((0.0, T), dt=dt)
    return pde.state


tp06_dts = [0.1, 0.05, 0.025, 0.0125]
fig, ax = plt.subplots()
for scheme, splitting in [
    ("rush_larsen", "godunov"),
    ("rush_larsen", "strang"),
    ("rush_larsen_2", "strang"),
]:
    errors, rates = self_convergence(
        lambda dt, scheme=scheme, splitting=splitting: tp06_final_v(scheme, splitting, dt), tp06_dts, 0.0015625,
    )
    print(f"{splitting:8s} {scheme:26s} rates {np.round(rates, 2)}")
    ax.loglog(tp06_dts, errors, "o-", label=f"{splitting}, {scheme}")
ax.loglog(tp06_dts, [200 * dt for dt in tp06_dts], "k--", label=r"$\mathcal{O}(\Delta t)$")
ax.loglog(tp06_dts, [400 * dt**2 for dt in tp06_dts], "k:", label=r"$\mathcal{O}(\Delta t^2)$")
ax.set_xlabel(r"$\Delta t$ (ms)")
ax.set_ylabel("RMS difference in $V$ to reference (mV)")
ax.legend()
fig.savefig("verification_tp06.png")

# %% [markdown]
# Godunov splitting and Strang splitting with the first-order `rush_larsen` stay at
# (or below) first order (rates 0.78, 1.05, 0.73 and 1.06, 0.74, 0.88). Strang splitting with `rush_larsen_2` approaches second order. In the
# [CLI](../docs/cli.md) this is `solver.splitting = "strang"` with
# `cell.scheme = "rush_larsen_2"`.

# %% [markdown]
# ## 7. Merging Strang half-steps
#
# Strang splitting takes two ODE half-steps per time step. The closing half-step of one step and
# the opening half-step of the next can be merged into one ODE step of length $\Delta t$
# (`merge_half_steps=True`), which costs what Godunov splitting costs. The scheme stays second
# order, but the cell scheme now steps with $\Delta t$ instead of $\Delta t/2$, so at the same
# $\Delta t$ its error is larger than plain Strang splitting's. The fair comparison is at the same
# cost: merged Strang at $\Delta t$ against plain Strang at $2\Delta t$.

# %%
from time import perf_counter

tp06_ref = tp06_final_v("rush_larsen_2", "strang", 0.0015625)
print(f"{'case':16s} {'dt (ms)':>8s} {'error (mV)':>11s} {'time (s)':>9s}")
for name, splitting, merge in [
    ("godunov", "godunov", False),
    ("strang", "strang", False),
    ("strang, merged", "strang", True),
]:
    for dt in (0.025, 0.0125):
        tic = perf_counter()
        v = tp06_final_v("rush_larsen_2", splitting, dt, merge_half_steps=merge)
        seconds = perf_counter() - tic
        print(f"{name:16s} {dt:8.4f} {rms_difference(v, tp06_ref):11.3f} {seconds:9.2f}")
errors, rates = self_convergence(
    lambda dt: tp06_final_v("rush_larsen_2", "strang", dt, merge_half_steps=True),
    tp06_dts,
    0.0015625,
)
print(f"strang, merged rates {np.round(rates, 2)}")

# %% [markdown]
# Merged Strang splitting at $\Delta t = 0.0125$ ms takes about as long as Godunov splitting at
# the same step and as plain Strang splitting at $\Delta t = 0.025$ ms, and is more accurate than
# both (errors of about 0.33 mV against 0.51 and 0.43 mV). At the same $\Delta t$ plain Strang
# splitting is about three times more accurate, at twice the cost. When you drive the time loop
# yourself, call `SplittingSolver.synchronize` before saving or post-processing $V$: between
# merged steps it still lacks its closing half-step.
# Merged Strang's rates (0.85, 1.4, 1.79) approach 2 as well, more slowly than plain Strang's
# (section 6).
