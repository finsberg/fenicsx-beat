# Mathematical background

This page gives a self-contained introduction to the mathematical models
that `fenicsx-beat` solves, and fixes the notation used consistently
throughout the [demos](../demos/index.md). It is a condensed version of
the material in Sundnes, Lines, Cai, Nielsen, Mardal and Tveito,
*Computing the Electrical Activity in the Heart* {cite}`sundnes2007computing`,
which is the standard reference for this field and the source we follow
below. If you want the full derivations, error analysis and numerical
background, that book is the natural next stop.

## 1. From single cells to continuum tissue

The heart muscle is made up of billions of individual, electrically excitable
cells. Each cell is surrounded by a membrane that separates the
**intracellular** space from the **extracellular** space, and actively
maintains a potential difference across it. When a cell is stimulated, ion
channels in the membrane open and close in a coordinated way, producing a
rapid **depolarization** followed by a slower **repolarization** — together
called an *action potential*. Neighbouring cells are electrically coupled
through gap junctions, so an action potential triggered in one place
propagates as a wave through the tissue.

Modelling every cell individually is computationally impossible for anything
larger than a small tissue sample. Instead, we use a *volume-averaging*
argument: at every point in space we define potentials and currents as
averages over a small volume that contains many cells, but is still small
compared to the size of the tissue. This is exactly the same idea used in
continuum mechanics to go from molecules to continuous solids and fluids.
Both the intracellular and extracellular spaces are then treated as
**continuous domains that coexist at every point** in the tissue — this is
the starting point of the *bidomain model*.

## 2. Notation used throughout this documentation

The following symbols are used consistently in this background page, in the
demos, and (as far as possible) map directly onto arguments in the `beat`
API:

| Symbol | Meaning | Corresponds to |
| --- | --- | --- |
| $u_i$, $u_e$ | Intracellular / extracellular potential | — |
| $v = u_i - u_e$ | Transmembrane potential | `pde.state` / `solver.pde.state` |
| $s$ | Vector of gating / concentration state variables | ODE `states` array |
| $M_i$, $M_e$ | Intracellular / extracellular conductivity tensors | `M_i` / `M_e` arguments to `beat.BidomainModel`; `sigma_i` / `sigma_e` arguments to `beat.pseudo_bidomain.VolumeConductor` (`sigma_i` also to `LeadFieldElectrodes`) |
| $M$ | Effective (monodomain) conductivity tensor | `M` argument to `beat.MonodomainModel` |
| $\chi$ | Membrane surface-to-volume ratio | `chi` argument to `beat.stimulation.define_stimulus` / `beat.conductivities.define_conductivity_tensor` |
| $C_m$ | Membrane capacitance (per unit area) | `C_m` argument to `beat.MonodomainModel` |
| $I_{ion}(v, s)$ | Total ionic current density through the membrane | Right-hand side of the cell model (`fun`) |
| $I_{stim}$ | Externally applied stimulus current | `beat.Stimulus` / `I_s` |
| $\theta$ | Splitting / time-stepping parameter, $\theta \in [0, 1]$ | `theta` argument to `beat.SplittingSolver` and `parameters["theta"]` |
| $t_a$ | Local activation time | Output of `beat.eikonal.solve_eikonal`, `t_a` argument to `beat.reaction_eikonal.FootCurrentODESolver` |
| $c_f$, $c_s$, $c_n$ | Conduction velocities along the fiber, sheet and sheet-normal directions | `velocities` argument to `beat.eikonal.EikonalSolver` |
| $\mathcal{V}$ | Velocity tensor of the eikonal equation | Built from `velocities` and `fibers` |
| $\phi_e$ | Extracellular potential in tissue and bath (equals $u_e$ in the tissue) | Output of `beat.pseudo_bidomain.VolumeConductor.solve` |
| $\sigma_b$ | Bath conductivity | `sigma_b` argument to `beat.pseudo_bidomain.VolumeConductor`, `g_b` to `define_augmented_conductivity_tensor` |

$\theta = 1$ gives (first-order) Godunov splitting and a backward Euler PDE
step, while $\theta = 0.5$ gives (second-order) Strang splitting and a
Crank–Nicolson PDE step; see Section 5 below.

## 3. The bidomain model

At each point of the heart we now have two potentials, $u_i$ and $u_e$, and
correspondingly two currents given by Ohm's law,

$$
J_i = -M_i \nabla u_i, \qquad J_e = -M_e \nabla u_e,
$$

where $M_i$ and $M_e$ are the (in general anisotropic) intracellular and
extracellular conductivity tensors. Charge is conserved in each domain up to
the current $I_{ion}$ that crosses the membrane (measured per unit membrane
area, so it is scaled by the surface-to-volume ratio $\chi$ to become a
current *per unit tissue volume*), giving

$$
-\nabla \cdot J_i = \chi C_m \frac{\partial v}{\partial t} + \chi I_{ion}(v, s), \qquad
-\nabla \cdot J_e = -\chi C_m \frac{\partial v}{\partial t} - \chi I_{ion}(v, s),
$$

where $v = u_i - u_e$ is the transmembrane potential and $C_m$ is the
membrane capacitance. Eliminating $u_i = u_e + v$ from these two equations
gives the standard formulation of the **bidomain model**,

$$
\nabla \cdot (M_i \nabla v) + \nabla \cdot (M_i \nabla u_e)
    = \chi C_m \frac{\partial v}{\partial t} + \chi I_{ion}(v, s) - \chi I_{stim},
$$
$$
\nabla \cdot (M_i \nabla v) + \nabla \cdot \big((M_i + M_e) \nabla u_e\big) = 0,
$$

together with an ODE system for the gating/concentration variables,

$$
\frac{\partial s}{\partial t} = f(v, s),
$$

and insulating boundary conditions $n \cdot (M_i \nabla v + M_i \nabla u_e) = 0$
and $n \cdot (M_e \nabla u_e) = 0$ on the heart surface (if the heart is
instead coupled to a surrounding conductive torso, these are replaced by
continuity conditions on the potential and the current; see Chapter 2 of
{cite}`sundnes2007computing`). Note that we have added the stimulus current
$I_{stim}$ explicitly on the right-hand side — this is how an external
pacing or defibrillation current enters the model, and it is treated
identically to $I_{ion}$ but with the opposite sign.

`beat.BidomainModel` implements this
system, solving for $v$ and $u_e$ together. The second equation determines
$u_e$ only up to a constant, so the model normalizes it to zero mean unless a
boundary condition grounds it instead; see the
[bidomain convergence demo](../demos/bidomain_convergence.py) for the
verification of both fields.

## 4. The monodomain model

The bidomain model is a coupled system for two unknown fields, $v$ and
$u_e$, and is comparatively expensive to solve. If we assume that the
intracellular and extracellular conductivity tensors are proportional,
$M_e = \lambda M_i$ for some scalar $\lambda$ (the *equal anisotropy ratio*
assumption), then $u_e$ can be eliminated algebraically and the bidomain
model collapses to a single scalar reaction-diffusion equation for $v$ alone,
the **monodomain model**:

$$
\chi C_m \frac{\partial v}{\partial t}
    = \nabla \cdot (M \nabla v) - \chi I_{ion}(v, s) + \chi I_{stim},
\qquad
M = \frac{\lambda}{1 + \lambda} M_i,
$$

with the ODE system $\partial s/\partial t = f(v, s)$ and the boundary
condition $n \cdot (M \nabla v) = 0$. The equal anisotropy assumption does
not hold exactly in real tissue, so the monodomain model is an
approximation — but it is far cheaper to solve and reproduces propagation
patterns well enough that it is the default choice for most simulation
studies, including all of the demos in this repository.
`beat.MonodomainModel` implements exactly this
equation. Note that the helper functions in
`beat.conductivities` and
`beat.stimulation` already divide the physical
conductivities and stimulus amplitude by $\chi$ before they reach
`MonodomainModel`, so what the assembled weak form actually contains is

$$
C_m \frac{\partial v}{\partial t} = \nabla \cdot (M \nabla v) - I_{ion}(v, s) + I_{stim},
$$

with $M$ and $I_{stim}$ already expressed "per unit membrane area" rather
than "per unit tissue volume".

## 5. Ionic current models

$I_{ion}(v, s)$ describes the current flowing through the ion channels,
pumps and exchangers embedded in the cell membrane, and it is here that most
of the biophysical detail of a specific cell type enters the model. Three
broad families are used in this repository's demos:

- **Phenomenological models**, such as the two-variable
  [FitzHugh–Nagumo model](../demos/fitzhughnagumo.py) used in the "Getting
  started" demo, are not derived from channel physiology but are simple
  dynamical systems tuned to reproduce the qualitative shape of an action
  potential (a fast upstroke, a plateau/recovery phase, and a refractory
  period). They are cheap to evaluate and useful for illustrating numerical
  methods and propagation patterns.
- **Detailed biophysical models** in the tradition of Hodgkin and Huxley's
  1952 model of the squid giant axon describe the current through each ion
  channel as the product of a maximum conductance, a driving force
  $(v - v_{eq})$ relative to the channel's Nernst equilibrium potential, and
  one or more dimensionless *gating variables* $g \in [0, 1]$ (the fraction
  of open channels), each obeying a first-order ODE
  $dg/dt = \alpha(v)(1 - g) - \beta(v) g$. The ten Tusscher–Panfilov and
  ToR-ORd models used in several demos (e.g. the
  [Niederer benchmark](../demos/niederer_benchmark.py), the
  [pacing train](../demos/pace_train.py), [PVC](../demos/pvc.py), and the
  [left/bi-ventricular ellipsoid](../demos/lv_endocardial.py) demos) are modern,
  highly detailed descendants of this formalism, with tens of state
  variables describing individual sodium, potassium and calcium currents,
  calcium handling, and more.
- **Cell-model files** for these detailed models are stored as CellML/`.ode`
  files under [`odes/`](https://github.com/finsberg/fenicsx-beat/tree/main/odes)
  and translated into fast Python right-hand-side functions using
  [`gotranx`](https://finsberg.github.io/gotranx), typically choosing a
  (generalized) Rush–Larsen scheme that treats the gating equations
  semi-analytically for numerical stability.

Regardless of which cell model is used, $I_{ion}$ and $s$ always play the
same role in the coupled system above: $I_{ion}(v, s)$ feeds back into the
PDE for $v$, while $v$ itself feeds into the ODE system
$\partial s/\partial t = f(v, s)$ that advances the gating and concentration
variables.

## 6. Solving the coupled system: operator splitting

The full monodomain problem couples a (typically stiff) nonlinear ODE system
at every point in space with a linear diffusion PDE. Instead of solving this
monolithic system directly, `fenicsx-beat` uses **operator splitting**: at
each time step, the ODE part and the PDE part are solved separately, using
whichever numerical method suits each sub-problem best (an explicit or
Rush–Larsen scheme for the stiff cell-model ODEs, and an implicit finite
element solve for the diffusion PDE).

Writing the monodomain equation as $\partial v/\partial t = (L_1 + L_2) v$
with

$$
L_1 v = -I_{ion}(v, s), \qquad
L_2 v = \frac{1}{C_m}\nabla \cdot (M \nabla v) + \frac{1}{C_m} I_{stim},
$$

one time step $[t_n, t_n + \Delta t]$ of the general $\theta$-splitting
scheme consists of three stages:

1. **ODE step.** Solve $\partial v/\partial t = -I_{ion}(v, s)$ together
   with $\partial s/\partial t = f(v, s)$ from $t_n$ to $t_n + \theta \Delta t$.
2. **PDE step.** Solve the linear diffusion equation
   $C_m \partial v/\partial t = \nabla \cdot (M \nabla v) + I_{stim}$ from
   $t_n$ to $t_n + \Delta t$, using a $\theta$-rule (a weighted average of
   the diffusion operator at the old and new time levels) for the time
   discretization and a standard finite element method in space.
3. **Corrective ODE step.** If $\theta \ne 1$, solve the same ODE system as
   in step 1 again, from $t_n + \theta \Delta t$ to $t_n + \Delta t$, to
   bring $v$ and $s$ to the same point in time.

Choosing $\theta = 1$ collapses step 3 and gives **Godunov splitting**
(first-order accurate in time, and consistent with a first-order backward
Euler PDE step); choosing $\theta = 1/2$ gives **Strang splitting** combined
with a **Crank–Nicolson** PDE step, which is second-order accurate overall.
This is exactly what `beat.SplittingSolver`
implements: its `theta` argument selects between the two, and the `pde` and
`ode` objects it is constructed from correspond to steps 2 and 1/3 above,
respectively. The [convergence](../demos/monodomain_convergence.py) and
[verification](../demos/verification.py) demos check this second-order behaviour
numerically using the method of manufactured solutions.

## 7. Alternative backends for the ODE/PDE steps

`beat.SplittingSolver` only depends on its `pde` and `ode` objects implementing a small
interface (`step`, `to_dolfin`/`from_dolfin`, ...), so either half of the split can be swapped out
independently. Two optional backends are available.

### 7.1 Fully implicit Runge–Kutta stepping (Irksome)

The $\theta$-rule used in Section 6 gives at most second-order accuracy in time (at $\theta = 1/2$),
and the explicit or Rush–Larsen schemes typically used for the ODE step in step 1/3 can struggle with
very stiff cell models. `beat.IrksomeMonodomainModel` and `beat.IrksomeODESolver` are alternative
implementations of the `pde` and `ode` roles in `beat.SplittingSolver`, built on top of the
[Irksome](https://firedrakeproject.org/Irksome) library, that instead advance $v$ (and, for the ODE
step, $s$) with a fully implicit Runge–Kutta method specified by a *Butcher tableau* — for example
`irksome.BackwardEuler()` (first order, matching Godunov splitting) or `irksome.GaussLegendre(2)`
(fourth order). Because they implement the same interface as `MonodomainModel` and
`beat.odesolver.DolfinODESolver`, an `IrksomeMonodomainModel` PDE step can be freely combined with a
plain `DolfinODESolver` ODE step, or vice versa, inside the same splitting solver.
`IrksomeMonodomainModel` can also be used entirely on its own, without any splitting at all, to solve a
pure diffusion-plus-stimulus problem implicitly and with high-order accuracy in time — useful when
there is no reaction term to split against, and hence no splitting error to bound the overall accuracy.
`beat.IrksomeMultiODESolver` is the Irksome counterpart to
`beat.odesolver.DolfinMultiODESolver`, for tissue with a different cell model in different regions (as
in the multi-layer ventricle demos); it shares one Butcher tableau across all regions.

This comes at a cost: each implicit Runge–Kutta stage requires solving a (in general nonlinear) system
over the whole mesh, so it is more expensive per time step than the explicit/$\theta$-rule schemes used
elsewhere in this repository. `irksome` is therefore an optional dependency (the `irksome` extra,
`python3 -m pip install fenicsx-beat[irksome]`), and the
[Irksome demo](../demos/irksome_model_gotranx.py) shows it in combination with a
`gotranx`-generated cell model.

### 7.2 Cell-model evaluation via `dolfinx-external-operator`

`beat.ExternalOperatorODESolver` and `beat.ExternalOperatorMultiODESolver` are drop-in alternatives to
`beat.odesolver.DolfinODESolver`/`DolfinMultiODESolver` built on
[`dolfinx-external-operator`](https://github.com/a-latyshev/dolfinx-external-operator)'s
`FEMExternalOperator`, which lets an ODE right-hand side be evaluated as a plain NumPy or Numba array
operation at every mesh point and spliced back into the dolfinx `Function` machinery, instead of
`DolfinODESolver`'s own hand-rolled loop. Unlike the Irksome classes above, these keep the *same*
`fun(states, t, parameters, dt)` cell-model convention already used throughout this repository
unchanged — the same `gotranx`-generated or hand-written functions that work with
`DolfinODESolver` work here too, with no separate UFL- or Irksome-specific variant needed. The
numerical scheme (e.g. forward Euler, Rush–Larsen) and its order of accuracy are unchanged; what
changes is only *how* it is evaluated and wired into the splitting solver, which mainly matters if
you want to batch the cell-model evaluation on a backend `dolfinx-external-operator` supports (Numba,
JAX, PyTorch) rather than beat's own optional Numba path in `single_cell.py`/`odesolver.py`.
`dolfinx-external-operator` is an optional dependency (the `external_operator` extra,
`python3 -m pip install fenicsx-beat[external_operator]`), and the
[external-operator demo](../demos/external_operator_gotranx.py) cross-checks it against
`DolfinODESolver` on a real `gotranx`-generated cell model and reports the current performance
trade-off honestly.

### 7.3 Removing the splitting error entirely: monolithic implicit coupling

Sections 7.1 and 7.2 both still fit inside the operator-splitting scheme of Section 6 — they change
*how* the PDE or ODE half-step is computed, not the fact that there are two half-steps with a splitting
error between them. Because `FEMExternalOperator` can supply a *derivative* as well as a value, it can
instead be used to solve the PDE and the cell-model ODEs **together**, in one fully implicit Newton
system per time step, with $v$ and every gating/concentration variable as unknowns simultaneously —
removing the operator-splitting error altogether (only the usual temporal and spatial discretization
error remains). This needs a cell model that can supply its own Jacobian; `gotranx`'s JAX code
generation backend combined with `jax.jacfwd`/`jax.vmap` gives this for free, with no hand-derived
derivatives, even through the `Conditional`-based branching common in cell models.

This is not currently packaged as a reusable `beat` class the way the classes above are —
`FEMExternalOperator` does not (yet) support operands built from a mixed-space `Function`, so a
monolithic coupling has to be built from one separate scalar `Function` per state plus an
$N \times N$ block Jacobian, which does not scale to large state counts as cleanly as the flat
`(num_states, num_points)` arrays used everywhere else in this repository. The
[monolithic coupling demo](../demos/monolithic_external_operator.py) is a validated prototype of the
pattern (checked against an independent, non-FEM reference) together with a quantitative comparison
against `SplittingSolver`, showing a several-times reduction in error at practical time
steps — worth knowing the pattern exists even though it is not (yet) a drop-in solver class.

## 8. The reaction-eikonal model

Everything above resolves the depolarization wavefront in space. In human ventricular tissue the
upstroke front is well under a millimeter wide, so a reaction-diffusion (R-D) model needs meshes of
roughly $100$–$250\ \mu$m and time steps of a few hundredths of a millisecond before conduction
velocity stops depending on the discretization; on coarser meshes the wave travels too slowly or
fails to propagate at all. The *reaction-eikonal* (R-E) model of Neic et al.
{cite}`neic2017efficient` avoids this by computing *when* each point activates from a cheap
geometric problem, the eikonal equation, and only then letting the cell model produce the action
potential. The front no longer has to be resolved, so the same model runs on meshes several times
coarser while keeping the activation sequence of a fine R-D simulation.

### 8.1 Activation times from the eikonal equation

Let $t_a(x)$ be the time at which the wavefront reaches the point $x$. If the front moves with
conduction velocity $c_f$ along the fibers, $c_s$ along the sheets and $c_n$ normal to them, $t_a$
satisfies the anisotropic **eikonal equation**

$$
\sqrt{\nabla t_a \cdot \mathcal{V}\, \nabla t_a} = 1,
\qquad
\mathcal{V} = c_f^2\, f f^\top + c_s^2\, s s^\top + c_n^2\, n n^\top,
$$

with $t_a = t_0$ at each stimulus site that fires at time $t_0$. Its solution is the earliest
arrival time over all paths from the stimulus sites, where a path's travel time is measured in the
metric $\mathcal{V}^{-1}$; where fronts from different sites meet they simply collide. The
velocities are inputs here, not outputs: in a monodomain model they scale roughly like
$\sqrt{M_\zeta / (\chi C_m)}$ in each direction $\zeta$, and in practice one measures them once from
a fine R-D simulation of the same tissue (the
[wedge activation demo](../demos/reaction_eikonal_wedge_activation.py) does exactly this).

`beat.eikonal.EikonalSolver` (or the one-shot `beat.eikonal.solve_eikonal`) computes $t_a$ at the
mesh vertices with the **Fast Iterative Method** (FIM) of Jeong and Whitaker
{cite}`jeong2008fast` in the form of Fu, Kirby and Whitaker {cite}`fu2013fast`. FIM keeps a list of
active vertices and repeatedly replaces each one's value by the smallest arrival time it can get
through any of its cells, given the current values at the cell's other vertices; a vertex leaves
the list once its value stops changing, and wakes its neighbours. Quadrilateral and hexahedral cells
are split into triangles and tetrahedra for this, and higher-order geometries use their vertices
only, so $t_a$ is always a piecewise linear ($P_1$) function. The kernel is compiled C (optionally
with OpenMP) and runs in parallel over MPI by exchanging the smallest value at shared vertices
between ranks; the [eikonal demo](../demos/eikonal.py) checks it against exact solutions.

### 8.2 Coupling activation back to the cell model: R-E$^+$ and R-E$^-$

The activation map is turned back into action potentials by a **foot current** that kicks each
point's cell model across its excitation threshold at $t = t_a$. In the notation of this page it is
a source of transmembrane potential,

$$
\left.\frac{\partial v}{\partial t}\right|_{foot}
    = \frac{A}{\tau_F}\, e^{(t - t_a)/\tau_F}
    \quad \text{for } 0 \le t - t_a \le T_{foot} \text{ and } v < V_{th},
$$

and zero otherwise, with $A = 0.91$ mV, $\tau_F = 0.25$ ms, $T_{foot} = 5$ ms and
$V_{th} = -30$ mV by default (`beat.reaction_eikonal.ExponentialFoot`). The exponential mimics the
rise of $v$ ahead of a real front; the threshold switches it off once the cell's own sodium current
has taken over. The time window follows the openCARP implementation, which starts the foot at
$t_a$; Eq. 27 of {cite}`neic2017efficient` states the window $[t_a - T_{foot}, t_a]$ instead,
which contradicts the paper's own appendix.

The foot current enters the model in one of two ways:

- **R-E$^+$** keeps diffusion. Each step solves the monodomain model of Section 4, without
  $I_{stim}$, with the foot current added to the reaction part:

  $$
  C_m \frac{\partial v}{\partial t}
      = \nabla \cdot (M \nabla v) - I_{ion}(v, s) + C_m \left.\frac{\partial v}{\partial t}\right|_{foot}.
  $$

  The foot guarantees activation no later than $t_a$, but diffusion can activate a point earlier.
  In `beat` this is `beat.SplittingSolver(pde, ode)` with an unstimulated
  `MonodomainModel` and the ODE solver wrapped in `beat.reaction_eikonal.FootCurrentODESolver`, so
  the operator splitting of Section 6 is unchanged.
- **R-E$^-$** drops diffusion entirely, $C_m\, \partial v/\partial t = -I_{ion}(v, s) +
  C_m\, \partial v/\partial t|_{foot}$, so every point evolves independently and no linear system is
  solved at all. This is `beat.SplittingSolver(pde=None, ode=FootCurrentODESolver(...))`: the same driver,
  without a PDE.

Both reproduce a fine R-D activation sequence on coarse meshes. They differ during repolarization:
diffusion smooths the boundaries between regions with different action potential durations, and
without it R-E$^-$ produces repolarization gradients that are far too steep, which shows up in the
T-wave of an ECG. R-E$^+$ keeps that smoothing at a modest extra cost. The
[wedge activation](../demos/reaction_eikonal_wedge_activation.py) and
[wedge repolarization](../demos/reaction_eikonal_wedge_repolarization.py) demos reproduce both
effects from the benchmark of {cite}`neic2017efficient`.

### 8.3 Measuring activation in a simulation

To compare an R-D or R-E simulation with $t_a$, the activation time has to be extracted from $v$.
`beat.activation.ActivationTracker` records it at every degree of freedom while the simulation runs,
either as the time of the steepest upstroke (maximum $\partial v/\partial t$, the usual definition
and the one used by {cite}`neic2017efficient`) or as the first upward crossing of a threshold
potential. Points that never activate keep the value NaN.

### 8.4 Bath loading and the pseudo-bidomain model

Tissue in contact with a conductive medium (the blood pool, the torso, or the bath of an
experimental preparation) conducts faster close to the surface: extracellular current can return
through the bath, which conducts much better than the interstitial space, so in a bidomain model
the wavefront runs ahead along the surface. This is called *bath loading*. A monodomain model has
no bath, but Bishop and Plank {cite}`bishop2011representing` showed that it reproduces the effect
if the conductivity is *augmented* in a thin layer of tissue along the interface. In each
eigendirection $\zeta$ of the conductivity tensors the monodomain conductivity becomes

$$
m_\zeta = \frac{g_{i\zeta}\, g_{e\zeta}}{g_{i\zeta} + g_{e\zeta}} \quad \text{in the bulk},
\qquad
m_\zeta = \frac{g_{i\zeta}\, \sigma_b}{g_{i\zeta} + \sigma_b} \quad \text{in the layer},
$$

where $M_i = \sum_\zeta g_{i\zeta}\,\zeta\zeta^\top$ and $M_e = \sum_\zeta g_{e\zeta}\,\zeta\zeta^\top$ before the division by $\chi$, and $M = \frac{1}{\chi}\sum_\zeta m_\zeta\,\zeta\zeta^\top$; that is, the harmonic mean of Section 4 with the bath conductivity $\sigma_b$ in place of the
extracellular one near the surface (Eqs. 14 and 15 of {cite}`neic2017efficient`).
`beat.pseudo_bidomain.interface_layer` marks the layer (three element rings by default, as in
{cite}`bishop2011representing`) and `beat.pseudo_bidomain.define_augmented_conductivity_tensor`
builds $M$ from it. The result goes straight into `beat.MonodomainModel`, so R-D and R-E$^+$ run
unchanged (R-E$^-$ has no diffusion and is unaffected).

The **pseudo-bidomain** model then recovers the extracellular potential. Given $v$ from the
augmented monodomain model, $\phi_e$ on the tissue $\Omega_T$ and the bath $\Omega_B$ solves the
elliptic part of the bidomain model of Section 3, extended to the bath:

$$
\nabla \cdot \big((M_i + M_e) \nabla \phi_e\big) = -\nabla \cdot (M_i \nabla v) \ \text{in } \Omega_T,
\qquad
\nabla \cdot (\sigma_b \nabla \phi_e) = 0 \ \text{in } \Omega_B,
$$

with $\phi_e$ and the normal current continuous across the interface, no current through the
outer boundary, and $\phi_e$ normalized to zero mean. In the tissue $\phi_e$ plays the role of
$u_e$. Because $v$ is known this is a single linear solve, needed only at the times $\phi_e$ is
wanted, and its cost is independent of the cell model. `beat.pseudo_bidomain.VolumeConductor`
assembles it once on a mesh of tissue and bath and returns $\phi_e$ for any $v$; the tissue mesh
of the simulation can be separate, with $v$ interpolated onto the tissue cells.

Recording $\phi_e$ at a few electrodes in every time step does not even need that solve. For an
electrode at $P$, the **lead field** $Z_P$ solves the same elliptic problem with a unit point
source at $P$ (and a uniform sink, so the pure-Neumann problem has a solution); taking $Z_P$ as
the test function in the problem for $\phi_e$, and $\phi_e$ as the test function in the problem
for $Z_P$, gives

$$
\phi_e(P) = -\int_{\Omega_T} M_i \nabla v \cdot \nabla Z_P \, dx,
$$

which also holds exactly for the discretized problems. `beat.pseudo_bidomain.LeadFieldElectrodes`
solves one lead field per electrode up front, after which an electrogram sample costs one
integral over the tissue. The [wedge electrogram demo](../demos/reaction_eikonal_wedge_electrograms.py)
uses both to reproduce Fig. 7 of {cite}`neic2017efficient`.

## 9. Where to go from here

- The [demos index](../demos/index.md) lists every demo in this repository, grouped by topic.
- The [FitzHugh–Nagumo demo](../demos/fitzhughnagumo.py) is the best starting
  point: it solves the monodomain model with the simplest possible ionic
  current model, both as a single-cell ODE and as a full 2D tissue
  simulation, and introduces the `beat` API step by step.
- The [diffusion demo](../demos/diffusion.py) shows the monodomain model with the
  ionic current turned off entirely, i.e. pure diffusion driven by a
  stimulus.
- The [convergence](../demos/monodomain_convergence.py) and
  [verification](../demos/verification.py) demos connect the numerical scheme
  described in Section 6 back to measured convergence rates.
- The remaining demos build up towards realistic simulations: conduction
  velocity and ECG estimation on a simple slab, pacing trains and premature
  ventricular complexes on a 1D cable, and full biophysically detailed
  simulations on left- and bi-ventricular geometries, including 12-lead ECG
  recovery.
- The [Irksome demo](../demos/irksome_model_gotranx.py) shows the fully implicit, high-order time
  stepping described in Section 7.1, the
  [external-operator demo](../demos/external_operator_gotranx.py) shows the drop-in ODE-step backend
  described in Section 7.2, and the
  [monolithic coupling demo](../demos/monolithic_external_operator.py) shows the fully implicit,
  no-splitting-error prototype described in Section 7.3.
- The [eikonal demo](../demos/eikonal.py) introduces the activation-time solver of Section 8.1, and
  the [wedge activation](../demos/reaction_eikonal_wedge_activation.py) and
  [wedge repolarization](../demos/reaction_eikonal_wedge_repolarization.py) demos compare R-D,
  R-E$^+$ and R-E$^-$ as described in Section 8.2, and the
  [wedge electrogram demo](../demos/reaction_eikonal_wedge_electrograms.py) computes extracellular
  potentials and electrograms with the pseudo-bidomain model of Section 8.4.
- For the full derivation of the bidomain and monodomain models, the
  physiology of the cell membrane, the Hodgkin–Huxley and Nernst–Planck
  formalisms, and a rigorous treatment of operator splitting and its order
  of accuracy, see Chapters 1–3 of {cite}`sundnes2007computing`.

```{bibliography}
:filter: docname in docnames
```
