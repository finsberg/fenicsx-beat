# Mathematical background

This page gives a self-contained introduction to the mathematical models
that `fenicsx-beat` solves, and fixes the notation used consistently
throughout the [demos](../demos/fitzhughnagumo.py). It is a condensed version of
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
| $M_i$, $M_e$ | Intracellular / extracellular conductivity tensors | — |
| $M$ | Effective (monodomain) conductivity tensor | `M` argument to `beat.MonodomainModel` |
| $\chi$ | Membrane surface-to-volume ratio | `chi` argument to `beat.stimulation.define_stimulus` / `beat.conductivities.define_conductivity_tensor` |
| $C_m$ | Membrane capacitance (per unit area) | `C_m` argument to `beat.MonodomainModel` |
| $I_{ion}(v, s)$ | Total ionic current density through the membrane | Right-hand side of the cell model (`fun`) |
| $I_{stim}$ | Externally applied stimulus current | `beat.Stimulus` / `I_s` |
| $\theta$ | Splitting / time-stepping parameter, $\theta \in [0, 1]$ | `theta` argument to `beat.MonodomainSplittingSolver` and `parameters["theta"]` |

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
This is exactly what `beat.MonodomainSplittingSolver`
implements: its `theta` argument selects between the two, and the `pde` and
`ode` objects it is constructed from correspond to steps 2 and 1/3 above,
respectively. The [convergence](../demos/monodomain_convergence.py) and
[verification](../demos/verification.py) demos check this second-order behaviour
numerically using the method of manufactured solutions.

## 7. Where to go from here

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
- For the full derivation of the bidomain and monodomain models, the
  physiology of the cell membrane, the Hodgkin–Huxley and Nernst–Planck
  formalisms, and a rigorous treatment of operator splitting and its order
  of accuracy, see Chapters 1–3 of {cite}`sundnes2007computing`.

```{bibliography}
:filter: docname in docnames
```
