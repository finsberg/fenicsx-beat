# Demos

This page indexes all the demos in `fenicsx-beat`, grouped the same way as in the sidebar. If you are
new to the library, start with the [mathematical background](../docs/math_background.md) page for the
model and notation used throughout, then work through the
[FitzHugh–Nagumo demo](fitzhughnagumo.py) below.

## Getting started

- [A simple example of excitable tissue: The FitzHugh–Nagumo model](fitzhughnagumo.py) — the best
  starting point. Solves the monodomain model with the simplest possible ionic current model, both as
  a single-cell ODE and as a full 2D tissue simulation, introducing the `beat` API step by step.

## Examples

- [Diffusion in a square domain with a stimulus in the lower left corner](diffusion.py) — the
  monodomain model with the ionic current switched off entirely, i.e. pure diffusion driven by a
  stimulus current.
- [Solving a simple ODE](simple_ode.py) — solving a toy ODE system with `beat`'s ODE solver at many
  points at once, the same kind of step used for the cell model in the full monodomain solver.
- [Endocardial stimulation of a left ventricle ellipsoid](lv_endocardial.py) — a full 3D simulation on
  an idealized left-ventricle geometry, with fibre directions, transmurally varying cell models
  (endocardial/mid-myocardial/epicardial), and endocardial stimulation.
- [Conduction velocity and ECG for slabs](slab.py) — conduction velocity and pseudo-ECG estimation on a
  simple slab of tissue.
- [Niederer benchmark](niederer_benchmark.py) — the standard cross-code monodomain benchmark
  {cite}`land2015verification`, comparing activation times across spatial and temporal resolutions.
- [Premature Ventricular Complexes (PVCs)](pvc.py) — reproducing an ectopic beat originating from a
  region of reduced repolarization reserve on a 1D cable {cite}`zhang2021mechanisms`.
- [Pacing train](pace_train.py) — the same 1D cable as the PVC demo, rapidly paced from one end.
- [Endocardial stimulation of a Bi-ventricular ellipsoid](biv_endocardial.py) — like the left-ventricle
  demo, but on a bi-ventricular geometry, including 12-lead ECG recovery.
- [Purkinje like stimulation of a realistic BiV geometry](ukb_atlas.py) — a realistic bi-ventricular
  geometry from a UK Biobank atlas, stimulated at many random endocardial points to mimic activation
  via the Purkinje network.

## Verification

- [Verifying Second-Order Temporal Convergence](verification.py) — checks the second-order-in-time
  convergence of Strang splitting combined with a Crank–Nicolson PDE step, using the method of
  manufactured solutions.
- [Monodomain convergence test](monodomain_convergence.py) — the companion check of the (first-order)
  spatial and Godunov-splitting convergence rates.

```{bibliography}
:filter: docname in docnames
```
