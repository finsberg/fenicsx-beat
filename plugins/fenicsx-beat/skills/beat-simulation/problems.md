# Problem catalogue

Match the user's goal to a starting point. **CLI** = a template exists (`beat init --template <name>`)
or the CLI schema can express the problem; **Python** = start from the demo. Demo links:
`https://finsberg.github.io/fenicsx-beat/demos/<name>.html` (explained) and
`https://github.com/finsberg/fenicsx-beat/blob/main/demos/<name>.py` (source). Read a demo's source
before adapting it; templates carry their own comments (including where they deviate from the demo).

## Starting points

| Problem | Route | Template | Demo | Key `beat` objects |
|---|---|---|---|---|
| First tissue simulation, learning the API | CLI or Python | `fitzhughnagumo` | [fitzhughnagumo](https://finsberg.github.io/fenicsx-beat/demos/fitzhughnagumo.html) | `beat.MonodomainModel`, `beat.odesolver.DolfinODESolver`, `beat.SplittingSolver` |
| Pure diffusion of a stimulus (no cell model) | CLI or Python | `diffusion` | [diffusion](https://finsberg.github.io/fenicsx-beat/demos/diffusion.html) | `beat.MonodomainModel` |
| Conduction velocity in a slab; pseudo-ECG | CLI or Python | `slab` | [slab](https://finsberg.github.io/fenicsx-beat/demos/slab.html) | `beat.conduction_velocity.measure_conduction_velocity`, `beat.conduction_velocity.calibrate_conductivities`, `beat.ECGRecovery` |
| Cross-code benchmark (Niederer et al. 2011), mesh/dt convergence of activation times | CLI or Python | `niederer_benchmark` | [niederer_benchmark](https://finsberg.github.io/fenicsx-beat/demos/niederer_benchmark.html) | `beat.activation.ActivationTracker` |
| Ectopic beats (PVCs) on a 1D cable | CLI (homogeneous cable) or Python (heterogeneous) | `pvc` | [pvc](https://finsberg.github.io/fenicsx-beat/demos/pvc.html) | `beat.odesolver.DolfinODESolver` |
| Rapid pacing train on a cable | CLI or Python | `pace_train` | [pace_train](https://finsberg.github.io/fenicsx-beat/demos/pace_train.html) | stimulus `period` / `num_pulses` |
| Idealized LV ellipsoid, endocardial stimulus, endo/mid/epi cells | CLI or Python | `lv_endocardial` | [lv_endocardial](https://finsberg.github.io/fenicsx-beat/demos/lv_endocardial.html) | `beat.utils.expand_layer`, `beat.odesolver.DolfinMultiODESolver` |
| Realistic bi-ventricular heart (UK Biobank atlas), Purkinje-like random endocardial stimuli | CLI or Python | `ukb_atlas` | [ukb_atlas](https://finsberg.github.io/fenicsx-beat/demos/ukb_atlas.html) | `beat.utils.expand_layer_biv`, `beat.stimulation.generate_random_activation` |
| Durrer-like endocardial activation sequence on a BiV ellipsoid; 12-lead ECG | Python | — | [durrer_activation](https://finsberg.github.io/fenicsx-beat/demos/durrer_activation.html); full pipeline (1 mm mesh, endo/mid/epi ToR-ORd paced to steady state, 12-lead ECG, MPI): [`scripts/durrer_activation_full.py`](https://github.com/finsberg/fenicsx-beat/blob/main/scripts/durrer_activation_full.py) | `beat.activation_sequence.EndocardialActivation`, `beat.ecg.ElectrodePotentials`, `beat.ecg.twelve_lead` |
| 12-lead ECG from a CLI run | CLI | any | — | `[postprocess.ecg]` with `postprocess.ecg.leads = "twelve-lead"` ([outputs.md](outputs.md)) |
| Spiral wave (S1–S2 cross-field), tip tracking | Python (the CLI can do S1–S2 with two `box` stimuli, but has no tip tracking) | — | [spiral_wave](https://finsberg.github.io/fenicsx-beat/demos/spiral_wave.html) | `beat.SplittingSolver`; tip tracking is a function in the demo (not `beat` API); the S2 timing is tuned to the demo's mesh, dt and TP06 |
| Spiral wave breakup (steep restitution) | Python | — | [spiral_wave_breakup](https://finsberg.github.io/fenicsx-beat/demos/spiral_wave_breakup.html) | — |
| Activation times only, fast (any mesh) | Python | — | [eikonal](https://finsberg.github.io/fenicsx-beat/demos/eikonal.html) | `beat.eikonal.solve_eikonal`, `beat.eikonal.EikonalSolver` |
| Many stimulation sites / parameter sweeps of activation | Python | — | [eikonal](https://finsberg.github.io/fenicsx-beat/demos/eikonal.html) (wedge section) | build one `beat.eikonal.EikonalSolver` and call `solve` per site with `beat.eikonal.Source` (a function of $x$ returning a boolean mask, or `(meshtags, value)`, plus a start time `t0`); velocities are in mesh units per ms along/across fibres; `openmp=True` or MPI |
| Activation, repolarization or electrograms on coarse meshes (reaction-eikonal, Neic et al. 2017) | Python | — | [reaction_eikonal_wedge_activation](https://finsberg.github.io/fenicsx-beat/demos/reaction_eikonal_wedge_activation.html), [reaction_eikonal_wedge_repolarization](https://finsberg.github.io/fenicsx-beat/demos/reaction_eikonal_wedge_repolarization.html), [reaction_eikonal_wedge_electrograms](https://finsberg.github.io/fenicsx-beat/demos/reaction_eikonal_wedge_electrograms.html) | `beat.reaction_eikonal.FootCurrentODESolver`, `beat.pseudo_bidomain.VolumeConductor`, `beat.pseudo_bidomain.LeadFieldElectrodes` |
| Bidomain (extracellular potential, unequal anisotropy) | Python | — | [bidomain_convergence](https://finsberg.github.io/fenicsx-beat/demos/bidomain_convergence.html) | `beat.BidomainModel` |
| Temporal / spatial convergence, verification | Python | — | [verification](https://finsberg.github.io/fenicsx-beat/demos/verification.html), [monodomain_convergence](https://finsberg.github.io/fenicsx-beat/demos/monodomain_convergence.html) | `beat.SplittingSolver` |
| Fully implicit high-order Runge–Kutta (Irksome) | CLI or Python | `irksome_model_gotranx` | [irksome_model_gotranx](https://finsberg.github.io/fenicsx-beat/demos/irksome_model_gotranx.html) | `beat.IrksomeMonodomainModel`, `beat.IrksomeODESolver` |
| ODE step via dolfinx-external-operator | CLI or Python | `external_operator_gotranx` | [external_operator_gotranx](https://finsberg.github.io/fenicsx-beat/demos/external_operator_gotranx.html) | `beat.ExternalOperatorODESolver` |
| Faster cell-model step (compiled C, optional OpenMP) | Python | — | [c_backend](https://finsberg.github.io/fenicsx-beat/demos/c_backend.html) | `beat.c_backend.from_ode` |
| Single cell only (pacing to steady state before tissue runs) | Python (CLI: `cell.steady_state`) | — | [pace_train](https://finsberg.github.io/fenicsx-beat/demos/pace_train.html), [slab](https://finsberg.github.io/fenicsx-beat/demos/slab.html) | `beat.single_cell.get_steady_state` |
| Many independent ODEs at once (toy model, the ODE step alone) | Python | — | [simple_ode](https://finsberg.github.io/fenicsx-beat/demos/simple_ode.html) | `beat.odesolver.DolfinODESolver` |

All ten templates: `slab`, `niederer_benchmark`, `fitzhughnagumo`, `diffusion`, `pvc`, `pace_train`,
`lv_endocardial`, `ukb_atlas`, `irksome_model_gotranx`, `external_operator_gotranx`. Without
`--template`, `beat init` writes `slab`.

## Choosing between the models

| Model | Gives | Cost | Use when |
|---|---|---|---|
| Monodomain (default) | $v$ everywhere: upstroke, repolarization, re-entry | resolve the upstroke: mesh ~0.1–0.25 mm for physiological models | the reference; arrhythmia, repolarization, anything with wave interaction |
| Reaction-eikonal R-E+ | activation from an eikonal solve, $v$ from the cell model plus diffusion | mesh can be ~1 mm | activation and ECG on whole hearts at modest cost |
| Reaction-eikonal R-E− | as R-E+ without the PDE | cheapest with cell models | activation-driven $v$ without electrotonic coupling |
| Eikonal only | activation times | seconds | activation maps, parameter sweeps, initial guesses |
| Bidomain | $v$ and the extracellular potential | about twice monodomain or more | extracellular potential itself, unequal anisotropy ratios, bath effects |

R-E+ is `beat.SplittingSolver(pde, wrapped_ode)` with the ODE solver wrapped in
`beat.reaction_eikonal.FootCurrentODESolver`; R-E− is `beat.SplittingSolver(None, wrapped_ode)`.
See the reaction-eikonal demos for the setup.

## Cell models

- Shipped with the templates as `.ode` files: FitzHugh–Nagumo, Mitchell–Schaeffer, ten Tusscher–
  Panfilov 2006 epicardial, ToR-ORd endocardial (`ToRORd_dynCl_endo.ode`; its `celltype` parameter
  selects endo 0, epi 1, mid 2).
- Own model: a gotranx `.ode` file goes straight into `cell.ode_file`; convert CellML first with
  `gotranx cellml2ode model.cellml`. Set `cell.v_name` to the membrane-potential state and zero the
  model's own stimulus amplitude in `cell.parameters` when the PDE stimulates.
- CellML sources: the Physiome Model Repository (https://models.physiomeproject.org).
