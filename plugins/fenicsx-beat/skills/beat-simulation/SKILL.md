---
name: beat-simulation
description: Set up, choose and run cardiac electrophysiology simulations with fenicsx-beat (the beat package on FEniCSx/dolfinx) — monodomain, reaction-eikonal and eikonal models, cell models (gotranx/CellML), slabs, ventricles and own meshes, activation maps, ECGs, spiral waves. Use when the user wants to simulate heart electrophysiology, write or fix a beat CLI config or beat Python script, pick a solver, cell scheme, time step or mesh size, or post-process beat results.
---

# Setting up a beat simulation

`beat` (package `fenicsx-beat`) solves the monodomain model — the reaction-diffusion PDE
$\chi C_m \partial v/\partial t = \nabla\cdot(M\nabla v) - \chi I_{ion}(v, s) + \chi I_{stim}$
coupled to cell-model ODEs for the states $s$ — by operator splitting, on FEniCSx/dolfinx meshes. It also has an eikonal
solver, the reaction-eikonal model, a bidomain model and ECG recovery. Two ways to drive it:

- **CLI** (`beat init / validate-config / geometry / run / post`): a TOML config, reproducible,
  restartable, MPI-ready; `beat post` recomputes outputs from saved results.
- **Python API**: everything the CLI cannot express (custom ODE code, the Durrer activation
  sequence, reaction-eikonal, bath/pseudo-bidomain, custom pipelines).

Your job: find out what the user really wants to simulate, agree on an approach with them, build
it, and check it before they spend hours on a run. Work through the five steps below in order. Do
not skip the interview because the request looks clear — the choices below change the result.

Reference files (read when the step needs them):

- [install.md](install.md) — getting dolfinx, beat and the extras
- [problems.md](problems.md) — problem catalogue: which template, demo and classes fit
- [numerics.md](numerics.md) — cell scheme, splitting, time step, mesh size, backends, MPI
- [geometry.md](geometry.md) — built-in geometries, the user's own mesh, markers and fibres
- [outputs.md](outputs.md) — what to save, how often, and recomputing outputs afterwards

## Step 1: Check the environment

Run (and read the output):

```bash
python -c "import dolfinx, beat; print('dolfinx', dolfinx.__version__, 'beat', beat.__version__)"
beat version   # only if the CLI route is likely; needs the `cli` extra
```

- dolfinx missing → it is not pip-installable on its own. Present the options in
  [install.md](install.md) with your recommendation (conda on a laptop/workstation, Spack on an HPC
  cluster, Docker for a quick isolated try, Debian/Ubuntu packages if they are on Ubuntu and want
  system packages) and let the user choose.
- **Version.** This skill describes beat 0.14 (defaults `rush_larsen_2` + Strang, merged half-steps
  in the CLI). An editable install can report a stale `beat.__version__`, so check features, not only
  the number: `python -c "import beat, inspect; print(inspect.signature(beat.SplittingSolver))"` must
  show `splitting` and `merge_half_steps`. If the installed beat is older, say so: recommend
  upgrading, or follow that version's docs and `help()` instead of the defaults described here.
- Later, once the plan is clear, check the extras it needs (`gotranx` for cell models,
  `cardiac_geometries` and `ldrb` for ellipsoids and fibres, `ukb` for the UK Biobank atlas,
  `pyvista` for images/movies, `io4dolfinx` for reading meshes and results).
- **Never install anything without asking.** Show the command and let the user run it or approve it.

## Step 2: Interview the user

Ask **one question at a time**, multiple choice with an "other" option, and say why it matters when
that is not obvious. Skip a topic the user already answered; confirm instead of re-asking. Skip a
topic that does not apply to where the conversation is heading (no cell-model question for
activation times from the eikonal model alone), and say why. If the user asked for the CLI but wants
something only Python can do (Durrer sequence, eikonal, reaction-eikonal, bath, custom ODE code,
spiral-tip tracking), say so as soon as it comes up, not only at Step 3. Topics, in order:

1. **Goal** — what question should the simulation answer? Conduction velocity; activation map;
   ECG (pseudo-ECG or 12-lead); arrhythmia / re-entry (spiral waves, S1–S2, breakup); effect of a
   drug or parameter change (channel block, cell type); numerics / verification / benchmark;
   learning the tool. The goal drives every later choice — get it concrete.
2. **Geometry** — 1D cable; 2D sheet; 3D slab; idealized LV or BiV ellipsoid; UK Biobank atlas
   heart; their own mesh. Own mesh → follow [geometry.md](geometry.md) *now* (check it can be read,
   which markers and fibres it has) before going on. Ask for the length unit.
3. **Cell model** — FitzHugh–Nagumo or Mitchell–Schaeffer (cheap, qualitative); ten Tusscher–
   Panfilov 2006 (TP06); ToR-ORd (human ventricle, endo/mid/epi variants); their own CellML or `.ode`
   file (converted with gotranx). Ask whether cells should be paced to steady state first and
   whether different regions (endo/mid/epi layers) need different cell types or parameters.
4. **Stimulus** — a marked region (e.g. the endocardium); a box; a point / corner; an S1–S2 or
   pacing train (period, number of pulses); many random Purkinje-like endocardial points; the
   Durrer-like endocardial activation sequence (`beat.activation_sequence.EndocardialActivation`,
   Python only). Stimulate in the PDE *or* the cell model, not both.
5. **Outputs and post-processing** — what to look at (movie of $v$, activation-time map, traces
   or APD at probe points, pseudo-ECG or 12-lead ECG with electrode positions, electrograms); how
   often to save; whether outputs must be recomputed later from the saved fields. Use [outputs.md](outputs.md) to
   explain the save-interval trade-off and recommend saving the raw fields once and recomputing
   movies/ECGs from them.
6. **Resources and accuracy** — laptop or cluster (how many MPI ranks), how much simulated time
   (one beat, many beats), quick-and-rough or converged (mesh size and time step,
   [numerics.md](numerics.md)).

## Step 3: Propose alternatives

Present **2–3 approaches with trade-offs and a recommendation**, using [problems.md](problems.md)
and [numerics.md](numerics.md). Typical axes:

- **Model**: full monodomain (reference; resolves upstroke, repolarization, re-entry) vs
  reaction-eikonal (activation and ECG on coarse meshes at a fraction of the cost; R-E+ keeps
  diffusion, R-E− has no PDE) vs eikonal only (activation times only, seconds) vs bidomain (when the
  extracellular potential itself, or a bath, matters).
- **Interface**: CLI config vs Python script — Python is required for the Durrer sequence,
  reaction-eikonal, eikonal, bath/pseudo-bidomain, custom ODE code or custom analysis.
- **Numerics**: the defaults (`rush_larsen_2`, Strang splitting, merged half-steps in the CLI) vs
  a more robust or cheaper setting; time step and mesh size against accuracy and cost.

Then **summarize the chosen setup** in a few lines (geometry, cell model, stimulus, outputs, model,
interface, dt, mesh size, end time, ranks) and get an explicit yes before building.

## Step 4: Build

**CLI route** (in a fresh folder the user agrees on):

```bash
beat init --template <closest> config.toml   # templates listed in problems.md
# edit config.toml for the agreed setup
beat validate-config config.toml             # prints the resolved config; fix every error
```

Show the user the resolved values that matter (geometry, `cell.scheme`, `solver.dt`,
`solver.end_time`, stimulus, `output.save_every`). Config keys and allowed values: the comments in
the template, `beat validate-config` errors, and the configuration reference
(https://finsberg.github.io/fenicsx-beat/docs/cli_reference.html). Quantities are pint strings
(`"0.05 ms"`, `"1400 cm**-1"`).

**Python route**: start from the closest demo or script in [problems.md](problems.md) (docs page
for the explanation, GitHub source for the code — demos and `scripts/` are not in the pip package).
Before using a `beat` call, read its signature in the installed version (`python -c "import beat,
inspect; print(inspect.signature(beat.SplittingSolver))"`, `help(...)`) — the API changes between
versions. When adapting a demo:

- it reads files relative to the repository (`../odes/...` cell models): download them from
  https://github.com/finsberg/fenicsx-beat/tree/main/odes or point to the user's own file, and use
  absolute paths;
- it writes generated code and results into the current directory — run it in its own folder;
- its end time and sizes are often cut down so the docs build fast — set them for the user's goal;
- tuned values (e.g. the S2 timing of the spiral-wave demo) must be tuned again when the mesh, dt or
  cell model change;
- helper functions defined in the demo (e.g. spiral-tip tracking) are demo code, not `beat` API —
  copy them along;
- units are whatever the demo chose (e.g. mm and µF/mm²) — keep them consistent, and check
  conductivities, $\chi$ and $C_m$ against the mesh unit.

Save the fields needed for post-processing with io4dolfinx and put post-processing in a separate
script ([outputs.md](outputs.md)).

## Step 5: Check, then hand over

1. **Offer a smoke run** before the real one: same setup with a short end time (a few ms past the
   stimulus, e.g. `beat run config.toml --set 'solver.end_time="20 ms"'`, into a separate
   `--output-folder`) and, if meshing
   is slow, a coarser mesh. One rank unless MPI itself is under test.
   For a Python script, the same with its end time cut down. For an eikonal solve, a coarse mesh
   is the smoke run.
2. **Check the smoke run**: values finite (no NaN); $v$ makes an upstroke (from rest to above
   about 0 mV for physiological models) where stimulated; **the wave propagates** — tissue away from
   the stimulus activates (activation map from `beat post`, or $v$ at a probe point). For eikonal
   activation times: finite everywhere, no unreached vertices (`allow_unreached=False` raises), and
   the latest activation time plausible for the size and velocity. A stimulus
   that fires but never spreads usually means the mesh is too coarse for the conductivities, the
   stimulus is too weak or too short, or the stimulus is applied twice/never.
3. **Hand over**: the full command (`beat run config.toml`, or `mpirun -n N beat run config.toml`;
   `python script.py` / `mpirun -n N python script.py`), roughly how long it will take (scale from
   the smoke run), where results land, how to post-process (`beat post config.toml`) and restart
   (`beat run config.toml --restart`).

**Never start a long run or an MPI run without asking** — offer the command; run it only on an
explicit yes.

## Ground rules

- **Do not invent API or config keys.** Check the installed package (`help`, `inspect.signature`,
  `beat validate-config`) or the docs (https://finsberg.github.io/fenicsx-beat). If something does
  not exist, say so and offer the closest thing that does.
- Use the notation of the docs: $v$ (transmembrane potential), $M$ (conductivity tensor),
  $\chi$ (surface-to-volume ratio), $C_m$ (membrane capacitance), $I_{ion}$, $I_{stim}$.
- Physical units matter: CLI quantities are pint strings; check the mesh's length unit
  (`geometry.unit`) against conductivities, $\chi$ and stimulus sizes.
- Known pitfalls to raise when relevant:
  - a too coarse mesh slows conduction or blocks it entirely (calibrate conductivities, refine,
    or use reaction-eikonal — [numerics.md](numerics.md));
  - `rush_larsen_2` steps $v$ explicitly, so a large `solver.dt` can blow up: use a smaller dt or
    `cell.scheme = "generalized_rush_larsen"` with Godunov splitting;
  - with merged Strang half-steps in Python (`beat.SplittingSolver(..., merge_half_steps=True)`)
    call `solver.synchronize()` before reading or saving the state (the CLI does this itself);
  - a stimulus in both the cell model (its own stimulus parameters) and the PDE stimulates twice —
    set the cell model's stimulus amplitude to zero when the PDE stimulates (the templates do);
  - steady state: a cell model started from its default initial state drifts for many beats; pace
    to steady state (`cell.steady_state`) when the result depends on it.
