# Command line interface

`fenicsx-beat` ships a `beat` command line tool that runs a tissue-level monodomain simulation
(and its postprocessing) purely from a `config.toml` file, so routine runs — including batch/array
jobs on an HPC cluster — don't need a Python script. This page covers installation, the workflow,
the commands and override mechanisms, the output folder layout, and the built-in templates. See
also [Running on a cluster](cli_cluster.md) and the generated [configuration
reference](cli_reference.md).

```{note}
The CLI drives one spatially homogeneous-or-layered cell model per simulation from one `.ode` file
(shared parameters, or per-region overrides via `[cell.regions.<name>]` — see the
[config reference](cli_reference.md)). Per-region *different* `.ode` files, and changing
ODE parameters at specific times during a run (`parameter_schedule`), are future work; the
[demos](../demos/index.md) that need them (e.g. [pvc.py](../demos/pvc.py),
[pace_train.py](../demos/pace_train.py)) are not fully reproducible via `config.toml` yet — see the
[templates table](#templates) below for the exact deviations.
```

## Installation

The CLI's dependencies (`pydantic`, `pydantic-pint`, `cardiac-geometriesx`, `gotranx`,
`io4dolfinx`, ...) are bundled in the `cli` extra:

```bash
pip install "fenicsx-beat[cli]"
```

Visualization from `beat post` additionally needs `pyvista` (e.g. via `pip install
"fenicsx-beat[docs]"`, or just `pip install pyvista`); it's optional and silently skipped with a
log warning if not installed.

```{warning}
If your config uses `solver.pde.type = "irksome"` / `solver.ode.type = "irksome"`, install the
`irksome` extra too -- but **not** straight from PyPI: `irksome[dolfinx]` on PyPI ignores
`backend="dolfinx"` and imports `firedrake` regardless. Install the `git+` version instead:

    pip install "irksome[dolfinx] @ git+https://github.com/firedrakeproject/Irksome.git"

Without it, any config that needs it fails fast with an install hint before doing any mesh/MPI
work; it does *not* fail silently.
```

## Quick start

```bash
beat init case/config.toml --template slab   # write a starter config (+ its .ode file)
beat validate-config case/config.toml        # parse + validate, print the resolved config
beat geometry case/config.toml               # generate/cache the mesh, then stop
beat run case/config.toml                    # run the simulation
beat post case/config.toml                   # activation times, pseudo-ECG, VTX, PNG/GIF
```

`beat init --template NAME` copies `NAME`'s `config.toml` (and any companion files, e.g. its
`.ode` model) from the package's built-in templates -- see the [templates table](#templates) for
the full list, one per row. Without `--template`, it defaults to `slab`, a thin box of tissue
stimulated at one end (the simplest possible geometry, good for checking the CLI wiring itself)
with `geometry.type = "box_slab"` -- a structured tetrahedral box built directly by `beat` in
memory on every run (cheap; nothing is written to disk), no `gmsh`/mesh-generation tool of your own
needed. `beat init --template lv_endocardial`/`biv_endocardial`/`ukb_atlas` instead generate a real
gmsh mesh (`geometry.type = "slab"`/`"lv_ellipsoid"`/`"biv_ellipsoid"`/`"ukb"`, via
`cardiac-geometriesx`, and `ukb-atlas` for the last one), which can take from seconds to minutes
depending on resolution. `beat geometry` (also run implicitly by `beat run` and `beat post`)
generates such a mesh once and reuses it on every later invocation: it is cached in its own
subfolder `geometry.folder/<hash>/`, keyed on a hash of the `[geometry]` section (everything except
`folder` and `unit`). Changing a geometry parameter therefore creates a *new* subfolder next to the
old one rather than replacing it, and `beat` never deletes `geometry.folder` itself or anything in
it that it didn't create -- `geometry.folder = "."` (the config's own directory) is safe. Delete
stale `<hash>/` subfolders by hand when you no longer need them. Run `beat geometry` on its own
first if you just want to warm that cache (e.g. on a cluster login node, see [Running on a
cluster](cli_cluster.md)); it logs the subfolder it used.

`geometry.type = "folder"` is the other case: point `folder` at a mesh you (or `geox`, or another
`cardiac-geometriesx`-compatible tool) already generated externally, with a `markers.json` describing
its facet/cell tags -- `beat` then only *loads* it, via
`cardiac_geometries.geometry.Geometry.from_folder`.

The points in `[postprocess.points]` are where `beat post` reports activation times; the
pseudo-ECG's electrodes go in `[postprocess.ecg.electrodes]` instead (see
[Pseudo-ECG](#pseudo-ecg)). A point outside the mesh has no activation time: `beat post` records
`null` for it (with a log warning), which is distinct from the `-1.0` it uses for a point that's
inside the mesh but simply hadn't activated by the end of the recorded run. Also note that
`postprocess.activation_threshold` is in the cell model's own units for `v`: a real ionic model's
`v` is in mV (so a physiological threshold is around `0.0`), while a normalized two-variable model
like Mitchell-Schaeffer (used by a couple of templates, e.g. `irksome_model_gotranx`) has `v`
roughly in `[0, 1]`, so its threshold should be something like `0.5` instead.

## Commands

`validate-config`, `geometry`, `run` and `post` all take a config path and accept repeatable
`--set KEY=VALUE` overrides. `init` takes an optional config path (default `config.toml`) but no
`--set` (there's no existing config to override yet). `version` takes neither -- it only prints
version numbers. The global `-v/--verbose`, `--log-all-cpus` and `--dry-run` flags are accepted
by every subcommand, either before or after the subcommand name: `beat -v run config.toml` and
`beat run config.toml -v` are equivalent.

| Command | Behaviour |
|---|---|
| `beat init [config.toml] [--template NAME] [--force]` | Write a starter config from a template (default: `slab`); `--force` overwrites existing files. |
| `beat validate-config config.toml` | Parse + validate + print the resolved config. Builds nothing (no mesh, no MPI-collective work). |
| `beat geometry config.toml` | Generate (or load) the geometry and stop. For a generated type the mesh is cached in `geometry.folder/<hash>/` and reused across the other commands and reruns. |
| `beat run config.toml [--restart] [--overwrite] [--output-folder P] [--petsc-options "..."]` | Run the simulation. |
| `beat post config.toml [--output-folder P]` | Activation times, the pseudo-ECG (with `[postprocess.ecg]`, see [Pseudo-ECG](#pseudo-ecg)), VTX conversion and visualizations from `results.bp`. |
| `beat version` | Versions of beat, dolfinx, mpi4py, petsc4py. |

`beat --dry-run <command> ...` (e.g. `beat --dry-run run config.toml --set 'solver.dt="0.02 ms"'`)
logs the raw, argparse-parsed arguments and exits -- it does **not** load the TOML file or resolve
`--set`/env overrides, so it can't catch a config or override mistake, only confirm which flags
argparse itself accepted. To actually check that a config (with its overrides applied) parses and
validates, use `beat validate-config` instead, which does load and resolve everything and prints
the fully resolved result.

(exit-codes)=
### Exit codes

- `0` success.
- `1` a configuration/validation error (`ConfigError`), a command-line usage error (unknown flag,
  missing argument), or a missing `cli` extra (the error names the `pip install
  "fenicsx-beat[cli]"` fix).
- `2` a runtime failure: a solver failure (e.g. a non-finite transmembrane potential), or any
  other unexpected error (mesh generation, I/O, ...). It is logged as a one-line error; run with
  `-v` for the full traceback.

`beat run` builds the whole simulation -- cell model, geometry, markers, stimuli, solvers -- before
it touches the output folder, so a failure during that setup leaves the output folder exactly as it
was (no `run.json`, and with `--overwrite` the previous results are *not* deleted). Only once the
run has started does a failure set `run.json: status = "failed"` (with the error message). Not
every `ConfigError` is caught at the same point, though all of them are caught before the actual
PDE/ODE solve loop (except a restart checkpoint that doesn't match, which is found when it's
loaded):

- Bad TOML, wrong units, an unknown TOML key or geometry/stimulus/solver `type`, a missing
  `cell.ode_file`, an unknown `cell.scheme`, or a `cell.parameters`/`cell.regions.*.parameters` name
  the generated cell model doesn't have -- all reported before any mesh is built or loaded (the
  cell-model code is generated from `cell.ode_file` itself and doesn't need a mesh).
- Marker names (`[[stimulus]]`, `cell.layers`) and fiber availability (`fibers = "from_geometry"`
  needing an `f0` from the geometry) can only be checked once the mesh has actually been built or
  loaded, since they *are* properties of that mesh -- so these are reported early in `beat run`,
  right after the geometry step, but not before it.

For a **generated** geometry (`slab`/`lv_ellipsoid`/`biv_ellipsoid`/`ukb`), building the mesh itself
can be by far the most expensive part of that early setup -- run `beat validate-config` (parse-time
checks only, no mesh) and then `beat geometry` (builds/caches the mesh once, cheaply reused by every
later command) before submitting a long or queued job, so that if a marker/fiber-config mistake
*is* still there, `beat run` fails within seconds against the already-cached mesh, rather than after
regenerating it inside the timed job.

(overrides)=
## Overrides

Config values can come from four places, in order of increasing precedence:

1. The TOML file itself.
2. Environment variables: `BEAT_<SECTION>__<KEY>`, e.g. `BEAT_SOLVER__DT="0.02 ms"`. The `BEAT_`
   prefix and `__` (double underscore) nesting delimiter are fixed, but the section/key names
   themselves are matched **case-insensitively** against the config schema, so `BEAT_EP__C_M` and
   `BEAT_ep__c_m` both resolve to `ep.C_m` (whose field name is mixed-case).
3. `--set dotted.key=value` (repeatable). `value` is parsed as a TOML literal, the same way it
   would appear on the right-hand side of a `key = value` line in the file:
   - `--set 'solver.dt="0.02 ms"'` (a quantity is still a quoted string)
   - `--set ep.conductivity.sigma_il=0.2`
   - `--set 'postprocess.points.P1=[0,0,0]'`
   - List elements are addressed **by index**: `--set 'stimulus.0.start="10 ms"'` sets the first
     `[[stimulus]]` table's `start`.
   - Unknown keys are an error (`ConfigError`), never silently dropped.
4. Dedicated flags on `beat run`/`beat post`: `--output-folder` and (on `beat run`)
   `--petsc-options`.

`--output-folder` overrides `output.folder` and, unlike every path *inside* the config file,
resolves against the **current working directory** rather than the config file's directory --
handy for array jobs launched from one shared directory (see
[Running on a cluster](cli_cluster.md)).

`--petsc-options "-ksp_type cg -pc_type hypre"` merges into `solver.petsc_options` (parsed with
`shlex`, each `-key value` pair; a bare `-flag` becomes `True`). Negative numbers are accepted as
option *values*, not mistaken for the next flag, e.g. `--petsc-options "-ksp_rtol -1e-6"`.

Relative paths written *inside* the config file (`cell.ode_file`, `geometry.folder` for
`type = "folder"`, `output.folder`) resolve against **the config file's own directory**, not the
current working directory -- so `beat run /abs/path/to/config.toml` from anywhere still finds
`ode_file`/the mesh next to the config, which matters once a cluster job script `cd`s elsewhere
before running `srun beat run ...`.

Whatever the config resolves to after all four layers, it's written out in full to
`output/config.resolved.toml` at the start of every `beat run` -- the single source of truth for
"what actually ran".

pydantic-settings' `CliSettingsSource` is deliberately not used here (poor support for lists of
discriminated unions, and an unwieldy auto-generated `--help`).

## Output folder layout

`beat run` writes into `output.folder` (default `output`, relative to the config file):

```text
output/
  config.resolved.toml     # the fully resolved configuration of the (latest) run
  run.json                 # versions, n_ranks, start/end wall time, status: running/finished/failed
  output.log               # log file (output_all_cpus.log too when running on >1 rank)
  cell_model_<hash>.py      # gotranx-generated code for cell.ode_file (hash: file contents + scheme); kept by --overwrite
  init_states/<region>_<hash>.npy   # cached single-cell steady state (only if cell.steady_state is set); kept by --overwrite
  results.bp               # io4dolfinx: v (+ output.fields), every output.save_every
  restart.bp                # io4dolfinx: v and every ODE state, every output.checkpoint_every and at the end
  restart.json              # the latest complete checkpoint's time/step and a hash of the run's physics
  performance.json          # timing summary (only with output.performance = true)
  post/                     # written by `beat post`, see below
```

`beat run` never writes VTX itself -- only the io4dolfinx files above. `beat post config.toml`
reads `results.bp` (which can happen later, on any number of ranks, independent of how many ranks
the run itself used) and writes into `post/`. The config given to `beat post` must describe the
same physics as the run that wrote `results.bp` -- the same check as for `--restart` (below),
against the hash in `restart.json`, or, if the run stopped before writing its first checkpoint,
against `config.resolved.toml` (whose `[postprocess]` is not read). Only `[output]`,
`[postprocess]` and the run length may differ; anything else (e.g. an edited `geometry.dx`) is
refused with a `ConfigError` naming `config.resolved.toml` to compare with, rather than crashing
or silently producing wrong results.

```text
output/post/
  v.bp                       # results.bp converted to VTX (ParaView), if postprocess.vtx (default true)
  activation_time.bp          # full-mesh local activation-time map (VTX)
  activation_times.json       # activation times at postprocess.points (null if a point is outside the mesh)
  voltage_final.png            # snapshot of v at the last saved time (needs pyvista)
  activation_time_map.png      # snapshot of the activation-time map (needs pyvista)
  voltage.gif                  # animation of v(t) over the whole run, if postprocess.make_gif (needs pyvista)
  ecg.csv                      # pseudo-ECG: time, then the potential at each electrode, if [postprocess.ecg]
  ecg.png                      # plot of ecg.csv (needs matplotlib)
  ecg_leads.csv                # time, then each lead, if postprocess.ecg.leads = "twelve-lead"
  ecg_leads.png                # the leads in the clinical 3x4 layout (needs matplotlib)
```

On more than one rank, the PNG/GIF previews show only rank 0's partition of the mesh (with a log
warning); the VTX files are always complete. Run `beat post` on a single rank for full previews.

(pseudo-ecg)=
### Pseudo-ECG

With a `[postprocess.ecg]` section, `beat post` also recovers the pseudo-ECG: the extracellular
potential at each electrode, in an infinite homogeneous conductor (`beat.ECGRecovery`), at every
saved time. It is computed in the same pass over `results.bp` as the activation map, so the ECG
adds no extra pass over the saved times. Without the section, `beat post` writes no ECG. The
electrodes are a table of name to position:

```toml
[postprocess.ecg]
unit = "cm"                     # the electrodes' length unit; default: geometry.unit
leads = "twelve-lead"           # or "none" (default): electrode potentials only
reference = "potential"         # or "position" (legacy simcardems), which needs twelve-lead
sigma_b = 1.0                   # bath conductivity (default 1.0)

[postprocess.ecg.electrodes]
LA = [4.0, -12.0, -7.0]
RA = [-15.0, 0.0, -10.0]
LL = [17.0, 11.0, 7.0]
V1 = [-3.0, 4.0, -9.0]
# ... V2 to V6
```

- `ecg.csv` has `time`, then one column per electrode, in the order given: the electrodes you
  give, and nothing else.
- `leads = "twelve-lead"` also writes `ecg_leads.csv` (`time`, then `I, II, III, aVR, aVL, aVF,
  V1, ..., V6`) and `ecg_leads.png`. It needs the electrodes `LA`, `RA`, `LL` and `V1` to `V6`.
  Any others (`RL`, or a body-surface set) are written to `ecg.csv` and otherwise unused. The lead
  `V1` and the electrode `V1` are in separate files, so they never share a column. With
  `leads = "none"`, `beat post` deletes an earlier run's `ecg_leads.*`; without the section it
  touches no `ecg*` file.
- `reference` sets how a negative pole of more than one electrode is formed. I, II and III are the
  same under both.
  - `"potential"` (the default; Einthoven, Goldberger, Wilson): the mean of its electrodes'
    potentials, e.g. `aVR = φ(RA) - (φ(LA) + φ(LL))/2`, and, against Wilson's central terminal,
    `V1 = φ(V1) - (φ(LA) + φ(RA) + φ(LL))/3`.
  - `"position"` (legacy simcardems): the potential at the mean of its electrodes' *positions*,
    e.g. `aVR = φ(RA) - φ(LA_LL_mid)` and `V1 = φ(V1) - φ(WCT_pt)`. These derived points are not
    electrodes and are not written to `ecg.csv`; an electrode may not share their names.
- The positions are converted from `unit` to `geometry.unit`, and each must have as many
  coordinates as the mesh has dimensions.
- An electrode may lie inside the mesh, but the value there is a near-field one, not a
  body-surface potential, and `beat post` logs a warning naming it.
- `beat post` checks the section once it has built the geometry, before it reads any saved time,
  and refuses (exit 1) a position of the wrong length or a lead set missing an electrode, naming
  them.

A header-less electrode CSV of the legacy simcardems / Alya kind (one `x,y,z` row per electrode,
in the order LA, RA, LL, RL, V1 to V6) is converted to this table once, with the script
`scripts/electrodes_to_toml.py` in the repository (<https://github.com/finsberg/fenicsx-beat>).
It is not installed by pip: run it from a clone or a downloaded copy. It needs only numpy.
`--names A,B,...` gives another order, and `-o FILE` writes to a file instead of stdout. Append
its output to the config:

```bash
python scripts/electrodes_to_toml.py electrodes.csv --unit cm > ecg.toml
```

`[postprocess]` is outside the physics hash, so a finished run can be post-processed again with
other electrodes or leads. Such a rerun also redoes the activation map and the previews, and the
VTX conversion unless it is switched off:

```bash
beat post config.toml --set 'postprocess.ecg.electrodes.V1=[-3.5, 4.0, -9.0]' --set postprocess.vtx=false
beat post config.toml --set 'postprocess.ecg.reference="position"' --set postprocess.vtx=false
```

### `--overwrite` and `--restart`

Re-running into a non-empty output folder (e.g. an array-job index collision, or simply rerunning
by hand) is refused by default -- `beat` never silently deletes anything:

- `--overwrite` deletes *only the artifacts `beat` itself wrote* (everything listed above, plus
  `post/`, except the content-hashed `cell_model_<hash>.py`/`init_states/` caches, which stay valid
  and are reused) and starts fresh. It does so only *after* the new config has been fully
  validated (the simulation is built first), so a mistake in the config never costs the previous
  results. Anything else in that folder -- notably your own `config.toml`/`.ode` files, if the
  output folder happens to be the config's own directory -- is left untouched.
- `--restart` continues from `restart.json`/`restart.bp` instead. It refuses if the run's
  **physics** has changed since the checkpoint was written: the check is a hash of the whole
  resolved config *excluding* the run length `solver.end_time`/`solver.num_beats`/`solver.BCL` (so
  extending the simulated time, or switching from `end_time` to `num_beats`/`BCL`, is fine;
  `BCL` only sets the run length, `num_beats x BCL` -- it doesn't pace anything, use a stimulus
  `period` for that, and `beat run` warns if no stimulus has one) and excluding `[output]` and
  `[postprocess]` entirely (change `save_every`, `checkpoint_every`, `performance`, any
  `[postprocess]` setting, freely across a restart). `geometry.folder` only matters for
  `geometry.type = "folder"` (where it *is* the mesh being simulated); for every generated
  geometry type it's just a cache location and is excluded like any other non-physics path.
  Restarting on a **different number of MPI ranks** than the original run is allowed. If the run
  stopped before its first checkpoint there is no `restart.json` yet, and both `--restart` and a
  plain rerun are refused with a message saying so -- use `--overwrite` to start over.
- A restart never rewrites a `results.bp` timestamp that's already there: io4dolfinx *appends* a
  duplicate write at an existing timestamp, and its reader returns the *first* match, so
  re-writing the same time would be silently ignored on read anyway -- the runner simply skips it.

## Units

Every physical quantity in the config is a pint string, `"<value> <unit>"` (e.g. `dt = "0.05 ms"`,
`sigma_el = "0.62 S/m"`) -- a bare number is rejected with a validation error naming the field.
The one field that needs a moment's thought is `stimulus.amplitude`: it's a *current density*, but
which dimension depends on the **stimulus type and, for a marker, the marker's own dimension** --
not the mesh's topological dimension. Precisely (`src/beat/cli/stimulus.py::_scaled_amplitude`,
mirroring `beat.stimulation.compute_effective_dim`):
`effective_dim = entity_dim + (3 - mesh.topology.dim)`, where `entity_dim` is the dimension of the
marked entity (facet or cell) for a `marker` stimulus, and the mesh's own topological dimension for
`box`/`random_endocardial`. A facet's `entity_dim` is always `mesh.topology.dim - 1`, so it always
cancels out to `effective_dim = 2` regardless of the mesh dimension; a cell marker's `entity_dim`
equals `mesh.topology.dim`, always cancelling to `effective_dim = 3`. In short:

- A `marker` stimulus on a **facet** marker is always areal, `uA/cm**2`, whatever the mesh's own
  dimension (1D, 2D or 3D).
- A `marker` stimulus on a **cell** marker, a `box` stimulus, and a `random_endocardial` stimulus
  are all always volumetric, `uA/cm**3`, likewise regardless of the mesh's own dimension.

A current per length (`uA/cm`) is therefore never valid. Get the dimension wrong and the config is
rejected before any solve, naming the mismatch: at parse time (`beat validate-config`) for `uA/cm`
and for a `box`/`random_endocardial` amplitude that isn't per volume; once the mesh is loaded for
a `marker` stimulus, whose facet-or-cell dimension is a property of the mesh.

(templates)=
## Templates

Each template under `src/beat/cli/templates/<name>/` is a runnable `config.toml` (plus any
companion `.ode` file) reproducing one of the tissue-level demos as closely as the CLI schema
allows; `beat init --template NAME` copies it (and its companions) into place. Where a demo does
something the schema can't express yet, the template's header comment documents the deviation --
summarized here:

| Template | Demo | Known deviation |
|---|---|---|
| `slab` | [slab.py](../demos/slab.py) | Cable partitioned into endo/mid/epi celltypes by raw x-position isn't expressible via `[cell.layers]` yet; homogeneous ToR-ORd region instead. |
| `niederer_benchmark` | [niederer_benchmark.py](../demos/niederer_benchmark.py) | -- |
| `fitzhughnagumo` | [fitzhughnagumo.py](../demos/fitzhughnagumo.py) | Rescaled from a dimensionless unit square to 100x100 mm; conductivity from the `Niederer` preset rather than the demo's arbitrary scalar `M`. |
| `diffusion` | [diffusion.py](../demos/diffusion.py) | The demo has no ODE/cell model at all (pure diffusion); the CLI always couples an ODE step, so this template ships a trivial `passive.ode` (`dv/dt = 0`) instead. |
| `pvc` | [pvc.py](../demos/pvc.py) | The demo's per-DOF g_Kr/g_Ks heterogeneity (right half of the cable) has no marker to key `[cell.layers]` off on an `interval` geometry; homogeneous cable instead (no PVC emerges, but pacing/cell model match). |
| `pace_train` | [pace_train.py](../demos/pace_train.py) | The demo switches its stimulus off at runtime (a parameter schedule, future work); this template uses a fixed PDE pulse train for the whole run instead. |
| `lv_endocardial` | [lv_endocardial.py](../demos/lv_endocardial.py) | -- |
| `biv_endocardial` | [biv_endocardial.py](../demos/biv_endocardial.py) | -- |
| `ukb_atlas` | [ukb_atlas.py](../demos/ukb_atlas.py) | Needs network access on first run (atlas download, cached afterwards with the mesh in `geometry.folder/<hash>/`). |
| `irksome_model_gotranx` | [irksome_model_gotranx.py](../demos/irksome_model_gotranx.py) | Uses `RadauIIA`/`stages=1` rather than the demo's `BackwardEuler()` (not expressible via `tableau`/`stages`); PDE stimulus box instead of the demo's non-default initial condition; uniform conductivity preset instead of the demo's piecewise-constant one. |
| `external_operator_gotranx` | [external_operator_gotranx.py](../demos/external_operator_gotranx.py) | Conductivity built from `[ep]` (the `Niederer` preset) rather than the demo's raw scalar `M`. |

`examples/cli/README.md` in the repository points at the same templates for anyone browsing the
source tree directly rather than an installed package.

## See also

- [Configuration reference](cli_reference.md) -- every section/field, generated from the pydantic
  models, so it can't drift from the code.
- [Running on a cluster](cli_cluster.md) -- SLURM array jobs, wall-time/`--restart` patterns, and
  solver advice for large meshes.
