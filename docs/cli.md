# Command line interface

`fenicsx-beat` ships a `beat` command line tool that runs a tissue-level monodomain simulation
(and some postprocessing) purely from a `config.toml` file, so you don't need to write a Python
script for routine runs. This page walks through the full workflow — building a mesh, writing a
`config.toml`, and running/postprocessing a simulation — for three geometries:
a [slab](#example-1-a-slab), an idealized [left-ventricle ellipsoid](#example-2-a-left-ventricle-ellipsoid),
and a realistic [UK Biobank bi-ventricular atlas](#example-3-a-uk-biobank-bi-ventricular-geometry)
mesh.

```{note}
The `beat` CLI currently supports a single, spatially homogeneous cell model per simulation (one
`.ode` file, one parameter set for the whole tissue). The [demos](../demos/index.md) — e.g.
[lv_endocardial.py](../demos/lv_endocardial.py) — go further, with transmurally varying cell
models (endocardial/mid-myocardial/epicardial) via `beat.odesolver.DolfinMultiODESolver`; that
isn't exposed through `config.toml` yet.
```

## Installation

The CLI's dependencies (`pydantic`, `cardiac-geometriesx`, `gotranx`, `io4dolfinx`, ...) are
bundled in the `cli` extra:

```bash
pip install "fenicsx-beat[cli]"
```

Visualization from `beat post` additionally needs `pyvista` (e.g. via `pip install
"fenicsx-beat[docs]"`, or just `pip install pyvista`) — it's optional and silently skipped with a
log warning if not installed.

## Overview of the workflow

1. **Build a mesh.** `cardiac-geometriesx` installs a `geox` command line tool that generates
   meshes (with fibres) for several standard geometries — run `geox --help` for the full list.
   This page covers `geox slab`, `geox lv-ellipsoid` and `geox ukb`.
2. **Get a cell model.** Point `cell.ode_file` at any `.ode`/CellML-derived model, e.g. one of the
   ones under [`odes/`](https://github.com/finsberg/fenicsx-beat/tree/main/odes) in the repository,
   or your own. `beat run` generates the Python code for it (via
   [`gotranx`](https://finsberg.github.io/gotranx)) the first time it's needed and reuses it after
   that.
3. **Write a `config.toml`.** `beat init config.toml` writes one with default values you can edit;
   `beat validate-config config.toml` checks it parses without actually running anything.
4. **Run it.** `beat run config.toml` runs the simulation and writes a VTX file (for viewing in
   ParaView) plus a checkpoint used by the postprocessing commands below.
5. **Postprocess it.** `beat ecg config.toml` recovers a pseudo-ECG at configured points; `beat
   post config.toml` computes local activation times (as a full-mesh map and at configured points)
   and, if `pyvista` is installed, renders PNG/GIF visualizations.

## The `config.toml` sections

| Section | Field | Meaning |
|---|---|---|
| `[mesh]` | `unit` | Length unit the mesh coordinates are in, e.g. `"mm"` or `"cm"` |
| | `folder` | Folder written by `geox` (or `cardiac_geometries.geometry.Geometry.save_folder`) |
| `[cell]` | `ode_file` | Path to the `.ode` cell model |
| | `module_name` | Where to cache the `gotranx`-generated Python module |
| | `scheme` | Integration scheme, e.g. `"generalized_rush_larsen"` |
| | `v_name` | Name of the transmembrane-potential state in the `.ode` file (usually `"v"` or `"V"`) |
| | `num_beats`, `BCL`, `dt` | Pacing protocol used to compute a steady-state initial condition for the single cell, before the tissue simulation starts |
| | `track_indices` | State names to record while computing the steady state (diagnostic plot/array in `output/init_states/`) |
| `[simulation]` | `num_beats`, `BCL` | Simulated time is `num_beats * BCL` |
| | `dt` | PDE (and ODE) time step |
| | `theta` | Splitting scheme parameter (`1.0` = Godunov, the default; `0.5` = Strang) |
| | `save_every_ms` | How often to write output |
| | `output_folder` | Where results/logs/checkpoints go (cleared at the start of each `beat run`) |
| `[stimulus]` | `marker` | Name of a **facet** marker (from the mesh's `markers.json`) where the stimulus is applied |
| | `amplitude`, `duration`, `start` | Stimulus current parameters, see {py:func}`beat.stimulation.define_stimulus` |
| `[ep]` | `chi`, `C_m` | Surface-to-volume ratio and membrane capacitance, see the [mathematical background](math_background.md) |
| | `conductivity.sigma_{i,e}{l,t}` | Intracellular/extracellular, longitudinal/transverse conductivities |
| `[postprocess]` | `points` | Named points (in `mesh.unit` coordinates), e.g. `{P1 = [0.0, 0.0, 0.0]}`, used by both `beat ecg` and `beat post` |
| | `activation_threshold` | Threshold on `v` used to determine local activation time |
| | `sigma_b` | Bath conductivity used by the ECG recovery, see {py:class}`beat.ecg.ECGRecovery` |
| | `make_gif` | If true, `beat post` also renders an animated GIF of `v(t)` (requires `pyvista`) |

Any field with physical units (`BCL`, `dt`, `duration`, `chi`, `C_m`, the conductivities, ...)
accepts a `"<value> <unit>"` string parsed by [pint](https://pint.readthedocs.io) — e.g. `dt =
"0.05 ms"` or `sigma_el = "6.2 mS/cm"`. `stimulus.amplitude` is the one exception: it's a plain
number, interpreted in the unit implied by the marker's dimension and `mesh.unit` (see
{py:func}`beat.stimulation.define_stimulus`).

A point used only for `beat ecg` doesn't need to lie inside the mesh — e.g. a far-field
"electrode" position — but `beat post` can't report an activation time there and records `null`
for it (with a log warning), instead of the `-1.0` it uses for a point that's inside the mesh but
simply hadn't activated by the end of the recorded run.

## Example 1: a slab

A thin slab of tissue, stimulated at one end — the simplest possible 3D geometry, good for
checking the CLI wiring itself. `geox slab` requires [gmsh](https://gmsh.info/); pass
`--no-create-fibers`'s opposite, `--create-fibers`, to also generate an analytic fibre field
(required — `beat run` needs one to build the conductivity tensor $M$).

```bash
geox slab mesh --lx 20 --ly 7 --lz 3 --dx 1.0 --create-fibers
```

This writes `mesh/markers.json` with facet markers `X0`, `X1`, `Y0`, `Y1`, `Z0`, `Z1` (the six
faces of the box) — `X0` is the natural place to stimulate to trigger a wave travelling down the
slab's long axis.

```toml
# config.toml
[mesh]
unit = "mm"
folder = "mesh"

[cell]
ode_file = "mitchell_schaeffer.ode"   # or e.g. odes/tentusscher_panfilov_2006/*.ode for a real run
num_beats = 20
BCL = "1000 ms"
dt = "0.05 ms"
module_name = "mitchell_schaeffer.py"
v_name = "v"
track_indices = ["v", "h"]

[simulation]
num_beats = 1
BCL = "20 ms"
dt = "0.05 ms"
save_every_ms = 1.0
output_folder = "output"

[stimulus]
marker = "X0"
amplitude = 5000.0
duration = "2.0 ms"

[postprocess]
activation_threshold = 0.5
sigma_b = 1.0

[postprocess.points]
P1 = [5.0, 3.5, 1.5]
P2 = [15.0, 3.5, 1.5]
```

```bash
beat run config.toml
beat ecg config.toml    # -> output/ecg.csv, output/ecg.png
beat post config.toml   # -> output/activation_time.xdmf, output/activation_times.json, ...
```

This mirrors the [conduction-velocity/ECG slab demo](../demos/slab.py), which computes conduction
velocity from the same kind of point-activation-time data by hand.

## Example 2: a left-ventricle ellipsoid

An idealized LV geometry (a truncated ellipsoid) — see the
[endocardial-stimulation demo](../demos/lv_endocardial.py) for the full multi-region version of
this same geometry.

```bash
geox lv-ellipsoid mesh --psize-ref 2.0 --create-fibers
```

(`--psize-ref` sets the target element size, in `cm`; `2.0` gives a coarse-but-fast mesh for a
quick trial run — the [demo](../demos/lv_endocardial.py) uses `0.15` for a much finer, much
slower mesh, more representative of a real study.)

`markers.json` has `ENDO`, `EPI` and `BASE` facet markers (plus point/ring markers used
internally by `cardiac_geometries`). Stimulating `ENDO` mimics activation spreading in from the
endocardium.

```toml
# config.toml
[mesh]
unit = "cm"
folder = "mesh"

[cell]
ode_file = "mitchell_schaeffer.ode"
num_beats = 20
BCL = "1000 ms"
dt = "0.05 ms"
module_name = "mitchell_schaeffer.py"
v_name = "v"
track_indices = ["v", "h"]

[simulation]
num_beats = 1
BCL = "20 ms"
dt = "0.05 ms"
save_every_ms = 1.0
output_folder = "output"

[stimulus]
marker = "ENDO"
amplitude = 5000.0
duration = "1.0 ms"

[postprocess]
activation_threshold = 0.5

[postprocess.points]
Epicardium = [0.0, 0.0, -9.5]
```

```bash
beat run config.toml
beat post config.toml
```

`output/activation_time_map.png` should show activation starting at the endocardium (dark) and
spreading outward to the epicardium (lighter) as the wave crosses the wall.

## Example 3: a UK Biobank bi-ventricular geometry

A realistic bi-ventricular geometry built from the UK Biobank statistical shape atlas (via the
optional [`ukb-atlas`](https://github.com/ComputationalPhysiology/ukb-atlas) dependency, pulled in
by `geox`) — see the [Purkinje-like stimulation demo](../demos/ukb_atlas.py) for a more elaborate
random-activation-pattern version of this geometry.

```bash
geox ukb mesh --char-length-max 2.0 --char-length-min 2.0 --create-fibers
```

The first run downloads the atlas data (cached under `~/.ukb/`) and can take a while, especially
at the default (fine) resolution — use a coarser `--char-length-max/min` (e.g. `5.0`) for a
quicker trial run. `markers.json` has `LV`, `RV` and `EPI` facet markers (plus the four valve
markers `MV`/`AV`/`PV`/`TV`); stimulating `LV` or `RV` mimics endocardial activation in one
chamber.

```toml
# config.toml
[mesh]
unit = "mm"
folder = "mesh"

[cell]
ode_file = "mitchell_schaeffer.ode"
num_beats = 20
BCL = "1000 ms"
dt = "0.05 ms"
module_name = "mitchell_schaeffer.py"
v_name = "v"
track_indices = ["v", "h"]

[simulation]
num_beats = 1
BCL = "20 ms"
dt = "0.05 ms"
save_every_ms = 1.0
output_folder = "output"

[stimulus]
marker = "LV"
amplitude = 500.0
duration = "1.0 ms"

[postprocess]
activation_threshold = 0.5

[postprocess.points]
# A far-field point outside the mesh: valid for `beat ecg`, reported as `null` (not an error)
# by `beat post`, which only reports activation times for points inside the tissue.
Torso = [200.0, 0.0, 0.0]
```

```bash
beat run config.toml
beat ecg config.toml
beat post config.toml
```

## A minimal `.ode` file, for trying this quickly

The examples above use the two-state
[Mitchell–Schaeffer model](https://www.sciencedirect.com/science/article/pii/S0092822302000809)
rather than a full ionic current model, purely so a first end-to-end run finishes in seconds:

```
parameters(
    tau_in = 0.3,
    tau_out = 6.0,
    tau_open = 120.0,
    tau_close = 150.0,
    v_gate = 0.13
)
states(
    v = 0.0,
    h = 1.0
)

h_inf = Conditional(Lt(v, v_gate), 1.0, 0.0)
tau_h = Conditional(Lt(v, v_gate), tau_open, tau_close)

dv_dt = h * (v**2 * (1.0 - v)) / tau_in - v / tau_out
dh_dt = (h_inf - h) / tau_h
```

Note that Mitchell–Schaeffer's `v` is a normalized action potential (roughly `0` to `1`, resting at
exactly `0`), not millivolts — that's why the examples above use
`postprocess.activation_threshold = 0.5` rather than a physiological voltage like `0.0` mV (which,
for this particular model, every point would trivially "cross" at rest, at `t=0`). For a real
study, swap in one of the ionic current models under
[`odes/`](https://github.com/finsberg/fenicsx-beat/tree/main/odes) (e.g. the ten
Tusscher–Panfilov or ToR-ORd models used in the other demos), a `num_beats` of at least 200 for
the single-cell steady state, and an activation threshold appropriate to that model's `v` (e.g.
`0.0` mV).
