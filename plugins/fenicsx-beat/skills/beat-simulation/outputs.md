# Outputs: what to save, how often, and recomputing outputs

Principle: **save the raw fields once, recompute everything else from them.** Movies, activation
maps and ECGs should never require re-running the simulation.

## What the user wants to look at

| Output | Needs | CLI | Python |
|---|---|---|---|
| Movie / snapshots of $v$ | $v$ at every frame | `beat post`: `post/v.bp` (ParaView), `voltage.gif` with `postprocess.make_gif = true` (needs pyvista) | write $v$ with io4dolfinx; render with pyvista or ParaView |
| Activation-time map | $v$ finely in time, or tracking during the run | `beat post`: `post/activation_time.bp`, map image | `beat.activation.ActivationTracker` (updated every step) |
| Activation times at points | as above | `postprocess.points` → `post/activation_times.json` (threshold `postprocess.activation_threshold`, in the cell model's units of $v$) | `beat.utils.PointEvaluator` + tracker |
| Traces / APD at probe points | $v$ at the points every step or every save | from `results.h5` | `beat.utils.PointEvaluator` |
| Pseudo-ECG at electrodes | $v$ every ~1 ms | `[postprocess.ecg]` → `post/ecg.csv`, `ecg.png` | `beat.ECGRecovery`, `beat.ecg.ElectrodePotentials` |
| 12-lead ECG | electrodes LA, RA, LL, V1–V6 | `postprocess.ecg.leads = "twelve-lead"` → `ecg_leads.csv`, `ecg_leads.png` | `beat.ecg.twelve_lead` |
| Electrograms in a bath | bounded tissue + bath | — | `beat.pseudo_bidomain.VolumeConductor`, `beat.pseudo_bidomain.LeadFieldElectrodes` |
| Other states (e.g. calcium) | the state field | `output.fields = ["cai"]` (state names of the cell model) | write the state's Function |

In Python, electrodes are a mapping name → position: `beat.ecg.twelve_lead().points(electrodes)`
gives the points for `beat.ecg.ElectrodePotentials(recovery, points)`; read the user's CSV into that
dict (check the column order and unit with the user).

Electrode positions are in the mesh's coordinate system (`postprocess.ecg.unit`, default
`geometry.unit`) and must lie outside the heart. For an Alya-order electrode CSV,
[`scripts/electrodes_to_toml.py`](https://github.com/finsberg/fenicsx-beat/blob/main/scripts/electrodes_to_toml.py)
converts it to the `[postprocess.ecg.electrodes]` table.

## How often to save

Ask, and explain the trade-off:

- File size ≈ number of saves × number of mesh nodes × fields × 8 bytes (a 1 M-node heart, 1 field,
  every 1 ms for 500 ms ≈ 4 GB).
- ECGs: ~1 ms is enough (clinical ECGs sample at 0.5–2 ms).
- Activation maps from saved $v$ are only as precise as the save interval (1 ms saves → ±0.5 ms
  activation times). For finer activation times without saving more, track activation during the run
  (Python: `beat.activation.ActivationTracker`), or save more often only for short runs.
- Movies: 1–5 ms per frame is plenty.
- Restart checkpoints (`output.checkpoint_every`, all ODE states) are separate from results; set them
  for long runs on clusters with time limits.

CLI default: `output.save_every = "1 ms"`, checkpoints only at the end.

## CLI: run once, post-process many times

```text
output/
  results.h5        # v (+ output.fields) every output.save_every
  restart.h5        # checkpoints (output.checkpoint_every and at the end)
  config.resolved.toml, run.json, output.log
  post/             # written by `beat post`
```

`beat post config.toml` reads `results.h5` and writes `post/`. Change `[postprocess]` (points, ECG
electrodes, leads, `postprocess.make_gif`, `postprocess.vtx`) or `[output]` and run `beat post`
again — no new simulation. It runs on any number of ranks, independent of the run (images show only
rank 0's part of the mesh on more than one rank, so render them serially). `beat post` refuses a
config whose physics differs from the run's; only `[output]`, `[postprocess]` and the run length may
change.

## Python: save with io4dolfinx, post-process in a separate script

`beat post` only reads CLI runs (it checks the run's config), so a Python run needs its own
post-processing script.

In the simulation script (with merged Strang half-steps, call `solver.synchronize()` first):

```python
import io4dolfinx

io4dolfinx.write_mesh(results_path, mesh)  # once
...
if step % save_every_steps == 0:
    solver.synchronize()  # only with merge_half_steps=True
    io4dolfinx.write_function(results_path, pde.state, time=t, name="v")
```

In a separate post-processing script, read it back on any number of ranks:

```python
mesh = io4dolfinx.read_mesh(results_path, MPI.COMM_WORLD)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
v = dolfinx.fem.Function(V)
for t in io4dolfinx.read_timestamps(results_path, MPI.COMM_WORLD, function_name="v"):
    io4dolfinx.read_function(results_path, v, time=t, name="v")
    ...  # ECG (beat.ECGRecovery), activation map, frames for a movie
```

Check signatures in the installed io4dolfinx (`help(io4dolfinx.write_function)`): the backend
(`backend="adios2"` or `"h5py"`) must be the same for writing and reading; the CLI uses `h5py`.
Use the same function space (family, degree) as the simulation when reading.
