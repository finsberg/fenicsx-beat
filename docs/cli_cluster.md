# Running on a cluster

The `beat` CLI is designed so one `config.toml` can drive an array-job parameter sweep, and a long
run can survive a wall-time kill and be continued (`--restart`), possibly on a different node
allocation. This page assumes SLURM but the same ideas (override flags, exit codes, restart) apply
to any scheduler.

(mesh-once-run-many)=
## Mesh once, run many

Generating a mesh (especially a realistic `lv_ellipsoid`/`biv_ellipsoid`/`ukb` geometry) can take
much longer than a short SLURM job's queue-wait budget wants, and every array-job task sharing
one `config.toml` would otherwise regenerate (or race to regenerate) it independently. Build it
once, on a login node or a short single-task job, before submitting the sweep:

```bash
beat validate-config config.toml   # parse-time checks only: no mesh, no MPI-collective work
beat geometry config.toml          # build/cache the mesh once
```

`beat geometry` is exactly the same step `beat run` would do lazily on first use, cached in
`geometry.folder` and keyed by a hash of the `[geometry]` section -- so running it up front is
purely an optimization, not a separate code path; every array-job task's own `beat run` reuses the
cache rather than rebuilding it. It's also the cheapest way to surface a marker-name typo
(`[[stimulus]]`, `cell.layers`) or a missing fiber field (`fibers = "from_geometry"`) ahead of the
timed sweep: those checks need the actual mesh, so they only run once `beat run` has built or
loaded the geometry (see [Exit codes](cli.md#exit-codes)) -- with the mesh already cached, that
happens within seconds rather than after a from-scratch mesh generation inside the job.

## A SLURM array job for a parameter sweep

`--output-folder` and `--set` let one base `config.toml` be varied per array-job task without
copying files:

```bash
#!/bin/bash
#SBATCH --job-name=beat-sweep
#SBATCH --array=0-9
#SBATCH --ntasks=64
#SBATCH --time=04:00:00

DT=(0.01 0.02 0.05 0.1 0.2 0.01 0.02 0.05 0.1 0.2)
srun beat run config.toml \
    --output-folder "runs/${SLURM_ARRAY_TASK_ID}" \
    --set "solver.dt=\"${DT[$SLURM_ARRAY_TASK_ID]} ms\"" \
    --overwrite
```

`--output-folder` resolves against the **current working directory** the job runs in (unlike every
path *inside* the config file, which resolves against the config file's own directory -- see
[Overrides](cli.md#overrides)), so `runs/${SLURM_ARRAY_TASK_ID}` above lands next to wherever the
job script itself runs from. `--overwrite` makes the task safe to resubmit: if that array index's
output folder already has results in it (e.g. a resubmit after a scheduler-level failure, or
resubmitting the whole array because task 3 failed), `beat` would otherwise refuse to touch it
rather than silently deleting a previous task's output from under a differently-indexed rerun.

## Surviving a wall-time kill: `--restart`

For a run whose simulated time exceeds what a single job's wall-time allows, set a checkpoint
interval and let the job resubmit itself onto the same output folder:

```bash
#!/bin/bash
#SBATCH --job-name=beat-longrun
#SBATCH --time=24:00:00
#SBATCH --ntasks=64

# output.checkpoint_every = "50 ms" in config.toml
if [ -f output/restart.json ]; then RESTART=--restart; fi
srun beat run config.toml $RESTART
```

Resubmit the same script (e.g. from a scheduler dependency chain, `sbatch --dependency=afterany`,
or cron) until `run.json: status == "finished"`. Each resubmission picks up `--restart`
automatically once `output/restart.json` exists (written after the first successful checkpoint).

**What may change across a restart, and what may not:** `beat` refuses `--restart` if the run's
*physics* has changed since the last checkpoint, comparing a hash of the whole resolved config
except `solver.end_time`/`solver.num_beats` and everything under `[output]`/`[postprocess]`. So
between restarts you may freely:

- Extend `solver.end_time` (or switch between `end_time` and `num_beats`/`BCL`) to run longer than
  originally configured.
- Change anything under `[output]` (`save_every`, `checkpoint_every`, `performance`, `log_every`,
  `fields`) or `[postprocess]`.
- Run on a **different number of MPI ranks** than the original job used (the checkpoint is read
  and redistributed across however many ranks the new job has).

But not, without `beat` refusing with an error naming the mismatch:

- `solver.dt`, `solver.theta`, `solver.pde`/`solver.ode` settings, `[ep]`, `[cell]` (including
  `cell.ode_file`'s *contents*, not just its path -- editing the `.ode` file itself also counts as
  a physics change), `[[stimulus]]`, or `[geometry]` (except `geometry.folder` for a *generated*
  geometry type, which is just a cache location there, not the physics itself -- it does count for
  `geometry.type = "folder"`, where it's the actual mesh being simulated).

## Exit codes for job-script branching

`0` success, `1` a configuration error (`ConfigError`), `2` a runtime/solver failure (e.g. a
blown-up, non-finite transmembrane potential). A parse-time mistake (bad TOML, wrong units, an
unknown key, an unknown `cell.parameters` name) is always caught before any mesh is built or
loaded; a marker-name or fiber-availability mistake (which needs the actual mesh to check) is
caught right after that -- still well before the collective solve loop, but only cheap in wall-time
if the geometry was already built/cached ahead of time (see [Mesh once, run
many](#mesh-once-run-many)). Either way, a job script can branch on the exit code directly:

```bash
srun beat run config.toml --restart
case $? in
  0) echo "done" ;;
  1) echo "config error, not retrying" >&2; exit 1 ;;
  2) echo "solver failed, resubmitting is unlikely to help without changing the config" >&2; exit 1 ;;
esac
```

`output/run.json` (`status: "running"|"finished"|"failed"`, plus `n_ranks`, wall-clock start/end,
and -- on a runtime failure -- the exception) is the same information in a form a later step or a
monitoring script can read back out of the output folder itself.

## Env var overrides with scheduler-provided variables

`BEAT_<SECTION>__<KEY>` env vars (matched case-insensitively against the schema, e.g.
`BEAT_SOLVER__DT`) sit below `--set` and above the TOML file in precedence -- convenient for
piping a scheduler's own environment straight through without constructing a `--set` string in the
job script:

```bash
export BEAT_OUTPUT__FOLDER="runs/${SLURM_ARRAY_TASK_ID}"
export BEAT_SOLVER__DT="${DT_MS} ms"
srun beat run config.toml
```

(Note `BEAT_OUTPUT__FOLDER` here still resolves against the config file's directory like any other
in-file path, since it's not the dedicated `--output-folder` flag -- use that flag instead if you
need current-working-directory-relative resolution, as in the array-job example above.)

## Solver advice for large meshes

The default `solver.pde.linear_solver = "direct"` uses a direct (MUMPS) factorization of the PDE
system -- fine for the small-to-moderate meshes in the templates, but its memory and time cost
grows faster than linearly with problem size. For a large mesh (a fine realistic ventricular
geometry, or a sweep run at production resolution), set:

```toml
[solver.pde]
linear_solver = "iterative"
```

which switches to a preconditioned conjugate-gradient solve (`ksp_type = "cg"`,
`pc_type = "hypre"`, `pc_hypre_type = "boomeramg"`) -- algebraic multigrid scales much better across
MPI ranks and mesh sizes than a direct factorization. Fine-tune further with
`--petsc-options`/`solver.petsc_options` if the default iterative tolerances need adjusting for a
particular mesh.
