# The C backend for cell models

`beat` advances the cell model with a function `fun(states, t, parameters, dt)` that acts on a
`(num_states, num_points)` array. `beat.c_backend` turns the C code that `gotranx` generates for a
single cell into such a function: it adds a loop over the points, compiles the result, and loads it
with `ctypes`. The result can be passed as `fun` to `beat.odesolver.DolfinODESolver` and
`beat.odesolver.DolfinMultiODESolver`. See the [demo](../demos/c_backend.py) for a comparison with
NumPy and Numba.

## Usage

From a `gotranx` ODE,

```python
import gotranx
import beat

ode = gotranx.load_ode("tentusscher_panfilov_2006_epi_cell.ode")
fun = beat.c_backend.from_ode(ode, scheme="generalized_rush_larsen")

solver = beat.odesolver.DolfinODESolver(..., fun=fun, ...)
```

or from C code that you already have, for example from `gotranx ode2c`,

```python
fun = beat.c_backend.compile_scheme(c_code, scheme="generalized_rush_larsen")
```

The function also has `state_index(name)`, `parameter_index(name)`, `init_state_values()` and
`init_parameter_values()`, which call the functions of the same name in the generated C.

## Options

| Option | Default | Meaning |
|---|---|---|
| `scheme` | `"generalized_rush_larsen"` | The name of the scheme (and of the C function) |
| `codegen_kwargs` (`from_ode` only) | `None` | Passed on to `gotranx.codegen.CCodeGenerator.scheme` |
| `cache_dir` | `$BEAT_C_CACHE`, else `~/.cache/beat/c_backend` | Where the libraries are stored |
| `cc` | `$CC`, else `cc` | The C compiler |
| `cflags` | `("-O3", "-march=native")` | The compiler flags |
| `openmp` | `False` | Compile with `-fopenmp` and run the loop over the points with OpenMP threads |
| `num_threads` | `OMP_NUM_THREADS`, else the OpenMP default | The number of threads (with `openmp=True`) |
| `comm` | `MPI.COMM_WORLD` | Rank 0 compiles, all ranks load |

The `states` and `parameters` must be C-contiguous float64 arrays. `parameters` is either one
vector for all points or a `(num_parameters, num_points)` array, for parameters that vary in space.
Pass `out=` to write the result into an existing array (it may be `states` itself).

## Parallel runs and clusters

- Rank 0 compiles the library and the other ranks load it from the cache directory, so the cache
  directory must be on a file system that all nodes share. If it is not, all ranks raise an error
  that says so. The home directory usually is shared; otherwise set `BEAT_C_CACHE`.
- The library is compiled inside the job, so `-march=native` targets the compute nodes. With a
  native flag the cache key includes the CPU features the compiler detects (its predefined macros),
  so a library built on a node with a different CPU (e.g. a login node with AVX-512) is not reused
  on the compute nodes. Pass `cflags=("-O3",)` for a library that runs anywhere.
- Several jobs can compile the same library into the same cache at the same time.

## OpenMP

With `openmp=True` the loop over the points runs on several threads. The points are handed out
dynamically, in chunks of 256, since schemes that adapt their sub-steps make some points much more
expensive than others. Threads and MPI ranks can be combined: with `R` ranks on a node of `C` cores,
use `num_threads` (or `OMP_NUM_THREADS`) of at most `C / R`, and make sure that the scheduler gives
each rank that many cores (for SLURM, `--cpus-per-task`). `beat.c_backend.openmp_available()` tells
whether the compiler supports OpenMP. Apple's clang does not, without extra setup.

## Requirements

A C compiler. Any working dolfinx installation has one, since FFCx uses it to compile the finite
element kernels. Windows is not supported.
