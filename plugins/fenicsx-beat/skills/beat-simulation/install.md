# Installing dolfinx, beat and the extras

`fenicsx-beat` needs a working FEniCSx/dolfinx (with MPI and PETSc), which pip cannot install on its
own. Present the options below, recommend one for the user's machine, and let them choose. Never
run an install command without the user's approval.

| Option | Best for | Notes |
|---|---|---|
| conda | laptops and workstations (Linux, macOS; Windows via WSL) | one command, includes beat |
| Docker / Apptainer | a quick isolated try, reproducible environments; Apptainer on clusters | beat preinstalled |
| Spack | HPC clusters (system MPI, compilers) | builds from source; slower to set up |
| Debian/Ubuntu packages | Ubuntu users who prefer system packages | then pip-install beat |

## conda (recommended on laptops and workstations)

```bash
conda create -n beat -c conda-forge fenicsx-beat
conda activate beat
```

`fenicsx-beat` on conda-forge pulls in dolfinx. Add `pyvista` (images, movies) and the extras below
as needed. To start from dolfinx alone: `conda install -c conda-forge fenics-dolfinx mpich pyvista`,
then `python3 -m pip install fenicsx-beat`.

## Docker

```bash
docker run -ti -v "$(pwd)":/home/shared -w /home/shared ghcr.io/finsberg/fenicsx-beat:latest
```

The image has dolfinx and beat installed; the `-v` mount makes the current folder visible inside.
For MPI runs inside the container you may need `--shm-size=2g` (or larger).

On HPC clusters Docker is usually not available; use Apptainer (Singularity) with the same image:

```bash
apptainer pull fenicsx-beat.sif docker://ghcr.io/finsberg/fenicsx-beat:latest
srun apptainer exec fenicsx-beat.sif python3 script.py   # or mpirun -n N apptainer exec fenicsx-beat.sif ...
```

MPI across nodes from a container needs an MPI compatible with the host's; ask the cluster's support
or use Spack instead.

## Spack (recommended on HPC clusters)

```bash
. spack/share/spack/setup-env.sh
spack env create beat
spack env activate beat
spack repo add https://github.com/FEniCS/spack-fenics.git
spack repo add https://github.com/scientificcomputing/spack_repos.git
spack add fenicsx-beat
spack concretize
spack install -j 4
```

The `scientificcomputing/spack_repos` repository (https://github.com/scientificcomputing/spack_repos)
provides the `fenicsx-beat` package; the FEniCS repository provides dolfinx. Check the package name
with `spack list beat` and the available versions with `spack info fenicsx-beat` before adding it;
on a cluster, configure the system MPI as an external package first (see the cluster's docs).

## Debian / Ubuntu packages

```bash
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt update
sudo apt install fenicsx
python3 -m pip install fenicsx-beat
```

(Use a virtual environment created with `--system-site-packages` so pip sees the system dolfinx.)

## Extras

Install only what the plan needs:

| Need | Install |
|---|---|
| the `beat` CLI (`beat init/run/post`) | `python3 -m pip install "fenicsx-beat[cli]"` |
| ellipsoid / UK Biobank geometries, fibres (cardiac-geometriesx, ukb-atlas, fenicsx-ldrb), demo dependencies | `python3 -m pip install "fenicsx-beat[demos]"` (the CLI's `[cli]` has cardiac-geometriesx but not ukb-atlas or fenicsx-ldrb) |
| cell models from `.ode`/CellML (gotranx) | included in `[cli]` and `[demos]`; alone: `python3 -m pip install gotranx` |
| images and movies | `python3 -m pip install pyvista` |
| fully implicit Runge–Kutta steps (Irksome) | `python3 -m pip install "irksome[dolfinx] @ git+https://github.com/firedrakeproject/Irksome.git"` — the PyPI release behind `fenicsx-beat[irksome]` imports firedrake and fails |
| ODE step via dolfinx-external-operator | `python3 -m pip install "fenicsx-beat[external_operator]"` |

Check what is installed with `python -c "import gotranx, cardiac_geometries, pyvista"` (one import
per package, to see which one is missing). Full install notes:
https://finsberg.github.io/fenicsx-beat (Install section of the README).
