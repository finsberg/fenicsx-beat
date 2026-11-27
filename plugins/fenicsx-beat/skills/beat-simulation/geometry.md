# Geometry: built-in shapes, the user's own mesh, markers and fibres

## Built-in geometries (CLI `geometry.type`)

| `geometry.type` | Shape | Built by | Main parameters |
|---|---|---|---|
| `interval` | 1D cable | dolfinx | `length`, `dx` |
| `rectangle` | 2D sheet | dolfinx | `lx`, `ly`, `dx` |
| `box_slab` | 3D box, structured tetrahedra | beat (in memory, no gmsh) | `lx`, `ly`, `lz`, `dx` |
| `slab` | 3D slab with rule-based fibres | cardiac-geometriesx (gmsh) | `lx`, `ly`, `lz`, `dx`, fibre angles |
| `lv_ellipsoid` | idealized left ventricle | cardiac-geometriesx (gmsh) | radii, `psize_ref`, fibre angles |
| `biv_ellipsoid` | idealized bi-ventricle | cardiac-geometriesx (gmsh) + fenicsx-ldrb fibres | `char_length`, fibre angles |
| `ukb` | UK Biobank atlas heart | ukb-atlas + cardiac-geometriesx (downloads the atlas once) | `mode`, `std`, `case`, `char_length_max`/`char_length_min` |
| `folder` | a mesh already on disk in cardiac-geometriesx's folder format | — | `folder` |

- `geometry.unit` is the length unit of the mesh coordinates (e.g. `"mm"`, `"cm"`); everything else
  (conductivities, $\chi$, stimulus boxes, electrode positions) is interpreted against it.
- `geometry.fibers` chooses the fibre field: from the geometry (rule-based / LDRB for the
  cardiac shapes), along an axis, or isotropic.
- gmsh-based types are generated once and cached in `geometry.folder/<hash>/`; run `beat geometry
  config.toml` serially before a parallel run.
- In Python: `beat.geometry.get_2D_slab_geometry`, `beat.geometry.get_3D_slab_geometry` for
  slabs; `cardiac_geometries` (cardiac-geometriesx) for ellipsoids and the atlas — see the
  `lv_endocardial`, `ukb_atlas` and `durrer_activation` demos.

## The user's own mesh

Do this as soon as the user says they have a mesh, before choosing anything else.

### 1. Can it be read?

Ask for the file (path and format, which software produced it, the length unit). Then try to read it
with io4dolfinx, choosing the backend from the extension:

| Extension | io4dolfinx backend | Notes |
|---|---|---|
| `.bp` | `adios2` | io4dolfinx / dolfinx checkpoints |
| `.h5` | `h5py` (io4dolfinx checkpoints) or `vtkhdf` (VTKHDF files) | try both |
| `.vtkhdf`, `.hdf` | `vtkhdf` | |
| `.xdmf` (+ `.h5`) | `xdmf` | dolfinx XDMF meshes |
| `.exo`, `.e` | `exodus` | needs `netCDF4` |
| `.vtu`, `.vtk`, other VTK formats | `pyvista` | read through pyvista, serial read |
| `.msh` (gmsh) | — | not io4dolfinx: use `dolfinx.io.gmsh.read_from_msh` (`dolfinx.io.gmshio` in older dolfinx) |

Check the backend names against the installed version first
(`python -c "import io4dolfinx.backends as b, pkgutil; print([m.name for m in pkgutil.iter_modules(b.__path__)])"`)
and the signature (`help(io4dolfinx.read_mesh)`). A check script (write it to a scratch file, run it
serially, then with `mpirun -n 2`):

```python
import sys
from mpi4py import MPI

import io4dolfinx

path, backend = sys.argv[1], sys.argv[2]
mesh = io4dolfinx.read_mesh(path, MPI.COMM_WORLD, backend=backend)
tdim = mesh.topology.dim
num_cells = mesh.topology.index_map(tdim).size_global
num_vertices = mesh.geometry.index_map().size_global
x = mesh.geometry.x
lo = MPI.COMM_WORLD.allreduce(x.min(axis=0), op=MPI.MIN)
hi = MPI.COMM_WORLD.allreduce(x.max(axis=0), op=MPI.MAX)
if MPI.COMM_WORLD.rank == 0:
    print("cell type:", mesh.topology.cell_name(), "dim:", tdim)
    print("cells:", num_cells, "vertices:", num_vertices)
    print("bounding box:", lo, hi)  # does the size match the stated unit?
```

Then look for markers: `io4dolfinx.read_meshtags(path, mesh, meshtag_name=..., backend=...)` with the
tag names the file has (ask, or inspect the file, e.g. with `h5py` or pyvista). Report cell type,
size, bounding box (a heart in mm is ~100 across; in cm ~10; in m ~0.1), and which markers exist.

### 2. If it cannot be read

Say so plainly. Ask the user to open a feature request at
https://github.com/scientificcomputing/io4dolfinx/issues and draft the issue text for them:
the file format and the software that wrote it, a small example file (or how to make one), the
command that failed, and the full error. In the meantime offer a conversion route:

- meshio: `meshio convert heart.ext heart.xdmf` (then the `xdmf` backend), for formats meshio reads;
- gmsh: open/import and export `.msh`, then `dolfinx.io.gmsh.read_from_msh`;
- the producing software's own export to VTU/XDMF.

### 3. What the problem needs from the mesh

- **Stimulus**: markers for the stimulated region (e.g. LV/RV endocardium facets), or a box/point in
  mesh coordinates.
- **Fibres**: from the file (a vector field), or computed with LDRB (`ldrb`, fenicsx-ldrb), which
  needs endocardium/epicardium/base markers. Without fibres, conduction is isotropic — say so.
- **Cell layers / regions**: endo/mid/epi via `beat.utils.expand_layer` /
  `beat.utils.expand_layer_biv` (needs endocardium/epicardium markers), or cell tags already in the
  file.
- **Resolution**: compare the edge length with [numerics.md](numerics.md) (mesh size); a coarse
  clinical mesh may need refinement, calibrated conductivities or reaction-eikonal.

### 4. Getting it into the CLI

The CLI loads own meshes with `geometry.type = "folder"`: a folder in cardiac-geometriesx's format
(mesh, tags, a `markers.json` mapping marker names to `[tag, dim]`, optional fibres), read by
`cardiac_geometries.geometry.Geometry.from_folder`. Build it in Python from the mesh read above:
construct `cardiac_geometries.geometry.Geometry(mesh=..., markers=..., ffun=..., f0=..., ...)` and
write it with its `save_folder(folder)` method (check `help(cardiac_geometries.geometry.Geometry)` in
the installed version), then load it back with `Geometry.from_folder` to confirm before pointing
`geometry.folder` at it. In the Python route, use the mesh, tags and fibres directly.
