"""Every template runs end-to-end when shrunk to a coarse mesh and a few time steps."""

import os
import shutil

from mpi4py import MPI

import pytest

from beat.cli import TEMPLATES_DIR, _available_templates
from beat.cli.overrides import load_config
from beat.cli.runner import run

# Irksome must be imported before any UFL form is built in this process (see
# beat.cli.solvers.check_backends_available's docstring); a module-level import (as
# tests/test_irksome_monodomain.py also does) makes that happen at collection time, before any
# test body -- including the non-irksome templates parametrized below, which do build UFL forms
# -- runs. Importing this file on its own (e.g. `pytest tests/test_cli_examples.py`) would
# otherwise fail on irksome_model_gotranx whenever it sorts after another template alphabetically.
try:
    import irksome  # noqa: F401
except ImportError:
    pass

# Per-template overrides that make it tiny. Keys are --set strings.
SHRINK = {
    # box_slab needs n = round(L/dx) >= 1 along every axis; the template's Ly=Lz=0.05 cm means
    # dx must stay <= 0.05 (0.1 would round Ly/dx and Lz/dx down to 0 cells and fail to mesh).
    "slab": ["geometry.dx=0.05"],
    "niederer_benchmark": [
        "geometry.dx=0.5",
        "geometry.lx=2.0",
        "geometry.ly=1.0",
        "geometry.lz=1.0",
        "stimulus.0.max=[0.5, 0.5, 0.5]",
    ],
    "fitzhughnagumo": ["geometry.dx=10.0"],
    "diffusion": ["geometry.dx=0.25"],
    "pvc": ["geometry.dx=0.5"],
    "pace_train": ["geometry.dx=0.5"],
    "lv_endocardial": ["geometry.psize_ref=10.0"],
    "biv_endocardial": ["geometry.char_length=2.0"],
    "ukb_atlas": [
        # 15.0 (the brief's suggested value) makes gmsh fail on this atlas/gmsh version with
        # "PLC Error: A segment and a facet intersect at point" (verified); 8.0 meshes reliably
        # while still being much coarser than the template's default (2.0).
        "geometry.char_length_max=8.0",
        "geometry.char_length_min=8.0",
        "stimulus.0.num_points=10",
    ],
    "irksome_model_gotranx": ["geometry.dx=0.25"],
    "external_operator_gotranx": ["geometry.dx=0.25"],
}
NEEDS = {
    # "slab" is deliberately absent: geometry.type = "box_slab" wraps beat.geometry's own
    # get_3D_slab_mesh/get_3D_slab_microstructure directly (no gmsh/cardiac-geometriesx).
    "lv_endocardial": "cardiac_geometries",
    "biv_endocardial": "cardiac_geometries",
    "ukb_atlas": "ukb",
    "irksome_model_gotranx": "irksome",
    "external_operator_gotranx": "dolfinx_external_operator",
}
SERIAL_ONLY = {"lv_endocardial", "biv_endocardial", "ukb_atlas"}  # gmsh meshing


def test_all_demo_templates_exist():
    assert set(_available_templates()) == set(SHRINK)


@pytest.mark.parametrize("name", sorted(SHRINK))
def test_template_runs(name, tmp_path):
    if name in NEEDS:
        pytest.importorskip(NEEDS[name])
    if name == "ukb_atlas" and not os.environ.get("BEAT_TEST_NETWORK"):
        pytest.skip("ukb_atlas needs network access to fetch the atlas; set BEAT_TEST_NETWORK=1")
    if name in SERIAL_ONLY and MPI.COMM_WORLD.size > 1:
        pytest.skip("mesh generation tested in serial")
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    if MPI.COMM_WORLD.rank == 0:
        shutil.copytree(TEMPLATES_DIR / name, tmp_path / name)
    MPI.COMM_WORLD.barrier()
    sets = [*SHRINK[name], 'output.save_every="0.1 ms"', *_short_run(tmp_path / name)]
    conf = load_config(tmp_path / name / "config.toml", sets=sets, environ={})
    conf.cell.steady_state = None  # never pre-pace in CI
    out = run(conf)
    assert (out / "results.bp").exists()


def _short_run(folder) -> list[str]:
    """Shrink the run length, keeping the template's choice of end_time vs num_beats/BCL
    (exactly one of them is allowed)."""
    import toml

    solver = toml.loads((folder / "config.toml").read_text()).get("solver", {})
    if "num_beats" in solver:
        return ["solver.num_beats=1", 'solver.BCL="0.2 ms"']
    return ['solver.end_time="0.2 ms"']
