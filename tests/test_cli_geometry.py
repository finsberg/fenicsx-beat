import json

from mpi4py import MPI

import numpy as np
import pytest

from beat.cli.config import Config, ConfigError
from beat.cli.geometry import (
    _install_generated,
    build_conductivity,
    build_fibers,
    build_geometry,
    cache_folder,
    get_marker,
)


def geom(tmp_path, **g):
    return Config.model_validate(
        {
            "geometry": g,
            "cell": {"ode_file": "m.ode"},
            "solver": {"dt": "0.1 ms", "end_time": "1 ms"},
        },
    )


def test_interval_has_end_markers(tmp_path):
    conf = geom(tmp_path, type="interval", length=1.0, dx=0.1)
    geo = build_geometry(conf.geometry, MPI.COMM_WORLD)
    assert geo.mesh.topology.dim == 1
    assert set(geo.markers) == {"X0", "X1"}
    n = geo.mesh.comm.allreduce(len(geo.ffun.find(geo.markers["X0"][0])), op=MPI.SUM)
    assert n == 1


def test_rectangle_and_box_slab_markers(tmp_path):
    rect = build_geometry(
        geom(tmp_path, type="rectangle", lx=1, ly=0.5, dx=0.25).geometry,
        MPI.COMM_WORLD,
    )
    assert set(rect.markers) == {"X0", "X1", "Y0", "Y1"}
    box = build_geometry(
        geom(tmp_path, type="box_slab", lx=1, ly=0.5, lz=0.5, dx=0.25).geometry,
        MPI.COMM_WORLD,
    )
    assert set(box.markers) == {"X0", "X1", "Y0", "Y1", "Z0", "Z1"}
    assert box.f0 is not None


def test_get_marker_unknown_lists_available(tmp_path):
    geo = build_geometry(
        geom(tmp_path, type="interval", length=1.0, dx=0.5).geometry,
        MPI.COMM_WORLD,
    )
    with pytest.raises(ConfigError, match="X0"):
        get_marker(geo, "ENDO")


def test_from_geometry_fibers_requires_f0(tmp_path):
    conf = geom(tmp_path, type="rectangle", dx=0.5, fibers={"type": "from_geometry"})
    geo = build_geometry(conf.geometry, MPI.COMM_WORLD)
    with pytest.raises(ConfigError, match="fiber"):
        build_fibers(geo, conf.geometry.fibers)


def test_axis_fibers_and_isotropic_conductivity(tmp_path):
    conf = geom(tmp_path, type="rectangle", dx=0.5, fibers={"type": "axis", "direction": "y"})
    geo = build_geometry(conf.geometry, MPI.COMM_WORLD)
    f0 = build_fibers(geo, conf.geometry.fibers)
    assert np.allclose(f0.value, [0.0, 1.0])
    iso = geom(tmp_path, type="rectangle", dx=0.5)
    M = build_conductivity(iso.ep, geo, iso.geometry.fibers)
    assert isinstance(M, float) and M > 0


@pytest.mark.skip_in_parallel
def test_generated_slab_is_cached(tmp_path):
    pytest.importorskip("cardiac_geometries")
    conf = geom(
        tmp_path,
        type="slab",
        lx=1.0,
        ly=0.3,
        lz=0.3,
        dx=0.15,
        folder=str(tmp_path / "geo"),
    )
    geo = build_geometry(conf.geometry, MPI.COMM_WORLD)
    assert geo.f0 is not None and "X0" in geo.markers
    sub = cache_folder(conf.geometry)
    assert sub.parent == tmp_path / "geo"
    stamp = (sub / "beat_geometry.json").stat().st_mtime_ns
    build_geometry(conf.geometry, MPI.COMM_WORLD)
    assert (sub / "beat_geometry.json").stat().st_mtime_ns == stamp


@pytest.mark.skip_in_parallel
def test_generated_slab_recovers_from_corrupt_cache_metadata(tmp_path):
    pytest.importorskip("cardiac_geometries")
    conf = geom(
        tmp_path,
        type="slab",
        lx=1.0,
        ly=0.3,
        lz=0.3,
        dx=0.15,
        folder=str(tmp_path / "geo"),
    )
    build_geometry(conf.geometry, MPI.COMM_WORLD)
    meta = cache_folder(conf.geometry) / "beat_geometry.json"
    meta.write_text("{not valid json")

    # A corrupt/truncated metadata file (e.g. left behind by a killed job) must be treated as
    # "needs regeneration", not raise (which would deadlock the other ranks under MPI).
    geo = build_geometry(conf.geometry, MPI.COMM_WORLD)
    assert geo.f0 is not None and "X0" in geo.markers
    # Regeneration must have replaced the corrupt file with a valid one.
    assert json.loads(meta.read_text())["hash"]


def _slab(tmp_path, folder, dx=0.15):
    return geom(tmp_path, type="slab", lx=1.0, ly=0.3, lz=0.3, dx=dx, folder=str(folder))


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("sub", ["geo", "."])
def test_regeneration_never_deletes_user_files(tmp_path, sub):
    """A stale/changed geometry cache must never delete geometry.folder or anything beat did
    not create there -- geometry.folder may well be the config's own directory (folder=".").
    """
    pytest.importorskip("cardiac_geometries")
    folder = (tmp_path / sub).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.toml").write_text("user config")
    (tmp_path / "my_model.ode").write_text("user ode")
    (folder / "notes.txt").write_text("keep me")
    a = _slab(tmp_path, folder, dx=0.15)
    build_geometry(a.geometry, MPI.COMM_WORLD)
    b = _slab(tmp_path, folder, dx=0.3)  # param change -> a new, separate cache entry
    geo = build_geometry(b.geometry, MPI.COMM_WORLD)
    assert "X0" in geo.markers
    for f in (tmp_path / "config.toml", tmp_path / "my_model.ode", folder / "notes.txt"):
        assert f.is_file(), f
    sub_a, sub_b = cache_folder(a.geometry), cache_folder(b.geometry)
    assert sub_a != sub_b and sub_a.parent == sub_b.parent == folder
    assert (sub_a / "beat_geometry.json").is_file()
    assert (sub_b / "beat_geometry.json").is_file()
    # Corrupt cache entry: only that beat-created subfolder is regenerated.
    (sub_a / "beat_geometry.json").write_text("{corrupt")
    build_geometry(a.geometry, MPI.COMM_WORLD)
    assert json.loads((sub_a / "beat_geometry.json").read_text())["hash"]
    assert (folder / "notes.txt").is_file() and (sub_b / "beat_geometry.json").is_file()


def test_install_generated_reuses_entry_completed_by_another_job(tmp_path):
    """Two jobs generating the same hash concurrently: the second to finish keeps the first's
    complete entry and discards its own temporary folder."""
    if MPI.COMM_WORLD.rank != 0:
        return
    target = tmp_path / "abc"
    target.mkdir()
    (target / "beat_geometry.json").write_text(json.dumps({"type": "slab", "hash": "h"}))
    (target / "mesh.txt").write_text("first job")
    tmp = tmp_path / ".tmp-abc-1"
    tmp.mkdir()
    (tmp / "mesh.txt").write_text("second job")
    _install_generated(tmp, target, "slab", "h")
    assert not tmp.exists()
    assert (target / "mesh.txt").read_text() == "first job"
    # A stale (hash-mismatch) entry is replaced.
    tmp.mkdir()
    (tmp / "mesh.txt").write_text("fresh")
    _install_generated(tmp, target, "slab", "h2")
    assert (target / "mesh.txt").read_text() == "fresh"
    assert json.loads((target / "beat_geometry.json").read_text())["hash"] == "h2"
    assert not tmp.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["abc"]
