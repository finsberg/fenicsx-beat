from mpi4py import MPI

import numpy as np
import pytest

from beat.cli.config import Config, ConfigError
from beat.cli.geometry import build_conductivity, build_fibers, build_geometry, get_marker


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
    stamp = (tmp_path / "geo" / "beat_geometry.json").stat().st_mtime_ns
    build_geometry(conf.geometry, MPI.COMM_WORLD)
    assert (tmp_path / "geo" / "beat_geometry.json").stat().st_mtime_ns == stamp
