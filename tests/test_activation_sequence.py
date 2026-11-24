from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import beat
from beat.activation_sequence import (
    DURRER_ROOT_SITES,
    EndocardialActivation,
    HeartFrame,
    RootSite,
)

cardiac_geometries = pytest.importorskip("cardiac_geometries")

# The ellipsoid is in cm; 2 mm/ms
SPEED = 0.2


@pytest.fixture(scope="module")
def geo(tmp_path_factory):
    comm = MPI.COMM_WORLD
    folder = comm.bcast(str(tmp_path_factory.mktemp("biv")) if comm.rank == 0 else None, root=0)
    cardiac_geometries.mesh.biv_ellipsoid(comm=comm, outdir=folder, char_length=0.5)
    return cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=folder)


@pytest.fixture(scope="module")
def act(geo):
    return EndocardialActivation.from_geometry(geo, mesh_unit="cm", conduction_velocity=SPEED)


def owned(u):
    return u.x.array[: u.function_space.dofmap.index_map.size_local]


def longest_endocardial_edge(geo):
    mesh = geo.mesh
    mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
    facets = np.concatenate([geo.ffun.find(geo.markers[k][0]) for k in ("LV", "RV")])
    nodes = dolfinx.mesh.entities_to_geometry(mesh, mesh.topology.dim - 1, facets.astype(np.int32))
    p = mesh.geometry.x[nodes]  # (facet, vertex, xyz)
    edges = np.linalg.norm(p - np.roll(p, 1, axis=1), axis=2)
    return MPI.COMM_WORLD.allreduce(float(edges.max(initial=0.0)), op=MPI.MAX)


def total(form):
    return MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)), op=MPI.SUM)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(surface="LA", height=0.5, position=0.0),
        dict(surface="LV", height=1.5, position=0.0),
        dict(surface="RV", height=0.5, position="lateral"),
    ],
)
def test_root_site_validation(kwargs):
    with pytest.raises(ValueError):
        RootSite(**kwargs)


def test_heart_frame(geo):
    m = geo.markers
    kwargs = dict(lv=m["LV"][0], rv=m["RV"][0], epi=m["EPI"][0], base=m["BASE"][0])
    frame = HeartFrame.from_facet_tags(geo.mesh, geo.ffun, **kwargs)
    # The ellipsoid: base at z = 1, apex towards -z, RV on the +x side of the LV
    assert frame.long_axis @ [0, 0, 1] > 0.9
    assert frame.septal @ [1, 0, 0] > 0.9
    assert np.isclose(frame.long_axis @ frame.septal, 0.0)
    assert np.allclose(np.cross(frame.long_axis, frame.septal), frame.anterior)
    z, angle, _ = frame.coordinates(np.vstack([frame.apex, frame.base_center]))
    assert np.allclose(z, [0.0, 1.0])
    # The point at (height, angle, radius) has these coordinates
    p = frame.point(0.4, 30.0, 1.2)
    assert np.allclose(np.hstack(frame.coordinates(p)), [0.4, 30.0, 1.2])
    # anterior_point fixes the sign of the anterior direction
    for sign in (1, -1):
        f = HeartFrame.from_facet_tags(
            geo.mesh,
            geo.ffun,
            anterior_point=[0, 5 * sign, 0],
            **kwargs,
        )
        assert f.anterior @ [0, sign, 0] > 0.9


def test_heart_frame_axis(geo):
    m = geo.markers
    kwargs = dict(lv=m["LV"][0], rv=m["RV"][0], epi=m["EPI"][0], base=m["BASE"][0])
    V = dolfinx.fem.functionspace(geo.mesh, ("P", 1))
    base = beat.activation_sequence._surface_dofs(V, geo.ffun, [m["BASE"][0]])
    x_base = V.tabulate_dof_coordinates()[base]
    # The base of the ellipsoid is the plane z = 1: by default the long axis is its normal, and
    # the whole base is at height 1
    frame = HeartFrame.from_facet_tags(geo.mesh, geo.ffun, **kwargs)
    assert np.allclose(frame.long_axis, [0, 0, 1])
    assert np.allclose(frame.coordinates(x_base)[0], 1.0)
    # From the apex to the centre of the base the axis leans towards the RV, and the base is
    # higher on the RV side than on the LV side
    tilted = HeartFrame.from_facet_tags(geo.mesh, geo.ffun, axis="apex_to_base", **kwargs)
    assert tilted.long_axis @ [1, 0, 0] > 0.1
    rv_side, lv_side = tilted.coordinates([[3.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])[0]
    assert rv_side > 1.0 > lv_side
    with pytest.raises(ValueError, match="axis"):
        HeartFrame.from_facet_tags(geo.mesh, geo.ffun, axis="long", **kwargs)


def test_activation_times(act, geo):
    t = act.activation_times
    x = t.function_space.tabulate_dof_coordinates()[: len(owned(t))]
    finite = np.isfinite(owned(t))
    assert MPI.COMM_WORLD.allreduce(int(finite.sum()), op=MPI.SUM) > 0
    # Only endocardial vertices below max_height are activated
    height, _, _ = act.frame.coordinates(x)
    assert (height[finite] <= act.max_height).all()
    endo = beat.activation_sequence._surface_dofs(
        t.function_space,
        geo.ffun,
        [geo.markers["LV"][0], geo.markers["RV"][0]],
    )
    assert np.isin(np.flatnonzero(finite), endo).all()
    # No vertex is activated much earlier than the straight-line bound from the root sites. The
    # bound is the minimum of one cone per root site, which is not convex: on an edge across the
    # line where two fronts meet, the linear interpolation of the eikonal update dips below it
    # by a fraction of h / SPEED (the bound is exact for one root site, see below)
    roots = list(act.roots.values())
    bound = np.min([r["delay"] + np.linalg.norm(x - r["x"], axis=1) / SPEED for r in roots], axis=0)
    h_max = longest_endocardial_edge(geo)
    assert (owned(t)[finite] >= bound[finite] - 0.25 * h_max / SPEED).all()
    for r in roots:
        at_root = np.linalg.norm(x - r["x"], axis=1) < 1e-10
        assert np.allclose(owned(t)[at_root], r["delay"])
    assert sorted(act.roots) == sorted(DURRER_ROOT_SITES)
    assert {r["delay"] for r in roots} == {0.0, 5.0}


@pytest.mark.parametrize("name", ["LV mid-septal", "RV free wall"])
def test_activation_times_single_root(geo, name):
    # From one root site the distance is convex and a geodesic is at least as long as the
    # straight line, so no vertex is activated earlier than the straight-line bound
    act = EndocardialActivation.from_geometry(
        geo,
        mesh_unit="cm",
        conduction_velocity=SPEED,
        root_sites={name: DURRER_ROOT_SITES[name]},
    )
    t = act.activation_times
    x = t.function_space.tabulate_dof_coordinates()[: len(owned(t))]
    finite = np.isfinite(owned(t))
    assert MPI.COMM_WORLD.allreduce(int(finite.sum()), op=MPI.SUM) > 0
    r = act.roots[name]
    bound = r["delay"] + np.linalg.norm(x - r["x"], axis=1) / SPEED
    assert (owned(t)[finite] >= bound[finite] - 1e-8).all()


def test_cell_times_are_the_earliest_endocardial_vertex(act):
    summary = act.summary()
    assert summary["first_stimulus"] == 0.0
    t_vertex = act.activation_times.x.array
    tmax = MPI.COMM_WORLD.allreduce(t_vertex[np.isfinite(t_vertex)].max(initial=0.0), op=MPI.MAX)
    assert 5.0 < summary["last_stimulus"] <= tmax
    assert summary["num_stimulated_cells"] > 0


def test_delay_scale_and_speed(geo, act):
    slow = EndocardialActivation.from_geometry(
        geo,
        mesh_unit="cm",
        conduction_velocity=beat.units.ureg.Quantity(1.0, "mm/ms"),
        delay_scale=2.0,
    )
    assert {r["delay"] for r in slow.roots.values()} == {0.0, 10.0}
    assert slow.summary()["last_stimulus"] > act.summary()["last_stimulus"]


def test_unreached_surface_is_not_stimulated(geo):
    lv_only = {k: v for k, v in DURRER_ROOT_SITES.items() if v.surface == "LV"}
    with pytest.warns(RuntimeWarning, match="not reached"):
        a = EndocardialActivation.from_geometry(
            geo,
            mesh_unit="cm",
            conduction_velocity=SPEED,
            root_sites=lv_only,
        )
    rv = beat.activation_sequence._surface_dofs(
        a.activation_times.function_space,
        geo.ffun,
        [geo.markers["RV"][0]],
    )
    lv = beat.activation_sequence._surface_dofs(
        a.activation_times.function_space,
        geo.ffun,
        [geo.markers["LV"][0]],
    )
    rv_only = np.setdiff1d(rv, lv)
    assert not np.isfinite(a.activation_times.x.array[rv_only]).any()


def test_from_geometry_needs_markers(geo):
    class Geo:
        mesh = geo.mesh
        ffun = geo.ffun
        markers = {k: v for k, v in geo.markers.items() if k != "BASE"}

    with pytest.raises(ValueError, match="base marker"):
        EndocardialActivation.from_geometry(Geo())


@pytest.mark.parametrize("dt", [0.05, 0.3])
def test_stimulus_charge_does_not_depend_on_dt(act, geo, dt):
    time = dolfinx.fem.Constant(geo.mesh, 0.0)
    stimulus = act.stimulus(time, dt, amplitude=40.0, duration=2.0, start=1.0)
    assert isinstance(stimulus, beat.Stimulus)
    form = dolfinx.fem.form(stimulus.expr * stimulus.dz)
    charge = 0.0
    for k in range(round(40.0 / dt)):
        time.value = (k + 0.5) * dt
        charge += dt * MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)
    stimulated = ufl.conditional(ufl.lt(act.cell_activation_times, 1e10), 1.0, 0.0)
    # 40 uA/cm^2 on a mesh in cm, for 2 ms, in every stimulated cell
    assert np.isclose(charge, 40.0 * 2.0 * total(stimulated * ufl.dx), rtol=1e-10)


def test_stimulus_amplitude_units(act, geo):
    ureg = beat.units.ureg
    time = dolfinx.fem.Constant(geo.mesh, 0.0)

    # A cell stimulated from t = 0 on each rank that has one
    W = act.cell_activation_times.function_space
    num_cells = geo.mesh.topology.index_map(3).size_local
    t_cell = act.cell_activation_times.x.array[W.dofmap.list[:num_cells, 0]]
    first = np.flatnonzero(t_cell == 0.0)

    def at_first(**kwargs):
        s = act.stimulus(time, 0.1, **kwargs)
        time.value = 0.5
        values = dolfinx.fem.Expression(s.expr, np.zeros((1, 3))).eval(
            geo.mesh,
            first[:1].astype(np.int32),
        )
        return MPI.COMM_WORLD.allreduce(float(np.max(values)) if len(values) else 0.0, op=MPI.MAX)

    assert np.isclose(at_first(amplitude=40.0), 40.0)
    assert np.isclose(at_first(amplitude=ureg.Quantity(0.4, "uA/mm**2")), 40.0)
    assert np.isclose(at_first(amplitude=ureg.Quantity(56000.0, "uA/cm**3"), chi=1400.0), 40.0)
    with pytest.raises(ValueError, match="chi"):
        act.stimulus(time, 0.1, amplitude=ureg.Quantity(56000.0, "uA/cm**3"))
    with pytest.raises(ValueError, match="duration"):
        act.stimulus(time, 0.1, duration=0.0)


def test_septal_and_free_wall_sites(act):
    frame = act.frame
    _, angle_s, radius_s = frame.coordinates(act.roots["RV septal"]["x"])
    _, angle_f, radius_f = frame.coordinates(act.roots["RV free wall"]["x"])
    # On the septal direction, not on the junctions of the RV with the LV
    assert abs(angle_s[0]) < 15.0
    assert abs(angle_f[0]) < 15.0
    assert radius_s[0] < radius_f[0]


def test_vertex_stimulates_more_cells_at_the_same_times(geo, act):
    vertex = EndocardialActivation.from_geometry(
        geo,
        mesh_unit="cm",
        conduction_velocity=SPEED,
        stimulated_cells="vertex",
    )
    t_facet = act.cell_activation_times.x.array
    t_vertex = vertex.cell_activation_times.x.array
    assert np.isfinite(t_vertex[np.isfinite(t_facet)]).all()
    assert vertex.summary()["num_stimulated_cells"] > act.summary()["num_stimulated_cells"]
    # A cell is stimulated at its earliest endocardial vertex, which is never later than the
    # earliest vertex of its endocardial facets
    both = np.isfinite(t_facet)
    assert (t_vertex[both] <= t_facet[both] + 1e-12).all()
    with pytest.raises(ValueError, match="stimulated_cells"):
        EndocardialActivation.from_geometry(geo, mesh_unit="cm", stimulated_cells="edge")
