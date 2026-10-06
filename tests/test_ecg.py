from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl
from packaging.version import Version

import beat

_dolfinx_version = Version(dolfinx.__version__)


def test_ecg():
    N = 5
    M = 1.0
    C_m = 1.0
    sigma_b = 1.0

    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, N, N, dolfinx.cpp.mesh.CellType.triangle)

    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    v = dolfinx.fem.Function(V)

    X = ufl.SpatialCoordinate(mesh)
    v_expr = (X[0] - 0.5) ** 2

    ecg = beat.ECGRecovery(v=v, M=M, C_m=C_m, sigma_b=sigma_b)
    p1 = (1.5, 0.5)
    p1_ecg = ecg.eval(p1)
    p2 = (10.0, 0.5)
    p2_ecg = ecg.eval(p2)
    p3 = (-0.5, 0.5)
    p3_ecg = ecg.eval(p3)
    ecg.solve()

    value = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(p1_ecg), op=MPI.SUM)
    assert np.isclose(value, 0.0)

    v.interpolate(dolfinx.fem.Expression(v_expr, beat.utils.interpolation_points(V)))
    ecg.solve()
    value_p1 = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(p1_ecg), op=MPI.SUM)
    value_p2 = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(p2_ecg), op=MPI.SUM)
    value_p3 = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(p3_ecg), op=MPI.SUM)

    # The solution should be symmetric with respect to the line x=0.5
    assert np.isclose(value_p1, value_p3)
    # Points further away from the source should have a smaller absolute potential
    assert abs(value_p2) < abs(value_p1)


_ELECTRODES = ("LA", "RA", "LL", "RL", "V1", "V2", "V3", "V4", "V5", "V6")
_LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
_NEEDED = {"LA", "RA", "LL", "V1", "V2", "V3", "V4", "V5", "V6"}


@pytest.mark.parametrize("shape", [(), (5,)])
def test_twelve_lead_potential_matches_goldberger_and_wilson(shape):
    rng = np.random.default_rng(1)
    phi = {k: rng.normal(size=shape) for k in _ELECTRODES}
    system = beat.ecg.twelve_lead()
    out = system.leads(phi)
    Vw = (phi["RA"] + phi["LA"] + phi["LL"]) / 3
    expected = {
        "I": phi["LA"] - phi["RA"],
        "II": phi["LL"] - phi["RA"],
        "III": phi["LL"] - phi["LA"],
        "aVR": 1.5 * (phi["RA"] - Vw),
        "aVL": 1.5 * (phi["LA"] - Vw),
        "aVF": 1.5 * (phi["LL"] - Vw),
    }
    for i in range(1, 7):
        expected[f"V{i}"] = phi[f"V{i}"] - Vw
    scale = max(np.max(np.abs(v)) for v in phi.values())
    assert tuple(out) == _LEAD_NAMES
    for k, v in expected.items():
        np.testing.assert_allclose(out[k], v, rtol=1e-14, atol=1e-14 * scale)
    assert system.names == _LEAD_NAMES
    assert set(system.electrodes) == _NEEDED
    assert system.layout is not None
    assert sorted(n for row in system.layout for n in row) == sorted(_LEAD_NAMES)


def test_twelve_lead_position_points_and_leads():
    rng = np.random.default_rng(2)
    x = {k: rng.normal(size=3) for k in _ELECTRODES}
    system = beat.ecg.twelve_lead("position")
    pts = system.points(x)
    np.testing.assert_allclose(pts["WCT_pt"], (x["LA"] + x["RA"] + x["LL"]) / 3, rtol=1e-15)
    np.testing.assert_allclose(pts["LA_LL_mid"], (x["LA"] + x["LL"]) / 2, rtol=1e-15)
    np.testing.assert_allclose(pts["RA_LL_mid"], (x["RA"] + x["LL"]) / 2, rtol=1e-15)
    np.testing.assert_allclose(pts["RA_LA_mid"], (x["RA"] + x["LA"]) / 2, rtol=1e-15)
    phi = {k: rng.normal() for k in pts}
    out = system.leads(phi)
    assert out["aVR"] == phi["RA"] - phi["LA_LL_mid"]
    assert out["aVL"] == phi["LA"] - phi["RA_LL_mid"]
    assert out["aVF"] == phi["LL"] - phi["RA_LA_mid"]
    for i in range(1, 7):
        assert out[f"V{i}"] == phi[f"V{i}"] - phi["WCT_pt"]
    ref = beat.ecg.twelve_lead().leads(phi)
    for k in ("I", "II", "III"):
        assert out[k] == ref[k]
    assert set(system.electrodes) == _NEEDED


def test_lead_system_is_generic():
    system = beat.ecg.LeadSystem((beat.ecg.Lead("X", {"A": 2.0, "B": -1.0}),))
    assert system.leads({"A": 1.0, "B": 3.0}) == {"X": -1.0}


def test_lead_system_names_missing_inputs():
    phi = {k: 1.0 for k in _ELECTRODES if k != "V6"}
    with pytest.raises(ValueError, match="V6"):
        beat.ecg.twelve_lead().leads(phi)
    x = {k: np.zeros(3) for k in _ELECTRODES if k != "LL"}
    with pytest.raises(ValueError, match="LL"):
        beat.ecg.twelve_lead("position").points(x)


def test_qt_interval():
    qrs_peak_time = 200  # ms
    t_peak_offset_ms = 200  # ms
    t_width_ms = 60  # ms
    t, y = beat.ecg.example(
        sampling_rate_hz=1000,
        duration_s=1,
        noise_amplitude=0.0,
        wander_amplitude=0.0,
        heart_rate_bpm=60,
        q_offset_ms=40,
        s_offset_ms=40,
        t_peak_offset_ms=t_peak_offset_ms,
        r_width_ms=20,
        q_width_ms=20,
        s_width_ms=30,
        t_width_ms=t_width_ms,
        qrs_peak_time=qrs_peak_time,
    )

    qt = beat.ecg.qt_interval(t=t, ecg_signal=y)

    # Start index should be close to the QRS peak time
    assert np.isclose(qt.start_index, qrs_peak_time, atol=2)

    # End index should be after t_peak_offset + about 2/3 of the t_width_ms

    assert np.isclose(qt.end_index, qrs_peak_time + t_peak_offset_ms + 2 * t_width_ms / 3, atol=5)

    assert np.isclose(qt.qt_interval, qt.end_index - qt.start_index)

    # import matplotlib.pyplot as plt
    # plt.plot(t, y)
    # plt.plot([t[qt.start_index]], [y[qt.start_index]], "ro", label="QT Interval")
    # plt.plot([t[qt.end_index]], [y[qt.end_index]], "go", label="QT Interval")
    # plt.savefig("ecg_qt_interval.png")


def test_electrode_potentials_one_solve_bit_for_bit():
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 8, 8)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    v = dolfinx.fem.Function(V)
    X = ufl.SpatialCoordinate(mesh)
    v.interpolate(
        dolfinx.fem.Expression(ufl.sin(2 * X[0]) * X[1], beat.utils.interpolation_points(V)),
    )
    recovery = beat.ECGRecovery(v=v)
    points = {"b": (1.5, 0.5), "a": (-0.5, 0.5), "c": (0.5, 2.0)}

    probe = beat.ecg.ElectrodePotentials(recovery, points)
    calls = []
    solve = recovery.solve
    recovery.solve = lambda: (calls.append(1), solve())[1]
    phi = probe()

    assert len(calls) == 1
    assert list(phi) == ["b", "a", "c"]
    for name, p in points.items():
        expected = mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(recovery.eval(p)),
            op=MPI.SUM,
        )
        assert expected == phi[name]
        assert type(phi[name]) is float


@pytest.mark.parametrize(("anisotropic", "C_m"), [(False, 1.0), (True, 2.0)])
def test_recovery_is_the_direct_integral_to_second_order(anisotropic, C_m):
    """beat's recovery against 1/(4 pi sigma_b C_m) int M grad v . r / |r|^3 dx."""
    comm = MPI.COMM_WORLD
    electrodes = {"far": (1.5, 0.5, 0.5), "corner": (-0.5, -0.5, 1.5), "near": (1.1, 0.5, 0.5)}

    def relative_errors(N):
        mesh = dolfinx.mesh.create_unit_cube(comm, N, N, N)
        V = dolfinx.fem.functionspace(mesh, ("P", 1))
        v = dolfinx.fem.Function(V)
        X = ufl.SpatialCoordinate(mesh)
        expr = ufl.sin(2 * X[0]) * X[1] + X[2] ** 2 * X[0]
        v.interpolate(dolfinx.fem.Expression(expr, beat.utils.interpolation_points(V)))
        M = (
            ufl.as_matrix([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]])
            if anisotropic
            else 1.0
        )
        recovery = beat.ECGRecovery(v=v, sigma_b=1.0, C_m=C_m, M=M)
        dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 4})
        flux = M * ufl.grad(v) if anisotropic else ufl.grad(v)
        direct = {}
        for name, p in electrodes.items():
            r = X - dolfinx.fem.Constant(mesh, np.asarray(p, dtype=dolfinx.default_scalar_type))
            direct[name] = dolfinx.fem.form(
                ufl.inner(flux, r) / ufl.dot(r, r) ** 1.5 / (4 * np.pi * C_m) * dx,
            )
        beat_forms = {name: recovery.eval(p) for name, p in electrodes.items()}
        recovery.solve()
        errors = {}
        for name in electrodes:
            b = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(beat_forms[name]), op=MPI.SUM)
            d = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(direct[name]), op=MPI.SUM)
            errors[name] = abs(b - d) / abs(d)
        return errors

    e16 = relative_errors(16)
    e32 = relative_errors(32)
    for name in electrodes:
        order = np.log2(e16[name] / e32[name])
        assert order >= 1.8, f"{name}: order {order:.2f}, errors {e16[name]:.3e} -> {e32[name]:.3e}"
