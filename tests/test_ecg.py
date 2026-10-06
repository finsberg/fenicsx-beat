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
