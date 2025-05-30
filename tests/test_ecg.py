from mpi4py import MPI

import dolfinx
import numpy as np
import ufl

import beat


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

    v.interpolate(dolfinx.fem.Expression(v_expr, V.element.interpolation_points()))
    ecg.solve()
    value_p1 = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(p1_ecg), op=MPI.SUM)
    value_p2 = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(p2_ecg), op=MPI.SUM)
    value_p3 = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(p3_ecg), op=MPI.SUM)

    # The solution should be symmetric with respect to the line x=0.5
    assert np.isclose(value_p1, value_p3)
    # Points further away from the source should have a smaller absolute potential
    assert abs(value_p2) < abs(value_p1)


def test_12_leads_ecg():
    N = 10
    x = np.ones(N)
    la = 1.2
    ra = 4.5
    ll = 3.6
    v1 = 1.0
    v2 = 2.0
    v3 = 3.0
    v4 = 4.0
    v5 = 5.0
    v6 = 6.0

    Vw = np.mean([la, ra, ll])

    ecg = beat.ecg.Leads12(
        LA=la * x,
        RA=ra * x,
        LL=ll * x,
        V1=v1 * x,
        V2=v2 * x,
        V3=v3 * x,
        V4=v4 * x,
        V5=v5 * x,
        V6=v6 * x,
    )

    for i, vi in enumerate([v1, v2, v3, v4, v5, v6], start=1):
        assert np.allclose(getattr(ecg, f"V{i}_"), vi - Vw)


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
