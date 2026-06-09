import json
import logging
import time
from unittest.mock import MagicMock

from mpi4py import MPI

from beat.telemetry import NullMonitor, PerformanceMonitor


def test_null_monitor():
    """Test that the NullMonitor behaves safely without raising errors."""
    monitor = NullMonitor()

    # Should yield safely
    with monitor.track_time("test_event"):
        pass

    # Should accept dummy KSP / None
    monitor.record_ksp(None)

    # Should accept float time steps
    monitor.advance_step(0.0, 0.1)


def test_performance_monitor_tracking():
    """Test that the context manager correctly accumulates time."""
    comm = MPI.COMM_WORLD
    monitor = PerformanceMonitor(comm=comm)

    with monitor.track_time("dummy_work"):
        time.sleep(0.01)

    assert "dummy_work" in monitor.timings
    assert monitor.timings["dummy_work"] >= 0.01

    # Test accumulation
    with monitor.track_time("dummy_work"):
        time.sleep(0.01)

    assert monitor.timings["dummy_work"] >= 0.02


def test_performance_monitor_record_ksp():
    """Test that KSP iterations and residuals are accumulated correctly."""
    comm = MPI.COMM_WORLD
    monitor = PerformanceMonitor(comm=comm)

    # Mock PETSc.KSP to avoid needing a full linear system setup
    mock_ksp = MagicMock()
    mock_ksp.getIterationNumber.return_value = 5
    mock_ksp.getResidualNorm.return_value = 1e-6
    mock_ksp.getConvergedReason.return_value = 2  # e.g., KSP_CONVERGED_RTOL

    monitor.record_ksp(mock_ksp)

    assert monitor.ksp_last_iterations == 5
    assert monitor.ksp_total_iterations == 5
    assert monitor.ksp_max_iterations == 5
    assert monitor.ksp_last_residual_norm == 1e-6
    assert monitor.ksp_last_converged_reason == 2

    # Second call to test max and total accumulation
    mock_ksp.getIterationNumber.return_value = 7
    monitor.record_ksp(mock_ksp)

    assert monitor.ksp_last_iterations == 7
    assert monitor.ksp_total_iterations == 12
    assert monitor.ksp_max_iterations == 7


def test_performance_monitor_logging(caplog):
    """Test that the monitor only logs at the specified frequency."""
    caplog.set_level(logging.INFO)
    comm = MPI.COMM_WORLD

    # Log every 2 steps
    monitor = PerformanceMonitor(log_frequency=2, comm=comm)

    with monitor.track_time("step_time"):
        pass

    monitor.advance_step(0.0, 0.1)
    assert len(caplog.records) == 0  # Shouldn't log on step 1

    monitor.advance_step(0.1, 0.2)
    assert len(caplog.records) == 1  # Should log on step 2
    assert "PDE step timing step=2" in caplog.records[0].message
    assert "step_time=" in caplog.records[0].message


def test_performance_monitor_save_summary(tmp_path):
    """Test that the monitor saves valid JSON to disk (and only on Rank 0)."""
    comm = MPI.COMM_WORLD
    monitor = PerformanceMonitor(comm=comm)

    # Manually populate some mock metrics
    monitor.step_counter = 10
    monitor.ksp_total_iterations = 45
    monitor.timings["test_metric"] = 1.234

    filepath = tmp_path / "summary.json"
    monitor.save_summary(filepath)

    if comm.rank == 0:
        assert filepath.exists()
        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["total_steps"] == 10
        assert data["ksp"]["total_iterations"] == 45
        assert "test_metric" in data["timings"]
        assert data["timings"]["test_metric"] == 1.234
    else:
        # On other MPI ranks, it should not save the file
        assert not filepath.exists()


def test_performance_monitor_display(capsys):
    """Test that the display outputs formatted tables to stdout (and only on Rank 0)."""
    comm = MPI.COMM_WORLD
    monitor = PerformanceMonitor(comm=comm)

    # Manually populate some mock metrics
    monitor.step_counter = 5
    monitor.timings["fast_op"] = 0.1
    monitor.timings["slow_op"] = 5.0

    monitor.display_summary()

    captured = capsys.readouterr()

    if comm.rank == 0:
        assert "PERFORMANCE SUMMARY" in captured.out
        assert "Total Steps:           5" in captured.out
        assert "slow_op" in captured.out
        assert "fast_op" in captured.out

        # Ensure that it sorted the outputs (slow_op should print before fast_op)
        assert captured.out.find("slow_op") < captured.out.find("fast_op")
    else:
        # On other MPI ranks, standard output should be empty
        assert captured.out == ""
