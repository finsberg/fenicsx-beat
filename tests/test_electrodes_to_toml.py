"""scripts/electrodes_to_toml.py: header-less electrode CSV to [postprocess.ecg]."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import toml
from cli_helpers import minimal_config_dict

from beat.cli.overrides import load_config

SCRIPT = Path(__file__).parents[1] / "scripts" / "electrodes_to_toml.py"
ALYA = ["LA", "RA", "LL", "RL", "V1", "V2", "V3", "V4", "V5", "V6"]

pytestmark = pytest.mark.skip_in_parallel


def run_script(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
    )


def load_with(tmp_path, converted: str):
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps(minimal_config_dict(tmp_path)) + "\n" + converted)
    return load_config(path, environ={})


def test_converts_legacy_simcardems_file_bit_for_bit(tmp_path):
    rows = np.random.default_rng(1).normal(scale=10.0, size=(10, 3))
    csv = tmp_path / "electrodes.csv"
    np.savetxt(csv, rows, fmt="%.18e", delimiter=",")
    res = run_script(csv, "--unit", "cm")
    assert res.returncode == 0, res.stderr
    conf = load_with(tmp_path, res.stdout)
    assert conf.postprocess.ecg.electrodes == {n: r.tolist() for n, r in zip(ALYA, rows)}
    assert list(conf.postprocess.ecg.electrodes) == ALYA
    assert conf.postprocess.ecg.unit == "cm"


def test_names_that_need_quoting(tmp_path):
    csv = tmp_path / "e.csv"
    csv.write_text("1.0,2.0,3.0\n4.0,5.0,6.0\n")
    out = tmp_path / "ecg.toml"
    res = run_script(csv, "--names", "V1 chest,a.b", "-o", out)
    assert res.returncode == 0, res.stderr
    assert res.stdout == ""
    conf = load_with(tmp_path, out.read_text())
    assert conf.postprocess.ecg.electrodes == {"V1 chest": [1.0, 2.0, 3.0], "a.b": [4.0, 5.0, 6.0]}
    assert conf.postprocess.ecg.unit is None  # no unit written: geometry.unit applies


def test_two_columns(tmp_path):
    csv = tmp_path / "e.csv"
    csv.write_text("1.0,2.0\n")
    res = run_script(csv, "--names", "A")
    assert res.returncode == 0, res.stderr
    assert load_with(tmp_path, res.stdout).postprocess.ecg.electrodes == {"A": [1.0, 2.0]}


def test_refuses_row_count(tmp_path):
    csv = tmp_path / "nine.csv"
    np.savetxt(csv, np.ones((9, 3)), delimiter=",")
    res = run_script(csv)
    assert res.returncode == 1
    assert "nine.csv" in res.stderr


def test_refuses_uneven_rows(tmp_path):
    csv = tmp_path / "uneven.csv"
    csv.write_text("1,2,3\n4,5\n")
    res = run_script(csv, "--names", "A,B")
    assert res.returncode == 1
    assert "uneven.csv" in res.stderr


def test_refuses_width(tmp_path):
    csv = tmp_path / "wide.csv"
    csv.write_text("1,2,3,4\n")
    res = run_script(csv, "--names", "A")
    assert res.returncode == 1
    assert "wide.csv" in res.stderr


def test_refuses_repeated_name(tmp_path):
    csv = tmp_path / "dup.csv"
    csv.write_text("1,2,3\n4,5,6\n")
    res = run_script(csv, "--names", "A,A")
    assert res.returncode == 1
    assert "dup.csv" in res.stderr
