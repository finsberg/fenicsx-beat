from pathlib import Path

import pytest
import toml
from cli_helpers import minimal_config_dict

from beat.cli.config import ConfigError, ms
from beat.cli.overrides import (
    apply_override,
    dump_config,
    env_overrides,
    load_config,
    parse_petsc_options,
    parse_value,
    physics_hash,
)


@pytest.fixture
def cfg_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps(minimal_config_dict(tmp_path)))
    return path


def test_parse_value_toml_literals():
    assert parse_value("1") == 1
    assert parse_value("0.5") == 0.5
    assert parse_value("true") is True
    assert parse_value('"0.02 ms"') == "0.02 ms"
    assert parse_value("[1, 2]") == [1, 2]
    assert parse_value("0.02 ms") == "0.02 ms"  # bare string fallback


def test_apply_override_nested_and_list_index():
    data = {"stimulus": [{"start": "0 ms"}]}
    apply_override(data, "stimulus.0.start", "10 ms")
    apply_override(data, "solver.dt", "0.1 ms")
    assert data == {"stimulus": [{"start": "10 ms"}], "solver": {"dt": "0.1 ms"}}


def test_apply_override_bad_index():
    with pytest.raises(ConfigError, match="index"):
        apply_override({"stimulus": []}, "stimulus.3.start", "1 ms")


def test_set_overrides_toml(cfg_file):
    conf = load_config(cfg_file, sets=['solver.dt="0.05 ms"'], environ={})
    assert ms(conf.solver.dt) == pytest.approx(0.05)


def test_set_without_equals_errors(cfg_file):
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        load_config(cfg_file, sets=["solver.dt"], environ={})


def test_unknown_set_key_errors(cfg_file):
    with pytest.raises(ConfigError, match="solver.dtt"):
        load_config(cfg_file, sets=["solver.dtt=1"], environ={})


def test_env_keys_are_case_insensitive(cfg_file):
    env = {"BEAT_EP__C_M": "2 uF/cm**2", "BEAT_SOLVER__DT": "0.05 ms", "HOME": "/x"}
    assert dict(env_overrides(env)) == {"ep.C_m": "2 uF/cm**2", "solver.dt": "0.05 ms"}
    conf = load_config(cfg_file, environ=env)
    assert conf.ep.C_m.to("uF/cm**2").magnitude == pytest.approx(2.0)


def test_precedence_toml_env_set_flag(cfg_file, tmp_path):
    env = {"BEAT_SOLVER__DT": "0.05 ms", "BEAT_OUTPUT__FOLDER": "env_out"}
    conf = load_config(
        cfg_file,
        sets=['solver.dt="0.02 ms"', 'output.folder="set_out"'],
        environ=env,
        output_folder=Path("flag_out"),
    )
    assert ms(conf.solver.dt) == pytest.approx(0.02)  # --set beats env
    assert conf.output.folder == (Path.cwd() / "flag_out").resolve()  # flag beats --set


def test_relative_paths_resolve_against_config_dir(tmp_path, monkeypatch):
    sub = tmp_path / "case"
    sub.mkdir()
    data = minimal_config_dict(sub)
    data["cell"]["ode_file"] = "ms.ode"
    data["output"]["folder"] = "out"
    (sub / "config.toml").write_text(toml.dumps(data))
    monkeypatch.chdir(tmp_path)  # run from another directory
    conf = load_config(sub / "config.toml", environ={})
    assert conf.cell.ode_file == sub / "ms.ode"
    assert conf.output.folder == sub / "out"
    assert conf.geometry.folder == sub / "geometry"


def test_validation_error_becomes_config_error(cfg_file):
    with pytest.raises(ConfigError, match="solver"):
        load_config(cfg_file, sets=["solver.dt=1"], environ={})  # no unit


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "nope.toml", environ={})


def test_petsc_options_flag(cfg_file):
    assert parse_petsc_options("-ksp_type cg -pc_type hypre -ksp_monitor") == {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_monitor": True,
    }
    conf = load_config(cfg_file, environ={}, petsc_options="-ksp_type cg")
    assert conf.solver.petsc_options["ksp_type"] == "cg"


def test_dump_roundtrip(cfg_file, tmp_path):
    conf = load_config(cfg_file, environ={})
    out = tmp_path / "resolved.toml"
    dump_config(conf, out)
    again = load_config(out, environ={})
    assert again.model_dump(mode="json") == conf.model_dump(mode="json")


def test_physics_hash_ignores_end_time_and_output(cfg_file):
    a = load_config(cfg_file, environ={})
    b = load_config(cfg_file, environ={}, sets=['solver.end_time="5 ms"', "output.log_every=3"])
    assert physics_hash(a) == physics_hash(b)


def test_physics_hash_changes_with_dt_and_ode_contents(cfg_file):
    a = load_config(cfg_file, environ={})
    b = load_config(cfg_file, environ={}, sets=['solver.dt="0.05 ms"'])
    hash_a = physics_hash(a)
    assert hash_a != physics_hash(b)
    a.cell.ode_file.write_text(a.cell.ode_file.read_text() + "\n# changed\n")
    assert physics_hash(load_config(cfg_file, environ={})) != hash_a
