from pathlib import Path

import pytest
import toml
from cli_helpers import MITCHELL_SCHAEFFER_ODE, minimal_config_dict

from beat.cli.config import ConfigError, ms
from beat.cli.overrides import (
    apply_override,
    dump_config,
    env_overrides,
    file_hash,
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


@pytest.mark.parametrize(
    "ecg",
    [
        {"electrodes": {"E": [2.0, 0.5]}},
        {"electrodes": {"E": [2.0, 0.5]}, "leads": "twelve-lead", "reference": "position"},
    ],
)
def test_dump_roundtrip_with_ecg(tmp_path, ecg):
    """config.resolved.toml holds the ECG section's defaults, reference = "potential" among
    them, also with leads = "none"; it loads back unchanged."""
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps(minimal_config_dict(tmp_path, postprocess={"ecg": ecg})))
    conf = load_config(path, environ={})
    out = tmp_path / "resolved.toml"
    dump_config(conf, out)
    assert toml.loads(out.read_text())["postprocess"]["ecg"]["reference"] == (
        ecg.get("reference", "potential")
    )
    again = load_config(out, environ={})
    assert again.model_dump(mode="json") == conf.model_dump(mode="json")


def test_load_config_exclude_drops_tables_before_validation(tmp_path):
    """beat 0.7.x's config.resolved.toml has [postprocess] sigma_b, which beat now refuses."""
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps(minimal_config_dict(tmp_path, postprocess={"sigma_b": 1.0})))
    with pytest.raises(ConfigError, match=r"postprocess\.ecg\.sigma_b"):
        load_config(path, environ={})
    conf = load_config(path, environ={}, exclude=("postprocess",))
    assert conf.postprocess == type(conf.postprocess)()


def test_dump_roundtrip_slab_template(tmp_path):
    """The shipped slab template has [postprocess.ecg] with leads = "none"."""
    from beat.cli import TEMPLATES_DIR

    conf = load_config(TEMPLATES_DIR / "slab" / "config.toml", environ={})
    assert conf.postprocess.ecg is not None and conf.postprocess.ecg.leads == "none"
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


def _folder_geometry_cfg_file(tmp_path, folder_name="mesh_a"):
    ode = tmp_path / "ms.ode"
    ode.write_text(MITCHELL_SCHAEFFER_ODE)
    data = {
        "geometry": {"type": "folder", "unit": "mm", "folder": folder_name},
        "cell": {"ode_file": str(ode), "v_name": "v"},
        "solver": {"dt": "0.1 ms", "end_time": "0.3 ms"},
        "output": {"folder": str(tmp_path / "output"), "save_every": "0.1 ms"},
    }
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps(data))
    return path


def test_physics_hash_folder_geometry_tracks_folder(tmp_path):
    cfg = _folder_geometry_cfg_file(tmp_path)
    a = load_config(cfg, environ={})
    b = load_config(cfg, environ={}, sets=['geometry.folder="mesh_b"'])
    assert physics_hash(a) != physics_hash(b)  # type="folder": folder IS the mesh


def test_physics_hash_ignores_folder_for_non_folder_geometry(cfg_file, tmp_path):
    # cfg_file's geometry.type is "rectangle": folder is just a mesh cache location.
    a = load_config(cfg_file, environ={})
    b = load_config(cfg_file, environ={}, sets=['geometry.folder="elsewhere"'])
    assert physics_hash(a) == physics_hash(b)

    data = minimal_config_dict(tmp_path, geometry={"type": "slab"})
    for key in ("lx", "ly", "dx"):
        data["geometry"].pop(key, None)
    slab_path = tmp_path / "slab.toml"
    slab_path.write_text(toml.dumps(data))
    c = load_config(slab_path, environ={})
    d = load_config(slab_path, environ={}, sets=['geometry.folder="elsewhere"'])
    assert physics_hash(c) == physics_hash(d)


def test_apply_override_through_scalar_errors():
    with pytest.raises(ConfigError, match="solver.dt.foo"):
        apply_override({"solver": {"dt": "0.1 ms"}}, "solver.dt.foo", "5")


def test_petsc_options_negative_number_value():
    assert parse_petsc_options("-ksp_rtol -1e-6 -pc_type hypre") == {
        "ksp_rtol": "-1e-6",
        "pc_type": "hypre",
    }
    assert parse_petsc_options("-ksp_max_it -3") == {"ksp_max_it": "-3"}


def test_file_hash_missing_file_errors(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        file_hash(tmp_path / "nope.txt")


def test_physics_hash_missing_ode_file_errors(cfg_file):
    conf = load_config(cfg_file, environ={})
    conf.cell.ode_file.unlink()
    with pytest.raises(ConfigError, match="not found"):
        physics_hash(conf)


def test_physics_hash_ignores_bcl(tmp_path):
    """solver.BCL only sets the run length (end = num_beats * BCL); it paces nothing, so it's
    excluded from the physics hash like end_time/num_beats."""
    by_time = tmp_path / "by_time.toml"
    by_time.write_text(toml.dumps(minimal_config_dict(tmp_path)))
    data = minimal_config_dict(tmp_path)
    data["solver"] = {"dt": "0.1 ms", "num_beats": 2, "BCL": "0.15 ms"}
    by_beats = tmp_path / "by_beats.toml"
    by_beats.write_text(toml.dumps(data))
    a = load_config(by_time, environ={})
    b = load_config(by_beats, environ={})
    c = load_config(by_beats, environ={}, sets=['solver.BCL="0.5 ms"'])
    assert physics_hash(a) == physics_hash(b) == physics_hash(c)
