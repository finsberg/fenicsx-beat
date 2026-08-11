from pathlib import Path

import pytest
from pydantic import ValidationError

from beat.config import CellConfig, Config, MeshConfig, SimulationConfig, StimulusConfig


def test_default_config_has_expected_sections():
    conf = Config()
    assert isinstance(conf.mesh, MeshConfig)
    assert isinstance(conf.cell, CellConfig)
    assert isinstance(conf.simulation, SimulationConfig)
    assert isinstance(conf.stimulus, StimulusConfig)


def test_default_config_values():
    conf = Config()
    assert conf.ep.chi.to("cm**-1").magnitude == pytest.approx(1400.0)
    assert conf.cell.v_name == "v"
    assert conf.simulation.theta == pytest.approx(1.0)
    assert conf.simulation.save_every_ms == pytest.approx(1.0)
    assert conf.stimulus.marker == "ENDO"
    assert conf.stimulus.amplitude == pytest.approx(5000.0)
    assert conf.stimulus.duration.to("ms").magnitude == pytest.approx(2.0)


def test_conductivity_accepts_alternative_units():
    conf = Config(ep={"conductivity": {"sigma_el": "6.2 mS/cm"}})
    assert conf.ep.conductivity.sigma_el.to("S/m").magnitude == pytest.approx(0.62)


def test_conductivity_rejects_incompatible_units():
    with pytest.raises(ValidationError):
        Config(ep={"conductivity": {"sigma_el": "5 ms"}})


def test_dump_and_parse_toml_roundtrip(tmp_path):
    conf = Config(
        stimulus={"marker": "X0", "amplitude": 1234.0},
        simulation={"num_beats": 2, "output_folder": str(tmp_path / "output")},
    )
    toml_path = tmp_path / "config.toml"
    conf.dump_toml(toml_path)
    assert toml_path.is_file()

    loaded = Config.parse_toml(toml_path)
    assert loaded.stimulus.marker == "X0"
    assert loaded.stimulus.amplitude == pytest.approx(1234.0)
    assert loaded.simulation.num_beats == 2
    assert loaded.ep.chi.to("cm**-1").magnitude == pytest.approx(
        conf.ep.chi.to("cm**-1").magnitude,
    )


def test_parse_toml_missing_file_raises(tmp_path):
    missing = tmp_path / "does-not-exist.toml"
    with pytest.raises(FileNotFoundError):
        Config.parse_toml(missing)


def test_parse_toml_rejects_unknown_section(tmp_path):
    toml_path = tmp_path / "config.toml"
    toml_path.write_text("[not_a_real_section]\nfoo = 1\n")
    with pytest.raises(ValidationError):
        Config.parse_toml(toml_path)


def test_mesh_config_default_unit_and_folder():
    mesh = MeshConfig()
    assert mesh.unit == "mm"
    assert Path(mesh.folder).name == "mesh"
