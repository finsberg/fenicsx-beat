import pytest
from pydantic import ValidationError

from beat.cli.config import Config, ms


def base(tmp_path, **kw):
    data = {
        "geometry": {"type": "rectangle", "lx": 1.0, "ly": 1.0, "dx": 0.25},
        "cell": {"ode_file": str(tmp_path / "m.ode")},
        "solver": {"dt": "0.1 ms", "end_time": "1 ms"},
    }
    data.update(kw)
    return data


def test_minimal_config_validates_and_converts_units(tmp_path):
    conf = Config.model_validate(base(tmp_path))
    assert ms(conf.solver.dt) == pytest.approx(0.1)
    assert conf.solver.t_end_ms() == pytest.approx(1.0)
    assert conf.geometry.fibers.type == "isotropic"  # default for rectangle
    assert conf.stimulus == []
    assert conf.cell.steady_state is None  # opt-in


def test_string_defaults_are_validated_into_quantities(tmp_path):
    conf = Config.model_validate(base(tmp_path))
    assert ms(conf.output.save_every) == pytest.approx(1.0)


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Config.model_validate(base(tmp_path, solver={"dt": "0.1 ms", "end_time": "1 ms", "dtt": 1}))


def test_unknown_geometry_type_lists_options(tmp_path):
    with pytest.raises(ValidationError, match="slab"):
        Config.model_validate(base(tmp_path, geometry={"type": "slap"}))


def test_quantity_without_unit_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="units"):
        Config.model_validate(base(tmp_path, solver={"dt": 0.1, "end_time": "1 ms"}))


def test_geometry_invalid_unit_string_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="valid unit"):
        Config.model_validate(
            base(
                tmp_path,
                geometry={"type": "rectangle", "lx": 1.0, "ly": 1.0, "dx": 0.25, "unit": "banana"},
            ),
        )


def test_stimulus_invalid_unit_string_is_rejected(tmp_path):
    stim = {"type": "marker", "marker": "X0", "amplitude": "5 banana"}
    with pytest.raises(ValidationError, match="valid quantity"):
        Config.model_validate(base(tmp_path, stimulus=[stim]))


def test_end_time_or_beats_exactly_one(tmp_path):
    with pytest.raises(ValidationError, match="exactly one"):
        Config.model_validate(base(tmp_path, solver={"dt": "0.1 ms"}))
    with pytest.raises(ValidationError, match="exactly one"):
        Config.model_validate(
            base(
                tmp_path,
                solver={"dt": "0.1 ms", "end_time": "1 ms", "num_beats": 2, "BCL": "1 s"},
            ),
        )
    conf = Config.model_validate(
        base(tmp_path, solver={"dt": "0.1 ms", "num_beats": 2, "BCL": "1 s"}),
    )
    assert conf.solver.t_end_ms() == pytest.approx(2000.0)


def test_stimulus_amplitude_must_be_a_current_density(tmp_path):
    stim = {"type": "marker", "marker": "X0", "amplitude": "5 mV"}
    with pytest.raises(ValidationError, match="current"):
        Config.model_validate(base(tmp_path, stimulus=[stim]))


def test_stimulus_period_must_exceed_duration(tmp_path):
    stim = {
        "type": "marker",
        "marker": "X0",
        "amplitude": "1 uA/cm**2",
        "duration": "5 ms",
        "period": "2 ms",
    }
    with pytest.raises(ValidationError, match="period"):
        Config.model_validate(base(tmp_path, stimulus=[stim]))


def test_num_pulses_requires_period(tmp_path):
    stim = {"type": "marker", "marker": "X0", "amplitude": "1 uA/cm**2", "num_pulses": 3}
    with pytest.raises(ValidationError, match="period"):
        Config.model_validate(base(tmp_path, stimulus=[stim]))


def test_box_min_max_checked(tmp_path):
    stim = {"type": "box", "min": [0.0, 1.0], "max": [1.0, 0.5], "amplitude": "1 uA/cm**3"}
    with pytest.raises(ValidationError, match="min < max"):
        Config.model_validate(base(tmp_path, stimulus=[stim]))


def test_regions_require_layers(tmp_path):
    cell = {"ode_file": "m.ode", "regions": {"endo": {"parameters": {"celltype": 0}}}}
    with pytest.raises(ValidationError, match="layers"):
        Config.model_validate(base(tmp_path, cell=cell))


def test_transmural_region_names(tmp_path):
    cell = {
        "ode_file": "m.ode",
        "layers": {"method": "transmural"},
        "regions": {"endoo": {"parameters": {}}},
    }
    with pytest.raises(ValidationError, match="endo, mid, epi"):
        Config.model_validate(base(tmp_path, cell=cell))
    cell["regions"] = {"endo": {"parameters": {"celltype": 0.0}}}
    conf = Config.model_validate(base(tmp_path, cell=cell))
    assert conf.cell.region_names() == ["mid", "endo", "epi"]


def test_cell_marker_region_names(tmp_path):
    cell = {
        "ode_file": "m.ode",
        "layers": {"method": "cell_markers", "map": {"scar": "SCAR"}},
        "regions": {"healthy": {"parameters": {}}},
    }
    with pytest.raises(ValidationError, match="scar"):
        Config.model_validate(base(tmp_path, cell=cell))


def test_conductivity_preset_and_override(tmp_path):
    conf = Config.model_validate(
        base(tmp_path, ep={"conductivity": {"preset": "Bishop", "sigma_il": "0.5 S/m"}}),
    )
    g = conf.ep.conductivity.resolved()
    assert g["g_il"].to("S/m").magnitude == pytest.approx(0.5)
    assert g["g_el"].to("S/m").magnitude == pytest.approx(0.12)  # Bishop


def test_conductivity_alternative_units(tmp_path):
    conf = Config.model_validate(base(tmp_path, ep={"conductivity": {"sigma_el": "6.2 mS/cm"}}))
    assert conf.ep.conductivity.resolved()["g_el"].to("S/m").magnitude == pytest.approx(0.62)


def test_generator_kwargs_exclude_cli_fields(tmp_path):
    conf = Config.model_validate(base(tmp_path, geometry={"type": "slab", "lx": 2.0}))
    kw = conf.geometry.generator_kwargs()
    assert kw["lx"] == 2.0
    assert not {"type", "unit", "folder", "fibers"} & set(kw)


def test_irksome_backends_parse(tmp_path):
    conf = Config.model_validate(
        base(
            tmp_path,
            solver={
                "dt": "0.1 ms",
                "end_time": "1 ms",
                "pde": {"type": "irksome", "tableau": "GaussLegendre", "stages": 2},
                "ode": {"type": "external_operator"},
            },
        ),
    )
    assert conf.solver.pde.stages == 2
    assert conf.solver.ode.type == "external_operator"
