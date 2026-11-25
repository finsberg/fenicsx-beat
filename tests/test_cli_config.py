from pathlib import Path

import pytest
import toml
from pydantic import ValidationError

from beat.cli.config import (
    SECOND_ORDER_SCHEMES,
    Config,
    SolverConfig,
    config_warnings,
    ms,
    splitting_order_warnings,
)


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


def test_random_endocardial_requires_at_least_one_marker(tmp_path):
    stim = {"type": "random_endocardial", "markers": [], "amplitude": "1 uA/cm**3"}
    with pytest.raises(ValidationError, match="at least 1 item"):
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


@pytest.mark.parametrize("stim_type", ["marker", "box", "random_endocardial"])
def test_stimulus_amplitude_per_length_is_rejected(tmp_path, stim_type):
    """The effective stimulus dimension is only ever 2 (facet marker) or 3 (cell marker, box,
    random_endocardial): uA/cm is never valid."""
    stim = {"type": stim_type, "amplitude": "1 uA/cm", **_STIM_EXTRA[stim_type]}
    with pytest.raises(ValidationError, match="amplitude"):
        Config.model_validate(base(tmp_path, stimulus=[stim]))


@pytest.mark.parametrize("stim_type", ["box", "random_endocardial"])
def test_volumetric_stimulus_requires_per_volume_amplitude(tmp_path, stim_type):
    stim = {"type": stim_type, "amplitude": "1 uA/cm**2", **_STIM_EXTRA[stim_type]}
    with pytest.raises(ValidationError, match="uA/cm\\*\\*3"):
        Config.model_validate(base(tmp_path, stimulus=[stim]))
    stim["amplitude"] = "1 uA/mm**3"
    Config.model_validate(base(tmp_path, stimulus=[stim]))


def test_marker_stimulus_accepts_areal_and_volumetric_amplitude(tmp_path):
    for amp in ("1 uA/cm**2", "1 uA/cm**3"):
        stim = {"type": "marker", "marker": "X0", "amplitude": amp}
        Config.model_validate(base(tmp_path, stimulus=[stim]))


_STIM_EXTRA = {
    "marker": {"marker": "X0"},
    "box": {"min": [0.0, 0.0], "max": [1.0, 1.0]},
    "random_endocardial": {},
}


def test_solver_run_length_fields_are_documented():
    fields = SolverConfig.model_fields
    assert "num_beats" in fields["end_time"].description
    assert "BCL" in fields["num_beats"].description
    assert "period" in fields["BCL"].description


# --- [postprocess.ecg] ------------------------------------------------------------------


def _load_with_postprocess(tmp_path, postprocess, sets=()):
    import toml
    from cli_helpers import minimal_config_dict

    from beat.cli.overrides import load_config

    path = tmp_path / "config.toml"
    path.write_text(toml.dumps(minimal_config_dict(tmp_path, postprocess=postprocess)))
    return load_config(path, sets=sets)


@pytest.mark.parametrize(
    "ecg, match",
    [
        ({}, "electrodes"),
        ({"electrodes": {}}, "electrodes"),
        ({"electrodes": {"E": [1.0]}}, "electrodes"),
        ({"electrodes": {"E": [1.0, 2.0]}, "unit": "ms"}, "unit"),
        ({"electrodes": {"E": [1.0, 2.0]}, "reference": "position"}, "reference = 'position'"),
    ],
)
def test_postprocess_ecg_invalid(tmp_path, ecg, match):
    from beat.cli.config import ConfigError

    with pytest.raises(ConfigError, match=match):
        _load_with_postprocess(tmp_path, {"ecg": ecg})


def test_postprocess_sigma_b_moved(tmp_path):
    from beat.cli.config import ConfigError

    with pytest.raises(ConfigError, match=r"postprocess\.ecg\.sigma_b"):
        _load_with_postprocess(tmp_path, {"sigma_b": 2.0})


def test_postprocess_ecg_valid_and_set_electrode(tmp_path):
    conf = _load_with_postprocess(
        tmp_path,
        {"ecg": {"electrodes": {"E": [1.0, 2.0, 3.0]}, "unit": "cm", "leads": "twelve-lead"}},
        sets=["postprocess.ecg.electrodes.V1=[1.0, 2.0]"],
    )
    ecg = conf.postprocess.ecg
    assert ecg.electrodes == {"E": [1.0, 2.0, 3.0], "V1": [1.0, 2.0]}
    assert ecg.leads == "twelve-lead"
    assert ecg.reference == "potential"
    assert ecg.sigma_b == 1.0


def test_postprocess_ecg_reference_with_leads(tmp_path):
    conf = _load_with_postprocess(
        tmp_path,
        {"ecg": {"electrodes": {"E": [1.0, 2.0]}, "leads": "twelve-lead", "reference": "position"}},
    )
    assert conf.postprocess.ecg.reference == "position"


def test_postprocess_ecg_potential_reference_without_leads(tmp_path):
    """reference = "potential" is the default, which config.resolved.toml writes out also
    with leads = "none"; only "position" needs a lead system."""
    conf = _load_with_postprocess(
        tmp_path,
        {"ecg": {"electrodes": {"E": [1.0, 2.0]}, "reference": "potential"}},
    )
    assert conf.postprocess.ecg.leads == "none"
    assert conf.postprocess.ecg.reference == "potential"


def test_postprocess_ecg_default_none(tmp_path):
    assert _load_with_postprocess(tmp_path, {}).postprocess.ecg is None


def test_splitting_defaults_to_godunov(tmp_path):
    assert Config.model_validate(base(tmp_path)).solver.splitting == "godunov"


def test_splitting_strang(tmp_path):
    data = base(tmp_path, solver={"dt": "0.1 ms", "end_time": "1 ms", "splitting": "strang"})
    assert Config.model_validate(data).solver.splitting == "strang"


@pytest.mark.parametrize(("theta", "name"), [(1.0, "godunov"), (0.5, "strang"), (1, "godunov")])
def test_deprecated_solver_theta_maps_to_splitting(tmp_path, caplog, theta, name):
    data = base(tmp_path, solver={"dt": "0.1 ms", "end_time": "1 ms", "theta": theta})
    conf = Config.model_validate(data)
    assert conf.solver.splitting == name
    assert "solver.theta is deprecated" in caplog.text
    assert "theta" not in conf.solver.model_dump(mode="json")


def test_deprecated_solver_theta_other_value_is_rejected(tmp_path):
    data = base(tmp_path, solver={"dt": "0.1 ms", "end_time": "1 ms", "theta": 0.3})
    with pytest.raises(ValidationError, match="only 1.0 \\(godunov\\) and 0.5 \\(strang\\)"):
        Config.model_validate(data)


@pytest.mark.parametrize("theta", [True, False])
def test_deprecated_solver_theta_rejects_booleans(tmp_path, theta):
    data = base(tmp_path, solver={"dt": "0.1 ms", "end_time": "1 ms", "theta": theta})
    with pytest.raises(ValidationError, match="only 1.0 \\(godunov\\) and 0.5 \\(strang\\)"):
        Config.model_validate(data)


def test_solver_theta_and_splitting_together_is_rejected(tmp_path):
    data = base(
        tmp_path,
        solver={"dt": "0.1 ms", "end_time": "1 ms", "theta": 0.5, "splitting": "strang"},
    )
    with pytest.raises(ValidationError, match="not the deprecated `theta`"):
        Config.model_validate(data)


def test_stiff_states_need_a_hybrid_scheme(tmp_path):
    cell = {
        "ode_file": str(tmp_path / "m.ode"),
        "scheme": "generalized_rush_larsen_2",
        "stiff_states": ["v"],
    }
    with pytest.raises(ValidationError, match="hybrid_"):
        Config.model_validate(base(tmp_path, cell=cell))


def test_stiff_states_with_hybrid_scheme(tmp_path):
    cell = {
        "ode_file": str(tmp_path / "m.ode"),
        "scheme": "hybrid_rush_larsen_2",
        "stiff_states": ["v"],
    }
    assert Config.model_validate(base(tmp_path, cell=cell)).cell.stiff_states == ["v"]


def _order_conf(tmp_path, splitting, scheme="generalized_rush_larsen", pde=None, ode=None):
    solver = {"dt": "0.1 ms", "end_time": "1 ms", "splitting": splitting}
    if pde is not None:
        solver["pde"] = pde
    if ode is not None:
        solver["ode"] = ode
    cell = {"ode_file": str(tmp_path / "m.ode"), "scheme": scheme}
    return Config.model_validate(base(tmp_path, cell=cell, solver=solver))


def _hybrid_cell(tmp_path, stiff_states):
    return {
        "ode_file": str(tmp_path / "m.ode"),
        "scheme": "hybrid_rush_larsen_2",
        "stiff_states": stiff_states,
    }


def test_stiff_states_are_sorted(tmp_path):
    cell = _hybrid_cell(tmp_path, ["m", "h", "j"])
    assert Config.model_validate(base(tmp_path, cell=cell)).cell.stiff_states == ["h", "j", "m"]


def test_stiff_states_order_does_not_change_the_physics_hash(tmp_path):
    from beat.cli.overrides import physics_hash

    (tmp_path / "m.ode").write_text("")
    hashes = {
        physics_hash(Config.model_validate(base(tmp_path, cell=_hybrid_cell(tmp_path, stiff))))
        for stiff in (["m", "h"], ["h", "m"])
    }
    assert len(hashes) == 1


def test_duplicate_stiff_states_are_rejected(tmp_path):
    with pytest.raises(ValidationError, match="duplicate.*'h'"):
        Config.model_validate(base(tmp_path, cell=_hybrid_cell(tmp_path, ["h", "m", "h"])))


@pytest.mark.parametrize("scheme", ["hybrid_rush_larsen", "hybrid_rush_larsen_2"])
def test_warning_for_hybrid_scheme_without_stiff_states(tmp_path, scheme):
    conf = _order_conf(tmp_path, "godunov", scheme=scheme)
    (msg,) = config_warnings(conf)
    assert "cell.stiff_states" in msg
    assert repr(scheme) in msg


def test_no_hybrid_warning_with_stiff_states(tmp_path):
    conf = Config.model_validate(base(tmp_path, cell=_hybrid_cell(tmp_path, ["h"])))
    assert config_warnings(conf) == []


def test_unknown_scheme_warns_it_is_unknown_not_first_order(tmp_path):
    pytest.importorskip("gotranx")
    (msg,) = config_warnings(_order_conf(tmp_path, "strang", scheme="rush_larsen_3"))
    assert "'rush_larsen_3' is not a gotranx scheme" in msg
    assert "first order" not in msg


def test_config_warnings_include_order_warnings(tmp_path):
    conf = _order_conf(tmp_path, "strang")
    assert config_warnings(conf) == splitting_order_warnings(conf)


def test_no_order_warning_for_godunov(tmp_path):
    assert splitting_order_warnings(_order_conf(tmp_path, "godunov")) == []


def test_no_order_warning_for_strang_with_second_order_scheme(tmp_path):
    conf = _order_conf(tmp_path, "strang", scheme="generalized_rush_larsen_2")
    assert splitting_order_warnings(conf) == []


def test_order_warning_for_strang_with_first_order_scheme(tmp_path):
    (msg,) = splitting_order_warnings(_order_conf(tmp_path, "strang"))
    assert "'generalized_rush_larsen'" in msg
    assert "generalized_rush_larsen_2" in msg
    assert "first order" in msg


def test_order_warning_for_strang_with_backward_euler_pde(tmp_path):
    conf = _order_conf(
        tmp_path,
        "strang",
        scheme="generalized_rush_larsen_2",
        pde={"type": "theta", "theta": 1.0},
    )
    (msg,) = splitting_order_warnings(conf)
    assert "solver.pde.theta = 1.0" in msg


def test_order_warning_skips_irksome_backends(tmp_path):
    conf = _order_conf(tmp_path, "strang", pde={"type": "irksome"}, ode={"type": "irksome"})
    assert splitting_order_warnings(conf) == []


def test_second_order_schemes_exist_in_gotranx():
    gotranx = pytest.importorskip("gotranx")
    assert SECOND_ORDER_SCHEMES <= {s.name for s in gotranx.schemes.Scheme}


def test_niederer_template_has_no_order_warning(tmp_path):
    import beat.cli

    path = Path(beat.cli.__file__).parent / "templates" / "niederer_benchmark" / "config.toml"
    conf = Config.model_validate(toml.loads(path.read_text()))
    assert conf.solver.splitting == "strang"
    assert splitting_order_warnings(conf) == []


def test_merge_half_steps_defaults_to_false(tmp_path):
    assert Config.model_validate(base(tmp_path)).solver.merge_half_steps is False


def test_merge_half_steps_with_strang(tmp_path):
    solver = {"dt": "0.1 ms", "end_time": "1 ms", "splitting": "strang", "merge_half_steps": True}
    assert Config.model_validate(base(tmp_path, solver=solver)).solver.merge_half_steps is True


def test_merge_half_steps_needs_strang(tmp_path):
    solver = {"dt": "0.1 ms", "end_time": "1 ms", "merge_half_steps": True}
    with pytest.raises(ValidationError, match="merge_half_steps"):
        Config.model_validate(base(tmp_path, solver=solver))
