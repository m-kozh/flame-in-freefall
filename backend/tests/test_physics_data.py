import numpy as np
import pandas as pd
import pytest

from data.loader import COLUMNS, DataValidationError, load_csv, validate_dataframe
from data.materials import get_material, resolve_material
from data.synthetic import generate_synthetic
from physics.flame_spread import OPTIMAL_FLOW_CM_S, evaluate_flame


def physics(o2, flow, pressure=101.3):
    material = get_material("cotton")
    return evaluate_flame(
        o2, pressure, flow, material.thickness_mm, material.density_kg_m3,
        material.specific_heat_j_kgk, material.vaporization_temp_k, material.limiting_o2_percent,
    )


def test_oxygen_and_pressure_increase_burning_rate():
    assert np.all(np.diff(physics(np.array([25., 30., 40.]), 10)["spread_rate_mm_s"]) > 0)
    assert physics(35, 10, 120)["spread_rate_mm_s"] > physics(35, 10, 50)["spread_rate_mm_s"]


def test_boundary_has_quench_and_blowoff_branches():
    low = np.geomspace(.05, OPTIMAL_FLOW_CM_S, 100)
    high = np.linspace(OPTIMAL_FLOW_CM_S, 200, 100)
    assert np.all(np.diff(physics(30, low)["limiting_o2_percent"]) < 0)
    assert np.all(np.diff(physics(30, high)["limiting_o2_percent"]) > 0)
    observed = {
        str(physics(40, 10)["regime"]), str(physics(25, .1)["regime"]),
        str(physics(25, 150)["regime"]), str(physics(5, 10)["regime"]),
    }
    assert observed == {"burning", "quench_extinction", "blowoff_extinction", "no_ignition"}
    result = physics(5, np.array([0., 10., 200.]))
    assert np.all(result["spread_rate_mm_s"] == 0)
    assert np.isfinite(np.log(result["potential_spread_rate_mm_s"])).all()


def test_generator_is_reproducible_and_exact_schema(tmp_path):
    a = generate_synthetic(400)
    pd.testing.assert_frame_equal(a, generate_synthetic(400))
    assert list(a.columns) == COLUMNS and a.experiment_id.nunique() == 20
    assert set(a.source) == {"synthetic"}
    assert not ((a.o2_percent > 32) & (a.pressure_kpa > 95) & (a.flow_cm_s > 35)).any()
    path = tmp_path / "data.csv"
    a.to_csv(path, index=False)
    loaded = load_csv(path)
    pd.testing.assert_frame_equal(a, loaded, check_exact=False)


def test_loader_rejects_missing_columns(tmp_path):
    path = tmp_path / "missing.csv"
    generate_synthetic(200).drop(columns="pressure_kpa").to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="pressure_kpa"):
        load_csv(path)


@pytest.mark.parametrize("column,value", [
    ("o2_percent", .21), ("pressure_kpa", 101300), ("flow_cm_s", -1),
    ("thickness_mm", 0), ("source", "unknown"), ("ignited", "maybe"),
    ("gravity_g", np.inf), ("material", "wood"),
])
def test_loader_rejects_wrong_units_and_values(column, value):
    frame = generate_synthetic(200)
    frame[column] = value
    with pytest.raises(DataValidationError):
        validate_dataframe(frame)


def test_loader_rejects_inconsistent_outcomes():
    frame = generate_synthetic(200)
    frame.loc[0, ["ignited", "extinguished", "spread_rate_mm_s"]] = [False, False, 1.]
    with pytest.raises(DataValidationError, match="Non-burning"):
        validate_dataframe(frame)
    frame.loc[0, "extinguished"] = True
    with pytest.raises(DataValidationError, match="requires ignited"):
        validate_dataframe(frame)


def test_material_aliases_and_error():
    assert resolve_material("COTTON") == "sibal_cotton_fiberglass"
    assert get_material("pmma_thick").thermally_thin is False
    with pytest.raises(ValueError, match="Valid ids"):
        resolve_material("wood")
