"""Contract validation for bounded research workloads and explicit uncertainty."""

import math

import pytest
from pydantic import ValidationError

from app.advanced_schemas import (
    AnalyzeRequest, BatchPredictRequest, Distribution, ExperimentDesignRequest,
    ModelCardResponse, OptimizeRequest, Tolerances,
)

SCENARIO = {"material": "cotton", "o2": 21, "pressure_kpa": 101.3, "flow_cm_s": 10}
BOUNDS = {"o2_range": [12, 35], "pressure_range": [50, 115], "flow_range": [0, 100]}


def test_defaults_canonicalize_nested_scenarios_and_materials():
    analysis = AnalyzeRequest(scenario=SCENARIO).model_dump()
    assert analysis["scenario"]["material"] == "sibal_cotton_fiberglass"
    assert analysis["scenario"]["o2_percent"] == 21
    assert analysis["scenario"]["thickness_mm"] == .33
    assert analysis["tolerances"] == {"o2_percent": 1, "pressure_kpa": 2, "flow_cm_s": 1}
    experiment = ExperimentDesignRequest(material="COTTON").model_dump()
    assert experiment["material"] == "sibal_cotton_fiberglass"
    assert experiment["thickness_mm"] == .33
    assert experiment["strategy"] == "information_gain"


def test_optimizer_allows_fixed_controls_and_endpoint_targets():
    for target in (0, 1):
        request = OptimizeRequest(
            scenario=SCENARIO,
            bounds={"o2_range": [21, 21], "pressure_range": [101.3, 101.3], "flow_range": [0, 0]},
            target_extinction_probability=target,
            tolerances={"o2_percent": 0, "pressure_kpa": 0, "flow_cm_s": 0},
        )
        assert request.target_extinction_probability == target
        assert request.model_dump()["bounds"]["flow_range"] == (0, 0)


@pytest.mark.parametrize("count", [1, 256])
def test_batch_accepts_bounded_workloads(count):
    assert len(BatchPredictRequest(points=[SCENARIO] * count).points) == count


@pytest.mark.parametrize("count", [0, 257])
def test_batch_rejects_empty_or_excess_workloads(count):
    with pytest.raises(ValidationError):
        BatchPredictRequest(points=[SCENARIO] * count)


@pytest.mark.parametrize("changes", [
    {"samples": 31}, {"samples": 513}, {"samples": True}, {"samples": 32.5},
    {"tolerances": {"o2_percent": -1}}, {"tolerances": {"o2_percent": 11}},
    {"tolerances": {"pressure_kpa": 21}}, {"tolerances": {"flow_cm_s": 21}},
    {"tolerances": {"flow_cm_s": True}}, {"tolerances": {"flow_cm_s": math.inf}},
    {"scenario": SCENARIO | {"o2_percent": 30}}, {"extra": 1},
])
def test_analysis_rejects_invalid_inputs(changes):
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate({"scenario": SCENARIO} | changes)


@pytest.mark.parametrize("changes", [
    {"bounds": {"o2_range": [35, 12], "pressure_range": [50, 115], "flow_range": [0, 100]}},
    {"bounds": BOUNDS | {"flow_range": [-1, 10]}},
    {"bounds": BOUNDS | {"o2_range": [.21, 35]}},
    {"bounds": BOUNDS | {"pressure_range": [50, 101300]}},
    {"bounds": BOUNDS | {"o2_range": [12, math.nan]}},
    {"bounds": BOUNDS | {"flow_range": [False, 10]}},
    {"resolution": 4}, {"resolution": 17}, {"resolution": True},
    {"k": 0}, {"k": 21}, {"k": False},
    {"target_extinction_probability": -.1}, {"target_extinction_probability": 1.1},
    {"target_extinction_probability": True}, {"supported_only": "false"},
    {"supported_only": 1}, {"bounds": {}}, {"unknown": 1},
])
def test_optimizer_rejects_invalid_inputs(changes):
    with pytest.raises(ValidationError):
        OptimizeRequest.model_validate({"scenario": SCENARIO, "bounds": BOUNDS} | changes)


@pytest.mark.parametrize("changes", [
    {"material": "wood"}, {"pressure_kpa": 101300}, {"thickness_mm": .001},
    {"resolution": 9}, {"resolution": 61}, {"resolution": True},
    {"k": 0}, {"k": 21}, {"strategy": "bayesian_guarantee"},
    {"exclude_tested": "false"}, {"exclude_tested": 0},
    {"flow_range": [10, 10]}, {"flow_range": [10, 0]},
    {"o2_range": [5, 5]}, {"o2_range": [5, 101]},
    {"flow_range": [0, math.inf]}, {"extra": 1},
])
def test_experiment_design_rejects_invalid_inputs(changes):
    with pytest.raises(ValidationError):
        ExperimentDesignRequest.model_validate({"material": "cotton"} | changes)


def test_experiment_design_can_include_zero_flow_and_strict_false():
    request = ExperimentDesignRequest(material="cotton", flow_range=[0, 100], exclude_tested=False)
    assert request.flow_range[0] == 0 and request.exclude_tested is False


def test_default_tolerances_are_independent():
    first = AnalyzeRequest(scenario=SCENARIO)
    second = AnalyzeRequest(scenario=SCENARIO)
    first.tolerances.o2_percent = 5
    assert second.tolerances.o2_percent == 1
    with pytest.raises(ValidationError):
        Tolerances(unknown=0)


def test_distribution_checks_quantiles_but_allows_mean_outside_central_quantiles():
    assert Distribution(min=0, p05=0, mean=2, p95=1, max=100).mean == 2
    with pytest.raises(ValidationError):
        Distribution(min=0, p05=3, mean=2, p95=1, max=100)


@pytest.mark.parametrize("provenance,ensemble_size", [(None, 20), (None, None)])
def test_model_card_accepts_legacy_and_future_predictor_metadata(provenance, ensemble_size):
    response = ModelCardResponse(
        model_version="legacy-or-pinn", model_id=None, data_source="synthetic",
        validation_status="unavailable", provenance=provenance,
        training_data_summary={"rows": 20}, features=[], ensemble_size=ensemble_size,
        uncertainty_method="Predictor-provided uncertainty; no validation report available.",
        validation=None, limitations=["Validation metadata unavailable."],
    )
    assert response.provenance is None
    assert response.ensemble_size == ensemble_size
