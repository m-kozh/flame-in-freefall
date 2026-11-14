"""Decision features are tested against analytic counterexamples, not only trees."""

import json

import numpy as np
import pandas as pd
import pytest

from app.advanced_schemas import AnalyzeResponse, OptimizeResponse
from data.materials import MATERIALS
from models.base import PredictionBatch
from models.residual_ensemble import physics_frame
from services.decision import DecisionService
from services.predictions import prediction_dicts

SCENARIO = {"material": "cotton", "o2_percent": 21., "pressure_kpa": 101.3, "flow_cm_s": 10.}
FIXED_BOUNDS = {"o2_range": (21., 21.), "pressure_range": (101.3, 101.3), "flow_range": (10., 10.)}
ZERO_TOLERANCE = {"o2_percent": 0., "pressure_kpa": 0., "flow_cm_s": 0.}


class AnalyticPredictor:
    """A non-tree predictor with known probabilities and support boundaries."""

    model_version = "analytic-test"
    model_id = "analytic-test-model"
    data_source = "synthetic"
    training_data = pd.DataFrame()

    def __init__(self, probability=None, support=None):
        self.probability = probability or (lambda frame: np.full(len(frame), .95))
        self.support = support or (lambda frame: np.ones(len(frame), dtype=bool))
        self.calls = []

    def predict(self, frame):
        self.calls.append(frame.copy())
        probability = np.asarray(self.probability(frame), dtype=float)
        rate = frame.o2_percent.to_numpy() / 10
        physical = physics_frame(frame)
        return PredictionBatch(
            mean=rate, lower=rate * .8, upper=rate * 1.2,
            extinction_probability=probability,
            physics_baseline=physical["spread_rate_mm_s"],
            extinction_lower=np.maximum(probability - .02, 0),
            extinction_upper=np.minimum(probability + .02, 1),
            uncertainty=np.full(len(frame), .01), physics_regime=physical["regime"],
            in_data_range=np.asarray(self.support(frame), dtype=bool),
            log_std=np.full(len(frame), .1),
        )


def test_real_dossier_is_reproducible_and_preserves_current_prediction(trained_bundle):
    model = trained_bundle["model"]
    service = DecisionService(model)
    result = service.analyze(SCENARIO, samples=32)
    AnalyzeResponse.model_validate(result)
    assert result == service.analyze(SCENARIO, samples=32)
    expected = prediction_dicts(model, pd.DataFrame([SCENARIO]))[0]
    assert result["current"] == expected
    assert len(result["material_comparison"]) == 5
    for comparison in result["material_comparison"]:
        assert comparison["prediction"]["inputs"]["thickness_mm"] == MATERIALS[comparison["material"]].thickness_mm
    assert len(result["preset_comparison"]) == 4
    assert result["robustness"]["sample_count"] == 40
    assert result["physics_disagreement"]["potential_baseline_mm_s"] > 0
    assert np.isfinite(result["physics_disagreement"]["learned_to_potential_ratio"])
    json.dumps(result, allow_nan=False)


def test_dossier_uses_one_batch_and_fixed_controls_have_no_artificial_spread():
    model = AnalyticPredictor()
    result = DecisionService(model).analyze(SCENARIO, ZERO_TOLERANCE, samples=32)
    AnalyzeResponse.model_validate(result)
    assert len(model.calls) == 1
    assert len(model.calls[0]) == 32 + 24  # Current + comparisons + sensitivity + corners.
    assert result["assessment"] == "supported"
    robust = result["robustness"]
    assert robust["extinction_probability"]["min"] == robust["extinction_probability"]["max"] == .95
    assert robust["minimum_sampled_extinction_lower"] == pytest.approx(.93)
    assert robust["decision_flip_fraction"] == 0
    for item in result["sensitivity"]:
        assert item["lower_input"] == item["upper_input"]
        assert item["extinction_probability_change"] == item["spread_rate_change_mm_s"] == 0


def test_dossier_intersects_tolerances_with_the_legal_domain():
    model = AnalyticPredictor()
    scenario = SCENARIO | {"o2_percent": 5., "pressure_kpa": 10., "flow_cm_s": 0.}
    result = DecisionService(model).analyze(scenario, samples=32)
    # Comparison rows use atmosphere presets; inspect only sensitivity and perturbations.
    sampled = model.calls[0].iloc[10:]
    assert sampled.o2_percent.between(5, 6).all()
    assert sampled.pressure_kpa.between(10, 12).all()
    assert sampled.flow_cm_s.between(0, 1).all()
    assert result["sensitivity"][0]["lower_input"] == 5
    assert result["sensitivity"][1]["lower_input"] == 10
    assert result["sensitivity"][2]["lower_input"] == 0
    corners = model.calls[0].iloc[-8:]
    assert set(corners.o2_percent) == {5., 6.}
    assert set(corners.pressure_kpa) == {10., 12.}
    assert set(corners.flow_cm_s) == {0., 1.}


def test_dossier_detects_boundary_flips_and_unsupported_perturbations():
    model = AnalyticPredictor(
        probability=lambda frame: 1 / (1 + np.exp(frame.o2_percent.to_numpy() - 21)),
        support=lambda frame: frame.o2_percent.to_numpy() <= 21.5,
    )
    result = DecisionService(model).analyze(SCENARIO, samples=64)
    assert 0 < result["robustness"]["decision_flip_fraction"] < 1
    assert 0 < result["robustness"]["support_fraction"] < 1
    assert result["assessment"] == "extrapolation"
    assert result["sensitivity"][0]["extinction_probability_change"] < 0
    assert result["robustness"]["minimum_sampled_extinction_lower"] == pytest.approx(1 / (1 + np.e) - .02)


def test_optimizer_preserves_low_and_high_flow_branches():
    # A U-shaped non-burning probability is high at both flow extremes.
    model = AnalyticPredictor(probability=lambda frame: .2 + .75 * np.clip(
        np.abs(np.log1p(frame.flow_cm_s.to_numpy()) - np.log(11)) / 1.8, 0, 1,
    ))
    bounds = FIXED_BOUNDS | {"flow_range": (0., 100.)}
    result = DecisionService(model).optimize(SCENARIO, bounds, resolution=5, k=20, tolerances=ZERO_TOLERANCE)
    OptimizeResponse.model_validate(result)
    flows = [row["prediction"]["inputs"]["flow_cm_s"] for row in result["candidates"]]
    assert result["status"] == "found"
    assert 0. in flows and 100. in flows
    costs = [row["change_cost"] for row in result["candidates"]]
    assert costs == sorted(costs)
    assert result["evaluated_count"] == 6  # Five grid points plus the off-grid current flow.


def test_optimizer_does_not_accept_a_centre_that_fails_a_corner():
    model = AnalyticPredictor(probability=lambda frame: np.where(frame.o2_percent.to_numpy() >= 21, .99, .1))
    result = DecisionService(model).optimize(SCENARIO, FIXED_BOUNDS)
    assert result["status"] == "not_found" and not result["target_achievable"]
    assert result["eligible_count"] == 1
    assert result["candidates"] == []
    assert result["best_available"]["prediction"]["extinction_probability"] == .99
    assert result["best_available"]["robustness"]["minimum_sampled_extinction_lower"] == pytest.approx(.08)
    assert len(model.calls) == 2 and len(model.calls[1]) == 8


def test_optimizer_requires_support_at_all_corners_and_can_expose_extrapolation():
    model = AnalyticPredictor(support=lambda frame: frame.o2_percent.to_numpy() <= 21.25)
    service = DecisionService(model)
    supported = service.optimize(SCENARIO, FIXED_BOUNDS)
    assert supported["eligible_count"] == 0
    assert supported["best_available"] is None and supported["pareto_front"] == []
    assert supported["status"] == "not_found"
    allowed = service.optimize(SCENARIO, FIXED_BOUNDS, supported_only=False)
    assert allowed["status"] == "found"
    assert allowed["best_available"]["robustness"]["support_fraction"] == pytest.approx(5 / 9)
    assert any("supported_only=false" in text for text in allowed["limitations"])


def test_optimizer_checks_farther_candidates_after_nearby_nominal_success_fails():
    model = AnalyticPredictor(probability=lambda frame: np.where(
        np.abs(frame.o2_percent.to_numpy() - 21) < .01, .99,
        np.where(frame.o2_percent.to_numpy() >= 21.8, .96, .1),
    ))
    bounds = FIXED_BOUNDS | {"o2_range": (20., 24.)}
    result = DecisionService(model).optimize(
        SCENARIO, bounds, resolution=5, k=1,
        tolerances={"o2_percent": .5, "pressure_kpa": 0., "flow_cm_s": 0.},
    )
    assert result["status"] == "found"
    assert result["candidates"][0]["prediction"]["inputs"]["o2_percent"] == 23.
    assert len(model.calls[1]) == 5 * 8  # Every eligible centre, not just the nominal top-k.


def test_optimizer_costs_deltas_and_bounds_are_consistent():
    model = AnalyticPredictor()
    bounds = {"o2_range": (20., 24.), "pressure_range": (100., 110.), "flow_range": (0., 100.)}
    result = DecisionService(model).optimize(SCENARIO, bounds, resolution=5, k=20, tolerances=ZERO_TOLERANCE)
    assert result["candidates"][0]["change_cost"] == 0
    assert all(delta == 0 for delta in result["candidates"][0]["deltas"].values())
    for row in result["candidates"]:
        point = row["prediction"]["inputs"]
        changes = np.array([point[name] - SCENARIO[name] for name in ["o2_percent", "pressure_kpa", "flow_cm_s"]])
        assert row["change_cost"] == pytest.approx(np.linalg.norm(changes / np.array([4., 10., 100.])))
        assert list(row["deltas"].values()) == pytest.approx(changes)
    outside = DecisionService(AnalyticPredictor()).optimize(
        SCENARIO, FIXED_BOUNDS | {"o2_range": (22., 24.)}, resolution=5, tolerances=ZERO_TOLERANCE,
    )
    assert outside["evaluated_count"] == 5
    assert all(row["prediction"]["inputs"]["o2_percent"] >= 22 for row in outside["candidates"])


def test_optimizer_returns_an_honest_truncated_pareto_frontier():
    # Probability increases continuously with the exact normalised change cost;
    # every distinct-cost grid point is nondominated.
    model = AnalyticPredictor(probability=lambda frame: .1 + .8 * np.sqrt(
        ((frame.o2_percent.to_numpy() - 20) / 4) ** 2
        + ((frame.pressure_kpa.to_numpy() - 100) / 10) ** 2,
    ) / np.sqrt(2))
    scenario = SCENARIO | {"o2_percent": 20., "pressure_kpa": 100.}
    bounds = {"o2_range": (20., 24.), "pressure_range": (100., 110.), "flow_range": (10., 10.)}
    result = DecisionService(model).optimize(scenario, bounds, resolution=16, tolerances=ZERO_TOLERANCE, target_extinction_probability=1.)
    assert result["status"] == "not_found" and result["best_available"] is not None
    assert result["pareto_front_total_count"] > 20
    assert result["pareto_front_truncated"] is True
    frontier = result["pareto_front"]
    assert len(frontier) == 20 and frontier[0]["change_cost"] == 0
    assert frontier[-1]["change_cost"] == pytest.approx(np.sqrt(2))
    for a, b in zip(frontier, frontier[1:]):
        assert a["change_cost"] <= b["change_cost"]
        assert a["robustness"]["minimum_sampled_extinction_lower"] <= b["robustness"]["minimum_sampled_extinction_lower"]


def test_pareto_counts_equally_good_alternatives_without_arbitrary_direction_bias():
    model = AnalyticPredictor(probability=lambda frame: .1 + .8 * np.abs(frame.o2_percent.to_numpy() - 21))
    bounds = FIXED_BOUNDS | {"o2_range": (20., 22.)}
    result = DecisionService(model).optimize(SCENARIO, bounds, resolution=5, tolerances=ZERO_TOLERANCE)
    # Symmetric lower/higher-oxygen alternatives have identical objectives.
    # They remain separate nondominated settings, rather than losing one branch.
    assert result["pareto_front_total_count"] == 5
    assert len(result["pareto_front"]) == 5
    assert {row["prediction"]["inputs"]["o2_percent"] for row in result["pareto_front"]} == {20., 20.5, 21., 21.5, 22.}


@pytest.mark.parametrize("kwargs", [{"samples": 31}, {"samples": True}, {"tolerances": {"flow_cm_s": -1}}])
def test_analysis_rejects_invalid_direct_call_options(kwargs):
    with pytest.raises(ValueError):
        DecisionService(AnalyticPredictor()).analyze(SCENARIO, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"resolution": 17}, {"resolution": True}, {"k": 0},
    {"target_extinction_probability": np.nan}, {"supported_only": "false"},
])
def test_optimization_rejects_invalid_direct_call_options(kwargs):
    with pytest.raises(ValueError):
        DecisionService(AnalyticPredictor()).optimize(SCENARIO, FIXED_BOUNDS, **kwargs)
