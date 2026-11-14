from types import SimpleNamespace

import numpy as np
import pytest

from app.advanced_schemas import ExperimentDesignResponse
from services.experiment_design import ExperimentDesignService, binary_entropy_bits, information_components


class ResearchEnsemble:
    model_version = "research-stub"
    data_source = "synthetic"
    model_id = None

    def __init__(self, measured="none", agrees=False):
        self.measured = measured
        self.agrees = agrees
        self.calls = 0

    def predict_members(self, frame):
        self.calls += 1
        # Identical aggregate p=.5: only the higher-oxygen half has member
        # disagreement. This distinguishes information gain from plain entropy.
        uncertain_model = frame.o2_percent.to_numpy() >= 30
        lower = np.where(uncertain_model & ~np.array(self.agrees), .05, .5)
        return {
            "extinction_probability": np.stack([lower, 1 - lower]),
            "in_data_range": np.ones(len(frame), dtype=bool),
        }

    def support_distance(self, frame):
        if self.measured == "all":
            return np.zeros(len(frame))
        return np.where(frame.o2_percent.to_numpy() == 10, 0., .3) if self.measured == "first" else np.full(len(frame), .3)


def test_entropy_separates_member_disagreement_from_class_ambiguity():
    values = information_components(np.array([[.5, 0, 1], [.5, 1, 1]]))
    np.testing.assert_allclose(values["predictive_entropy_bits"], [1, 1, 0])
    np.testing.assert_allclose(values["expected_member_entropy_bits"], [1, 0, 0])
    np.testing.assert_allclose(values["ensemble_information_gain_proxy_bits"], [0, 1, 0])
    np.testing.assert_array_equal(binary_entropy_bits(np.array([0., 1.])), [0., 0.])


@pytest.mark.parametrize("invalid", [np.array([np.nan]), np.array([1.01]), np.array([-.01])])
def test_entropy_rejects_invalid_probabilities(invalid):
    with pytest.raises(ValueError):
        binary_entropy_bits(invalid)


def test_information_strategy_prefers_epistemic_split_and_preserves_diversity():
    model = ResearchEnsemble()
    service = ExperimentDesignService(model)
    payload = dict(material="cotton", flow_range=(0, 100), o2_range=(10, 50), resolution=10, k=8)
    result = service.design(**payload)
    ExperimentDesignResponse.model_validate(result)
    assert len(result["points"]) == 8
    assert result["points"][0]["score"] == 1.
    assert all(point["o2"] >= 30 and point["ensemble_information_gain_proxy_bits"] > .7 for point in result["points"])
    coordinates = np.array([[(p["o2"] - 10) / 40, np.log1p(p["flow_cm_s"]) / np.log1p(100)] for p in result["points"]])
    for index in range(len(coordinates)):
        assert np.all(np.linalg.norm(coordinates[index + 1:] - coordinates[index], axis=1) >= .1 - 1e-12)
    result["points"][0]["score"] = -999
    assert service.design(**payload)["points"][0]["score"] == 1
    assert model.calls == 1


def test_measured_candidates_are_excluded_before_score_normalisation():
    service = ExperimentDesignService(ResearchEnsemble(measured="first"))
    result = service.design("cotton", flow_range=(0, 100), o2_range=(10, 50), resolution=10, k=20)
    assert result["excluded_tested_count"] == 10 and result["eligible_count"] == 90
    assert all(point["nearest_test_distance"] > .025 and point["o2"] > 10 for point in result["points"])


def test_no_eligible_or_no_information_is_reported_without_invented_priority():
    empty = ExperimentDesignService(ResearchEnsemble(measured="all")).design("cotton", resolution=10)
    assert empty["eligible_count"] == 0 and empty["points"] == []
    identical = ExperimentDesignService(ResearchEnsemble(agrees=True)).design("cotton", resolution=10)
    assert identical["eligible_count"] > 0 and identical["points"] == []
    assert any("nonzero" in note for note in identical["method_notes"])
    boundary = ExperimentDesignService(ResearchEnsemble(agrees=True)).design("cotton", strategy="boundary", resolution=10)
    assert boundary["points"] and all(point["ensemble_information_gain_proxy_bits"] == 0 for point in boundary["points"])


def test_real_ensemble_design_respects_shapes_and_bounds(trained_bundle):
    model = trained_bundle["model"]
    service = ExperimentDesignService(model)
    result = service.design("cotton", flow_range=(0, 200), o2_range=(5, 100), resolution=10)
    ExperimentDesignResponse.model_validate(result)
    assert result["candidate_count"] == 100
    assert len({(p["o2"], p["flow_cm_s"]) for p in result["points"]}) == len(result["points"])
    for point in result["points"]:
        assert 0 <= point["flow_cm_s"] <= 200
        assert point["ensemble_information_gain_proxy_bits"] <= point["predictive_entropy_bits"] + 1e-12


def test_future_model_requires_member_adapter():
    service = ExperimentDesignService(SimpleNamespace())
    with pytest.raises(NotImplementedError, match="adapter"):
        service.design("cotton", resolution=10)
