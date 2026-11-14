from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app

SCENARIO = {"material": "cotton", "o2": 21, "pressure_kpa": 101.3, "flow_cm_s": 10}
BOUNDS = {"o2_range": [12, 35], "pressure_range": [50, 115], "flow_range": [.1, 100]}


def test_batch_is_identical_to_single_predictions_and_keeps_order(client):
    points = [SCENARIO, SCENARIO | {"o2": 34, "material": "nomex"}, SCENARIO | {"flow_cm_s": 0}]
    response = client.post("/predict/batch", json={"points": points})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["count"] == 3 and len(result["model_id"]) == 64
    singles = [client.post("/predict", json=point).json() for point in points]
    assert result["predictions"] == singles


def test_analyze_returns_reproducible_comparisons_and_input_uncertainty(client):
    request = {"scenario": SCENARIO, "samples": 32}
    response = client.post("/analyze", json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result == client.post("/analyze", json=request).json()
    assert result["current"] == client.post("/predict", json=SCENARIO).json()
    assert len(result["material_comparison"]) == 5
    assert len(result["preset_comparison"]) == 4
    assert len(result["sensitivity"]) == 3
    assert result["robustness"]["sample_count"] == 40
    assert result["assessment"] in {"supported", "extrapolation"}
    assert result["limitations"] and result["data_source"] == "synthetic"


def test_fixed_point_optimization_reports_both_found_and_unmet_target(client, trained_bundle):
    row = trained_bundle["df"].iloc[0]
    scenario = {name: row[name] for name in ["material", "o2_percent", "pressure_kpa", "flow_cm_s", "thickness_mm"]}
    bounds = {key: [scenario[name], scenario[name]] for key, name in [("o2_range", "o2_percent"), ("pressure_range", "pressure_kpa"), ("flow_range", "flow_cm_s")]}
    request = {"scenario": scenario, "bounds": bounds, "target_extinction_probability": 0., "tolerances": {"o2_percent": 0, "pressure_kpa": 0, "flow_cm_s": 0}}
    response = client.post("/optimize", json=request)
    assert response.status_code == 200, response.text
    found = response.json()
    assert found["evaluated_count"] == found["eligible_count"] == 1
    assert found["status"] == "found" and found["target_achievable"]
    assert found["candidates"][0]["change_cost"] == 0
    assert found["candidates"][0]["robustness"]["support_fraction"] == 1
    not_found = client.post("/optimize", json=request | {"target_extinction_probability": 1.}).json()
    assert not_found["status"] == "not_found" and not not_found["target_achievable"]
    assert not_found["candidates"] == [] and not_found["best_available"] is not None


def test_experiment_design_exposes_information_and_source(client):
    response = client.post("/experiments/design", json={"material": "cotton", "resolution": 10, "k": 3})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["candidate_count"] == 100 and result["strategy"] == "information_gain"
    assert result["points"] and len(result["points"]) <= 3
    for point in result["points"]:
        assert np.isclose(point["ensemble_information_gain_proxy_bits"], point["predictive_entropy_bits"] - point["expected_member_entropy_bits"], atol=1e-12)
    assert result["data_source"] == "synthetic" and len(result["model_id"]) == 64


def test_model_card_contains_real_out_of_fold_evidence(client, trained_bundle):
    response = client.get("/model-card")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["validation_status"] == "available"
    assert result["provenance"]["model_id"] == trained_bundle["model"].model_id
    assert result["validation"]["scores"] == client.get("/metrics").json()["models"]
    assert result["training_data_summary"]["n_rows"] == 800
    for report in result["validation"]["diagnostics"]["models"].values():
        assert sum(group["count"] for group in report["calibration"]["bins"]) == 800


@pytest.mark.parametrize("path,payload", [
    ("/predict/batch", {"points": []}),
    ("/analyze", {"scenario": SCENARIO, "samples": 10000}),
    ("/optimize", {"scenario": SCENARIO, "bounds": BOUNDS, "resolution": 100}),
    ("/experiments/design", {"material": "cotton", "exclude_tested": "false"}),
])
def test_advanced_invalid_requests_return_actionable_422(client, path, payload):
    response = client.post(path, json=payload)
    assert response.status_code == 422
    assert response.json()["detail"] and response.json()["data_source"] == "synthetic"


def test_advanced_inference_without_model_is_503(tmp_path):
    with TestClient(create_app(model_path=tmp_path / "missing.joblib", metrics_path=tmp_path / "missing.json")) as missing:
        for path, payload in [
            ("/predict/batch", {"points": [SCENARIO]}), ("/analyze", {"scenario": SCENARIO}),
            ("/optimize", {"scenario": SCENARIO, "bounds": BOUNDS}),
            ("/experiments/design", {"material": "cotton"}),
        ]:
            response = missing.post(path, json=payload)
            assert response.status_code == 503 and response.json()["data_source"] == "synthetic"
        assert missing.get("/model-card").status_code == 503


@pytest.mark.parametrize("component", ["dataset_sha256", "physics_material_sha256", "training_config_sha256", "fitted_models_sha256", "model_id"])
def test_same_row_count_cannot_disguise_wrong_metrics(trained_bundle, component):
    metrics = deepcopy(trained_bundle["metrics"])
    metrics["provenance"][component] = "0" * 64
    with TestClient(create_app(predictor=trained_bundle["model"], metrics=metrics)) as session:
        assert session.get("/metrics").status_code == 503
        card = session.get("/model-card").json()
        assert card["validation_status"] == "unavailable" and card["validation"] is None
        assert session.post("/predict", json=SCENARIO).status_code == 200


def test_alternative_predictor_retains_dossiers_but_needs_information_adapter(trained_bundle):
    model = trained_bundle["model"]
    alternative = SimpleNamespace(model_version=model.model_version, data_source=model.data_source, training_data=model.training_data, predict=model.predict, support_distance=model.support_distance)
    with TestClient(create_app(predictor=alternative)) as session:
        assert session.post("/analyze", json={"scenario": SCENARIO, "samples": 32}).status_code == 200
        card = session.get("/model-card")
        assert card.status_code == 200 and card.json()["ensemble_size"] is None
        assert card.json()["model_id"] is None and card.json()["validation_status"] == "unavailable"
        unsupported = session.post("/experiments/design", json={"material": "cotton", "resolution": 10})
        assert unsupported.status_code == 501 and "adapter" in unsupported.json()["detail"]


def test_future_model_without_retained_observations_has_readable_model_card(trained_bundle):
    import pandas as pd

    model = trained_bundle["model"]
    alternative = SimpleNamespace(model_version="future-pinn", data_source="nasa", training_data=pd.DataFrame(), predict=model.predict, support_distance=model.support_distance)
    with TestClient(create_app(predictor=alternative)) as session:
        response = session.get("/model-card")
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["validation_status"] == "unavailable"
        assert result["training_data_summary"]["n_rows"] == 0
        assert result["training_data_summary"]["by_material"] == []
        assert result["training_data_summary"]["quality_limitations"]


def test_changed_runtime_physics_disables_stale_model_instead_of_reusing_identity(trained_bundle, monkeypatch):
    monkeypatch.setattr("models.provenance.physics_material_fingerprint", lambda: "0" * 64)
    with TestClient(create_app(predictor=trained_bundle["model"], metrics=trained_bundle["metrics"])) as session:
        result = session.get("/health").json()
        assert result["status"] == "model_not_loaded" and result["model_loaded"] is False
        assert session.post("/predict", json=SCENARIO).status_code == 503


def test_request_trace_is_exposed_to_frontend_for_success_and_validation(client):
    for path, payload in [("/predict", SCENARIO), ("/predict/batch", {"points": []})]:
        response = client.post(path, json=payload, headers={"X-Request-ID": "pitch-demo-42", "Origin": "http://localhost:3000"})
        assert response.headers["X-Request-ID"] == "pitch-demo-42"
        assert "Server-Timing" in response.headers["Access-Control-Expose-Headers"]
        assert float(response.headers["Server-Timing"].split("=")[1]) >= 0


def test_unexpected_error_retains_source_trace_and_cors_without_leaking_details(trained_bundle):
    model = trained_bundle["model"]
    def fail(frame):
        raise RuntimeError("PRIVATE_ERROR_DETAIL")
    broken = SimpleNamespace(model_version=model.model_version, data_source=model.data_source, training_data=model.training_data, predict=fail)
    with TestClient(create_app(predictor=broken), raise_server_exceptions=False) as session:
        response = session.post("/predict", json=SCENARIO, headers={"X-Request-ID": "failure-demo", "Origin": "http://localhost:5173"})
        assert response.status_code == 500
        assert response.headers["X-Request-ID"] == "failure-demo"
        assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
        assert response.json()["data_source"] == "synthetic"
        assert "PRIVATE_ERROR_DETAIL" not in response.text
