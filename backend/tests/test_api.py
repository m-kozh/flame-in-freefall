"""Exercise the public contract through HTTP, including frontend compatibility."""

from copy import deepcopy
import json

from fastapi.testclient import TestClient
import pytest

from app.main import create_app


PREDICT = {
    "material": "cotton", "o2": 21,
    "pressure_kpa": 101.3, "flow_cm_s": 10,
}


def assert_response(response, status=200):
    assert response.status_code == status, response.text
    body = response.json()
    assert body["data_source"] == "synthetic"
    return body


@pytest.mark.parametrize("path", ["/health", "/materials", "/presets", "/metrics"])
def test_public_metadata_endpoints(client, trained_bundle, path):
    body = assert_response(client.get(path))
    if path == "/health":
        assert body["status"] == "ok" and body["model_loaded"]
    elif path == "/materials":
        assert len(body["materials"]) == 5
        assert any("cotton" in material["aliases"] for material in body["materials"])
    elif path == "/presets":
        assert len(body["presets"]) == 4
        assert any(preset["illustrative"] for preset in body["presets"])
    else:
        # Response validation must preserve notes, seeds and exact split audits.
        assert body == trained_bundle["metrics"]
        assert set(body["models"]) == {"physics_only", "ml_only", "physics_plus_ml_residual"}


def test_prediction_compatibility_and_aliases(client):
    frontend = assert_response(client.post("/predict", json=PREDICT))
    canonical = dict(PREDICT, material="sibal_cotton_fiberglass", o2_percent=21)
    canonical.pop("o2")
    assert assert_response(client.post("/predict", json=canonical)) == frontend
    assert assert_response(client.post("/predict", json=PREDICT | {"o2_percent": 21})) == frontend
    assert isinstance(frontend["spread_rate_mm_s"], float)
    interval = frontend["spread_rate_interval"]
    assert interval["lower"] <= frontend["spread_rate_mm_s"] == interval["mean"] <= interval["upper"]
    assert frontend["spread_ci"] == [interval["lower"], interval["upper"]]
    assert frontend["extinction_prob"] == frontend["extinction_probability"]
    assert frontend["extinction_ci"][0] <= frontend["extinction_prob"] <= frontend["extinction_ci"][1]
    assert frontend["inputs"]["thickness_mm"] == 0.33
    assert (frontend["regime"] == "burning") == (frontend["extinction_prob"] < 0.5)
    detailed = assert_response(client.post("/predict/detailed", json=PREDICT))
    assert detailed["spread_rate_mm_s"] == interval
    assert {key: value for key, value in detailed.items() if key != "spread_rate_mm_s"} == {
        key: value for key, value in frontend.items() if key != "spread_rate_mm_s"
    }


def test_get_and_post_maps_have_same_axes_and_cells(client):
    query = {"material": "cotton", "pressure_kpa": 70, "resolution": 10}
    get_map = assert_response(client.get("/map", params=query))
    post_map = assert_response(client.post("/map", json=query))
    assert get_map == post_map
    assert get_map["x_axis"] == "o2" and get_map["y_axis"] == "flow_cm_s"
    assert get_map["x"] == get_map["o2_values"]
    assert get_map["y"] == get_map["flow_values"]
    assert get_map["z"] == get_map["extinction_probability"]
    for matrix in (get_map["z"], get_map["spread_rate"], get_map["uncertainty"]):
        assert len(matrix) == len(get_map["y"]) == 10
        assert all(len(row) == len(get_map["x"]) == 10 for row in matrix)
    assert all(point["source"] == "synthetic" for point in get_map["tested_points"])


@pytest.mark.parametrize("k", [1, 20])
def test_suggest_accepts_both_count_boundaries(client, k):
    body = assert_response(client.get("/suggest", params={"material": "cotton", "k": k}))
    assert len(body["points"]) == k
    assert len({(point["o2"], point["flow_cm_s"]) for point in body["points"]}) == k
    assert all(point["o2"] == point["o2_percent"] and point["reason"] for point in body["points"])


@pytest.mark.parametrize("target", ["spread_rate", "extinction"])
def test_get_and_post_explanations_are_additive(client, target):
    global_result = assert_response(client.get("/explain", params={"target": target}))
    local = assert_response(client.post("/explain", json=PREDICT | {"target": target}))
    assert global_result["local_contributions"] == []
    assert global_result["base_value"] is None and global_result["prediction"] is None
    assert local["global_importance"] == global_result["global_importance"]
    assert local["features"] == [
        {"name": item["feature"], "importance": item["importance"]}
        for item in local["global_importance"]
    ]
    assert sum(item["importance"] for item in local["global_importance"]) == pytest.approx(1)
    assert local["base_value"] + sum(item["contribution"] for item in local["local_contributions"]) == pytest.approx(local["prediction"])


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/predict", PREDICT | {"o2_percent": 22}),
        ("post", "/predict", PREDICT | {"o2": True}),
        ("post", "/predict", PREDICT | {"pressure_kpa": 151}),
        ("post", "/predict", PREDICT | {"material": "wood"}),
        ("post", "/predict", PREDICT | {"unknown_field": 1}),
        ("post", "/predict/detailed", {"material": "cotton"}),
        ("post", "/map", {"material": "cotton", "flow_range": [2, 1]}),
        ("post", "/map", {"material": "cotton", "o2_range": [30, 30]}),
        ("post", "/map", {"material": "cotton", "resolution": True}),
        ("get", "/map", {"material": "cotton", "flow_min": 0}),
        ("get", "/map", {"material": "cotton", "resolution": 4}),
        ("get", "/map", {"material": "cotton", "resolution": 101}),
        ("get", "/suggest", {"material": "cotton", "k": 0}),
        ("get", "/suggest", {"material": "cotton", "k": 21}),
        ("get", "/suggest", {"material": "cotton", "extra": "rejected"}),
        ("get", "/explain", {"target": "not_a_target"}),
        ("post", "/explain", {"material": "cotton"}),
    ],
)
def test_invalid_inputs_return_clear_422(client, method, path, payload):
    kwargs = {"params" if method == "get" else "json": payload}
    body = assert_response(client.request(method, path, **kwargs), 422)
    assert body["detail"] and isinstance(body["detail"], list)
    assert all(error["loc"] and error["msg"] for error in body["detail"])
    assert all(error["loc"][0] == ("query" if method == "get" else "body") for error in body["detail"])


@pytest.mark.parametrize("literal", ["NaN", "Infinity"])
def test_nonfinite_json_inputs_return_serializable_validation_errors(client, literal):
    # httpx intentionally rejects nonfinite values through its json= helper.
    content = json.dumps(PREDICT).replace('"o2": 21', f'"o2": {literal}')
    body = assert_response(client.post("/predict", content=content, headers={"Content-Type": "application/json"}), 422)
    assert any(error["type"] == "finite_number" for error in body["detail"])
    json.dumps(body, allow_nan=False)


@pytest.mark.parametrize("origin", ["http://localhost:3000", "http://localhost:5173"])
def test_cors_accepts_frontend_origins(client, origin):
    response = client.options("/predict", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_cors_rejects_unlisted_origin(client):
    response = client.options("/predict", headers={
        "Origin": "https://unlisted.example", "Access-Control-Request-Method": "POST",
    })
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_missing_model_keeps_health_and_returns_inference_503(tmp_path):
    with TestClient(create_app(model_path=tmp_path / "missing.joblib", metrics_path=tmp_path / "missing.json")) as missing:
        health = assert_response(missing.get("/health"))
        assert health["status"] == "model_not_loaded" and not health["model_loaded"]
        requests = [
            ("post", "/predict", {"json": PREDICT}),
            ("post", "/predict/detailed", {"json": PREDICT}),
            ("get", "/map", {"params": {"material": "cotton"}}),
            ("get", "/suggest", {"params": {"material": "cotton"}}),
            ("get", "/explain", {}),
            ("get", "/metrics", {}),
        ]
        for method, path, kwargs in requests:
            body = assert_response(missing.request(method, path, **kwargs), 503)
            assert "train_model.py" in body["detail"]


@pytest.mark.parametrize("mismatch", [None, "model_version", "data_source", "n_rows"])
def test_missing_or_mismatched_metrics_return_503(trained_bundle, tmp_path, mismatch):
    metrics = None if mismatch is None else deepcopy(trained_bundle["metrics"])
    if mismatch is not None:
        metrics[mismatch] = {"model_version": "other-version", "data_source": "nasa", "n_rows": -1}[mismatch]
    with TestClient(create_app(
        predictor=trained_bundle["model"], metrics=metrics,
        metrics_path=tmp_path / "missing.json",
    )) as session:
        assert_response(session.get("/health"))
        body = assert_response(session.get("/metrics"), 503)
        assert "Metrics" in body["detail"]


def test_openapi_exposes_fixed_prediction_formats(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    schemas = spec["components"]["schemas"]
    assert schemas["PredictResponse"]["properties"]["spread_rate_mm_s"]["type"] == "number"
    assert schemas["DetailedPredictResponse"]["properties"]["spread_rate_mm_s"]["$ref"].endswith("/SpreadInterval")
    assert set(schemas["SpreadInterval"]["properties"]) == {"mean", "lower", "upper"}
    assert schemas["PredictRequest"]["additionalProperties"] is False
    assert schemas["PredictRequest"]["properties"]["o2"]["type"] == "number"
    assert {tuple(option["required"]) for option in schemas["PredictRequest"]["anyOf"]} == {("o2",), ("o2_percent",)}
    for path, name in [("/predict", "PredictResponse"), ("/predict/detailed", "DetailedPredictResponse")]:
        schema = spec["paths"][path]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        assert schema["$ref"].endswith(f"/{name}")


def test_alternate_predictor_keeps_api_and_requires_own_explanation_adapter(trained_bundle):
    """A future non-tree implementation can reuse inference routes unchanged."""
    from types import SimpleNamespace

    delegate = trained_bundle["model"]
    alternate = SimpleNamespace(
        model_version=delegate.model_version, data_source=delegate.data_source,
        training_data=delegate.training_data, predict=delegate.predict,
        support_distance=delegate.support_distance,
    )
    with TestClient(create_app(predictor=alternate, metrics=trained_bundle["metrics"])) as session:
        assert_response(session.post("/predict", json=PREDICT))
        assert_response(session.get("/map", params={"material": "cotton", "resolution": 5}))
        assert_response(session.get("/suggest", params={"material": "cotton", "k": 2}))
        result = assert_response(session.get("/explain"), 501)
        assert "adapter" in result["detail"]
