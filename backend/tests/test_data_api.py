"""Evidence counts and missingness are checked independently of model training."""

from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.data_routes import router
from data.nasa import nasa_training_rows

EXPECTED = {"nasa-cr-182185": (13, 11), "psi-25-bass-ii": (129, 0), "bass-ii-tm-20210011385": (41, 0)}


def evidence_client(training_data=None, evidence=None):
    application = FastAPI()
    application.include_router(router)
    application.state.predictor = None if training_data is None else SimpleNamespace(training_data=training_data, training_evidence=evidence, data_source="mixed")
    return TestClient(application)


def test_catalog_remains_available_without_a_model():
    with evidence_client() as client:
        response = client.get("/datasets")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["data_source"] == "nasa"
    cards = {row["id"]: row for row in result["datasets"]}
    assert set(cards) == set(EXPECTED)
    for identifier, (count, training_count) in EXPECTED.items():
        assert cards[identifier]["record_count"] == count
        assert cards[identifier]["training_row_count"] == training_count
        assert cards[identifier]["included_in_loaded_model"] is False
        assert cards[identifier]["source_url"].startswith("https://")
        assert cards[identifier]["source_files"] and cards[identifier]["limitations"]


@pytest.mark.parametrize("identifier,counts", EXPECTED.items())
def test_dataset_details_and_observations_preserve_real_record_counts(identifier, counts):
    with evidence_client() as client:
        detail = client.get(f"/datasets/{identifier}")
        observations = client.get(f"/datasets/{identifier}/observations")
    assert detail.status_code == observations.status_code == 200
    assert detail.json()["data_source"] == observations.json()["data_source"] == "nasa"
    assert detail.json()["dataset"]["record_count"] == observations.json()["count"] == counts[0]
    assert len(observations.json()["observations"]) == counts[0]
    assert observations.json()["dataset_id"] == identifier
    records = observations.json()["observations"]
    assert len({row["record_id"] for row in records}) == len(records)
    assert all(isinstance(row["source_reference"], dict) and row["source_reference"] for row in records)
    assert all(isinstance(row["training_eligible"], bool) for row in records)
    assert all(row["exclusion_reason"] for row in records if not row["training_eligible"])


def test_censored_and_extinguished_rates_remain_null_in_evidence():
    with evidence_client() as client:
        records = client.get("/datasets/nasa-cr-182185/observations").json()["observations"]
    censored = [row for row in records if not row["training_eligible"]]
    assert len(censored) == 2
    assert all(row["spread_rate_mm_s"] is None for row in censored)
    assert all(row["spread_rate_upper_bound_mm_s"] == .5 for row in censored)
    assert all(row["rate_observability"] == "left_censored_below_0.5_mm_s" for row in censored)
    eligible = [row for row in records if row["training_eligible"]]
    assert len(eligible) == 11
    assert sum(row["spread_rate_mm_s"] is None for row in eligible) == 3
    assert sum(row["spread_rate_mm_s"] is not None for row in eligible) == 8
    assert records[0]["spread_rate_mm_s"] == 8.7  # .87 cm/s, explicitly converted.


@pytest.mark.parametrize("path", ["/datasets/does-not-exist", "/datasets/does-not-exist/observations"])
def test_unknown_dataset_has_nasa_error_source_even_with_mixed_model(path):
    training_data = pd.DataFrame({"source": ["nasa"], "campaign": ["NASA-CR-182185"]})
    with evidence_client(training_data) as client:
        response = client.get(path)
    assert response.status_code == 404
    assert response.json()["data_source"] == "nasa"
    assert "GET /datasets" in response.json()["detail"]


@pytest.mark.parametrize("change,included", [(None, True), ("source", False), ("campaign", False), ("o2_percent", False)])
def test_inclusion_is_based_on_actual_curated_training_values(change, included):
    training_data = nasa_training_rows()
    if change is not None:
        training_data = training_data.copy()
        training_data[change] = {"source": "synthetic", "campaign": "another-campaign", "o2_percent": 50.}[change]
    with evidence_client(training_data) as client:
        cards = {row["id"]: row for row in client.get("/datasets").json()["datasets"]}
    assert cards["nasa-cr-182185"]["included_in_loaded_model"] is included
    assert not cards["psi-25-bass-ii"]["included_in_loaded_model"]
    assert not cards["bass-ii-tm-20210011385"]["included_in_loaded_model"]


def test_recorded_verified_training_snapshot_takes_precedence():
    evidence = {"verified_curated_row_count": 11, "dataset_ids": ["nasa-cr-182185"], "unverified_nasa_row_count": 0}
    with evidence_client(pd.DataFrame(), evidence) as client:
        result = client.get("/datasets/nasa-cr-182185").json()
    assert result["dataset"]["included_in_loaded_model"] is True
    with evidence_client(nasa_training_rows(), evidence | {"verified_curated_row_count": 0}) as client:
        result = client.get("/datasets/nasa-cr-182185").json()
    assert result["dataset"]["included_in_loaded_model"] is False


def test_empty_or_nonstandard_external_training_data_does_not_break_metadata():
    for frame in [pd.DataFrame(), pd.DataFrame({"unrelated": [1]})]:
        with evidence_client(frame) as client:
            result = client.get("/datasets/nasa-cr-182185")
        assert result.status_code == 200
        assert result.json()["dataset"]["included_in_loaded_model"] is False


def test_openapi_documents_evidence_shapes_and_error_envelopes():
    with evidence_client() as client:
        spec = client.get("/openapi.json").json()
    for path in ["/datasets", "/datasets/{dataset_id}", "/datasets/{dataset_id}/observations"]:
        responses = spec["paths"][path]["get"]["responses"]
        assert {"200", "404", "422"}.issubset(responses)
    fields = spec["components"]["schemas"]["DatasetCard"]["properties"]
    assert {"source_files", "training_row_count", "included_in_loaded_model", "observation_window_s"}.issubset(fields)
