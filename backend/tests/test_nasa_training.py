from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from data.nasa import nasa_training_rows
from data.synthetic import generate_synthetic
from models.residual_ensemble import ResidualEnsemble
from models.train import train_and_evaluate
from scripts.validate_nasa import evaluate_nasa_external


@pytest.fixture(scope="module")
def nasa_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mixed_nasa")
    frame = pd.concat([generate_synthetic(400), nasa_training_rows()], ignore_index=True)
    metrics = train_and_evaluate(frame, directory, n_members=2, n_splits=3)
    return ResidualEnsemble.load(directory / "model.joblib"), metrics


def test_real_observations_reach_fitted_artifact_and_evidence_snapshot(nasa_bundle):
    model, metrics = nasa_bundle
    assert model.data_source == metrics["data_source"] == "mixed"
    assert len(model.training_data) == 411
    assert model.training_evidence["verified_curated_row_count"] == 11
    assert model.training_evidence["unverified_nasa_row_count"] == 0
    assert model.training_evidence["observation_window_s"] == 5.18
    assert model.provenance["training_config"]["training_evidence"] == model.training_evidence
    assert metrics["training_evidence"] == model.training_evidence


def test_nasa_report_is_held_out_whole_and_scored_separately(nasa_bundle):
    _, metrics = nasa_bundle
    containing_test = [fold for fold in metrics["cv"]["folds"] if "NASA-CR-182185" in fold["test_experiment_ids"]]
    assert len(containing_test) == 1
    assert "NASA-CR-182185" not in containing_test[0]["train_experiment_ids"]
    for report in metrics["validation"]["models"].values():
        nasa = next(part for part in report["by_source"] if part["source"] == "nasa")
        assert nasa["n_rows"] == 11 and nasa["n_burning"] == 8 and nasa["n_nonburning"] == 3
        assert nasa["n_experiments"] == 1
        assert len(report["by_source"]) == 2


def test_mixed_api_exposes_real_training_use_and_original_map_observations(nasa_bundle):
    model, metrics = nasa_bundle
    with TestClient(create_app(predictor=model, metrics=metrics)) as client:
        response = client.get("/datasets").json()
        cards = {card["id"]: card for card in response["datasets"]}
        assert cards["nasa-cr-182185"]["included_in_loaded_model"]
        assert not cards["psi-25-bass-ii"]["included_in_loaded_model"]
        card = client.get("/model-card").json()
        assert card["data_source"] == "mixed" and card["validation_status"] == "available"
        cellulose = next(row for row in card["training_data_summary"]["by_material"] if row["material"] == "cellulose_thin")
        assert cellulose["source_counts"] == {"synthetic": 80, "nasa": 11}
        grid = client.get("/map", params={"material": "cellulose_thin", "pressure_kpa": 101.325, "thickness_mm": .076, "resolution": 5}).json()
        actual = [point for point in grid["tested_points"] if point["source"] == "nasa"]
        assert len(actual) == 11
        assert all(point["campaign"] == "NASA-CR-182185" and point["thickness_mm"] == .076 for point in actual)
        point = nasa_training_rows().iloc[0]
        prediction = client.post("/predict", json={key: point[key] for key in ["material", "o2_percent", "pressure_kpa", "flow_cm_s", "thickness_mm"]}).json()
        assert prediction["data_source"] == "mixed"
        assert prediction["in_data_range"]


def test_training_evidence_cannot_be_relabelled_after_fitting(nasa_bundle, monkeypatch, tmp_path):
    model, _ = nasa_bundle
    forged = deepcopy(model.training_evidence)
    forged["observation_window_s"] = 100000
    with monkeypatch.context() as patch:
        patch.setattr(model, "training_evidence", forged)
        with pytest.raises(ValueError, match="evidence"):
            model.save(tmp_path / "forged.joblib")


def test_training_predictions_cannot_be_called_external_nasa_validation(nasa_bundle):
    model, _ = nasa_bundle
    with pytest.raises(ValueError, match="without NASA-CR-182185"):
        evaluate_nasa_external(model)


@pytest.mark.parametrize("renamed_ids,relabeled_source", [(False, False), (True, False), (True, True)])
def test_external_validation_detects_report_aliases_and_relabelled_values(renamed_ids, relabeled_source):
    data = nasa_training_rows()
    data["campaign"] = "import-alias"
    if renamed_ids:
        data["experiment_id"] = "new-batch"
    if relabeled_source:
        data["source"] = "synthetic"
        data.loc[data.extinguished, "spread_rate_mm_s"] = 0.
    with pytest.raises(ValueError, match="potential report or observation overlap"):
        evaluate_nasa_external(SimpleNamespace(training_data=data))
