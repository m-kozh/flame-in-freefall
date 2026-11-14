"""Source-aware OOF reporting and retained NASA evidence must remain honest."""

from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data.synthetic import generate_synthetic
from models.validation import build_validation_report
from services.model_card import ModelCardService


def _report_inputs():
    frame = pd.DataFrame({
        "material": ["cellulose_thin"] * 4,
        "experiment_id": ["NASA-CR-182185"] * 2 + ["synthetic-batch"] * 2,
        "source": ["nasa", "nasa", "synthetic", "synthetic"],
        "ignited": [True, True, False, False], "extinguished": False,
        "spread_rate_mm_s": [1., 2., np.nan, np.nan],
    })
    values = {
        "mean": np.array([1.2, 1.8, 1., 1.]),
        "probability": np.array([.1, .2, .8, .9]),
        "lower": np.full(4, .5), "upper": np.full(4, 2.5),
    }
    physics = values | {"lower": np.full(4, np.nan), "upper": np.full(4, np.nan)}
    return frame, {"physics_only": physics, "ml_only": values, "physics_plus_ml_residual": values}


def _card_model(source="mixed", snapshot=True):
    data = generate_synthetic(200).head(4).copy()
    data.loc[:1, "source"] = "nasa"
    data.loc[:1, "campaign"] = "NASA-CR-182185"
    data.loc[:1, "experiment_id"] = "NASA-CR-182185-whole-report"
    model = SimpleNamespace(model_version="test-model", data_source=source, training_data=data)
    if snapshot:
        model.training_evidence = {
            "datasets": [{"id": "nasa-cr-182185", "training_row_count": 2}],
            "source_counts": {"nasa": 2, "synthetic": 2}, "snapshot_note": "retained at fit time",
        }
    return model


def _fake_registry(monkeypatch, summary):
    module = ModuleType("data.nasa")
    calls = []
    def read(frame):
        calls.append(frame)
        return summary
    module.training_evidence_summary = read
    monkeypatch.setitem(sys.modules, "data.nasa", module)
    return calls


def test_source_slices_use_same_oof_arrays_and_match_hand_calculated_scores():
    frame, predictions = _report_inputs()
    report = build_validation_report(frame, predictions)
    for name, diagnostics in report["models"].items():
        slices = {item["source"]: item for item in diagnostics["by_source"]}
        assert set(slices) == {"nasa", "synthetic"}
        assert sum(row["n_rows"] for row in slices.values()) == len(frame)
        nasa = slices["nasa"]
        assert nasa["n_burning"] == 2 and nasa["n_nonburning"] == 0
        assert nasa["n_experiments"] == 1
        error = np.log([1.2, 1.8]) - np.log([1., 2.])
        assert nasa["log_spread_mae"] == pytest.approx(np.mean(np.abs(error)))
        assert nasa["log_spread_rmse"] == pytest.approx(np.sqrt(np.mean(error ** 2)))
        assert nasa["extinction_brier"] == pytest.approx(.025)
        assert nasa["extinction_roc_auc"] is None  # A single class has no defined AUC.
        assert nasa["interval_coverage_90"] == (None if name == "physics_only" else 1.)
        synthetic = slices["synthetic"]
        assert synthetic["log_spread_mae"] is None
        assert synthetic["log_spread_rmse"] is None
        assert synthetic["interval_coverage_90"] is None
        assert synthetic["extinction_roc_auc"] is None
        assert synthetic["extinction_brier"] == pytest.approx(.025)
    json.dumps(report, allow_nan=False)


def test_unknown_source_column_does_not_invent_a_nasa_slice():
    frame, predictions = _report_inputs()
    report = build_validation_report(frame.drop(columns="source"), predictions)
    assert all(item["by_source"] == [] for item in report["models"].values())


def test_material_source_counts_show_nasa_coverage_without_global_claim():
    model = _card_model()
    card = ModelCardService(model).get_card()
    summary = card["training_data_summary"]
    material = summary["by_material"][0]
    assert summary["source_counts"] == {"nasa": 2, "synthetic": 2}
    assert material["source_counts"] == {"nasa": 2, "synthetic": 2}
    assert sum(material["source_counts"].values()) == material["n_rows"]
    notes = " ".join(card["limitations"])
    assert "5.18-second" in notes and "one thin cellulose" in notes
    assert "not an overall NASA validation" in notes
    assert card["validation_status"] == "unavailable"  # No made-up OOF evidence.


@pytest.mark.parametrize("change", ["other_report", "no_nasa_rows", "synthetic_model"])
def test_short_window_limitation_is_tied_to_the_report_actually_in_training(change):
    model = _card_model()
    if change == "other_report":
        model.training_data.loc[:1, ["campaign", "experiment_id"]] = ["BASS-II", "different-report"]
    elif change == "no_nasa_rows":
        model.training_data["source"] = "synthetic"
    else:
        model.data_source = "synthetic"
    notes = " ".join(ModelCardService(model).get_card()["limitations"])
    assert "5.18-second" not in notes


def test_training_snapshot_is_preferred_to_current_registry_and_copied(monkeypatch):
    model = _card_model()
    expected = deepcopy(model.training_evidence)
    calls = _fake_registry(monkeypatch, {"datasets": [{"id": "changed-registry"}]})
    service = ModelCardService(model)
    card = service.get_card()
    assert card["training_evidence"] == expected and calls == []
    card["training_evidence"]["datasets"][0]["id"] = "mutated-response"
    assert model.training_evidence == expected
    assert service.get_card()["training_evidence"] == expected


def test_saved_provenance_snapshot_is_used_when_attribute_is_missing(monkeypatch):
    model = _card_model(snapshot=False)
    snapshot = {"datasets": [{"id": "nasa-cr-182185"}], "snapshot_note": "stored manifest"}
    model.provenance = {"training_config": {"training_evidence": snapshot}}
    calls = _fake_registry(monkeypatch, {"datasets": []})
    card = ModelCardService(model).get_card()
    assert card["training_evidence"] == snapshot and calls == []


def test_legacy_evidence_is_explicitly_current_registry_derived(monkeypatch):
    model = _card_model(snapshot=False)
    current = {"datasets": [{"id": "nasa-cr-182185"}], "source_counts": {"nasa": 2, "synthetic": 2}}
    calls = _fake_registry(monkeypatch, current)
    service = ModelCardService(model)
    evidence = service.get_card()["training_evidence"]
    assert len(calls) == 1 and calls[0] is model.training_data
    assert evidence["metadata_origin"] == "current_registry_derived"
    assert "not proof" in evidence["metadata_note"]
    assert "metadata_origin" not in current
    evidence["datasets"].clear()
    assert current["datasets"] == [{"id": "nasa-cr-182185"}]
