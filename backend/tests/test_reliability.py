"""Evidence tests for probability diagnostics, provenance and model-card honesty."""

from copy import copy, deepcopy
from dataclasses import replace
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from data.synthetic import generate_synthetic
from models.provenance import (
    build_provenance, dataset_fingerprint, metrics_match_model, stable_fingerprint,
    validate_runtime_provenance,
)
from models.residual_ensemble import ResidualEnsemble
from models.validation import build_validation_report, probability_reliability
from services.model_card import ModelCardService


def test_dataset_fingerprint_ignores_order_but_detects_values_and_duplicates():
    data = generate_synthetic(200)
    first = dataset_fingerprint(data)
    assert first == dataset_fingerprint(data.sample(frac=1, random_state=7).reset_index(drop=True))
    changed = data.copy()
    changed.loc[0, "o2_percent"] = np.nextafter(changed.loc[0, "o2_percent"], np.inf)
    assert first != dataset_fingerprint(changed)
    assert first != dataset_fingerprint(pd.concat([data, data.iloc[[0]]], ignore_index=True))
    assert first != dataset_fingerprint(data.assign(source="nasa"))


def test_probability_reliability_matches_hand_calculation():
    result = probability_reliability([0, 1, 0, 1], [.1, .3, .6, .9], n_bins=2)
    assert result["ece"] == pytest.approx(.275)
    assert result["brier"] == pytest.approx(.2175)
    assert result["bins"] == [
        {"lower": 0., "upper": .5, "count": 2, "mean_probability": .2, "observed_frequency": .5},
        {"lower": .5, "upper": 1., "count": 2, "mean_probability": .75, "observed_frequency": .5},
    ]
    parts = result["brier_decomposition"]
    assert parts["reliability"] == pytest.approx(.07625)
    assert parts["resolution"] == 0
    assert parts["uncertainty"] == .25
    assert parts["binning_residual"] != 0  # Do not claim the unbinned identity.
    assert parts["reconstructed"] + parts["binning_residual"] == pytest.approx(result["brier"])


def test_reliability_includes_empty_bins_and_probability_endpoints():
    result = probability_reliability([0, 1], [0, 1])
    assert sum(item["count"] for item in result["bins"]) == 2
    assert result["bins"][0]["count"] == result["bins"][-1]["count"] == 1
    assert result["bins"][1]["mean_probability"] is None
    assert result["ece"] == 0 and result["brier"] == 0
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError):
        probability_reliability([0, 1], [np.nan, 1])
    with pytest.raises(ValueError):
        probability_reliability([0, 1], [.2, 1.1])


def test_validation_undefined_slices_are_null_and_json_finite():
    data = pd.DataFrame({
        "material": ["unburned", "unburned", "burned", "burned"],
        "experiment_id": ["one", "one", "two", "two"],
        "ignited": [False, False, True, True], "extinguished": False,
        "spread_rate_mm_s": [np.nan, np.nan, 1., 2.],
    })
    values = {
        "mean": np.array([1., 1., 1.1, 1.8]), "probability": np.array([.9, .8, .2, .1]),
        "lower": np.full(4, np.nan), "upper": np.full(4, np.nan),
    }
    report = build_validation_report(data, {"physics_only": values})
    rows = {row["material"]: row for row in report["models"]["physics_only"]["by_material"]}
    assert rows["unburned"]["n_burning"] == 0
    assert rows["unburned"]["log_spread_mae"] is None
    assert rows["unburned"]["extinction_roc_auc"] is None
    assert rows["burned"]["interval_coverage_90"] is None
    json.dumps(report, allow_nan=False)


def test_training_adds_identity_and_complete_oof_diagnostics(trained_bundle):
    model = trained_bundle["model"]
    metrics = trained_bundle["metrics"]
    assert metrics["model_id"] == model.model_id == metrics["provenance"]["model_id"]
    assert metrics["provenance"] == model.provenance
    assert model.provenance["dataset_sha256"] == dataset_fingerprint(trained_bundle["df"])
    assert model.provenance["training_config"]["validation"]["n_splits"] == 3
    assert set(metrics["validation"]["models"]) == set(metrics["models"])
    for name, diagnostics in metrics["validation"]["models"].items():
        assert sum(item["count"] for item in diagnostics["calibration"]["bins"]) == metrics["n_rows"]
        assert diagnostics["calibration"]["brier"] == pytest.approx(metrics["models"][name]["extinction_brier"])
        assert sum(row["n_rows"] for row in diagnostics["by_material"]) == metrics["n_rows"]
        assert sum(row["n_rows"] for row in diagnostics["by_experiment"]) == metrics["n_rows"]
    json.dumps(metrics, allow_nan=False)


def test_fingerprints_reproduce_and_detect_training_configuration(trained_bundle):
    model = trained_bundle["model"]
    validation = model.provenance["training_config"]["validation"]
    rebuilt = build_provenance(model, validation_config=validation)
    assert rebuilt == model.provenance
    changed = build_provenance(model, validation_config=validation | {"n_splits": 5})
    assert changed["dataset_sha256"] == rebuilt["dataset_sha256"]
    assert changed["fitted_models_sha256"] == rebuilt["fitted_models_sha256"]
    assert changed["training_config_sha256"] != rebuilt["training_config_sha256"]
    assert changed["model_id"] != rebuilt["model_id"]
    assert stable_fingerprint({"one": 1, "two": [2., 3.]}) == stable_fingerprint({"two": [2., 3.], "one": 1})


def test_member_predictions_reproduce_public_outputs_and_support(trained_bundle, monkeypatch):
    import models.residual_ensemble as module

    model = trained_bundle["model"]
    inputs = trained_bundle["df"].iloc[::47]
    original_physics = module._physics_prepared
    calls = []

    def count_physics(frame):
        calls.append(len(frame))
        return original_physics(frame)

    monkeypatch.setattr(module, "_physics_prepared", count_physics)
    members = model.predict_members(inputs)
    assert calls == [len(inputs)]
    assert members["extinction_probability"].shape == (model.n_members, len(inputs))
    assert members["log_spread_rate"].shape == (model.n_members, len(inputs))
    batch = model.predict(inputs)
    assert calls == [len(inputs), len(inputs)]
    np.testing.assert_array_equal(members["extinction_probability"].mean(axis=0), batch.extinction_probability)
    np.testing.assert_array_equal(np.exp(members["log_spread_rate"]).mean(axis=0), batch.mean)
    np.testing.assert_array_equal(members["in_data_range"], batch.in_data_range)


def test_model_card_uses_matching_evidence_and_returns_isolated_values(trained_bundle):
    model = trained_bundle["model"]
    service = ModelCardService(model, trained_bundle["metrics"])
    card = service.get_card()
    assert card["validation_status"] == "available"
    assert card["model_id"] == model.model_id
    summary = card["training_data_summary"]
    assert summary["n_rows"] == len(model.training_data)
    assert summary["source_counts"] == {"synthetic": len(model.training_data)}
    assert sum(row["n_rows"] for row in summary["by_material"]) == summary["n_rows"]
    assert card["validation"]["scores"] == trained_bundle["metrics"]["models"]
    assert any("synthetic" in note.lower() for note in card["limitations"])
    card["provenance"]["dataset_sha256"] = "corrupted"
    card["validation"]["scores"].clear()
    assert service.get_card()["provenance"] == model.provenance
    assert service.get_card()["validation"]["scores"] == trained_bundle["metrics"]["models"]
    json.dumps(service.get_card(), allow_nan=False)


@pytest.mark.parametrize("missing", [True, False])
def test_model_card_does_not_invent_or_accept_mismatched_validation(trained_bundle, missing):
    metrics = None if missing else deepcopy(trained_bundle["metrics"])
    if metrics is not None:
        metrics["provenance"]["dataset_sha256"] = "different_dataset_same_row_count"
    card = ModelCardService(trained_bundle["model"], metrics).get_card()
    assert card["validation_status"] == "unavailable" and card["validation"] is None


def test_legacy_card_labels_missing_identity_without_fabricating_validation(trained_bundle):
    legacy = copy(trained_bundle["model"])
    del legacy.provenance
    del legacy.model_id
    card = ModelCardService(legacy).get_card()
    assert card["model_id"] is None and card["provenance"] is None
    assert card["validation_status"] == "unavailable"
    assert any("legacy" in note.lower() for note in card["limitations"])


@pytest.mark.parametrize("component", [
    "dataset_sha256", "physics_material_sha256", "training_config_sha256", "fitted_models_sha256", "model_id",
])
def test_report_matching_rejects_every_component_mismatch(trained_bundle, component):
    model = trained_bundle["model"]
    metrics = deepcopy(trained_bundle["metrics"])
    assert metrics_match_model(model, metrics)
    metrics["provenance"][component] = "wrong_but_row_count_and_version_still_match"
    assert not metrics_match_model(model, metrics)
    assert ModelCardService(model, metrics).get_card()["validation_status"] == "unavailable"


@pytest.mark.parametrize("malformed", [[], {}, "not_a_dictionary", None, {"model_id": "incomplete"}])
def test_report_matching_rejects_malformed_provenance(trained_bundle, malformed):
    metrics = deepcopy(trained_bundle["metrics"])
    metrics["provenance"] = malformed
    assert not metrics_match_model(trained_bundle["model"], metrics)
    model = copy(trained_bundle["model"])
    model.provenance = malformed
    assert not metrics_match_model(model, trained_bundle["metrics"])


def test_legacy_matching_is_allowed_only_when_both_sides_lack_provenance(trained_bundle):
    legacy = copy(trained_bundle["model"])
    del legacy.provenance
    del legacy.model_id
    metrics = deepcopy(trained_bundle["metrics"])
    del metrics["provenance"]
    del metrics["model_id"]
    assert metrics_match_model(legacy, metrics)
    assert not metrics_match_model(trained_bundle["model"], metrics)
    assert not metrics_match_model(legacy, trained_bundle["metrics"])
    metrics["n_rows"] = -1
    assert not metrics_match_model(legacy, metrics)


def test_load_rejects_changed_physics_source_fingerprint(trained_bundle, monkeypatch):
    import models.provenance as provenance_module

    monkeypatch.setattr(provenance_module, "_physics_source_sha256", lambda: "0" * 64)
    with pytest.raises(ValueError, match="physics source.*changed.*retrain"):
        ResidualEnsemble.load(trained_bundle["artifact_dir"] / "model.joblib")


def test_load_rejects_changed_runtime_physics_constants(trained_bundle, monkeypatch):
    import physics.flame_spread as physics_module

    monkeypatch.setattr(physics_module, "CONSTANTS", replace(physics_module.CONSTANTS, gas_conductivity_w_mk=.08))
    with pytest.raises(ValueError, match="physics.*changed.*retrain"):
        ResidualEnsemble.load(trained_bundle["artifact_dir"] / "model.joblib")


def test_load_rejects_changed_material_values(trained_bundle, monkeypatch):
    from data.materials import MATERIALS

    material = MATERIALS["cellulose_thin"]
    monkeypatch.setitem(MATERIALS, material.id, replace(material, specific_heat_j_kgk=material.specific_heat_j_kgk + 1))
    with pytest.raises(ValueError, match="material properties changed.*retrain"):
        ResidualEnsemble.load(trained_bundle["artifact_dir"] / "model.joblib")


def test_load_and_save_reject_changed_ensemble_constants(trained_bundle, monkeypatch, tmp_path):
    import models.residual_ensemble as ensemble_module

    monkeypatch.setattr(ensemble_module, "CONSTANTS", replace(ensemble_module.CONSTANTS, normal_90_quantile=1.5))
    with pytest.raises(ValueError, match="ensemble constants changed.*retrain"):
        ResidualEnsemble.load(trained_bundle["artifact_dir"] / "model.joblib")
    with pytest.raises(ValueError, match="ensemble constants changed.*retrain"):
        trained_bundle["model"].save(tmp_path / "must_not_write.joblib")
    assert not (tmp_path / "must_not_write.joblib").exists()


@pytest.mark.parametrize("malformed", [[], {}, "bad", {"schema_version": True}, {"schema_version": 2}])
def test_runtime_validation_rejects_malformed_new_metadata(trained_bundle, malformed):
    model = copy(trained_bundle["model"])
    model.provenance = malformed
    with pytest.raises(ValueError, match="provenance mismatch.*retrain"):
        validate_runtime_provenance(model)


def test_runtime_validation_allows_legacy_and_other_model_protocols(trained_bundle, tmp_path):
    from types import SimpleNamespace

    legacy = copy(trained_bundle["model"])
    del legacy.provenance
    del legacy.model_id
    validate_runtime_provenance(legacy)
    path = tmp_path / "legacy.joblib"
    joblib.dump(legacy, path)
    loaded = ResidualEnsemble.load(path)
    assert not hasattr(loaded, "provenance")
    np.testing.assert_array_equal(
        loaded.predict(trained_bundle["df"].head()).mean,
        legacy.predict(trained_bundle["df"].head()).mean,
    )
    validate_runtime_provenance(SimpleNamespace(provenance=None))
    # Another model owns its own versioned metadata rather than this tree schema.
    validate_runtime_provenance(SimpleNamespace(provenance={"schema_version": "custom_pinn"}))
