"""Scientific and service contracts, using independently generated test data."""

from collections import Counter
from copy import copy

import numpy as np
import pandas as pd
import pytest

import models.residual_ensemble as ensemble_module
from data.synthetic import generate_synthetic
from models.base import Predictor
from models.residual_ensemble import RAW_FEATURES, ResidualEnsemble
from models.train import grouped_splits
from services.explain import ExplainService
from services.flammability_map import FlammabilityMapService, _boundary_points
from services.suggest import SuggestService


def test_group_kfold_holds_out_whole_experiments_once(trained_bundle):
    data = trained_bundle["df"]
    test_counts = np.zeros(len(data), dtype=int)
    for train_indices, test_indices in grouped_splits(data, n_splits=3):
        training_ids = set(data.iloc[train_indices].experiment_id)
        testing_ids = set(data.iloc[test_indices].experiment_id)
        assert training_ids.isdisjoint(testing_ids)
        assert training_ids | testing_ids == set(data.experiment_id)
        expected_test_rows = np.flatnonzero(data.experiment_id.isin(testing_ids))
        np.testing.assert_array_equal(np.sort(test_indices), expected_test_rows)
        test_counts[test_indices] += 1
    np.testing.assert_array_equal(test_counts, np.ones(len(data), dtype=int))


def test_training_metrics_preserve_real_group_split_audit(trained_bundle):
    metrics = trained_bundle["metrics"]
    data = trained_bundle["df"]
    assert metrics["cv"]["method"] == "GroupKFold"
    assert metrics["cv"]["group_column"] == "experiment_id"
    test_group_counts = Counter()
    for fold in metrics["cv"]["folds"]:
        training_ids = set(fold["train_experiment_ids"])
        testing_ids = set(fold["test_experiment_ids"])
        assert training_ids.isdisjoint(testing_ids)
        assert training_ids | testing_ids == set(data.experiment_id)
        assert fold["test_rows"] == int(data.experiment_id.isin(testing_ids).sum())
        assert fold["train_rows"] + fold["test_rows"] == len(data)
        test_group_counts.update(testing_ids)
    assert test_group_counts == Counter({name: 1 for name in data.experiment_id.unique()})
    assert set(metrics["models"]) == {"physics_only", "ml_only", "physics_plus_ml_residual"}
    assert metrics["models"]["physics_only"]["interval_coverage_90"] is None


def test_ml_only_uses_no_physics_and_bootstraps_complete_groups(trained_bundle, monkeypatch):
    data = trained_bundle["df"]
    sampled_row_indices = []
    original_fit = ensemble_module.LGBMClassifier.fit

    def record_classifier_rows(classifier, X, y, *args, **kwargs):
        sampled_row_indices.append(X.index.to_list())
        return original_fit(classifier, X, y, *args, **kwargs)

    def physics_must_not_be_used(*args, **kwargs):
        raise AssertionError("The ML-only learner consulted the physics baseline.")

    monkeypatch.setattr(ensemble_module.LGBMClassifier, "fit", record_classifier_rows)
    monkeypatch.setattr(ensemble_module, "_physics_prepared", physics_must_not_be_used)
    model = ResidualEnsemble(n_members=2, use_physics=False).fit(data)
    assert model.feature_names == RAW_FEATURES
    assert model.feature_frame(data.head()).columns.to_list() == RAW_FEATURES
    assert len(sampled_row_indices) == 2
    for row_indices, group_draw in zip(sampled_row_indices, model.bootstrap_groups, strict=True):
        assert len(group_draw) == data.experiment_id.nunique()
        assert set(group_draw) <= set(data.experiment_id)
        row_counts = Counter(row_indices)
        group_counts = Counter(group_draw)
        # Each row of a selected experiment appears equally often, including
        # multiplicity when a complete experiment is selected more than once.
        for index, experiment_id in enumerate(data.experiment_id):
            assert row_counts[index] == group_counts[experiment_id]


def test_prediction_intervals_are_finite_ordered_and_probabilities_bounded(trained_bundle):
    model = trained_bundle["model"]
    assert isinstance(model, Predictor)
    batch = model.predict(trained_bundle["df"].iloc[::13])
    for values in [batch.mean, batch.lower, batch.upper, batch.extinction_probability, batch.uncertainty, batch.log_std]:
        assert np.isfinite(values).all()
    assert (batch.lower > 0).all()
    assert (batch.lower <= batch.mean).all()
    assert (batch.mean <= batch.upper).all()
    assert (batch.extinction_lower <= batch.extinction_probability).all()
    assert (batch.extinction_probability <= batch.extinction_upper).all()
    assert (batch.extinction_lower >= 0).all()
    assert (batch.extinction_upper <= 1).all()
    assert ((batch.uncertainty >= 0) & (batch.uncertainty <= .5)).all()
    assert model.noise_estimation == "group_out_of_bag"
    assert model.oob_prediction_rows > 0


def test_model_save_load_preserves_predictions_without_source_csv(trained_bundle, tmp_path):
    model = trained_bundle["model"]
    inputs = trained_bundle["df"].iloc[::67]
    before = model.predict(inputs)
    path = tmp_path / "nested" / "model.joblib"
    model.save(path)
    loaded = ResidualEnsemble.load(path)
    after = loaded.predict(inputs)
    for field in ["mean", "lower", "upper", "extinction_probability", "uncertainty", "physics_baseline"]:
        np.testing.assert_allclose(getattr(before, field), getattr(after, field))
    np.testing.assert_array_equal(before.in_data_range, after.in_data_range)
    pd.testing.assert_frame_equal(loaded.training_data, model.training_data)
    assert loaded.model_version == model.model_version
    assert loaded.data_source == model.data_source


def test_training_observation_supported_and_extreme_query_flagged(trained_bundle):
    model = trained_bundle["model"]
    observed = trained_bundle["df"].iloc[[0]].copy()
    extrapolated = observed.assign(o2_percent=100., pressure_kpa=150., flow_cm_s=200.)
    inputs = pd.concat([observed, extrapolated], ignore_index=True)
    assert model.predict(inputs).in_data_range.tolist() == [True, False]
    distances = model.support_distance(inputs)
    assert distances[0] == pytest.approx(0, abs=1e-8)
    assert distances[1] > distances[0]


@pytest.mark.parametrize("expected_source", ["nasa", "mixed"])
def test_model_source_is_derived_from_training_rows(expected_source):
    data = generate_synthetic(n_rows=600)
    data["source"] = "nasa"
    if expected_source == "mixed":
        data.loc[len(data) // 2:, "source"] = "synthetic"
    model = ResidualEnsemble(n_members=2).fit(data)
    assert model.data_source == expected_source
    assert set(model.training_data.source) == ({"nasa"} if expected_source == "nasa" else {"nasa", "synthetic"})


def test_non_microgravity_training_rejected(trained_bundle):
    data = trained_bundle["df"].copy()
    data.loc[0, "gravity_g"] = .02
    with pytest.raises(ValueError, match="microgravity"):
        ResidualEnsemble(n_members=2).fit(data)


def test_map_orientation_aliases_and_cached_result_isolation(trained_bundle, monkeypatch):
    model = trained_bundle["model"]
    original_predict = model.predict
    prediction_calls = []

    def count_predictions(inputs):
        prediction_calls.append(len(inputs))
        return original_predict(inputs)

    monkeypatch.setattr(model, "predict", count_predictions)
    service = FlammabilityMapService(model)
    first = service.get_map("cotton", pressure_kpa=70., resolution=7)
    assert first["x"] == first["o2_values"]
    assert first["y"] == first["flow_values"]
    assert first["z"] == first["extinction_probability"]
    assert np.asarray(first["z"]).shape == (7, 7)
    row, column = 1, 5
    point = pd.DataFrame([{
        "material": "cotton", "pressure_kpa": 70.,
        "flow_cm_s": first["flow_values"][row],
        "o2_percent": first["o2_values"][column],
    }])
    expected = original_predict(point)
    assert first["z"][row][column] == pytest.approx(expected.extinction_probability[0])
    assert first["spread_rate"][row][column] == pytest.approx(expected.mean[0])
    first["z"][row][column] = -999
    first["data_source"] = "corrupted"
    cached = service.get_map("sibal", pressure_kpa=70., resolution=7)
    assert prediction_calls == [49]
    assert cached["z"][row][column] == pytest.approx(expected.extinction_probability[0])
    assert cached["data_source"] == model.data_source

    # A replacement predictor gets its own cache and its own provenance badge.
    replacement = copy(model)
    replacement.data_source = "nasa"
    replacement.training_data = model.training_data.assign(source="nasa")
    replacement_map = FlammabilityMapService(replacement).get_map("cotton", pressure_kpa=70., resolution=7)
    assert replacement_map["data_source"] == "nasa"
    assert all(point["source"] == "nasa" for point in replacement_map["tested_points"])
    assert cached["data_source"] == "synthetic"


def test_boundary_uses_highest_crossing_and_omits_rows_without_crossing():
    flow = np.array([1., 2., 3.])
    oxygen = np.array([10., 20., 30., 40., 50.])
    probability = np.array([
        [.9, .1, .9, .1, .1],
        [.9, .8, .7, .6, .51],
        [1., .8, .5, .2, .1],
    ])
    assert _boundary_points(flow, oxygen, probability) == [
        {"flow_cm_s": 1., "o2_percent": 35.},
        {"flow_cm_s": 3., "o2_percent": 30.},
    ]


def test_suggestions_are_ranked_diverse_and_explained(trained_bundle):
    model = trained_bundle["model"]
    service = SuggestService(model, FlammabilityMapService(model))
    result = service.suggest("cotton", pressure_kpa=70., k=8)
    points = result["points"]
    assert len(points) == 8
    assert len({(point["o2"], point["flow_cm_s"]) for point in points}) == 8
    scores = [point["score"] for point in points]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= score <= 1 for score in scores)
    assert all(point["o2"] == point["o2_percent"] for point in points)
    assert all(point["reason"].isascii() and "test" in point["reason"] for point in points)
    assert all(point["pressure_kpa"] == 70. for point in points)
    coordinates = np.array([
        [(point["o2"] - 8.) / 47., np.log(point["flow_cm_s"] / .1) / np.log(1500.)]
        for point in points
    ])
    pairwise = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    assert np.min(pairwise[np.triu_indices(len(points), k=1)]) >= .1 - 1e-10
    assert result["data_source"] == model.data_source


@pytest.mark.parametrize("target", ["spread_rate", "extinction"])
def test_shap_adds_to_full_ensemble_raw_target_and_supports_global_only(trained_bundle, target):
    model = trained_bundle["model"]
    inputs = {"material": "cotton", "o2": 21., "pressure_kpa": 101.3, "flow_cm_s": 10.}
    service = ExplainService(model)
    result = service.explain(inputs, target=target)
    members = model.regressors if target == "spread_rate" else model.classifiers
    features = model.feature_frame(pd.DataFrame([inputs])).to_numpy()
    expected_raw_mean = np.mean([
        member.booster_.predict(features, raw_score=True, num_threads=1)[0]
        for member in members
    ])
    explained_total = result["base_value"] + sum(item["contribution"] for item in result["local_contributions"])
    assert result["prediction"] == pytest.approx(expected_raw_mean, abs=1e-7)
    assert explained_total == pytest.approx(expected_raw_mean, abs=1e-7)
    assert {item["feature"] for item in result["local_contributions"]} == set(model.feature_names)
    importance = result["global_importance"]
    assert sum(item["importance"] for item in importance) == pytest.approx(1.)
    assert all(item["importance"] >= 0 for item in importance)
    assert result["features"] == [{"name": item["feature"], "importance": item["importance"]} for item in importance]
    global_only = service.explain(target=target)
    assert global_only["local_contributions"] == []
    assert global_only["base_value"] is None
    assert global_only["prediction"] is None
    assert global_only["data_source"] == model.data_source
