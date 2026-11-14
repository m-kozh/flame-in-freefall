"""Grouped, out-of-experiment evaluation and final model training.

All three approaches share the same GroupKFold splits. Never randomly split
rows: neighbouring observations from one experiment are correlated.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

from app.config import ARTIFACT_DIR, ENSEMBLE_SIZE, MODEL_VERSION, RANDOM_SEED
from data.loader import validate_dataframe
from models.provenance import build_provenance
from models.residual_ensemble import ResidualEnsemble, physics_frame
from models.validation import build_validation_report

PHYSICS_PROBABILITY_TRANSITION_PERCENT = 1.2  # Illustrative fixed soft threshold, not calibrated.
LOG_RATE_FLOOR = 1e-8


def grouped_splits(df: pd.DataFrame, n_splits: int = 5, seed: int = RANDOM_SEED):
    if df.experiment_id.nunique() < n_splits:
        raise ValueError(f"GroupKFold needs at least {n_splits} distinct experiment_id values.")
    # Shuffle whole groups, never rows. Sorted equal-size batches otherwise align
    # with this generator's material cycle and accidentally hold out one material.
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(df, groups=df["experiment_id"]):
        train_groups = set(df.iloc[train_index].experiment_id)
        test_groups = set(df.iloc[test_index].experiment_id)
        assert train_groups.isdisjoint(test_groups), "Experiment leakage detected."
        yield train_index, test_index


def _scores(df, prediction, probability, lower=None, upper=None):
    burning = (df.ignited & ~df.extinguished).to_numpy()
    observed = df.spread_rate_mm_s.to_numpy()[burning]
    log_error = np.log(np.maximum(prediction[burning], LOG_RATE_FLOOR)) - np.log(observed)
    target = (~(df.ignited & ~df.extinguished)).astype(int).to_numpy()
    return {
        "log_spread_mae": float(mean_absolute_error(np.log(observed), np.log(np.maximum(prediction[burning], LOG_RATE_FLOOR)))),
        "log_spread_rmse": float(np.sqrt(np.mean(log_error ** 2))),
        "extinction_roc_auc": float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else None,
        "extinction_brier": float(brier_score_loss(target, probability)),
        "interval_coverage_90": None if lower is None else float(np.mean((observed >= lower[burning]) & (observed <= upper[burning]))),
    }


def train_and_evaluate(
    frame: pd.DataFrame, artifact_dir: str | Path = ARTIFACT_DIR,
    n_members: int = ENSEMBLE_SIZE, n_splits: int = 5, seed: int = RANDOM_SEED,
) -> dict:
    df = validate_dataframe(frame)
    if (df.gravity_g > 0.01).any():
        raise ValueError("This model only supports microgravity data (gravity_g <= 0.01). Filter other gravity regimes explicitly before training.")
    if not (df.ignited & ~df.extinguished).any():
        raise ValueError("Training requires positive spread-rate observations from sustained-burning tests.")
    n = len(df)
    keys = ["physics_only", "ml_only", "physics_plus_ml_residual"]
    out = {key: {field: np.full(n, np.nan) for field in ["mean", "probability", "lower", "upper"]} for key in keys}
    folds = []
    for fold, (train_index, test_index) in enumerate(grouped_splits(df, n_splits, seed), 1):
        training = df.iloc[train_index].reset_index(drop=True)
        testing = df.iloc[test_index].reset_index(drop=True)
        folds.append({
            "fold": fold,
            "train_experiment_ids": sorted(training.experiment_id.unique().tolist()),
            "test_experiment_ids": sorted(testing.experiment_id.unique().tolist()),
            "train_rows": len(training), "test_rows": len(testing),
        })
        physical = physics_frame(testing)
        out["physics_only"]["mean"][test_index] = physical["potential_spread_rate_mm_s"]
        out["physics_only"]["probability"][test_index] = 1 / (1 + np.exp(np.clip(physical["o2_margin_percent"] / PHYSICS_PROBABILITY_TRANSITION_PERCENT, -40, 40)))
        for key, use_physics in [("ml_only", False), ("physics_plus_ml_residual", True)]:
            estimator = ResidualEnsemble(n_members=n_members, seed=seed + fold, use_physics=use_physics).fit(training)
            batch = estimator.predict(testing)
            for field, values in [("mean", batch.mean), ("probability", batch.extinction_probability), ("lower", batch.lower), ("upper", batch.upper)]:
                out[key][field][test_index] = values
        print(f"Grouped fold {fold}/{n_splits}: {len(training)} train / {len(testing)} held-out rows", flush=True)
    scores = {}
    for key in keys:
        values = out[key]
        if not np.isfinite(values["mean"]).all() or not np.isfinite(values["probability"]).all():
            raise RuntimeError("Incomplete or nonfinite out-of-fold predictions.")
        if np.any((values["probability"] < 0) | (values["probability"] > 1)):
            raise RuntimeError("Extinction probability outside [0, 1].")
        if key != "physics_only":
            if not (np.isfinite(values["lower"]).all() and np.isfinite(values["upper"]).all()):
                raise RuntimeError("Nonfinite out-of-fold prediction interval.")
            if np.any((values["lower"] <= 0) | (values["upper"] < values["lower"])):
                raise RuntimeError("Invalid out-of-fold prediction interval ordering.")
        scores[key] = _scores(
            df, values["mean"], values["probability"],
            None if key == "physics_only" else values["lower"],
            None if key == "physics_only" else values["upper"],
        )
    final_model = ResidualEnsemble(n_members=n_members, seed=seed).fit(df)
    final_model.provenance = build_provenance(final_model, validation_config={
        "method": "GroupKFold", "group_column": "experiment_id", "n_splits": n_splits,
        "shuffle": True, "seed": seed,
    })
    final_model.model_id = final_model.provenance["model_id"]
    metrics = {
        "model_version": MODEL_VERSION, "data_source": final_model.data_source,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": n, "n_experiments": df.experiment_id.nunique(),
        "ensemble_size": n_members, "seed": seed,
        "cv": {"method": "GroupKFold", "group_column": "experiment_id", "n_splits": n_splits, "shuffle": True, "random_state": seed, "folds": folds},
        "models": scores,
        "model_id": final_model.model_id, "provenance": final_model.provenance,
        "validation": build_validation_report(df, out),
        "training_evidence": final_model.training_evidence,
        "notes": [
            "All results are out-of-fold; entire experiments are held out together.",
            "Speed errors and interval coverage are conditional on observed sustained burning; the physics model uses its positive potential rate for the same conditional comparison.",
            "Physics extinction probabilities use sigmoid(-oxygen_margin/1.2), an assumed soft threshold rather than calibrated probabilities.",
            "Nominal 90% speed intervals combine bootstrap model spread and training-group OOB residual noise; coverage is measured, not guaranteed.",
            "Synthetic-data accuracy does not demonstrate accuracy on NASA measurements or fire-safety suitability.",
        ],
    }
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Stage both files before replacement, so failed serialization does not overwrite good artifacts.
    staged_model = artifact_dir / "model.pending.joblib"
    staged_metrics = artifact_dir / "metrics.pending.json"
    final_model.save(staged_model)
    staged_metrics.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    staged_model.replace(artifact_dir / "model.joblib")
    staged_metrics.replace(artifact_dir / "metrics.json")
    return metrics
