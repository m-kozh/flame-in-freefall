"""Out-of-experiment reliability diagnostics, never a training calibration step."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def probability_reliability(y_true, probability, n_bins: int = 10) -> dict:
    """Equal-width probability bins, ECE, and an honest binned Brier decomposition.

    Within-bin probability variation prevents a generic exact three-term empirical
    decomposition. We report its residual explicitly rather than hiding it.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError("Reliability requires nonempty one-dimensional aligned arrays.")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Reliability labels must be 0/1 and inputs must be finite.")
    if ((p < 0) | (p > 1)).any() or isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("Probabilities must be in [0,1] and n_bins must be a positive integer.")
    assignment = np.minimum((p * n_bins).astype(int), n_bins - 1)
    event_rate = float(y.mean())
    bins = []
    ece = reliability = resolution = 0.0
    for index in range(n_bins):
        selected = assignment == index
        count = int(selected.sum())
        predicted = float(p[selected].mean()) if count else None
        observed = float(y[selected].mean()) if count else None
        bins.append({
            "lower": index / n_bins, "upper": (index + 1) / n_bins,
            "count": count, "mean_probability": predicted, "observed_frequency": observed,
        })
        if count:
            weight = count / len(y)
            ece += weight * abs(predicted - observed)
            reliability += weight * (predicted - observed) ** 2
            resolution += weight * (observed - event_rate) ** 2
    brier = float(np.mean((p - y) ** 2))
    uncertainty = event_rate * (1 - event_rate)
    reconstructed = reliability - resolution + uncertainty
    clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {
        "n_rows": len(y), "n_bins": n_bins, "binning": "equal_width",
        "positive_class": "no_sustained_flame", "event_rate": event_rate,
        "ece": float(ece), "brier": brier,
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "bins": bins,
        "brier_decomposition": {
            "reliability": float(reliability), "resolution": float(resolution),
            "uncertainty": float(uncertainty), "reconstructed": float(reconstructed),
            "binning_residual": float(brier - reconstructed),
            "method": "Binned approximation; brier = reconstructed + binning_residual.",
        },
    }


def _slice_metrics(frame: pd.DataFrame, values: dict, rows: np.ndarray) -> dict:
    sustained = (frame.ignited & ~frame.extinguished).to_numpy()[rows]
    observed = frame.spread_rate_mm_s.to_numpy()[rows][sustained]
    predicted = np.asarray(values["mean"])[rows][sustained]
    probability = np.asarray(values["probability"])[rows]
    target = (~sustained).astype(int)
    has_interval = np.isfinite(np.asarray(values["lower"])[rows]).all()
    errors = np.log(np.maximum(predicted, 1e-8)) - np.log(observed)
    return {
        "n_rows": int(rows.sum()), "n_burning": int(sustained.sum()),
        "n_nonburning": int((~sustained).sum()),
        "log_spread_mae": float(np.abs(errors).mean()) if len(errors) else None,
        "log_spread_rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
        "extinction_roc_auc": float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else None,
        "extinction_brier": float(np.mean((target - probability) ** 2)),
        "interval_coverage_90": float(np.mean(
            (observed >= np.asarray(values["lower"])[rows][sustained])
            & (observed <= np.asarray(values["upper"])[rows][sustained])
        )) if len(observed) and has_interval else None,
    }


def build_validation_report(frame: pd.DataFrame, predictions: dict, n_bins: int = 10) -> dict:
    """Summarise already-created OOF predictions; never fit or tune on these labels."""
    target = (~(frame.ignited & ~frame.extinguished)).astype(int).to_numpy()
    models = {}
    for name, values in predictions.items():
        material_slices = []
        experiment_slices = []
        source_slices = []
        for column, destination in [("material", material_slices), ("experiment_id", experiment_slices), ("source", source_slices)]:
            # Older callers may supply only the labels needed for diagnostics.
            # An absent source column is unknown provenance, never NASA by default.
            if column not in frame:
                continue
            for group in sorted(frame[column].unique()):
                selected = frame[column].eq(group).to_numpy()
                summary = {column: str(group), **_slice_metrics(frame, values, selected)}
                if column == "source":
                    summary["n_experiments"] = int(frame.loc[selected, "experiment_id"].nunique())
                destination.append(summary)
        models[name] = {
            "calibration": probability_reliability(target, values["probability"], n_bins=n_bins),
            "by_material": material_slices, "by_experiment": experiment_slices,
            "by_source": source_slices,
        }
    return {
        "method": "GroupKFold out-of-fold predictions", "n_rows": len(frame),
        "n_bins": n_bins, "models": models,
        "notes": [
            "Diagnostics use only outer-fold predictions; they do not calibrate or tune the fitted model.",
            "ECE depends on finite sample size and equal-width binning; inspect bin counts, not ECE alone.",
            "Brier combines calibration, resolution and outcome uncertainty; the binned decomposition includes its approximation residual.",
            "Speed errors and coverage are conditional on observed sustained burning; undefined slice metrics are null.",
            "Small or single-class experiment slices are descriptive and are not independent guarantees of generalisation.",
            "Source slices retain the same outer-fold predictions; a NASA-labelled slice is not a claim that all materials or pressures were validated by NASA.",
        ],
    }
