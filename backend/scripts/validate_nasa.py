"""Evaluate an unseen NASA report without mistaking fitted rows for validation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ARTIFACT_DIR
from data.nasa import dataset_catalog, nasa_training_rows
from models.residual_ensemble import ResidualEnsemble
from models.train import _scores


def evaluate_nasa_external(model) -> dict:
    observations = nasa_training_rows()
    training = model.training_data
    named_overlap = any(
        training[column].astype(str).str.contains(r"182185|19890014267", case=False, regex=True).any()
        for column in ["campaign", "experiment_id"] if column in training
    )
    # Campaigns can be renamed during an import. Compare the actual observations
    # as well, without relying on a source tag or a conveniently different ID.
    columns = ["material", "thickness_mm", "o2_percent", "pressure_kpa", "flow_cm_s", "gravity_g", "ignited", "extinguished", "spread_rate_mm_s"]

    def signature(row):
        result = []
        for name in columns:
            value = row[name]
            if name == "material":
                result.append(str(value))
            elif name in {"ignited", "extinguished"}:
                result.append(bool(value))
            elif name == "spread_rate_mm_s" and (pd.isna(value) or (row["extinguished"] and value == 0)):
                result.append(None)
            else:
                # CSV roundtrips can differ by one binary digit; this tolerance
                # is much smaller than the source measurement precision.
                result.append(round(float(value), 9))
        return tuple(result)

    observed = {signature(row) for _, row in observations.iterrows()}
    value_overlap = set(columns).issubset(training.columns) and any(
        signature(row) in observed for _, row in training.iterrows()
    )
    if named_overlap or value_overlap:
        raise ValueError("External validation requires a model trained without NASA-CR-182185; potential report or observation overlap was detected, including renamed or relabelled rows. Use the preserved synthetic-only baseline.")
    prediction = model.predict(observations)
    card = next(card for card in dataset_catalog() if card["id"] == "nasa-cr-182185")
    return {
        "evaluation_type": "whole_report_external_transfer",
        "model_version": model.model_version, "model_id": getattr(model, "model_id", None),
        "model_data_source": model.data_source, "evaluation_data_source": "nasa",
        "dataset_id": card["id"], "n_rows": len(observations),
        "n_measured_rates": int(observations.spread_rate_mm_s.notna().sum()),
        "n_extinguished": int(observations.extinguished.sum()),
        "n_report_groups": observations.experiment_id.nunique(),
        "support_fraction": float(np.mean(prediction.in_data_range)),
        "metrics": _scores(
            observations, prediction.mean, prediction.extinction_probability,
            prediction.lower, prediction.upper,
        ),
        "scope": card["scope"], "observation_window_s": card["observation_window_s"],
        "source_files": card["source_files"],
        "notes": [
            "No report identifier or matching numerical observation was found in the model's training data; rounded numerical matching is a conservative overlap check, not proof of campaign independence.",
            "Eight positive rates and three observed extinctions are evaluated; two left-censored rate records are retained as evidence but excluded from the complete-target CSV.",
            "A no-extinction label describes only the 5.18-second observation window, not indefinite flame survival.",
            "One report and eight measured rates are a small transfer check, not an independent multi-campaign validation of general fire safety.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ARTIFACT_DIR / "synthetic_baseline" / "model.joblib")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "synthetic_baseline" / "nasa_external_validation.json")
    args = parser.parse_args()
    try:
        result = evaluate_nasa_external(ResidualEnsemble.load(args.model))
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"NASA transfer check: {result['n_rows']} conditions, {result['n_report_groups']} whole-report group")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Local support fraction: {result['support_fraction']:.1%}")
    print(f"Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
