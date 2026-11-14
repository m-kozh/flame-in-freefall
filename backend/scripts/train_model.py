"""Evaluate three approaches, then save the hybrid model and metrics."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ARTIFACT_DIR, DATA_PATH
from data.loader import load_csv
from models.train import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--members", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = train_and_evaluate(load_csv(args.data), args.artifacts, args.members, args.folds, args.seed)
    print("\nModel                         log MAE  log RMSE  ROC-AUC  Brier   90% coverage")
    for name, row in metrics["models"].items():
        coverage = "N/A" if row["interval_coverage_90"] is None else f"{row['interval_coverage_90']:.3f}"
        auc = "N/A" if row["extinction_roc_auc"] is None else f"{row['extinction_roc_auc']:.3f}"
        print(f"{name:29s} {row['log_spread_mae']:.3f}    {row['log_spread_rmse']:.3f}     {auc:7s}  {row['extinction_brier']:.3f}   {coverage}")
    print(f"\nArtifacts saved to {args.artifacts.resolve()}")
    print(f"Model identity: {metrics['model_id']}")
    print("Out-of-fold reliability (10 equal-width probability bins):")
    for name, row in metrics["validation"]["models"].items():
        print(f"  {name}: ECE={row['calibration']['ece']:.4f}; log loss={row['calibration']['log_loss']:.4f}")
    if metrics.get("training_evidence", {}).get("nasa_row_count", 0):
        print("Whole-report held-out NASA slice (never training-set accuracy):")
        for name, row in metrics["validation"]["models"].items():
            nasa = next(part for part in row["by_source"] if part["source"] == "nasa")
            coverage = "N/A" if nasa["interval_coverage_90"] is None else f"{nasa['interval_coverage_90']:.3f}"
            print(f"  {name}: {nasa['n_rows']} rows / {nasa['n_experiments']} group(s), log MAE={nasa['log_spread_mae']:.3f}, coverage={coverage}")


if __name__ == "__main__":
    main()
