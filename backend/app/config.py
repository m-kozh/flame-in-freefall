"""Paths and operational defaults; independent of the current working directory."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DATA_PATH = ROOT / "data" / "experiments.csv"
DATA_PATH = Path(os.environ.get("FLAME_DATA_PATH", ROOT / "data" / "experiments_mixed.csv"))
ARTIFACT_DIR = Path(os.environ.get("FLAME_ARTIFACT_DIR", ROOT / "artifacts"))
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
MODEL_VERSION = "residual-ensemble-0.3.0"
API_VERSION = "0.3.0"
RANDOM_SEED = 42
ENSEMBLE_SIZE = 20
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5173"]
