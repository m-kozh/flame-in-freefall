"""Tests build their own small artifacts; no existing CSV/model is required."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.main import create_app
from data.synthetic import generate_synthetic
from models.residual_ensemble import ResidualEnsemble
from models.train import train_and_evaluate


@pytest.fixture(scope="session")
def trained_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("flame_artifacts")
    frame = generate_synthetic(n_rows=800)
    metrics = train_and_evaluate(frame, directory, n_members=4, n_splits=3)
    return {
        "model": ResidualEnsemble.load(directory / "model.joblib"),
        "metrics": metrics, "artifact_dir": directory, "df": frame,
    }


@pytest.fixture(scope="session")
def client(trained_bundle):
    with TestClient(create_app(predictor=trained_bundle["model"], metrics=trained_bundle["metrics"])) as session:
        yield session
