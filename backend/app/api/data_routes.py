"""Public, source-preserving NASA evidence endpoints, available before training."""

import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.data_schemas import (
    DatasetDetailResponse, DatasetListResponse, DatasetObservationsResponse,
)
from app.schemas import ErrorResponse
from data.loader import COLUMNS
from data.nasa import dataset_catalog, dataset_observations, training_evidence_summary

router = APIRouter(tags=["NASA observations"], responses={
    404: {"model": ErrorResponse, "description": "Unknown NASA dataset identifier."},
    422: {"model": ErrorResponse, "description": "Invalid input."},
})

TRAINING_DATASET_ID = "nasa-cr-182185"
def _included_in_model(dataset_id: str, request: Request) -> bool:
    if dataset_id != TRAINING_DATASET_ID:
        # The PSI catalogue and partially reported BASS measurements are source
        # evidence, not manufactured complete-target training rows.
        return False
    predictor = getattr(request.app.state, "predictor", None)
    evidence = getattr(predictor, "training_evidence", None)
    if evidence is None:
        data = getattr(predictor, "training_data", None)
        if not isinstance(data, pd.DataFrame) or data.empty or not set(COLUMNS).issubset(data.columns):
            return False
        # A source/campaign string by itself is not proof that the curated NASA
        # measurements were used. The fallback checks their actual values.
        evidence = training_evidence_summary(data)
    if not isinstance(evidence, dict):
        return False
    count = evidence.get("verified_curated_row_count", 0)
    dataset_ids = evidence.get("dataset_ids", [])
    return (
        isinstance(count, int) and not isinstance(count, bool) and count > 0
        and isinstance(dataset_ids, list) and dataset_id in dataset_ids
    )


def _card(record: dict, request: Request) -> dict:
    return {**record, "included_in_loaded_model": _included_in_model(record["id"], request)}


def _find(dataset_id: str) -> dict | None:
    return next((record for record in dataset_catalog() if record["id"] == dataset_id), None)


def _not_found(dataset_id: str) -> JSONResponse:
    # The application's general exception handler reports the loaded model's
    # source. Here the resource is NASA evidence even if no model is loaded or
    # the inference model is synthetic/mixed.
    return JSONResponse(status_code=404, content={
        "detail": f"Unknown NASA dataset '{dataset_id}'. See GET /datasets for valid identifiers.",
        "data_source": "nasa",
    })


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets(request: Request):
    return {"data_source": "nasa", "datasets": [_card(record, request) for record in dataset_catalog()]}


@router.get("/datasets/{dataset_id}", response_model=DatasetDetailResponse)
def get_dataset(dataset_id: str, request: Request):
    record = _find(dataset_id)
    if record is None:
        return _not_found(dataset_id)
    return {"data_source": "nasa", "dataset": _card(record, request)}


@router.get("/datasets/{dataset_id}/observations", response_model=DatasetObservationsResponse)
def get_observations(dataset_id: str):
    if _find(dataset_id) is None:
        return _not_found(dataset_id)
    observations = dataset_observations(dataset_id)
    return {
        "data_source": "nasa", "dataset_id": dataset_id,
        "count": len(observations), "observations": observations,
    }
