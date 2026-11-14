"""Stable frontend endpoints; scientific work lives in models and services."""

from typing import Annotated

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from app.config import MODEL_VERSION
from app.schemas import (
    DetailedPredictResponse, ErrorResponse, ExplainQuery, ExplainRequest, ExplainResponse,
    HealthResponse, MapQuery, MapRequest, MapResponse, MaterialsResponse,
    MetricsResponse, PredictRequest, PredictResponse, PresetsResponse,
    SuggestQuery, SuggestResponse,
)
from data.materials import material_dicts
from data.presets import PRESETS
from services.predictions import prediction_dicts

router = APIRouter(responses={
    422: {"model": ErrorResponse, "description": "Invalid input with source metadata."},
    503: {"model": ErrorResponse, "description": "Model or matching metrics unavailable."},
})
MODEL_MISSING = "Model is not loaded. Run `python scripts/train_model.py` and restart the server."


def data_source(request: Request) -> str:
    predictor = getattr(request.app.state, "predictor", None)
    return predictor.data_source if predictor is not None else "synthetic"


def require_model(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail=MODEL_MISSING)
    return predictor


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    return {
        "status": "ok" if predictor is not None else "model_not_loaded",
        "model_loaded": predictor is not None,
        "model_version": predictor.model_version if predictor is not None else MODEL_VERSION,
        "data_source": data_source(request),
    }


@router.get("/materials", response_model=MaterialsResponse)
def materials(request: Request):
    return {"materials": material_dicts(), "data_source": data_source(request)}


@router.get("/presets", response_model=PresetsResponse)
def presets(request: Request):
    return {"presets": PRESETS, "data_source": data_source(request)}


def _prediction(payload: PredictRequest, request: Request) -> dict:
    predictor = require_model(request)
    return prediction_dicts(predictor, pd.DataFrame([payload.model_dump()]))[0]


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    return _prediction(payload, request)


@router.post("/predict/detailed", response_model=DetailedPredictResponse)
def predict_detailed(payload: PredictRequest, request: Request):
    result = _prediction(payload, request)
    result["spread_rate_mm_s"] = result["spread_rate_interval"].copy()
    return result


@router.post("/map", response_model=MapResponse)
def post_map(payload: MapRequest, request: Request):
    require_model(request)
    return request.app.state.map_service.get_map(**payload.model_dump())


@router.get("/map", response_model=MapResponse)
def get_map(request: Request, params: Annotated[MapQuery, Query()]):
    return post_map(params.to_request(), request)


@router.get("/suggest", response_model=SuggestResponse)
def suggest(request: Request, params: Annotated[SuggestQuery, Query()]):
    require_model(request)
    return request.app.state.suggest_service.suggest(**params.model_dump())


def _explain(request: Request, inputs=None, target="spread_rate"):
    require_model(request)
    try:
        return request.app.state.explain_service.explain(inputs, target)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.get("/explain", response_model=ExplainResponse, responses={501: {"model": ErrorResponse}})
def global_explain(request: Request, params: Annotated[ExplainQuery, Query()]):
    return _explain(request, target=params.target)


@router.post("/explain", response_model=ExplainResponse, responses={501: {"model": ErrorResponse}})
def local_explain(payload: ExplainRequest, request: Request):
    return _explain(request, inputs=payload.model_dump(exclude={"target"}), target=payload.target)


@router.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request):
    require_model(request)
    result = getattr(request.app.state, "metrics", None)
    if result is None:
        raise HTTPException(status_code=503, detail="Metrics are unavailable or do not match the loaded model; retrain and restart the server.")
    return result
