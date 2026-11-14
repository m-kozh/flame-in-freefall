"""Additive decision and research endpoints; legacy paths retain their shapes."""

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from app.advanced_schemas import (
    AnalyzeRequest, AnalyzeResponse, BatchPredictRequest, BatchPredictResponse,
    ExperimentDesignRequest, ExperimentDesignResponse, ModelCardResponse,
    OptimizeRequest, OptimizeResponse,
)
from app.api.routes import require_model
from app.schemas import ErrorResponse
from services.predictions import model_meta, prediction_dicts

router = APIRouter(tags=["Decision and research"], responses={
    422: {"model": ErrorResponse, "description": "Invalid input or requested computation outside allowed bounds."},
    503: {"model": ErrorResponse, "description": "A trained model is unavailable."},
})


@router.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest, request: Request):
    predictor = require_model(request)
    frame = pd.DataFrame([point.model_dump() for point in payload.points])
    predictions = prediction_dicts(predictor, frame)
    return {**model_meta(predictor), "count": len(predictions), "predictions": predictions}


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, request: Request):
    require_model(request)
    return request.app.state.decision_service.analyze(**payload.model_dump())


@router.post("/optimize", response_model=OptimizeResponse)
def optimize(payload: OptimizeRequest, request: Request):
    require_model(request)
    return request.app.state.decision_service.optimize(**payload.model_dump())


@router.post("/experiments/design", response_model=ExperimentDesignResponse, responses={501: {"model": ErrorResponse}})
def design_experiments(payload: ExperimentDesignRequest, request: Request):
    require_model(request)
    try:
        return request.app.state.experiment_design_service.design(**payload.model_dump())
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.get("/model-card", response_model=ModelCardResponse)
def model_card(request: Request):
    require_model(request)
    return request.app.state.model_card_service.get_card()
