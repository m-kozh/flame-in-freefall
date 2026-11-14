"""Run from backend/: python -m uvicorn app.main:app --reload --port 8000."""

import json
import logging
import math
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import data_source, router
from app.api.advanced_routes import router as advanced_router
from app.api.data_routes import router as data_router
from app.config import API_VERSION, CORS_ORIGINS, METRICS_PATH, MODEL_PATH
from models.base import Predictor
from models.provenance import metrics_match_model, validate_runtime_provenance
from models.residual_ensemble import ResidualEnsemble
from services.explain import ExplainService
from services.decision import DecisionService
from services.experiment_design import ExperimentDesignService
from services.flammability_map import FlammabilityMapService
from services.model_card import ModelCardService
from services.suggest import SuggestService

logger = logging.getLogger(__name__)


def _json_safe(value):
    """Validation errors can contain ValueError or NaN, neither is valid JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def create_app(
    predictor: Predictor | None = None, metrics: dict | None = None,
    model_path: str | Path = MODEL_PATH, metrics_path: str | Path = METRICS_PATH,
) -> FastAPI:
    """Inject another Predictor here to use a future PINN with the same API.

    Artifact loading happens once at startup; startup never silently generates
    data or trains. Missing artifacts leave liveness available and inference 503.
    """
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        loaded = predictor
        if loaded is None and Path(model_path).is_file():
            try:
                loaded = ResidualEnsemble.load(model_path)
            except Exception:
                logger.exception("Model artifact could not be loaded; regenerate it using scripts/train_model.py.")
        if loaded is not None:
            try:
                validate_runtime_provenance(loaded)
            except ValueError:
                logger.exception("Model identity no longer matches runtime physics or inference configuration; retrain before serving predictions.")
                loaded = None
        application.state.predictor = loaded
        application.state.metrics = None
        if loaded is not None:
            application.state.map_service = FlammabilityMapService(loaded)
            application.state.suggest_service = SuggestService(loaded, application.state.map_service)
            application.state.explain_service = ExplainService(loaded)
            application.state.decision_service = DecisionService(loaded)
            application.state.experiment_design_service = ExperimentDesignService(loaded)
            supplied_metrics = metrics
            if supplied_metrics is None and Path(metrics_path).is_file():
                try:
                    supplied_metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    logger.exception("Metrics could not be loaded.")
            if supplied_metrics is not None:
                if metrics_match_model(loaded, supplied_metrics):
                    application.state.metrics = supplied_metrics
                else:
                    logger.error("Metrics metadata does not match loaded model; /metrics will return 503.")
            application.state.model_card_service = ModelCardService(loaded, application.state.metrics)
        yield

    application = FastAPI(
        title="Flame in Freefall", version=API_VERSION, lifespan=lifespan,
        description="Physics-informed microgravity combustion demonstration; synthetic training data and approximate physics.",
    )
    application.add_middleware(
        CORSMiddleware, allow_origins=CORS_ORIGINS,
        allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Server-Timing", "X-Model-ID"],
    )

    @application.middleware("http")
    async def trace_request(request: Request, call_next):
        """Correlate frontend failures and measure complete application latency."""
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", supplied) else uuid.uuid4().hex
        started = time.perf_counter()
        request.state.request_id = request_id
        request.state.started = started
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["Server-Timing"] = f"app;dur={duration_ms:.2f}"
        predictor_instance = getattr(request.app.state, "predictor", None)
        model_id = getattr(predictor_instance, "model_id", None)
        if isinstance(model_id, str):
            response.headers["X-Model-ID"] = model_id
        logger.info("request_id=%s method=%s path=%s status=%s duration_ms=%.2f", request_id, request.method, request.url.path, response.status_code, duration_ms)
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": _json_safe(exc.errors()), "data_source": data_source(request)})

    @application.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": _json_safe(exc.detail), "data_source": data_source(request)}, headers=exc.headers)

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.exception("Unexpected API failure", exc_info=exc)
        headers = {"X-Request-ID": getattr(request.state, "request_id", uuid.uuid4().hex)}
        started = getattr(request.state, "started", None)
        if started is not None:
            headers["Server-Timing"] = f"app;dur={(time.perf_counter() - started) * 1000:.2f}"
        origin = request.headers.get("Origin")
        if origin in CORS_ORIGINS:
            headers.update({"Access-Control-Allow-Origin": origin, "Vary": "Origin", "Access-Control-Expose-Headers": "X-Request-ID, Server-Timing"})
        return JSONResponse(status_code=500, content={"detail": "Unexpected server error; see backend logs.", "data_source": data_source(request)}, headers=headers)

    application.include_router(router)
    application.include_router(advanced_router)
    application.include_router(data_router)
    return application


app = create_app()
