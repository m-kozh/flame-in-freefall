"""One response serializer shared by single, batch and decision predictions."""

import numpy as np
import pandas as pd

from models.base import PredictionBatch, Predictor
from models.residual_ensemble import physics_frame, prepare_inputs
from physics.flame_spread import OPTIMAL_FLOW_CM_S

INPUT_COLUMNS = ["material", "o2_percent", "pressure_kpa", "flow_cm_s", "thickness_mm"]


def model_meta(predictor: Predictor) -> dict:
    return {
        "model_version": predictor.model_version, "data_source": predictor.data_source,
        "model_id": getattr(predictor, "model_id", None),
    }


def prediction_dicts(
    predictor: Predictor, frame: pd.DataFrame, prediction: PredictionBatch | None = None,
) -> list[dict]:
    """Serialize an already computed batch, or evaluate it exactly once.

    Physics extinction subtypes and confidence are explanatory heuristics; model
    probabilities and conditional speed intervals remain the scientific outputs.
    """
    if len(frame) == 0:
        return []
    inputs = prepare_inputs(frame).loc[:, INPUT_COLUMNS]
    result = prediction if prediction is not None else predictor.predict(inputs)
    physical = physics_frame(inputs)
    responses = []
    for index, row in enumerate(inputs.to_dict(orient="records")):
        probability = float(result.extinction_probability[index])
        physics_regime = str(result.physics_regime[index])
        regime = "burning"
        if probability >= .5:
            regime = physics_regime
            if regime == "burning":
                if row["o2_percent"] < physical["minimum_o2_percent"][index]:
                    regime = "no_ignition"
                else:
                    regime = "quench_extinction" if row["flow_cm_s"] < OPTIMAL_FLOW_CM_S else "blowoff_extinction"
        in_range = bool(result.in_data_range[index])
        uncertainty = float(result.uncertainty[index])
        if not in_range or uncertainty >= .15:
            confidence = "low"
        elif uncertainty < .08 and result.log_std[index] < .3:
            confidence = "high"
        else:
            confidence = "medium"
        interval = {
            "mean": float(result.mean[index]), "lower": float(result.lower[index]),
            "upper": float(result.upper[index]),
        }
        responses.append({
            "spread_rate_mm_s": interval["mean"], "spread_rate_interval": interval,
            "spread_ci": [interval["lower"], interval["upper"]],
            "extinction_probability": probability, "extinction_prob": probability,
            "extinction_ci": [float(result.extinction_lower[index]), float(result.extinction_upper[index])],
            "uncertainty": uncertainty, "regime": regime,
            "physics_baseline_mm_s": float(result.physics_baseline[index]), "physics_regime": physics_regime,
            "confidence": confidence, "in_data_range": in_range, "inputs": row,
            "model_version": predictor.model_version, "data_source": predictor.data_source,
        })
    return responses
