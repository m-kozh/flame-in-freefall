"""Transparent uncertainty sampling with a boundary preference and diversity."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.materials import resolve_material
from models.base import Predictor
from physics.flame_spread import OPTIMAL_FLOW_CM_S
from services.flammability_map import FlammabilityMapService


@dataclass(frozen=True)
class SuggestConstants:
    candidate_resolution: int = 50
    candidate_flow_min: float = 0.1
    candidate_flow_max: float = 150.0
    candidate_o2_min: float = 8.0
    candidate_o2_max: float = 55.0
    boundary_probability_width: float = 0.25
    minimum_boundary_weight: float = 0.1
    sparse_distance_scale: float = 0.25
    minimum_point_separation: float = 0.10


CONSTANTS = SuggestConstants()


class SuggestService:
    def __init__(self, predictor: Predictor, map_service: FlammabilityMapService):
        self.predictor = predictor
        self.map_service = map_service

    def suggest(self, material: str, pressure_kpa: float = 101.3, k: int = 5) -> dict:
        material = resolve_material(material)
        c = CONSTANTS
        grid = self.map_service.get_map(
            material, pressure_kpa=pressure_kpa,
            flow_range=(c.candidate_flow_min, c.candidate_flow_max),
            o2_range=(c.candidate_o2_min, c.candidate_o2_max),
            resolution=c.candidate_resolution, flow_scale="log",
        )
        oxygen, flow = np.meshgrid(grid["o2_values"], grid["flow_values"])
        candidates = pd.DataFrame({
            "material": material, "o2_percent": oxygen.ravel(),
            "flow_cm_s": flow.ravel(), "pressure_kpa": pressure_kpa,
        })
        probability = np.asarray(grid["extinction_probability"]).ravel()
        uncertainty = np.asarray(grid["uncertainty"]).ravel()
        distance = self.predictor.support_distance(candidates)
        boundary_weight = c.minimum_boundary_weight + (1 - c.minimum_boundary_weight) * np.exp(
            -((probability - .5) / c.boundary_probability_width) ** 2
        )
        sparse_weight = 1 + np.clip(distance / c.sparse_distance_scale, 0, 1)
        # Do not manufacture larger uncertainty outside the data domain: use
        # the ensemble SD as reported, and expose the separate acquisition score.
        raw_score = uncertainty * boundary_weight * sparse_weight
        maximum = raw_score.max()
        score = raw_score / maximum if maximum > 0 else raw_score
        coordinates = np.column_stack([
            (oxygen.ravel() - c.candidate_o2_min) / (c.candidate_o2_max - c.candidate_o2_min),
            np.log(flow.ravel() / c.candidate_flow_min) / np.log(c.candidate_flow_max / c.candidate_flow_min),
        ])
        order = np.argsort(-score, kind="stable")
        chosen = []
        for index in order:
            if all(np.linalg.norm(coordinates[index] - coordinates[other]) >= c.minimum_point_separation for other in chosen):
                chosen.append(int(index))
                if len(chosen) == k:
                    break
        # The bounded dense candidate grid has ample distinct points for k<=20.
        if len(chosen) < k:
            for index in order:
                if int(index) not in chosen:
                    chosen.append(int(index))
                    if len(chosen) == k:
                        break
        chosen.sort(key=lambda index: -score[index])
        points = []
        for index in chosen:
            speed = float(flow.ravel()[index])
            oxygen_value = float(oxygen.ravel()[index])
            branch = "low-flow quenching" if speed < OPTIMAL_FLOW_CM_S else "high-flow blowoff"
            near_boundary = abs(probability[index] - .5) < .25
            context = f"near the {branch} boundary" if near_boundary else f"in the {branch} region"
            coverage = "sparse nearby test coverage" if distance[index] > .1 else "a poorly resolved flame limit"
            points.append({
                "o2_percent": oxygen_value, "o2": oxygen_value,
                "pressure_kpa": float(pressure_kpa), "flow_cm_s": speed,
                "uncertainty": float(uncertainty[index]),
                "extinction_probability": float(probability[index]), "score": float(score[index]),
                "reason": f"Ensemble disagreement (SD {uncertainty[index]:.2f}) {context} and {coverage} make this a useful next test (P(extinction)={probability[index]:.2f}).",
            })
        return {
            "points": points, "material": material, "pressure_kpa": float(pressure_kpa),
            "model_version": self.predictor.model_version, "data_source": self.predictor.data_source,
        }
