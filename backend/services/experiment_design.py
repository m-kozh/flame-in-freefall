"""Diverse, full-ensemble experimental design with interpretable acquisition.

The entropy difference is inspired by BALD. Bootstrap trees are not posterior
samples, so the output is explicitly an information-gain proxy, in bits.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from data.materials import get_material
from models.base import Predictor
from services.predictions import model_meta


@dataclass(frozen=True)
class DesignConstants:
    minimum_separation: float = 0.10
    tested_distance_threshold: float = 0.025
    boundary_probability_width: float = 0.25
    minimum_boundary_weight: float = 0.10
    sparse_distance_scale: float = 0.25
    cache_size: int = 16
    minimum_nonzero_score: float = 1e-12


CONSTANTS = DesignConstants()


def binary_entropy_bits(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Member probabilities must be finite and in [0, 1].")
    # 0 log(0) has the limiting value zero; explicitly handle both endpoints.
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(
            np.where(probability > 0, probability * np.log2(probability), 0)
            + np.where(probability < 1, (1 - probability) * np.log2(1 - probability), 0)
        )


def information_components(member_probabilities: np.ndarray) -> dict[str, np.ndarray]:
    probabilities = np.asarray(member_probabilities, dtype=float)
    if probabilities.ndim != 2 or min(probabilities.shape) == 0:
        raise ValueError("Expected member probabilities with shape [members, candidates].")
    member_entropy = binary_entropy_bits(probabilities).mean(axis=0)
    predictive_entropy = binary_entropy_bits(probabilities.mean(axis=0))
    return {
        "predictive_entropy_bits": predictive_entropy,
        "expected_member_entropy_bits": member_entropy,
        "ensemble_information_gain_proxy_bits": np.maximum(predictive_entropy - member_entropy, 0),
    }


class ExperimentDesignService:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor
        self._cached_grid = lru_cache(maxsize=CONSTANTS.cache_size)(self._evaluate_grid)

    def _evaluate_grid(self, material, pressure, thickness, flow_range, o2_range, resolution):
        predict_members = getattr(self.predictor, "predict_members", None)
        if not callable(predict_members):
            raise NotImplementedError("Information-driven design requires a member-prediction adapter; /suggest remains available.")
        flows = np.expm1(np.linspace(*np.log1p(flow_range), resolution))
        flows = np.clip(flows, *flow_range)
        flows[0], flows[-1] = flow_range
        oxygen = np.linspace(*o2_range, resolution)
        oxygen_grid, flow_grid = np.meshgrid(oxygen, flows, indexing="xy")
        frame = pd.DataFrame({
            "material": material, "o2_percent": oxygen_grid.ravel(),
            "pressure_kpa": pressure, "flow_cm_s": flow_grid.ravel(), "thickness_mm": thickness,
        })
        members = predict_members(frame)
        probabilities = np.asarray(members["extinction_probability"], dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(frame):
            raise ValueError("Member adapter must return [members, candidates] probabilities.")
        components = information_components(probabilities)
        support = members.get("in_data_range")
        if support is None:
            support = self.predictor.predict(frame).in_data_range
        support = np.asarray(support, dtype=bool)
        if support.shape != (len(frame),):
            raise ValueError("Member adapter returned invalid in_data_range shape.")
        distance = np.asarray(self.predictor.support_distance(frame), dtype=float)
        if distance.shape != (len(frame),) or not np.isfinite(distance).all() or np.any(distance < 0):
            raise ValueError("Support distances must be finite nonnegative candidate distances.")
        return {
            "oxygen": oxygen_grid.ravel(), "flow": flow_grid.ravel(),
            "probability": probabilities.mean(axis=0), "uncertainty": probabilities.std(axis=0),
            "support": support, "distance": distance, **components,
        }

    def design(
        self, material: str, pressure_kpa: float = 101.3,
        flow_range=(.1, 150.), o2_range=(8., 55.), resolution: int = 40,
        k: int = 5, strategy: str = "information_gain", exclude_tested: bool = True,
        thickness_mm: float | None = None,
    ) -> dict:
        sample = get_material(material)
        pressure = float(pressure_kpa)
        thickness = sample.thickness_mm if thickness_mm is None else float(thickness_mm)
        flows = tuple(float(value) for value in flow_range)
        oxygen = tuple(float(value) for value in o2_range)
        if len(flows) != 2 or len(oxygen) != 2 or not np.isfinite([pressure, thickness, *flows, *oxygen]).all():
            raise ValueError("Experimental-design ranges must have two finite bounds.")
        if not 0 <= flows[0] < flows[1] <= 200 or not 5 <= oxygen[0] < oxygen[1] <= 100:
            raise ValueError("Experimental-design ranges must be ordered and within API limits.")
        if not 10 <= pressure <= 150 or not .01 <= thickness <= 10:
            raise ValueError("Experimental-design pressure or thickness outside API limits.")
        if isinstance(resolution, bool) or not isinstance(resolution, int) or not 10 <= resolution <= 60:
            raise ValueError("resolution must be an integer in [10, 60].")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
            raise ValueError("k must be an integer in [1, 20].")
        if strategy not in {"information_gain", "boundary", "disagreement"}:
            raise ValueError("Unknown experimental-design strategy.")
        if not isinstance(exclude_tested, bool):
            raise ValueError("exclude_tested must be boolean.")
        grid = self._cached_grid(sample.id, pressure, thickness, flows, oxygen, resolution)
        probability, distance = grid["probability"], grid["distance"]
        c = CONSTANTS
        boundary_weight = c.minimum_boundary_weight + (1 - c.minimum_boundary_weight) * np.exp(
            -((probability - .5) / c.boundary_probability_width) ** 2
        )
        sparse_weight = 1 + np.clip(distance / c.sparse_distance_scale, 0, 1)
        acquisition = {
            "information_gain": grid["ensemble_information_gain_proxy_bits"],
            "boundary": grid["predictive_entropy_bits"],
            "disagreement": grid["uncertainty"],
        }[strategy]
        raw_score = acquisition * boundary_weight * sparse_weight
        excluded = (distance <= c.tested_distance_threshold) if exclude_tested else np.zeros(len(probability), dtype=bool)
        eligible = np.flatnonzero(~excluded)
        best_score = float(raw_score[eligible].max()) if len(eligible) else 0.
        informative = best_score > c.minimum_nonzero_score
        scores = raw_score / best_score if informative else np.zeros_like(raw_score)
        coordinates = np.column_stack([
            (grid["oxygen"] - oxygen[0]) / (oxygen[1] - oxygen[0]),
            (np.log1p(grid["flow"]) - np.log1p(flows[0])) / (np.log1p(flows[1]) - np.log1p(flows[0])),
        ])
        order = eligible[np.argsort(-scores[eligible], kind="stable")] if informative else []
        chosen = []
        for index in order:
            if all(np.linalg.norm(coordinates[index] - coordinates[other]) >= c.minimum_separation for other in chosen):
                chosen.append(int(index))
                if len(chosen) >= k:
                    break
        points = []
        for index in chosen:
            gain = float(grid["ensemble_information_gain_proxy_bits"][index])
            coverage = "inside local data support" if grid["support"][index] else "outside local data support and requiring experimental validation"
            points.append({
                "o2": float(grid["oxygen"][index]), "o2_percent": float(grid["oxygen"][index]),
                "pressure_kpa": pressure, "flow_cm_s": float(grid["flow"][index]),
                "thickness_mm": thickness, "extinction_probability": float(probability[index]),
                "uncertainty": float(grid["uncertainty"][index]),
                **{name: float(grid[name][index]) for name in ["predictive_entropy_bits", "expected_member_entropy_bits", "ensemble_information_gain_proxy_bits"]},
                "in_data_range": bool(grid["support"][index]), "nearest_test_distance": float(distance[index]),
                "score": float(scores[index]),
                "score_components": {
                    "acquisition": float(acquisition[index]), "boundary_weight": float(boundary_weight[index]),
                    "sparse_weight": float(sparse_weight[index]),
                },
                "reason": f"The ensemble information proxy is {gain:.3f} bits at P(extinction)={probability[index]:.2f}, {coverage}.",
            })
        notes = [
            "Entropy difference is a BALD-inspired ensemble information-gain proxy; bootstrap members are not exact posterior samples.",
            f"exclude_tested removes candidates within support-distance {c.tested_distance_threshold} of a measured observation.",
            "Between-member disagreement and within-member entropy describe the ensemble, not an identified decomposition of real physical uncertainty.",
            "Diversity is enforced in normalised oxygen/log1p-flow coordinates; fewer than k points may be returned.",
            "Source and out-of-support flags must remain visible; no prediction here certifies a safe atmosphere.",
        ]
        if not informative:
            notes.append("No eligible candidate had nonzero acquisition value; no artificial priority was assigned.")
        return {
            **model_meta(self.predictor), "material": sample.id, "pressure_kpa": pressure,
            "strategy": strategy, "candidate_count": len(probability),
            "excluded_tested_count": int(excluded.sum()), "eligible_count": len(eligible),
            "minimum_separation": c.minimum_separation, "points": points,
            "method_notes": notes,
        }
