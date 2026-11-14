"""Reproducible scenario dossiers and bounded, uncertainty-aware counterfactuals.

All conclusions are model-based research suggestions. The finite tolerance samples
and ensemble quantiles used here are not certified lower bounds or safety limits.
Both services use the Predictor contract, without inspecting tree internals.
"""

from itertools import product

import numpy as np
import pandas as pd

from data.materials import MATERIALS, get_material
from data.presets import PRESETS
from models.base import Predictor
from physics.flame_spread import evaluate_flame
from services.predictions import model_meta, prediction_dicts

CONTROLS = ("o2_percent", "pressure_kpa", "flow_cm_s")
DOMAIN = {"o2_percent": (5., 100.), "pressure_kpa": (10., 150.), "flow_cm_s": (0., 200.)}
DEFAULT_TOLERANCES = {"o2_percent": 1., "pressure_kpa": 2., "flow_cm_s": 1.}
MAX_TOLERANCES = {"o2_percent": 10., "pressure_kpa": 20., "flow_cm_s": 20.}
BOUND_KEYS = ("o2_range", "pressure_range", "flow_range")
RANDOM_SEED = 42
PARETO_LIMIT = 20


def _number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not a boolean.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _scenario(scenario: dict) -> dict:
    """Small domain adapter for direct Python use; HTTP validation remains upstream."""
    material = get_material(scenario["material"])
    oxygen = scenario.get("o2_percent", scenario.get("o2"))
    if oxygen is None:
        raise ValueError("Scenario requires o2_percent or its o2 alias.")
    if "o2" in scenario and "o2_percent" in scenario:
        if _number(scenario["o2"], "o2") != _number(oxygen, "o2_percent"):
            raise ValueError("o2 and o2_percent must agree.")
    result = {
        "material": material.id,
        "o2_percent": _number(oxygen, "o2_percent"),
        "pressure_kpa": _number(scenario["pressure_kpa"], "pressure_kpa"),
        "flow_cm_s": _number(scenario["flow_cm_s"], "flow_cm_s"),
        "thickness_mm": _number(
            material.thickness_mm if scenario.get("thickness_mm") is None else scenario["thickness_mm"], "thickness_mm",
        ),
    }
    for name, (lo, hi) in DOMAIN.items():
        if not lo <= result[name] <= hi:
            raise ValueError(f"{name} must be in [{lo}, {hi}].")
    if not .01 <= result["thickness_mm"] <= 10:
        raise ValueError("thickness_mm must be in [0.01, 10].")
    return result


def _tolerances(values: dict | None) -> dict:
    result = dict(DEFAULT_TOLERANCES)
    if values is not None:
        if set(values) - set(CONTROLS):
            raise ValueError("Unknown tolerance; only oxygen, pressure and flow are supported.")
        result.update(values)
    for name, maximum in MAX_TOLERANCES.items():
        result[name] = _number(result[name], name)
        if not 0 <= result[name] <= maximum:
            raise ValueError(f"Tolerance {name} must be in [0, {maximum}].")
    return result


def _tolerance_box(points: np.ndarray, tolerances: dict) -> tuple[np.ndarray, np.ndarray]:
    widths = np.array([tolerances[name] for name in CONTROLS])
    legal_lower = np.array([DOMAIN[name][0] for name in CONTROLS])
    legal_upper = np.array([DOMAIN[name][1] for name in CONTROLS])
    return np.maximum(points - widths, legal_lower), np.minimum(points + widths, legal_upper)


def _corner_points(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return eight corners per centre, including duplicates on fixed dimensions."""
    switches = np.asarray(list(product((False, True), repeat=3)))
    return np.where(switches[None, :, :], upper[:, None, :], lower[:, None, :])


def _rows(base: dict, coordinates: np.ndarray) -> list[dict]:
    return [dict(base, **dict(zip(CONTROLS, point, strict=True))) for point in coordinates]


def _distribution(values: np.ndarray) -> dict:
    minimum, maximum = float(np.min(values)), float(np.max(values))
    # Floating summation of a constant array can put its mean a few ulps outside
    # min/max; preserve the mathematical bounds used by response validation.
    return {
        "min": minimum, "p05": float(np.clip(np.quantile(values, .05), minimum, maximum)),
        "mean": float(np.clip(np.mean(values), minimum, maximum)),
        "p95": float(np.clip(np.quantile(values, .95), minimum, maximum)), "max": maximum,
    }


def _limitations(predictor: Predictor, material: str) -> list[str]:
    messages = [
        "Model-based research insights are not a fire-safety certification or an approved operating recommendation.",
        "Conditional spread rates describe a sustained flame; they are not an unconditional fire-risk score.",
        "Ensemble lower quantiles describe member disagreement and are not calibrated probability guarantees.",
        "Tolerance samples and corners do not establish an exhaustive worst case over all possible conditions.",
    ]
    if predictor.data_source == "synthetic":
        messages.append("The model is trained entirely on synthetic observations, not NASA combustion measurements.")
    elif predictor.data_source == "mixed":
        messages.append("Training combines synthetic and NASA-labelled observations; inspect the model card for provenance.")
    if not get_material(material).thermally_thin:
        messages.append("This material's library sample is outside the thermally thin baseline's intended validity.")
    return messages


class DecisionService:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    def analyze(self, scenario: dict, tolerances: dict | None = None, samples: int = 128) -> dict:
        """Compare a scenario and propagate explicit input tolerances in one batch."""
        if isinstance(samples, bool) or not isinstance(samples, int) or not 32 <= samples <= 512:
            raise ValueError("samples must be an integer in [32, 512].")
        base = _scenario(scenario)
        tolerances = _tolerances(tolerances)
        center = np.array([[base[name] for name in CONTROLS]])
        lower, upper = _tolerance_box(center, tolerances)
        random_points = np.random.default_rng(RANDOM_SEED).uniform(lower[0], upper[0], size=(samples, 3))
        corners = _corner_points(lower, upper)[0]

        inputs = [base]
        material_indices = []
        for material in MATERIALS.values():
            material_indices.append((material, len(inputs)))
            inputs.append(dict(base, material=material.id, thickness_mm=material.thickness_mm))
        preset_indices = []
        for preset in PRESETS:
            preset_indices.append((preset, len(inputs)))
            inputs.append(dict(base, **{name: preset[name] for name in CONTROLS}))
        sensitivity_indices = []
        for index, name in enumerate(CONTROLS):
            sensitivity_indices.append((name, len(inputs), len(inputs) + 1))
            inputs.extend([dict(base, **{name: lower[0, index]}), dict(base, **{name: upper[0, index]})])
        perturbation_start = len(inputs)
        inputs.extend(_rows(base, np.vstack([random_points, corners])))
        frame = pd.DataFrame(inputs)
        prediction = self.predictor.predict(frame)
        records = prediction_dicts(self.predictor, frame, prediction=prediction)
        current = records[0]

        material = get_material(base["material"])
        physical = evaluate_flame(
            base["o2_percent"], base["pressure_kpa"], base["flow_cm_s"], base["thickness_mm"],
            material.density_kg_m3, material.specific_heat_j_kgk,
            material.vaporization_temp_k, material.limiting_o2_percent,
        )
        potential = float(physical["potential_spread_rate_mm_s"])
        ratio = float(prediction.mean[0] / potential)
        disagreement = bool(bool(physical["burns"]) != (prediction.extinction_probability[0] < .5))
        interpretation = (
            "The learned sustained-burning verdict differs from the approximate physical boundary."
            if disagreement else "The learned sustained-burning verdict agrees with the approximate physical boundary."
        )
        probability = prediction.extinction_probability[perturbation_start:]
        supported = prediction.in_data_range[perturbation_start:]
        support_fraction = float(np.mean(supported))
        assessment = "supported" if bool(prediction.in_data_range[0]) and supported.all() else "extrapolation"
        limitations = _limitations(self.predictor, base["material"])
        limitations.append("Material comparisons use each material's library thickness; the samples are not geometrically interchangeable.")
        if assessment == "extrapolation":
            limitations.append("The current scenario or at least one tolerance sample is outside local training support.")

        return {
            **model_meta(self.predictor), "current": current,
            "physics_disagreement": {
                "potential_baseline_mm_s": potential, "learned_to_potential_ratio": ratio,
                "sustained_burning_disagreement": disagreement, "interpretation": interpretation,
            },
            "material_comparison": [
                {"material": sample.id, "thermally_thin": sample.thermally_thin, "prediction": records[index]}
                for sample, index in material_indices
            ],
            "preset_comparison": [
                {"preset_id": preset["id"], "illustrative": preset["illustrative"], "prediction": records[index]}
                for preset, index in preset_indices
            ],
            "sensitivity": [
                {
                    "parameter": name, "lower_input": float(inputs[low][name]), "upper_input": float(inputs[high][name]),
                    "lower_prediction": records[low], "upper_prediction": records[high],
                    "extinction_probability_change": float(prediction.extinction_probability[high] - prediction.extinction_probability[low]),
                    "spread_rate_change_mm_s": float(prediction.mean[high] - prediction.mean[low]),
                }
                for name, low, high in sensitivity_indices
            ],
            "robustness": {
                "seed": RANDOM_SEED, "random_samples": samples, "corner_samples": 8,
                "sample_count": samples + 8,
                "extinction_probability": _distribution(probability),
                "spread_rate_mm_s": _distribution(prediction.mean[perturbation_start:]),
                "minimum_sampled_extinction_lower": float(prediction.extinction_lower[perturbation_start:].min()),
                "support_fraction": support_fraction,
                "decision_flip_fraction": float(np.mean((probability < .5) != (prediction.extinction_probability[0] < .5))),
            },
            "assessment": assessment, "limitations": limitations,
        }

    def optimize(
        self, scenario: dict, bounds: dict, target_extinction_probability: float = .9,
        resolution: int = 10, k: int = 5, supported_only: bool = True,
        tolerances: dict | None = None,
    ) -> dict:
        """Search both flow branches and validate every eligible tolerance box."""
        if isinstance(resolution, bool) or not isinstance(resolution, int) or not 5 <= resolution <= 16:
            raise ValueError("resolution must be an integer in [5, 16].")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
            raise ValueError("k must be an integer in [1, 20].")
        target = _number(target_extinction_probability, "target_extinction_probability")
        if not 0 <= target <= 1:
            raise ValueError("target_extinction_probability must be in [0, 1].")
        if not isinstance(supported_only, bool):
            raise ValueError("supported_only must be a boolean.")
        base = _scenario(scenario)
        tolerances = _tolerances(tolerances)
        if set(bounds) != set(BOUND_KEYS):
            raise ValueError("bounds requires exactly o2_range, pressure_range, flow_range.")
        axes, ranges = [], []
        for name, key in zip(CONTROLS, BOUND_KEYS, strict=True):
            pair = bounds[key]
            if len(pair) != 2:
                raise ValueError(f"{key} requires [min, max].")
            lo, hi = [_number(value, key) for value in pair]
            if not DOMAIN[name][0] <= lo <= hi <= DOMAIN[name][1]:
                raise ValueError(f"{key} must be ordered within {DOMAIN[name]}.")
            ranges.append((lo, hi))
            if lo == hi:
                axis = np.array([lo])
            elif name == "flow_cm_s":
                axis = np.expm1(np.linspace(np.log1p(lo), np.log1p(hi), resolution))
                axis[0], axis[-1] = lo, hi  # Preserve inclusive bounds exactly.
            else:
                axis = np.linspace(lo, hi, resolution)
            axes.append(axis)
        coordinates = np.asarray(list(product(*axes)), dtype=float)
        center = np.asarray([base[name] for name in CONTROLS], dtype=float)
        limits = np.asarray(ranges)
        if np.all((center >= limits[:, 0]) & (center <= limits[:, 1])):
            coordinates = np.vstack([coordinates, center])
        coordinates = np.unique(coordinates, axis=0)
        frame = pd.DataFrame(_rows(base, coordinates))
        prediction = self.predictor.predict(frame)
        center_records = prediction_dicts(self.predictor, frame, prediction=prediction)
        width = limits[:, 1] - limits[:, 0]
        scaled_difference = np.zeros_like(coordinates)
        free = width > 0
        scaled_difference[:, free] = (coordinates[:, free] - center[free]) / width[free]
        costs = np.linalg.norm(scaled_difference, axis=1)
        center_eligible = np.flatnonzero(prediction.in_data_range if supported_only else np.ones(len(frame), dtype=bool))

        robust_lower = np.full(len(frame), np.nan)
        support_fraction = np.zeros(len(frame))
        eligible = np.zeros(len(frame), dtype=bool)
        if len(center_eligible):
            lower, upper = _tolerance_box(coordinates[center_eligible], tolerances)
            corner_coordinates = _corner_points(lower, upper).reshape(-1, 3)
            corner_frame = pd.DataFrame(_rows(base, corner_coordinates))
            corner_prediction = self.predictor.predict(corner_frame)
            corner_lower = corner_prediction.extinction_lower.reshape(-1, 8)
            corner_support = corner_prediction.in_data_range.reshape(-1, 8)
            robust_lower[center_eligible] = np.minimum(prediction.extinction_lower[center_eligible], corner_lower.min(axis=1))
            support_fraction[center_eligible] = (
                prediction.in_data_range[center_eligible].astype(int) + corner_support.sum(axis=1)
            ) / 9
            eligible[center_eligible] = corner_support.all(axis=1) if supported_only else True
        indices = np.flatnonzero(eligible)
        feasible = [int(i) for i in indices if robust_lower[i] >= target]
        feasible.sort(key=lambda i: (costs[i], -robust_lower[i], i))

        def candidate(index: int) -> dict:
            return {
                "prediction": center_records[index], "change_cost": float(costs[index]),
                "deltas": {name: float(coordinates[index, j] - center[j]) for j, name in enumerate(CONTROLS)},
                "robustness": {
                    "corner_count": 8, "minimum_sampled_extinction_lower": float(robust_lower[index]),
                    "support_fraction": float(support_fraction[index]),
                },
            }

        # With two objectives, a cost-ordered running maximum identifies the
        # finite-set frontier in O(n log n). Equal-cost/equal-probability
        # alternatives are retained: neither strictly dominates the other.
        ordered = sorted(indices.tolist(), key=lambda i: (costs[i], -robust_lower[i], i))
        frontier = []
        highest = -np.inf
        position = 0
        while position < len(ordered):
            end = position + 1
            cost = costs[ordered[position]]
            while end < len(ordered) and costs[ordered[end]] == cost:
                end += 1
            group = ordered[position:end]
            group_maximum = robust_lower[group[0]]
            if group_maximum > highest:
                frontier.extend(index for index in group if robust_lower[index] == group_maximum)
                highest = group_maximum
            position = end
        total_frontier = len(frontier)
        if total_frontier > PARETO_LIMIT:
            selected = np.linspace(0, total_frontier - 1, PARETO_LIMIT, dtype=int)
            frontier = [frontier[index] for index in selected]
        best = min(indices.tolist(), key=lambda i: (-robust_lower[i], costs[i], i)) if len(indices) else None
        limitations = _limitations(self.predictor, base["material"])
        limitations.extend([
            "The result is limited to the requested finite grid; not_found does not prove mathematical infeasibility.",
            "Change cost ignores fixed axes; deltas still report any changes on those axes.",
            "Tolerance corners are clipped to the legal API domain and may extend beyond the centre search bounds.",
        ])
        if not supported_only:
            limitations.append("Unsupported conditions were eligible because supported_only=false; inspect support_fraction before interpreting candidates.")
        if not len(indices):
            limitations.append("No candidate satisfied the required centre-and-corner training-support checks.")
        return {
            **model_meta(self.predictor), "status": "found" if feasible else "not_found",
            "target_achievable": bool(feasible), "target_extinction_probability": target,
            "evaluated_count": len(frame), "eligible_count": int(len(indices)),
            "candidates": [candidate(index) for index in feasible[:k]],
            "best_available": candidate(best) if best is not None else None,
            "pareto_front": [candidate(index) for index in frontier],
            "pareto_front_total_count": total_frontier,
            "pareto_front_truncated": total_frontier > PARETO_LIMIT,
            "limitations": limitations,
        }
