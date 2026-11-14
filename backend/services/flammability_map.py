"""Batched flammability maps with a small, instance-local prediction cache."""

from copy import deepcopy
from functools import lru_cache

import numpy as np
import pandas as pd

from data.materials import get_material
from data.presets import PRESETS
from models.base import Predictor
from models.residual_ensemble import physics_frame


def _boundary_points(
    flow_values: np.ndarray, o2_values: np.ndarray, probability: np.ndarray,
) -> list[dict]:
    """Find every row's highest oxygen crossing, without joining absent branches."""
    points = []
    for flow, row in zip(flow_values, probability):
        offsets = row - 0.5
        crossings = o2_values[offsets == 0].tolist()
        # Strict sign changes avoid dividing by zero on constant-probability cells.
        indices = np.flatnonzero(offsets[:-1] * offsets[1:] < 0)
        for index in indices:
            fraction = -offsets[index] / (offsets[index + 1] - offsets[index])
            crossings.append(
                float(o2_values[index] + fraction * (o2_values[index + 1] - o2_values[index]))
            )
        if crossings:
            points.append({"flow_cm_s": float(flow), "o2_percent": float(max(crossings))})
    return points


class FlammabilityMapService:
    """One batch prediction per uncached grid; callers receive isolated results.

    Construct a fresh service when loading a new model. Its cache belongs only to
    that predictor instance, so cached maps cannot survive a model replacement.
    """

    def __init__(self, predictor: Predictor):
        self.predictor = predictor
        self._cached_map = lru_cache(maxsize=32)(self._compute_map)

    def get_map(
        self,
        material,
        pressure_kpa=101.3,
        flow_range=(0.5, 100.0),
        o2_range=(12.0, 40.0),
        resolution=50,
        flow_scale="log",
        thickness_mm=None,
    ) -> dict:
        """Return rows=flow, columns=oxygen, matching the frontend heat-map axes."""
        sample = get_material(material)
        pressure = float(pressure_kpa)
        thickness = sample.thickness_mm if thickness_mm is None else float(thickness_mm)
        flow_limits = tuple(float(value) for value in flow_range)
        oxygen_limits = tuple(float(value) for value in o2_range)
        # API validation is authoritative; these checks also make direct use clear.
        if len(flow_limits) != 2 or len(oxygen_limits) != 2:
            raise ValueError("flow_range and o2_range must each contain [min, max].")
        if not np.isfinite([pressure, thickness, *flow_limits, *oxygen_limits]).all():
            raise ValueError("Map parameters must be finite numbers.")
        if not 10 <= pressure <= 150 or not 0.01 <= thickness <= 10:
            raise ValueError("pressure_kpa must be in [10, 150]; thickness_mm in [0.01, 10].")
        if not 0 <= flow_limits[0] < flow_limits[1] <= 200:
            raise ValueError("flow_range must satisfy 0 <= min < max <= 200 cm/s.")
        if not 5 <= oxygen_limits[0] < oxygen_limits[1] <= 100:
            raise ValueError("o2_range must satisfy 5 <= min < max <= 100 percent.")
        if isinstance(resolution, bool) or not isinstance(resolution, int) or not 5 <= resolution <= 100:
            raise ValueError("resolution must be an integer in [5, 100].")
        if flow_scale not in ("log", "linear"):
            raise ValueError("flow_scale must be 'log' or 'linear'.")
        if flow_scale == "log" and flow_limits[0] <= 0:
            raise ValueError("Logarithmic flow spacing requires a positive minimum flow.")
        # Canonical material id and explicit default thickness make aliases and
        # equivalent requests share a cache entry. No mutable objects enter keys.
        result = self._cached_map(
            sample.id, pressure, flow_limits, oxygen_limits,
            resolution, flow_scale, thickness,
        )
        return deepcopy(result)

    def _compute_map(
        self, material, pressure, flow_limits, oxygen_limits, resolution,
        flow_scale, thickness,
    ) -> dict:
        spacing = np.geomspace if flow_scale == "log" else np.linspace
        flow_values = spacing(*flow_limits, resolution)
        o2_values = np.linspace(*oxygen_limits, resolution)
        oxygen_grid, flow_grid = np.meshgrid(o2_values, flow_values, indexing="xy")
        grid = pd.DataFrame({
            "material": material,
            "o2_percent": oxygen_grid.ravel(),
            "pressure_kpa": pressure,
            "flow_cm_s": flow_grid.ravel(),
            "thickness_mm": thickness,
        })
        prediction = self.predictor.predict(grid)
        shape = (resolution, resolution)
        probability = prediction.extinction_probability.reshape(shape)
        physical_limit = physics_frame(grid)["limiting_o2_percent"].reshape(shape)[:, 0]
        flow_list, oxygen_list = flow_values.tolist(), o2_values.tolist()
        probability_list = probability.tolist()
        return {
            "material": material,
            "pressure_kpa": pressure,
            "thickness_mm": thickness,
            "flow_scale": flow_scale,
            "flow_values": flow_list,
            "o2_values": oxygen_list,
            "spread_rate": prediction.mean.reshape(shape).tolist(),
            "extinction_probability": probability_list,
            "uncertainty": prediction.uncertainty.reshape(shape).tolist(),
            "boundary": _boundary_points(flow_values, o2_values, probability),
            "physics_boundary": [
                {"flow_cm_s": float(flow), "o2_percent": float(limit)}
                for flow, limit in zip(flow_values, physical_limit)
                if oxygen_limits[0] <= limit <= oxygen_limits[1]
            ],
            "presets_overlay": [
                {
                    "id": preset["id"], "name": preset["name"],
                    "o2_percent": preset["o2_percent"],
                    "pressure_kpa": preset["pressure_kpa"],
                    "flow_cm_s": preset["flow_cm_s"],
                    "matches_map_pressure": abs(preset["pressure_kpa"] - pressure) <= 10,
                }
                for preset in PRESETS
            ],
            "tested_points": self._tested_points(material, pressure),
            "x_axis": "o2", "y_axis": "flow_cm_s",
            "x": oxygen_list, "y": flow_list, "z": probability_list,
            "model_version": self.predictor.model_version,
            "data_source": self.predictor.data_source,
        }

    def _tested_points(self, material: str, pressure: float) -> list[dict]:
        data = self.predictor.training_data
        if data.empty:
            return []
        selected = data.loc[
            data["material"].eq(material) & data["pressure_kpa"].sub(pressure).abs().le(10)
        ].head(500)
        return [
            {
                "o2_percent": float(row.o2_percent), "o2": float(row.o2_percent),
                "flow_cm_s": float(row.flow_cm_s),
                "pressure_kpa": float(row.pressure_kpa),
                "burned": bool(row.ignited and not row.extinguished),
                "source": str(row.source), "experiment_id": str(row.experiment_id),
                "campaign": str(row.campaign), "thickness_mm": float(row.thickness_mm),
            }
            for row in selected.itertuples(index=False)
        ]
