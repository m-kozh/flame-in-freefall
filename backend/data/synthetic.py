"""Reproducible, deliberately incomplete demonstration experiments, not NASA data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.loader import COLUMNS, validate_dataframe
from data.materials import MATERIALS
from physics.flame_spread import evaluate_flame


@dataclass(frozen=True)
class SyntheticAssumptions:
    seed: int = 42
    rows: int = 2000
    experiments: int = 20
    log_rate_noise_sigma: float = 0.12
    batch_log_rate_sigma: float = 0.06
    extinction_transition_percent: float = 1.2
    thickness_relative_sigma: float = 0.06
    # Each experimental campaign occupies one small patch, not the whole domain.
    patches: tuple = (
        (12., 26., 80., 115., .15, 12.),
        (18., 35., 45., 80., 3., 35.),
        (22., 45., 65., 105., 20., 100.),
        (14., 40., 90., 120., 1., 25.),
    )


ASSUMPTIONS = SyntheticAssumptions()


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def generate_synthetic(n_rows: int = 2000, n_experiments: int = 20, seed: int = 42) -> pd.DataFrame:
    """Generate the standard CSV schema with a smooth, hidden model discrepancy.

    Group offsets model unobserved batch conditions. The excluded high-oxygen,
    high-pressure, high-flow corner makes a genuine hole in training support.
    """
    if n_experiments < 5 or n_rows < 10 * n_experiments:
        raise ValueError("Use at least 5 experiments and 10 rows per experiment.")
    a = ASSUMPTIONS
    rng = np.random.default_rng(seed)
    materials = list(MATERIALS.values())
    records = []
    for group in range(n_experiments):
        material = materials[group % len(materials)]
        patch = a.patches[(group // len(materials)) % len(a.patches)]
        n = n_rows // n_experiments + int(group < n_rows % n_experiments)
        o_lo, o_hi, p_lo, p_hi, u_lo, u_hi = patch
        o2 = rng.uniform(o_lo, o_hi, n)
        pressure = rng.uniform(p_lo, p_hi, n)
        flow = np.exp(rng.uniform(np.log(u_lo), np.log(u_hi), n))
        hole = (o2 > 32) & (pressure > 95) & (flow > 35)
        while hole.any():
            o2[hole] = rng.uniform(o_lo, o_hi, hole.sum())
            pressure[hole] = rng.uniform(p_lo, p_hi, hole.sum())
            flow[hole] = np.exp(rng.uniform(np.log(u_lo), np.log(u_hi), hole.sum()))
            hole = (o2 > 32) & (pressure > 95) & (flow > 35)
        thickness = material.thickness_mm * np.clip(rng.normal(1, a.thickness_relative_sigma, n), .8, 1.2)
        physics = evaluate_flame(
            o2, pressure, flow, thickness, material.density_kg_m3,
            material.specific_heat_j_kgk, material.vaporization_temp_k, material.limiting_o2_percent,
        )
        # Hidden smooth discrepancy: the learner sees observations, never these terms.
        residual = (
            .32 * np.sin(o2 / 9) + .24 * np.cos(np.log1p(flow))
            + .22 * np.log(pressure / 80) + .10 * np.log(material.density_kg_m3 / 550)
            + rng.normal(0, a.batch_log_rate_sigma)
        )
        boundary_bias = 1.8 * np.sin(np.log1p(flow)) + .8 * np.cos(o2 / 10) + .6 * np.log(pressure / 80)
        p_extinction = _sigmoid((boundary_bias - physics["o2_margin_percent"]) / a.extinction_transition_percent)
        sustained = rng.random(n) > p_extinction
        ignited = sustained | (rng.random(n) < _sigmoid((o2 - physics["minimum_o2_percent"] + 3) / 1.5))
        rates = physics["potential_spread_rate_mm_s"] * np.exp(residual + rng.normal(0, a.log_rate_noise_sigma, n))
        records.append(pd.DataFrame({
            "experiment_id": f"SYN-{group + 1:02d}", "campaign": "synthetic",
            "material": material.id, "thickness_mm": thickness,
            "o2_percent": o2, "pressure_kpa": pressure, "flow_cm_s": flow,
            "gravity_g": 0., "ignited": ignited, "extinguished": ignited & ~sustained,
            "spread_rate_mm_s": np.where(sustained, rates, np.nan), "source": "synthetic",
        }))
    return validate_dataframe(pd.concat(records, ignore_index=True).loc[:, COLUMNS])
