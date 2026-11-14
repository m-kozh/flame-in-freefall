"""Vectorised thermally thin, opposed-flow flame-spread demonstration.

The de Ris / Delichatsios-inspired heat-balance expression and all numerical
constants below are hackathon approximations. They are not a validated fire
safety model, and require checks against the relevant material and literature.
The U-shaped oxygen limit is an illustrative mechanism, not a fitted law.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicsConstants:
    """All assumed constants; SI units unless the field name specifies otherwise."""

    heat_balance_coefficient: float = np.pi / 4
    gas_conductivity_w_mk: float = 0.04  # Assumed effective gas conductivity.
    ambient_temperature_k: float = 293.15
    flame_temperature_intercept_k: float = 900.0
    flame_temperature_per_o2_percent_k: float = 35.0
    minimum_flame_superheat_k: float = 25.0  # Keeps the latent rate positive.
    reference_pressure_kpa: float = 101.3
    pressure_rate_exponent: float = 0.15  # Mild pressure enhancement of spread.
    pressure_limit_coefficient_percent: float = 2.0
    reference_thickness_mm: float = 0.2
    thickness_limit_coefficient_percent: float = 1.2
    quench_coefficient_percent_cm_s: float = 14.0
    quench_flow_offset_cm_s: float = 0.4
    blowoff_coefficient_percent: float = 0.6
    blowoff_flow_scale_cm_s: float = 10.0
    blowoff_exponent: float = 1.5
    mm_per_m: float = 1000.0


# These are deliberately visible assumptions, not precise physical constants.
CONSTANTS = PhysicsConstants()


def _optimal_flow_cm_s() -> float:
    """Solve the analytic derivative equation for the unique positive minimum.

    With t=sqrt(u), derivative=0 becomes
    t**5 + 2*s*t**3 + s**2*t - A*U**1.5/(1.5*B) = 0.
    The left side is strictly increasing for t>=0, so there is one positive root.
    """
    c = CONSTANTS
    rhs = (
        c.quench_coefficient_percent_cm_s
        * c.blowoff_flow_scale_cm_s**c.blowoff_exponent
        / (c.blowoff_coefficient_percent * c.blowoff_exponent)
    )
    s = c.quench_flow_offset_cm_s
    roots = np.roots([1.0, 0.0, 2 * s, 0.0, s * s, -rhs])
    positive = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
    return float(positive[0] ** 2)


OPTIMAL_FLOW_CM_S = _optimal_flow_cm_s()


def _flow_limit_increment(flow_cm_s: np.ndarray) -> np.ndarray:
    c = CONSTANTS
    return (
        c.quench_coefficient_percent_cm_s / (flow_cm_s + c.quench_flow_offset_cm_s)
        + c.blowoff_coefficient_percent
        * (flow_cm_s / c.blowoff_flow_scale_cm_s) ** c.blowoff_exponent
    )


def evaluate_flame(
    o2_percent,
    pressure_kpa,
    flow_cm_s,
    thickness_mm,
    density_kg_m3,
    specific_heat_j_kgk,
    vaporization_temp_k,
    limiting_o2_percent,
) -> dict[str, np.ndarray]:
    """Evaluate broadcast-compatible scalar/array inputs without Python row loops.

    ``limiting_o2_percent`` is the material's base oxygen limit before flow,
    thickness and pressure corrections. Oxygen is a percentage (21, not .21).
    Every returned value is a NumPy array, including a 0-D array for scalar input.
    The potential rate remains positive in extinction regions so that residual
    models can learn departures from the baseline without taking log(0).
    """
    values = np.broadcast_arrays(
        *[
            np.asarray(value, dtype=float)
            for value in (
                o2_percent,
                pressure_kpa,
                flow_cm_s,
                thickness_mm,
                density_kg_m3,
                specific_heat_j_kgk,
                vaporization_temp_k,
                limiting_o2_percent,
            )
        ]
    )
    o2, pressure, flow, thickness, density, heat_capacity, vapor_temp, base_limit = values
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("Physics inputs must contain finite numbers only.")
    if np.any(pressure <= 0) or np.any(thickness <= 0):
        raise ValueError("pressure_kpa and thickness_mm must be positive.")
    if np.any(flow < 0) or np.any(o2 < 0) or np.any(o2 > 100):
        raise ValueError("flow_cm_s must be nonnegative; o2_percent must be in [0, 100].")
    if np.any(density <= 0) or np.any(heat_capacity <= 0):
        raise ValueError("Material density and specific heat must be positive.")
    c = CONSTANTS
    if np.any(vapor_temp <= c.ambient_temperature_k):
        raise ValueError("vaporization_temp_k must exceed ambient temperature.")

    # A simple monotonic flame-temperature approximation, not equilibrium chemistry.
    flame_temp = c.flame_temperature_intercept_k + c.flame_temperature_per_o2_percent_k * o2
    superheat = np.maximum(flame_temp - vapor_temp, c.minimum_flame_superheat_k)
    thickness_m = thickness / c.mm_per_m
    potential_rate = (
        c.heat_balance_coefficient
        * c.gas_conductivity_w_mk
        * superheat
        / (density * heat_capacity * thickness_m * (vapor_temp - c.ambient_temperature_k))
        * (pressure / c.reference_pressure_kpa) ** c.pressure_rate_exponent
        * c.mm_per_m
    )

    # Low-flow radiation losses and high-flow finite residence time form a U.
    adjusted_limit = (
        base_limit
        - c.pressure_limit_coefficient_percent * np.log(pressure / c.reference_pressure_kpa)
        + c.thickness_limit_coefficient_percent * np.log(thickness / c.reference_thickness_mm)
    )
    limit = adjusted_limit + _flow_limit_increment(flow)
    minimum_limit = adjusted_limit + _flow_limit_increment(np.asarray(OPTIMAL_FLOW_CM_S))
    margin = o2 - limit
    burns = margin >= 0
    regime = np.where(
        burns,
        "burning",
        np.where(
            o2 < minimum_limit,
            "no_ignition",
            np.where(flow < OPTIMAL_FLOW_CM_S, "quench_extinction", "blowoff_extinction"),
        ),
    )
    return {
        "spread_rate_mm_s": np.asarray(np.where(burns, potential_rate, 0.0)),
        "potential_spread_rate_mm_s": np.asarray(potential_rate),
        "burns": np.asarray(burns),
        "regime": np.asarray(regime),
        "limiting_o2_percent": np.asarray(limit),
        "minimum_o2_percent": np.asarray(minimum_limit),
        "o2_margin_percent": np.asarray(margin),
    }
