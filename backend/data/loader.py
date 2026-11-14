"""One explicit CSV schema for synthetic and future NASA observations.

Column suffixes specify units. Ranges catch common mistakes (Pa instead of kPa,
oxygen fraction instead of percent); no loader can infer every unit mistake.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data.materials import resolve_material

COLUMNS = [
    "experiment_id", "campaign", "material", "thickness_mm", "o2_percent",
    "pressure_kpa", "flow_cm_s", "gravity_g", "ignited", "extinguished",
    "spread_rate_mm_s", "source",
]
RANGES = {
    "thickness_mm": (0.01, 10), "o2_percent": (5, 100),
    "pressure_kpa": (10, 150), "flow_cm_s": (0, 200), "gravity_g": (0, 1),
}


class DataValidationError(ValueError):
    """A human-readable error in a source data file."""


def _bad_rows(mask: pd.Series) -> str:
    return ", ".join(str(i + 2) for i in np.flatnonzero(mask.to_numpy())[:5])


def validate_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(COLUMNS) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(COLUMNS))
    if missing or extra:
        raise DataValidationError(
            f"CSV schema mismatch: missing columns={missing}; unexpected columns={extra}. "
            "Use the documented column names and units (o2_percent, pressure_kpa, flow_cm_s, thickness_mm)."
        )
    if frame.empty:
        raise DataValidationError("CSV must contain at least one observation.")
    df = frame.loc[:, COLUMNS].copy().reset_index(drop=True)
    for column in ["experiment_id", "campaign", "material", "source"]:
        invalid = df[column].isna() | df[column].astype(str).str.strip().eq("")
        if invalid.any():
            raise DataValidationError(f"{column} must be nonempty; CSV rows {_bad_rows(invalid)}.")
        df[column] = df[column].astype(str).str.strip()
    try:
        df["material"] = df["material"].map(resolve_material)
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc
    if not df["source"].isin(["synthetic", "nasa"]).all():
        raise DataValidationError("source must be 'synthetic' or 'nasa' (lowercase).")
    for column, (lo, hi) in RANGES.items():
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = ~np.isfinite(values) | ~values.between(lo, hi)
        if invalid.any():
            raise DataValidationError(
                f"{column} must be finite and in [{lo}, {hi}] in the column's named units; "
                f"CSV rows {_bad_rows(invalid)}. Oxygen uses percent, pressure kPa, flow cm/s, thickness mm."
            )
        df[column] = values.astype(float)
    bool_values = {"true": True, "false": False, "1": True, "0": False}
    for column in ["ignited", "extinguished"]:
        parsed = df[column].astype(str).str.strip().str.lower().map(bool_values)
        if parsed.isna().any():
            raise DataValidationError(f"{column} must be true/false or 1/0; CSV rows {_bad_rows(parsed.isna())}.")
        df[column] = parsed.astype(bool)
    inconsistent = ~df["ignited"] & df["extinguished"]
    if inconsistent.any():
        raise DataValidationError("extinguished=true requires ignited=true; an unignited sample cannot self-extinguish.")
    rate = pd.to_numeric(df["spread_rate_mm_s"], errors="coerce")
    # A blank is allowed for non-burning tests, but malformed strings are never blanks.
    provided = df["spread_rate_mm_s"].notna() & df["spread_rate_mm_s"].astype(str).str.strip().ne("")
    malformed = provided & (~np.isfinite(rate) | (rate < 0))
    burning = df["ignited"] & ~df["extinguished"]
    if malformed.any() or (burning & (rate.isna() | (rate <= 0))).any():
        raise DataValidationError("spread_rate_mm_s must be finite and positive for sustained burning; otherwise blank or zero.")
    if (~burning & rate.fillna(0).ne(0)).any():
        raise DataValidationError("Non-burning observations must have blank or zero spread_rate_mm_s.")
    df["spread_rate_mm_s"] = rate.astype(float)
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"Data file not found: {path}. Run python scripts/generate_data.py first.")
    try:
        # Preserve identifiers such as 001 instead of silently converting them to integers.
        frame = pd.read_csv(path, dtype={"experiment_id": str, "campaign": str, "material": str, "source": str})
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Cannot read CSV {path}: {exc}") from exc
    return validate_dataframe(frame)
