"""Offline NASA evidence snapshots and deliberately narrow training conversion.

Catalogue entries are evidence, not automatically training examples. Original
source hashes and local snapshot hashes have different roles and are both kept.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pandas as pd

from data.loader import COLUMNS, validate_dataframe

CURATED_DIR = Path(__file__).resolve().parent / "nasa_curated"
CR_DATASET_ID = "nasa-cr-182185"
CR_CAMPAIGN = "NASA-CR-182185"


class SourceIntegrityError(ValueError):
    """Downloaded source or bundled snapshot does not match its recorded hash."""


def verify_sha256(path: str | Path, expected: str) -> str:
    """Verify a file without relying on a network service or a local filename."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise SourceIntegrityError(
            f"SHA256 mismatch for {Path(path).name}: expected {expected}, got {actual}."
        )
    return actual


def _registry() -> dict:
    return json.loads((CURATED_DIR / "registry.json").read_text(encoding="utf-8"))


def _entry(dataset_id: str) -> dict:
    for entry in _registry()["datasets"]:
        if entry["id"] == dataset_id:
            return entry
    raise KeyError(dataset_id)


def _snapshot_bytes(entry: dict) -> bytes:
    spec = entry["curated_file"]
    path = CURATED_DIR / spec["name"]
    verify_sha256(path, spec["sha256"])
    return path.read_bytes()


def dataset_catalog() -> list[dict]:
    """Return stable public cards; model inclusion is decided by the API layer."""
    keys = (
        "id", "title", "source_url", "doi", "license", "record_count",
        "training_row_count", "observation_window_s", "scope", "source_files",
        "limitations",
    )
    cards = []
    for entry in _registry()["datasets"]:
        _snapshot_bytes(entry)
        card = {key: copy.deepcopy(entry[key]) for key in keys}
        card["source_files"] = [
            {key: file[key] for key in ("name", "url", "sha256")}
            for file in entry["source_files"]
        ]
        cards.append(card)
    return cards


def dataset_observations(dataset_id: str) -> list[dict]:
    """Return source-specific observations, preserving missing/censored values.

    Unknown IDs raise KeyError. Returned dictionaries are independent copies;
    callers cannot alter a future request or the training source in memory.
    """
    entry = _entry(dataset_id)
    content = _snapshot_bytes(entry)
    if entry["curated_file"]["format"] == "json":
        records = json.loads(content)
    else:
        rows = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
        records = [
            {
                "record_id": f"PSI-25-BASS-II-catalogue-{index:03d}",
                "catalogue_row": index,
                "source_reference": {
                    "url": entry["source_url"], "study_id": "PSI-25",
                    "study_version": 4, "file": "PSI-25_BASS-II.csv",
                    "csv_data_row": index,
                },
                "training_eligible": False,
                "exclusion_reason": (
                    "The catalogue supplies no complete spread-rate and binary "
                    "ignition/extinction targets. Time-varying air/fan strings "
                    "are preserved without inventing a numerical condition."
                ),
                "source_fields": dict(row),
            }
            for index, row in enumerate(rows, 1)
        ]
    if len(records) != entry["record_count"]:
        raise SourceIntegrityError(f"Unexpected observation count for {dataset_id}.")
    return records


def nasa_training_rows() -> pd.DataFrame:
    """The 11 complete Table I records, with one conservative report group.

    Two positive but left-censored rates are excluded. An extinction has an
    absent rate, not a fabricated zero. 'Not extinguished' refers only to the
    5.18-second observation window. Fuel speed is the environmental relative
    flow; characteristic velocity would include and leak the target flame rate.
    """
    rows = []
    for observation in dataset_observations(CR_DATASET_ID):
        if not observation["training_eligible"]:
            continue
        raw = observation["source_values"]
        rate = raw["spread_rate_cm_s"]
        rows.append({
            "experiment_id": CR_CAMPAIGN,
            "campaign": CR_CAMPAIGN,
            "material": "cellulose_thin",
            "thickness_mm": float(Decimal("0.0038") * 2 * 10),
            "o2_percent": float(Decimal(raw["o2_percent"])),
            "pressure_kpa": 101.325,
            "flow_cm_s": float(Decimal(raw["fuel_speed_cm_s"])),
            "gravity_g": 0.0,
            "ignited": True,
            "extinguished": observation["extinguished"],
            "spread_rate_mm_s": None if rate is None else float(Decimal(rate) * 10),
            "source": "nasa",
        })
    return validate_dataframe(pd.DataFrame(rows, columns=COLUMNS))


def _signature(row: pd.Series) -> tuple | None:
    """Normalize validated CSV values for exact curated-record recognition."""
    if any(column not in row.index for column in COLUMNS):
        return None
    numeric = {
        "thickness_mm", "o2_percent", "pressure_kpa", "flow_cm_s",
        "gravity_g", "spread_rate_mm_s",
    }
    values = []
    try:
        for column in COLUMNS:
            value = row[column]
            if column in numeric:
                values.append(None if pd.isna(value) else float(value))
            elif column in {"ignited", "extinguished"}:
                parsed = {"true": True, "false": False, "1": True, "0": False}.get(str(value).lower())
                if parsed is None:
                    return None
                values.append(parsed)
            else:
                values.append(str(value))
    except (ValueError, TypeError):
        return None
    return tuple(values)


def training_evidence_summary(frame: pd.DataFrame) -> dict:
    """Deterministic evidence metadata for the actual selected training frame.

    Source identifies NASA rows and campaign identifies their candidate dataset;
    complete value matching determines whether labels belong to the checked snapshot.
    Unknown or modified NASA rows never inherit the curated observation window.
    """
    def strings(column: str) -> pd.Series:
        if column in frame:
            return frame[column].fillna("").astype(str)
        return pd.Series("", index=frame.index, dtype=str)

    source = strings("source")
    campaign = strings("campaign")
    mask = source.str.lower().eq("nasa")
    campaign_conflicts = campaign.eq(CR_CAMPAIGN) & ~mask
    selected = frame.loc[mask]
    known_signatures = {_signature(row) for _, row in nasa_training_rows().iterrows()}
    verified_count = sum(
        _signature(row) in known_signatures for _, row in selected.iterrows()
    )
    unverified_count = len(selected) - verified_count
    registry = _registry()
    cr = _entry(CR_DATASET_ID)
    limitations = list(cr["limitations"]) if verified_count else []
    if unverified_count:
        limitations.append(
            "Some selected NASA-labelled rows do not exactly match the independently "
            "verified curated snapshot; their labels, observation windows and source "
            "provenance have not been independently verified."
        )
    if not len(selected):
        limitations.append("No NASA observations were selected for this training frame.")
    if campaign_conflicts.any():
        limitations.append(
            "Some rows use the curated NASA campaign identifier but are not marked "
            "source=nasa; they are not counted as NASA training evidence."
        )
    if verified_count:
        limitations.append(
            "These finite-window labels do not establish indefinite sustained burning. "
            "One report-level group does not support an independently validated NASA-only model."
        )

    def counts(column: str) -> dict[str, int]:
        return dict(sorted(Counter(strings(column).loc[mask]).items()))

    return {
        "metadata_origin": "saved_training_snapshot",
        "registry_version": registry["registry_version"],
        "registry_sha256": hashlib.sha256((CURATED_DIR / "registry.json").read_bytes()).hexdigest(),
        "nasa_row_count": int(len(selected)),
        "verified_curated_row_count": int(verified_count),
        "unverified_nasa_row_count": int(unverified_count),
        "campaign_source_conflict_row_count": int(campaign_conflicts.sum()),
        "dataset_ids": [CR_DATASET_ID] if verified_count else [],
        "campaign_counts": counts("campaign"),
        "material_counts": counts("material"),
        "verified_materials": ["cellulose_thin"] if verified_count else [],
        "verified_pressure_kpa": [101.325] if verified_count else [],
        "observation_window_s": 5.18 if verified_count and not unverified_count else None,
        "verified_observation_window_s": 5.18 if verified_count else None,
        "target_definitions": {
            "ignited": "All curated Table I conditions concern previously ignited flames.",
            "extinguished_true": "Extinction observed during the 5.18-second microgravity interval.",
            "extinguished_false": "No extinction observed within 5.18 seconds; survival is right-censored at the end of the observation window.",
            "spread_rate_mm_s": "Measured time-average cm/s multiplied by ten, for eight non-extinguished conditions; absent after observed extinction.",
            "censored_rates": "Two non-extinguished conditions report rates below 0.5 mm/s and are excluded from the complete-target training schema.",
            "applicability": "Definitions apply only to exact matches of the NASA-CR-182185 curated rows; unknown NASA labels are unverified.",
        },
        "source_files": [
            {key: file[key] for key in ("name", "url", "sha256")}
            for file in cr["source_files"]
        ] if verified_count else [],
        "limitations": limitations,
    }
