"""Reproducible scientific fingerprints, independent of row order and timestamps.

These hashes describe what was trained; they are not authenticity signatures.
Dataset floats use their exact IEEE-754 value via ``float.hex``. A CSV reader that
changes the last binary digit can therefore change the fingerprint: retain the
validated training CSV and parser version when reproducing a run.
"""

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from data.loader import COLUMNS, validate_dataframe
from data.materials import MATERIALS
import physics.flame_spread as physics_module


def _canonical(value):
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("Fingerprint metadata must contain finite floats.")
        return {"float_hex": float(value).hex()}
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"Unsupported fingerprint value: {type(value).__name__}")


def stable_fingerprint(value) -> str:
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Hash the validated standard schema, preserving duplicates but ignoring row order."""
    data = validate_dataframe(df)
    rows = []
    for row in data.itertuples(index=False, name=None):
        # Only the non-burning speed column can be missing after validation.
        canonical_row = [None if pd.isna(value) else value for value in row]
        rows.append(json.dumps(_canonical(canonical_row), separators=(",", ":"), ensure_ascii=False))
    return stable_fingerprint({"schema": COLUMNS, "rows": sorted(rows)})


def physics_material_fingerprint() -> str:
    properties = [
        "density_kg_m3", "specific_heat_j_kgk", "thickness_mm",
        "vaporization_temp_k", "thermally_thin", "limiting_o2_percent",
    ]
    return stable_fingerprint({
        "formula": "thermally_thin_heat_balance_with_u_shaped_oxygen_limit_v1",
        "source_sha256": _physics_source_sha256(),
        "physics_constants": asdict(physics_module.CONSTANTS),
        "materials": {
            material_id: {name: getattr(material, name) for name in properties}
            for material_id, material in MATERIALS.items()
        },
    })


def _physics_source_sha256() -> str:
    """Normalise line endings so a Windows checkout has the same source hash."""
    source = Path(physics_module.__file__).read_text(encoding="utf-8")
    normalised = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def fitted_models_fingerprint(model) -> str:
    return stable_fingerprint({
        "regressors": [member.booster_.model_to_string() for member in model.regressors],
        "classifiers": [member.booster_.model_to_string() for member in model.classifiers],
        "noise_log_std": model.noise_log_std,
    })


def build_provenance(model, validation_config: dict | None = None) -> dict:
    """Fingerprint dataset, assumptions, training configuration and fitted trees.

    Validation configuration is part of the validated-model identity, although it
    does not alter the final fitted trees. The fitted-model hash is separate.
    """
    # Local import avoids a circular dependency during ResidualEnsemble.fit.
    from models.residual_ensemble import CONSTANTS as ENSEMBLE_CONSTANTS

    config = {
        "model_version": model.model_version, "n_members": model.n_members,
        "seed": model.seed, "use_physics": model.use_physics,
        "features": list(model.feature_names), "constants": asdict(ENSEMBLE_CONSTANTS),
        "lightgbm_parameters": model._parameters(model.seed),
        "validation": validation_config,
        "training_evidence": getattr(model, "training_evidence", None),
        "runtime_versions": {
            "python": platform.python_version(),
            **{package: version(package) for package in ["numpy", "pandas", "scikit-learn", "lightgbm"]},
        },
    }
    hashes = {
        "dataset_sha256": dataset_fingerprint(model.training_data),
        "physics_material_sha256": physics_material_fingerprint(),
        "training_config_sha256": stable_fingerprint(config),
        "fitted_models_sha256": fitted_models_fingerprint(model),
    }
    model_id = stable_fingerprint({"schema_version": 1, **hashes})
    return {
        "schema_version": 1, **hashes, "model_id": model_id,
        "training_config": config,
        "methods": {
            "algorithm": "SHA256",
            "dataset": "Validated standard-schema rows sorted as canonical JSON; duplicates preserved; floats encoded with float.hex; missing non-burning rates encoded as null.",
            "row_order_invariant": True,
            "float_precision": "Exact binary64 values; CSV parser roundtrip differences may change the dataset hash.",
            "physics_material": "LF-normalised physics/flame_spread.py source SHA256, formula identifier, runtime physics constants and material thermal properties; not a literature validation.",
            "fitted_models": "Ordered LightGBM model text and fitted log-noise scale; independent of pickle bytes and timestamps.",
            "model_id": "SHA256 of fingerprint schema and the four component hashes; validation configuration is included in training_config.",
        },
    }


def metrics_match_model(predictor, metrics: dict | None) -> bool:
    """Match report evidence to its artifact without trusting row counts alone.

    Both old artifacts and old metrics may lack provenance, preserving legacy
    inference. Once either side records it, all component hashes and the complete
    model identity must be present and match. Malformed JSON metadata is rejected.
    """
    if not isinstance(metrics, dict):
        return False
    try:
        n_rows = len(predictor.training_data)
    except (AttributeError, TypeError):
        return False
    if not (
        isinstance(metrics.get("n_rows"), int)
        and not isinstance(metrics.get("n_rows"), bool)
        and metrics.get("model_version") == getattr(predictor, "model_version", None)
        and metrics.get("data_source") == getattr(predictor, "data_source", None)
        and metrics["n_rows"] == n_rows
    ):
        return False
    model_provenance = getattr(predictor, "provenance", None)
    report_provenance = metrics.get("provenance")
    if model_provenance is None and report_provenance is None:
        return True
    if not isinstance(model_provenance, dict) or not isinstance(report_provenance, dict):
        return False
    keys = [
        "dataset_sha256", "physics_material_sha256", "training_config_sha256",
        "fitted_models_sha256", "model_id",
    ]
    for key in keys:
        recorded = model_provenance.get(key)
        if not isinstance(recorded, str) or not recorded or report_provenance.get(key) != recorded:
            return False
    model_id = getattr(predictor, "model_id", None)
    return (
        isinstance(model_id, str) and bool(model_id)
        and model_id == model_provenance["model_id"] == metrics.get("model_id")
    )


def validate_runtime_provenance(model) -> None:
    """Reject a recorded artifact whose live assumptions or identity have changed.

    This schema belongs to ResidualEnsemble. Other Predictor implementations own
    their own artifact validation. Legacy ensembles without recorded provenance
    remain loadable, with their unverified status exposed by the model card.
    These consistency checks are not a signature or a defence against malicious
    pickle files; only trusted local artifacts should ever be deserialised.
    """
    from models.residual_ensemble import (
        CONSTANTS as ensemble_constants, PHYSICS_FEATURES, RAW_FEATURES, ResidualEnsemble,
    )

    if not isinstance(model, ResidualEnsemble):
        return
    provenance = getattr(model, "provenance", None)
    if provenance is None:
        return

    def reject(reason):
        raise ValueError(
            f"Artifact runtime provenance mismatch: {reason}. "
            "Run python scripts/train_model.py to retrain with the current physics, materials and model configuration."
        )

    if not isinstance(provenance, dict) or type(provenance.get("schema_version")) is not int or provenance["schema_version"] != 1:
        reject("missing or unsupported provenance schema")
    keys = ["dataset_sha256", "physics_material_sha256", "training_config_sha256", "fitted_models_sha256"]
    for key in [*keys, "model_id"]:
        value = provenance.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            reject(f"malformed {key}")
    identity = stable_fingerprint({"schema_version": 1, **{key: provenance[key] for key in keys}})
    if identity != provenance["model_id"] or getattr(model, "model_id", None) != identity:
        reject("model identity does not agree with its recorded component fingerprints")
    config = provenance.get("training_config")
    if not isinstance(config, dict) or not isinstance(config.get("constants"), dict):
        reject("malformed training configuration")
    try:
        if stable_fingerprint(config) != provenance["training_config_sha256"]:
            reject("recorded training configuration fingerprint is inconsistent")
        if "training_evidence" in config and stable_fingerprint(config["training_evidence"]) != stable_fingerprint(getattr(model, "training_evidence", None)):
            reject("training evidence metadata changed since fingerprinting")
        if physics_material_fingerprint() != provenance["physics_material_sha256"]:
            reject("physics source, physics constants or material properties changed since training")
        if stable_fingerprint(config["constants"]) != stable_fingerprint(asdict(ensemble_constants)):
            reject("ensemble constants changed since training")
        expected_features = RAW_FEATURES + (PHYSICS_FEATURES if model.use_physics else [])
        if config.get("features") != expected_features or model.feature_names != expected_features:
            reject("runtime feature definition changed since training")
        for key in ["model_version", "n_members", "seed", "use_physics"]:
            if config.get(key) != getattr(model, key):
                reject(f"artifact {key} differs from its recorded training configuration")
        if dataset_fingerprint(model.training_data) != provenance["dataset_sha256"]:
            reject("artifact training observations changed since fingerprinting")
        if fitted_models_fingerprint(model) != provenance["fitted_models_sha256"]:
            reject("fitted trees or noise scale changed since fingerprinting")
    except (TypeError, KeyError, AttributeError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Artifact runtime provenance mismatch:"):
            raise
        reject(f"cannot validate malformed artifact metadata ({type(exc).__name__})")
