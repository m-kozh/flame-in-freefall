"""A readable model card backed by training evidence rather than marketing claims."""

from copy import deepcopy

from models.provenance import metrics_match_model
from services.predictions import model_meta

RANGE_COLUMNS = ["o2_percent", "pressure_kpa", "flow_cm_s", "thickness_mm", "gravity_g"]
LIMITATIONS = [
    "The physics baseline and material properties are hackathon approximations, not validated fire-safety parameters.",
    "Nominal 90% speed intervals combine bootstrap disagreement and empirical group-OOB residual noise; coverage is measured, not guaranteed.",
    "Extinction member quantiles are ensemble-disagreement summaries, not calibrated confidence bounds for the true probability.",
    "Ensemble agreement outside the local data support does not establish reliability; in_data_range is an empirical heuristic.",
    "Predictions concern sustained opposed-flow microgravity burning; sample geometry, ignition method and campaign differences require scientific review before combining real datasets.",
    "The thermally thin physics assumption does not hold for the thick PMMA example.",
]


class ModelCardService:
    def __init__(self, predictor, metrics: dict | None = None):
        self.predictor = predictor
        self.metrics = metrics

    def _training_summary(self) -> dict:
        data = self.predictor.training_data
        required = {"material", "experiment_id", "source", "ignited", "extinguished", *RANGE_COLUMNS}
        missing = sorted(required - set(data.columns))
        if data.empty or missing:
            return {
                "n_rows": len(data),
                "n_experiments": int(data.experiment_id.nunique()) if "experiment_id" in data else None,
                "source_counts": {}, "by_material": [],
                "quality_limitations": [
                    "Standard training observations were not retained or have incomplete columns; no empirical training-support summary is available.",
                    f"Missing standard columns: {', '.join(missing) if missing else 'none (empty table)' }.",
                ],
            }
        per_material = []
        for material, rows in data.groupby("material", sort=True):
            sustained = rows.ignited & ~rows.extinguished
            per_material.append({
                "material": str(material), "n_rows": len(rows),
                "n_burning": int(sustained.sum()), "n_nonburning": int((~sustained).sum()),
                "source_counts": {str(name): int(count) for name, count in rows.source.value_counts().sort_index().items()},
                "input_ranges": {
                    name: {"min": float(rows[name].min()), "max": float(rows[name].max())}
                    for name in RANGE_COLUMNS
                },
            })
        return {
            "n_rows": len(data), "n_experiments": int(data.experiment_id.nunique()),
            "source_counts": {str(name): int(count) for name, count in data.source.value_counts().sort_index().items()},
            "by_material": per_material,
            "quality_limitations": [
                "Range and schema validation cannot detect every unit mistake or verify the physical quality of an observation.",
                "Source labels record supplied provenance; they do not independently certify NASA origin or licensing.",
                "Experiment identifiers must represent genuinely correlated batches and be unique across campaigns.",
            ],
        }

    def _training_evidence(self) -> dict:
        """Keep the training-time snapshot even if the live registry changes."""
        snapshot = getattr(self.predictor, "training_evidence", None)
        if not isinstance(snapshot, dict):
            provenance = getattr(self.predictor, "provenance", None)
            config = provenance.get("training_config", {}) if isinstance(provenance, dict) else {}
            snapshot = config.get("training_evidence") if isinstance(config, dict) else None
        if isinstance(snapshot, dict):
            return deepcopy(snapshot)
        try:
            from data.nasa import training_evidence_summary
        except ModuleNotFoundError as exc:
            if exc.name != "data.nasa":
                raise
            # Keeps legacy artifacts inspectable during an incremental deployment
            # in which the optional evidence registry has not been installed yet.
            evidence = {"datasets": [], "status": "registry_unavailable"}
        else:
            evidence = deepcopy(training_evidence_summary(self.predictor.training_data))
        evidence["metadata_origin"] = "current_registry_derived"
        evidence["metadata_note"] = "Legacy artifact has no retained training-evidence snapshot; this summary uses the current registry and is not proof of what was recorded at training time."
        return evidence

    def _has_short_window_nasa_rows(self) -> bool:
        data = self.predictor.training_data
        if data.empty or "source" not in data:
            return False
        nasa_rows = data.loc[data["source"].eq("nasa")]
        for column in ("campaign", "experiment_id"):
            if column in nasa_rows and nasa_rows[column].astype(str).str.contains(
                r"182185|19890014267", case=False, regex=True,
            ).any():
                return True
        return False

    def get_card(self) -> dict:
        predictor = self.predictor
        raw_provenance = getattr(predictor, "provenance", None)
        provenance = deepcopy(raw_provenance) if isinstance(raw_provenance, dict) else None
        valid = bool(provenance) and metrics_match_model(predictor, self.metrics) and isinstance(self.metrics.get("validation"), dict)
        limitations = list(LIMITATIONS)
        if predictor.data_source in {"synthetic", "mixed"}:
            limitations.insert(0, "Training includes synthetic observations; synthetic-data accuracy is a method demonstration and does not validate performance on NASA measurements.")
        if predictor.data_source in {"nasa", "mixed"} and self._has_short_window_nasa_rows():
            limitations.extend([
                "NASA-CR-182185 labels describe a 5.18-second microgravity observation window; no observed extinction does not establish indefinite sustained burning.",
                "The NASA-CR-182185 training subset covers one thin cellulose sample and one nominal pressure; other materials and pressures lack real observations from that report.",
                "NASA source labels identify observations used in training and are not an overall NASA validation or approval of this model.",
            ])
        if provenance is None:
            limitations.append("This legacy or malformed artifact has no usable recorded fingerprints; its training identity cannot be verified through the new provenance checks.")
        if not valid:
            limitations.append("Matching grouped out-of-fold reliability diagnostics are unavailable; no validation report has been inferred from training predictions.")
        return {
            **model_meta(predictor),
            "validation_status": "available" if valid else "unavailable",
            "provenance": provenance, "training_data_summary": self._training_summary(),
            "training_evidence": self._training_evidence(),
            "features": list(getattr(predictor, "feature_names", [])),
            "ensemble_size": getattr(predictor, "n_members", None),
            "uncertainty_method": (
                "Group bootstrap for model disagreement; training-group OOB residual noise plus member log variance for nominal 90% conditional speed intervals."
                if hasattr(predictor, "regressors") and hasattr(predictor, "classifiers")
                else str(getattr(predictor, "uncertainty_method", "No uncertainty-method metadata was recorded for this predictor."))
            ),
            "validation": {
                "scores": deepcopy(self.metrics["models"]),
                "diagnostics": deepcopy(self.metrics["validation"]),
                "cv": deepcopy(self.metrics["cv"]),
            } if valid else None,
            "limitations": limitations,
        }
