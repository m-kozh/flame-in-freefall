"""Group-bootstrap LightGBM correction of an explicitly approximate baseline.

No experiment is divided when a bootstrap training set is drawn. The same
implementation can omit all physics features to provide the ML-only comparator.
The nominal 90% rate interval combines model disagreement with group-out-of-bag
residual noise. It is an empirical demonstration, not a coverage guarantee.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neighbors import NearestNeighbors

from app.config import MODEL_VERSION
from data.loader import validate_dataframe
from data.materials import MATERIALS, resolve_material
from models.base import PredictionBatch
from physics.flame_spread import evaluate_flame


@dataclass(frozen=True)
class EnsembleConstants:
    """Visible, modest hyperparameters chosen for the hackathon data size."""

    n_estimators: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 15
    reg_lambda: float = 1.0
    normal_90_quantile: float = 1.6448536269514722
    minimum_noise_log_std: float = 0.03
    bootstrap_attempt_limit: int = 200
    maximum_training_gravity_g: float = 0.01
    support_neighbor_multiplier: float = 1.0
    minimum_support_threshold: float = 0.08
    maximum_support_threshold: float = 0.22
    support_o2_scale_percent: float = 20.0
    support_pressure_scale_kpa: float = 50.0
    support_log_flow_scale: float = float(np.log1p(100.0))
    # Thickness is checked against its measured range as well as included with
    # modest weight in local distance, so normal sample variation does not mask
    # an oxygen/pressure/flow hole. These are illustrative support scales.
    support_thickness_relative_scale: float = 0.6
    unsupported_material_distance: float = 2.0


CONSTANTS = EnsembleConstants()
RAW_FEATURES = [
    "o2_percent", "pressure_kpa", "flow_cm_s", "thickness_mm",
    "density_kg_m3", "specific_heat_j_kgk", "vaporization_temp_k",
]
PHYSICS_FEATURES = [
    "log_physics_spread_rate", "physics_o2_margin_percent", "physics_regime_code",
]
REGIME_CODES = {
    "burning": 0.0, "quench_extinction": 1.0,
    "blowoff_extinction": 2.0, "no_ignition": 3.0,
}


def prepare_inputs(X: pd.DataFrame) -> pd.DataFrame:
    """Resolve aliases and add material properties without requiring CSV labels."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    frame = X.copy().reset_index(drop=True)
    if "o2" in frame:
        if "o2_percent" in frame and not np.allclose(
            frame["o2"].astype(float), frame["o2_percent"].astype(float),
            equal_nan=False,
        ):
            raise ValueError("o2 and o2_percent must agree when both are provided.")
        if "o2_percent" not in frame:
            frame["o2_percent"] = frame["o2"]
    required = {"material", "o2_percent", "pressure_kpa", "flow_cm_s"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction inputs are missing columns: {missing}.")
    frame["material"] = frame["material"].map(resolve_material)
    materials = frame["material"].map(MATERIALS.get)
    defaults = materials.map(lambda material: material.thickness_mm)
    if "thickness_mm" not in frame:
        frame["thickness_mm"] = defaults
    else:
        frame["thickness_mm"] = frame["thickness_mm"].fillna(defaults)
    for column in ["o2_percent", "pressure_kpa", "flow_cm_s", "thickness_mm"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"{column} must contain finite numbers only.")
    for column in [
        "density_kg_m3", "specific_heat_j_kgk", "vaporization_temp_k",
        "limiting_o2_percent",
    ]:
        frame[column] = materials.map(lambda material: getattr(material, column)).astype(float)
    return frame


def _physics_prepared(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return evaluate_flame(**{
        column: frame[column].to_numpy(dtype=float)
        for column in [
            "o2_percent", "pressure_kpa", "flow_cm_s", "thickness_mm",
            "density_kg_m3", "specific_heat_j_kgk", "vaporization_temp_k",
            "limiting_o2_percent",
        ]
    })


def physics_frame(X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Public vectorised physics helper used by the physics-only evaluation."""
    return _physics_prepared(prepare_inputs(X))


def _features_prepared(frame: pd.DataFrame, use_physics: bool, physics=None) -> pd.DataFrame:
    features = frame.loc[:, RAW_FEATURES].copy()
    if use_physics:
        if physics is None:
            physics = _physics_prepared(frame)
        features["log_physics_spread_rate"] = np.log(physics["potential_spread_rate_mm_s"])
        features["physics_o2_margin_percent"] = physics["o2_margin_percent"]
        features["physics_regime_code"] = pd.Series(physics["regime"]).map(REGIME_CODES).to_numpy()
    return features.astype(float)


def feature_frame(X: pd.DataFrame, use_physics: bool = True) -> pd.DataFrame:
    """Raw seven features, optionally followed by the three named physics features."""
    return _features_prepared(prepare_inputs(X), use_physics)


def _support_coordinates(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        frame["o2_percent"].to_numpy(), frame["pressure_kpa"].to_numpy(),
        np.log1p(frame["flow_cm_s"].to_numpy()), frame["thickness_mm"].to_numpy(),
    ])


class ResidualEnsemble:
    """A physics residual model, or a physics-free log-rate comparator.

    ``uncertainty`` is the standard deviation of member extinction probabilities.
    It captures model disagreement only. Rate intervals additionally include an
    empirical observation-error scale derived from whole held-out groups within
    this model's own training data.
    """

    def __init__(self, n_members: int = 20, seed: int = 42, use_physics: bool = True):
        if isinstance(n_members, bool) or not isinstance(n_members, int) or n_members < 1:
            raise ValueError("n_members must be a positive integer.")
        self.n_members = n_members
        self.seed = seed
        self.use_physics = use_physics
        self.model_version = MODEL_VERSION if use_physics else "ml-only-0.1.0"
        self.data_source = "synthetic"
        self.feature_names = RAW_FEATURES + (PHYSICS_FEATURES if use_physics else [])
        self.regressors: list[LGBMRegressor] = []
        self.classifiers: list[LGBMClassifier] = []
        self.training_data = pd.DataFrame()
        self.noise_log_std = CONSTANTS.minimum_noise_log_std
        self.support_models: dict[str, dict] = {}
        self.bootstrap_groups: list[list[str]] = []
        self.provenance = None
        self.model_id = None
        self.training_evidence = None

    def _parameters(self, member_seed: int) -> dict:
        c = CONSTANTS
        return {
            "n_estimators": c.n_estimators, "learning_rate": c.learning_rate,
            "num_leaves": c.num_leaves, "min_child_samples": c.min_child_samples,
            "reg_lambda": c.reg_lambda, "random_state": member_seed,
            "n_jobs": 1, "verbosity": -1, "deterministic": True,
            "force_col_wise": True,
        }

    def feature_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        return feature_frame(X, use_physics=self.use_physics)

    def fit(self, df: pd.DataFrame) -> "ResidualEnsemble":
        """Fit on this dataset only; no outer validation rows are consulted."""
        data = validate_dataframe(df)
        if (data["gravity_g"] > CONSTANTS.maximum_training_gravity_g).any():
            raise ValueError(
                "This predictor is restricted to microgravity: gravity_g must be <= 0.01. "
                "Filter or separately model ground-gravity experiments before training."
            )
        sustained = (data["ignited"] & ~data["extinguished"]).to_numpy()
        extinct = (~sustained).astype(int)
        if not sustained.any() or np.unique(extinct).size < 2:
            raise ValueError("Training requires both sustained-burning and non-burning observations.")
        groups = data["experiment_id"].to_numpy()
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError("At least two experiment_id groups are required for group-bootstrap uncertainty.")
        self.training_data = data.copy()
        sources = data["source"].unique()
        self.data_source = str(sources[0]) if len(sources) == 1 else "mixed"
        self.regressors = []
        self.classifiers = []
        self.bootstrap_groups = []
        frame = prepare_inputs(data)
        training_physics = _physics_prepared(frame) if self.use_physics else None
        features = _features_prepared(frame, self.use_physics, physics=training_physics)
        self.feature_names = features.columns.tolist()
        baseline_log = (
            np.log(training_physics["potential_spread_rate_mm_s"])
            if self.use_physics else np.zeros(len(data))
        )
        observed_log = np.full(len(data), np.nan)
        observed_log[sustained] = np.log(data.loc[sustained, "spread_rate_mm_s"].to_numpy())
        target = observed_log - baseline_log
        rng = np.random.default_rng(self.seed)
        indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
        oob_log_predictions = np.full((self.n_members, len(data)), np.nan)
        for member in range(self.n_members):
            for _ in range(CONSTANTS.bootstrap_attempt_limit):
                sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                indices = np.concatenate([indices_by_group[group] for group in sampled])
                if sustained[indices].any() and np.unique(extinct[indices]).size == 2:
                    break
            else:
                raise ValueError("Cannot form a group-bootstrap sample containing both outcome classes.")
            self.bootstrap_groups.append(sampled.tolist())
            parameters = self._parameters(self.seed + member)
            regressor = LGBMRegressor(objective="regression", **parameters)
            classifier = LGBMClassifier(objective="binary", **parameters)
            reg_indices = indices[sustained[indices]]
            regressor.fit(features.iloc[reg_indices], target[reg_indices])
            classifier.fit(features.iloc[indices], extinct[indices])
            self.regressors.append(regressor)
            self.classifiers.append(classifier)
            # An entire experiment must be absent before its error can estimate noise.
            oob = ~np.isin(groups, np.unique(sampled)) & sustained
            if oob.any():
                oob_log_predictions[member, oob] = (
                    baseline_log[oob]
                    + regressor.booster_.predict(features.loc[oob].to_numpy(), num_threads=1)
                )
        self._estimate_noise(features, target, baseline_log, observed_log, sustained, groups, oob_log_predictions)
        self._fit_support(frame)
        from data.nasa import training_evidence_summary
        from models.provenance import build_provenance

        self.training_evidence = training_evidence_summary(data)
        self.provenance = build_provenance(self)
        self.model_id = self.provenance["model_id"]
        return self

    def _estimate_noise(
        self, features, target, baseline_log, observed_log, sustained, groups, predictions,
    ) -> None:
        counts = np.isfinite(predictions).sum(axis=0)
        usable = sustained & (counts > 0)
        errors = []
        member_variances = []
        if usable.any():
            selected = predictions[:, usable]
            means = np.nansum(selected, axis=0) / counts[usable]
            variances = np.nansum((selected - means) ** 2, axis=0) / counts[usable]
            errors.extend((observed_log[usable] - means).tolist())
            member_variances.extend(variances.tolist())
        # Tiny ensembles may produce no OOB predictions. Leave whole groups out
        # explicitly rather than substituting overoptimistic training residuals.
        if not errors:
            for group in np.unique(groups):
                held_out = (groups == group) & sustained
                train = (groups != group) & sustained
                if not held_out.any() or not train.any():
                    continue
                temporary = LGBMRegressor(objective="regression", **self._parameters(self.seed))
                temporary.fit(features.loc[train], target[train])
                estimate = baseline_log[held_out] + temporary.booster_.predict(
                    features.loc[held_out].to_numpy(), num_threads=1,
                )
                errors.extend((observed_log[held_out] - estimate).tolist())
                member_variances.extend([0.0] * int(held_out.sum()))
        if not errors:
            raise ValueError("Need sustained-burning observations in at least two experiment groups to estimate uncertainty.")
        # OOB errors already include some member disagreement. Subtract its
        # estimated variance before adding member variance back at prediction.
        variance = float(np.mean(np.square(errors)) - np.mean(member_variances))
        self.noise_log_std = float(np.sqrt(max(variance, CONSTANTS.minimum_noise_log_std**2)))
        self.noise_estimation = "group_out_of_bag"
        self.oob_prediction_rows = int(usable.sum())

    def _fit_support(self, frame: pd.DataFrame) -> None:
        self.support_models = {}
        coordinates = _support_coordinates(frame)
        for material in frame["material"].unique():
            material_rows = frame["material"].eq(material).to_numpy()
            values = coordinates[material_rows]
            scale = np.array([
                CONSTANTS.support_o2_scale_percent,
                CONSTANTS.support_pressure_scale_kpa,
                CONSTANTS.support_log_flow_scale,
                MATERIALS[material].thickness_mm * CONSTANTS.support_thickness_relative_scale,
            ])
            normalised = values / scale
            neighbors = NearestNeighbors(n_jobs=1).fit(normalised)
            if len(values) >= 2:
                leave_one_out = neighbors.kneighbors(normalised, n_neighbors=2)[0][:, 1]
                threshold = np.clip(
                    np.quantile(leave_one_out, 0.95) * CONSTANTS.support_neighbor_multiplier,
                    CONSTANTS.minimum_support_threshold, CONSTANTS.maximum_support_threshold,
                )
            else:
                threshold = CONSTANTS.minimum_support_threshold
            self.support_models[material] = {
                "scale": scale, "minimum": values.min(axis=0), "maximum": values.max(axis=0),
                "neighbors": neighbors, "threshold": float(threshold),
            }

    def _check_fitted(self) -> None:
        if not self.regressors or not self.classifiers:
            raise ValueError("Model has not been fitted. Train or load an artifact first.")

    def _support_prepared(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        coordinates = _support_coordinates(frame)
        distance = np.full(len(frame), CONSTANTS.unsupported_material_distance, dtype=float)
        inside = np.zeros(len(frame), dtype=bool)
        for material in frame["material"].unique():
            support = self.support_models.get(material)
            if support is None:
                continue
            rows = frame["material"].eq(material).to_numpy()
            values = coordinates[rows]
            nearest = support["neighbors"].kneighbors(
                values / support["scale"], n_neighbors=1, return_distance=True,
            )[0][:, 0]
            distance[rows] = nearest
            within_ranges = (
                (values >= support["minimum"] - 1e-9)
                & (values <= support["maximum"] + 1e-9)
            ).all(axis=1)
            inside[rows] = within_ranges & (nearest <= support["threshold"])
        return distance, inside

    def support_distance(self, X: pd.DataFrame) -> np.ndarray:
        """Distance to the nearest same-material training observation, normalised."""
        self._check_fitted()
        return self._support_prepared(prepare_inputs(X))[0]

    def _member_predictions_prepared(self, frame: pd.DataFrame):
        """Shared vectorised member evaluation, with physics calculated once."""
        physics = _physics_prepared(frame)
        features = _features_prepared(frame, self.use_physics, physics=physics).to_numpy()
        baseline_log = (
            np.log(physics["potential_spread_rate_mm_s"])
            if self.use_physics else np.zeros(len(frame))
        )
        member_log_rates = np.stack([
            baseline_log + model.booster_.predict(features, num_threads=1)
            for model in self.regressors
        ])
        member_probabilities = np.stack([
            model.booster_.predict(features, num_threads=1)
            for model in self.classifiers
        ])
        member_log_rates = np.clip(member_log_rates, -30.0, 30.0)
        return member_log_rates, member_probabilities, physics

    def predict_members(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Optional ensemble capability for information-driven experimental design.

        Arrays have shape (members, observations). These bootstrap members are
        empirical alternatives, not exact Bayesian posterior samples.
        """
        self._check_fitted()
        frame = prepare_inputs(X)
        if frame.empty:
            raise ValueError("Prediction requires at least one input row.")
        log_rates, probability, _ = self._member_predictions_prepared(frame)
        return {
            "extinction_probability": probability, "log_spread_rate": log_rates,
            "in_data_range": self._support_prepared(frame)[1],
        }

    def predict(self, X: pd.DataFrame) -> PredictionBatch:
        self._check_fitted()
        frame = prepare_inputs(X)
        if frame.empty:
            raise ValueError("Prediction requires at least one input row.")
        member_log_rates, member_probabilities, physics = self._member_predictions_prepared(frame)
        mean = np.exp(member_log_rates).mean(axis=0)
        center = member_log_rates.mean(axis=0)
        log_std = np.sqrt(member_log_rates.var(axis=0) + self.noise_log_std**2)
        lower = np.minimum(np.exp(center - CONSTANTS.normal_90_quantile * log_std), mean)
        upper = np.maximum(np.exp(center + CONSTANTS.normal_90_quantile * log_std), mean)
        probability = np.clip(member_probabilities.mean(axis=0), 0.0, 1.0)
        extinction_lower = np.minimum(np.quantile(member_probabilities, 0.05, axis=0), probability)
        extinction_upper = np.maximum(np.quantile(member_probabilities, 0.95, axis=0), probability)
        _, in_range = self._support_prepared(frame)
        return PredictionBatch(
            mean=mean, lower=lower, upper=upper,
            extinction_probability=probability,
            physics_baseline=physics["spread_rate_mm_s"],
            extinction_lower=extinction_lower, extinction_upper=extinction_upper,
            uncertainty=member_probabilities.std(axis=0),
            physics_regime=physics["regime"], in_data_range=in_range, log_std=log_std,
        )

    def save(self, path: str | Path) -> None:
        """Persist the model, source data, support index, and uncertainty metadata."""
        self._check_fitted()
        from models.provenance import validate_runtime_provenance

        validate_runtime_provenance(self)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> "ResidualEnsemble":
        """Load a locally trained trusted artifact (joblib is Python serialisation)."""
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise ValueError("Artifact does not contain a ResidualEnsemble predictor.")
        model._check_fitted()
        from models.provenance import validate_runtime_provenance

        validate_runtime_provenance(model)
        return model
