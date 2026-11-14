"""Tree SHAP explanations of the complete bootstrap ensemble.

The additive targets are the mean member log-rate correction and mean member
extinction log-odds. They differ from transforming the final arithmetic mean
rate/probability, because exp and sigmoid are nonlinear. Keeping this distinction
explicit makes waterfall charts mathematically meaningful.
"""

from copy import deepcopy
import warnings

import numpy as np
import pandas as pd


GLOBAL_SAMPLE_LIMIT = 128
GLOBAL_SAMPLE_SEED = 42
TARGETS = {"spread_rate", "extinction"}


def _single_output_values(values, n_rows: int, n_features: int) -> np.ndarray:
    """Normalise old/list and current/multi-output SHAP return conventions."""
    if isinstance(values, (list, tuple)):
        if not values:
            raise ValueError("SHAP returned no output arrays.")
        # Binary classifiers sometimes return one array per class. Class 1 is
        # extinction in our training labels; regression has a single output.
        values = values[-1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        if values.shape[:2] == (n_rows, n_features):
            values = values[..., -1]
        elif values.shape[1:] == (n_rows, n_features):
            values = values[-1]
    if values.shape != (n_rows, n_features):
        raise ValueError(f"Unsupported SHAP output shape {values.shape}; expected {(n_rows, n_features)}.")
    return values


class ExplainService:
    def __init__(self, predictor):
        self.predictor = predictor
        self._explainer_cache: dict[str, list] = {}
        self._global_cache: dict[str, list[dict]] = {}

    def _members(self, target: str) -> list:
        attribute = "regressors" if target == "spread_rate" else "classifiers"
        members = getattr(self.predictor, attribute, None)
        if not members or not all(hasattr(model, "booster_") for model in members):
            raise NotImplementedError(
                "Tree SHAP requires a fitted LightGBM ensemble; a future PINN must provide its own explanation adapter."
            )
        if not hasattr(self.predictor, "feature_frame"):
            raise NotImplementedError("This predictor does not provide a tree feature adapter.")
        return members

    def _explainers(self, target: str) -> list:
        if target not in self._explainer_cache:
            # Lazy import keeps prediction, maps, and health independent of SHAP
            # initialisation; only explanation requests need this dependency.
            import shap

            self._explainer_cache[target] = [
                shap.TreeExplainer(
                    model.booster_, model_output="raw",
                    feature_perturbation="tree_path_dependent",
                )
                for model in self._members(target)
            ]
        return self._explainer_cache[target]

    def _ensemble_shap(self, features: pd.DataFrame, target: str) -> tuple[np.ndarray, float]:
        values = []
        base_values = []
        for explainer in self._explainers(target):
            with warnings.catch_warnings():
                # This version-compatibility notice repeats for every member;
                # both advertised shapes are handled explicitly below.
                warnings.filterwarnings(
                    "ignore",
                    message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
                    category=UserWarning,
                    module=r"shap\.explainers\._tree",
                )
                member_values = explainer.shap_values(features, check_additivity=True)
            values.append(_single_output_values(member_values, len(features), features.shape[1]))
            base_values.append(float(np.asarray(explainer.expected_value).reshape(-1)[-1]))
        return np.mean(values, axis=0), float(np.mean(base_values))

    def _global(self, target: str) -> list[dict]:
        if target not in self._global_cache:
            training = getattr(self.predictor, "training_data", None)
            if training is None or training.empty:
                raise NotImplementedError("Global SHAP requires a representative training-data sample.")
            sample = training.sample(
                n=min(GLOBAL_SAMPLE_LIMIT, len(training)), random_state=GLOBAL_SAMPLE_SEED,
            ).reset_index(drop=True)
            features = self.predictor.feature_frame(sample)
            ensemble_values, _ = self._ensemble_shap(features, target)
            importance = np.abs(ensemble_values).mean(axis=0)
            total = float(importance.sum())
            if total > 0:
                importance = importance / total
            self._global_cache[target] = sorted([
                {"feature": feature, "importance": float(value)}
                for feature, value in zip(features.columns, importance, strict=True)
            ], key=lambda item: item["importance"], reverse=True)
        return deepcopy(self._global_cache[target])

    def explain(self, inputs: dict | None = None, target: str = "spread_rate") -> dict:
        """Return cached global importance and optional per-input contributions."""
        if target not in TARGETS:
            raise ValueError("target must be 'spread_rate' or 'extinction'.")
        members = self._members(target)
        global_importance = self._global(target)
        local = []
        base_value = None
        prediction = None
        if inputs is not None:
            features = self.predictor.feature_frame(pd.DataFrame([inputs]))
            shap_values, base_value = self._ensemble_shap(features, target)
            prediction = float(np.mean([
                model.booster_.predict(features.to_numpy(), raw_score=True, num_threads=1)[0]
                for model in members
            ]))
            if not np.isclose(base_value + shap_values[0].sum(), prediction, atol=1e-7, rtol=1e-6):
                raise ValueError("SHAP contributions do not add to the declared ensemble raw-output target.")
            local = sorted([
                {
                    "feature": name,
                    "value": float(features.iloc[0][name]),
                    "contribution": float(contribution),
                }
                for name, contribution in zip(features.columns, shap_values[0], strict=True)
            ], key=lambda item: abs(item["contribution"]), reverse=True)
        if target == "extinction":
            units = "log-odds of extinction"
        elif getattr(self.predictor, "use_physics", True):
            units = "ln(spread rate / physics baseline)"
        else:
            units = "ln(spread rate)"
        return {
            "target": target,
            "global_importance": global_importance,
            "features": [
                {"name": item["feature"], "importance": item["importance"]}
                for item in global_importance
            ],
            "local_contributions": local,
            "base_value": base_value,
            "prediction": prediction,
            "contribution_units": units,
            "model_version": self.predictor.model_version,
            "data_source": self.predictor.data_source,
        }
