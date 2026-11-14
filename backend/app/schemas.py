"""Stable HTTP schemas and unit-aware validation for both frontend API forms."""

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator,
)

from data.materials import get_material, resolve_material


DataSource = Literal["synthetic", "nasa", "mixed"]
Regime = Literal["burning", "quench_extinction", "blowoff_extinction", "no_ignition"]
ExplainTarget = Literal["spread_rate", "extinction"]


def _reject_boolean(value):
    # Python considers bool an int; JSON true must never mean 1 kPa or 1 cm/s.
    if isinstance(value, bool):
        raise ValueError("Boolean values are not accepted as numbers.")
    return value


Number = Annotated[float, BeforeValidator(_reject_boolean), Field(allow_inf_nan=False)]
Oxygen = Annotated[Number, Field(ge=5, le=100, description="Oxygen mole percent; 21 means 21%.")]
Pressure = Annotated[Number, Field(ge=10, le=150, description="Total pressure in kPa.")]
Flow = Annotated[Number, Field(ge=0, le=200, description="Opposed ambient flow speed in cm/s.")]
Thickness = Annotated[Number, Field(ge=0.01, le=10, description="Full sample thickness in mm.")]
Resolution = Annotated[int, BeforeValidator(_reject_boolean), Field(ge=5, le=100)]
SuggestionCount = Annotated[int, BeforeValidator(_reject_boolean), Field(ge=1, le=20)]
Probability = Annotated[Number, Field(ge=0, le=1)]
Nonnegative = Annotated[Number, Field(ge=0)]
ProbabilitySD = Annotated[Number, Field(ge=0, le=0.5)]


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


class MaterialInput(InputModel):
    material: str

    @field_validator("material")
    @classmethod
    def canonical_material(cls, value: str) -> str:
        return resolve_material(value)


class MaterialThicknessInput(MaterialInput):
    thickness_mm: Thickness | None = None

    @model_validator(mode="after")
    def fill_default_thickness(self):
        if self.thickness_mm is None:
            self.thickness_mm = get_material(self.material).thickness_mm
        return self


def _oxygen_alias_schema(schema: dict) -> None:
    """Expose both accepted keys to OpenAPI client generators as well as prose."""
    schema["properties"]["o2"] = {
        **schema["properties"]["o2_percent"],
        "title": "O2", "description": "Alias of o2_percent; both keys must agree if supplied together.",
    }
    schema["required"] = [name for name in schema.get("required", []) if name != "o2_percent"]
    schema["anyOf"] = [{"required": ["o2_percent"]}, {"required": ["o2"]}]


class PredictRequest(MaterialThicknessInput):
    model_config = ConfigDict(json_schema_extra=_oxygen_alias_schema)
    o2_percent: Oxygen
    pressure_kpa: Pressure
    flow_cm_s: Flow

    @model_validator(mode="before")
    @classmethod
    def resolve_oxygen_alias(cls, value):
        if isinstance(value, dict) and "o2" in value:
            value = dict(value)
            alias = value.pop("o2")
            if "o2_percent" in value:
                try:
                    equal = float(_reject_boolean(alias)) == float(_reject_boolean(value["o2_percent"]))
                except (TypeError, ValueError):
                    raise ValueError("o2 and o2_percent must be valid numbers and agree when both are supplied.") from None
                if not equal:
                    raise ValueError("o2 and o2_percent must agree when both are supplied.")
            else:
                value["o2_percent"] = alias
        return value


class ExplainRequest(PredictRequest):
    target: ExplainTarget = "spread_rate"


def _check_ranges(flow_min, flow_max, o2_min, o2_max, flow_scale):
    if flow_min >= flow_max:
        raise ValueError("Flow range requires min < max (cm/s).")
    if o2_min >= o2_max:
        raise ValueError("Oxygen range requires min < max (percent).")
    if flow_scale == "log" and flow_min <= 0:
        raise ValueError("Logarithmic flow spacing requires flow_min > 0 cm/s.")


class MapRequest(MaterialThicknessInput):
    pressure_kpa: Pressure = 101.3
    flow_range: tuple[Flow, Flow] = (0.5, 100.0)
    o2_range: tuple[Oxygen, Oxygen] = (12.0, 40.0)
    resolution: Resolution = 50
    flow_scale: Literal["log", "linear"] = "log"

    @model_validator(mode="after")
    def check_ranges(self):
        _check_ranges(*self.flow_range, *self.o2_range, self.flow_scale)
        return self


class MapQuery(MaterialThicknessInput):
    pressure_kpa: Pressure = 101.3
    flow_min: Flow = 0.5
    flow_max: Flow = 100.0
    o2_min: Oxygen = 12.0
    o2_max: Oxygen = 40.0
    resolution: Resolution = 50
    flow_scale: Literal["log", "linear"] = "log"

    @model_validator(mode="after")
    def check_ranges(self):
        _check_ranges(self.flow_min, self.flow_max, self.o2_min, self.o2_max, self.flow_scale)
        return self

    def to_request(self) -> MapRequest:
        return MapRequest(
            material=self.material, pressure_kpa=self.pressure_kpa,
            thickness_mm=self.thickness_mm,
            flow_range=(self.flow_min, self.flow_max),
            o2_range=(self.o2_min, self.o2_max),
            resolution=self.resolution, flow_scale=self.flow_scale,
        )


class SuggestQuery(MaterialInput):
    pressure_kpa: Pressure = 101.3
    k: SuggestionCount = 5


class ExplainQuery(InputModel):
    target: ExplainTarget = "spread_rate"


class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SourceResponse(ResponseModel):
    data_source: DataSource


class ModelResponse(SourceResponse):
    model_version: str


class HealthResponse(ModelResponse):
    status: Literal["ok", "model_not_loaded"]
    model_loaded: bool


class MaterialProperties(ResponseModel):
    id: str
    name: str
    aliases: list[str]
    density_kg_m3: Number
    specific_heat_j_kgk: Number
    thickness_mm: Number
    vaporization_temp_k: Number
    thermally_thin: bool
    limiting_o2_percent: Number | None = None
    notes: str


class MaterialsResponse(SourceResponse):
    materials: list[MaterialProperties]


class AtmospherePreset(ResponseModel):
    id: str
    name: str
    o2_percent: Number
    pressure_kpa: Number
    flow_cm_s: Number
    illustrative: bool
    note: str


class PresetsResponse(SourceResponse):
    presets: list[AtmospherePreset]


class SpreadInterval(ResponseModel):
    mean: Nonnegative
    lower: Nonnegative
    upper: Nonnegative


class PredictionInputs(ResponseModel):
    material: str
    o2_percent: Number
    pressure_kpa: Number
    flow_cm_s: Number
    thickness_mm: Number


class PredictResponse(ModelResponse):
    spread_rate_mm_s: Nonnegative
    spread_rate_interval: SpreadInterval
    spread_ci: tuple[Nonnegative, Nonnegative]
    extinction_probability: Probability
    extinction_prob: Probability
    extinction_ci: tuple[Probability, Probability]
    uncertainty: ProbabilitySD
    regime: Regime
    physics_baseline_mm_s: Nonnegative
    physics_regime: Regime
    confidence: Literal["high", "medium", "low"]
    in_data_range: bool
    inputs: PredictionInputs


class DetailedPredictResponse(PredictResponse):
    spread_rate_mm_s: SpreadInterval


class BoundaryPoint(ResponseModel):
    flow_cm_s: Number
    o2_percent: Number


class PresetOverlay(ResponseModel):
    id: str
    name: str
    o2_percent: Number
    pressure_kpa: Number
    flow_cm_s: Number
    matches_map_pressure: bool


class TestedPoint(ResponseModel):
    o2_percent: Number
    o2: Number
    flow_cm_s: Number
    pressure_kpa: Number
    burned: bool
    source: Literal["synthetic", "nasa"]
    experiment_id: str
    campaign: str | None = None
    thickness_mm: Number | None = None


class MapResponse(ModelResponse):
    material: str
    pressure_kpa: Number
    thickness_mm: Number
    flow_scale: Literal["log", "linear"]
    flow_values: list[Number]
    o2_values: list[Number]
    spread_rate: list[list[Nonnegative]]
    extinction_probability: list[list[Probability]]
    uncertainty: list[list[ProbabilitySD]]
    boundary: list[BoundaryPoint]
    physics_boundary: list[BoundaryPoint]
    presets_overlay: list[PresetOverlay]
    tested_points: list[TestedPoint]
    x_axis: Literal["o2"]
    y_axis: Literal["flow_cm_s"]
    x: list[Number]
    y: list[Number]
    z: list[list[Probability]]


class SuggestedPoint(ResponseModel):
    o2_percent: Number
    o2: Number
    pressure_kpa: Number
    flow_cm_s: Number
    uncertainty: ProbabilitySD
    extinction_probability: Probability
    score: Probability
    reason: str


class SuggestResponse(ModelResponse):
    points: list[SuggestedPoint]
    material: str
    pressure_kpa: Number


class FeatureImportance(ResponseModel):
    feature: str
    importance: Nonnegative


class FeatureAlias(ResponseModel):
    name: str
    importance: Nonnegative


class LocalContribution(ResponseModel):
    feature: str
    value: Number
    contribution: Number


class ExplainResponse(ModelResponse):
    target: ExplainTarget
    global_importance: list[FeatureImportance]
    features: list[FeatureAlias]
    local_contributions: list[LocalContribution]
    base_value: Number | None
    prediction: Number | None
    contribution_units: str


class ErrorResponse(SourceResponse):
    # Validation errors retain FastAPI's detail list, including locations/input;
    # operational errors use the same conventional detail string.
    detail: str | list[dict[str, Any]]


class ModelMetrics(ResponseModel):
    log_spread_mae: Nonnegative
    log_spread_rmse: Nonnegative
    extinction_roc_auc: Probability | None
    extinction_brier: Probability
    interval_coverage_90: Probability | None


class FoldAudit(ResponseModel):
    fold: int
    train_experiment_ids: list[str]
    test_experiment_ids: list[str]
    train_rows: int
    test_rows: int


class CrossValidationAudit(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    method: Literal["GroupKFold"]
    group_column: Literal["experiment_id"]
    n_splits: int
    folds: list[FoldAudit] = Field(default_factory=list)


class MetricsResponse(ModelResponse):
    # Preserve additional training metadata verbatim instead of filtering notes,
    # seeds, or future validation details from artifacts/metrics.json.
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    trained_at: str
    n_rows: int
    n_experiments: int
    cv: CrossValidationAudit
    models: dict[str, ModelMetrics]
