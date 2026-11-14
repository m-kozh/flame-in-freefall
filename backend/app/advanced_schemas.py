"""Typed additive research APIs; the original frontend contract stays intact."""

from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, StrictBool, model_validator

from app.schemas import (
    Flow, InputModel, MaterialThicknessInput, ModelResponse, Nonnegative, Number,
    Oxygen, PredictRequest, PredictResponse, Pressure, Probability, ProbabilitySD,
    Resolution, ResponseModel, SuggestionCount, Thickness, _reject_boolean,
)

Count = Annotated[int, BeforeValidator(_reject_boolean), Field(ge=0)]
SampleCount = Annotated[int, BeforeValidator(_reject_boolean), Field(ge=32, le=512)]
OptimizeResolution = Annotated[Resolution, Field(le=16)]
ExperimentResolution = Annotated[Resolution, Field(ge=10, le=60)]
ExperimentStrategy = Literal["information_gain", "boundary", "disagreement"]


class AdvancedMeta(ModelResponse):
    model_id: str | None


class Tolerances(InputModel):
    o2_percent: Annotated[Number, Field(ge=0, le=10)] = 1.0
    pressure_kpa: Annotated[Number, Field(ge=0, le=20)] = 2.0
    flow_cm_s: Annotated[Number, Field(ge=0, le=20)] = 1.0


class BatchPredictRequest(InputModel):
    points: Annotated[list[PredictRequest], Field(min_length=1, max_length=256)]


class BatchPredictResponse(AdvancedMeta):
    count: Annotated[Count, Field(ge=1, le=256)]
    predictions: Annotated[list[PredictResponse], Field(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def check_count(self):
        if self.count != len(self.predictions):
            raise ValueError("count must equal the number of predictions.")
        return self


class AnalyzeRequest(InputModel):
    scenario: PredictRequest
    tolerances: Tolerances = Field(default_factory=Tolerances)
    samples: SampleCount = 128


class OptimizeBounds(InputModel):
    o2_range: tuple[Oxygen, Oxygen]
    pressure_range: tuple[Pressure, Pressure]
    flow_range: tuple[Flow, Flow]

    @model_validator(mode="after")
    def check_order(self):
        for name in ("o2_range", "pressure_range", "flow_range"):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} requires min <= max; equal limits fix a control.")
        return self


class OptimizeRequest(InputModel):
    scenario: PredictRequest
    bounds: OptimizeBounds
    target_extinction_probability: Probability = 0.9
    resolution: OptimizeResolution = 10
    k: SuggestionCount = 5
    supported_only: StrictBool = True
    tolerances: Tolerances = Field(default_factory=Tolerances)


class ExperimentDesignRequest(MaterialThicknessInput):
    pressure_kpa: Pressure = 101.3
    flow_range: tuple[Flow, Flow] = (0.1, 150.0)
    o2_range: tuple[Oxygen, Oxygen] = (8.0, 55.0)
    resolution: ExperimentResolution = 40
    k: SuggestionCount = 5
    strategy: ExperimentStrategy = "information_gain"
    exclude_tested: StrictBool = True

    @model_validator(mode="after")
    def check_order(self):
        if self.flow_range[0] >= self.flow_range[1]:
            raise ValueError("flow_range requires min < max (cm/s).")
        if self.o2_range[0] >= self.o2_range[1]:
            raise ValueError("o2_range requires min < max (percent).")
        return self


class Distribution(ResponseModel):
    min: Nonnegative
    p05: Nonnegative
    mean: Nonnegative
    p95: Nonnegative
    max: Nonnegative

    @model_validator(mode="after")
    def check_order(self):
        if not self.min <= self.p05 <= self.p95 <= self.max:
            raise ValueError("Distribution quantiles must be ordered within min/max.")
        if not self.min <= self.mean <= self.max:
            raise ValueError("Distribution mean must lie within min/max.")
        return self


class PhysicsDisagreement(ResponseModel):
    potential_baseline_mm_s: Nonnegative
    learned_to_potential_ratio: Nonnegative
    sustained_burning_disagreement: bool
    interpretation: str


class MaterialComparison(ResponseModel):
    material: str
    thermally_thin: bool
    prediction: PredictResponse


class PresetComparison(ResponseModel):
    preset_id: str
    illustrative: bool
    prediction: PredictResponse


class Sensitivity(ResponseModel):
    parameter: Literal["o2_percent", "pressure_kpa", "flow_cm_s"]
    lower_input: Number
    upper_input: Number
    lower_prediction: PredictResponse
    upper_prediction: PredictResponse
    extinction_probability_change: Number
    spread_rate_change_mm_s: Number


class Robustness(ResponseModel):
    seed: Count
    random_samples: Count
    corner_samples: Count
    sample_count: Count
    extinction_probability: Distribution
    spread_rate_mm_s: Distribution
    minimum_sampled_extinction_lower: Probability
    support_fraction: Probability
    decision_flip_fraction: Probability


class AnalyzeResponse(AdvancedMeta):
    current: PredictResponse
    physics_disagreement: PhysicsDisagreement
    material_comparison: list[MaterialComparison]
    preset_comparison: list[PresetComparison]
    sensitivity: list[Sensitivity]
    robustness: Robustness
    assessment: Literal["supported", "extrapolation"]
    limitations: list[str]


class ControlDeltas(ResponseModel):
    o2_percent: Number
    pressure_kpa: Number
    flow_cm_s: Number


class CandidateRobustness(ResponseModel):
    corner_count: Count
    minimum_sampled_extinction_lower: Probability
    support_fraction: Probability


class OptimizeCandidate(ResponseModel):
    prediction: PredictResponse
    change_cost: Nonnegative
    deltas: ControlDeltas
    robustness: CandidateRobustness


class OptimizeResponse(AdvancedMeta):
    status: Literal["found", "not_found"]
    target_achievable: bool
    target_extinction_probability: Probability
    evaluated_count: Count
    eligible_count: Count
    candidates: Annotated[list[OptimizeCandidate], Field(max_length=20)]
    best_available: OptimizeCandidate | None
    pareto_front: Annotated[list[OptimizeCandidate], Field(max_length=20)]
    pareto_front_total_count: Count
    pareto_front_truncated: bool
    limitations: list[str]


class AcquisitionScore(ResponseModel):
    acquisition: Nonnegative
    boundary_weight: Nonnegative
    sparse_weight: Nonnegative


class ExperimentPoint(ResponseModel):
    o2: Oxygen
    o2_percent: Oxygen
    pressure_kpa: Pressure
    flow_cm_s: Flow
    thickness_mm: Thickness
    extinction_probability: Probability
    uncertainty: ProbabilitySD
    predictive_entropy_bits: Annotated[Nonnegative, Field(le=1)]
    expected_member_entropy_bits: Annotated[Nonnegative, Field(le=1)]
    ensemble_information_gain_proxy_bits: Annotated[Nonnegative, Field(le=1)]
    in_data_range: bool
    nearest_test_distance: Nonnegative
    score: Probability
    score_components: AcquisitionScore
    reason: str


class ExperimentDesignResponse(AdvancedMeta):
    material: str
    pressure_kpa: Pressure
    strategy: ExperimentStrategy
    candidate_count: Count
    excluded_tested_count: Count
    eligible_count: Count
    minimum_separation: Nonnegative
    points: Annotated[list[ExperimentPoint], Field(max_length=20)]
    method_notes: list[str]


class ModelCardResponse(AdvancedMeta):
    validation_status: Literal["available", "unavailable"]
    provenance: dict[str, Any] | None
    training_data_summary: dict[str, Any]
    features: list[str]
    ensemble_size: Count | None
    uncertainty_method: str
    validation: dict[str, Any] | None
    limitations: list[str]
    training_evidence: dict[str, Any] | None = None
