# Flame in Freefall — decision and research APIs (v0.2)

This is an additive extension. Every endpoint and field in [API.md](API.md) remains
supported with the same type. New endpoints use the same units, validation errors,
CORS policy and `data_source` badges. All requests reject unknown fields and
nonfinite/boolean numeric inputs. A missing model returns 503.

These APIs produce **model-based research insights**, not a safety certification.
Bootstrap members are not exact Bayesian posterior samples, and synthetic training
accuracy does not establish accuracy on spacecraft experiments.

## Shared types

`PredictReq`, `PredictRes`, and `DataSource` are defined in API.md. All scenario
inputs also accept `o2` and resolve material aliases/default thickness as before.

```ts
type Meta = { model_version: string; data_source: DataSource; model_id: string | null };
type Tolerances = { o2_percent?: number; pressure_kpa?: number; flow_cm_s?: number };
// defaults: ±1 percentage point O2, ±2 kPa, ±1 cm/s
// allowed nonnegative maxima: 10 percentage points, 20 kPa, 20 cm/s
type Distribution = { min: number; p05: number; mean: number; p95: number; max: number };
```

## POST /predict/batch

One vectorised model call for 1–256 scenarios. Predictions preserve input order and
are identical to the individual `/predict` schema, including conditional speed,
uncertainty and supported-region checks.

```json
{"points":[{"material":"cotton","o2":21,"pressure_kpa":101.3,"flow_cm_s":10},{"material":"nomex","o2":34,"pressure_kpa":56.5,"flow_cm_s":10}]}
```

Response: `Meta & { count: number; predictions: PredictRes[] }`.

## POST /analyze

Generate a reproducible scenario dossier with one batched prediction.

```json
{"scenario":{"material":"cotton","o2":21,"pressure_kpa":101.3,"flow_cm_s":10},"tolerances":{"o2_percent":1,"pressure_kpa":2,"flow_cm_s":1},"samples":128}
```

`samples` is the number of uniform random perturbations (32–512, default128), with
fixed seed42; eight tolerance-box corners are also evaluated. Tolerance boxes are
intersected with the valid API domain. A zero tolerance fixes that control.

Response fields:

- `current: PredictRes` — exactly the shared prediction shape.
- `physics_disagreement: {potential_baseline_mm_s, learned_to_potential_ratio,
  sustained_burning_disagreement, interpretation}` — ratios use the positive
  potential baseline, avoiding division by the extinction-gated zero rate.
- `material_comparison: {material, thermally_thin, prediction: PredictRes}[]` — all
  five materials at identical atmosphere/flow and each material's library thickness;
  this is not a claim that geometrically different samples are interchangeable.
- `preset_comparison: {preset_id, illustrative, prediction: PredictRes}[]` — current
  material/thickness at all four atmosphere presets, including their assumed flow.
- `sensitivity: {parameter, lower_input, upper_input, lower_prediction: PredictRes,
  upper_prediction: PredictRes, extinction_probability_change,
  spread_rate_change_mm_s}[]` — three finite differences at the tolerance limits;
  this is local sensitivity, not causal inference or a global monotonicity claim.
- `robustness: {seed, random_samples, corner_samples, sample_count,
  extinction_probability: Distribution, spread_rate_mm_s: Distribution,
  minimum_sampled_extinction_lower, support_fraction, decision_flip_fraction}`.
  Quantiles describe the **distribution of predictions across input conditions**,
  not an additional calibrated confidence interval or an exhaustive worst case.
- `assessment: "supported" | "extrapolation"`, `limitations: string[]`, plus Meta.

## POST /optimize

Search the whole allowed O2–pressure–flow grid, preserving both branches of the
nonmonotonic U-shaped boundary. This proposes settings to investigate experimentally.

```json
{"scenario":{"material":"cotton","o2":21,"pressure_kpa":101.3,"flow_cm_s":10},"bounds":{"o2_range":[12,35],"pressure_range":[50,115],"flow_range":[0.1,100]},"target_extinction_probability":0.9,"resolution":10,"k":5,"supported_only":true,"tolerances":{"o2_percent":1,"pressure_kpa":2,"flow_cm_s":1}}
```

`bounds` is required; ranges use the ordinary physical API limits and allow min=max
to fix a variable. `resolution`=5–16 (default10) per free axis, at most4096 central
grid points; `k`=1–20 (default5). Flow uses log1p spacing to include zero when allowed.
Current conditions are added only if inside the requested bounds.

Each candidate has `{prediction: PredictRes, change_cost,
deltas: {o2_percent, pressure_kpa, flow_cm_s},
robustness: {corner_count, minimum_sampled_extinction_lower, support_fraction}}`.
Change cost is Euclidean distance from the starting controls, normalised by each
requested range; fixed axes have no cost contribution. Lower cost is preferable.

Eligibility requires supported centres when `supported_only=true`; the target uses
the minimum of centre and eight clipped-neighbour corner **member lower quantiles**.
All corners must be supported when `supported_only=true`. These are empirical model
checks at sampled points, not a certified probability lower bound or safety guarantee.

Response: Meta plus `status: "found" | "not_found"`, `target_achievable: boolean`,
`target_extinction_probability`, `evaluated_count`, `eligible_count`,
`candidates` (at most k feasible candidates, increasing change_cost),
`best_available` (best eligible candidate even if target fails, or null),
`pareto_front` (at most20 nondominated cost/minimum-lower-probability candidates),
`pareto_front_total_count`, `pareto_front_truncated`, and `limitations: string[]`.
`not_found` only describes this bounded sampled search, not mathematical infeasibility.

## POST /experiments/design

Design a diverse next experimental batch using full-ensemble uncertainty.

```json
{"material":"cotton","pressure_kpa":56.5,"flow_range":[0.1,150],"o2_range":[8,55],"resolution":40,"k":5,"strategy":"information_gain","exclude_tested":true}
```

Ranges follow `/map`; flow is log1p spaced and can include0. `resolution`=10–60
(default40), `k`=1–20. `strategy` is `information_gain` (default), `boundary`, or
`disagreement`. Optional `thickness_mm` defaults to the library value.

For member extinction probabilities p_m:

```text
predictive_entropy_bits = H(mean(p_m))
expected_member_entropy_bits = mean(H(p_m))
ensemble_information_gain_proxy_bits = H(mean(p_m)) - mean(H(p_m))
```

The last value is a BALD-inspired **proxy**, not measured future information gain.
It separates between-member epistemic disagreement from each member's predicted
class ambiguity; it does not prove an aleatoric/epistemic decomposition of reality.

Response: Meta plus `material`, `pressure_kpa`, `strategy`, `candidate_count`,
`excluded_tested_count`, `eligible_count`, `minimum_separation`, `points`,
`method_notes: string[]`. Each point contains:

```ts
type ExperimentPoint = {
  o2: number; o2_percent: number; pressure_kpa: number; flow_cm_s: number;
  thickness_mm: number; extinction_probability: number; uncertainty: number;
  predictive_entropy_bits: number; expected_member_entropy_bits: number;
  ensemble_information_gain_proxy_bits: number; in_data_range: boolean;
  nearest_test_distance: number; score: number;
  score_components: {acquisition: number; boundary_weight: number; sparse_weight: number};
  reason: string;
};
```

Scores are normalised within eligible candidates and sorted descending. Greedy
selection enforces a normalised O2/log1p-flow minimum spacing; the service may return
fewer than k points if the eligibility/diversity constraints cannot be met. Measured
points are excluded when the same-material, four-dimensional support distance is
at most0.025 (scales:20 percentage points O2,50kPa,ln101 for log1p-flow, and
0.6×library thickness). Diversity distance is0.10 in normalised request O2/log1p-flow
coordinates. When all eligible scores are at most1e-12, return an empty point list
and an explicit note instead of inventing an ordering of zero-information points.
If a future model lacks member predictions this endpoint returns501, while existing
`/suggest` continues to work through the original Predictor protocol.

## GET /model-card

No body. Meta plus:

- `validation_status: "available" | "unavailable"`.
- `provenance`: dataset, physics/material, training-configuration and fitted-model
  SHA256 fingerprints, plus reproducible model_id and fingerprint method details.
- `training_data_summary`: rows, experiment count, source counts, per-material
  burning/nonburning counts and ranges; input data quality limitations.
- `features: string[]`, `ensemble_size`, `uncertainty_method`.
- `validation`: existing 3-model scores plus out-of-fold probability reliability
  bins/ECE/log loss and material/experiment slices. Single-class/no-burning slices
  return null where a metric is undefined. Brier's binned decomposition includes
  its residual term instead of claiming an exact unbinned identity.
- `limitations: string[]`, including synthetic data, approximate physics, empirical
  interval coverage, and ensemble agreement outside data support.

Newly trained model and metrics carry matching fingerprints. Startup rejects
metrics from a different dataset or fitted-model identity even if row counts match.
Runtime physics source/material constants and inference constants must match the
saved identity; newly fingerprinted models fail to load if these change and require
retraining. Legacy artifacts without fingerprints remain inferable and are labelled
accordingly. Fingerprints identify content; they are not authenticity signatures.

The existing `/metrics` response gains additive `validation` and `provenance` fields;
all old keys retain their meaning. [API.md](API.md) remains the frontend's v1 contract.

## Request tracing

Responses include `X-Request-ID` and `Server-Timing: app;dur=...` (milliseconds).
A caller-supplied request id of1–64 letters/digits/periods/underscores/hyphens is
preserved; other values are replaced with a generated id. Fingerprinted models
also expose `X-Model-ID`. Both frontend origins can read these headers through CORS.
This makes a frontend failure traceable to the corresponding backend log entry.

Scientific references: [BALD original paper](https://arxiv.org/abs/1112.5745),
[probability calibration and Brier-score interpretation](https://scikit-learn.org/stable/modules/calibration.html).
