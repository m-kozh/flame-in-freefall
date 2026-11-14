# Flame in Freefall — Backend API Contract

Additional batch prediction, scenario analysis, constrained research search,
information-driven experimental design and model-card endpoints are documented in
[ADVANCED_API.md](ADVANCED_API.md). Existing v1 endpoints keep their original types.
NASA source observations, dataset endpoints, eligibility and observation-window
semantics are documented in [NASA_DATA.md](NASA_DATA.md).

**Status: stable (v1, frontend agreement confirmed 2026-11-13).** Field names in this document will not be renamed or removed.
New optional fields may be added later; clients should ignore fields they do not know.

- Base URL (dev): `http://localhost:8000`
- All bodies are JSON (`Content-Type: application/json`).
- CORS is enabled for `http://localhost:3000` and `http://localhost:5173`.
- Interactive docs (auto-generated): `http://localhost:8000/docs`

> **Every response carries `data_source`** (`"synthetic"` | `"nasa"` | `"mixed"`).
> While it is `"synthetic"` the UI must show a "synthetic data" badge: the model
> is currently trained on generated data, not on NASA measurements.

The v0.3 default is `"mixed"` (2000 synthetic +11 verified NASA rows); the examples
below illustrate the fixed schema rather than a current source/accuracy claim.
NASA observation windows and per-material coverage are exposed by `/datasets` and
`/model-card`; a mixed source badge is not a NASA validation badge.

---

## 0. Notes for the frontend (differences from the types you sent)

Your draft types are supported almost unchanged. The response of each endpoint is a
**superset**: it contains the canonical field names *and* your names as aliases.

| Your draft | What the API does |
|---|---|
| `PredictReq.o2` | Accepted. `o2_percent` (canonical) and `o2` are both valid request keys. |
| `PredictRes.spread_rate_mm_s: number` | Supported exactly. The additional `spread_rate_interval` object has `{mean, lower, upper}`. `POST /predict/detailed` provides the original object's shape under `spread_rate_mm_s`. |
| `spread_ci`, `extinction_prob`, `extinction_ci`, `in_data_range`, `physics_baseline_mm_s` | Present exactly as you typed them. |
| `GET /map?material=..&pressure_kpa=..` | Supported (plus `POST /map` with a body for custom ranges). |
| `MapRes.x_axis / y_axis / x / y / z / uncertainty / tested_points` | Present exactly as you typed them. `z` is the extinction probability. |
| `SuggestRes.points[]` with `o2` | Present. Each point has both `o2` and `o2_percent`. |
| `GET /explain` → `features: {name, importance}[]` | Supported. `POST /explain` additionally returns per-prediction contributions. |
| `material: "cotton"` | Accepted as an alias of `sibal_cotton_fiberglass`. See `GET /materials` for ids and aliases. |

Aliases are plain duplicates of the canonical value (same numbers). Use whichever you prefer.
If both `o2` and `o2_percent` are supplied they must be equal; conflicts return 422.
Unknown request fields are rejected. Numeric inputs must be finite; booleans are not
accepted as numbers. The sample responses below illustrate the schema, not measured results.

---

## 1. Conventions

### 1.1 Units

| Field | Unit | Meaning |
|---|---|---|
| `o2_percent` (alias `o2`) | % by volume (mole %) | Oxygen concentration, e.g. `21` for air. **Not** a 0–1 fraction. |
| `pressure_kpa` | kPa | Total cabin pressure. 101.3 kPa = 14.7 psia. |
| `flow_cm_s` | cm/s | Speed of the ambient (opposed) air flow past the sample. |
| `thickness_mm` | mm | Full sample thickness. |
| `spread_rate_mm_s`, `physics_baseline_mm_s` | mm/s | Flame spread rate. |
| `extinction_probability` (alias `extinction_prob`) | 0–1 | Probability that the sample does **not** sustain a flame (fails to ignite or self-extinguishes). |
| `uncertainty` | 0–0.5 | Standard deviation of the extinction probability across the model ensemble. `0` = all ensemble members agree. Values above ~0.15 mean strong disagreement. |
| temperatures | K | |
| density | kg/m³ | |
| specific heat | J/(kg·K) | |

### 1.2 Input ranges (anything outside → `422`)

| Field | Allowed range | Default |
|---|---|---|
| `material` | an id or alias from `GET /materials` (case-insensitive) | required |
| `o2_percent` | 5 – 100 | required |
| `pressure_kpa` | 10 – 150 | required for `/predict`; `101.3` where optional |
| `flow_cm_s` | 0 – 200 | required |
| `thickness_mm` | 0.01 – 10 | the material's library thickness |
| `resolution` | 5 – 100 | `50` |
| `k` | 1 – 20 | `5` |

### 1.3 Matrix orientation (important for the heat map)

All 2-D arrays returned by `/map` use the same indexing:

```
matrix[i][j]  ->  flow_cm_s = flow_values[i]   (row    = y axis)
                  o2_percent = o2_values[j]    (column = x axis)
```

This is the `z[row][col]` convention of Plotly / ECharts heat maps with
`x = o2_values` and `y = flow_values`. Transpose if you want oxygen on the y axis
(the classic "U-shaped" flammability map has flow on x and oxygen on y).

### 1.4 Errors

**422 — invalid input** (FastAPI/Pydantic standard format, one entry per problem):

```json
{
  "data_source": "synthetic",
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "o2_percent"],
      "msg": "Input should be less than or equal to 100",
      "input": 140
    }
  ]
}
```

Unknown material uses the same shape:

```json
{
  "data_source": "synthetic",
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "material"],
      "msg": "Value error, Unknown material 'wood'. Valid ids: cellulose_thin, pmma_thin, pmma_thick, sibal_cotton_fiberglass, nomex",
      "input": "wood"
    }
  ]
}
```

`loc[0]` is `"body"` for POST bodies and `"query"` for GET query parameters.
To show an error, join `loc.slice(1)` and `msg`.

**503 — model not trained yet** (run `python scripts/train_model.py` on the backend):

```json
{ "detail": "Model is not loaded. Run `python scripts/train_model.py` and restart the server.", "data_source": "synthetic" }
```

`/metrics` also returns 503 if metrics are missing or inconsistent with the loaded
model. A future predictor without a SHAP adapter returns 501 from `/explain`, using
the same `{detail, data_source}` shape. Unknown paths (404) and unexpected backend
errors (500) also include source metadata; API documentation and CORS preflight
responses use their standard HTTP formats.

---

## 2. Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + model version |
| GET | `/materials` | Material library |
| GET | `/presets` | Cabin atmosphere presets |
| POST | `/predict` | Single prediction with uncertainty |
| POST | `/predict/detailed` | Original object-shaped spread-rate response |
| POST / GET | `/map` | Flammability map over (flow, O2) |
| GET | `/suggest` | "Next best experiment" recommendations |
| POST / GET | `/explain` | Feature importance (SHAP) |
| GET | `/metrics` | Cross-validation metrics of the 3 models |

---

### 2.1 `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "residual-ensemble-0.1.0",
  "data_source": "synthetic"
}
```

`status` is `"ok"` or `"model_not_loaded"`. The endpoint always returns `200`.

---

### 2.2 `GET /materials`

```json
{
  "materials": [
    {
      "id": "sibal_cotton_fiberglass",
      "name": "SIBAL-like cotton/fiberglass fabric",
      "aliases": ["cotton", "sibal"],
      "density_kg_m3": 550.0,
      "specific_heat_j_kgk": 1300.0,
      "thickness_mm": 0.33,
      "vaporization_temp_k": 620.0,
      "thermally_thin": true,
      "limiting_o2_percent": 14.0,
      "notes": "Illustrative material properties; composition and thermal parameters require verification."
    }
  ],
  "data_source": "synthetic"
}
```

Material ids (stable): `cellulose_thin`, `pmma_thin`, `pmma_thick`,
`sibal_cotton_fiberglass`, `nomex`.
Property values are approximations and may be tuned; ids will not change.
`limiting_o2_percent` is the illustrative material base oxygen limit (%) before
the flow, pressure and thickness corrections; it is not a certified material LOC.
`thermally_thin: false` means the physics baseline is outside its range of validity
for that material — show predictions for it as indicative only.

---

### 2.3 `GET /presets`

```json
{
  "presets": [
    {
      "id": "iss_standard",
      "name": "ISS standard atmosphere",
      "o2_percent": 21.0,
      "pressure_kpa": 101.3,
      "flow_cm_s": 10.0,
      "illustrative": false,
      "note": "Sea-level-like air. Flow of 10 cm/s is an illustrative assumption, not a NASA ventilation specification."
    }
  ],
  "data_source": "synthetic"
}
```

Preset ids (stable): `iss_standard`, `exploration_atmosphere` (34% O2, 56.5 kPa),
`lunar_base_upper_bound` (30% O2, 70.3 kPa), `spacesuit_pure_o2` (100% O2, 29.6 kPa,
`illustrative: true`).

---

### 2.4 `POST /predict`

**Request**

```json
{
  "material": "sibal_cotton_fiberglass",
  "o2_percent": 21.0,
  "pressure_kpa": 101.3,
  "flow_cm_s": 10.0,
  "thickness_mm": 0.33
}
```

`thickness_mm` is optional. `o2` is accepted instead of `o2_percent`.

**Response `200`**

```json
{
  "spread_rate_mm_s": 1.84,
  "spread_rate_interval": { "mean": 1.84, "lower": 1.31, "upper": 2.52 },
  "spread_ci": [1.31, 2.52],
  "extinction_probability": 0.06,
  "extinction_prob": 0.06,
  "extinction_ci": [0.01, 0.14],
  "uncertainty": 0.04,
  "regime": "burning",
  "physics_baseline_mm_s": 1.62,
  "physics_regime": "burning",
  "confidence": "high",
  "in_data_range": true,
  "inputs": {
    "material": "sibal_cotton_fiberglass",
    "o2_percent": 21.0,
    "pressure_kpa": 101.3,
    "flow_cm_s": 10.0,
    "thickness_mm": 0.33
  },
  "model_version": "residual-ensemble-0.1.0",
  "data_source": "synthetic"
}
```

| Field | Type | Meaning |
|---|---|---|
| `spread_rate_mm_s` | number | Predicted mean flame spread rate **if the flame is sustained**, in mm/s. |
| `spread_rate_interval` | `{mean, lower, upper}` | The mean and nominal 90% prediction interval. Bootstrap variation is combined with training-fold residual noise; coverage is evaluated on held-out experiment groups and is not guaranteed to equal 90%. |
| `spread_ci` | `[number, number]` | Alias: `[lower, upper]`. |
| `extinction_probability` | number 0–1 | P(no sustained flame). `extinction_prob` is an alias. |
| `extinction_ci` | `[number, number]` | 5th–95th percentile of the extinction probability across the ensemble. |
| `uncertainty` | number | See 1.1. |
| `regime` | string | The model's verdict, one of `"burning"`, `"quench_extinction"`, `"blowoff_extinction"`, `"no_ignition"`. It is `"burning"` exactly when `extinction_probability < 0.5`. |
| `physics_baseline_mm_s` | number | What the physics formula alone predicts (`0` if the physics baseline says the flame does not survive). Use it for "physics vs learned". |
| `physics_regime` | string | The physics baseline's own verdict (same 4 values). |
| `confidence` | `"high"` \| `"medium"` \| `"low"` | Summary of ensemble agreement and distance from training data. |
| `in_data_range` | boolean | `false` = the query is outside the region covered by training data for this material (extrapolation — show a warning). |
| `inputs` | object | The inputs actually used, with the material alias resolved and the default thickness filled in. |

Regimes:

- `burning` — sustained flame spread.
- `quench_extinction` — flow too slow; radiative heat loss wins (the microgravity-specific danger zone boundary).
- `blowoff_extinction` — flow too fast; the flame is blown off.
- `no_ignition` — oxygen too low to burn at any flow speed.

UI hint: when `regime != "burning"`, show "does not sustain a flame" and de-emphasise
`spread_rate_mm_s` (it is the conditional rate *if* it burned).

`POST /predict/detailed` accepts the identical request and returns the identical fields,
except `spread_rate_mm_s` is `{mean, lower, upper}`. This fixed compatibility endpoint
supports the initial backend specification without changing the frontend endpoint's type.

Complete `/predict/detailed` response example (same request as above):

```json
{
  "spread_rate_mm_s": {"mean": 1.84, "lower": 1.31, "upper": 2.52},
  "spread_rate_interval": {"mean": 1.84, "lower": 1.31, "upper": 2.52},
  "spread_ci": [1.31, 2.52],
  "extinction_probability": 0.06,
  "extinction_prob": 0.06,
  "extinction_ci": [0.01, 0.14],
  "uncertainty": 0.04,
  "regime": "burning",
  "physics_baseline_mm_s": 1.62,
  "physics_regime": "burning",
  "confidence": "high",
  "in_data_range": true,
  "inputs": {
    "material": "sibal_cotton_fiberglass",
    "o2_percent": 21.0,
    "pressure_kpa": 101.3,
    "flow_cm_s": 10.0,
    "thickness_mm": 0.33
  },
  "model_version": "residual-ensemble-0.1.0",
  "data_source": "synthetic"
}
```

Non-burning regime subtypes are heuristic physical labels, not independently learned
mechanism diagnoses. `extinction_ci` measures ensemble disagreement, not a calibrated
confidence interval for the true probability.

`confidence` is heuristic: low when outside local training support or extinction SD
is at least 0.15; high when supported, SD is below 0.08 and total log-rate SD is below
0.30; medium otherwise. These thresholds are display choices, not safety criteria.

---

### 2.5 `POST /map` and `GET /map`

Flammability map for one material at one pressure, over a grid of flow speed × oxygen.

**POST request** (all fields except `material` optional)

```json
{
  "material": "sibal_cotton_fiberglass",
  "pressure_kpa": 101.3,
  "flow_range": [0.5, 100.0],
  "o2_range": [12.0, 40.0],
  "resolution": 50,
  "flow_scale": "log",
  "thickness_mm": 0.33
}
```

**GET form** — same thing with query parameters, all optional except `material`:

```
GET /map?material=cotton&pressure_kpa=70
GET /map?material=cotton&pressure_kpa=70&flow_min=0.5&flow_max=100&o2_min=12&o2_max=40&resolution=50&flow_scale=log
```

| Field | Default | Notes |
|---|---|---|
| `pressure_kpa` | `101.3` | |
| `flow_range` / `flow_min`,`flow_max` | `[0.5, 100]` | `min < max`. With `flow_scale: "log"`, `min` must be > 0. |
| `o2_range` / `o2_min`,`o2_max` | `[12, 40]` | `min < max` |
| `resolution` | `50` | Grid is `resolution × resolution`. |
| `flow_scale` | `"log"` | `"log"` or `"linear"` spacing of `flow_values`. Log spacing shows the low-flow quenching branch clearly — use a log axis for flow. |

**Response `200`** (arrays shortened; a real response has `resolution` entries per axis)

```json
{
  "material": "sibal_cotton_fiberglass",
  "pressure_kpa": 101.3,
  "thickness_mm": 0.33,
  "flow_scale": "log",

  "flow_values": [0.5, 2.9, 17.1, 100.0],
  "o2_values": [12.0, 21.3, 30.7, 40.0],
  "spread_rate": [
    [0.21, 0.74, 1.90, 3.10],
    [0.35, 1.52, 2.61, 3.84],
    [0.33, 1.71, 2.95, 4.20],
    [0.18, 0.92, 2.40, 3.75]
  ],
  "extinction_probability": [
    [0.99, 0.93, 0.41, 0.07],
    [0.98, 0.12, 0.02, 0.01],
    [0.97, 0.05, 0.01, 0.01],
    [0.99, 0.81, 0.09, 0.03]
  ],
  "uncertainty": [
    [0.01, 0.06, 0.21, 0.05],
    [0.01, 0.08, 0.01, 0.01],
    [0.02, 0.04, 0.01, 0.01],
    [0.01, 0.12, 0.07, 0.02]
  ],

  "boundary": [
    { "flow_cm_s": 0.5, "o2_percent": 29.6 },
    { "flow_cm_s": 2.9, "o2_percent": 16.8 },
    { "flow_cm_s": 17.1, "o2_percent": 15.9 },
    { "flow_cm_s": 100.0, "o2_percent": 23.4 }
  ],
  "physics_boundary": [
    { "flow_cm_s": 0.5, "o2_percent": 27.1 },
    { "flow_cm_s": 2.9, "o2_percent": 16.2 },
    { "flow_cm_s": 17.1, "o2_percent": 15.5 },
    { "flow_cm_s": 100.0, "o2_percent": 22.0 }
  ],

  "presets_overlay": [
    {
      "id": "iss_standard",
      "name": "ISS standard atmosphere",
      "o2_percent": 21.0,
      "pressure_kpa": 101.3,
      "flow_cm_s": 10.0,
      "matches_map_pressure": true
    }
  ],

  "tested_points": [
    {
      "o2_percent": 21.0,
      "o2": 21.0,
      "flow_cm_s": 10.2,
      "pressure_kpa": 101.3,
      "burned": true,
      "source": "synthetic",
      "experiment_id": "SYN-03"
    }
  ],

  "x_axis": "o2",
  "y_axis": "flow_cm_s",
  "x": [12.0, 21.3, 30.7, 40.0],
  "y": [0.5, 2.9, 17.1, 100.0],
  "z": [
    [0.99, 0.93, 0.41, 0.07],
    [0.98, 0.12, 0.02, 0.01],
    [0.97, 0.05, 0.01, 0.01],
    [0.99, 0.81, 0.09, 0.03]
  ],

  "model_version": "residual-ensemble-0.1.0",
  "data_source": "synthetic"
}
```

| Field | Meaning |
|---|---|
| `flow_values`, `o2_values` | Grid axes (ascending). |
| `spread_rate[i][j]` | Mean spread rate in mm/s (conditional on burning), at `flow_values[i]`, `o2_values[j]`. |
| `extinction_probability[i][j]` | P(no sustained flame), 0–1. |
| `uncertainty[i][j]` | Ensemble disagreement on the extinction probability (see 1.1). |
| `boundary` | The learned flammability boundary: for each flow value, the oxygen level where `extinction_probability` crosses 0.5 (linear interpolation; if a row crosses more than once the highest crossing is returned). Sorted by `flow_cm_s`. Flow values with no crossing inside `o2_range` are omitted, so the list may be shorter than `flow_values` (or empty). |
| `physics_boundary` | Same curve from the physics baseline alone (its limiting oxygen concentration), for a "physics vs learned" overlay. Same omission rule. |
| `presets_overlay` | All cabin presets, to draw as markers. `matches_map_pressure` is `true` when the preset's pressure is within 10 kPa of the map's pressure (otherwise the marker is only indicative, since the map is drawn for a different pressure). |
| `tested_points` | Training-data tests for this material whose pressure is within 10 kPa of the map's pressure. `burned` = sustained flame. Capped at 500 points. |
| `x_axis`, `y_axis`, `x`, `y`, `z` | Aliases: `x` = `o2_values`, `y` = `flow_values`, `z` = `extinction_probability`. |

Performance: a 50 × 50 map returns in under 1 s; identical requests are cached.

---

### 2.6 `GET /suggest`

Active learning: which experiments would teach the model the most? Candidates are scored
by ensemble disagreement, weighted towards the flammability boundary and towards regions
with no nearby test data. The returned points are spread out (never `k` near-duplicates).

```
GET /suggest?material=cotton
GET /suggest?material=cotton&pressure_kpa=56.5&k=5
```

| Query param | Default |
|---|---|
| `material` | required |
| `pressure_kpa` | `101.3` |
| `k` | `5` |

**Response `200`**

```json
{
  "points": [
    {
      "o2_percent": 17.5,
      "o2": 17.5,
      "pressure_kpa": 56.5,
      "flow_cm_s": 1.4,
      "uncertainty": 0.23,
      "extinction_probability": 0.47,
      "score": 1.0,
      "reason": "Model is split on whether this burns (P(extinction) = 0.47) and the nearest test is far away; a test here pins down the low-flow quenching limit."
    }
  ],
  "material": "sibal_cotton_fiberglass",
  "pressure_kpa": 56.5,
  "model_version": "residual-ensemble-0.1.0",
  "data_source": "synthetic"
}
```

`points` is sorted by `score` descending (`score` is 0–1, relative to the best
candidate). `reason` is one plain-English sentence, ready to display.

---

### 2.7 `POST /explain` and `GET /explain`

SHAP explanations of the machine-learning part of the model.

- `GET /explain` → global importance only (`local_contributions` is `[]`).
  Optional query: `?target=spread_rate` (default) or `?target=extinction`.
- `POST /explain` → global importance **and** the contributions for one prediction.

**POST request** — same body as `/predict`, plus optional `target`:

```json
{
  "material": "sibal_cotton_fiberglass",
  "o2_percent": 21.0,
  "pressure_kpa": 101.3,
  "flow_cm_s": 10.0,
  "target": "spread_rate"
}
```

**Response `200`**

```json
{
  "target": "spread_rate",
  "global_importance": [
    { "feature": "flow_cm_s", "importance": 0.48 },
    { "feature": "o2_percent", "importance": 0.31 },
    { "feature": "pressure_kpa", "importance": 0.21 }
  ],
  "features": [
    { "name": "flow_cm_s", "importance": 0.48 },
    { "name": "o2_percent", "importance": 0.31 },
    { "name": "pressure_kpa", "importance": 0.21 }
  ],
  "local_contributions": [
    { "feature": "flow_cm_s", "value": 10.0, "contribution": 0.11 },
    { "feature": "o2_percent", "value": 21.0, "contribution": -0.04 },
    { "feature": "pressure_kpa", "value": 101.3, "contribution": 0.02 }
  ],
  "base_value": 0.03,
  "prediction": 0.12,
  "contribution_units": "ln(spread rate / physics baseline)",
  "model_version": "residual-ensemble-0.1.0",
  "data_source": "synthetic"
}
```

| Field | Meaning |
|---|---|
| `target` | `"spread_rate"`: explains the learned correction to the physics spread rate. `"extinction"`: explains the extinction classifier. |
| `global_importance` | Mean absolute SHAP value per feature over the training data, normalised so all importances sum to 1, sorted descending. `features` is an alias with `name` instead of `feature`. |
| `local_contributions` | SHAP value of each feature for this one prediction, sorted by absolute contribution. Positive = pushes the output up. `value` is the feature's input value. |
| `base_value`, `prediction` | `base_value + sum(contribution) = prediction`. Useful for a waterfall chart. `null` for GET. |
| `contribution_units` | `"ln(spread rate / physics baseline)"` for `spread_rate` (so `+0.10` ≈ 10% faster than the physics formula says); `"log-odds of extinction"` for `extinction`. |

SHAP values and base values are averaged across **all** ensemble members. `prediction`
is their mean raw residual or mean raw log-odds, respectively. Because exp/sigmoid are
nonlinear, it is not the log of the final arithmetic mean spread rate or the logit of
the final averaged extinction probability. The physics features describe the positive
potential rate before the extinction gate; this avoids log(0) and lets the ML correct
a wrong physics extinction decision.

Feature names (stable):

| Feature | Meaning |
|---|---|
| `o2_percent`, `pressure_kpa`, `flow_cm_s`, `thickness_mm` | The inputs. |
| `density_kg_m3`, `specific_heat_j_kgk`, `vaporization_temp_k` | Material properties. |
| `log_physics_spread_rate` | ln of the physics-formula spread rate. |
| `physics_o2_margin_percent` | Oxygen level minus the physics baseline's limiting oxygen concentration (positive = physics says flammable). |
| `physics_regime_code` | Physics verdict as a number: 0 burning, 1 quench, 2 blowoff, 3 no ignition. |

---

### 2.8 `GET /metrics`

Returns the contents of `artifacts/metrics.json`, written by training.
Metrics come from **grouped cross-validation** (whole experiments are held out, so the
numbers measure generalisation to unseen experiments, not memorisation).

```json
{
  "model_version": "residual-ensemble-0.1.0",
  "data_source": "synthetic",
  "trained_at": "2026-11-13T19:02:11Z",
  "n_rows": 2000,
  "n_experiments": 20,
  "cv": { "method": "GroupKFold", "group_column": "experiment_id", "n_splits": 5 },
  "models": {
    "physics_only": {
      "log_spread_mae": 0.31,
      "log_spread_rmse": 0.39,
      "extinction_roc_auc": 0.88,
      "extinction_brier": 0.14,
      "interval_coverage_90": null
    },
    "ml_only": {
      "log_spread_mae": 0.27,
      "log_spread_rmse": 0.36,
      "extinction_roc_auc": 0.90,
      "extinction_brier": 0.11,
      "interval_coverage_90": 0.78
    },
    "physics_plus_ml_residual": {
      "log_spread_mae": 0.16,
      "log_spread_rmse": 0.21,
      "extinction_roc_auc": 0.95,
      "extinction_brier": 0.08,
      "interval_coverage_90": 0.87
    }
  }
}
```

(The numbers above are placeholders — call the endpoint for the real ones.)

| Metric | Meaning | Better is |
|---|---|---|
| `log_spread_mae`, `log_spread_rmse` | Error of ln(spread rate) on burning tests. `0.10` ≈ 10% error. | lower |
| `extinction_roc_auc` | How well burning and non-burning tests are separated (0.5 = chance, 1 = perfect). | higher |
| `extinction_brier` | Mean squared error of the extinction probability. | lower |
| `interval_coverage_90` | Fraction of held-out tests whose true spread rate lies inside the 90% interval. Ideal ≈ 0.90. `null` for the physics model (it has no interval). | closer to 0.90 |

Speed metrics compare conditional-on-burning rates for all three models; the physics
model uses its positive potential rate before the extinction gate. Its extinction
probability is the illustrative fixed sigmoid(-oxygen_margin / 1.2), not a fitted
probability calibration. Additional `cv.folds`, `ensemble_size`, `seed`, and `notes`
fields record the exact validation procedure and experiment split audit.

---

## 3. TypeScript types (copy-paste)

```ts
type DataSource = "synthetic" | "nasa" | "mixed";
type Regime = "burning" | "quench_extinction" | "blowoff_extinction" | "no_ignition";

type PredictInputs = {
  material: string;
  pressure_kpa: number;
  flow_cm_s: number;
  thickness_mm?: number;
};
type PredictReq = PredictInputs & (
  { o2_percent: number; o2?: never } | { o2: number; o2_percent?: never }
);

type PredictRes = {
  spread_rate_mm_s: number;
  spread_rate_interval: { mean: number; lower: number; upper: number };
  spread_ci: [number, number];
  extinction_probability: number;
  extinction_prob: number;
  extinction_ci: [number, number];
  uncertainty: number;
  regime: Regime;
  physics_baseline_mm_s: number;
  physics_regime: Regime;
  confidence: "high" | "medium" | "low";
  in_data_range: boolean;
  inputs: Required<PredictInputs> & { o2_percent: number };
  model_version: string;
  data_source: DataSource;
};

type MapReq = {
  material: string;
  pressure_kpa?: number;
  flow_range?: [number, number];
  o2_range?: [number, number];
  resolution?: number;
  flow_scale?: "log" | "linear";
  thickness_mm?: number;
};

type DetailedPredictRes = Omit<PredictRes, "spread_rate_mm_s"> & {
  spread_rate_mm_s: PredictRes["spread_rate_interval"];
};

type BoundaryPoint = { flow_cm_s: number; o2_percent: number };

type MapRes = {
  material: string;
  pressure_kpa: number;
  thickness_mm: number;
  flow_scale: "log" | "linear";
  flow_values: number[];
  o2_values: number[];
  spread_rate: number[][];
  extinction_probability: number[][];
  uncertainty: number[][];
  boundary: BoundaryPoint[];
  physics_boundary: BoundaryPoint[];
  presets_overlay: {
    id: string; name: string; o2_percent: number; pressure_kpa: number;
    flow_cm_s: number; matches_map_pressure: boolean;
  }[];
  tested_points: {
    o2_percent: number; o2: number; flow_cm_s: number; pressure_kpa: number;
    burned: boolean; source: string; experiment_id: string;
  }[];
  x_axis: "o2"; y_axis: "flow_cm_s";
  x: number[]; y: number[]; z: number[][];
  model_version: string;
  data_source: DataSource;
};

type SuggestRes = {
  points: {
    o2_percent: number; o2: number; pressure_kpa: number; flow_cm_s: number;
    uncertainty: number; extinction_probability: number; score: number; reason: string;
  }[];
  material: string;
  pressure_kpa: number;
  model_version: string;
  data_source: DataSource;
};

type ExplainRes = {
  target: "spread_rate" | "extinction";
  global_importance: { feature: string; importance: number }[];
  features: { name: string; importance: number }[];
  local_contributions: { feature: string; value: number; contribution: number }[];
  base_value: number | null;
  prediction: number | null;
  contribution_units: string;
  model_version: string;
  data_source: DataSource;
};
```

---

## 4. Honesty notes (please keep these visible in the UI)

- The model is a **physics baseline plus a learned correction**. The physics constants are
  hackathon approximations, not validated values.
- With `data_source: "synthetic"` every number is a demonstration of the method, not a
  fire-safety result.
- `in_data_range: false` means the model is extrapolating; `confidence: "low"` can
  also indicate strong ensemble disagreement inside the supported region.
