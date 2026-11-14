// API contract shared with the backend (FastAPI). Do not change without telling the backend dev.

// POST /predict
export type PredictReq = { o2: number; pressure_kpa: number; flow_cm_s: number; material: string };
export type PredictRes = {
  spread_rate_mm_s: number;
  spread_ci: [number, number];
  extinction_prob: number;
  extinction_ci: [number, number];
  physics_baseline_mm_s: number; // classical formula, so we can show "physics vs learned"
  in_data_range: boolean; // false = extrapolating, show warning
  // optional extras from the real backend
  regime?: "burning" | "quench_extinction" | "blowoff_extinction" | "no_ignition";
  confidence?: "high" | "medium" | "low";
  data_source?: DataSource;
};

export type DataSource = "synthetic" | "nasa" | "mixed";

// GET /map?material=cotton&pressure_kpa=70
export type TestedPoint = {
  o2: number;
  flow_cm_s: number;
  burned: boolean;
  source: string; // "nasa" | "synthetic" (real backend) or a free-text citation
  experiment_id?: string;
  campaign?: string | null;
  pressure_kpa?: number;
};
export type BoundaryPoint = { flow_cm_s: number; o2_percent: number };
export type MapRes = {
  x_axis: "o2";
  y_axis: "flow_cm_s";
  x: number[];
  y: number[];
  z: number[][]; // extinction probability per cell, z[yIndex][xIndex]
  uncertainty: number[][]; // real backend: ensemble SD 0–0.5; mock: 0–1
  tested_points: TestedPoint[];
  flow_scale?: "log" | "linear"; // y spacing
  boundary?: BoundaryPoint[]; // model P(extinction)=0.5 line
  physics_boundary?: BoundaryPoint[]; // physics-only line
  data_source?: DataSource;
};

// GET /suggest?material=cotton
export type SuggestPoint = {
  o2: number;
  flow_cm_s: number;
  pressure_kpa: number;
  uncertainty: number;
  reason: string;
  score?: number; // 0–1 acquisition priority (real backend)
  extinction_probability?: number;
};
export type SuggestRes = { points: SuggestPoint[] };

// GET /explain
export type ExplainTarget = "extinction" | "spread_rate";
export type ExplainRes = { features: { name: string; importance: number }[]; target?: ExplainTarget; contribution_units?: string };

// GET /materials  (optional — frontend falls back to src/data/materials.ts)
export type Material = { id: string; name: string; seen_in?: string; thickness_mm?: number; aliases?: string[] };
export type MaterialsRes = { materials: Material[] };

// GET /health → { "status": "ok" }

// POST /optimize — smallest change to the controls that makes extinction likely
export type OptimizeReq = {
  scenario: { material: string; o2: number; pressure_kpa: number; flow_cm_s: number };
  bounds: { o2_range: [number, number]; pressure_range: [number, number]; flow_range: [number, number] };
  target_extinction_probability: number;
  resolution: number;
  k: number;
  supported_only: boolean;
};
export type OptimizeCandidate = {
  prediction: PredictRes & { inputs?: { o2_percent: number; pressure_kpa: number; flow_cm_s: number } };
  change_cost: number;
  deltas: { o2_percent: number; pressure_kpa: number; flow_cm_s: number };
};
export type OptimizeRes = {
  status: "found" | "not_found";
  target_achievable: boolean;
  candidates: OptimizeCandidate[];
  best_available: OptimizeCandidate | null;
  limitations?: string[];
};
