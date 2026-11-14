// MOCK MODEL — stands in for the backend until it is live.
// Shapes match src/lib/types.ts exactly. Numbers are illustrative, NOT NASA data.
import type { ExplainRes, MapRes, PredictReq, PredictRes, SuggestRes, TestedPoint } from "./types";
import { MATERIALS } from "@/data/materials";

const P_ISS = 101.3;

function params(material: string) {
  return MATERIALS.find((m) => m.id === material)?.mock ?? { o2crit: 18, k: 1 };
}

// Critical O2 (%) below which the flame dies: rises at low flow (radiative quenching),
// at high flow (blow-off) and slightly at reduced pressure.
function critO2(material: string, p: number, flow: number) {
  const { o2crit } = params(material);
  return o2crit + (P_ISS - p) * 0.05 + 6 * Math.exp(-flow / 1.2) + 0.08 * Math.max(0, flow - 25);
}

function burnProb(material: string, o2: number, p: number, flow: number) {
  return 1 / (1 + Math.exp(-(o2 - critO2(material, p, flow)) / 1.4));
}

function physicsBaseline(material: string, o2: number, p: number, flow: number) {
  const margin = o2 - critO2(material, p, flow);
  if (margin < -2) return 0;
  const { k } = params(material);
  return k * 0.8 * Math.pow((margin + 2) / 10, 1.5) * Math.pow(flow + 1, 0.25) * Math.pow(p / P_ISS, 0.6);
}

// Seeded RNG so the mock tested points are stable between reloads.
function rng(seed: number) {
  return () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
}

// Mock "tested points": deliberately clustered near ISS-like conditions (≤26% O2),
// mirroring the real situation that almost nothing was tested at Moon-base oxygen.
function testedPoints(material: string): (TestedPoint & { p: number })[] {
  const r = rng(material.split("").reduce((a, c) => a + c.charCodeAt(0), 7));
  const pts: (TestedPoint & { p: number })[] = [];
  for (let i = 0; i < 28; i++) {
    const o2 = 15 + r() * 11;
    const flow = 0.5 + r() * 38;
    const p = r() > 0.3 ? P_ISS : 70.7;
    const burned = burnProb(material, o2, p, flow) + (r() - 0.5) * 0.2 > 0.5;
    pts.push({ o2: +o2.toFixed(1), flow_cm_s: +flow.toFixed(1), burned, p, source: "MOCK — replace with PSI test matrix" });
  }
  return pts;
}

function uncertainty(material: string, o2: number, flow: number, p: number) {
  let best = Infinity;
  for (const t of testedPoints(material)) {
    const d = ((o2 - t.o2) / 4) ** 2 + ((flow - t.flow_cm_s) / 8) ** 2 + ((p - t.p) / 20) ** 2;
    if (d < best) best = d;
  }
  return Math.min(0.95, 0.05 + (1 - Math.exp(-best / 4)));
}

const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

export function mockPredict(req: PredictReq): PredictRes {
  const { o2, pressure_kpa: p, flow_cm_s: flow, material } = req;
  const ext = 1 - burnProb(material, o2, p, flow);
  const base = physicsBaseline(material, o2, p, flow);
  const residual = base * (0.12 * Math.sin(o2 / 4) + 0.05);
  const spread = ext > 0.95 ? 0 : base + residual;
  const u = uncertainty(material, o2, flow, p);
  const sw = 0.15 + 0.8 * u;
  const ew = 0.05 + 0.4 * u;
  return {
    spread_rate_mm_s: +spread.toFixed(2),
    spread_ci: [+(spread * (1 - sw)).toFixed(2), +(spread * (1 + sw)).toFixed(2)],
    extinction_prob: +ext.toFixed(3),
    extinction_ci: [+clamp(ext - ew).toFixed(3), +clamp(ext + ew).toFixed(3)],
    physics_baseline_mm_s: +base.toFixed(2),
    in_data_range: u < 0.5,
  };
}

export function mockMap(material: string, p: number): MapRes {
  const x: number[] = [];
  const y: number[] = [];
  for (let o = 12; o <= 45; o += 0.5) x.push(o);
  for (let f = 0; f <= 40; f += 1) y.push(f);
  const z = y.map((f) => x.map((o) => +(1 - burnProb(material, o, p, f)).toFixed(3)));
  const unc = y.map((f) => x.map((o) => +uncertainty(material, o, f, p).toFixed(3)));
  const tested = testedPoints(material)
    .filter((t) => Math.abs(t.p - p) < 20)
    .map(({ p: _p, ...t }) => t);
  return { x_axis: "o2", y_axis: "flow_cm_s", x, y, z, uncertainty: unc, tested_points: tested };
}

export function mockSuggest(material: string): SuggestRes {
  const cands: { o2: number; flow_cm_s: number; pressure_kpa: number; uncertainty: number }[] = [];
  for (const p of [56.5, 70.3]) for (let o = 26; o <= 36; o += 2) for (const f of [2, 5, 10, 20]) {
    cands.push({ o2: o, flow_cm_s: f, pressure_kpa: p, uncertainty: uncertainty(material, o, f, p) });
  }
  cands.sort((a, b) => b.uncertainty - a.uncertainty);
  const picked: typeof cands = [];
  for (const c of cands) {
    if (picked.every((q) => Math.abs(q.o2 - c.o2) + Math.abs(q.flow_cm_s - c.flow_cm_s) / 5 + Math.abs(q.pressure_kpa - c.pressure_kpa) / 10 > 3)) picked.push(c);
    if (picked.length === 4) break;
  }
  return {
    points: picked.map((c) => ({
      ...c,
      uncertainty: +c.uncertainty.toFixed(2),
      reason: `No tested points near ${c.o2}% O₂ / ${c.pressure_kpa} kPa — inside the planned exploration-atmosphere range, where the model is least certain.`,
    })),
  };
}

export function mockExplain(): ExplainRes {
  return {
    features: [
      { name: "Oxygen concentration", importance: 0.46 },
      { name: "Airflow speed", importance: 0.31 },
      { name: "Material", importance: 0.15 },
      { name: "Pressure", importance: 0.08 },
    ],
  };
}
