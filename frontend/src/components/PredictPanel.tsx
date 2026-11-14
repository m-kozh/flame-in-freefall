"use client";
import { useEffect, useState } from "react";
import Plot, { axis, darkLayout } from "./Plot";
import { getExplain, predict } from "@/lib/api";
import type { ExplainRes, ExplainTarget, PredictRes } from "@/lib/types";
import type { Material } from "@/lib/types";
import type { Environment } from "@/data/environments";
import { Badge, LowConfidence, verdictOf } from "./Verdicts";

const FEATURE_LABELS: Record<string, string> = {
  o2_percent: "Oxygen concentration",
  flow_cm_s: "Airflow speed",
  pressure_kpa: "Pressure",
  physics_o2_margin_percent: "O₂ margin above physics limit",
  log_physics_spread_rate: "Physics spread rate (de Ris)",
  specific_heat_j_kgk: "Material specific heat",
  thickness_mm: "Sample thickness",
  density_kg_m3: "Material density",
  vaporization_temp_k: "Vaporization temperature",
  physics_regime_code: "Physics regime",
};
const featureLabel = (n: string) => FEATURE_LABELS[n] ?? n;

const REGIME_LABELS: Record<string, string> = {
  burning: "Burning",
  quench_extinction: "Low-flow quenching",
  blowoff_extinction: "Blow-off",
  no_ignition: "No ignition",
};

function RangeBar({ value, lo, hi, max, fmt }: { value: number; lo: number; hi: number; max: number; fmt: (n: number) => string }) {
  const pct = (n: number) => `${Math.min(100, (n / max) * 100)}%`;
  return (
    <div className="mt-2">
      <div className="relative h-2 rounded-full bg-surface-2">
        <div className="absolute h-2 rounded-full bg-fire/40" style={{ left: pct(lo), width: `calc(${pct(hi)} - ${pct(lo)})` }} />
        <div className="absolute -top-1 h-4 w-1 rounded bg-ink" style={{ left: pct(value) }} />
      </div>
      <div className="mt-1 text-xs text-ink-3">
        95% interval: {fmt(lo)} – {fmt(hi)}
      </div>
    </div>
  );
}

export default function PredictPanel({ env, material }: { env: Environment; material: string; materials?: Material[] }) {
  const [res, setRes] = useState<PredictRes | null>(null);
  const [explain, setExplain] = useState<ExplainRes | null>(null);
  const [target, setTarget] = useState<ExplainTarget>("extinction");

  useEffect(() => {
    let alive = true;
    predict({ o2: env.o2, pressure_kpa: env.pressure_kpa, flow_cm_s: env.flow_cm_s, material }).then((r) => alive && setRes(r));
    return () => {
      alive = false;
    };
  }, [env.o2, env.pressure_kpa, env.flow_cm_s, material]);

  useEffect(() => {
    let alive = true;
    getExplain(target).then((e) => alive && setExplain(e));
    return () => {
      alive = false;
    };
  }, [target]);

  // Normalise and keep the features that matter, with readable names
  const feats = explain
    ? (() => {
        const total = explain.features.reduce((a, f) => a + f.importance, 0) || 1;
        return explain.features
          .map((f) => ({ name: featureLabel(f.name), importance: f.importance / total }))
          .filter((f) => f.importance >= 0.005)
          .sort((a, b) => b.importance - a.importance)
          .slice(0, 6);
      })()
    : [];

  const burn = res ? 1 - res.extinction_prob : 0;
  const learned = res ? res.spread_rate_mm_s - res.physics_baseline_mm_s : 0;

  return (
    <div>
      <p className="mb-4 text-sm text-ink-3">
        Model internals for your scenario: {env.o2}% O₂ · {env.pressure_kpa} kPa · {env.flow_cm_s} cm/s. Change the inputs at the top.
      </p>
      <div className="space-y-6">
        {res && (
          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-xl border border-line bg-surface p-5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="text-sm text-ink-2">Chance the fire keeps burning</div>
                <div className="flex flex-wrap gap-2">
                  {!res.in_data_range && <LowConfidence />}
                  <Badge v={verdictOf(res)} />
                </div>
              </div>
              <div className="mt-2 text-4xl font-semibold tabular-nums text-ink">{(burn * 100).toFixed(0)}%</div>
              {res.regime && (
                <div className="mt-1 text-xs text-ink-2">
                  Regime: <span className="text-ink">{REGIME_LABELS[res.regime] ?? res.regime}</span>
                  {res.confidence && <> · model confidence {res.confidence}</>}
                </div>
              )}
              <RangeBar value={burn} lo={1 - res.extinction_ci[1]} hi={1 - res.extinction_ci[0]} max={1} fmt={(n) => `${(n * 100).toFixed(0)}%`} />
            </div>
            <div className="rounded-xl border border-line bg-surface p-5">
              <div className="text-sm text-ink-2">Flame spread rate</div>
              <div className="mt-2 text-4xl font-semibold tabular-nums text-ink">
                {res.spread_rate_mm_s.toFixed(2)} <span className="text-lg text-ink-3">mm/s</span>
              </div>
              <RangeBar value={res.spread_rate_mm_s} lo={res.spread_ci[0]} hi={res.spread_ci[1]} max={Math.max(res.spread_ci[1] * 1.2, 0.5)} fmt={(n) => `${n.toFixed(2)} mm/s`} />
              <div className="mt-3 text-xs text-ink-2">
                Physics baseline {res.physics_baseline_mm_s.toFixed(2)} mm/s · learned correction {learned >= 0 ? "+" : ""}
                {learned.toFixed(2)} mm/s
              </div>
            </div>
            {!res.in_data_range && (
              <div className="rounded-xl border border-warn/50 bg-warn/10 p-4 text-sm text-ink md:col-span-2">
                <strong>Extrapolating.</strong> No NASA tests are close to these conditions, so treat this as a hypothesis — and a candidate for the next experiment.
              </div>
            )}
          </div>
        )}

        {explain && (
          <div className="rounded-xl border border-line bg-surface p-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-ink">What drives the prediction</div>
                <div className="text-xs text-ink-3">Mean |SHAP| feature importance across the model ensemble</div>
              </div>
              <div className="flex rounded-lg border border-line p-0.5 text-xs">
                {(["extinction", "spread_rate"] as const).map((t) => (
                  <button key={t} onClick={() => setTarget(t)} className={`rounded-md px-2.5 py-1 ${target === t ? "bg-surface-2 text-ink" : "text-ink-3"}`}>
                    {t === "extinction" ? "Will it burn?" : "Spread rate"}
                  </button>
                ))}
              </div>
            </div>
            <div className="h-[240px]">
              <Plot
                data={[
                  {
                    type: "bar",
                    orientation: "h",
                    x: [...feats].reverse().map((f) => f.importance),
                    y: [...feats].reverse().map((f) => f.name),
                    marker: { color: "#3987e5" },
                    text: [...feats].reverse().map((f) => `${(f.importance * 100).toFixed(0)}%`),
                    textposition: "outside",
                    textfont: { color: "#c3c2b7" },
                    hovertemplate: "%{y}: %{x:.0%}<extra></extra>",
                    cliponaxis: false,
                  },
                ]}
                layout={{
                  ...darkLayout,
                  autosize: true,
                  margin: { l: 210, r: 40, t: 8, b: 32 },
                  xaxis: { ...axis, tickformat: ".0%", range: [0, Math.max(0.01, ...feats.map((f) => f.importance)) * 1.2] },
                  yaxis: { ...axis, gridcolor: "rgba(0,0,0,0)" },
                  bargap: 0.45,
                }}
                config={{ displayModeBar: false, responsive: true }}
                useResizeHandler
                style={{ width: "100%", height: "100%" }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
