"use client";
import { useEffect, useState } from "react";
import { predict, predictBatch } from "@/lib/api";
import type { Material, PredictRes } from "@/lib/types";
import { ENVIRONMENTS, type Environment } from "@/data/environments";
import { Badge, LowConfidence, verdictOf } from "./Verdicts";

const REGIME_TEXT: Record<string, string> = {
  burning: "The flame sustains itself and keeps spreading.",
  quench_extinction: "Too little airflow: in microgravity there is no rising hot air, so the flame starves of oxygen and goes out (low-flow quenching).",
  blowoff_extinction: "Too much airflow: the flow cools the flame and pushes it off the material (blow-off).",
  no_ignition: "Not enough oxygen for the material to ignite at all.",
};

const fmt = (n: number, d = 1) => (Math.abs(n) < 1 && d > 0 ? n.toFixed(2) : n.toFixed(d));

// Lever = the closest value of ONE control (others unchanged) where the model
// estimates ≥90% chance the fire goes out on its own. null = none found in range.
type Lever = { kind: "oxygen" | "airflow"; value: number | null; prob: number | null; scanned: boolean };
const TARGET = 0.9;
const O2_STEPS = (o2: number) => {
  const out: number[] = [];
  for (let v = Math.round(o2 * 2) / 2; v >= 5; v -= 0.5) out.push(+v.toFixed(1));
  return out.slice(0, 200);
};
const FLOW_STEPS = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200];

export default function RiskReport({ env, material, materials }: { env: Environment; material: string; materials: Material[] }) {
  const [res, setRes] = useState<PredictRes | null>(null);
  const [iss, setIss] = useState<PredictRes | null>(null);
  const [o2Lever, setO2Lever] = useState<Lever>({ kind: "oxygen", value: null, prob: null, scanned: false });
  const [flowLever, setFlowLever] = useState<Lever>({ kind: "airflow", value: null, prob: null, scanned: false });
  const [loading, setLoading] = useState(true);
  const matName = materials.find((m) => m.id === material)?.name ?? material;
  const ISS = ENVIRONMENTS[0];

  useEffect(() => {
    let alive = true;
    setLoading(true);
    const p = env.pressure_kpa;
    const o2s = O2_STEPS(env.o2);
    Promise.all([
      predict({ material, o2: env.o2, pressure_kpa: p, flow_cm_s: env.flow_cm_s }),
      predict({ material, o2: ISS.o2, pressure_kpa: ISS.pressure_kpa, flow_cm_s: env.flow_cm_s }),
      // Lever 1: lower oxygen only — first (highest) O2 where extinction ≥ 90%
      predictBatch(o2s.map((o2) => ({ material, o2, pressure_kpa: p, flow_cm_s: env.flow_cm_s }))),
      // Lever 2: change airflow only — closest flow (log distance) where extinction ≥ 90%
      predictBatch(FLOW_STEPS.map((f) => ({ material, o2: env.o2, pressure_kpa: p, flow_cm_s: f }))),
    ]).then(([r, i, ob, fb]) => {
      if (!alive) return;
      setRes(r);
      setIss(i);
      const oi = ob.predictions.findIndex((x) => x.extinction_prob >= TARGET);
      setO2Lever({ kind: "oxygen", value: oi >= 0 ? o2s[oi] : null, prob: oi >= 0 ? ob.predictions[oi].extinction_prob : null, scanned: true });
      let best = -1;
      FLOW_STEPS.forEach((f, k) => {
        if (fb.predictions[k]?.extinction_prob >= TARGET) {
          const d = Math.abs(Math.log1p(f) - Math.log1p(env.flow_cm_s));
          if (best < 0 || d < Math.abs(Math.log1p(FLOW_STEPS[best]) - Math.log1p(env.flow_cm_s))) best = k;
        }
      });
      setFlowLever({ kind: "airflow", value: best >= 0 ? FLOW_STEPS[best] : null, prob: best >= 0 ? fb.predictions[best].extinction_prob : null, scanned: true });
      setLoading(false);
    });
    return () => {
      alive = false;
    };
  }, [env.o2, env.pressure_kpa, env.flow_cm_s, material]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!res) return <div className="h-[260px] animate-pulse rounded-2xl bg-surface" />;

  const burn = 1 - res.extinction_prob;
  const v = verdictOf(res);
  const headline =
    v === "dangerous" ? `${matName} would keep burning` : v === "safe" ? `${matName} would likely go out on its own` : `${matName} is on the edge — it may or may not keep burning`;
  const cmPerMin = res.spread_rate_mm_s * 6;
  const issBurn = iss ? 1 - iss.extinction_prob : null;

  const leverText = (l: Lever) => {
    if (!l.scanned) return null;
    const pct = l.prob != null ? ` (model: ${(l.prob * 100).toFixed(0)}% chance it goes out)` : "";
    if (l.kind === "oxygen") {
      if (l.value == null) return "Lowering oxygen alone (down to 5%) isn't enough, according to the model.";
      if (l.value >= env.o2) return "Already self-extinguishing at this oxygen level.";
      return `Lower oxygen from ${env.o2}% to about ${l.value}% — keeping pressure and airflow the same${pct}.`;
    }
    if (l.value == null) return "Changing airflow alone (0–200 cm/s) isn't enough at this oxygen level, according to the model.";
    if (l.value === env.flow_cm_s) return "Already self-extinguishing at this airflow.";
    return l.value < env.flow_cm_s
      ? `Cut airflow over the material to about ${fmt(l.value)} cm/s — in microgravity, still air starves the flame of oxygen${pct}.`
      : `Raise airflow to about ${fmt(l.value)} cm/s — strong flow can blow the flame off the material${pct}.`;
  };

  return (
    <section className="rounded-2xl border border-line bg-surface p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-base font-semibold text-ink">Fire risk report</h2>
        <div className="flex flex-wrap items-center gap-2">
          {loading && <span className="text-xs text-ink-3">updating…</span>}
          {!res.in_data_range && <LowConfidence />}
          <Badge v={v} />
        </div>
      </div>

      <p className="mt-3 text-xl font-semibold leading-snug text-ink">{headline}.</p>
      <p className="mt-1 text-sm text-ink-2">
        At {env.o2}% oxygen, {env.pressure_kpa} kPa ({env.pressure_psi} psi) and {env.flow_cm_s} cm/s airflow, in microgravity.
      </p>

      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        <div className="rounded-xl bg-surface-2 p-3">
          <div className="text-xs text-ink-3">Chance it keeps burning</div>
          <div className="text-2xl font-semibold tabular-nums text-ink">{(burn * 100).toFixed(0)}%</div>
          <div className="text-[11px] text-ink-3">
            range {((1 - res.extinction_ci[1]) * 100).toFixed(0)}–{((1 - res.extinction_ci[0]) * 100).toFixed(0)}%
          </div>
        </div>
        <div className="rounded-xl bg-surface-2 p-3">
          <div className="text-xs text-ink-3">Flame spread speed</div>
          <div className="text-2xl font-semibold tabular-nums text-ink">
            {res.spread_rate_mm_s.toFixed(2)} <span className="text-sm text-ink-3">mm/s</span>
          </div>
          <div className="text-[11px] text-ink-3">≈ {cmPerMin.toFixed(1)} cm per minute if it burns</div>
        </div>
        <div className="rounded-xl bg-surface-2 p-3">
          <div className="text-xs text-ink-3">Same material in ISS air</div>
          <div className="text-2xl font-semibold tabular-nums text-ink">{issBurn != null ? `${(issBurn * 100).toFixed(0)}%` : "—"}</div>
          <div className="text-[11px] text-ink-3">21% O₂ · 101.3 kPa · same airflow</div>
        </div>
      </div>

      {res.regime && (
        <p className="mt-3 text-sm text-ink-2">
          <span className="text-ink">Why: </span>
          {REGIME_TEXT[res.regime] ?? res.regime}
          {res.confidence && <span className="text-ink-3"> Model confidence: {res.confidence}.</span>}
        </p>
      )}

      {!res.in_data_range && (
        <div className="mt-3 rounded-lg border border-warn/50 bg-warn/10 p-3 text-sm text-ink">
          No NASA tests are close to these conditions — treat this as a hypothesis, and a candidate for the next experiment.
        </div>
      )}

      <div className="mt-4 border-t border-line pt-4">
        <div className="text-sm font-semibold text-ink">{v === "safe" ? "It already self-extinguishes. Margins:" : "What would make it go out on its own (≥90% chance)"}</div>
        <ul className="mt-2 space-y-2 text-sm text-ink-2">
          {[o2Lever, flowLever].map((l) => {
            const t = leverText(l);
            return t ? (
              <li key={l.kind} className="flex gap-2">
                <span className="mt-0.5 text-fire-soft">→</span>
                <span>{t}</span>
              </li>
            ) : null;
          })}
        </ul>
        <p className="mt-3 text-[11px] text-ink-3">
          Model estimates from scanning one setting at a time — research insight, not a safety certification. Settings to investigate experimentally.
        </p>
      </div>
    </section>
  );
}
