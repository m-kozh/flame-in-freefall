"use client";
import { useEffect, useState } from "react";
import { predict } from "@/lib/api";
import type { PredictRes } from "@/lib/types";
import type { Material } from "@/lib/types";
import { ENVIRONMENTS, type Environment } from "@/data/environments";

type Verdict = "safe" | "risky" | "dangerous";

export function verdictOf(r: PredictRes): Verdict {
  const burn = 1 - r.extinction_prob;
  if (burn >= 0.7) return "dangerous";
  if (burn <= 0.3) return "safe";
  return "risky";
}

export const VERDICT_STYLE: Record<Verdict, { label: string; icon: string; color: string }> = {
  safe: { label: "Self-extinguishes", icon: "✓", color: "var(--color-ok)" },
  risky: { label: "Borderline", icon: "!", color: "var(--color-warn)" },
  dangerous: { label: "Sustains fire", icon: "✕", color: "var(--color-crit)" },
};

export function LowConfidence() {
  return (
    <span className="shrink-0 whitespace-nowrap rounded-full border border-dashed border-ink-3 px-2 py-0.5 text-[11px] text-ink-2" title="No NASA tests near these conditions — model is extrapolating">
      ? low confidence
    </span>
  );
}

export function Badge({ v }: { v: Verdict }) {
  const s = VERDICT_STYLE[v];
  return (
    <span className="inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border px-2 py-0.5 text-xs font-medium text-ink" style={{ borderColor: s.color }}>
      <span className="grid h-4 w-4 place-items-center rounded-full text-[10px] font-bold text-bg" style={{ background: s.color }}>
        {s.icon}
      </span>
      {s.label}
    </span>
  );
}

const ISS = ENVIRONMENTS[0];

export default function Verdicts({ env, materials }: { env: Environment; materials: Material[] }) {
  const [rows, setRows] = useState<{ id: string; name: string; sub: string; iss: PredictRes; here: PredictRes }[]>([]);

  useEffect(() => {
    let alive = true;
    Promise.all(
      materials.map(async (m) => ({
        id: m.id,
        name: m.name,
        sub: m.seen_in ? `Tested in ${m.seen_in}` : m.thickness_mm ? `${m.thickness_mm} mm sample` : "",
        iss: await predict({ o2: ISS.o2, pressure_kpa: ISS.pressure_kpa, flow_cm_s: ISS.flow_cm_s, material: m.id }),
        here: await predict({ o2: env.o2, pressure_kpa: env.pressure_kpa, flow_cm_s: env.flow_cm_s, material: m.id }),
      })),
    ).then((r) => alive && setRows(r));
    return () => {
      alive = false;
    };
  }, [env, materials]);

  return (
    <div className="divide-y divide-line">
      {rows.map((r) => {
        const before = verdictOf(r.iss);
        const after = verdictOf(r.here);
        const flipped = env.id !== "iss" && before !== after;
        return (
          <div key={r.id} className="flex flex-wrap items-center gap-x-4 gap-y-2 py-3">
            <div className="min-w-[180px] flex-1">
              <div className="text-sm font-medium text-ink">{r.name}</div>
              {r.sub && <div className="text-xs text-ink-3">{r.sub}</div>}
            </div>
            {env.id !== "iss" && (
              <>
                <div className="flex items-center gap-2 text-xs text-ink-3">
                  ISS <Badge v={before} />
                </div>
                <span className="text-ink-3">→</span>
              </>
            )}
            <div className="flex items-center gap-2 text-xs text-ink-3">
              {env.name} <Badge v={after} />
              {!r.here.in_data_range && <LowConfidence />}
            </div>
            <div className="w-full text-xs text-ink-2 sm:w-auto sm:text-right">
              <span>
                Burn {(100 * (1 - r.here.extinction_prob)).toFixed(0)}% · spread {r.here.spread_rate_mm_s.toFixed(2)} mm/s
              </span>
              {flipped && <span className="ml-2 font-semibold text-fire-soft">▲ became riskier</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}
