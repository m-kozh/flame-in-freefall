"use client";
import { useMemo, useState } from "react";
import data from "@/data/investigations.json";
import type { Environment } from "@/data/environments";

export type Investigation = {
  id: string;
  name: string;
  psi_id: string | null;
  category: string[];
  hardware: string;
  fuel: string;
  o2_pct: number[] | null;
  pressure_kpa: number[] | null;
  flow_cm_s: number[] | null;
  key_findings: string[];
  habitat_relevance: string[];
  source_url: string;
  note?: string;
};

export const INVESTIGATIONS = data as Investigation[];

// Simple relevance score: does the test envelope reach the habitat's O2 / pressure,
// and is the investigation tagged for that habitat?
export function relevance(inv: Investigation, env: Environment) {
  let score = 0;
  const why: string[] = [];
  if (inv.habitat_relevance.includes(env.id)) {
    score += 3;
    why.push(`tagged for ${env.name}`);
  }
  if (inv.o2_pct) {
    const [lo, hi] = inv.o2_pct;
    if (env.o2 >= lo && env.o2 <= hi) {
      score += 3;
      why.push(`tested at ${env.o2}% O₂`);
    } else {
      const gap = env.o2 > hi ? env.o2 - hi : lo - env.o2;
      score += Math.max(0, 2 - gap / 5);
      if (gap <= 5) why.push(`tested within ${gap.toFixed(1)}% O₂`);
    }
  }
  if (inv.pressure_kpa) {
    const [lo, hi] = inv.pressure_kpa;
    if (env.pressure_kpa >= lo - 5 && env.pressure_kpa <= hi + 5) {
      score += 1.5;
      why.push("pressure covered");
    }
  }
  return { score, why };
}

const CATS = ["all", "prevention", "detection", "suppression"] as const;
const CAT_STYLE: Record<string, string> = {
  prevention: "border-fire/60 text-fire-soft",
  detection: "border-accent/60 text-accent",
  suppression: "border-ink-3 text-ink-2",
};

const range = (r: number[] | null, unit: string) => (r ? (r[0] === r[1] ? `${r[0]} ${unit}` : `${r[0]}–${r[1]} ${unit}`) : "—");

export default function Explorer({ env }: { env: Environment }) {
  const [cat, setCat] = useState<(typeof CATS)[number]>("all");
  const [q, setQ] = useState("");

  const list = useMemo(() => {
    return INVESTIGATIONS.map((inv) => ({ inv, ...relevance(inv, env) }))
      .filter(({ inv }) => cat === "all" || inv.category.includes(cat))
      .filter(({ inv }) => !q || (inv.name + inv.fuel + inv.key_findings.join(" ")).toLowerCase().includes(q.toLowerCase()))
      .sort((a, b) => b.score - a.score);
  }, [env, cat, q]);

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {CATS.map((c) => (
          <button
            key={c}
            onClick={() => setCat(c)}
            className={`rounded-full border px-3 py-1 text-sm capitalize ${cat === c ? "border-ink bg-ink text-bg" : "border-line text-ink-2 hover:border-ink-3"}`}
          >
            {c}
          </button>
        ))}
        <input
          placeholder="Search fuel, finding…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          className="ml-auto w-full rounded-lg border border-line bg-surface-2 px-3 py-1.5 text-sm text-ink placeholder:text-ink-3 sm:w-64"
        />
      </div>
      <p className="mb-4 text-sm text-ink-3">
        Ranked by relevance to <span className="text-ink">{env.name}</span> ({env.o2}% O₂, {env.pressure_kpa} kPa). {list.length} investigations.
      </p>
      <div className="grid gap-4 md:grid-cols-2">
        {list.map(({ inv, score, why }, i) => (
          <article key={inv.id} className="flex flex-col rounded-xl border border-line bg-surface p-5">
            <div className="flex items-start justify-between gap-3">
              <h3 className="text-base font-semibold leading-snug text-ink">
                <span className="mr-2 text-ink-3">#{i + 1}</span>
                {inv.name}
              </h3>
              <span className="shrink-0 rounded-md bg-surface-2 px-2 py-0.5 font-mono text-xs text-ink-2" title="Relevance score">
                {score.toFixed(1)}
              </span>
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {inv.category.map((c) => (
                <span key={c} className={`rounded-full border px-2 py-0.5 text-xs capitalize ${CAT_STYLE[c]}`}>
                  {c}
                </span>
              ))}
              {inv.psi_id && <span className="rounded-full border border-line px-2 py-0.5 font-mono text-xs text-ink-2">{inv.psi_id}</span>}
            </div>
            <dl className="mt-3 grid grid-cols-3 gap-2 text-xs">
              <div>
                <dt className="text-ink-3">O₂</dt>
                <dd className="text-ink">{range(inv.o2_pct, "%")}</dd>
              </div>
              <div>
                <dt className="text-ink-3">Pressure</dt>
                <dd className="text-ink">{range(inv.pressure_kpa, "kPa")}</dd>
              </div>
              <div>
                <dt className="text-ink-3">Airflow</dt>
                <dd className="text-ink">{range(inv.flow_cm_s, "cm/s")}</dd>
              </div>
            </dl>
            <p className="mt-3 text-xs text-ink-3">
              <span className="text-ink-2">Fuel:</span> {inv.fuel}
            </p>
            <ul className="mt-3 list-disc space-y-1 pl-4 text-sm text-ink-2">
              {inv.key_findings.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
            {why.length > 0 && <p className="mt-3 text-xs text-ink-3">Why ranked: {why.join(" · ")}</p>}
            {inv.note && <p className="mt-2 text-xs text-warn">{inv.note}</p>}
            <div className="mt-auto flex gap-4 pt-4 text-sm">
              <a className="text-accent hover:underline" href={inv.source_url} target="_blank" rel="noreferrer">
                Source ↗
              </a>
              <a className="text-accent hover:underline" href={`https://psi.nasa.gov/physci/repo/search?q=${encodeURIComponent(inv.name.split("(")[0].trim())}`} target="_blank" rel="noreferrer">
                Find in PSI ↗
              </a>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
