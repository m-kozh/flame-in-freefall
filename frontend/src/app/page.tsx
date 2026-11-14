"use client";
import { useEffect, useState } from "react";
import FlammabilityMap from "@/components/FlammabilityMap";
import Verdicts from "@/components/Verdicts";
import PredictPanel from "@/components/PredictPanel";
import Explorer from "@/components/Explorer";
import Chat from "@/components/Chat";
import { ENVIRONMENTS } from "@/data/environments";
import { getMap, getSuggest, useBackendStatus, useDataSource, useMaterials, type BackendStatus } from "@/lib/api";
import type { MapRes, SuggestPoint } from "@/lib/types";

const ALL_TABS = [
  { id: "atmosphere", label: "Atmosphere shift" },
  { id: "predict", label: "Predict" },
  { id: "evidence", label: "Evidence (PSI)" },
  { id: "ask", label: "Ask AI" },
] as const;
// "Ask AI" only appears once ANTHROPIC_API_KEY is set on the deployment (then redeploy).
const CHAT_ENABLED = process.env.NEXT_PUBLIC_CHAT_ENABLED === "1";
const TABS = ALL_TABS.filter((t) => t.id !== "ask" || CHAT_ENABLED);

const STATUS: Record<BackendStatus, { label: string; cls: string; title: string }> = {
  mock: { label: "Mock model — demo data", cls: "border-warn/60 text-warn", title: "BACKEND_URL is not set" },
  connecting: { label: "Connecting to model…", cls: "border-ink-3 text-ink-2", title: "Waking the backend (can take ~30 s on free hosting)" },
  live: { label: "● Live model", cls: "border-ok/60 text-ok", title: "Connected to the prediction backend" },
  fallback: { label: "Model offline — showing demo data", cls: "border-crit/60 text-crit", title: "Backend request failed; using the local mock" },
};

export default function Home() {
  const materials = useMaterials();
  const status = useBackendStatus();
  const dataSource = useDataSource();
  const [envId, setEnvId] = useState("moon");
  const [tab, setTab] = useState<(typeof ALL_TABS)[number]["id"]>("atmosphere");
  const [material, setMaterial] = useState<string>("cotton");
  const [view, setView] = useState<"burn" | "uncertainty">("burn");
  const [map, setMap] = useState<MapRes | null>(null);
  const [suggest, setSuggest] = useState<SuggestPoint[]>([]);
  const env = ENVIRONMENTS.find((e) => e.id === envId)!;

  useEffect(() => {
    if (materials.length && !materials.some((m) => m.id === material)) setMaterial(materials[0].id);
  }, [materials, material]);

  useEffect(() => {
    let alive = true;
    getMap(material, env.pressure_kpa).then((m) => alive && setMap(m));
    getSuggest(material, env.pressure_kpa).then((s) => alive && setSuggest(s.points));
    return () => {
      alive = false;
    };
  }, [material, env.pressure_kpa]);

  return (
    <main className="mx-auto max-w-7xl px-4 pb-16 sm:px-6">
      <header className="flex flex-wrap items-center justify-between gap-4 py-6">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-full bg-fire shadow-[0_0_16px_4px_rgba(217,89,38,0.5)]" />
            <h1 className="text-xl font-semibold tracking-tight text-ink">Burn Boundary</h1>
          </div>
          <p className="mt-1 max-w-xl text-sm text-ink-2">
            Every Moon-base fire safety decision rests on ISS data taken in a different atmosphere. See where that data runs out — and what to test next.
          </p>
        </div>
        <span className={`rounded-full border px-3 py-1 text-xs ${STATUS[status].cls}`} title={STATUS[status].title}>
          {STATUS[status].label}
          {status === "live" && dataSource && (
            <span className="text-ink-2"> · trained on {dataSource === "mixed" ? "synthetic + NASA data" : dataSource === "nasa" ? "NASA data" : "synthetic data"}</span>
          )}
        </span>
      </header>

      {/* Environment switch */}
      <section className="rounded-2xl border border-line bg-surface p-4">
        <div className="mb-3 text-xs font-medium uppercase tracking-wider text-ink-3">Habitat atmosphere</div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
          {ENVIRONMENTS.map((e) => (
            <button
              key={e.id}
              onClick={() => setEnvId(e.id)}
              className={`rounded-xl border p-3 text-left transition ${e.id === envId ? "border-fire bg-fire/10" : "border-line bg-surface-2 hover:border-ink-3"}`}
            >
              <div className="text-sm font-semibold text-ink">{e.name}</div>
              <div className="mt-0.5 font-mono text-xs text-ink-2">
                {e.o2}% O₂ · {e.pressure_psi} psi
              </div>
            </button>
          ))}
        </div>
        <p className="mt-3 text-xs text-ink-3">
          {env.note} Gravity: {env.gravity}.
        </p>
      </section>

      <nav className="mt-6 flex gap-1 overflow-x-auto border-b border-line">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`whitespace-nowrap border-b-2 px-4 py-2.5 text-sm ${tab === t.id ? "border-fire text-ink" : "border-transparent text-ink-3 hover:text-ink-2"}`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className="mt-6">
        {tab === "atmosphere" && (
          <div className="grid gap-6 xl:grid-cols-[1.4fr_1fr]">
            <section className="rounded-2xl border border-line bg-surface p-5">
              <div className="flex flex-wrap items-end justify-between gap-3">
                <div>
                  <h2 className="text-base font-semibold text-ink">Flammability boundary</h2>
                  <p className="text-xs text-ink-3">
                    {materials.find((m) => m.id === material)?.name ?? material} at {env.pressure_kpa} kPa
                  </p>
                </div>
                <div className="flex gap-2">
                  <select className="rounded-lg border border-line bg-surface-2 px-2 py-1.5 text-sm text-ink" value={material} onChange={(e) => setMaterial(e.target.value)}>
                    {materials.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                  <div className="flex rounded-lg border border-line p-0.5 text-sm">
                    {(["burn", "uncertainty"] as const).map((v) => (
                      <button key={v} onClick={() => setView(v)} className={`rounded-md px-2.5 py-1 capitalize ${view === v ? "bg-surface-2 text-ink" : "text-ink-3"}`}>
                        {v === "burn" ? "Burn risk" : "Model uncertainty"}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
              {env.o2 > 45 && (
                <div className="mt-3 rounded-lg border border-crit/50 bg-crit/10 p-3 text-sm text-ink">
                  {env.name} ({env.o2}% O₂) is off this chart — beyond every microgravity test of solid materials. Any prediction here is pure extrapolation.
                </div>
              )}
              <div className="mt-2">{map ? <FlammabilityMap map={map} env={env} suggestions={suggest} view={view} /> : <div className="h-[560px] animate-pulse rounded-lg bg-surface-2" />}</div>
            </section>

            <div className="space-y-6">
              <section className="rounded-2xl border border-line bg-surface p-5">
                <h2 className="text-base font-semibold text-ink">Material verdicts</h2>
                <p className="text-xs text-ink-3">At {env.flow_cm_s} cm/s cabin airflow{env.id !== "iss" ? ", compared with ISS air" : ""}</p>
                <div className="mt-2">
                  <Verdicts env={env} materials={materials} />
                </div>
              </section>

              <section className="rounded-2xl border border-line bg-surface p-5">
                <h2 className="flex items-center gap-2 text-base font-semibold text-ink">
                  <span className="text-warn">★</span> Recommended next experiments
                </h2>
                <p className="text-xs text-ink-3">Where the model is least certain — the highest-value tests for ISS or LUCI</p>
                <ol className="mt-3 space-y-3">
                  {suggest.map((s, i) => (
                    <li key={i} className="rounded-lg bg-surface-2 p-3">
                      <div className="flex items-center justify-between font-mono text-sm text-ink">
                        <span>
                          {s.o2.toFixed(1)}% O₂ · {s.flow_cm_s < 1 ? s.flow_cm_s.toFixed(2) : s.flow_cm_s.toFixed(1)} cm/s · {s.pressure_kpa} kPa
                        </span>
                        <span className="text-xs text-ink-2">
                          {s.score != null ? `priority ${(s.score * 100).toFixed(0)}%` : `uncertainty ${(s.uncertainty * 100).toFixed(0)}%`}
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-ink-3">{s.reason}</p>
                    </li>
                  ))}
                </ol>
              </section>
            </div>
          </div>
        )}
        {tab === "predict" && <PredictPanel env={env} materials={materials} />}
        {tab === "evidence" && <Explorer env={env} />}
        {tab === "ask" && CHAT_ENABLED && <Chat env={env} />}
      </div>

      <footer className="mt-12 border-t border-line pt-4 text-xs text-ink-3">
        Data: NASA Physical Sciences Informatics (PSI) combustion investigations and linked publications. Built for NASA Space Apps 2026 — Flame in Freefall. Not affiliated with or endorsed by NASA.
      </footer>
    </main>
  );
}
