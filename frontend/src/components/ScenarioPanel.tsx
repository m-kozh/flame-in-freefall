"use client";
import { ENVIRONMENTS, KPA_PER_PSI, customEnv, type Environment } from "@/data/environments";
import type { Material } from "@/lib/types";

function Field({
  label,
  unit,
  value,
  min,
  max,
  step,
  onChange,
  hint,
}: {
  label: string;
  unit: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  hint?: string;
}) {
  return (
    <label className="block">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-sm text-ink-2">{label}</span>
        <span className="flex items-baseline gap-1">
          <input
            type="number"
            inputMode="decimal"
            className="w-20 rounded-md border border-line bg-surface-2 px-2 py-0.5 text-right font-mono text-sm text-ink"
            value={value}
            min={min}
            max={max}
            step={step}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              if (Number.isFinite(v)) onChange(Math.min(max, Math.max(min, v)));
            }}
          />
          <span className="w-10 text-xs text-ink-3">{unit}</span>
        </span>
      </div>
      <input type="range" className="w-full" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
      {hint && <div className="mt-0.5 text-[11px] text-ink-3">{hint}</div>}
    </label>
  );
}

export default function ScenarioPanel({
  env,
  setEnv,
  material,
  setMaterial,
  materials,
}: {
  env: Environment;
  setEnv: (e: Environment) => void;
  material: string;
  setMaterial: (m: string) => void;
  materials: Material[];
}) {
  const edit = (patch: Partial<Pick<Environment, "o2" | "pressure_kpa" | "flow_cm_s">>) =>
    setEnv(customEnv(patch.o2 ?? env.o2, patch.pressure_kpa ?? env.pressure_kpa, patch.flow_cm_s ?? env.flow_cm_s));

  return (
    <section className="rounded-2xl border border-line bg-surface p-4 sm:p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-semibold text-ink">Your scenario</h2>
        <span className="text-xs text-ink-3">Pick a habitat to start, then change any value</span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
        {ENVIRONMENTS.map((e) => (
          <button
            key={e.id}
            onClick={() => setEnv(e)}
            className={`rounded-xl border p-3 text-left transition ${e.id === env.id ? "border-fire bg-fire/10" : "border-line bg-surface-2 hover:border-ink-3"}`}
          >
            <div className="text-sm font-semibold text-ink">{e.name}</div>
            <div className="mt-0.5 font-mono text-xs text-ink-2">
              {e.o2}% O₂ · {e.pressure_psi} psi
            </div>
          </button>
        ))}
      </div>

      <div className="mt-5 grid gap-5 md:grid-cols-4">
        <label className="block">
          <div className="mb-1 text-sm text-ink-2">Material that catches fire</div>
          <select className="w-full rounded-lg border border-line bg-surface-2 px-3 py-2 text-sm text-ink" value={material} onChange={(e) => setMaterial(e.target.value)}>
            {materials.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
          <div className="mt-0.5 text-[11px] text-ink-3">{materials.find((m) => m.id === material)?.thickness_mm ? `${materials.find((m) => m.id === material)?.thickness_mm} mm sample` : " "}</div>
        </label>
        <Field label="Oxygen" unit="%" value={env.o2} min={5} max={100} step={0.5} onChange={(o2) => edit({ o2 })} hint="Earth air = 21%" />
        <Field
          label="Cabin pressure"
          unit="kPa"
          value={env.pressure_kpa}
          min={10}
          max={150}
          step={0.5}
          onChange={(pressure_kpa) => edit({ pressure_kpa })}
          hint={`= ${(env.pressure_kpa / KPA_PER_PSI).toFixed(1)} psi · sea level = 101.3 kPa`}
        />
        <Field label="Airflow over the material" unit="cm/s" value={env.flow_cm_s} min={0} max={100} step={0.5} onChange={(flow_cm_s) => edit({ flow_cm_s })} hint="Cabin fans ≈ 5–20 cm/s" />
      </div>
      <p className="mt-3 text-xs text-ink-3">
        {env.note} Gravity: {env.gravity}.
      </p>
    </section>
  );
}
