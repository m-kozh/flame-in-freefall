// Talks to the backend through the /backend proxy (see next.config.ts).
// If BACKEND_URL isn't set, or the backend fails, it falls back to the local mock
// so the site never shows a broken screen. The header badge shows which is in use.
"use client";
import { useEffect, useState } from "react";
import type { DataSource, ExplainRes, ExplainTarget, MapRes, Material, MaterialsRes, PredictReq, PredictRes, SuggestRes } from "./types";
import { mockExplain, mockMap, mockPredict, mockSuggest } from "./mock";
import { MATERIALS } from "@/data/materials";

export const BACKEND_ENABLED = process.env.NEXT_PUBLIC_BACKEND_ENABLED === "1";
const BASE = "/backend";
const TIMEOUT_MS = 45_000; // free-tier hosts (Render) can take ~30–50 s to wake up

// ---- status shared with the header badge ----
export type BackendStatus = "mock" | "connecting" | "live" | "fallback";
let status: BackendStatus = BACKEND_ENABLED ? "connecting" : "mock";
const listeners = new Set<(s: BackendStatus) => void>();
function setStatus(s: BackendStatus) {
  if (s === status) return;
  status = s;
  listeners.forEach((l) => l(s));
}
// Which data the live model was trained on (backend sends data_source on every response)
let dataSource: DataSource | null = null;
const dsListeners = new Set<(d: DataSource | null) => void>();
export function useDataSource() {
  const [d, set] = useState(dataSource);
  useEffect(() => {
    dsListeners.add(set);
    set(dataSource);
    return () => {
      dsListeners.delete(set);
    };
  }, []);
  return d;
}

export function useBackendStatus() {
  const [s, set] = useState(status);
  useEffect(() => {
    listeners.add(set);
    set(status);
    return () => {
      listeners.delete(set);
    };
  }, []);
  return s;
}

async function call<T>(path: string, init: RequestInit | undefined, fallback: () => T): Promise<T> {
  if (!BACKEND_ENABLED) return fallback();
  try {
    const r = await fetch(`${BASE}${path}`, { ...init, signal: AbortSignal.timeout(TIMEOUT_MS) });
    if (!r.ok) throw new Error(`${path} → HTTP ${r.status}`);
    const data = (await r.json()) as T;
    const ds = (data as { data_source?: DataSource })?.data_source;
    if (ds && ds !== dataSource) {
      dataSource = ds;
      dsListeners.forEach((l) => l(ds));
    }
    setStatus("live");
    return data;
  } catch (e) {
    console.warn("[backend] falling back to mock:", e);
    setStatus("fallback");
    return fallback();
  }
}

const enc = encodeURIComponent;

export const predict = (req: PredictReq) =>
  call<PredictRes>("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req) }, () => mockPredict(req));

export const getMap = (material: string, pressure_kpa: number) =>
  call<MapRes>(`/map?material=${enc(material)}&pressure_kpa=${pressure_kpa}&o2_min=12&o2_max=45&flow_min=0.1&flow_max=100`, undefined, () => mockMap(material, pressure_kpa));

export const getSuggest = (material: string, pressure_kpa: number) =>
  call<SuggestRes>(`/suggest?material=${enc(material)}&pressure_kpa=${pressure_kpa}&k=4`, undefined, () => mockSuggest(material));

export const getExplain = (target: ExplainTarget = "extinction") =>
  call<ExplainRes>(`/explain?target=${target}`, undefined, () => mockExplain());

const LOCAL_MATERIALS: Material[] = MATERIALS.map(({ id, name, seenIn }) => ({ id, name, seen_in: seenIn }));
export const getMaterials = () => call<MaterialsRes>("/materials", undefined, () => ({ materials: LOCAL_MATERIALS }));

export function useMaterials() {
  const [m, setM] = useState<Material[]>(LOCAL_MATERIALS);
  useEffect(() => {
    getMaterials().then((r) => r.materials?.length && setM(r.materials));
  }, []);
  return m;
}
