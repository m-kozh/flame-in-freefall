"use client";
import Plot, { axis, darkLayout } from "./Plot";
import type { Annotation as Annotations, Data, Shape } from "plotly.js";
import type { MapRes, SuggestPoint, TestedPoint } from "@/lib/types";
import type { Environment } from "@/data/environments";

// Sequential single-hue ramps (dark surface → bright = more).
const BURN_SCALE: [number, string][] = [
  [0, "#1a1a19"],
  [0.25, "#4a2614"],
  [0.5, "#8a3b17"],
  [0.75, "#c24f20"],
  [1, "#e8703a"],
];
const UNC_SCALE: [number, string][] = [
  [0, "#1a1a19"],
  [0.5, "#55544f"],
  [1, "#c3c2b7"],
];

const isNasa = (p: TestedPoint) => p.source.toLowerCase() === "nasa";
const label = (p: TestedPoint) =>
  isNasa(p) ? `NASA ${p.campaign ?? ""} ${p.experiment_id ?? ""}`.trim() : p.experiment_id ? `${p.source} (${p.experiment_id})` : p.source;

export default function FlammabilityMap({
  map,
  env,
  suggestions,
  view,
}: {
  map: MapRes;
  env: Environment;
  suggestions: SuggestPoint[];
  view: "burn" | "uncertainty";
}) {
  const logY = map.flow_scale === "log";
  const burn = map.z.map((row) => row.map((v) => +(1 - v).toFixed(3)));
  const uncMax = Math.max(0.05, ...map.uncertainty.flat());
  const xMin = map.x[0];
  const xMax = map.x[map.x.length - 1];
  const onChart = env.o2 <= xMax;
  const nearSuggest = suggestions.filter((s) => Math.abs(s.pressure_kpa - env.pressure_kpa) < 15);

  // On a log axis, extend the range to include suggestions below the grid (e.g. 0.1 cm/s)
  const ys = [map.y[0], ...nearSuggest.map((s) => s.flow_cm_s), ...map.tested_points.map((p) => p.flow_cm_s)].filter((v) => !logY || v > 0);
  const yLo = Math.min(...ys);
  const yHi = map.y[map.y.length - 1];
  const yRange = logY ? [Math.log10(yLo * 0.9), Math.log10(yHi)] : [map.y[0], yHi];

  const groups: { name: string; pts: TestedPoint[]; marker: Record<string, unknown> }[] = [
    { name: "NASA test — burned", pts: map.tested_points.filter((p) => isNasa(p) && p.burned), marker: { symbol: "diamond", size: 12, color: "#3987e5", line: { color: "#f0efec", width: 2 } } },
    { name: "NASA test — went out", pts: map.tested_points.filter((p) => isNasa(p) && !p.burned), marker: { symbol: "diamond-open", size: 12, color: "#3987e5", line: { color: "#3987e5", width: 3 } } },
    { name: "Other test — burned", pts: map.tested_points.filter((p) => !isNasa(p) && p.burned), marker: { symbol: "circle", size: 7, color: "#f0efec", opacity: 0.85, line: { color: "#111110", width: 1.5 } } },
    { name: "Other test — went out (×)", pts: map.tested_points.filter((p) => !isNasa(p) && !p.burned), marker: { symbol: "x", size: 7, color: "#f0efec", opacity: 0.85, line: { color: "#111110", width: 1 } } },
  ];

  const data: Data[] = [
    view === "burn"
      ? {
          type: "heatmap",
          x: map.x,
          y: map.y,
          z: burn,
          zmin: 0,
          zmax: 1,
          colorscale: BURN_SCALE,
          colorbar: { title: { text: "Burn<br>probability", side: "top" }, thickness: 10, outlinewidth: 0, tickformat: ".0%" },
          hovertemplate: "O₂ %{x:.1f}%<br>Airflow %{y:.2f} cm/s<br>Burn probability %{z:.0%}<extra></extra>",
          showlegend: false,
        }
      : {
          type: "heatmap",
          x: map.x,
          y: map.y,
          z: map.uncertainty,
          zmin: 0,
          zmax: uncMax,
          colorscale: UNC_SCALE,
          colorbar: { title: { text: "Model<br>disagreement", side: "top" }, thickness: 10, outlinewidth: 0 },
          hovertemplate: "O₂ %{x:.1f}%<br>Airflow %{y:.2f} cm/s<br>Disagreement %{z:.2f}<extra></extra>",
          showlegend: false,
        },
  ];

  if (map.physics_boundary?.length) {
    data.push({
      type: "scatter",
      mode: "lines",
      name: "Physics-only boundary",
      x: map.physics_boundary.map((b) => b.o2_percent),
      y: map.physics_boundary.map((b) => b.flow_cm_s),
      line: { color: "#8d8c84", width: 2, dash: "dot" },
      hovertemplate: "Physics-only limit<br>%{x:.1f}% O₂ at %{y:.2f} cm/s<extra></extra>",
    });
  }
  if (map.boundary?.length) {
    data.push({
      type: "scatter",
      mode: "lines",
      name: "Model boundary (50% burn)",
      x: map.boundary.map((b) => b.o2_percent),
      y: map.boundary.map((b) => b.flow_cm_s),
      line: { color: "#f0efec", width: 2 },
      hovertemplate: "Model 50% limit<br>%{x:.1f}% O₂ at %{y:.2f} cm/s<extra></extra>",
    });
  }

  for (const g of groups) {
    if (!g.pts.length) continue;
    data.push({
      type: "scatter",
      mode: "markers",
      name: g.name,
      x: g.pts.map((p) => p.o2),
      y: g.pts.map((p) => p.flow_cm_s),
      text: g.pts.map(label),
      marker: g.marker as never,
      hovertemplate: `${g.name}<br>O₂ %{x:.1f}%, %{y:.2f} cm/s<br>%{text}<extra></extra>`,
    });
  }

  if (nearSuggest.length) {
    data.push({
      type: "scatter",
      mode: "markers",
      name: "Suggested next test",
      x: nearSuggest.map((s) => s.o2),
      y: nearSuggest.map((s) => s.flow_cm_s),
      text: nearSuggest.map((s) => `${s.pressure_kpa} kPa${s.score != null ? ` · priority ${(s.score * 100).toFixed(0)}%` : ""}`),
      marker: { symbol: "star", size: 16, color: "#fab219", line: { color: "#111110", width: 1.5 } },
      hovertemplate: "Suggested next experiment<br>O₂ %{x:.1f}%, %{y:.2f} cm/s<br>%{text}<extra></extra>",
    });
  }

  if (onChart) {
    data.push({
      type: "scatter",
      mode: "markers",
      name: `${env.name} cabin (${env.flow_cm_s} cm/s)`,
      x: [env.o2],
      y: [env.flow_cm_s],
      marker: { symbol: "cross-thin", size: 18, color: "#f0efec", line: { color: "#f0efec", width: 3 } },
      hovertemplate: `${env.name}<br>${env.o2}% O₂ · ${env.pressure_kpa} kPa · ${env.flow_cm_s} cm/s<extra></extra>`,
    });
  }

  const shapes: Partial<Shape>[] = onChart
    ? [
        { type: "line", x0: env.o2, x1: env.o2, yref: "paper", y0: 0, y1: 1, line: { color: "#f0efec", width: 1.5, dash: "dash" } },
        { type: "line", x0: 21, x1: 21, yref: "paper", y0: 0, y1: 1, line: { color: "#8d8c84", width: 1, dash: "dot" } },
      ]
    : [];
  const annotations: Partial<Annotations>[] = onChart
    ? [
        { x: env.o2, yref: "paper", y: 1.0, text: env.name, showarrow: false, xanchor: "left", xshift: 4, yanchor: "top", font: { color: "#f0efec", size: 12 } },
        ...(env.o2 !== 21
          ? [{ x: 21, yref: "paper" as const, y: 1.0, text: "ISS (21%)", showarrow: false, xanchor: "right" as const, xshift: -4, yanchor: "top" as const, font: { color: "#8d8c84", size: 11 } }]
          : []),
      ]
    : [];

  return (
    <div className="h-[560px] w-full">
      <Plot
        data={data}
        layout={{
          ...darkLayout,
          autosize: true,
          xaxis: { ...axis, title: { text: "Oxygen concentration (% by volume)" }, range: [xMin, xMax] },
          yaxis: { ...axis, type: logY ? "log" : "linear", title: { text: `Airflow speed (cm/s${logY ? ", log scale" : ""})` }, range: yRange },
          shapes,
          annotations,
          legend: { orientation: "h", x: 0, y: -0.16, font: { size: 11 } },
          margin: { ...darkLayout.margin, b: 110 },
        }}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
