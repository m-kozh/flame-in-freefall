"use client";
import dynamic from "next/dynamic";
import type { PlotParams } from "react-plotly.js";

// Plotly touches `window`, so load it client-side only.
const Plot = dynamic<PlotParams>(
  async () => {
    const Plotly = (await import("plotly.js-dist-min")).default;
    const createPlotlyComponent = (await import("react-plotly.js/factory")).default;
    return createPlotlyComponent(Plotly);
  },
  { ssr: false, loading: () => <div className="h-full w-full animate-pulse rounded-lg bg-surface-2" /> },
);

export default Plot;

export const darkLayout = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#c3c2b7", family: "ui-sans-serif, system-ui, sans-serif", size: 12 },
  margin: { l: 56, r: 16, t: 16, b: 48 },
  hoverlabel: { bgcolor: "#232321", bordercolor: "#2f2f2c", font: { color: "#f0efec" } },
};

export const axis = {
  gridcolor: "#2a2a28",
  zeroline: false,
  linecolor: "#2f2f2c",
  tickcolor: "#2f2f2c",
};
