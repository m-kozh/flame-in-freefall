// Materials offered in the UI. `mock` params only drive the local mock model.
export const MATERIALS = [
  { id: "cotton", name: "Cotton fabric", seenIn: "Saffire IV–V", mock: { o2crit: 17, k: 1.0 } },
  { id: "sibal", name: "Cotton/fiberglass (SIBAL)", seenIn: "BASS, BASS-II, Saffire I–V", mock: { o2crit: 19, k: 0.7 } },
  { id: "pmma", name: "PMMA (acrylic)", seenIn: "BASS-II, Saffire IV–VI", mock: { o2crit: 19.5, k: 0.4 } },
  { id: "nomex", name: "Nomex fabric", seenIn: "Saffire V", mock: { o2crit: 25, k: 0.5 } },
] as const;

export type MaterialId = (typeof MATERIALS)[number]["id"];
