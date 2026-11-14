export type Environment = {
  id: string;
  name: string;
  o2: number; // % by volume
  pressure_kpa: number;
  pressure_psi: number;
  gravity: string;
  flow_cm_s: number; // assumed typical cabin ventilation for verdicts
  note: string;
};

// Values from the challenge brief unless noted.
export const ENVIRONMENTS: Environment[] = [
  {
    id: "iss",
    name: "ISS",
    o2: 21,
    pressure_kpa: 101.3,
    pressure_psi: 14.7,
    gravity: "Microgravity",
    flow_cm_s: 10,
    note: "Earth-like air. Where almost all PSI combustion data was collected.",
  },
  {
    id: "moon",
    name: "Moon base",
    o2: 30,
    pressure_kpa: 70.3,
    pressure_psi: 10.2,
    gravity: "1/6 g (no PSI data)",
    flow_cm_s: 10,
    note: "Brief: likely ≥30% O₂ at ≤10.2 psi to shorten spacesuit pre-breathe — 40%+ more oxygen than Earth air.",
  },
  {
    id: "rover",
    name: "Pressurized rover",
    o2: 34,
    pressure_kpa: 56.5,
    pressure_psi: 8.2,
    gravity: "1/6 g (no PSI data)",
    flow_cm_s: 10,
    note: "NASA's proposed exploration atmosphere: ~34% O₂ at 8.2 psia (NTRS 20230015380).",
  },
  {
    id: "mars",
    name: "Mars habitat",
    o2: 30,
    pressure_kpa: 70.3,
    pressure_psi: 10.2,
    gravity: "0.38 g (no PSI data)",
    flow_cm_s: 10,
    note: "Assumed same atmosphere as the Moon base. No partial-gravity data exists in PSI yet.",
  },
  {
    id: "suit",
    name: "Spacesuit",
    o2: 100,
    pressure_kpa: 29.6,
    pressure_psi: 4.3,
    gravity: "Any",
    flow_cm_s: 10,
    note: "100% O₂ at 4.3 psi — far outside every tested condition in PSI.",
  },
];
