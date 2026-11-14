"""Atmosphere examples; all flow speeds are assumptions for demonstration.

最终 pitch 前请对照 NASA 资料核实这些数值。
NASA exploration-atmosphere context: https://ntrs.nasa.gov/citations/20230015380
NASA ISS context: https://www.nasa.gov/international-space-station/
These links supply context, not validation of all values or flow assumptions.
"""


PRESETS: list[dict] = [
    {
        "id": "iss_standard",
        "name": "ISS standard atmosphere",
        "o2_percent": 21.0,
        "pressure_kpa": 101.3,
        "flow_cm_s": 10.0,
        "illustrative": False,
        "note": "Approximately Earth sea-level air. Flow of 10 cm/s is a demonstration assumption, not a NASA cabin-flow specification. Verify before the final pitch: https://www.nasa.gov/international-space-station/",
    },
    {
        "id": "exploration_atmosphere",
        "name": "Proposed NASA exploration atmosphere",
        "o2_percent": 34.0,
        "pressure_kpa": 56.5,
        "flow_cm_s": 10.0,
        "illustrative": False,
        "note": "Proposed exploration-atmosphere example: approximately 34% oxygen at 8.2 psia (56.5 kPa). Flow of 10 cm/s is assumed. Verify before the final pitch: https://ntrs.nasa.gov/citations/20230015380",
    },
    {
        "id": "lunar_base_upper_bound",
        "name": "Illustrative lunar habitat atmosphere",
        "o2_percent": 30.0,
        "pressure_kpa": 70.3,
        "flow_cm_s": 10.0,
        "illustrative": True,
        "note": "Illustrative scenario only; this stable preset id is not a NASA safety limit or approved lunar habitat design. Oxygen, pressure and flow are demonstration choices. Verify before the final pitch.",
    },
    {
        "id": "spacesuit_pure_o2",
        "name": "Illustrative near-pure-oxygen suit scenario",
        "o2_percent": 100.0,
        "pressure_kpa": 29.6,
        "flow_cm_s": 5.0,
        "illustrative": True,
        "note": "Illustrative near-pure-oxygen, low-pressure scenario only; do not interpret it as a prebreathing protocol or suit safety assessment. Flow of 5 cm/s is assumed. Verify all values against NASA sources before the final pitch.",
    },
]
