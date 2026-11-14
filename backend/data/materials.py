"""Small material library for the hackathon demonstration.

Every property here is an assumed representative value. Real sample density,
thickness, thermal response and limiting oxygen concentration depend on test
conditions and manufacturing. These values are not certified material data.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Material:
    id: str
    name: str
    aliases: tuple[str, ...]
    density_kg_m3: float
    specific_heat_j_kgk: float
    thickness_mm: float
    vaporization_temp_k: float
    thermally_thin: bool
    limiting_o2_percent: float
    notes: str


_APPROXIMATE = "Approximate hackathon assumptions; verify against sample-specific measurements."

MATERIALS: dict[str, Material] = {
    "cellulose_thin": Material(
        id="cellulose_thin",
        name="Thin cellulose tissue",
        aliases=("cellulose", "paper", "tissue"),
        density_kg_m3=250.0,
        specific_heat_j_kgk=1400.0,
        thickness_mm=0.12,
        vaporization_temp_k=600.0,
        thermally_thin=True,
        limiting_o2_percent=14.5,
        notes=_APPROXIMATE + " Vaporization temperature represents an effective pyrolysis temperature.",
    ),
    "pmma_thin": Material(
        id="pmma_thin",
        name="Thin PMMA sheet",
        aliases=("pmma",),
        density_kg_m3=1180.0,
        specific_heat_j_kgk=1470.0,
        thickness_mm=0.25,
        vaporization_temp_k=650.0,
        thermally_thin=True,
        limiting_o2_percent=16.0,
        notes=_APPROXIMATE,
    ),
    "pmma_thick": Material(
        id="pmma_thick",
        name="Thick PMMA sheet",
        aliases=(),
        density_kg_m3=1180.0,
        specific_heat_j_kgk=1470.0,
        thickness_mm=3.0,
        vaporization_temp_k=650.0,
        thermally_thin=False,
        limiting_o2_percent=16.0,
        notes=_APPROXIMATE + " The thermally thin baseline is outside its intended validity for this sample.",
    ),
    "sibal_cotton_fiberglass": Material(
        id="sibal_cotton_fiberglass",
        name="SIBAL cotton-fiberglass fabric",
        aliases=("cotton", "sibal"),
        density_kg_m3=550.0,
        specific_heat_j_kgk=1300.0,
        thickness_mm=0.33,
        vaporization_temp_k=620.0,
        thermally_thin=True,
        limiting_o2_percent=14.0,
        notes=_APPROXIMATE + " Inspired by SIBAL-type fabric experiments; no blend ratio or measured property claim is made.",
    ),
    "nomex": Material(
        id="nomex",
        name="Nomex fabric",
        aliases=(),
        density_kg_m3=1380.0,
        specific_heat_j_kgk=1400.0,
        thickness_mm=0.25,
        vaporization_temp_k=720.0,
        thermally_thin=True,
        limiting_o2_percent=29.0,
        notes=_APPROXIMATE + " An effective pyrolysis temperature is used; decomposition is not vaporization.",
    ),
}

_MATERIAL_ALIASES = {
    name.casefold(): material.id
    for material in MATERIALS.values()
    for name in (material.id, *material.aliases)
}


def resolve_material(name: str) -> str:
    """Resolve a stable id or case-insensitive alias, or give a useful error."""
    if isinstance(name, str):
        canonical = _MATERIAL_ALIASES.get(name.strip().casefold())
        if canonical is not None:
            return canonical
    raise ValueError(
        f"Unknown material {name!r}. Valid ids: {', '.join(MATERIALS)}"
    )


def get_material(name: str) -> Material:
    return MATERIALS[resolve_material(name)]


def material_dicts() -> list[dict]:
    """Return fresh serialisable dictionaries without exposing mutable library data."""
    result = []
    for material in MATERIALS.values():
        item = asdict(material)
        item["aliases"] = list(material.aliases)
        result.append(item)
    return result
