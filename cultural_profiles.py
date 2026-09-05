"""[P1-ARQ25-F7-CULTURE · 2026-09-05] Perfiles culturales parametrizables (Fase 7 del roadmap 2.5, capa V2.4).

I16 — **Cultura separada del mercado**: `market_country` (H4) sigue mandando en precios, catálogo, moneda y
unidades; la CULTURA (platos, técnicas, sabores, horarios, inspiración) es un eje propio con seis perfiles
derivados de las seis bibliotecas existentes. El usuario elige una cocina principal y hasta dos secundarias
con intensidad (ocasional / frecuente / predominante); sin elección, la cocina del país de compra —sugerida,
nunca inferida como identidad— sigue siendo la de siempre (comportamiento byte-idéntico al de la Fase 6).

Motor genérico + presets (§9.1, «un `if/elif` por cultura» prohibido): cada preset es DATA (`PROFILES`).
`profile_for_day(weights, day_index)` reparte los días de forma determinista según los pesos (misma política
⇒ mismo reparto), así el blueprint, la inspiración del generador, los candidatos del registry y el juez
culinario hablan de la MISMA cocina para el mismo día.

Knob `MEALFIT_CULTURAL_PROFILES` (True). Apagado ⇒ cultura = país de compra (legacy).
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from knobs import _env_bool

logger = logging.getLogger(__name__)

INTENSITY_WEIGHT = {"ocasional": 0.15, "frecuente": 0.30, "predominante": 0.45}
MAX_SECONDARY = 2
MIN_MAIN_WEIGHT = 0.5

# Presets: identidad estable (`profile_id`) → datos. `library` = biblioteca de plantillas / registry
# (`dish_registry.LIBRARIES`), `market_default` = país cuya cocina representa (para el legado y los copys).
PROFILES: dict[str, dict] = {
    "dominican_criolla": {
        "name_es": "Cocina dominicana", "library": "do", "market_default": "DO",
        "staples": ["arroz", "habichuelas", "plátano", "yuca", "guineo verde", "casabe", "auyama", "batata"],
        "dish_families": ["locrio", "moro", "sancocho", "mangú", "guisado criollo", "pastelón", "tostones"],
        "techniques": ["guisado", "hervido+majado", "sofrito", "horneado", "plancha"],
        "flavor_base": ["ajo", "cebolla", "ají cubanela", "orégano dominicano", "cilantro", "sazón"],
        "slot_affinity": {"desayuno": ["mangú", "avena", "huevo", "casabe"], "almuerzo": ["arroz + habichuela + proteína"],
                          "cena": ["ligero", "víveres", "ensalada"], "merienda": ["fruta", "batida", "casabe"]},
        "main_meal": "almuerzo",
    },
    "puertorico_criolla": {
        "name_es": "Cocina puertorriqueña", "library": "pr", "market_default": "PR",
        "staples": ["arroz", "gandules", "plátano", "yuca", "pana", "habichuelas"],
        "dish_families": ["arroz con gandules", "mofongo", "asopao", "pastelón", "tostones", "pernil"],
        "techniques": ["guisado", "majado", "horneado", "frito ligero", "plancha"],
        "flavor_base": ["sofrito", "adobo", "sazón con culantro y achiote", "ajo", "recao"],
        "slot_affinity": {"desayuno": ["avena", "huevo", "pan"], "almuerzo": ["arroz + proteína"], "cena": ["ligero"], "merienda": ["fruta"]},
        "main_meal": "almuerzo",
    },
    "mexico_casera": {
        "name_es": "Cocina mexicana", "library": "mx", "market_default": "MX",
        "staples": ["tortilla de maíz", "frijoles", "arroz", "nopal", "chile", "aguacate"],
        "dish_families": ["tacos", "guisados", "caldos", "enfrijoladas", "chilaquiles", "pozole"],
        "techniques": ["guisado", "asado", "comal", "hervido", "horneado"],
        "flavor_base": ["chile", "cebolla", "ajo", "cilantro", "comino", "epazote", "limón"],
        "slot_affinity": {"desayuno": ["huevo", "frijoles", "tortilla", "avena"], "almuerzo": ["guisado + arroz + tortilla"], "cena": ["ligero", "sopa"], "merienda": ["fruta", "yogur"]},
        "main_meal": "almuerzo",
    },
    "colombia_casera": {
        "name_es": "Cocina colombiana", "library": "co", "market_default": "CO",
        "staples": ["arroz", "papa", "arepa", "frijol", "plátano", "yuca"],
        "dish_families": ["bandeja", "sancocho", "ajiaco", "arepa", "sudado", "lentejas"],
        "techniques": ["sudado", "guisado", "asado", "hervido", "plancha"],
        "flavor_base": ["hogao", "cebolla", "ajo", "comino", "cilantro", "guascas"],
        "slot_affinity": {"desayuno": ["arepa", "huevo", "caldo", "avena"], "almuerzo": ["sopa + seco"], "cena": ["ligero"], "merienda": ["fruta", "arepa"]},
        "main_meal": "almuerzo",
    },
    "spain_mediterranea": {
        "name_es": "Cocina española", "library": "es", "market_default": "ES",
        "staples": ["aceite de oliva", "pan", "patata", "legumbres", "arroz", "tomate"],
        "dish_families": ["cocido", "guiso", "tortilla", "ensalada", "pescado al horno", "revuelto"],
        "techniques": ["sofrito", "guisado", "plancha", "horneado", "sartén"],
        "flavor_base": ["aceite de oliva", "ajo", "pimentón", "cebolla", "laurel", "perejil"],
        "slot_affinity": {"desayuno": ["tostada", "huevo", "yogur", "fruta"], "almuerzo": ["plato principal + guarnición"], "cena": ["ligero", "tortilla", "ensalada"], "merienda": ["fruta", "pan"]},
        "main_meal": "almuerzo",
    },
    "us_everyday": {
        "name_es": "Cocina estadounidense cotidiana", "library": "us", "market_default": "US",
        "staples": ["pollo", "pan integral", "avena", "arroz", "papa", "vegetales", "huevo"],
        "dish_families": ["bowl", "sándwich", "ensalada", "asado", "sopa", "pasta"],
        "techniques": ["horneado", "plancha", "salteado", "asado", "frío"],
        "flavor_base": ["ajo", "cebolla", "pimienta", "hierbas", "limón", "mostaza"],
        "slot_affinity": {"desayuno": ["avena", "huevo", "yogur", "tostada"], "almuerzo": ["bowl", "sándwich", "ensalada"], "cena": ["proteína + vegetales"], "merienda": ["fruta", "yogur", "nueces"]},
        "main_meal": "cena",
    },
}
_PROFILE_BY_MARKET = {p["market_default"]: pid for pid, p in PROFILES.items()}
DEFAULT_PROFILE = "dominican_criolla"


def cultural_profiles_enabled() -> bool:
    return _env_bool("MEALFIT_CULTURAL_PROFILES", True)


def profile_ids() -> list[str]:
    return list(PROFILES.keys())


def is_profile(pid: Any) -> bool:
    return str(pid or "") in PROFILES


def profile_for_market(country: Any) -> str:
    try:
        from constants import canonicalize_country
        cc = canonicalize_country(country)
    except Exception:
        cc = str(country or "DO").upper()
    return _PROFILE_BY_MARKET.get(cc, DEFAULT_PROFILE)


def library_for_profile(pid: Any) -> str:
    return (PROFILES.get(str(pid or "")) or PROFILES[DEFAULT_PROFILE])["library"]


def country_for_profile(pid: Any) -> str:
    """País cuya biblioteca/registry representa la cocina (NO el país de compra)."""
    return (PROFILES.get(str(pid or "")) or PROFILES[DEFAULT_PROFILE])["market_default"]


def profile_name_es(pid: Any) -> str:
    return (PROFILES.get(str(pid or "")) or {}).get("name_es") or str(pid or "")


# ----------------------------------------------------------------------------- pesos
def normalize_weights(weights: Iterable[dict], *, default_profile: Optional[str] = None) -> list[dict]:
    """Mezcla válida: principal primero con ≥ 0.5, hasta 2 secundarias, suma 1.0, ids conocidos, sin duplicados.
    Puro; entrada inválida ⇒ `[{default, 1.0}]`."""
    fallback = [{"profile_id": default_profile or DEFAULT_PROFILE, "weight": 1.0}]
    clean: list[tuple[str, float]] = []
    seen: set[str] = set()
    for w in weights or []:
        if not isinstance(w, dict):
            continue
        pid = str(w.get("profile_id") or "")
        try:
            wt = float(w.get("weight"))
        except (TypeError, ValueError):
            continue
        if not is_profile(pid) or pid in seen or wt <= 0:
            continue
        seen.add(pid)
        clean.append((pid, wt))
    if not clean:
        return fallback
    clean.sort(key=lambda x: -x[1])
    clean = clean[: 1 + MAX_SECONDARY]
    total = sum(w for _, w in clean)
    clean = [(p, w / total) for p, w in clean]
    main_p, main_w = clean[0]
    if main_w < MIN_MAIN_WEIGHT:
        rest = 1.0 - MIN_MAIN_WEIGHT
        others = clean[1:]
        others_total = sum(w for _, w in others) or 1.0
        clean = [(main_p, MIN_MAIN_WEIGHT)] + [(p, rest * w / others_total) for p, w in others]
    out = [{"profile_id": p, "weight": round(w, 4)} for p, w in clean]
    drift = round(1.0 - sum(o["weight"] for o in out), 4)
    out[0]["weight"] = round(out[0]["weight"] + drift, 4)
    return out


def weights_from_form_field(value: Any) -> Optional[list[dict]]:
    """`cultureProfiles` del formulario: `{"main": id, "secondary": [{"profile_id": id, "intensity": ocasional|frecuente|predominante}]}`
    o una lista `[{"profile_id","weight"}]`. None si no hay elección."""
    if not value:
        return None
    if isinstance(value, list):
        ws = normalize_weights(value)
        return ws if ws and (len(ws) > 1 or ws[0]["weight"] == 1.0) else None
    if not isinstance(value, dict):
        return None
    main = str(value.get("main") or "")
    if not is_profile(main):
        return None
    secondary = []
    for s in (value.get("secondary") or [])[:MAX_SECONDARY]:
        if not isinstance(s, dict):
            continue
        pid = str(s.get("profile_id") or "")
        if not is_profile(pid) or pid == main:
            continue
        secondary.append((pid, INTENSITY_WEIGHT.get(str(s.get("intensity") or "ocasional").lower(), INTENSITY_WEIGHT["ocasional"])))
    sec_total = min(0.5, sum(w for _, w in secondary))
    scale = (sec_total / sum(w for _, w in secondary)) if secondary and sum(w for _, w in secondary) > sec_total else 1.0
    raw = [{"profile_id": main, "weight": 1.0 - sec_total}] + [{"profile_id": p, "weight": w * scale} for p, w in secondary]
    return normalize_weights(raw, default_profile=main)


def culture_weights_for_form(form_data: Optional[dict]) -> list[dict]:
    """Pesos de cocina para un formulario: elección explícita si existe; si no, la cocina del país de compra
    (sugerida). Con el knob apagado, siempre la del país de compra."""
    form = form_data if isinstance(form_data, dict) else {}
    try:
        from constants import country_for_form_data as _market_gate
        _market = _market_gate(form)  # respeta el knob de países: apagado ⇒ DO, sin mirar el campo crudo
    except Exception:
        _market = form.get("country")
    market_profile = profile_for_market(_market)
    if not cultural_profiles_enabled():
        return [{"profile_id": market_profile, "weight": 1.0}]
    chosen = weights_from_form_field(form.get("cultureProfiles"))
    return chosen or [{"profile_id": market_profile, "weight": 1.0}]


def main_profile_id(weights: Optional[Iterable[dict]]) -> str:
    ws = normalize_weights(weights or [])
    return ws[0]["profile_id"]


def profile_for_day(weights: Optional[Iterable[dict]], day_index: int) -> str:
    """Reparto determinista por día según pesos (secuencia de menor discrepancia): con 0.7/0.3 los días
    caen 7/3 de cada 10, siempre en el mismo orden. Un solo perfil ⇒ ese perfil siempre."""
    ws = normalize_weights(weights or [])
    if len(ws) == 1:
        return ws[0]["profile_id"]
    d = max(0, int(day_index or 0))
    # cuotas acumuladas: el perfil cuyo «déficit» (peso × días − asignados) es mayor recibe el día
    assigned = {w["profile_id"]: 0 for w in ws}
    choice = ws[0]["profile_id"]
    for i in range(d + 1):
        best, best_gap = None, None
        for w in ws:
            gap = w["weight"] * (i + 1) - assigned[w["profile_id"]]
            if best is None or gap > best_gap + 1e-9:
                best, best_gap = w["profile_id"], gap
        assigned[best] += 1
        choice = best
    return choice


def cultural_country_for_form_data(form_data: Optional[dict], day_index: Optional[int] = None) -> str:
    """País cuya COCINA guía las superficies culturales (plantillas, inspiración, juez, hábitos de franja).
    Con `day_index`, la cocina asignada a ese día; sin él, la principal. Con el knob apagado o sin elección,
    coincide con el país de compra: legado byte-idéntico."""
    ws = culture_weights_for_form(form_data)
    pid = profile_for_day(ws, day_index) if day_index is not None else ws[0]["profile_id"]
    return country_for_profile(pid)


def heading_for_weights(weights: Optional[Iterable[dict]]) -> str:
    """«INSPIRACIÓN DOMINICANA» (legado byte-idéntico para DO puro) o, en mezcla,
    «INSPIRACIÓN: COCINA DOMINICANA 70 % · COCINA ESPAÑOLA 30 %»."""
    ws = normalize_weights(weights or [])
    if len(ws) == 1:
        pid = ws[0]["profile_id"]
        if pid == DEFAULT_PROFILE:
            return "INSPIRACIÓN DOMINICANA"
        try:
            from constants import COUNTRY_PROFILES
            nm = (COUNTRY_PROFILES.get(country_for_profile(pid)) or {}).get("name_es")
            if nm:
                return f"INSPIRACIÓN DE {nm.upper()}"
        except Exception:
            pass
        return f"INSPIRACIÓN: {profile_name_es(pid).upper()}"
    return "INSPIRACIÓN: " + " · ".join(f"{profile_name_es(w['profile_id']).upper()} {int(round(w['weight'] * 100))} %" for w in ws)


def describe_weights_es(weights: Optional[Iterable[dict]]) -> str:
    ws = normalize_weights(weights or [])
    if len(ws) == 1:
        return profile_name_es(ws[0]["profile_id"])
    return " · ".join(f"{profile_name_es(w['profile_id'])} {int(round(w['weight'] * 100))} %" for w in ws)


__all__ = [
    "PROFILES", "DEFAULT_PROFILE", "INTENSITY_WEIGHT", "MAX_SECONDARY", "MIN_MAIN_WEIGHT",
    "cultural_profiles_enabled", "profile_ids", "is_profile", "profile_for_market", "library_for_profile",
    "country_for_profile", "profile_name_es", "normalize_weights", "weights_from_form_field",
    "culture_weights_for_form", "main_profile_id", "profile_for_day", "cultural_country_for_form_data",
    "heading_for_weights", "describe_weights_es",
]
