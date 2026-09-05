"""[P1-ARQ25-F6-DISH-REGISTRY · 2026-09-05] Dish Registry compilado (Fase 6 del roadmap 2.5, capa V2.3).

Compilador + snapshot inmutable + loader de runtime (§7.3):

  1. Fuente curada versionada: `data/dish_templates*.json` (6 bibliotecas, 338 plantillas con `template_id`
     acuñado en Fase 2) + `constituents` (nombre + gramos): en las 5 beta vienen en la plantilla; en DO los
     aporta `data/dish_constituents_do.json` (curación de `scripts/build_dish_constituents_do.py`).
  2. `compile_library` valida el esquema, resuelve CADA constituyente contra `master_ingredients`
     (nombre canónico + alias, sin acentos) y deriva atributos intrínsecos por porción a partir de las
     columnas por 100 g del catálogo: sodio, potasio, fósforo, grasa saturada, azúcares, carga glucémica
     potencial, densidad energética, procesados y clases de alérgenos (vocabulario SSOT
     `graph_orchestrator._ALLERGEN_SYNONYMS`). **Cero tags clínicos manuales** (§7.2): nunca
     `safe_for_diabetes`; la elegibilidad se evalúa en runtime sobre la plantilla ya dimensionada.
  3. `write_snapshot` genera `data/registry/dish_registry_<lib>_v<version>.json`, reproducible bit a bit:
     sin timestamps, claves ordenadas, `snapshot_hash` = sha256 del contenido canónico; `source_hash`
     (plantillas + constituyentes) y `catalog_fingerprint` (nombres + nutrición del catálogo) explican
     cualquier cambio de hash.
  4. Runtime: `load_registry(country)` carga el snapshot de la versión activa
     (knob `MEALFIT_DISH_REGISTRY_SNAPSHOT`, default "1"), cacheado; `template_candidates` sirve al
     allocator (Fase 3) candidatos por franja/familia; `registry_hash` viaja en el blueprint y en la métrica
     de fidelidad (los benchmarks guardan su hash).

Gate de la fase: 100 % de constituyentes resuelve o queda excluido EXPLÍCITAMENTE (`excluded[]` con motivo);
snapshot reproducible desde la fuente. Fail-open total en runtime: sin snapshot ⇒ None, nada bloquea.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Iterable, Optional

from knobs import _env_str

logger = logging.getLogger(__name__)

REGISTRY_SCHEMA_VERSION = 2
COMPILER_VERSION = 2
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
REGISTRY_DIR = os.path.join(DATA_DIR, "registry")

# biblioteca → (archivo de plantillas, país de mercado, perfil cultural)
LIBRARIES: dict[str, tuple[str, str, str]] = {
    "do": ("dish_templates.json", "DO", "dominican_criolla"),
    "es": ("dish_templates_es.json", "ES", "espanola"),
    "mx": ("dish_templates_mx.json", "MX", "mexicana"),
    "co": ("dish_templates_co.json", "CO", "colombiana"),
    "pr": ("dish_templates_pr.json", "PR", "puertorriquena"),
    "us": ("dish_templates_us.json", "US", "estadounidense"),
}
_COUNTRY_TO_LIB = {v[1]: k for k, v in LIBRARIES.items()}

# Umbrales por PORCIÓN para los atributos intrínsecos (señales, no veredictos clínicos)
RISK_THRESHOLDS = {
    "sodium_high_mg": 600.0, "potassium_high_mg": 700.0, "phosphorus_high_mg": 350.0,
    "sat_fat_high_g": 6.0, "sugar_high_g": 25.0, "glycemic_load_high_net_carbs_g": 75.0, "energy_dense_kcal": 800.0,
}
_PROCESSED_TOKENS = ("salami", "jamon", "chorizo", "longaniza", "tocineta", "pepperoni", "salchich", "chicharron",
                     "sobrasada", "morcilla", "butifarra", "chistorra", "cecina", "lomo embuchado", "panceta",
                     "mortadela", "pavochon")
_NUTRIENT_COLS = {
    "kcal": "kcal_per_100g", "protein_g": "protein_g_per_100g", "carbs_g": "carbs_g_per_100g", "fats_g": "fats_g_per_100g",
    "fiber_g": "fiber_g_per_100g", "sodium_mg": "sodium_mg_per_100g", "potassium_mg": "potassium_mg_per_100g",
    "phosphorus_mg": "phosphorus_mg_per_100g", "saturated_fat_g": "saturated_fat_g_per_100g", "sugars_g": "sugars_g_per_100g",
}


def registry_snapshot_version() -> str:
    v = str(_env_str("MEALFIT_DISH_REGISTRY_SNAPSHOT", "1") or "1").strip()
    return re.sub(r"[^0-9A-Za-z._-]", "", v) or "1"


def snapshot_path(library: str, version: Optional[str] = None) -> str:
    return os.path.join(REGISTRY_DIR, f"dish_registry_{library}_v{version or registry_snapshot_version()}.json")


# ----------------------------------------------------------------------------- utilidades puras
def _norm(s: Any) -> str:
    try:
        from constants import strip_accents
        return re.sub(r"\s+", " ", strip_accents(str(s or "")).lower()).strip()
    except Exception:
        return str(s or "").lower().strip()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def _f(v: Any) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def build_catalog_index(rows: Iterable[dict]) -> dict:
    """`{nombre_normalizado: fila}` con alias. Nombre canónico primero (los alias no lo pisan)."""
    idx: dict[str, dict] = {}
    rows = [r for r in (rows or []) if isinstance(r, dict) and r.get("name")]
    for r in rows:
        idx.setdefault(_norm(r["name"]), r)
    for r in rows:
        for a in (r.get("aliases") or []):
            if a:
                idx.setdefault(_norm(a), r)
    return idx


def resolve_constituent(name: str, index: dict) -> Optional[dict]:
    n = _norm(name)
    if not n:
        return None
    row = index.get(n)
    if row:
        return row
    # singular/plural simple (Fresas → Fresa) y viceversa
    for cand in (n[:-1] if n.endswith("s") else n + "s", n[:-2] if n.endswith("es") else None):
        if cand and cand in index:
            return index[cand]
    return None


def catalog_fingerprint(rows: Iterable[dict]) -> str:
    slim = sorted(
        [str(r.get("name"))] + [str(round(_f(r.get(col)), 3)) for col in _NUTRIENT_COLS.values()]
        for r in (rows or []) if isinstance(r, dict) and r.get("name")
    )
    return _sha(slim)[:16]


def allergen_classes_for(names: Iterable[str]) -> list[str]:
    """Clases de alérgeno (vocabulario SSOT de graph_orchestrator) presentes en los constituyentes."""
    try:
        from graph_orchestrator import _ALLERGEN_SYNONYMS as vocab
    except Exception:
        return []
    out: set[str] = set()
    norm_names = [_norm(n) for n in names if n]
    for cls, tokens in (vocab or {}).items():
        for tok in (tokens or []):
            t = _norm(tok)
            if not t:
                continue
            for n in norm_names:
                if re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", n):
                    out.add(str(cls))
                    break
            if str(cls) in out:
                break
    return sorted(out)


def derive_risk_attributes(nutrition: dict, constituent_names: Iterable[str]) -> dict:
    """Señales intrínsecas por porción (§7.2). Booleanos derivados de umbrales + lista de alérgenos."""
    n = nutrition or {}
    net_carbs = max(0.0, _f(n.get("carbs_g")) - _f(n.get("fiber_g")))
    names = list(constituent_names)
    processed = sorted({nm for nm in names if any(tok in _norm(nm) for tok in _PROCESSED_TOKENS)})
    return {
        "sodium_high": _f(n.get("sodium_mg")) >= RISK_THRESHOLDS["sodium_high_mg"],
        "potassium_high": _f(n.get("potassium_mg")) >= RISK_THRESHOLDS["potassium_high_mg"],
        "phosphorus_high": _f(n.get("phosphorus_mg")) >= RISK_THRESHOLDS["phosphorus_high_mg"],
        "sat_fat_high": _f(n.get("saturated_fat_g")) >= RISK_THRESHOLDS["sat_fat_high_g"],
        "sugar_high": _f(n.get("sugars_g")) >= RISK_THRESHOLDS["sugar_high_g"],
        "glycemic_load_high": net_carbs >= RISK_THRESHOLDS["glycemic_load_high_net_carbs_g"],
        "energy_dense": _f(n.get("kcal")) >= RISK_THRESHOLDS["energy_dense_kcal"],
        "processed_meat": bool(processed),
        "processed_items": processed,
        "allergens": allergen_classes_for(names),
        "net_carbs_g": round(net_carbs, 1),
    }


# ----------------------------------------------------------------------------- logística y editorial (§7.2)
# [P1-ARQ25-F6-REGISTRY-PROMPT · 2026-09-05] Metadata batch/freezer/shelf-life y estado editorial. Todo
# DERIVADO (técnica de la plantilla + vida útil por ingrediente del catálogo) y marcado como estimación:
# ningún número clínico, ninguna curación manual escondida en el snapshot.
_TECH_RULES = (
    # (tokens de técnica, batch_friendly, freezer_friendly, prep_min, difficulty)
    (("sopa", "sancocho", "asopao", "crema", "guisado", "estofado", "mechad"), True, True, 50, "media"),
    (("horneado", "al horno", "airfryer", "horno"), True, True, 40, "media"),
    (("masa horneada", "masa hervida", "masa al sart", "masa a la plancha"), True, True, 35, "media"),
    (("hervido", "majado", "vapor"), True, True, 30, "baja"),
    (("plancha", "salteado", "sart", "parrilla", "revuelto", "frito"), False, False, 20, "baja"),
    (("frio", "frío", "licuado", "masa fria", "masa fría", "tibio"), False, False, 10, "baja"),
)
_PERISHABLE_CATEGORIES = ("proteínas", "proteinas", "carnes", "pescados", "mariscos", "lácteos", "lacteos", "vegetales", "frutas", "verduras")


def derive_logistics(template: dict, resolved: list, index: dict) -> dict:
    tech = _norm(template.get("technique"))
    batch, freezer, prep, diff = True, True, 30, "media"
    for tokens, b, fz, pm, d in _TECH_RULES:
        if any(t in tech for t in tokens):
            batch, freezer, prep, diff = b, fz, pm, d
            break
    shelf = []
    for r in resolved:
        row = index.get(_norm(r.get("canonical"))) or {}
        sl = row.get("shelf_life_days")
        try:
            sl = int(sl) if sl is not None else None
        except (TypeError, ValueError):
            sl = None
        if sl is not None and sl > 0:
            shelf.append(sl)
    return {
        "batch_friendly": bool(batch), "freezer_friendly": bool(freezer),
        "min_shelf_life_days": min(shelf) if shelf else None,
        "prep_minutes_est": int(prep), "difficulty_est": diff, "estimated": True,
    }


def derive_editorial(template: dict, library: str) -> dict:
    try:
        from plan_policy import TEMPLATE_ALIASES
        aliases = sorted(k for k, v in (TEMPLATE_ALIASES or {}).items() if v == template.get("name"))
    except Exception:
        aliases = []
    return {"status": "curated", "source": f"data/dish_templates{'' if library == 'do' else '_' + library}.json",
            "display_name": {"es": template.get("name")}, "aliases": aliases, "media": []}


# ----------------------------------------------------------------------------- compilación
def _constituents_source(library: str, template: dict, do_constituents: Optional[dict]) -> tuple[list, list[str]]:
    """(constituyentes [{name, grams}], declarados_sin_resolver) según la biblioteca."""
    if library == "do":
        entry = ((do_constituents or {}).get("templates") or {}).get(str(template.get("name") or ""))
        if entry:
            return list(entry.get("constituents") or []), list(entry.get("declared_unresolved") or [])
        # [P1-ARQ25-F7-CULTURE] plantillas DO nuevas sin entrada curada: valen los constituyentes inline (mismo
        # contrato que las bibliotecas beta); sin ninguno de los dos, la plantilla queda sin constituyentes.
        if not (template.get("constituents") or []):
            return [], []
    cons = template.get("constituents") or []
    out = []
    for c in cons:
        if isinstance(c, dict) and c.get("name"):
            out.append({"name": c["name"], "grams": _f(c.get("grams", c.get("g")))})
    return out, []


def compile_template(template: dict, index: dict, *, library: str, constituents: list, declared_unresolved: list[str]) -> dict:
    resolved, excluded = [], []
    for c in constituents:
        name = str((c or {}).get("name") or "")
        grams = _f((c or {}).get("grams"))
        row = resolve_constituent(name, index)
        if not row:
            excluded.append({"name": name, "grams": grams, "reason": "not_in_catalog"})
            continue
        if grams <= 0:
            excluded.append({"name": name, "grams": grams, "reason": "no_grams"})
            continue
        try:
            from plan_policy import ingredient_id_for
            iid = ingredient_id_for(row["name"])
        except Exception:
            iid = _norm(row["name"]).replace(" ", "_")
        resolved.append({"name": name, "canonical": row["name"], "ingredient_id": iid, "grams": round(grams, 1)})
    for name in declared_unresolved or []:
        excluded.append({"name": str(name), "grams": None, "reason": "declared_unresolved"})
    nutrition = {k: 0.0 for k in _NUTRIENT_COLS}
    for r in resolved:
        row = index.get(_norm(r["canonical"])) or {}
        factor = r["grams"] / 100.0
        for k, col in _NUTRIENT_COLS.items():
            nutrition[k] += _f(row.get(col)) * factor
    nutrition = {k: round(v, 2) for k, v in nutrition.items()}
    names = [r["canonical"] for r in resolved]
    status = "ok" if resolved and not [e for e in excluded if e["reason"] == "not_in_catalog"] else ("partial" if resolved else "excluded")
    body = {
        "template_id": template.get("template_id"), "template_version": template.get("template_version"),
        "name": template.get("name"), "slots": sorted(template.get("slots") or []),
        "base": template.get("base"), "protein": template.get("protein"), "technique": template.get("technique"),
        "transform": bool(template.get("transform")), "library": library,
        "constituents": resolved, "excluded": excluded, "status": status,
        "serving_g": round(sum(r["grams"] for r in resolved), 1),
        "nutrition_per_serving": nutrition,
        "intrinsic_risk_attributes": derive_risk_attributes(nutrition, names),
        "logistics": derive_logistics(template, resolved, index),
        "editorial": derive_editorial(template, library),
    }
    body["content_hash"] = _sha(body)[:16]
    return body


def compile_library(library: str, *, catalog_rows: Optional[list] = None, version: Optional[str] = None,
                    templates: Optional[list] = None, do_constituents: Optional[dict] = None) -> dict:
    """Snapshot de UNA biblioteca. Determinista: misma fuente + mismo catálogo ⇒ mismo `snapshot_hash`."""
    lib = str(library or "do").lower()
    if lib not in LIBRARIES:
        raise ValueError(f"biblioteca desconocida: {library!r}")
    fname, country, culture = LIBRARIES[lib]
    path = os.path.join(DATA_DIR, fname)
    if templates is None:
        import dish_library
        templates = dish_library.load_dish_templates(path)
    if catalog_rows is None:
        from shopping_calculator import get_master_ingredients
        catalog_rows = list(get_master_ingredients() or [])
    if lib == "do" and do_constituents is None:
        p = os.path.join(DATA_DIR, "dish_constituents_do.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                do_constituents = json.load(f)
    index = build_catalog_index(catalog_rows)
    compiled = []
    for t in templates:
        if not isinstance(t, dict) or not t.get("template_id"):
            continue
        cons, declared = _constituents_source(lib, t, do_constituents)
        compiled.append(compile_template(t, index, library=lib, constituents=cons, declared_unresolved=declared))
    compiled.sort(key=lambda x: str(x.get("template_id")))
    source_material = {
        "templates": [{k: t.get(k) for k in ("name", "slots", "base", "protein", "technique", "transform", "constituents")}
                      for t in templates if isinstance(t, dict)],
        "do_constituents": (do_constituents or {}).get("templates") if lib == "do" else None,
    }
    n_cons = sum(len(c["constituents"]) + len(c["excluded"]) for c in compiled)
    n_res = sum(len(c["constituents"]) for c in compiled)
    snap = {
        "schema_version": REGISTRY_SCHEMA_VERSION, "compiler_version": COMPILER_VERSION,
        "registry_version": str(version or registry_snapshot_version()),
        "library": lib, "country": country, "culture": culture,
        "source_hash": _sha(source_material)[:16], "catalog_fingerprint": catalog_fingerprint(catalog_rows),
        "risk_thresholds": RISK_THRESHOLDS,
        "stats": {
            "templates": len(compiled),
            "ok": sum(1 for c in compiled if c["status"] == "ok"),
            "partial": sum(1 for c in compiled if c["status"] == "partial"),
            "excluded": sum(1 for c in compiled if c["status"] == "excluded"),
            "constituents": n_cons, "resolved": n_res,
            "resolution_pct": round(100.0 * n_res / n_cons, 1) if n_cons else 0.0,
        },
        "templates": compiled,
    }
    snap["snapshot_hash"] = _sha(snap)
    return snap


def write_snapshot(snap: dict, path: Optional[str] = None) -> str:
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    p = path or snapshot_path(snap["library"], snap.get("registry_version"))
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(_canonical_json(snap))
        f.write("\n")
    return p


def verify_snapshot(snap: dict) -> bool:
    """El hash declarado coincide con el contenido (integridad, reproducibilidad)."""
    body = {k: v for k, v in (snap or {}).items() if k != "snapshot_hash"}
    return bool(snap) and _sha(body) == snap.get("snapshot_hash")


# ----------------------------------------------------------------------------- runtime
_CACHE: dict[str, Optional[dict]] = {}


def library_for_country(country: Optional[str]) -> str:
    """Biblioteca del país. Un código explícito de una biblioteca conocida se respeta tal cual (el registry
    es un DATO por país, no una puerta de producto); lo demás pasa por el SSOT `country_for_form_data`
    (que aplica el knob maestro del sistema de países y cae a DO)."""
    raw = str(country or "").strip().upper()
    if raw in _COUNTRY_TO_LIB:
        return _COUNTRY_TO_LIB[raw]
    try:
        from constants import country_for_form_data
        canon = country_for_form_data({"country": country}) if country else "DO"
    except Exception:
        canon = "DO"
    return _COUNTRY_TO_LIB.get(str(canon or "DO").upper(), "do")


def load_registry(country: Optional[str] = None, version: Optional[str] = None) -> Optional[dict]:
    """Snapshot activo para el país (fail-open: None si no existe o está corrupto)."""
    lib = library_for_country(country)
    p = snapshot_path(lib, version)
    if p in _CACHE:
        return _CACHE[p]
    snap = None
    try:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                snap = json.load(f)
            if not verify_snapshot(snap):
                logger.warning(f"[ARQ25-F6] snapshot corrupto (hash no coincide): {p}")
                snap = None
    except Exception as e:
        logger.warning(f"[ARQ25-F6] snapshot no cargado {p}: {e!r}")
        snap = None
    _CACHE[p] = snap
    return snap


def registry_hash(country: Optional[str] = None) -> Optional[str]:
    snap = load_registry(country)
    return snap.get("snapshot_hash") if snap else None


_SLOT_ALIASES_ES = {
    "breakfast": "desayuno", "desayuno": "desayuno",
    "lunch": "almuerzo", "almuerzo": "almuerzo", "comida": "almuerzo",
    "dinner": "cena", "cena": "cena",
    "snack": "merienda", "merienda": "merienda", "merienda_am": "merienda", "merienda_pm": "merienda",
    "colacion": "merienda", "colación": "merienda",
}


def canonical_slot_es(slot: Any) -> str:
    """Franja en el vocabulario del registry (español). El motor canoniza a inglés (`meal_slot` → `dinner`);
    las plantillas hablan español (`cena`). Acepta ambos y alias; desconocido ⇒ tal cual en minúsculas."""
    s = _norm(slot)
    return _SLOT_ALIASES_ES.get(s, s)


def template_candidates(country: Optional[str], slot: str, family: Optional[str] = None, *, k: int = 6,
                        exclude_allergens: Iterable[str] = ()) -> list[dict]:
    """Candidatos del registry para el allocator: `status='ok'`, franja compatible, familia de proteína
    compatible (vía `horizon.family_matches`), sin las clases de alérgeno excluidas. Orden estable."""
    snap = load_registry(country)
    if not snap:
        return []
    ex = {str(a).lower() for a in exclude_allergens or ()}
    slot_es = canonical_slot_es(slot)
    out = []
    try:
        from horizon import family_matches
    except Exception:
        family_matches = None
    for t in snap.get("templates") or []:
        if t.get("status") != "ok" or slot_es not in (t.get("slots") or []):
            continue
        if ex and ex.intersection({a.lower() for a in (t.get("intrinsic_risk_attributes") or {}).get("allergens", [])}):
            continue
        if family and family_matches is not None:
            prot = str(t.get("protein") or "none").lower()
            if prot not in ("none", "mixta") and not family_matches(family, prot):
                continue
        out.append({"template_id": t["template_id"], "name": t["name"], "protein": t.get("protein"),
                    "technique": t.get("technique"), "transform": t.get("transform"),
                    "logistics": t.get("logistics") or {}})
        if len(out) >= max(1, int(k)):
            break
    return out


__all__ = [
    "REGISTRY_SCHEMA_VERSION", "COMPILER_VERSION", "LIBRARIES", "RISK_THRESHOLDS", "REGISTRY_DIR",
    "registry_snapshot_version", "snapshot_path", "build_catalog_index", "resolve_constituent", "catalog_fingerprint",
    "allergen_classes_for", "derive_risk_attributes", "compile_template", "compile_library", "write_snapshot",
    "verify_snapshot", "library_for_country", "load_registry", "registry_hash", "template_candidates",
    "derive_logistics", "derive_editorial",
]
