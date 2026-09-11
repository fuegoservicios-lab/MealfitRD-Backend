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

import pantry_durability as _pd  # [F7-G] SSOT de durabilidad
import hashlib
import json
import logging
import math
import os
import re
from typing import Any, Iterable, Optional

from knobs import _env_str

logger = logging.getLogger(__name__)

REGISTRY_SCHEMA_VERSION = 3   # [F7-G] durabilidad por constituyente + logística de despensa
COMPILER_VERSION = 3
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
    # [P1-PLANT-MILK-NOT-DAIRY · 2026-09-06] «Leche de coco» salía etiquetada como LÁCTEO, y con ella cualquier
    # plato que la use. No es el fallo de subcadena de siempre —el matcher ya compara por palabra completa— sino
    # el contrario: «leche» ES una palabra completa dentro de «leche de coco», y aun así la bebida no lleva
    # lácteo. Medido al añadir los desayunos sin lácteos ni huevo: la batida de lechosa quedaba fuera del filtro
    # de un alérgico a la leche por un alérgeno que no tiene.
    #
    # El daño va en las dos direcciones: excluye platos seguros para quien no tolera lácteos, y le dice a quien
    # sí los tolera que ese plato lleva algo que no lleva.
    # Y lo mismo por el otro lado del vocabulario: «mantequilla» es palabra láctea, pero la de maní no lleva
    # leche — medido aquí mismo: «Mantequilla de maní» devolvía lacteos + lactosa + mani.
    _leches_vegetales = ("leche de coco", "leche de almendra", "leche de almendras", "leche de soya",
                         "leche de soja", "leche de avena", "leche de arroz", "leche de anacardo",
                         "leche vegetal", "bebida de almendras", "bebida de avena", "bebida de soya",
                         "mantequilla de mani", "mantequilla de almendra", "mantequilla de almendras",
                         "mantequilla de anacardo", "mantequilla de semillas", "crema de cacahuate",
                         "crema de mani", "crema de coco", "queso vegano", "yogur vegetal", "yogurt vegetal",
                         # [ARQ27-P1-05 · 2026-09-06] «yogur vegetal» estaba; «Yogur de coco», que es como
                         # se llama la FILA del catálogo, no. Auditadas las 347 filas contra el guard de
                         # alergias: era la ÚNICA discrepancia entre lo que el registry declara y lo que el
                         # scan decide — el registry la marcaba láctea y el scan no. Un plato con yogur de
                         # coco quedaba fuera del pool de un alérgico a la leche por un alérgeno que no
                         # tiene. `test_las_dos_capas_de_lacteo_coinciden` ancla la paridad para la próxima.
                         "yogur de coco", "yogurt de coco", "yogur de soya", "yogurt de soya",
                         "yogur de almendras", "yogurt de almendras", "yogur de anacardo")
    # La salsa de soya corriente lleva trigo (solo el tamari no): para el registry cuenta como gluten.
    vocab = dict(vocab or {})
    vocab["gluten"] = list(vocab.get("gluten") or []) + ["salsa de soya", "salsa soya", "soy sauce"]
    for cls, tokens in vocab.items():
        for tok in (tokens or []):
            t = _norm(tok)
            if not t:
                continue
            _es_lacteo = "lacte" in str(cls).lower() or "lactos" in str(cls).lower()
            for n in norm_names:
                # [P1-PLANT-MILK-NOT-DAIRY] la bebida vegetal no aporta la clase láctea (sí las demás: una
                # bebida de almendras SIGUE siendo frutos secos, y así se declara por su propio token).
                if _es_lacteo and any(v in n for v in _leches_vegetales):
                    continue
                # [P1-ARQ25-F7-CULTURE · revisión curatorial] tolerante a plural: el vocabulario dice «sardina»,
                # «fideo», «almeja» y el catálogo «Sardinas en lata», «Fideos», «Almejas» — con frontera de palabra
                # estricta, tres bibliotecas servían sardinas sin la clase «pescado».
                if re.search(r"(?<![a-z])" + re.escape(t) + r"(?:e?s)?(?![a-z])", n):
                    out.add(str(cls))
                    break
            if str(cls) in out:
                break
    return sorted(out)


# [ARQ27-P0-03 · 2026-09-06] Qué nutriente sostiene cada señal de riesgo. Si ese nutriente es
# DESCONOCIDO en alguno de los constituyentes, la señal no puede decir `False`: dice `None`.
_RISK_SOURCES = {
    "sodium_high": ("sodium_mg",), "potassium_high": ("potassium_mg",),
    "phosphorus_high": ("phosphorus_mg",), "sat_fat_high": ("saturated_fat_g",),
    "sugar_high": ("sugars_g",), "energy_dense": ("kcal",),
    "glycemic_load_high": ("carbs_g", "fiber_g"),
}


def derive_risk_attributes(nutrition: dict, constituent_names: Iterable[str],
                           unknown: Iterable[str] = ()) -> dict:
    """Señales intrínsecas por porción (§7.2), derivadas de umbrales + lista de alérgenos.

    [ARQ27-P0-03] Tri-estado: `True` / `False` / `None`. `None` es «no lo sé», y aparece cuando algún
    constituyente no trae ese nutriente en el catálogo. Antes `_f(None)` devolvía 0,0, la suma
    incorporaba ese cero como si fuera un dato y la señal salía `False` — un plato con Hoja santa se
    le presentaba a un perfil renal como «fósforo bajo» cuando la verdad era que nadie lo había
    medido. Cero medido y cero por ausencia no son el mismo número. Contrato I20."""
    n = nutrition or {}
    unk = {str(u) for u in (unknown or ())}

    def _flag(key, valor):
        return None if unk.intersection(_RISK_SOURCES.get(key, ())) else valor

    net_carbs = max(0.0, _f(n.get("carbs_g")) - _f(n.get("fiber_g")))
    names = list(constituent_names)
    processed = sorted({nm for nm in names if any(tok in _norm(nm) for tok in _PROCESSED_TOKENS)})
    return {
        "sodium_high": _flag("sodium_high", _f(n.get("sodium_mg")) >= RISK_THRESHOLDS["sodium_high_mg"]),
        "potassium_high": _flag("potassium_high", _f(n.get("potassium_mg")) >= RISK_THRESHOLDS["potassium_high_mg"]),
        "phosphorus_high": _flag("phosphorus_high", _f(n.get("phosphorus_mg")) >= RISK_THRESHOLDS["phosphorus_high_mg"]),
        "sat_fat_high": _flag("sat_fat_high", _f(n.get("saturated_fat_g")) >= RISK_THRESHOLDS["sat_fat_high_g"]),
        "sugar_high": _flag("sugar_high", _f(n.get("sugars_g")) >= RISK_THRESHOLDS["sugar_high_g"]),
        "glycemic_load_high": _flag("glycemic_load_high", net_carbs >= RISK_THRESHOLDS["glycemic_load_high_net_carbs_g"]),
        "energy_dense": _flag("energy_dense", _f(n.get("kcal")) >= RISK_THRESHOLDS["energy_dense_kcal"]),
        "processed_meat": bool(processed),
        "processed_items": processed,
        "allergens": allergen_classes_for(names),
        "net_carbs_g": None if unk.intersection(_RISK_SOURCES["glycemic_load_high"]) else round(net_carbs, 1),
    }


# ----------------------------------------------------------------------------- logística y editorial (§7.2)
# [P1-ARQ25-F6-REGISTRY-PROMPT · 2026-09-05] Metadata batch/freezer/shelf-life y estado editorial. Todo
# DERIVADO (técnica de la plantilla + vida útil por ingrediente del catálogo) y marcado como estimación:
# ningún número clínico, ninguna curación manual escondida en el snapshot.
_TECH_RULES = (
    # (tokens de técnica, batch_friendly, freezer_friendly, prep_min, difficulty)
    (("sopa", "sancocho", "asopao", "crema", "guisado", "estofado", "mechad"), True, True, 50, "media"),
    (("horneado", "al horno", "airfryer", "horno", "asado"), True, True, 40, "media"),
    (("masa horneada", "masa hervida", "masa al sart", "masa a la plancha"), True, True, 35, "media"),
    (("hervido", "majado", "vapor"), True, True, 30, "baja"),
    (("plancha", "salteado", "sart", "parrilla", "revuelto", "frito", "tortilla"), False, False, 20, "baja"),
    # [P1-MINUTOS-DE-LA-RECETA · 2026-09-10] `crudo`/`ensamblado`/`batido` no casaban NINGUNA fila y
    # caían al defecto de 30: 21 plantillas, 16 de ellas crudas. «Yogurt griego con guineo» —pelar un
    # guineo— anunciaba media hora. Un defecto que se aplica en silencio no es un defecto: es un dato
    # inventado con la misma cara que uno medido.
    (("crudo", "ensamblado", "frio", "frío", "licuado", "batido", "masa fria", "masa fría", "tibio"),
     False, False, 10, "baja"),
)

# ----------------------------------------------------------------------------- tiempo declarado (§7.2)
# [P1-MINUTOS-DE-LA-RECETA · 2026-09-10] La receta escrita YA dice cuánto tarda cada paso; la ficha lo
# contradecía con una tabla por técnica. Medido contra la tercera ronda de juicio humano: la suma de
# los pasos acierta el número que pidió el dueño en 13 de 13 platos (±5 min).
_RE_RANGO = re.compile(r"(\d+)\s*(?:-|–|a)\s*(\d+)\s*(minutos?|segundos?)", re.I)
_RE_SUELTO = re.compile(r"(\d+)\s*(minutos?|segundos?)", re.I)
# Un paso que dice «mientras» no viene DESPUÉS del anterior: corre DENTRO. Sumarlo infló «Víveres
# guisados con garbanzos» a 125 min contando los 25 de pelar los víveres encima de los 90 del hervor
# que la propia receta dice que están ocurriendo a la vez.
_RE_SOLAPE = re.compile(r"\b(?:mientras(?:\s+tanto)?|en\s+lo\s+que|al\s+mismo\s+tiempo|entre\s+tanto)\b", re.I)
MARGEN_MANIPULACION_MIN = 3   # picar, pelar, montar: lo que ninguna receta cronometra
PISO_MINUTOS = 5              # ni el plato más simple se arma en menos


def _tiempo_de_un_paso(paso) -> float:
    """Minutos que ESE paso declara. Del rango se toma el tope: promete que no pasará de ahí."""
    resto = str(paso or "")
    total = 0.0
    for m in _RE_RANGO.finditer(resto):
        alto = float(m.group(2))
        if m.group(3).lower().startswith("segundo"):
            alto /= 60.0
        total += alto
    resto = _RE_RANGO.sub(" ", resto)              # el tope del rango ya se contó: no lo cuentes otra vez
    for m in _RE_SUELTO.finditer(resto):
        v = float(m.group(1))
        if m.group(2).lower().startswith("segundo"):
            v /= 60.0
        total += v
    return total


def minutos_de_los_pasos(pasos) -> Optional[int]:
    """Minutos que la receta DECLARA, redondeados a 5. `None` si no hay receta.

    Una receta SIN ningún tiempo (montar un yogurt con guineo) no es un dato ausente: es un plato de
    ensamblaje, y vale el piso. Por eso el `None` se reserva para «no hay pasos» — mezclar los dos
    casos devolvería la tabla justo para los platos que la tabla peor estima.
    """
    if not pasos:
        return None
    total = 0.0
    previo = 0.0
    for p in pasos:
        t = _tiempo_de_un_paso(p)
        # Lo que corre dentro del paso anterior sólo añade lo que se le SALGA por arriba.
        total += max(0.0, t - previo) if _RE_SOLAPE.search(str(p or "")) else t
        previo = t
    return int(math.ceil(max(PISO_MINUTOS, total + MARGEN_MANIPULACION_MIN) / 5.0) * 5)
_PERISHABLE_CATEGORIES = ("proteínas", "proteinas", "carnes", "pescados", "mariscos", "lácteos", "lacteos", "vegetales", "frutas", "verduras")


def derive_logistics(template: dict, resolved: list, index: dict, pasos: Optional[list] = None) -> dict:
    tech = _norm(template.get("technique"))
    batch, freezer, prep, diff = True, True, 30, "media"
    fuente = "defecto"
    for tokens, b, fz, pm, d in _TECH_RULES:
        if any(t in tech for t in tokens):
            batch, freezer, prep, diff = b, fz, pm, d
            fuente = "tecnica"
            break
    de_receta = minutos_de_los_pasos(pasos)
    if de_receta is not None:
        prep, fuente = de_receta, "receta"
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
    dur = _pd.durability_of(resolved)
    return {
        "batch_friendly": bool(batch), "freezer_friendly": bool(freezer),
        "min_shelf_life_days": min(shelf) if shelf else None,
        # [F7-G · corregido P1-REGISTRY-CANTIDADES-EN-CRUDO 2026-09-09] Días que aguantan sus INGREDIENTES
        # CRUDOS sin congelador / congelando la proteína, y si son de despensa (≥ 21 días). NO es la vida
        # del plato ya cocinado: una tortilla de papa dura 3 días, y su `days_fresh_min` dice 35 porque esa
        # es la pregunta que el ciclo de una sola compra necesita — «¿puedo COCINAR esto el día 25?» —, no
        # «¿cuánto aguanta el plato hecho?». `pantry_durability.template_fits` ya lo decía así; este
        # comentario decía lo contrario, y esa contradicción es la que la revisión humana del 09-sep
        # encontró en 35 de 35 platos. Si algún día este número se le enseña a un usuario, hay que
        # etiquetarlo como vida de los INGREDIENTES o se convierte en una afirmación falsa sobre comida.
        "days_fresh_min": dur["days_fresh_min"], "days_with_freezer_min": dur["days_with_freezer_min"], "pantry_only": dur["pantry_only"],
        # `estimated` sigue describiendo el BLOQUE (tanda, congelador, vida útil son derivados);
        # `prep_minutes_source` dice de dónde salió ESTE número: `receta` lo declara la receta escrita,
        # `tecnica` lo estima la tabla, `defecto` es el 30 de relleno — que ahora se ve.
        "prep_minutes_est": int(prep), "prep_minutes_source": fuente,
        "difficulty_est": diff, "estimated": True,
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
        # [revisión curatorial F7] constituyentes INLINE explícitos ganan a la tabla curada (que el script rellena por
        # reglas para las plantillas sin entrada a mano): lo explícito manda sobre lo generado.
        inline = [c for c in (template.get("constituents") or []) if isinstance(c, dict) and c.get("name")]
        if inline:
            return [{"name": c["name"], "grams": _f(c.get("grams", c.get("g")))} for c in inline], []
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


# [ARQ27-P0-02 · 2026-09-06] Las tres exclusiones que impiden llamar ÍNTEGRA a una plantilla.
# Antes solo contaba `not_in_catalog`: `declared_unresolved` y `no_grams` se ignoraban para el estado,
# así que cuatro plantillas DO figuraban `status=ok` con exclusiones dentro — incluida una «Batida de
# zapote» cuyo zapote no está resuelto. El plato salía al pool con su nombre prometiendo un
# ingrediente que la receta compilada no tiene.
#
# «Cada ingrediente resuelto o excluido» cuadra un contador; no certifica que la receta esté completa.
# La única salida de esto es que la fuente marque el constituyente `optional: true` — una excepción
# curada y VISIBLE, no un silencio.
_BLOCKING_EXCLUSIONS = ("not_in_catalog", "declared_unresolved", "no_grams")


def compile_template(template: dict, index: dict, *, library: str, constituents: list,
                     declared_unresolved: list[str], pasos: Optional[list] = None) -> dict:
    resolved, excluded = [], []
    for c in constituents:
        name = str((c or {}).get("name") or "")
        grams = _f((c or {}).get("grams"))
        opt = bool((c or {}).get("optional"))
        row = resolve_constituent(name, index)
        if not row:
            excluded.append({"name": name, "grams": grams, "reason": "not_in_catalog", "optional": opt})
            continue
        if grams <= 0:
            excluded.append({"name": name, "grams": grams, "reason": "no_grams", "optional": opt})
            continue
        try:
            from plan_policy import ingredient_id_for
            iid = ingredient_id_for(row["name"])
        except Exception:
            iid = _norm(row["name"]).replace(" ", "_")
        # [P1-ARQ25-F7-CULTURE · subfase G] durabilidad SSOT (pantry_durability): lo que el registry sabe de cuánto aguanta
        _dur = _pd.classify(row["name"], row.get("category"))
        resolved.append({"name": name, "canonical": row["name"], "ingredient_id": iid, "grams": round(grams, 1),
                         "durability": _dur["cls"], "days_fresh": _dur["days_fresh"]})
    for name in declared_unresolved or []:
        excluded.append({"name": str(name), "grams": None, "reason": "declared_unresolved", "optional": False})
    # [ARQ27-P0-03] Un nutriente AUSENTE en el catálogo no suma cero: se anota y la señal que depende
    # de él sale `None`. Medido el 06-sep sobre el catálogo vivo: 5 de 347 filas sin `phosphorus_mg`
    # (Hoja santa, Chontaduro, Champús, Borojó, Achiote — el lote beta), que sostienen 7 constituyentes
    # ya compilados. Los otros nueve nutrientes están completos hoy; el contrato es lo que impide que
    # la próxima fila con un hueco vuelva a certificarse como cero.
    nutrition = {k: 0.0 for k in _NUTRIENT_COLS}
    unknown: dict[str, list] = {}
    for r in resolved:
        row = index.get(_norm(r["canonical"])) or {}
        factor = r["grams"] / 100.0
        for k, col in _NUTRIENT_COLS.items():
            v = row.get(col)
            if v is None or str(v).strip() == "":
                unknown.setdefault(k, []).append(r["canonical"])
                continue
            nutrition[k] += _f(v) * factor
    nutrition = {k: round(v, 2) for k, v in nutrition.items()}
    unknown = {k: sorted(set(v)) for k, v in sorted(unknown.items())}
    names = [r["canonical"] for r in resolved]
    blocking = [e for e in excluded if e["reason"] in _BLOCKING_EXCLUSIONS and not e.get("optional")]
    status = "ok" if resolved and not blocking else ("partial" if resolved else "excluded")
    body = {
        "template_id": template.get("template_id"), "template_version": template.get("template_version"),
        "name": template.get("name"), "slots": sorted(template.get("slots") or []),
        "base": template.get("base"), "protein": template.get("protein"), "technique": template.get("technique"),
        "transform": bool(template.get("transform")), "library": library,
        "constituents": resolved, "excluded": excluded, "status": status,
        # [P1-REGISTRY-CANTIDADES-EN-CRUDO · 2026-09-09] Suma de los gramos CRUDOS, no el peso del plato
        # servido: 80 g de arroz seco pesan ~240 g cocidos. Las macros de arriba se calculan sobre ese
        # mismo estado crudo contra un catálogo en crudo, así que son CORRECTAS — lo que engaña es el
        # nombre, que promete una ración. Medido el 09-sep: 179 de 179 plantillas. No renombrar la clave
        # a la ligera: entra en `snapshot_hash`, y cambiarlo expira las 6 firmas curatoriales y el
        # `registry_hash` de los planes vivos. Si se le va a enseñar un peso al usuario, hay que calcular
        # el rendimiento tras cocción, no reetiquetar éste.
        "serving_g": round(sum(r["grams"] for r in resolved), 1),
        "nutrition_per_serving": nutrition,
        "nutrition_unknown": unknown,
        "intrinsic_risk_attributes": derive_risk_attributes(nutrition, names, unknown.keys()),
        "logistics": derive_logistics(template, resolved, index, pasos),
        "editorial": derive_editorial(template, library),
    }
    body["content_hash"] = _sha(body)[:16]
    return body


def recipe_steps_index(library: str) -> dict:
    """`template_id` → pasos escritos, leídos del snapshot de recetas. `{}` si esa biblioteca no tiene.

    Se lee AQUÍ y no vía `recipe_library.recipe_for_dish_name` a propósito: aquella resuelve por
    NOMBRE y está gateada por `MEALFIT_RECIPE_LIBRARY_SELECT`, que es una decisión de servicio. La
    compilación del registry no puede depender de un knob de runtime, o el mismo árbol daría dos
    snapshots distintos según el `.env` de quien compile — que es exactamente cómo mi `.env` local
    dio «0 de 14 días» en la sonda del día determinista.
    """
    p = os.path.join(REGISTRY_DIR, f"recipe_library_{str(library or 'do').lower()}_v1.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            por_id = json.load(f).get("por_id") or {}
    except Exception as e:                                                     # noqa: BLE001
        logger.warning(f"[P1-MINUTOS-DE-LA-RECETA] recetario {library} ilegible: {e!r}")
        return {}
    return {str(k): (v or {}).get("pasos") or [] for k, v in por_id.items() if isinstance(v, dict)}


def compile_library(library: str, *, catalog_rows: Optional[list] = None, version: Optional[str] = None,
                    templates: Optional[list] = None, do_constituents: Optional[dict] = None,
                    recipes: Optional[dict] = None) -> dict:
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
    if recipes is None:
        recipes = recipe_steps_index(lib)
    index = build_catalog_index(catalog_rows)
    compiled = []
    for t in templates:
        if not isinstance(t, dict) or not t.get("template_id"):
            continue
        cons, declared = _constituents_source(lib, t, do_constituents)
        compiled.append(compile_template(t, index, library=lib, constituents=cons,
                                         declared_unresolved=declared,
                                         pasos=(recipes or {}).get(str(t.get("template_id")))))
    compiled.sort(key=lambda x: str(x.get("template_id")))
    source_material = {
        "templates": [{k: t.get(k) for k in ("name", "slots", "base", "protein", "technique", "transform", "constituents")}
                      for t in templates if isinstance(t, dict)],
        "do_constituents": (do_constituents or {}).get("templates") if lib == "do" else None,
        # [P1-MINUTOS-DE-LA-RECETA] La receta entra en el hash de la fuente porque ahora DECIDE un
        # campo del snapshot. Va sólo el número derivado, no el texto: reescribir un paso sin tocar
        # sus tiempos no debe expirar las firmas curatoriales, y cambiar un tiempo sí.
        "recipe_minutes": {k: minutos_de_los_pasos(v) for k, v in sorted((recipes or {}).items())} or None,
    }
    n_cons = sum(len(c["constituents"]) + len(c["excluded"]) for c in compiled)
    n_res = sum(len(c["constituents"]) for c in compiled)
    # [ARQ27-P0-02] `excluded` (plantillas con status `excluded`) y `constituents_excluded` (LÍNEAS de
    # ingrediente excluidas) se leían como si fueran la misma cifra. Son dos poblaciones distintas y
    # ahora se nombran distinto; la identidad de abajo obliga a que cuadren.
    n_cons_ex = sum(len(c["excluded"]) for c in compiled)
    n_unknown = sum(1 for c in compiled if c.get("nutrition_unknown"))
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
            "constituents": n_cons, "resolved": n_res, "constituents_excluded": n_cons_ex,
            "templates_with_unknown_nutrient": n_unknown,
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


_BY_ID: dict = {}


def templates_by_id(country: Optional[str] = None) -> dict:
    """[ARQ27-P1-04 · 2026-09-06] `template_id` → plantilla, para resolver un CandidateSet FIJADO a un
    run sin volver a filtrar.

    La caché va indexada por el **hash del snapshot**, no por el nombre de la biblioteca ni por
    «v1»: el criterio del gap dice «resolver por hash de contenido, no solo por nombre v1», y una
    caché por nombre serviría plantillas viejas después de recompilar sin cambiar de versión — que
    es justo lo que el repo hace hoy (recompilar en `v1`)."""
    snap = load_registry(country)
    if not snap:
        return {}
    key = (library_for_country(country), str(snap.get("snapshot_hash") or ""))
    idx = _BY_ID.get(key)
    if idx is None:
        idx = {str(t.get("template_id")): t for t in (snap.get("templates") or [])
               if t.get("template_id")}
        _BY_ID[key] = idx
    return idx


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


# ----------------------------------------------------------------------------- dieta (ARQ27-P0-01)
# [ARQ27-P0-01 · 2026-09-06] El selector perdía la dieta. `template_candidates` filtraba franja,
# familia y alérgenos y NUNCA miraba el tipo de dieta, así que el bloque 📐 del prompt le ofrecía a un
# vegano paella de pollo y conejo, y a una vegetariana yogur y jamón. Medido sobre los seis snapshots
# el 06-sep: **1.109 de 1.646 candidatos ofrecidos (67,4 %) eran incompatibles** con la dieta pedida,
# en 2 dietas × 6 bibliotecas × 4 franjas.
#
# Las guardas finales seguían ahí y el LLM podía proponer otra variante, así que esto NO era una tasa
# de planes peligrosos. Era capacidad desperdiciada y una instrucción contradictoria: el prompt le
# pedía al modelo un plato que su propia dieta prohíbe, y el arreglo llegaba más tarde y más caro.
#
# La comprobación NO es una tabla nueva: reusa `_diet_pool_item_banned`, el mismo guard SSOT
# (`_scan_diet_violations`, mismos `_DIET_*_TERMS`, mismo matcher word-boundary, misma excusa
# plant-adjacent «carne de soya» / «leche de coco») que ya decide esto en el skeleton y en el
# day-gen. La lección de P1-DIET-CANON-SSOT es que una cuarta tabla de dieta deriva y sirve pollo a
# vegetarianas; aquí no hay tabla, hay adaptador.
#
# Y resuelve por TODOS los constituyentes, no por la etiqueta `protein`: una plantilla `none` puede
# llevar lácteos y una `mixta` puede llevar jamón. La etiqueta dice qué protagoniza el plato, jamás
# qué contiene.
_DIET_VERDICT_CACHE: dict[tuple, bool] = {}
_DIET_UNAVAILABLE = "unavailable"


def _diet_scope(diet: Any) -> Optional[str]:
    """Dieta canónica cuando RESTRINGE; `None` cuando no hay nada que filtrar (balanced/desconocida);
    `_DIET_UNAVAILABLE` cuando restringe pero el guard SSOT no se pudo cargar."""
    if not diet:
        return None
    try:
        from constants import canonicalize_diet_type
        canon = canonicalize_diet_type(diet)
    except Exception:
        canon = str(diet or "").strip().lower()
    if canon not in ("vegan", "vegetarian", "pescatarian"):
        return None
    try:
        from graph_orchestrator import _diet_pool_item_banned  # noqa: F401
    except Exception as e:
        logger.warning(f"[ARQ27-P0-01] guard de dieta no disponible ({e!r}) → no se ofrecen candidatos "
                       f"para '{canon}'. Sin bloque de registro (conducta previa a F6); ofrecer carne "
                       f"a un vegano no es una degradación aceptable.")
        return _DIET_UNAVAILABLE
    return canon


def _template_violates_diet(t: dict, canon: str) -> bool:
    """¿Algún constituyente de esta plantilla viola la dieta? Memoizado por `content_hash`: el snapshot
    es inmutable, así que cada plantilla se evalúa una vez por dieta y por proceso."""
    key = (t.get("content_hash") or t.get("template_id"), canon)
    hit = _DIET_VERDICT_CACHE.get(key)
    if hit is not None:
        return hit
    from graph_orchestrator import _diet_pool_item_banned
    verdict = False
    for c in (t.get("constituents") or []):
        nm = c.get("canonical") or c.get("name")
        if nm and _diet_pool_item_banned(nm, canon):
            verdict = True
            break
    _DIET_VERDICT_CACHE[key] = verdict
    return verdict


def _buyable(t: dict, market_country: Any) -> bool:
    """[ARQ27-P1-07] ¿Se compran en ese mercado todos los constituyentes de la plantilla?"""
    try:
        from catalog_capability import template_buyable_in
        return template_buyable_in(
            [c.get("canonical") or c.get("name") for c in (t.get("constituents") or [])], market_country)
    except Exception:
        return True


def _template_uses_excluded_food(t: dict, excluded: list, pnm) -> bool:
    """[P1-PLAN-LOTE-3 · B5] ¿Algún constituyente de la plantilla es uno de los alimentos excluidos?"""
    for c in (t.get("constituents") or []):
        for nombre in (c.get("name"), c.get("canonical")):
            if not nombre:
                continue
            for e in excluded:
                if _norm(e) == _norm(nombre):
                    return True
                try:
                    if pnm is not None and pnm(str(nombre), str(e)):
                        return True
                except Exception:
                    continue
    return False


def template_candidates(country: Optional[str], slot: str, family: Optional[str] = None, *, k: int = 6,
                        exclude_allergens: Iterable[str] = (), need_days: Optional[int] = None,
                        allow_frozen: bool = False, prefer_batch: bool = False,
                        diet: Any = None, require_known_nutrients: Iterable[str] = (),
                        market_country: Any = None, rotate: int = 0,
                        budget_tier: Any = None, exclude_foods: Iterable[str] = ()) -> list[dict]:
    """Candidatos del registry para el allocator: `status='ok'`, franja compatible, familia de proteína
    compatible (vía `horizon.family_matches_template`), sin las clases de alérgeno excluidas y sin
    violar la dieta declarada (ARQ27-P0-01). Orden estable.

    `require_known_nutrients` (ARQ27-P0-03) descarta las plantillas cuyo dato de ESE nutriente sea
    desconocido — el criterio del gap: un faltante de sodio/K/P bloquea solo los escenarios que
    exigen ese dato, no todos. Un perfil renal no puede recibir un plato cuyo fósforo nadie midió.

    `market_country` (ARQ27-P1-07) descarta lo que ese mercado no vende. La biblioteca es de la
    COCINA y el catálogo es del MERCADO (I16), y no tienen por qué coincidir: 26 plantillas
    dominicanas piden Casabe u Orégano dominicano, que ni ES ni US llevan en catálogo. Sin este
    filtro, a una cocina dominicana comprando en Estados Unidos se le ofrecían igual. Desconocido no
    recorta nada.

    [ARQ27-P1-04 · 2026-09-06] El orden era el de INSERCIÓN en el fichero con un corte al llegar a `k`:
    añadir una plantilla al principio del JSON cambiaba los candidatos de todas las consultas, y las
    últimas del fichero no se ofrecían jamás — el «ranking favorece siempre los primeros IDs» que el
    gap nombra. Ahora se recogen TODAS las compatibles y se ordenan por hash del `template_id` con la
    consulta como sal: determinista (mismo snapshot ⇒ mismo orden), independiente del orden de
    inserción, y distinto por franja y familia. `rotate` (el índice del día) desplaza la lista para que
    días consecutivos con la misma franja y familia no reciban siempre la misma cabeza.

    [P1-CANDIDATO-CON-PRECIO · 2026-09-09] `budget_tier` (`low`/`medium`/`custom`/`high`/`unlimited`,
    el que la política ya compiló en `effective["budget"]["tier"]`) recorta la cola CARA del conjunto.
    Medido: al presupuesto ajustado el prompt pide «evita mariscos caros» y el modelo puso Cangrejo
    2 lb = RD$958 — pedir no es imponer. Y elegir barato o caro son 5,8× sobre el ciclo, así que la
    elección del plato no es un factor del coste: es EL factor. La aritmética y la frontera viven en
    `dish_cost` porque el precio es del MERCADO y esto es la COCINA (I16), igual que `_buyable`.
    Sin tier ⇒ no se poda. Fail-open."""
    snap = load_registry(country)
    if not snap:
        return []
    diet_canon = _diet_scope(diet)
    if diet_canon is _DIET_UNAVAILABLE:
        return []
    need = {str(x) for x in (require_known_nutrients or ())}
    ex = {str(a).lower() for a in exclude_allergens or ()}
    # [P1-PLAN-LOTE-3 · 2026-09-11 · B5] `diet.exclusions` (los «no me gusta» del formulario) llegaban al prompt
    # como texto y a ningún filtro: el CandidateSet fijado al run podía traer el alimento que el usuario
    # pidió no ver. Identidad por `pantry_names_match` (case/acentos/plural, por token completo) — la misma
    # regla de la Nevera — y nunca por subcadena («res» ⊂ «queso fresco»).
    excl_foods = [str(x).strip() for x in (exclude_foods or ()) if str(x or "").strip()]
    _pnm = None
    if excl_foods:
        try:
            from constants import pantry_names_match as _pnm
        except Exception:
            _pnm = None
    slot_es = canonical_slot_es(slot)
    out = []
    compatibles = []   # [P1-CANDIDATO-CON-PRECIO] la plantilla ENTERA, en paralelo a `out`: costear
                       # necesita `constituents`, que el dict del candidato no lleva (ni debe llevar)
    try:
        from horizon import family_matches_template
    except Exception:
        family_matches_template = None
    for t in snap.get("templates") or []:
        if t.get("status") != "ok" or slot_es not in (t.get("slots") or []):
            continue
        if ex and ex.intersection({a.lower() for a in (t.get("intrinsic_risk_attributes") or {}).get("allergens", [])}):
            continue
        if diet_canon and _template_violates_diet(t, diet_canon):
            continue
        if need and need.intersection(t.get("nutrition_unknown") or {}):
            continue
        if market_country and not _buyable(t, market_country):
            continue
        if excl_foods and _template_uses_excluded_food(t, excl_foods, _pnm):
            continue
        if family and family_matches_template is not None:
            prot = str(t.get("protein") or "none").lower()
            if prot not in ("none", "mixta") and not family_matches_template(
                    family, prot, [c.get("canonical") or c.get("name") for c in (t.get("constituents") or [])]):
                continue
        # [F7-G] compra única: el plato debe aguantar hasta ese día del ciclo (sin congelador, o congelando si el modo lo permite)
        if need_days:
            _lg = t.get("logistics") or {}
            if not _pd.template_fits(_lg.get("days_fresh_min"), _lg.get("days_with_freezer_min"), int(need_days), bool(allow_frozen)):
                continue
        out.append({"template_id": t["template_id"], "name": t["name"], "protein": t.get("protein"),
                    "technique": t.get("technique"), "transform": t.get("transform"),
                    "logistics": t.get("logistics") or {}, "pantry_only": bool((t.get("logistics") or {}).get("pantry_only"))})
        compatibles.append(t)
    # [P1-CANDIDATO-CON-PRECIO · 2026-09-09] La poda va AQUÍ: antes del orden por hash y del corte a
    # `k`. Podar después dejaría fuera candidatos baratos sólo por su posición en la lista. El coste
    # se calcula, se poda y SE DESCARTA — no entra en el dict del candidato, porque los candidatos se
    # fijan al run y entran en `slice_hash` → `input_hash`: un precio ahí ataría la huella de un plan
    # al cron de inflación. Fail-open: cualquier fallo del catálogo deja la conducta previa.
    if budget_tier and out:
        try:
            import dish_cost as _dc
            if _dc.price_filter_enabled() and _dc.fraccion_asequible(budget_tier) is not None:
                _precios = _dc.tabla_de_precios()
                out = _dc.poda_por_presupuesto(
                    [(c, _dc.costo_racion(t, _precios)) for c, t in zip(out, compatibles)], budget_tier)
        except Exception as _e_precio:
            logger.debug(f"[P1-CANDIDATO-CON-PRECIO] sin poda por precio ({_e_precio!r}): conducta previa")
    # [ARQ27-P1-04] Orden por hash de CONTENIDO, no por posición en el fichero. Sin corte temprano: hay
    # que ver todas las compatibles para poder ordenarlas, y una biblioteca son ~100 plantillas.
    _salt = f"{library_for_country(country)}:{slot_es}:{_norm(family) if family else ''}"
    out.sort(key=lambda c: hashlib.sha256(
        f"{_salt}:{c['template_id']}".encode("utf-8")).hexdigest())
    if rotate and out:
        _r = int(rotate) % len(out)
        out = out[_r:] + out[:_r]
    if prefer_batch:
        # [P1-STEP14-SHOPPING-COOKING] «Cocino por tandas»: primero las plantillas que rinden para varios días
        # (`logistics.batch_friendly`). `sort` es estable, así que dentro de cada grupo se conserva el
        # orden por hash de arriba; el corte a `k` va DESPUÉS.
        out.sort(key=lambda c: 0 if (c.get("logistics") or {}).get("batch_friendly") else 1)
    return out[:max(1, int(k))]


__all__ = [
    "REGISTRY_SCHEMA_VERSION", "COMPILER_VERSION", "LIBRARIES", "RISK_THRESHOLDS", "REGISTRY_DIR",
    "registry_snapshot_version", "snapshot_path", "build_catalog_index", "resolve_constituent", "catalog_fingerprint",
    "allergen_classes_for", "derive_risk_attributes", "compile_template", "compile_library", "write_snapshot",
    "templates_by_id",
    "verify_snapshot", "library_for_country", "load_registry", "registry_hash", "template_candidates",
    "derive_logistics", "derive_editorial",
    "minutos_de_los_pasos", "recipe_steps_index",
]
