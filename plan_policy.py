"""[P1-ARQ25-F2-PLANPOLICY · 2026-09-02] Fase 2 del roadmap 2.5: `PlanPolicy` (capa V2.3).

Qué es: la política de planificación que el usuario PIDE (`requested`) y la que el sistema
APLICA (`effective`), con la lista de relajaciones (`relaxations[]`: campo, pedido, aplicado,
reason code, evidencia) para que el frontend pueda explicar «solicitaste X; aplicamos Y por Z».

Qué NO es (todavía): en `shadow` la política se compila, se persiste y se MIDE contra lo que V1
produjo, sin influir en la generación. `enforce` llega con el allocator (Fase 3).

Decisiones del dueño registradas aquí:
  · #4 (2026-09-02): el presupuesto es un LÍMITE DURO donde hay precios (mercado nativo). Si
    el pedido queda por debajo del piso de las metas, el compilador no lo modifica en silencio:
    emite `budget_below_floor` con `action=waiting_user` y la cifra mínima, para que el usuario
    suba el presupuesto o ajuste las metas ANTES de gastar el crédito. En países sin precios el
    presupuesto es orientativo (`budget_advisory_no_prices`), nunca fingido.
  · Los nombres de alimentos siguen siendo el SSOT del motor; `ingredient_id` es una clave
    estable DERIVADA del nombre canónico (slug), nunca un reemplazo.
  · `template_id` se acuña al CARGAR las bibliotecas (hash estable de biblioteca+base+nombre+
    técnica, con tabla de alias para renombres): cero reescritura de los JSON.

Precedencia (§6.3): 1 seguridad clínica y alergias · 2 dieta · 3 disponibilidad en el mercado ·
4 restricciones duras (exclusiones, congelar, presupuesto duro) · 5 anclas y recurrencia ·
6 preferencias suaves · 7 optimización interna. Una restricción dura imposible ⇒ `waiting_user`
con reason code; nunca modificación silenciosa.

Knob: `MEALFIT_PLAN_POLICY_MODE` = off | shadow | enforce (default off).
Doc: backend/docs/plan_policy_f2.md. Test: tests/test_p1_arq25_f2_planpolicy.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

POLICY_SCHEMA_VERSION = 1
COMPILER_VERSION = "2026-09-02.1"
POLICY_MODES = ("off", "shadow", "enforce")
RECURRENCE_MODES = ("routine", "balanced", "explore")
SLOTS = ("breakfast", "lunch", "dinner", "snack")
FREEZER_MODES = ("none", "limited", "full")
BATCH_MODES = ("never", "sometimes", "often")
MAX_ANCHORS = 8
# [P1-ARQ25-F4-FORM · 2026-09-03] Lo que el formulario progresivo (Fase 4) escribe además del
# V1. Su presencia marca `source.form_version = "v2"`; su ausencia cae a los defaults V1, así que
# un cliente viejo sigue produciendo la misma política que ayer.
FORM_V2_FIELDS = ("mealOrganization", "freezerMode", "freshTopup", "batchCooking", "stapleAnchors", "cultureProfiles")  # [P1-ARQ25-F7-CULTURE] cultureProfiles
PREPARATION_MODES = ("vary_preparation", "same_preparation")

_SLOT_ALIASES = {
    "desayuno": "breakfast", "breakfast": "breakfast",
    "almuerzo": "lunch", "lunch": "lunch", "comida": "lunch",
    "cena": "dinner", "dinner": "dinner",
    "merienda": "snack", "snack": "snack", "colacion": "snack", "colación": "snack",
}
_NONE_SENTINELS = {"", "ninguna", "ninguno", "none", "no", "n/a", "na", "nada", "sin"}
_CYCLE_DAYS_FALLBACK = {"weekly": 7, "biweekly": 15, "monthly": 30}
def _culture_weights_for(form: dict, cc: str) -> list:
    try:
        from cultural_profiles import culture_weights_for_form
        return culture_weights_for_form({**(form or {}), "country": cc})
    except Exception:
        return [{"profile_id": _PROFILE_BY_COUNTRY.get(cc, "dominican_criolla"), "weight": 1.0}]


_PROFILE_BY_COUNTRY = {
    "DO": "dominican_criolla", "US": "us_everyday", "ES": "spain_mediterranea",
    "MX": "mexico_casera", "PR": "puertorico_criolla", "CO": "colombia_casera",
}
# Clases de proteína animal para la dieta (nombres canónicos del motor, en español, sin acentos).
_MEAT = ("pollo", "res", "carne", "cerdo", "puerco", "pavo", "chivo", "cordero", "salami", "jamon",
         "tocino", "chorizo", "longaniza", "mortadela", "salchicha", "pechuga", "muslo", "bistec")
_FISH = ("pescado", "atun", "salmon", "bacalao", "sardina", "camaron", "camarones", "mariscos",
         "pulpo", "calamar", "cangrejo", "langosta", "tilapia", "merluza", "dorado", "chillo")
_ANIMAL_OTHER = ("huevo", "huevos", "leche", "queso", "yogurt", "yogur", "mantequilla", "miel", "crema")
# Clases de alérgenos declaradas (formulario) → tokens que las nombran en los alimentos.
_ALLERGEN_CLASS_TOKENS = {
    "lacteos": ("leche", "queso", "yogurt", "yogur", "mantequilla", "crema", "requeson", "lacteo"),
    "mariscos": ("camaron", "camarones", "marisco", "mariscos", "pulpo", "calamar", "cangrejo", "langosta", "mejillon", "almeja"),
    "pescado": ("pescado", "atun", "salmon", "bacalao", "sardina", "tilapia", "merluza", "dorado", "chillo"),
    "mani": ("mani", "cacahuate", "cacahuete"),
    "frutos secos": ("almendra", "almendras", "nuez", "nueces", "merey", "anacardo", "pistacho", "avellana", "pecana"),
    "huevo": ("huevo", "huevos"),
    "gluten": ("trigo", "pan", "pasta", "harina", "avena", "cebada", "centeno", "galleta", "tortilla de trigo", "bulgur"),
    "soya": ("soya", "soja", "tofu", "edamame", "tempeh"),
}


# ═══════════════════════════════════════════════════════ knob
def policy_mode() -> str:
    try:
        from knobs import _env_str
        return _env_str("MEALFIT_PLAN_POLICY_MODE", "off", choices=set(POLICY_MODES))
    except Exception:
        raw = str(os.environ.get("MEALFIT_PLAN_POLICY_MODE", "off") or "off").strip().lower()
        return raw if raw in POLICY_MODES else "off"


def policy_active() -> bool:
    return policy_mode() != "off"


# ═══════════════════════════════════════════════════════ ids estables
def _strip_accents(s: str) -> str:
    try:
        from constants import strip_accents
        return strip_accents(s)
    except Exception:
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", _strip_accents(str(s or "")).strip().lower())


def ingredient_id_for(name: Any) -> str:
    """Clave estable derivada del nombre canónico: minúsculas, sin acentos, `_` entre tokens."""
    n = _norm(name)
    n = re.sub(r"[^a-z0-9]+", "_", n).strip("_")
    return n or "unknown"


def canonical_name_for(ingredient_id: str, catalog_names: Optional[Iterable[str]] = None,
                       catalog_rows: Optional[Iterable[dict]] = None) -> Optional[str]:
    """Nombre canónico del motor para un `ingredient_id`.

    Primero por igualdad de slug contra los nombres. Si no casa y se pasan filas del catálogo, se
    prueba contra sus ALIAS.

    [P1-ARQ27-F2-IDENTIDAD · 2026-09-06] El puente de alias que el gap pedía como rollback. El id
    se deriva del NOMBRE (`ingredient_id_for`), así que renombrar un alimento en el catálogo
    huérfana todo lo que ya lo citaba — y un id acuñado desde un alias no resolvía nunca. Medido:
    1.137 alias del catálogo producen ids que no casan con ningún nombre canónico.

    Daño VIVO hoy: cero. Solo hay 2 `ingredient_id` persistidos en 96 planes y los dos son
    canónicos. Es una fragilidad latente, no un incendio — y por eso el puente es aditivo: no
    fusiona filas ni cambia ningún id existente.

    Los alias NO crean alimentos: varios alias resuelven al MISMO nombre canónico, que es lo que
    el criterio «alias no aumenta alimentos únicos» exige."""
    if not ingredient_id:
        return None
    for n in (catalog_names or []):
        if ingredient_id_for(n) == ingredient_id:
            return str(n)
    for row in (catalog_rows or []):
        if not isinstance(row, dict):
            continue
        for a in (row.get("aliases") or []):
            if ingredient_id_for(a) == ingredient_id:
                return str(row.get("name") or "") or None
    return None


# ── template_id ─────────────────────────────────────────────────────────────
# Renombres: nombre ACTUAL → nombre con el que se acuñó el id (así un renombre no cambia el id).
# [P1-PLAN-LOTE-11 · 2026-09-11 · C8] Los tres renombres que decidió el dueño al curar las plantillas sin receta:
# un plato SUSTITUIDO lleva el nombre de lo que de verdad trae (jamón de pavo, filete de pescado blanco) y la menta,
# que no se compra, sale del título. Conservan su `template_id` histórico: la biblioteca de recetas, los
# blueprints de los planes vivos y las fichas de curación lo citan. tooltip-anchor: TEMPLATE_ALIASES (test_p1_plan_lote_11.py)
TEMPLATE_ALIASES: dict[str, str] = {
    "Frutas picadas con limón": "Frutas picadas con limón y menta",
    "Mangú con jamón de pavo a la plancha": "Mangú con salami de pavo a la plancha (versión magra)",
    "Filete de pescado blanco al horno con vegetales y batata": "Chillo al horno con vegetales y batata asada",
    # [P1-PLAN-LOTE-16 · 2026-09-12 · A9] Dos renombres del dueño en la ficha de los 28 platos pendientes: la «sal
    # mínima» no es parte del nombre de un plato, y los macarrones eran coditos con gouda (el catálogo no tiene «queso
    # curado»; el título prometía lo que no traía).
    "Huevo duro con aguacate": "Huevo duro con aguacate y sal mínima",
    "Coditos con salsa de tomate y queso gouda": "Macarrones con salsa de tomate y queso curado",
}
# [P1-PLAN-LOTE-16 · 2026-09-12 · A9] La TÉCNICA también entra en el id (`mint_template_id`), así que corregirla lo
# cambiaría igual que un renombre. El dueño corrigió 16 técnicas en la ficha A9 («crudo» no describe un plato con
# casabe tostado; «batido» esconde que las claras se cuecen). Nombre ACTUAL → técnica con la que se ACUÑÓ el id.
# tooltip-anchor: TEMPLATE_MINT_TECHNIQUE (test_p1_plan_lote_16.py)
TEMPLATE_MINT_TECHNIQUE: dict[str, str] = {
    "Yogurt griego con avena tostada y maní": "crudo",
    "Batida de leche con claras, avena y guineo": "batido",
    "Queso cottage con casabe, tomate y aguacate": "crudo",
    "Tilapia al horno con yuca y cebolla": "horneado",
    "Atún en agua con casabe y cebolla": "crudo",
    "Yogurt griego con guineo": "crudo",
    "Queso cottage con lechosa": "crudo",
    "Batida de leche con claras y avena": "batido",
    "Sardinas en lata con casabe": "crudo",
    "Queso cottage con casabe y tomate": "crudo",
    "Pechuga desmenuzada con casabe y cebolla": "hervido",
    "Queso de hoja con tomate y cebolla": "crudo",
    "Coditos con salsa de tomate y queso gouda": "guisado",
    "Ensalada de garbanzos con huevo duro, aceitunas y cebolla": "frío",
    "Trinxat de col y patata con huevo": "salteado",
    "Arroz a la cubana con huevo y salsa de tomate": "sartén",
}
TEMPLATE_VERSION = 1


def library_key_for_path(path: str) -> str:
    base = os.path.basename(str(path or "")).lower()
    m = re.match(r"dish_templates(?:_([a-z]{2}))?\.json$", base)
    if not m:
        return "do"
    return (m.group(1) or "do").lower()


def mint_template_id(template: dict, library: str = "do") -> str:
    name = str(template.get("name") or "")
    minted_name = TEMPLATE_ALIASES.get(name, name)
    minted_technique = TEMPLATE_MINT_TECHNIQUE.get(name, template.get("technique"))  # [P1-PLAN-LOTE-16]
    raw = "|".join([
        str(library or "do").lower(),
        _norm(template.get("base")),
        _norm(minted_name),
        _norm(minted_technique),
    ])
    return "tpl_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def attach_template_ids(templates: list, library: str = "do") -> list:
    """Sella `template_id`/`template_version` en cada plantilla (in place). Idempotente."""
    seen: dict[str, str] = {}
    for t in templates or []:
        if not isinstance(t, dict):
            continue
        tid = mint_template_id(t, library)
        if tid in seen and seen[tid] != str(t.get("name")):
            logger.warning(f"[P1-ARQ25-F2-PLANPOLICY] template_id duplicado {tid}: {seen[tid]!r} vs {t.get('name')!r}")
        seen[tid] = str(t.get("name"))
        t["template_id"] = tid
        t["template_version"] = TEMPLATE_VERSION
    return templates


def template_id_coverage() -> dict:
    """Cobertura por biblioteca (las 6): total, con id, únicos. Para el gate de la fase."""
    out = {}
    try:
        import dish_library
        data_dir = os.path.join(os.path.dirname(os.path.abspath(dish_library.__file__)), "data")
        for fn in sorted(os.listdir(data_dir)):
            if not re.match(r"dish_templates(_[a-z]{2})?\.json$", fn):
                continue
            path = os.path.join(data_dir, fn)
            templates = dish_library.load_dish_templates(path)
            ids = [t.get("template_id") for t in templates if isinstance(t, dict)]
            out[library_key_for_path(path)] = {
                "templates": len(templates),
                "with_id": sum(1 for i in ids if i),
                "unique": len(set(i for i in ids if i)),
            }
    except Exception as e:
        logger.warning(f"[P1-ARQ25-F2-PLANPOLICY] cobertura de template_id no disponible: {e}")
    return out


# ═══════════════════════════════════════════════════════ adapters (formulario V1 → requested)
def _clean_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        parts = re.split(r"[,;\n]+", v)
    elif isinstance(v, (list, tuple, set)):
        parts = []
        for x in v:
            if isinstance(x, dict):
                x = x.get("name") or x.get("label") or x.get("value") or ""
            parts.append(str(x))
    else:
        parts = [str(v)]
    out, seen = [], set()
    for p in parts:
        s = str(p).strip()
        if not s or _norm(s) in _NONE_SENTINELS:
            continue
        k = _norm(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _int_or(v: Any, default: int) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return int(default)


def _cycle_days(form: dict) -> int:
    key = str(form.get("groceryDuration") or "weekly").strip().lower()
    try:
        from nutrition_calculator import _GROCERY_DURATION_DAYS
        return int(_GROCERY_DURATION_DAYS.get(key, 7))
    except Exception:
        return int(_CYCLE_DAYS_FALLBACK.get(key, 7))


def _batch_from_cooking_time(v: Any) -> str:
    s = _norm(v)
    if not s:
        return "sometimes"
    if "15" in s or "rapid" in s or "quick" in s:
        return "never"
    if "30" in s or "45" in s:
        return "sometimes"
    return "often"


def _country(form: dict, country: Optional[str]) -> str:
    if country:
        return str(country).upper()
    try:
        from constants import country_for_form_data
        return str(country_for_form_data(form) or "DO").upper()
    except Exception:
        return str(form.get("country") or "DO").upper()


def policy_from_form(form_data: dict, *, country: Optional[str] = None) -> dict:
    """Política SOLICITADA a partir del formulario V1 (adapters, §Fase 2)."""
    form = form_data or {}
    cc = _country(form, country)
    try:
        from constants import canonicalize_diet_type
        diet = canonicalize_diet_type(form.get("dietType"))
    except Exception:
        diet = "balanced"
    allergies = _clean_list(form.get("allergies")) + _clean_list(form.get("otherAllergies"))
    conditions = _clean_list(form.get("medicalConditions")) + _clean_list(form.get("otherConditions"))
    dislikes = _clean_list(form.get("dislikes")) + _clean_list(form.get("otherDislikes"))
    staples = _clean_list(form.get("stapleFoods") if form.get("stapleFoods") is not None else form.get("staple_foods"))
    gm = str(form.get("mealOrganization") or "balanced").strip().lower()
    if gm not in RECURRENCE_MODES:
        gm = "balanced"
    # [P1-ARQ25-F4-FORM] «Mis básicos» como editor de anclas: `stapleAnchors` trae, por básico,
    # franja(s), frecuencia semanal (min/max por 7 días) y misma/variada preparación. Los nombres
    # siguen en `stapleFoods` (SSOT del motor y de la Nevera); aquí solo se enriquecen.
    detail_by_id: dict = {}
    raw_anchors = form.get("stapleAnchors")
    if isinstance(raw_anchors, list):
        for item in raw_anchors:
            if isinstance(item, dict) and item.get("name"):
                detail_by_id[ingredient_id_for(item["name"])] = item
    names = list(staples)
    for item in detail_by_id.values():
        if str(item.get("name")) not in names:
            names.append(str(item.get("name")))
    anchors, seen = [], set()
    for s in names:
        iid = ingredient_id_for(s)
        if iid in seen:
            continue
        seen.add(iid)
        d = detail_by_id.get(iid) or {}
        slots: list = []
        for sl in (d.get("slots") or []):
            canon = _SLOT_ALIASES.get(_norm(sl))
            if canon and canon not in slots:
                slots.append(canon)
        lo, hi = _int_or(d.get("min_per_7d"), 2), _int_or(d.get("max_per_7d"), 7)
        lo, hi = max(0, min(7, lo)), max(0, min(7, hi))
        if lo > hi:
            lo, hi = hi, lo
        prep = str(d.get("preparation_mode") or "vary_preparation")
        if prep not in PREPARATION_MODES:
            prep = "vary_preparation"
        anchors.append({
            "ingredient_id": iid, "name": s, "slots": slots,
            "min_per_7d": lo, "max_per_7d": hi, "preparation_mode": prep,
            # [P1-ANCHOR-PORTION · 2026-09-07] CUÁNTO, no solo cuándo y cuán a menudo.
            "portion": _portion_of(d),
        })
    cycle = _cycle_days(form)
    freezer = str(form.get("freezerMode") or "limited").strip().lower()
    if freezer not in FREEZER_MODES:
        freezer = "limited"
    batch = str(form.get("batchCooking") or "").strip().lower()
    if batch not in BATCH_MODES:
        batch = _batch_from_cooking_time(form.get("cookingTime"))
    topup_raw = str(form.get("freshTopup") or "").strip().lower()
    fresh_topup_days = None if cycle <= 7 else (None if topup_raw in ("no", "false", "0") else 7)
    tier = str(form.get("budget") or "").strip().lower() or None
    amount = form.get("budgetAmount")
    try:
        amount = float(amount) if amount not in (None, "", "null") else None
    except (TypeError, ValueError):
        amount = None
    currency = str(form.get("budgetCurrency") or "").strip().upper() or None
    try:
        household = max(1, min(12, int(float(form.get("householdSize") or 1))))
    except (TypeError, ValueError):
        household = 1
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "market_country": cc,
        "recurrence": {"global_mode": gm, "slot_modes": {s: gm for s in SLOTS}},
        "food_anchors": anchors,
        "shopping": {
            "main_cycle_days": cycle, "fresh_topup_days": fresh_topup_days,
            "freezer_mode": freezer, "batch_cooking": batch,
        },
        "diet": {"type": diet, "allergies": allergies, "exclusions": dislikes},
        "clinical": {"conditions": conditions},
        "budget": {"tier": tier, "amount": amount, "currency": currency, "period_days": cycle, "mode": "hard"},
        "household_size": household,
        # [P1-ARQ25-F7-CULTURE · 2026-09-05] cultura ≠ mercado (I16): elección explícita del usuario (principal +
        # hasta 2 secundarias) o, sin elección, la cocina del país de compra — SUGERIDA, no inferida como identidad.
        "culture_weights": _culture_weights_for(form, cc),
        "source": {
            "form_version": "v2" if any(form.get(k) not in (None, "", [], {}) for k in FORM_V2_FIELDS) else "v1",
            "adapter": COMPILER_VERSION,
        },
    }


# ═══════════════════════════════════════════════════════ compiler (requested → effective)
def _stem(t: str) -> str:
    t = str(t or "")
    if len(t) > 4 and t.endswith("es"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def _matches(a: str, b: str) -> bool:
    try:
        from constants import pantry_names_match
        # sin alias del catálogo: el compilador es puro (sin DB) y determinista
        if pantry_names_match(str(a), str(b), use_catalog_aliases=False):
            return True
    except Exception:
        pass
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # plural/singular español sencillo: «huevos revueltos» casa con «huevo»
    sa, sb = {_stem(t) for t in na.split()}, {_stem(t) for t in nb.split()}
    return _stem(na) in sb or _stem(nb) in sa


def _allergen_tokens_for(allergy: str) -> tuple:
    n = _norm(allergy)
    for cls, toks in _ALLERGEN_CLASS_TOKENS.items():
        if cls in n or any(t == n for t in toks):
            return toks
    return (n,) if n else ()


def _anchor_hits_allergy(anchor_name: str, allergies: list[str]) -> Optional[str]:
    an = _norm(anchor_name)
    stems = {_stem(t) for t in an.split()}
    for a in allergies:
        for tok in _allergen_tokens_for(a):
            if tok and (tok in an.split() or _stem(tok) in stems or an.startswith(tok) or _matches(anchor_name, tok)):
                return a
    return None


def _anchor_hits_diet(anchor_name: str, diet: str) -> bool:
    an = _norm(anchor_name)
    toks = an.split()

    def _in(group):
        return any(g == t or t.startswith(g) for g in group for t in toks)

    if diet == "vegan":
        return _in(_MEAT) or _in(_FISH) or _in(_ANIMAL_OTHER)
    if diet == "vegetarian":
        return _in(_MEAT) or _in(_FISH)
    if diet == "pescatarian":
        return _in(_MEAT)
    return False


# ─────────────────────────────────────────────────────────────────────────────────────────────
# La RACIÓN de un ancla
#
# [P1-ANCHOR-PORTION · 2026-09-07] `stapleAnchors` sabía decir *cuándo* (franjas) y *cuán a
# menudo* (min/max por 7 días), pero no **cuánto**. Sin ese dato, «desayuno 10 claras» no tiene
# dónde entrar: no es una alergia, ni un disgusto, ni un básico más — es una CANTIDAD.
#
# Medido antes de escribirlo, sobre 96 planes vivos: las claras entregadas son 1, 2, 3, 4, 5 y 6,
# y ahí se cortan en seco. Esa distribución que muere justo en el tope es la firma de
# `MAX_EGG_WHITES_PER_MEAL = 6`; nadie ha recibido nunca más, pidiera lo que pidiera.
#
# Este P-fix **solo transporta la intención**: los topes siguen recortando exactamente igual. Lo
# que cambia es que el recorte deja de ser mudo — se registra como relajación «pediste N,
# aplicamos M», que es la mitad del daño. Honrar la petición es el paso siguiente y toca cinco
# capas (prompt, dos caps, solver y gate de variedad); mezclarlo aquí sería subir un tope global
# y dar 10 claras a quien no las pidió.
#
# LA UNIDAD ES OBLIGATORIA, y no es burocracia: sin ella «150» de pollo se leería como 150
# unidades. Es la lección de `P1-UNKNOWN-UNIT-NOT-WHOLE` — una unidad desconocida no es una
# unidad entera. Sin unidad se descarta la ración y se dice por qué.
_PORTION_MAX = 100.0          # cota anti-basura, NO un límite clínico (el encargo lo excluye)


def _portion_of(detail: dict) -> Optional[dict]:
    """`{"qty": float, "unit": str}` de un item de `stapleAnchors`, o `None` si no lo trae."""
    raw = (detail or {}).get("portion")
    if not isinstance(raw, dict):
        return None
    try:
        qty = float(raw.get("qty"))
    except (TypeError, ValueError):
        return None
    unit = str(raw.get("unit") or "").strip().lower()
    if not unit or qty <= 0:
        return None
    return {"qty": qty, "unit": unit}


def _relax(rels: list, *, field: str, requested: Any, applied: Any, reason: str, rank: int,
           evidence: Optional[dict] = None, action: str = "applied") -> None:
    rels.append({
        "field": field, "requested": requested, "applied": applied,
        "reason_code": reason, "rank": rank, "action": action, "evidence": evidence or {},
    })


def compile_policy(requested: dict, *, context: Optional[dict] = None) -> tuple[dict, list]:
    """Aplica la precedencia §6.3 y devuelve `(effective, relaxations)`.

    `context` (todo opcional): `budget_floor_dop` (float), `budget_reference_dop` (float),
    `pricing_mode` ('native' | 'beta_no_prices'), `known_ingredients` (iterable de nombres del
    catálogo del país). Sin contexto, los pasos que lo necesitan se anotan como omitidos: el
    compilador nunca inventa evidencia.
    """
    ctx = context or {}
    eff = json.loads(json.dumps(requested or {}, default=str))
    rels: list = []
    diet = str((eff.get("diet") or {}).get("type") or "balanced")
    allergies = list((eff.get("diet") or {}).get("allergies") or [])
    anchors = list(eff.get("food_anchors") or [])

    # 1. seguridad clínica y alergias
    kept = []
    for a in anchors:
        hit = _anchor_hits_allergy(str(a.get("name") or a.get("ingredient_id")), allergies)
        if hit:
            _relax(rels, field="food_anchors", requested=a.get("name"), applied=None,
                   reason="anchor_conflicts_allergy", rank=1, evidence={"allergy": hit})
        else:
            kept.append(a)
    anchors = kept
    # 2. dieta
    kept = []
    for a in anchors:
        if _anchor_hits_diet(str(a.get("name") or a.get("ingredient_id")), diet):
            _relax(rels, field="food_anchors", requested=a.get("name"), applied=None,
                   reason="anchor_conflicts_diet", rank=2, evidence={"diet": diet})
        else:
            kept.append(a)
    anchors = kept
    # 3. disponibilidad real en el mercado
    known = ctx.get("known_ingredients")
    if known is not None:
        # [ARQ27-P1-07] Constancia POSITIVA de que el paso corrió. Sin ella, un plan sin anclas
        # descartadas es indistinguible de un plan cuyo mercado nadie comprobó — y el embudo no
        # puede medir la diferencia entre «pasó el filtro» y «no hubo filtro».
        eff.setdefault("notes", []).append("market_check_applied")
        known_list = [str(k) for k in known]
        kept = []
        for a in anchors:
            name = str(a.get("name") or "")
            if any(_matches(name, k) for k in known_list):
                kept.append(a)
            else:
                _relax(rels, field="food_anchors", requested=name, applied=None,
                       reason="anchor_not_in_market", rank=3,
                       evidence={"market_country": eff.get("market_country")})
        anchors = kept
    else:
        eff.setdefault("notes", []).append("market_check_skipped")
    # 4. restricciones duras: presupuesto (decisión #4) y congelador/reposición
    budget = dict(eff.get("budget") or {})
    pricing_mode = ctx.get("pricing_mode")
    if pricing_mode == "beta_no_prices":
        if budget.get("mode") == "hard":
            _relax(rels, field="budget.mode", requested="hard", applied="advisory",
                   reason="budget_advisory_no_prices", rank=4,
                   evidence={"market_country": eff.get("market_country")})
        budget["mode"] = "advisory"
    floor = ctx.get("budget_floor_dop")
    ref = ctx.get("budget_reference_dop")
    if floor is not None:
        try:
            budget["floor_dop"] = round(float(floor), 2)
        except (TypeError, ValueError):
            pass
    if ref is not None:
        try:
            budget["reference_dop"] = round(float(ref), 2)
        except (TypeError, ValueError):
            pass
    amount_dop = ctx.get("budget_amount_dop")
    if (budget.get("mode") == "hard" and amount_dop is not None and floor is not None
            and float(amount_dop) < float(floor)):
        # Nunca se modifica en silencio: la cifra se conserva y el usuario decide.
        budget["status"] = "below_floor"
        _relax(rels, field="budget.amount", requested=budget.get("amount"), applied=budget.get("amount"),
               reason="budget_below_floor", rank=4, action="waiting_user",
               evidence={"floor_dop": round(float(floor), 2), "amount_dop": round(float(amount_dop), 2)})
    else:
        budget["status"] = "ok" if budget.get("mode") == "hard" else "advisory"
    eff["budget"] = budget
    shopping = dict(eff.get("shopping") or {})
    # [P1-ARQ25-F7-CULTURE · subfase G] Sin congelador ni reposición en un ciclo > 7 días ya NO se acorta el ciclo:
    # el usuario pidió UNA compra y el motor la respeta con proteínas de despensa (huevo, enlatados, legumbres, queso
    # curado) desde el día 8 — el registry filtra por durabilidad y el validador lo vigila. Se declara, no se cambia.
    if (int(shopping.get("main_cycle_days") or 7) > 7 and shopping.get("freezer_mode") == "none"
            and not shopping.get("fresh_topup_days")):
        _relax(rels, field="shopping.freezer_mode", requested="none", applied="none",
               reason="pantry_proteins_after_first_week", rank=4, action="applied", evidence={"fresh_days": 7})
    eff["shopping"] = shopping
    # 5. anclas y recurrencia
    for a in anchors:
        lo = int(a.get("min_per_7d") if a.get("min_per_7d") is not None else 0)
        hi = int(a.get("max_per_7d") if a.get("max_per_7d") is not None else 7)
        lo2, hi2 = max(0, min(7, lo)), max(0, min(7, hi))
        if lo2 > hi2:
            lo2 = hi2
        if (lo2, hi2) != (lo, hi):
            _relax(rels, field=f"food_anchors[{a.get('ingredient_id')}].per_7d", requested=[lo, hi],
                   applied=[lo2, hi2], reason="recurrence_clamped", rank=5)
        a["min_per_7d"], a["max_per_7d"] = lo2, hi2
        # [P1-ANCHOR-PORTION · 2026-09-07] La ración sobrevive a `compile_policy` o se cae
        # DICIÉNDOLO. Una petición descartada en silencio es la que hace que el usuario repita
        # el formulario creyendo que no le hicieron caso.
        _p = a.get("portion")
        if _p is not None:
            _pn = _portion_of({"portion": _p})
            if _pn is None:
                _relax(rels, field=f"food_anchors[{a.get('ingredient_id')}].portion",
                       requested=_p, applied=None, reason="portion_invalid", rank=5)
            elif _pn["qty"] > _PORTION_MAX:
                _relax(rels, field=f"food_anchors[{a.get('ingredient_id')}].portion",
                       requested=_pn, applied={"qty": _PORTION_MAX, "unit": _pn["unit"]},
                       reason="portion_out_of_range", rank=5)
                _pn = {"qty": _PORTION_MAX, "unit": _pn["unit"]}
            a["portion"] = _pn
        a["slots"] = [_SLOT_ALIASES.get(_norm(s), _norm(s)) for s in (a.get("slots") or []) if _norm(s)]
        if a.get("preparation_mode") not in ("vary_preparation", "same_preparation"):
            a["preparation_mode"] = "vary_preparation"
    if len(anchors) > MAX_ANCHORS:
        _relax(rels, field="food_anchors", requested=len(anchors), applied=MAX_ANCHORS,
               reason="anchors_capped", rank=5, evidence={"dropped": [a.get("name") for a in anchors[MAX_ANCHORS:]]})
        anchors = anchors[:MAX_ANCHORS]
    anchors.sort(key=lambda a: str(a.get("ingredient_id") or ""))
    eff["food_anchors"] = anchors
    rec = dict(eff.get("recurrence") or {})
    if rec.get("global_mode") not in RECURRENCE_MODES:
        rec["global_mode"] = "balanced"
    rec["slot_modes"] = {s: (rec.get("slot_modes") or {}).get(s, rec["global_mode"]) for s in SLOTS}
    eff["recurrence"] = rec
    # 6-7. preferencias suaves / optimización interna: sin reglas todavía (Fase 3)
    eff["schema_version"] = POLICY_SCHEMA_VERSION
    eff["compiler_version"] = COMPILER_VERSION
    eff["policy_hash"] = policy_hash(eff)
    return eff, rels


_HASH_VOLATILE = {"policy_hash", "compiled_at", "notes", "source"}


def policy_hash(policy: dict) -> str:
    """sha256 del JSON canónico (claves ordenadas, sin campos volátiles). Misma entrada ⇒ mismo hash."""
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in sorted(o.items()) if k not in _HASH_VOLATILE}
        if isinstance(o, list):
            items = [_clean(v) for v in o]
            # el orden de una lista no es semántica de la política (anclas, alergias, exclusiones)
            return sorted(items, key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False, default=str))
        return o
    canon = json.dumps(_clean(policy or {}), sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


_REASON_COPY = {
    "anchor_conflicts_allergy": "Quitamos «{requested}» de tus básicos: choca con una alergia declarada ({allergy}).",
    "anchor_conflicts_diet": "Quitamos «{requested}» de tus básicos: no encaja con tu dieta ({diet}).",
    "anchor_not_in_market": "«{requested}» no está en el catálogo de tu país; no lo usamos como básico.",
    "budget_advisory_no_prices": "En tu país aún no hay precios: el presupuesto es orientativo, no un límite.",
    "budget_below_floor": "Tu presupuesto ({amount_dop}) está por debajo del mínimo para un plan que cumpla tus metas ({floor_dop}). Súbelo o ajusta las metas.",
    "cycle_shortened_no_freezer_no_topup": "Sin congelador ni reposición de frescos, el ciclo de compra pasa a 7 días.",
    "pantry_proteins_after_first_week": "Sin congelador ni reposición de frescos: la proteína fresca es para la primera semana; después huevos, enlatados, legumbres y queso curado.",
    "recurrence_clamped": "La frecuencia pedida se ajustó al rango posible (0–7 por semana).",
    "anchors_capped": "Solo los primeros {applied} básicos se usan como anclas.",
    # [P1-ANCHOR-PORTION · 2026-09-07] El usuario ve QUÉ pidió y QUÉ se aplicó. Sin este copy la
    # relajación existiría en el jsonb y no llegaría al panel «solicitaste / aplicamos / por qué».
    "portion_invalid": "No entendimos la cantidad que pediste para este básico (falta la unidad "
                       "o no es un número), así que usamos la ración normal.",
    "portion_out_of_range": "La cantidad pedida es demasiado alta para una ración; la ajustamos.",
}


def explain_relaxations(relaxations: list) -> list[str]:
    out = []
    for r in relaxations or []:
        tpl = _REASON_COPY.get(str(r.get("reason_code")), "{reason_code}")
        ev = dict(r.get("evidence") or {})
        try:
            out.append(tpl.format(requested=r.get("requested"), applied=r.get("applied"),
                                  reason_code=r.get("reason_code"), **ev))
        except (KeyError, IndexError):
            out.append(str(r.get("reason_code")))
    return out


# ═══════════════════════════════════════════════════════ compilación desde el formulario (con contexto real)
def compile_from_form(form_data: dict, *, country: Optional[str] = None) -> dict:
    """requested + effective + relaxations + hash, con el contexto que el sistema ya conoce
    (piso de presupuesto, referencia, modo de precios). Nunca lanza: fail-open a política vacía."""
    try:
        requested = policy_from_form(form_data, country=country)
        ctx: dict = {}
        try:
            from constants import pricing_mode_for_form_data
            ctx["pricing_mode"] = pricing_mode_for_form_data(form_data)
        except Exception:
            pass
        try:
            from nutrition_calculator import min_budget_for_goals, build_budget_reference, _budget_usd_to_dop
            info = min_budget_for_goals(form_data) or {}
            if info.get("min_budget_dop") is not None:
                ctx["budget_floor_dop"] = float(info["min_budget_dop"])
            ref = build_budget_reference(form_data)
            if isinstance(ref, dict) and ref.get("reference_rd") is not None:
                ctx["budget_reference_dop"] = float(ref["reference_rd"])
            b = requested.get("budget") or {}
            if b.get("tier") == "custom" and b.get("amount") is not None:
                ctx["budget_amount_dop"] = float(b["amount"]) * (_budget_usd_to_dop() if b.get("currency") == "USD" else 1.0)
        except Exception as e:
            logger.debug(f"[P1-ARQ25-F2-PLANPOLICY] contexto de presupuesto no disponible: {e}")
        # [ARQ27-P1-07 · 2026-09-06] El paso 3 de `compile_policy` (disponibilidad real en el
        # mercado) pedía `known_ingredients` y NADIE se lo pasaba: el único constructor que usa
        # producción dejaba el contexto vacío, así que el paso se anotaba `market_check_skipped` y no
        # corría nunca — 57 de 57 planes según el registro de F3. `None` (catálogo ilegible o vacío)
        # conserva ese `skipped` a propósito: ausente no es lo mismo que «este país no vende nada».
        try:
            from catalog_capability import known_ingredient_names
            _known = known_ingredient_names((requested or {}).get("market_country") or country)
            if _known:
                ctx["known_ingredients"] = _known
        except Exception as e:
            logger.debug(f"[ARQ27-P1-07] capacidad de mercado no disponible: {e}")
        effective, rels = compile_policy(requested, context=ctx)
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "compiler_version": COMPILER_VERSION,
            "requested": requested,
            "effective": effective,
            "relaxations": rels,
            "policy_hash": effective.get("policy_hash"),
            "compiled_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.warning(f"[P1-ARQ25-F2-PLANPOLICY] compile_from_form falló (fail-open): {type(e).__name__}: {e}")
        return {"schema_version": POLICY_SCHEMA_VERSION, "compiler_version": COMPILER_VERSION,
                "requested": {}, "effective": {}, "relaxations": [], "policy_hash": None, "error": type(e).__name__}


# ═══════════════════════════════════════════════════════ shadow: distancia política ↔ plan V1
def _ingredient_names(meal: dict) -> list[str]:
    out = []
    for i in (meal.get("ingredients") or []):
        if isinstance(i, dict):
            n = i.get("name") or i.get("item")
        else:
            n = i
        if n:
            out.append(str(n))
    return out


def measure_plan_against_policy(plan_data: dict, effective: dict, *, total_days_requested: Optional[int] = None) -> dict:
    """Mide, sin influir, cuánto se aleja el plan producido por V1 de la política efectiva.

    `total_days_requested`: el postprocess sella ANTES de que el campo viva en `plan_data`
    (medido en prod 2026-09-02: `cycle_match=None` en el primer plan shadow); el caller lo pasa."""
    days = [d for d in ((plan_data or {}).get("days") or []) if isinstance(d, dict)]
    n_days = len(days)
    per_day = [set(n for m in (d.get("meals") or []) if isinstance(m, dict) for n in _ingredient_names(m)) for d in days]
    anchors_out, scores = [], []
    for a in (effective or {}).get("food_anchors") or []:
        name = str(a.get("name") or a.get("ingredient_id"))
        present = sum(1 for names in per_day if any(_matches(name, n) or ingredient_id_for(n) == a.get("ingredient_id") for n in names))
        scaled = round(present * 7.0 / n_days, 2) if n_days else 0.0
        lo, hi = a.get("min_per_7d", 0), a.get("max_per_7d", 7)
        ok = (lo <= scaled <= hi) if n_days else None
        anchors_out.append({"ingredient_id": a.get("ingredient_id"), "days_present": present, "per_7d": scaled, "ok": ok})
        if ok is not None:
            scores.append(1.0 if ok else 0.0)
    anchor_cov = round(sum(scores) / len(scores), 3) if scores else None
    excl = list(((effective or {}).get("diet") or {}).get("exclusions") or []) + list(((effective or {}).get("diet") or {}).get("allergies") or [])
    violations = sorted({n for names in per_day for n in names for x in excl if _anchor_hits_allergy(n, [x]) or _matches(n, x)})
    cycle_req = int(((effective or {}).get("shopping") or {}).get("main_cycle_days") or 0)
    cycle_plan = int(total_days_requested or (plan_data or {}).get("total_days_requested") or 0)
    cycle_match = (cycle_req == cycle_plan) if (cycle_req and cycle_plan) else None
    br = (plan_data or {}).get("budget_reconciliation") or {}
    budget_over = None
    try:
        if br.get("reference_rd") and br.get("estimated_cycle_rd") is not None:
            budget_over = float(br["estimated_cycle_rd"]) > float(br["reference_rd"])
        elif br.get("status"):
            # forma real del reconcile (P1-BUDGET-RECONCILE): status excedido|dentro|ajustado
            budget_over = str(br["status"]).lower() in ("excedido", "over", "exceeded")
    except (TypeError, ValueError):
        budget_over = None
    # el componente de exclusiones solo cuenta si había algo que excluir (None ≠ 0 falso)
    excl_comp = None if not excl else (0.0 if violations else 1.0)
    comps = [c for c in (anchor_cov, excl_comp, None if cycle_match is None else (1.0 if cycle_match else 0.0),
                         None if budget_over is None else (0.0 if budget_over else 1.0)) if c is not None]
    distance = round(1.0 - (sum(comps) / len(comps)), 3) if comps else None
    return {
        "schema_version": POLICY_SCHEMA_VERSION, "days_measured": n_days, "anchors": anchors_out,
        "anchor_coverage": anchor_cov, "exclusion_violations": violations[:12], "cycle_match": cycle_match,
        "budget_over": budget_over, "distance": distance, "measured_at": datetime.now(timezone.utc).isoformat(),
    }


def stamp_plan_policy(plan_data: dict, form_data: dict, *, country: Optional[str] = None,
                      total_days_requested: Optional[int] = None) -> Optional[dict]:
    """Sella `plan_data['_plan_policy']` (compilación) y `_plan_policy_shadow` (medición).
    Solo cuando el knob no está en `off`. Nunca lanza."""
    if not policy_active() or not isinstance(plan_data, dict):
        return None
    try:
        compiled = compile_from_form(form_data, country=country)
        # [P1-PORTION-HONORED · 2026-09-07] Se sella si la política estaba EN VIGOR para esta
        # persona. Sin este campo, quien lee el plan al persistirlo no puede distinguir un usuario
        # del canary de uno en modo sombra — y aplicaría una excepción de ración a quien solo
        # estaba siendo medido. El dato viaja en `form_data`, pero `form_data` no llega al persist.
        compiled["enforced"] = bool((form_data or {}).get("_policy_enforced"))
        plan_data["_plan_policy"] = compiled
        if compiled.get("effective"):
            plan_data["_plan_policy_shadow"] = measure_plan_against_policy(
                plan_data, compiled["effective"], total_days_requested=total_days_requested)
        return compiled
    except Exception as e:
        logger.warning(f"[P1-ARQ25-F2-PLANPOLICY] stamp_plan_policy falló (fail-open): {e}")
        return None


def emit_policy_shadow_metric(user_id: Optional[str], plan_id: Optional[str], compiled: dict, shadow: Optional[dict]) -> None:
    """`pipeline_metrics` node=`plan_policy_shadow` (best-effort). Base del benchmark de fidelidad."""
    try:
        from db import execute_sql_write
        meta = {
            "plan_id": plan_id, "policy_hash": compiled.get("policy_hash"),
            "relaxations": [r.get("reason_code") for r in (compiled.get("relaxations") or [])],
            "distance": (shadow or {}).get("distance"), "anchor_coverage": (shadow or {}).get("anchor_coverage"),
            "exclusion_violations": len((shadow or {}).get("exclusion_violations") or []),
            "cycle_match": (shadow or {}).get("cycle_match"), "budget_over": (shadow or {}).get("budget_over"),
            "mode": policy_mode(), "compiler_version": COMPILER_VERSION,
        }
        execute_sql_write(
            "INSERT INTO pipeline_metrics (user_id, session_id, node, duration_ms, retries, "
            "tokens_estimated, confidence, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
            (user_id, "__policy__", "plan_policy_shadow", 0, 0, 0, (shadow or {}).get("distance"), json.dumps(meta, default=str)),
        )
    except Exception as e:
        logger.debug(f"[P1-ARQ25-F2-PLANPOLICY] métrica shadow no persistida: {e}")


# [P2-FORM-SNAPSHOT-LOG · 2026-09-05] Lo que el usuario PIDIÓ en el asistente, paso a paso y legible. Nace de una
# pregunta del dueño («¿lo vegetariano lo elegiste a propósito?») que no debería hacer falta: la respuesta está en el
# formulario, y quien revisa un plan debe leerla ANTES de juzgar el plato. Sin datos sensibles de salud detallados
# (peso, edad, condiciones textuales): solo las ELECCIONES del asistente. Puro; nunca lanza.
FORM_CHOICE_FIELDS = (
    "appMode", "planSource", "totalDays", "mainGoal", "goal", "dietType", "allergies", "otherAllergies",
    "medicalConditions", "dislikes", "otherDislikes", "country", "cultureProfiles", "mealOrganization",
    "stapleAnchors", "stapleFoods", "groceryDuration", "freshTopup", "freezerMode", "batchCooking",
    "budget", "budgetAmount", "budgetCurrency", "householdSize", "cookingTime", "activityLevel",
    "scheduleType", "locale",
)


def form_choices(form: Optional[dict]) -> dict:
    """Subconjunto ESTABLE del formulario con las elecciones del asistente (solo claves presentes)."""
    f = form if isinstance(form, dict) else {}
    out = {}
    for k in FORM_CHOICE_FIELDS:
        if k in f and f.get(k) not in (None, "", [], {}):
            v = f.get(k)
            out[k] = v if isinstance(v, (str, int, float, bool, list, dict)) else str(v)
    return out


def form_choices_summary(form: Optional[dict]) -> str:
    """Una línea legible: `dieta=vegetarian · país=US · cocina=dominican_criolla+us_everyday(frecuente) · …`."""
    try:
        f = form if isinstance(form, dict) else {}
        if not f:
            return "(formulario vacío)"
        parts = []
        g = f.get("mainGoal") or f.get("goal")
        if g: parts.append(f"objetivo={g}")
        if f.get("appMode"): parts.append(f"modo={f.get('appMode')}")
        if f.get("totalDays"): parts.append(f"días={f.get('totalDays')}")
        if f.get("dietType"): parts.append(f"dieta={f.get('dietType')}")
        al = [str(x) for x in (f.get("allergies") or []) if x] + ([str(f.get("otherAllergies"))] if f.get("otherAllergies") else [])
        parts.append("alergias=" + (",".join(al) if al else "ninguna"))
        mc = [str(x) for x in (f.get("medicalConditions") or []) if x]
        if mc: parts.append("condiciones=" + ",".join(mc))
        dl = [str(x) for x in (f.get("dislikes") or []) if x]
        if dl: parts.append("no_gusta=" + ",".join(dl))
        if f.get("country"): parts.append(f"país={f.get('country')}")
        cp = f.get("cultureProfiles")
        if isinstance(cp, dict) and cp.get("main"):
            sec = [f"{s.get('profile_id')}({s.get('intensity') or 'ocasional'})" for s in (cp.get("secondary") or []) if isinstance(s, dict) and s.get("profile_id")]
            parts.append("cocina=" + "+".join([str(cp.get("main"))] + sec))
        elif cp:
            parts.append(f"cocina={cp}")
        if f.get("mealOrganization"): parts.append(f"recurrencia={f.get('mealOrganization')}")
        sa = f.get("stapleAnchors") or f.get("stapleFoods")
        if sa: parts.append(f"anclas={sa if isinstance(sa, str) else ','.join(str(x.get('name') if isinstance(x, dict) else x) for x in sa)}")
        shop = [str(f.get(k)) for k in ("groceryDuration", "freshTopup", "freezerMode", "batchCooking") if f.get(k)]
        if shop: parts.append("compra=" + "/".join(shop))
        if f.get("budget"): parts.append(f"presupuesto={f.get('budget')}" + (f"({f.get('budgetAmount')} {f.get('budgetCurrency') or ''})".rstrip() if f.get("budgetAmount") else ""))
        if f.get("householdSize"): parts.append(f"hogar={f.get('householdSize')}")
        if f.get("cookingTime"): parts.append(f"tiempo_cocina={f.get('cookingTime')}")
        if f.get("activityLevel"): parts.append(f"actividad={f.get('activityLevel')}")
        if f.get("locale"): parts.append(f"idioma={f.get('locale')}")
        return " · ".join(parts) if parts else "(formulario vacío)"
    except Exception:
        return "(formulario ilegible)"


# ═══════════════════════════════════════════════════════ el tope efectivo de un alimento contable
#
# [P1-PORTION-HONORED · 2026-09-07] Segunda mitad de P1-ANCHOR-PORTION: la ración pedida deja de
# ser solo telemetría y **levanta el techo** de ese alimento para esa persona.
#
# Tres decisiones que hacen esto seguro, y las tres importan más que el cambio en sí:
#
# 1. **`max(default, pedido)`, nunca `pedido` a secas.** Una petición sube el techo; jamás lo baja.
#    Si alguien escribe «2 claras», el motor conserva su margen de 6 en vez de apretarse.
# 2. **Solo con la política EN VIGOR** (`enforced`). Es el mismo canary que ya gobierna el resto de
#    la Fase 3: la primera persona que reciba 10 claras será una a la que se le encendió a
#    propósito, no toda la base de usuarios de golpe.
# 3. **Solo unidades de PIEZA.** «10 unidades» eleva un tope que cuenta piezas; «150 g» no dice
#    nada sobre cuántas piezas caben, y tratarlo como tal es la trampa de
#    `P1-UNKNOWN-UNIT-NOT-WHOLE` por la puerta de atrás.
#
# Fail-safe absoluto: sin política, sin ancla, con unidad rara o ante cualquier excepción devuelve
# el `default` que le pasan. Este helper no puede EMPEORAR el comportamiento actual de nadie.
_UNIDADES_DE_PIEZA = frozenset({"unidad", "unidades", "ud", "uds", "pieza", "piezas"})


def portion_cap_for(effective: Optional[dict], name: Any, default: float,
                    *, enforced: bool = False) -> float:
    """Techo de PIEZAS para `name`, elevado a la ración que la persona pidió."""
    try:
        if not enforced or not isinstance(effective, dict) or not name:
            return default
        iid = ingredient_id_for(name)
        if not iid:
            return default
        for a in (effective.get("food_anchors") or []):
            if not isinstance(a, dict) or a.get("ingredient_id") != iid:
                continue
            p = a.get("portion")
            if not isinstance(p, dict):
                return default
            if str(p.get("unit") or "").strip().lower() not in _UNIDADES_DE_PIEZA:
                return default
            q = float(p.get("qty") or 0)
            return max(float(default), q) if q > 0 else default
        return default
    except Exception:
        return default


def build_count_caps_override(plan_policy: Optional[dict], base: dict) -> Optional[dict]:
    """Copia de `base` con los techos elevados a las raciones pedidas, o `None` si no aplica.

    [P1-PORTION-HONORED · 2026-09-07] La otra mitad del paso: el techo por conteo que corre AL
    PERSISTIR (`_cap_unrealistic_portions`) no ve `form_data`, así que lee la política sellada en
    el propio plan. Sin ella —o con la política solo en modo sombra— devuelve `None` y el llamador
    usa el diccionario global de siempre.

    Devuelve una COPIA: mutar `_REALISM_COUNT_CAPS` le cambiaría el techo a todo el proceso, que
    es exactamente el fallo que este diseño evita (una petición de una persona no puede tocar el
    plan de otra).

    La clave del diccionario es el sustantivo en SINGULAR y sin acentos que usa el motor
    («clara», «huevo»), no el nombre del catálogo: así lo consulta el cap.
    """
    try:
        if not isinstance(plan_policy, dict) or not plan_policy.get("enforced"):
            return None
        eff = plan_policy.get("effective")
        if not isinstance(eff, dict):
            return None
        fuera = dict(base or {})
        tocado = False
        for a in (eff.get("food_anchors") or []):
            if not isinstance(a, dict):
                continue
            nombre = str(a.get("name") or "")
            # El techo se eleva SOLO para el sustantivo cabecera del ancla. Comparar por
            # subcadena hacía que «Clara de huevo» subiera también el tope de «huevo»: pedir 10
            # claras habría autorizado 10 huevos enteros. Es `"res" ⊂ "fresco"` una vez más, y la
            # cabecera es lo que distingue las tres formas del huevo, que es justo el caso.
            _tok = (_norm(nombre).split() or [""])[0]
            for clave in list(fuera.keys()):
                if clave and clave in (_tok, _tok.rstrip("s")):
                    nuevo = portion_cap_for(eff, nombre, float(fuera[clave]), enforced=True)
                    if nuevo > float(fuera[clave]):
                        fuera[clave] = nuevo
                        tocado = True
        return fuera if tocado else None
    except Exception:
        return None
