"""[P1-ARQ25-F3-HORIZON · 2026-09-02] Fase 3 del roadmap 2.5: Full-Horizon Blueprint + fidelidad.

Qué es: el ALLOCATOR del horizonte (§6.5). Antes de pedirle un solo plato al LLM, reparte los
7/15/30 días de la política efectiva (Fase 2) en un `blueprint` determinista: franjas por día,
anclas y sus cuotas (qué día y en qué franja), familia de proteína por día, límites de repetición
exacta y de ingrediente según la banda de recurrencia, ventanas de frescos/congelación y las
fronteras de los chunks alineadas con `split_with_absorb` (H2: `days_offset`/`days_count`).

Cómo viaja: el blueprint se persiste CON el run (`plan_generation_runs.blueprint`) y cada chunk
recibe una REBANADA inmutable (`pipeline_snapshot["_blueprint_slice"]`), cuyo hash entra en
`input_hash`. Si algo tiene que relajarse, la decisión vuelve al compilador (Fase 2) o a este
allocator: nunca se esconde dentro de un prompt.

Qué cambia según el modo (`MEALFIT_PLAN_POLICY_MODE`):
  · `shadow`: el blueprint se construye, se persiste y se MIDE (`plan_policy_fidelity` en
    `pipeline_metrics`); el seeder y los gates siguen byte a byte como V1.
  · `enforce` (global, o por usuario vía `MEALFIT_PLAN_POLICY_ENFORCE_USERS` = canary «dueño →
    test → flip»): el seeder obedece la rebanada (proteína del día, anclas del día), los prompts
    de TODAS las superficies (§6.6) reciben el bloque 📐 de la política, y los validadores de
    fidelidad (ancla ausente, banda de recurrencia, repetición exacta) SUSTITUYEN a los gates de
    repetición de V1 que contradicen la banda pedida. `MEALFIT_FIDELITY_GATE` = warn | block.

Motivo neutral versionado: la renovación ya no se codifica como `update_reason='variety'`
(«más variedad siempre es mejor», §6.1). El motivo canónico es `renewal.v1`; `variety` se acepta
como alias legado en `is_renewal_reason` para que ningún cliente viejo rompa.

Los nombres de alimentos siguen siendo el SSOT del motor: aquí se usan tal cual salen de la
política (`name`) y se comparan con `plan_policy._matches` — jamás se traducen ni se renombran.

Knobs: MEALFIT_FIDELITY_GATE (warn), MEALFIT_PLAN_POLICY_ENFORCE_USERS (""),
MEALFIT_SHOPPING_PROJECTION_JOBS (True).
Doc: backend/docs/plan_policy_f3.md. Test: tests/test_p1_arq25_f3_horizon.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

BLUEPRINT_SCHEMA_VERSION = 1
ALLOCATOR_VERSION = "2026-09-02.1"
FIDELITY_GATE_MODES = ("warn", "block")
FIDELITY_METRIC_NODE = "plan_policy_fidelity"

# Motivo neutral versionado (§6.1/§6.6): la renovación hereda la política, no pide «variedad».
RENEWAL_REASON = "renewal"
RENEWAL_REASON_VERSION = 1
RENEWAL_REASON_VERSIONED = f"{RENEWAL_REASON}.v{RENEWAL_REASON_VERSION}"
LEGACY_RENEWAL_REASONS = frozenset({"variety"})

# Claves internas que viajan en `form_data` (whitelist `_TRUSTED_INTERNAL_FORM_KEYS`).
BLUEPRINT_SLICE_KEY = "_blueprint_slice"
POLICY_EFFECTIVE_KEY = "_plan_policy_effective"
POLICY_ENFORCED_KEY = "_policy_enforced"
POLICY_DAY_INDEX_KEY = "_policy_day_index"
FIDELITY_REPORT_KEY = "_fidelity_report"

SLOT_ORDER = ("breakfast", "snack", "lunch", "snack", "dinner", "snack")
_SLOT_ES = {"breakfast": "desayuno", "lunch": "almuerzo", "dinner": "cena", "snack": "merienda"}

# Familias de proteína por dieta (nombres tal como aparecen en el catálogo/seeder).
_FAMILIES_BY_DIET = {
    "vegan": ("Lentejas", "Garbanzos", "Habichuelas", "Tofu", "Guandules"),
    "vegetarian": ("Huevo", "Queso", "Lentejas", "Garbanzos", "Yogur", "Habichuelas"),
    "pescatarian": ("Pescado", "Atún", "Huevo", "Camarones", "Lentejas", "Queso"),
    "_default": ("Pollo", "Pescado", "Huevo", "Res", "Cerdo", "Atún", "Pavo", "Lentejas", "Habichuelas"),
}

# Banda de recurrencia → límites de repetición por 7 días (§6.1: «respetar la banda pedida»).
_RECURRENCE_LIMITS = {
    "routine": {"max_exact_repeat_per_7d": 7, "max_ingredient_days_per_7d": 7,
                "protein_pool_size": 3, "same_day_protein_repeat_ok": True},
    "balanced": {"max_exact_repeat_per_7d": 2, "max_ingredient_days_per_7d": 5,
                 "protein_pool_size": 5, "same_day_protein_repeat_ok": False},
    "explore": {"max_exact_repeat_per_7d": 1, "max_ingredient_days_per_7d": 3,
                "protein_pool_size": 9, "same_day_protein_repeat_ok": False},
}

# Ingredientes que no cuentan para «ingrediente repetido demasiados días» (sazón/base de cocina).
_INGREDIENT_DAYS_EXEMPT = frozenset({
    "aceite", "aceite de oliva", "sal", "ajo", "cebolla", "limon", "limón", "agua", "pimienta",
    "oregano", "orégano", "vinagre", "cilantro", "perejil", "laurel", "comino", "sazon", "sazón",
    "mantequilla", "aji", "ají", "pimiento", "tomate", "azucar", "azúcar", "miel", "canela",
})


# ═══════════════════════════════════════════════════════════════════ knobs
def fidelity_gate_mode() -> str:
    """`MEALFIT_FIDELITY_GATE` = warn | block (default warn). Se lee en cada llamada."""
    try:
        from knobs import _env_str
        return _env_str("MEALFIT_FIDELITY_GATE", "warn", choices=set(FIDELITY_GATE_MODES))
    except Exception:
        raw = str(os.environ.get("MEALFIT_FIDELITY_GATE", "warn") or "warn").strip().lower()
        return raw if raw in FIDELITY_GATE_MODES else "warn"


def policy_mode_for_user(user_id: Optional[str] = None) -> str:
    """Modo efectivo para un usuario: `enforce` global, o `shadow` global + usuario en la lista
    canary `MEALFIT_PLAN_POLICY_ENFORCE_USERS` (uuids separados por coma) ⇒ `enforce`."""
    try:
        from plan_policy import policy_mode
        mode = policy_mode()
    except Exception:
        mode = "off"
    if mode != "shadow" or not user_id:
        return mode
    allow = os.environ.get("MEALFIT_PLAN_POLICY_ENFORCE_USERS", "") or ""
    if allow.strip():
        allowed = {u.strip().lower() for u in allow.split(",") if u.strip()}
        if str(user_id).strip().lower() in allowed:
            return "enforce"
    return mode


def policy_enforced(user_id: Optional[str] = None) -> bool:
    return policy_mode_for_user(user_id) == "enforce"


def shopping_projection_jobs_enabled() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_SHOPPING_PROJECTION_JOBS", True)
    except Exception:
        return str(os.environ.get("MEALFIT_SHOPPING_PROJECTION_JOBS", "true")).strip().lower() in ("1", "true", "yes", "on")


# ═══════════════════════════════════════════════════════════ motivo neutral
def is_renewal_reason(reason: Any) -> bool:
    """`renewal`, `renewal.vN` o el alias legado `variety` ⇒ renovación del plan."""
    r = str(reason or "").strip().lower()
    if not r:
        return False
    if r in LEGACY_RENEWAL_REASONS:
        return True
    return r == RENEWAL_REASON or r.startswith(RENEWAL_REASON + ".v")


def normalize_update_reason(reason: Any) -> Optional[str]:
    """El alias legado se canoniza al motivo versionado; los demás motivos no se tocan."""
    if reason is None:
        return None
    r = str(reason).strip()
    if not r:
        return None
    return RENEWAL_REASON_VERSIONED if is_renewal_reason(r) else r


def default_swap_reason(user_id: Optional[str] = None) -> str:
    """Motivo por defecto de un swap sin motivo: neutral bajo `enforce`, legado si no."""
    return RENEWAL_REASON_VERSIONED if policy_enforced(user_id) else "variety"


# ═══════════════════════════════════════════════════════════════ utilidades
def _norm(s: Any) -> str:
    try:
        from plan_policy import _norm as _pn
        return _pn(s)
    except Exception:
        return str(s or "").strip().lower()


def _matches(a: str, b: str) -> bool:
    try:
        from plan_policy import _matches as _pm
        return bool(_pm(a, b))
    except Exception:
        return _norm(a) == _norm(b)


def _ingredient_id(name: Any) -> str:
    try:
        from plan_policy import ingredient_id_for
        return ingredient_id_for(name)
    except Exception:
        return _norm(name).replace(" ", "_")


def _canon_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canon_json(obj).encode("utf-8")).hexdigest()


def meal_slot(meal: dict) -> Optional[str]:
    """Franja canónica de una comida del plan (`type`/`meal`/`slot`), o None."""
    if not isinstance(meal, dict):
        return None
    raw = meal.get("slot") or meal.get("type") or meal.get("meal") or meal.get("meal_type") or ""
    try:
        from plan_policy import _SLOT_ALIASES
        n = _norm(raw)
        for k, v in _SLOT_ALIASES.items():
            if n == k or n.startswith(k):
                return v
    except Exception:
        pass
    return None


def _meal_texts(meal: dict) -> list[str]:
    """Nombre + ingredientes de una comida, como textos (los nombres son el SSOT)."""
    out = []
    if not isinstance(meal, dict):
        return out
    if meal.get("name"):
        out.append(str(meal["name"]))
    for i in (meal.get("ingredients") or []):
        n = (i.get("name") or i.get("item")) if isinstance(i, dict) else i
        if n:
            out.append(str(n))
    return out


def anchor_in_text(anchor: str, text: str) -> bool:
    """¿El ancla está DENTRO del texto de un ingrediente («¾ cucharada de mantequilla de maní»,
    «½ kiwi en cubos»)? Contención token a token sobre la clave canónica de la Nevera (misma
    normalización de cantidad/acentos/plural que `pantry_names_match`), no igualdad: la fila de
    la Nevera exige el mismo número de tokens, pero aquí la pregunta es «¿aparece el ancla?».
    [P1-PANTRY-KEY-VULGAR-FRACTIONS · 2026-09-03]"""
    if _matches(anchor, text):
        return True
    try:
        from constants import canonical_pantry_key, _pantry_token_variants
        ka, kt = canonical_pantry_key(anchor), canonical_pantry_key(text)
        if not ka or not kt:
            return False
        stop = {"de", "del", "la", "el", "con", "en", "y", "al"}
        need = [t for t in ka.split() if t not in stop]
        have = set()
        for t in kt.split():
            have |= _pantry_token_variants(t)
        return bool(need) and all(_pantry_token_variants(t) & have for t in need)
    except Exception:
        return False


def _meal_has(meal: dict, name: str, iid: Optional[str] = None) -> bool:
    for t in _meal_texts(meal):
        if anchor_in_text(name, t) or (iid and _ingredient_id(t) == iid):
            return True
    return False


def _meals_per_day(form: Optional[dict], default: int = 4) -> int:
    form = form or {}
    for k in ("num_meals", "mealsPerDay", "meals_per_day", "mealFrequency"):
        v = form.get(k)
        try:
            n = int(float(v))
            if 1 <= n <= 8:
                return n
        except (TypeError, ValueError):
            continue
    return default


def slots_for_day(meals_per_day: int) -> list[str]:
    """Franjas de un día según el nº de comidas (3 → desayuno/almuerzo/cena; 4 → + merienda…)."""
    n = max(1, min(8, int(meals_per_day or 3)))
    if n <= 3:
        return list(("breakfast", "lunch", "dinner")[:n])
    if n == 4:
        return ["breakfast", "lunch", "snack", "dinner"]
    base = ["breakfast", "snack", "lunch", "snack", "dinner"]
    while len(base) < n:
        base.append("snack")
    return base[:n]


# ═══════════════════════════════════════════════════════════════════ chunks
def chunk_boundaries(total_days: int, base: Optional[int] = None) -> list[dict]:
    """Fronteras de los chunks, la MISMA aritmética que la cola (`split_with_absorb`, H2)."""
    from constants import PLAN_CHUNK_SIZE, split_with_absorb
    base = int(base or PLAN_CHUNK_SIZE)
    total = max(1, int(total_days or 1))
    sizes = split_with_absorb(total, base) if total > base else [total]
    out, offset = [], 0
    for i, n in enumerate(sizes):
        n = int(n)
        if n <= 0:
            continue
        out.append({"chunk_index": i, "week_number": i + 1, "days_offset": offset, "days_count": n})
        offset += n
    return out


# ═══════════════════════════════════════════════════════════════ familias
def protein_families_for(effective: dict) -> list[str]:
    """Familias de proteína permitidas por la política (dieta, alergias, exclusiones)."""
    eff = effective or {}
    diet = str(((eff.get("diet") or {}).get("type")) or "balanced").lower()
    allergies = list((eff.get("diet") or {}).get("allergies") or [])
    exclusions = list((eff.get("diet") or {}).get("exclusions") or [])
    base = _FAMILIES_BY_DIET.get(diet) or _FAMILIES_BY_DIET["_default"]
    try:
        from plan_policy import _anchor_hits_allergy, _anchor_hits_diet
    except Exception:  # pragma: no cover - plan_policy siempre está
        return list(base)
    out = []
    for fam in base:
        if _anchor_hits_allergy(fam, allergies):
            continue
        if _anchor_hits_diet(fam, diet):
            continue
        if any(_matches(fam, x) for x in exclusions):
            continue
        out.append(fam)
    return out


def repetition_limits_for(mode: str, window_days: int = 7) -> dict:
    """Límites de repetición para una ventana de N días, escalados desde la banda por 7 días."""
    m = str(mode or "balanced").lower()
    lim = dict(_RECURRENCE_LIMITS.get(m) or _RECURRENCE_LIMITS["balanced"])
    n = max(1, int(window_days or 7))
    lim["mode"] = m
    lim["window_days"] = n
    lim["max_exact_repeat"] = max(1, math.ceil(lim["max_exact_repeat_per_7d"] * n / 7.0))
    lim["max_ingredient_days"] = max(1, math.ceil(lim["max_ingredient_days_per_7d"] * n / 7.0))
    return lim


# ═══════════════════════════════════════════════════════════════ blueprint
def _schedule_days(total_days: int, min7: int, max7: int) -> list[int]:
    """Días (0-based) en que se programa un ancla: el mínimo pedido escalado al horizonte,
    sin superar el máximo; repartidos de forma pareja y determinista."""
    t = max(1, int(total_days))
    lo = max(0, int(min7 or 0))
    hi = max(lo, int(max7 if max7 is not None else 7))
    target = math.ceil(lo * t / 7.0)
    cap = math.floor(hi * t / 7.0)
    if target > cap:
        target = max(min(round(lo * t / 7.0), t), 0)
    target = max(0, min(t, target))
    if target == 0:
        return []
    if target >= t:
        return list(range(t))
    days, used = [], set()
    for k in range(target):
        d = int((k + 0.5) * t / target)
        d = max(0, min(t - 1, d))
        while d in used and d < t - 1:
            d += 1
        while d in used and d > 0:
            d -= 1
        used.add(d)
        days.append(d)
    return sorted(days)


def build_blueprint(effective: dict, *, total_days: int, base: Optional[int] = None,
                    meals_per_day: Optional[int] = None) -> dict:
    """Blueprint determinista del horizonte completo (§6.5). Misma política ⇒ mismo blueprint."""
    eff = effective or {}
    total = max(1, int(total_days or 1))
    mpd = int(meals_per_day or 4)
    slots = slots_for_day(mpd)
    rec = eff.get("recurrence") or {}
    gmode = str(rec.get("global_mode") or "balanced").lower()
    if gmode not in _RECURRENCE_LIMITS:
        gmode = "balanced"
    slot_modes = {s: str((rec.get("slot_modes") or {}).get(s) or gmode).lower() for s in ("breakfast", "lunch", "dinner", "snack")}
    limits7 = repetition_limits_for(gmode, 7)
    families = protein_families_for(eff)
    pool = families[: max(1, min(len(families), int(limits7["protein_pool_size"])))] if families else []
    chunks = chunk_boundaries(total, base)
    chunk_of = {}
    for c in chunks:
        for d in range(c["days_offset"], c["days_offset"] + c["days_count"]):
            chunk_of[d] = c["chunk_index"]

    anchors_out, per_day_anchors = [], {d: [] for d in range(total)}
    for a in (eff.get("food_anchors") or []):
        if not isinstance(a, dict):
            continue
        name = str(a.get("name") or a.get("ingredient_id") or "").strip()
        if not name:
            continue
        iid = a.get("ingredient_id") or _ingredient_id(name)
        slot = None
        for s in (a.get("slots") or []):
            s = str(s).lower()
            if s in _SLOT_ES:
                slot = s
                break
        min7 = int(a.get("min_per_7d") or 0)
        max7 = int(a.get("max_per_7d") if a.get("max_per_7d") is not None else 7)
        days = _schedule_days(total, min7, max7)
        anchors_out.append({
            "ingredient_id": iid, "name": name, "slot": slot, "min_per_7d": min7, "max_per_7d": max7,
            "preparation_mode": a.get("preparation_mode") or "vary_preparation",
            "scheduled_days": days,
        })
        for d in days:
            per_day_anchors[d].append({"ingredient_id": iid, "name": name, "slot": slot})

    culture = (eff.get("culture_weights") or [{}])[0] if isinstance(eff.get("culture_weights"), list) else {}
    profile_id = str((culture or {}).get("profile_id") or "dominican_criolla")
    days_out = []
    for d in range(total):
        days_out.append({
            "day_index": d,
            "chunk_index": chunk_of.get(d, 0),
            "slots": list(slots),
            "anchors": per_day_anchors[d],
            "protein": (pool[d % len(pool)] if pool else None),
            "culture": {s: profile_id for s in slots},
        })

    shopping = eff.get("shopping") or {}
    cycle = int(shopping.get("main_cycle_days") or total)
    fresh_windows = shopping_projection_windows(eff, total)
    freezer = str(shopping.get("freezer_mode") or "limited").lower()
    bp = {
        "schema_version": BLUEPRINT_SCHEMA_VERSION,
        "allocator_version": ALLOCATOR_VERSION,
        "policy_hash": eff.get("policy_hash"),
        "total_days": total,
        "meals_per_day": mpd,
        "recurrence": {"global_mode": gmode, "slot_modes": slot_modes},
        "repetition_limits": {k: limits7[k] for k in ("max_exact_repeat_per_7d", "max_ingredient_days_per_7d", "same_day_protein_repeat_ok")},
        "protein_families": families,
        "protein_pool": pool,
        "anchors": anchors_out,
        "days": days_out,
        "chunks": chunks,
        "fresh_windows": fresh_windows,
        "freezer": {"mode": freezer, "freeze_horizon_days": _freeze_horizon_days(freezer, total)},
        "main_cycle_days": cycle,
        "culture_profile": profile_id,
    }
    bp["blueprint_hash"] = blueprint_hash(bp)
    bp["built_at"] = datetime.now(timezone.utc).isoformat()
    return bp


_BP_VOLATILE = {"blueprint_hash", "built_at"}


def blueprint_hash(bp: dict) -> str:
    return _sha({k: v for k, v in (bp or {}).items() if k not in _BP_VOLATILE})


def slice_for_chunk(bp: dict, days_offset: int, days_count: int) -> dict:
    """Rebanada INMUTABLE del blueprint para un chunk (H2: `days_offset`/`days_count`)."""
    bp = bp or {}
    off = max(0, int(days_offset or 0))
    n = max(1, int(days_count or 1))
    days = [dict(d) for d in (bp.get("days") or []) if off <= int(d.get("day_index", -1)) < off + n]
    limits = repetition_limits_for((bp.get("recurrence") or {}).get("global_mode"), len(days) or n)
    anchors_in = []
    for a in (bp.get("anchors") or []):
        sched = [d for d in (a.get("scheduled_days") or []) if off <= d < off + n]
        anchors_in.append({**{k: a[k] for k in a if k != "scheduled_days"}, "scheduled_days": sched})
    windows = [w for w in (bp.get("fresh_windows") or []) if int(w.get("end_day", 0)) > off and int(w.get("start_day", 0)) < off + n]
    sl = {
        "schema_version": BLUEPRINT_SCHEMA_VERSION,
        "allocator_version": bp.get("allocator_version") or ALLOCATOR_VERSION,
        "blueprint_hash": bp.get("blueprint_hash"),
        "policy_hash": bp.get("policy_hash"),
        "total_days": bp.get("total_days"),
        "days_offset": off,
        "days_count": n,
        "recurrence": bp.get("recurrence") or {},
        "repetition_limits": {"max_exact_repeat": limits["max_exact_repeat"],
                              "max_ingredient_days": limits["max_ingredient_days"],
                              "same_day_protein_repeat_ok": limits["same_day_protein_repeat_ok"]},
        "anchors": anchors_in,
        "days": days,
        "fresh_windows": windows,
        "freezer": bp.get("freezer") or {},
    }
    sl["slice_hash"] = slice_hash(sl)
    return sl


def slice_hash(sl: dict) -> str:
    return _sha({k: v for k, v in (sl or {}).items() if k != "slice_hash"})


def chunk_input_hash(fingerprint: str, sl: Optional[dict]) -> str:
    """`input_hash` del chunk: huella del formulario + hash de su rebanada (si la hay)."""
    fp = str(fingerprint or "")
    if not isinstance(sl, dict) or not sl:
        return fp
    return hashlib.sha256(f"{fp}:{sl.get('slice_hash') or slice_hash(sl)}".encode("utf-8")).hexdigest()


# ════════════════════════════════════════════════════ persistencia / lookup
def persist_run_blueprint(run_id: Optional[str], bp: Optional[dict]) -> bool:
    """Guarda el blueprint en la fila del run (§6.5: «persistido con el run»). Best-effort."""
    if not run_id or not isinstance(bp, dict) or not bp:
        return False
    try:
        from db import execute_sql_write
        from psycopg.types.json import Jsonb
        execute_sql_write(
            "UPDATE plan_generation_runs SET blueprint = %s, blueprint_hash = %s, allocator_version = %s WHERE id = %s",
            (Jsonb(bp), bp.get("blueprint_hash"), bp.get("allocator_version") or ALLOCATOR_VERSION, run_id),
        )
        return True
    except Exception as e:
        logger.warning(f"[P1-ARQ25-F3-HORIZON] blueprint no persistido en el run {str(run_id)[:8]}: {e}")
        return False


def _run_blueprint_for_plan(plan_id: Optional[str]) -> Optional[dict]:
    if not plan_id:
        return None
    try:
        from db import execute_sql_query
        row = execute_sql_query(
            "SELECT blueprint FROM plan_generation_runs WHERE plan_id = %s AND blueprint IS NOT NULL "
            "AND blueprint <> '{}'::jsonb ORDER BY created_at DESC LIMIT 1",
            (plan_id,), fetch_one=True,
        )
        bp = (row or {}).get("blueprint")
        if isinstance(bp, str):
            bp = json.loads(bp)
        return bp if isinstance(bp, dict) and bp.get("days") else None
    except Exception as e:
        logger.debug(f"[P1-ARQ25-F3-HORIZON] blueprint del run no disponible: {e}")
        return None


def compiled_policy_for_form(form_data: dict, *, country: Optional[str] = None) -> Optional[dict]:
    """Compilación de la Fase 2 (fail-open) cuando el knob no está en `off`."""
    try:
        from plan_policy import compile_from_form, policy_active
        if not policy_active():
            return None
        c = compile_from_form(form_data or {}, country=country)
        return c if isinstance(c, dict) and c.get("effective") else None
    except Exception as e:
        logger.debug(f"[P1-ARQ25-F3-HORIZON] política no compilada: {e}")
        return None


def blueprint_for_plan(plan_id: Optional[str], form_data: dict, total_days: int, *,
                       country: Optional[str] = None) -> Optional[dict]:
    """El blueprint del run del plan; si el plan no nació en la cola, se reconstruye desde el
    formulario (misma política ⇒ mismo blueprint ⇒ mismo hash)."""
    bp = _run_blueprint_for_plan(plan_id)
    if bp:
        return bp
    compiled = compiled_policy_for_form(form_data, country=country)
    if not compiled:
        return None
    try:
        return build_blueprint(compiled["effective"], total_days=int(total_days or 1), meals_per_day=_meals_per_day(form_data))
    except Exception as e:
        logger.warning(f"[P1-ARQ25-F3-HORIZON] build_blueprint falló (fail-open): {e}")
        return None


def effective_policy_for_plan(plan_data: Optional[dict], form_data: Optional[dict] = None) -> Optional[dict]:
    """Política efectiva de un plan ya persistido (`_plan_policy.effective`, Fase 2) o, si no la
    lleva, compilada del formulario. None cuando el knob está en `off`."""
    try:
        from plan_policy import policy_active
        if not policy_active():
            return None
    except Exception:
        return None
    pp = (plan_data or {}).get("_plan_policy") if isinstance(plan_data, dict) else None
    if isinstance(pp, dict) and isinstance(pp.get("effective"), dict) and pp["effective"]:
        return pp["effective"]
    if not isinstance(form_data, dict) or not form_data:
        return None
    compiled = compiled_policy_for_form(form_data)
    return compiled["effective"] if compiled else None


def inject_policy_into_pipeline_data(pipeline_data: dict, *, form_data: dict, total_days: int,
                                     days_offset: int = 0, days_count: Optional[int] = None,
                                     user_id: Optional[str] = None, blueprint: Optional[dict] = None,
                                     compiled: Optional[dict] = None) -> Optional[dict]:
    """Deja en `pipeline_data` la rebanada del chunk + la política efectiva + el flag enforce.
    Devuelve el blueprint usado (para persistirlo con el run) o None si la política está `off`."""
    if not isinstance(pipeline_data, dict):
        return None
    compiled = compiled or compiled_policy_for_form(form_data)
    if not compiled:
        return None
    eff = compiled.get("effective") or {}
    try:
        bp = blueprint or build_blueprint(eff, total_days=int(total_days or 1), meals_per_day=_meals_per_day(form_data))
        n = int(days_count or pipeline_data.get("_days_to_generate") or bp.get("total_days") or 1)
        pipeline_data[BLUEPRINT_SLICE_KEY] = slice_for_chunk(bp, int(days_offset or 0), n)
        pipeline_data[POLICY_EFFECTIVE_KEY] = eff
        pipeline_data[POLICY_ENFORCED_KEY] = policy_enforced(user_id or pipeline_data.get("user_id"))
        return bp
    except Exception as e:
        logger.warning(f"[P1-ARQ25-F3-HORIZON] inyección de la rebanada omitida (fail-open): {e}")
        return None


def attach_policy_to_swap_form(form: dict, user_id: Optional[str], *, plan_data: Optional[dict] = None,
                               plan_id: Optional[str] = None, day_index: Optional[int] = None) -> Optional[dict]:
    """Las superficies post-generación (swap, regen de día) leen la política del plan vivo."""
    if not isinstance(form, dict):
        return None
    eff = None
    try:
        from plan_policy import policy_active
        if not policy_active():
            return None
        if plan_data is None and user_id and user_id != "guest":
            from db import execute_sql_query
            pid = plan_id or form.get("plan_id")
            if pid:
                row = execute_sql_query(
                    "SELECT plan_data->'_plan_policy' AS pp FROM meal_plans WHERE id = %s AND user_id = %s",
                    (pid, user_id), fetch_one=True)
            else:
                row = execute_sql_query(
                    "SELECT plan_data->'_plan_policy' AS pp FROM meal_plans WHERE user_id = %s "
                    "ORDER BY created_at DESC LIMIT 1", (user_id,), fetch_one=True)
            pp = (row or {}).get("pp")
            if isinstance(pp, str):
                pp = json.loads(pp)
            plan_data = {"_plan_policy": pp} if isinstance(pp, dict) else {}
        eff = effective_policy_for_plan(plan_data, None)   # el formulario del swap NO es el del wizard
    except Exception as e:
        logger.debug(f"[P1-ARQ25-F3-HORIZON] política del swap no disponible: {e}")
        eff = None
    # siempre server-side: un cliente no decide la política que el motor obedece
    form.pop(BLUEPRINT_SLICE_KEY, None)
    if eff:
        form[POLICY_EFFECTIVE_KEY] = eff
        form[POLICY_ENFORCED_KEY] = policy_enforced(user_id)
        if day_index is not None:
            form[POLICY_DAY_INDEX_KEY] = int(day_index)
    else:
        form.pop(POLICY_EFFECTIVE_KEY, None)
        form[POLICY_ENFORCED_KEY] = False
    return eff


# ═══════════════════════════════════════════════════════════════════ prompt
_MODE_ES = {
    "routine": "RUTINA: repetir platos y bases es CORRECTO y deseado",
    "balanced": "EQUILIBRADA: alguna repetición está bien; no fuerces novedad en cada comida",
    "explore": "EXPLORAR: máxima novedad; evita repetir platos",
}


def policy_prompt_block(effective: Optional[dict], sl: Optional[dict] = None, *, surface: str,
                        day_index: Optional[int] = None, slot: Optional[str] = None,
                        enforced: bool = True) -> str:
    """Bloque 📐 para el LLM: banda de recurrencia, anclas (del día si se conoce) y límites de
    repetición. Vacío cuando no hay política o no está en `enforce` (prompts byte-idénticos)."""
    if not enforced or not isinstance(effective, dict) or not effective:
        return ""
    try:
        rec = effective.get("recurrence") or {}
        gmode = str(rec.get("global_mode") or "balanced").lower()
        lines = [f"\n📐 [POLÍTICA DEL PLAN · fidelidad, no «variedad» por defecto · {surface}]",
                 f"- Recurrencia solicitada — {_MODE_ES.get(gmode, _MODE_ES['balanced'])}."]
        anchors = [a for a in (effective.get("food_anchors") or []) if isinstance(a, dict) and (a.get("name") or a.get("ingredient_id"))]
        if anchors:
            lines.append("- Alimentos ancla del usuario (deben aparecer con la frecuencia pedida): " + "; ".join(
                f"{a.get('name') or a.get('ingredient_id')} ({a.get('min_per_7d', 0)}–{a.get('max_per_7d', 7)} de cada 7 días"
                + (f", {_SLOT_ES.get(str(a.get('slots')[0]).lower(), a.get('slots')[0])}" if a.get("slots") else "") + ")"
                for a in anchors) + ".")
        if isinstance(sl, dict) and sl.get("days"):
            per_day = []
            for d in sl["days"]:
                if day_index is not None and int(d.get("day_index", -1)) != int(day_index):
                    continue
                rel = int(d.get("day_index", 0)) - int(sl.get("days_offset") or 0) + 1
                parts = []
                if d.get("anchors"):
                    parts.append("anclas: " + ", ".join(
                        f"{a.get('name')}" + (f" ({_SLOT_ES.get(a.get('slot'), a.get('slot'))})" if a.get("slot") else "")
                        for a in d["anchors"]))
                if d.get("protein"):
                    parts.append(f"proteína principal sugerida: {d['protein']}")
                if parts:
                    per_day.append(f"Día {rel}: " + "; ".join(parts))
            if per_day:
                lines.append("- Reparto de este bloque → " + " | ".join(per_day) + ".")
            lim = sl.get("repetition_limits") or {}
            if lim.get("max_exact_repeat"):
                lines.append(f"- En este bloque un MISMO plato puede repetirse hasta {int(lim['max_exact_repeat'])} vez/veces; "
                             f"un ingrediente principal hasta {int(lim.get('max_ingredient_days') or 7)} día(s).")
        if slot:
            s = meal_slot({"type": slot}) or str(slot).lower()
            slot_anchors = [a for a in anchors if any(str(x).lower() == s for x in (a.get("slots") or []))]
            if slot_anchors:
                lines.append(f"- Esta comida es {_SLOT_ES.get(s, s)}: ancla de la franja → " + ", ".join(
                    str(a.get("name")) for a in slot_anchors) + ".")
        lines.append("- No «diversifiques» un ancla ni cambies la base de un plato por variedad: la política manda sobre la variedad.")
        return "\n".join(lines) + "\n"
    except Exception as e:
        logger.debug(f"[P1-ARQ25-F3-HORIZON] policy_prompt_block falló: {e}")
        return ""


# [P1-ARQ25-F3-HORIZON · canary 2026-09-03] Una familia («Pescado») debe casar con el NOMBRE real del
# pool («Sardinas en lata», «Filete de pescado blanco»): `_matches` compara raíces de palabra y no
# sabe que una sardina es pescado. Medido en el primer plan del canary (nevera estricta): las 3
# familias programadas cayeron al fallback porque el pool era sardinas/mozzarella/hígado.
_FAMILY_TOKENS = {
    "pescado": ("pescado", "atun", "salmon", "bacalao", "sardina", "camaron", "camarones", "mariscos", "pulpo",
                "calamar", "cangrejo", "langosta", "tilapia", "merluza", "dorado", "chillo", "mero", "filete de pescado"),
    "pollo": ("pollo", "pechuga", "muslo", "gallina"),
    "res": ("res", "carne", "bistec", "higado", "molida", "falda", "chuleta de res"),
    "cerdo": ("cerdo", "puerco", "chuleta", "tocino", "jamon", "costilla"),
    "huevo": ("huevo", "huevos", "clara", "claras"),
    "queso": ("queso", "mozzarella", "cheddar", "ricotta", "requeson"),
    "atun": ("atun",), "pavo": ("pavo",), "yogur": ("yogur", "yogurt"),
    "lentejas": ("lenteja", "lentejas"), "garbanzos": ("garbanzo", "garbanzos"),
    "habichuelas": ("habichuela", "habichuelas", "frijol", "frijoles"), "tofu": ("tofu",),
    "camarones": ("camaron", "camarones"), "guandules": ("guandul", "guandules"),
}


# [P2-F3-FAMILY-INJECT-EMPTY-PANTRY · 2026-09-03] Nombre CANÓNICO del catálogo que representa a cada
# familia (no existen filas «Pollo»/«Pescado»/«Res» a secas). Se usa SOLO cuando el pool del seeder no
# trae ningún miembro de la familia programada Y la Nevera está por debajo del umbral del guard
# (`PANTRY_GUARD_MIN_ITEMS`, cuando el guard de despensa no aplica): con la Nevera vacía el sorteo
# ponderado del seeder entregaba «Habas / Guisantes secos / Queso de hoja» como proteína del día a un
# perfil de ganancia muscular (135 g), y el revisor rechazó 2 intentos seguidos por déficit de
# proteína (plan 8f364c87, 11:13 y 11:17 UTC). Con Nevera real el pool sale de ella y no se inyecta.
_FAMILY_REPRESENTATIVE = {
    "pollo": "Pechuga de pollo", "pescado": "Filete de pescado blanco", "huevo": "Huevo", "res": "Carne de res",
    "cerdo": "Cerdo", "atun": "Atún en agua", "pavo": "Pechuga de pavo", "lentejas": "Lentejas",
    "habichuelas": "Habichuelas", "garbanzos": "Garbanzos", "tofu": "Tofu", "camarones": "Camarones",
    "queso": "Queso blanco", "yogur": "Yogur natural", "guandules": "Guandules",
}


def family_representative(family: str) -> Optional[str]:
    return _FAMILY_REPRESENTATIVE.get(_norm(family)) or _FAMILY_REPRESENTATIVE.get(_stem_word(_norm(family)))


def family_matches(family: str, candidate: str) -> bool:
    """¿El nombre del pool pertenece a la familia programada? Por clase de alimento, no por raíz."""
    if _matches(family, candidate):
        return True
    fam = _norm(family)
    cand = _norm(candidate)
    toks = _FAMILY_TOKENS.get(fam) or _FAMILY_TOKENS.get(_stem_word(fam)) or ()
    words = {_stem_word(w) for w in cand.split()}
    return any((" " in t and t in cand) or _stem_word(t) in words for t in toks)


def _stem_word(t: str) -> str:
    try:
        from plan_policy import _stem
        return _stem(t)
    except Exception:
        return t


def apply_slice_to_seeder_pools(sl: dict, chosen: list, pool: list, *, days: int, inject_missing: bool = False) -> list:
    """Proteína por día según la rebanada: para cada día busca en el pool del seeder un nombre
    que case con la familia programada; si no hay, conserva la elección del seeder — o, con
    `inject_missing` (Nevera por debajo del umbral del guard), inyecta el representante canónico
    de la familia. En modo rutina la misma proteína puede repetirse entre días; en los demás se
    evita si es posible."""
    try:
        days = max(1, int(days or 1))
        chosen = list(chosen or [])
        pool = list(pool or [])
        if not isinstance(sl, dict) or not sl.get("days") or not pool:
            return chosen
        routine = str((sl.get("recurrence") or {}).get("global_mode") or "").lower() == "routine"
        out, used = [], set()
        for i in range(days):
            d = sl["days"][i] if i < len(sl["days"]) else None
            fam = (d or {}).get("protein")
            pick = None
            if fam:
                for cand in pool:
                    if family_matches(fam, cand) and (routine or cand.lower() not in used):
                        pick = cand
                        break
            if pick is None and fam and inject_missing:
                rep_name = family_representative(fam)
                if rep_name and (routine or rep_name.lower() not in used):
                    pick = rep_name
            if pick is None:
                for cand in chosen + pool:
                    if cand.lower() not in used:
                        pick = cand
                        break
            if pick is None:
                pick = chosen[i % len(chosen)] if chosen else pool[i % len(pool)]
            used.add(pick.lower())
            out.append(pick)
        return out
    except Exception as e:
        logger.debug(f"[P1-ARQ25-F3-HORIZON] apply_slice_to_seeder_pools falló: {e}")
        return list(chosen or [])


# ═══════════════════════════════════════════════════════════════ fidelidad
def fidelity_issues(days: list, sl: Optional[dict], effective: Optional[dict], *,
                    meals_per_day: Optional[int] = None) -> list[dict]:
    """Validadores de fidelidad sobre los días producidos (ventana = este chunk):
    `anchor_missing_day`, `anchor_slot_mismatch`, `recurrence_above_band`, `recurrence_below_band`,
    `exact_repeat_exceeded`, `ingredient_days_exceeded`. Puro, sin red, nunca lanza."""
    out: list[dict] = []
    try:
        days = [d for d in (days or []) if isinstance(d, dict)]
        n = len(days)
        if n == 0:
            return out
        eff = effective or {}
        sl = sl if isinstance(sl, dict) else {}
        if not sl and not eff:
            return out
        rec_mode = str(((sl.get("recurrence") or eff.get("recurrence") or {}).get("global_mode")) or "balanced").lower()
        limits = sl.get("repetition_limits") or {
            k: v for k, v in repetition_limits_for(rec_mode, n).items()
            if k in ("max_exact_repeat", "max_ingredient_days", "same_day_protein_repeat_ok")}
        off = int(sl.get("days_offset") or 0)
        anchors = sl.get("anchors") or [
            {"ingredient_id": a.get("ingredient_id"), "name": a.get("name") or a.get("ingredient_id"),
             "slot": (str(a["slots"][0]).lower() if a.get("slots") else None),
             "min_per_7d": a.get("min_per_7d", 0), "max_per_7d": a.get("max_per_7d", 7),
             "scheduled_days": []}
            for a in (eff.get("food_anchors") or []) if isinstance(a, dict)]
        sched_by_day: dict[int, list] = {}
        for d in (sl.get("days") or []):
            sched_by_day[int(d.get("day_index", -1))] = list(d.get("anchors") or [])

        # 1) anclas por día (rebanada) — día ausente / franja equivocada
        # [P1-PANTRY-KEY-VULGAR-FRACTIONS · 2026-09-03] Solo las anclas CON franja se exigen en su
        # día (rutina explícita). Las anclas sin franja prometen una CUOTA en la ventana: se
        # validan en (2) como `anchor_under_scheduled` — «lo puso el día 3 en vez del 2» no es
        # infidelidad, y en `block` habría quemado un reintento sobre un plan correcto.
        missing_for: dict[str, int] = {}
        scheduled_in_window: dict[str, int] = {}
        for i, day in enumerate(days):
            abs_idx = off + i
            for a in sched_by_day.get(abs_idx, []):
                name, iid, slot = str(a.get("name")), a.get("ingredient_id"), a.get("slot")
                scheduled_in_window[iid or name] = scheduled_in_window.get(iid or name, 0) + 1
                if not slot:
                    continue
                meals = [m for m in (day.get("meals") or []) if isinstance(m, dict)]
                in_any = [m for m in meals if _meal_has(m, name, iid)]
                if not in_any:
                    missing_for[iid or name] = missing_for.get(iid or name, 0) + 1
                    out.append({
                        "code": "anchor_missing_day", "severity": "high", "day": i + 1, "anchor": name,
                        "message": (f"ANCLA AUSENTE (fidelidad a la política del usuario): el día {i + 1} debía incluir "
                                    f"{name}" + (f" en {_SLOT_ES.get(slot, slot)}" if slot else "") +
                                    " y no aparece. Inclúyelo ese día" + (f" en {_SLOT_ES.get(slot, slot)}" if slot else "") +
                                    " sin cambiar los demás platos."),
                    })
                elif slot and not any(meal_slot(m) == slot for m in in_any):
                    out.append({
                        "code": "anchor_slot_mismatch", "severity": "medium", "day": i + 1, "anchor": name,
                        "message": (f"ANCLA EN FRANJA EQUIVOCADA: el día {i + 1} {name} debía ir en "
                                    f"{_SLOT_ES.get(slot, slot)} y aparece en otra comida. Muévelo a {_SLOT_ES.get(slot, slot)}."),
                    })

        # 2) banda de recurrencia por ancla (escalada a la ventana)
        for a in anchors:
            name, iid = str(a.get("name")), a.get("ingredient_id")
            present = sum(1 for day in days if any(_meal_has(m, name, iid) for m in (day.get("meals") or []) if isinstance(m, dict)))
            hi = math.ceil(int(a.get("max_per_7d") if a.get("max_per_7d") is not None else 7) * n / 7.0)
            lo = math.floor(int(a.get("min_per_7d") or 0) * n / 7.0)
            sched_n = scheduled_in_window.get(iid or name, 0)
            if sched_n and present < sched_n and (iid or name) not in missing_for:
                out.append({
                    "code": "anchor_under_scheduled", "severity": "high", "anchor": name, "present": present, "scheduled": sched_n,
                    "message": (f"ANCLA POR DEBAJO DE LO PROGRAMADO (fidelidad a la política del usuario): {name} debía "
                                f"aparecer en {sched_n} día(s) de este bloque y aparece en {present}. Añádelo en "
                                f"{sched_n - present} día(s) más sin cambiar el resto."),
                })
                missing_for[iid or name] = sched_n - present
            if present > hi:
                out.append({
                    "code": "recurrence_above_band", "severity": "medium", "anchor": name, "present": present, "max": hi,
                    "message": (f"ANCLA POR ENCIMA DE SU BANDA: {name} aparece en {present} de {n} días; el usuario pidió "
                                f"como máximo {a.get('max_per_7d', 7)} de cada 7 (≈{hi} en este bloque). Sustitúyelo en "
                                f"{present - hi} día(s)."),
                })
            elif present < lo and (iid or name) not in missing_for:
                out.append({
                    "code": "recurrence_below_band", "severity": "medium", "anchor": name, "present": present, "min": lo,
                    "message": (f"ANCLA POR DEBAJO DE SU BANDA: {name} aparece en {present} de {n} días; el usuario pidió "
                                f"al menos {a.get('min_per_7d', 0)} de cada 7 (≈{lo} en este bloque). Añádelo en "
                                f"{lo - present} día(s) más."),
                })

        # 3) repetición exacta del mismo plato (nombre normalizado) por encima del límite
        max_exact = int(limits.get("max_exact_repeat") or 1)
        counts: dict[str, int] = {}
        labels: dict[str, str] = {}
        for day in days:
            for m in (day.get("meals") or []):
                if isinstance(m, dict) and m.get("name"):
                    k = _norm(m["name"])
                    counts[k] = counts.get(k, 0) + 1
                    labels.setdefault(k, str(m["name"]))
        for k, c in sorted(counts.items()):
            if c > max_exact:
                out.append({
                    "code": "exact_repeat_exceeded", "severity": "medium", "dish": labels[k], "count": c, "max": max_exact,
                    "message": (f"PLATO REPETIDO MÁS DE LO QUE PIDE LA POLÍTICA: «{labels[k]}» aparece {c} veces en este "
                                f"bloque; con recurrencia {rec_mode} el máximo es {max_exact}. Cambia {c - max_exact} de "
                                f"esas apariciones por otra preparación o plato."),
                })

        # 4) ingrediente principal en demasiados días (no anclas, no sazón)
        max_ing_days = int(limits.get("max_ingredient_days") or n)
        if rec_mode == "explore" and max_ing_days < n:
            anchor_ids = {a.get("ingredient_id") for a in anchors} | {_ingredient_id(a.get("name")) for a in anchors}
            ing_days: dict[str, set] = {}
            ing_label: dict[str, str] = {}
            for i, day in enumerate(days):
                for m in (day.get("meals") or []):
                    if not isinstance(m, dict):
                        continue
                    for ing in (m.get("ingredients") or []):
                        nm = (ing.get("name") or ing.get("item")) if isinstance(ing, dict) else ing
                        if not nm:
                            continue
                        nn = _norm(nm)
                        if len(nn) < 4 or nn in _INGREDIENT_DAYS_EXEMPT:
                            continue
                        iid = _ingredient_id(nm)
                        if iid in anchor_ids:
                            continue
                        ing_days.setdefault(iid, set()).add(i)
                        ing_label.setdefault(iid, str(nm))
            for iid, ds in sorted(ing_days.items()):
                if len(ds) > max_ing_days:
                    out.append({
                        "code": "ingredient_days_exceeded", "severity": "low", "ingredient": ing_label[iid],
                        "days": len(ds), "max": max_ing_days,
                        "message": (f"INGREDIENTE EN DEMASIADOS DÍAS para la recurrencia {rec_mode}: {ing_label[iid]} aparece "
                                    f"en {len(ds)} de {n} días (máximo {max_ing_days}). Sustitúyelo en los días sobrantes."),
                    })
    except Exception as e:
        logger.debug(f"[P1-ARQ25-F3-HORIZON] fidelity_issues falló (fail-open): {e}")
        return out
    return out


def fidelity_report(days: list, sl: Optional[dict], effective: Optional[dict], *, surface: str,
                    meals_per_day: Optional[int] = None) -> dict:
    issues = fidelity_issues(days, sl, effective, meals_per_day=meals_per_day)
    n_checks = 0
    if isinstance(sl, dict):
        n_checks += sum(len(d.get("anchors") or []) for d in (sl.get("days") or []))
        n_checks += len(sl.get("anchors") or [])
    n_checks += 2  # repetición exacta + ingrediente
    score = round(max(0.0, 1.0 - (len(issues) / float(max(1, n_checks)))), 3)
    return {
        "schema_version": BLUEPRINT_SCHEMA_VERSION, "surface": str(surface or "")[:40],
        "slice_hash": (sl or {}).get("slice_hash") if isinstance(sl, dict) else None,
        "policy_hash": (effective or {}).get("policy_hash") if isinstance(effective, dict) else None,
        "days_checked": len([d for d in (days or []) if isinstance(d, dict)]),
        "issues": issues, "codes": sorted({i["code"] for i in issues}), "score": score,
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }


_VARIETY_REPEAT_FAMILIES = ("PLATO-BASE REPETIDO", "MISMO PLATO REPETIDO ENTRE DÍAS", "MISMA PROTEÍNA REPETIDA")


def filter_variety_issues_for_policy(issues: list, effective: Optional[dict], *, enforced: bool) -> list:
    """En `enforce`, los gates de REPETICIÓN de V1 que contradicen la banda pedida se retiran
    (los de coherencia —fruta+salado, fruta repetida— se conservan): rutina ⇒ todos los de
    repetición; equilibrada ⇒ solo los que hablan de un ancla; explorar ⇒ ninguno se retira."""
    issues = list(issues or [])
    if not enforced or not isinstance(effective, dict) or not effective:
        return issues
    mode = str(((effective.get("recurrence") or {}).get("global_mode")) or "balanced").lower()
    if mode == "explore":
        return issues
    anchor_tokens = set()
    for a in (effective.get("food_anchors") or []):
        for t in _norm((a or {}).get("name") or (a or {}).get("ingredient_id") or "").split():
            if len(t) >= 3:
                anchor_tokens.add(t[:-1] if t.endswith("s") else t)
    kept = []
    for it in issues:
        txt = str(it)
        is_repeat = any(f in txt for f in _VARIETY_REPEAT_FAMILIES)
        if not is_repeat:
            kept.append(it)
            continue
        if mode == "routine":
            continue
        low = _norm(txt)
        if any(tok in low for tok in anchor_tokens):
            continue
        kept.append(it)
    return kept


def exclude_anchors_from_fatigue(fatigued: Iterable, effective: Optional[dict]) -> list:
    """Aprendizaje: un ancla nunca se «fatiga» — el usuario pidió repetirla (§6.6)."""
    items = list(fatigued or [])
    if not isinstance(effective, dict) or not effective.get("food_anchors"):
        return items
    names = [str(a.get("name") or a.get("ingredient_id") or "") for a in effective["food_anchors"] if isinstance(a, dict)]
    return [f for f in items if not any(_matches(n, str(f)) for n in names if n)]


def rank_days_by_policy(days: list, effective: Optional[dict]) -> list:
    """Shift/shuffle degradado: ordena candidatos por cobertura de anclas (estable). Sin política
    devuelve la lista tal cual."""
    if not isinstance(effective, dict) or not effective.get("food_anchors") or not days:
        return list(days or [])
    names = [(str(a.get("name") or a.get("ingredient_id") or ""), a.get("ingredient_id"))
             for a in effective["food_anchors"] if isinstance(a, dict)]

    def _cov(day):
        meals = (day.get("meals") or []) if isinstance(day, dict) else []
        return sum(1 for n, iid in names if n and any(_meal_has(m, n, iid) for m in meals if isinstance(m, dict)))

    return sorted(list(days), key=lambda d: -_cov(d))


def emit_fidelity_metric(user_id: Optional[str], plan_id: Optional[str], report: dict, *, mode: str,
                         gate: Optional[str] = None, rejected: bool = False) -> None:
    """`pipeline_metrics` node=`plan_policy_fidelity` (best-effort): base del gate de la fase."""
    try:
        from db import execute_sql_write
        meta = {
            "plan_id": plan_id, "surface": report.get("surface"), "slice_hash": report.get("slice_hash"),
            "policy_hash": report.get("policy_hash"), "codes": report.get("codes"),
            "n_issues": len(report.get("issues") or []), "days_checked": report.get("days_checked"),
            "score": report.get("score"), "mode": mode, "gate": gate or fidelity_gate_mode(),
            "rejected": bool(rejected), "allocator_version": ALLOCATOR_VERSION,
        }
        execute_sql_write(
            "INSERT INTO pipeline_metrics (user_id, session_id, node, duration_ms, retries, "
            "tokens_estimated, confidence, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
            (user_id if user_id and user_id != "guest" else None, "__policy__", FIDELITY_METRIC_NODE, 0, 0, 0,
             report.get("score"), json.dumps(meta, default=str)),
        )
    except Exception as e:
        logger.debug(f"[P1-ARQ25-F3-HORIZON] métrica de fidelidad no persistida: {e}")


def review_fidelity_gate(plan: dict, form_data: dict, variety_issues: list, *, attempt: int,
                         max_attempts: int) -> tuple[list, list]:
    """Punto único del revisor: (a) mide la fidelidad del plan contra su rebanada y la sella en
    `plan['_fidelity_report']` + métrica; (b) en `enforce` filtra los gates de repetición de V1
    que contradicen la banda; (c) con `MEALFIT_FIDELITY_GATE=block` y no en el intento final,
    devuelve los mensajes de rechazo. Devuelve (variety_issues_filtrados, rechazos_fidelidad)."""
    variety_issues = list(variety_issues or [])
    try:
        form_data = form_data or {}
        eff = form_data.get(POLICY_EFFECTIVE_KEY)
        sl = form_data.get(BLUEPRINT_SLICE_KEY)
        if not isinstance(eff, dict) or not eff:
            return variety_issues, []
        enforced = bool(form_data.get(POLICY_ENFORCED_KEY))
        days = (plan or {}).get("days") if isinstance(plan, dict) else None
        report = fidelity_report(days or [], sl if isinstance(sl, dict) else None, eff, surface="review_plan_node")
        gate = fidelity_gate_mode()
        rejects: list = []
        if enforced and gate == "block" and int(attempt) < int(max_attempts):
            rejects = [i["message"] for i in report["issues"] if i.get("severity") in ("high", "medium")]
        mode = "enforce" if enforced else "shadow"
        report["enforced"] = enforced
        report["mode"] = mode  # [P1-ARQ25-F4-FORM] la pantalla «solicitaste / aplicamos» lee si el motor obedeció
        report["gate"] = gate
        report["rejected"] = bool(rejects)
        if isinstance(plan, dict):
            plan[FIDELITY_REPORT_KEY] = report
        emit_fidelity_metric(form_data.get("user_id"), form_data.get("_caller_target_plan_id"), report,
                             mode=mode, gate=gate, rejected=bool(rejects))
        filtered = filter_variety_issues_for_policy(variety_issues, eff, enforced=enforced)
        if report["issues"]:
            logger.warning(
                f"📐 [P1-ARQ25-F3-HORIZON] fidelidad {mode}/{gate}: {len(report['issues'])} issue(s) "
                f"{report['codes']} score={report['score']} rechazo={bool(rejects)} "
                f"variety_gates={len(variety_issues)}→{len(filtered)}")
        return filtered, rejects
    except Exception as e:
        logger.warning(f"[P1-ARQ25-F3-HORIZON] review_fidelity_gate falló (fail-open): {e}")
        return variety_issues, []


# ═══════════════════════════════════════════════════ shopping / proyección
def _freeze_horizon_days(freezer_mode: str, total_days: int) -> int:
    m = str(freezer_mode or "limited").lower()
    if m == "none":
        return 0
    if m == "full":
        return int(total_days)
    return min(7, int(total_days))


def shopping_projection_windows(effective: Optional[dict], total_days: int) -> list[dict]:
    """Ventanas de compra 7/15/30: la principal + los top-ups de frescos cada `fresh_topup_days`."""
    eff = effective or {}
    total = max(1, int(total_days or 1))
    shopping = eff.get("shopping") or {}
    cycle = int(shopping.get("main_cycle_days") or total)
    cycle = max(1, min(cycle, total))
    out = [{"kind": "main", "start_day": 0, "end_day": total, "days": total, "cycle_days": cycle, "fresh_only": False}]
    topup = shopping.get("fresh_topup_days")
    try:
        topup = int(topup) if topup not in (None, "", False) else 0
    except (TypeError, ValueError):
        topup = 0
    if topup > 0 and total > topup:
        start = topup
        while start < total:
            end = min(total, start + topup)
            out.append({"kind": "fresh_topup", "start_day": start, "end_day": end, "days": end - start,
                        "cycle_days": topup, "fresh_only": True})
            start = end
    return out


def stamp_demand_windows(plan_data: dict, effective: Optional[dict] = None) -> Optional[dict]:
    """Añade a `_ingredient_demand` (H6) las ventanas de frescos/congelación de la política.
    Idempotente; si no hay política, no toca nada."""
    if not isinstance(plan_data, dict):
        return None
    try:
        eff = effective or effective_policy_for_plan(plan_data)
        if not eff:
            return None
        total = int(plan_data.get("total_days_requested") or len(plan_data.get("days") or []) or 1)
        try:
            from shopping_calculator import INGREDIENT_DEMAND_KEY, shopping_source_days
            total = max(total, len(shopping_source_days(plan_data) or []))
            key = INGREDIENT_DEMAND_KEY
        except Exception:
            key = "_ingredient_demand"
        freezer = str(((eff.get("shopping") or {}).get("freezer_mode")) or "limited").lower()
        stamp = plan_data.get(key) if isinstance(plan_data.get(key), dict) else {}
        stamp = dict(stamp)
        stamp["windows"] = shopping_projection_windows(eff, total)
        stamp["freezer_mode"] = freezer
        stamp["freeze_horizon_days"] = _freeze_horizon_days(freezer, total)
        stamp["policy_hash"] = eff.get("policy_hash")
        stamp["windows_schema"] = BLUEPRINT_SCHEMA_VERSION
        plan_data[key] = stamp
        return stamp
    except Exception as e:
        logger.debug(f"[P1-ARQ25-F3-HORIZON] stamp_demand_windows falló: {e}")
        return None


def enqueue_shopping_projection_job(plan_id: Optional[str], user_id: Optional[str], *, revision: Optional[int],
                                    effective: Optional[dict], total_days: int) -> Optional[str]:
    """Outbox `plan_jobs` (`job_type='shopping_projection'`): la lista 7/15/30 como PROYECCIÓN.
    Nadie la consume hasta la Fase 5; dedup por plan+revisión+política. Solo bajo `enforce`."""
    if not plan_id or not user_id or user_id == "guest" or not effective:
        return None
    if not shopping_projection_jobs_enabled() or not policy_enforced(user_id):
        return None
    try:
        from db import execute_sql_write
        from psycopg.types.json import Jsonb
        windows = shopping_projection_windows(effective, total_days)
        payload = {"schema_version": BLUEPRINT_SCHEMA_VERSION, "policy_hash": effective.get("policy_hash"),
                   "total_days": int(total_days), "windows": windows,
                   "freezer_mode": str(((effective.get("shopping") or {}).get("freezer_mode")) or "limited")}
        dedup = f"shopping_projection:{plan_id}:{int(revision or 0)}:{_sha(payload)[:12]}"
        rows = execute_sql_write(
            "INSERT INTO plan_jobs (job_type, plan_id, user_id, plan_revision, dedup_key, payload) "
            "VALUES ('shopping_projection', %s, %s, %s, %s, %s) ON CONFLICT (dedup_key) DO NOTHING RETURNING id",
            (plan_id, user_id, int(revision) if isinstance(revision, int) else None, dedup, Jsonb(payload)),
            returning=True,
        ) or []
        return str(rows[0]["id"]) if rows else None
    except Exception as e:
        logger.debug(f"[P1-ARQ25-F3-HORIZON] proyección de compra no encolada: {e}")
        return None


__all__ = [
    "BLUEPRINT_SCHEMA_VERSION", "ALLOCATOR_VERSION", "FIDELITY_GATE_MODES", "FIDELITY_METRIC_NODE",
    "RENEWAL_REASON", "RENEWAL_REASON_VERSION", "RENEWAL_REASON_VERSIONED", "LEGACY_RENEWAL_REASONS",
    "BLUEPRINT_SLICE_KEY", "POLICY_EFFECTIVE_KEY", "POLICY_ENFORCED_KEY", "POLICY_DAY_INDEX_KEY", "FIDELITY_REPORT_KEY",
    "fidelity_gate_mode", "policy_mode_for_user", "policy_enforced", "shopping_projection_jobs_enabled",
    "is_renewal_reason", "normalize_update_reason", "default_swap_reason",
    "meal_slot", "slots_for_day", "chunk_boundaries", "protein_families_for", "repetition_limits_for",
    "build_blueprint", "blueprint_hash", "slice_for_chunk", "slice_hash", "chunk_input_hash",
    "persist_run_blueprint", "compiled_policy_for_form", "blueprint_for_plan", "effective_policy_for_plan",
    "inject_policy_into_pipeline_data", "attach_policy_to_swap_form",
    "policy_prompt_block", "apply_slice_to_seeder_pools", "family_matches", "family_representative",
    "anchor_in_text", "fidelity_issues", "fidelity_report", "filter_variety_issues_for_policy", "exclude_anchors_from_fatigue",
    "rank_days_by_policy", "emit_fidelity_metric", "review_fidelity_gate",
    "shopping_projection_windows", "stamp_demand_windows", "enqueue_shopping_projection_job",
]
