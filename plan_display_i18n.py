# backend/plan_display_i18n.py
"""Motor de enriquecimiento de la capa de display i18n del plan.

tooltip-anchor: P1-PLAN-DISPLAY-I18N

Decisión arquitectónica (spec `docs/superpowers/specs/2026-08-19-plan-display-i18n-design.md`):
el plan se GENERA y PERSISTE siempre en español canónico — los nombres de alimentos y platos
son identificadores del sistema (`pantry_names_match`, coherence guard, backstop de alergias
resuelven por esos strings exactos; P1-I18N-DASHBOARD). Este módulo NUNCA lee ese campo de
vuelta ni condiciona su propia conducta a él — solo ESCRIBE un campo paralelo de solo-lectura:

    meal["_display"][locale] = {"name", "description", "recipe", "ingredients"}

que el frontend consume con fallback campo a campo al original si falta (fase 2, fuera de
este módulo). es-DO jamás lleva `_display` — byte-idéntico a hoy.

Contrato:
    enrich_plan_display(plan_id, user_id, locale, day_indices=None) -> dict
        {"enriched_meals": int, "skipped": str | None}. Jamás lanza (fail-open TOTAL — spec
        "Global Constraints": cualquier excepción del enriquecimiento es un warning + el plan
        se queda sin `_display` + el frontend cae a español, nunca rompe generación/swap/lectura).

    schedule_plan_display_enrichment(plan_id, user_id, locale, day_indices=None) -> None
        Wrapper fire-and-forget: thread background + dedupe. Los 5 disparadores (Task 2/3)
        llaman a este, no a `enrich_plan_display` directo.

Validación determinista (spec, sección "Contrato de datos"):
    - `recipe` e `ingredients` de salida deben tener la MISMA longitud que el original, en el
      mismo orden (alineados por índice) — si no, el meal ENTERO se descarta (no se persiste
      _display para ese meal).
    - Cada línea de `ingredients` debe contener el nombre canónico español del alimento como
      substring accent-insensitive (extraído del original con `constants.strip_accents`). Si
      la línea traducida lo pierde, SOLO esa línea cae de vuelta al original español (fallback
      per-línea, NO descarta el meal — "un gloss que pierde el identificador es peor que no
      tener gloss"). Si el original no tiene canónico identificable (p.ej. tras remover
      cantidad/unidad no queda texto), la línea pasa sin check.

Costo: NUNCA a `api_usage` (cero crédito de usuario) — telemetría a `llm_usage_events` vía
`db.log_llm_usage_event(node="plan_display_i18n")`.

Knobs (auto-registrados en `_KNOBS_REGISTRY` vía `knobs._env_bool/_env_str/_env_float`):
    MEALFIT_PLAN_DISPLAY_I18N            default True  — kill switch total.
    MEALFIT_PLAN_DISPLAY_I18N_MODEL      default flash — convención P3-PREVIEW-MODEL-KNOB.
    MEALFIT_PLAN_DISPLAY_I18N_TIMEOUT_S  default 60.0  — timeout del cliente LLM.
"""

import json
import logging
import re
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

from knobs import _env_bool, _env_str, _env_float
from constants import strip_accents
from prompts.chat_agent import _COACH_LANGUAGE_NAMES
from llm_provider import ChatDeepSeek, DEEPSEEK_FLASH
from langchain_core.messages import SystemMessage
from db import update_plan_data_atomic, log_llm_usage_event
from db_core import execute_sql_query

# ============================================================
# Knobs
# ============================================================


def _plan_display_i18n_enabled() -> bool:
    return _env_bool("MEALFIT_PLAN_DISPLAY_I18N", True)


def _plan_display_i18n_model_name() -> str:
    return _env_str("MEALFIT_PLAN_DISPLAY_I18N_MODEL", DEEPSEEK_FLASH)


def _plan_display_i18n_timeout_s() -> float:
    return _env_float("MEALFIT_PLAN_DISPLAY_I18N_TIMEOUT_S", 60.0)


# ============================================================
# Dedupe: lock in-process + marker cross-worker en app_kv_store
# (mismo patrón que `agent.py::_try_claim_title_lock_cross_worker`)
# ============================================================

_INFLIGHT: set = set()
_INFLIGHT_LOCK = threading.Lock()
_ENRICH_LOCK_TTL_S = 300  # 5 min — mismo TTL que el title-lock de referencia.


def _try_claim_enrich_lock_cross_worker(plan_id: str, locale: str) -> bool:
    """Claim atómico cross-worker vía UPSERT en `app_kv_store`. Espejo de
    `agent.py::_try_claim_title_lock_cross_worker`. Best-effort: si la DB no
    responde, retorna True (fail-open) — preferimos duplicar una llamada
    flash barata a bloquear el enriquecimiento por un outage del KV.
    """
    try:
        _now_ts = time.time()
        _kv_key = f"plan_display_enrich:{plan_id}:{locale}"
        result = execute_sql_query(
            """
            INSERT INTO app_kv_store (key, value)
            VALUES (%s, jsonb_build_object('started_at', %s::float))
            ON CONFLICT (key) DO UPDATE SET
                value = jsonb_build_object('started_at', %s::float),
                updated_at = NOW()
            WHERE COALESCE((app_kv_store.value->>'started_at')::float, 0)
                  < %s::float
            RETURNING key
            """,
            (_kv_key, _now_ts, _now_ts, _now_ts - _ENRICH_LOCK_TTL_S),
            fetch_all=True,
        )
        return bool(result)
    except Exception as e:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] claim de lock cross-worker falló (fail-open) "
            f"plan={plan_id} locale={locale}: {e!r}"
        )
        return True


# ============================================================
# Lectura del plan (ownership AND user_id, mismo patrón de routers/plans.py)
# ============================================================


def _fetch_plan_data(plan_id: str, user_id: str) -> Optional[dict]:
    try:
        row = execute_sql_query(
            "SELECT plan_data FROM meal_plans WHERE id = %s AND user_id = %s",
            (plan_id, user_id),
            fetch_one=True,
        )
    except Exception as e:
        logger.warning(f"[P1-PLAN-DISPLAY-I18N] SELECT plan falló plan={plan_id}: {e!r}")
        return None
    if not row:
        return None
    pd = row.get("plan_data")
    if isinstance(pd, str):
        try:
            pd = json.loads(pd)
        except Exception:
            return None
    if not isinstance(pd, dict):
        return None
    return pd


# ============================================================
# Extracción del nombre canónico (misma limpieza accent-insensitive que
# `constants.strip_accents`) — quita el prefijo de cantidad/unidad de una
# línea de ingrediente original ("30 g Habichuelas rojas" -> "Habichuelas
# rojas"). Si tras la limpieza no queda texto, no hay canónico identificable.
# ============================================================

_QTY_UNIT_PREFIX_RE = re.compile(
    r"^\s*[\d.,/½¼¾\s]*\s*"
    r"(kg|kilos?|g|gr|gramos?|ml|mls?|l|litros?|tazas?|taza|cditas?|cdtas?|cdas?|cucharadas?|"
    r"cucharaditas?|unidad(?:es)?|unids?|piezas?|pza|oz|onzas?|lb|lbs|libras?)?\s*",
    re.IGNORECASE,
)
_PARENTHETICAL_RE = re.compile(r"\([^)]*\)")


def _extract_canonical_name(ingredient_line: str) -> str:
    if not isinstance(ingredient_line, str):
        return ""
    s = ingredient_line.strip()
    if not s:
        return ""
    s = _QTY_UNIT_PREFIX_RE.sub("", s, count=1).strip()
    s = _PARENTHETICAL_RE.sub("", s).strip()
    return s


# ============================================================
# Prompt (UN lote por llamada) + parseo JSON estricto
# ============================================================

_JSON_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _build_prompt(targets: list, locale: str) -> str:
    idioma = _COACH_LANGUAGE_NAMES.get(locale, locale)
    lines = []
    for i, t in enumerate(targets):
        lines.append(f"{i}. NAME: {t['name']}")
        lines.append(f"   DESCRIPTION: {t['description']}")
        lines.append("   RECIPE:")
        for step_idx, step in enumerate(t["recipe"]):
            lines.append(f"     [{step_idx}] {step}")
        lines.append("   INGREDIENTS:")
        for ing_idx, ing in enumerate(t["ingredients"]):
            lines.append(f"     [{ing_idx}] {ing}")
    meals_block = "\n".join(lines)
    return (
        f"Traduce estos platos de un plan de comidas dominicano al {idioma}, para LECTURA del "
        f"usuario. El sistema sigue operando en español canónico internamente — esto es "
        f"SOLAMENTE una capa de display paralela.\n\n"
        f"REGLAS ESTRICTAS:\n"
        f"1. Responde 'name'/'description'/'recipe' EXCLUSIVAMENTE en {idioma}. EXCEPCIÓN "
        f"dentro de 'ingredients': cada línea lleva el nombre del alimento en el formato "
        f'"English gloss (Nombre canónico en español)" — el nombre canónico español SIEMPRE '
        f"debe aparecer literal, sin traducir, tal como en el original (es un identificador "
        f"del sistema).\n"
        f"2. Los arrays 'recipe' e 'ingredients' de salida DEBEN tener EXACTAMENTE la misma "
        f"cantidad de elementos que el original, en el MISMO orden (alineados por índice).\n"
        f"3. Responde SOLO con JSON válido, sin markdown ni texto fuera del JSON, con este "
        f"contrato exacto:\n"
        f'{{"meals":[{{"i":0,"name":"...","description":"...","recipe":["...","..."],'
        f'"ingredients":["...","..."]}}]}}\n\n'
        f"PLATOS ORIGINALES (español — NO traduzcas los nombres de alimentos fuera del gloss "
        f"de ingredients):\n"
        f"{meals_block}"
    )


def _parse_json_response(raw: str) -> Optional[dict]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    cleaned = _JSON_CODE_FENCE_RE.sub("", raw).strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


# ============================================================
# Validación por meal + construcción del `_display` final
# ============================================================


def _validate_and_build_display(original: dict, item: dict) -> Optional[dict]:
    name = item.get("name")
    description = item.get("description")
    recipe = item.get("recipe")
    ingredients = item.get("ingredients")

    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(description, str) or not description.strip():
        return None
    if not isinstance(recipe, list) or len(recipe) != len(original["recipe"]):
        return None
    if not isinstance(ingredients, list) or len(ingredients) != len(original["ingredients"]):
        return None

    final_ingredients = []
    for idx, translated_line in enumerate(ingredients):
        translated_line = translated_line if isinstance(translated_line, str) else ""
        original_line = original["ingredients"][idx]
        original_line = original_line if isinstance(original_line, str) else str(original_line)
        canonical = _extract_canonical_name(original_line)
        if not canonical:
            # Sin canónico identificable en el original: la línea pasa sin check
            # (spec: "si el original no tiene canónico identificable, la línea
            # pasa sin check").
            final_ingredients.append(translated_line.strip() or original_line)
            continue
        if strip_accents(canonical).lower() in strip_accents(translated_line).lower():
            final_ingredients.append(translated_line)
        else:
            # Un gloss que pierde el identificador es peor que no tener gloss:
            # se descarta ESA línea (no el meal) -> fallback al original español.
            final_ingredients.append(original_line)

    final_recipe = [step if isinstance(step, str) else str(step) for step in recipe]

    return {
        "name": name.strip(),
        "description": description.strip(),
        "recipe": final_recipe,
        "ingredients": final_ingredients,
    }


def _emit_usage_telemetry(plan_id: str, user_id: str, model_name: str, response) -> None:
    """Best-effort. NUNCA toca `api_usage` — solo `llm_usage_events` (libro de
    costo, node="plan_display_i18n"). Cualquier fallo se traga en silencio."""
    try:
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
        log_llm_usage_event(
            user_id=user_id,
            plan_id=plan_id,
            model=model_name,
            node="plan_display_i18n",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as e:
        logger.debug(f"[P1-PLAN-DISPLAY-I18N] telemetría falló (best-effort): {e!r}")


# ============================================================
# Motor
# ============================================================


def enrich_plan_display(
    plan_id: str,
    user_id: str,
    locale: str,
    day_indices: Optional[list] = None,
) -> dict:
    """Enriquece `_display[locale]` de los meals de `plan_id` (ownership
    `AND user_id`). JAMÁS lanza — fail-open total (ver docstring del módulo).

    Returns:
        {"enriched_meals": int, "skipped": str | None}
    """
    key = (plan_id, locale)
    try:
        if not _plan_display_i18n_enabled():
            return {"enriched_meals": 0, "skipped": "knob_off"}
        if not isinstance(locale, str) or locale not in _COACH_LANGUAGE_NAMES:
            # Cubre es-DO (nunca en el dict) y cualquier locale inválido.
            return {"enriched_meals": 0, "skipped": "locale"}

        plan_data = _fetch_plan_data(plan_id, user_id)
        if plan_data is None:
            return {"enriched_meals": 0, "skipped": "not_found"}

        days = plan_data.get("days")
        if not isinstance(days, list) or not days:
            return {"enriched_meals": 0, "skipped": "no_days"}

        with _INFLIGHT_LOCK:
            if key in _INFLIGHT:
                return {"enriched_meals": 0, "skipped": "dedupe_inprocess"}
            _INFLIGHT.add(key)

        try:
            if not _try_claim_enrich_lock_cross_worker(plan_id, locale):
                return {"enriched_meals": 0, "skipped": "dedupe_locked"}

            idx_set = set(day_indices) if day_indices is not None else None
            targets = []
            for day_idx, day in enumerate(days):
                if idx_set is not None and day_idx not in idx_set:
                    continue
                if not isinstance(day, dict):
                    continue
                meals = day.get("meals")
                if not isinstance(meals, list):
                    continue
                for meal_idx, meal in enumerate(meals):
                    if not isinstance(meal, dict):
                        continue
                    recipe = meal.get("recipe")
                    ingredients = meal.get("ingredients")
                    targets.append(
                        {
                            "day_idx": day_idx,
                            "meal_idx": meal_idx,
                            "name": meal.get("name") or "",
                            "description": meal.get("description") or "",
                            "recipe": recipe if isinstance(recipe, list) else [],
                            "ingredients": ingredients if isinstance(ingredients, list) else [],
                        }
                    )

            if not targets:
                return {"enriched_meals": 0, "skipped": "no_meals"}

            prompt = _build_prompt(targets, locale)
            model_name = _plan_display_i18n_model_name()

            try:
                llm = ChatDeepSeek(
                    model=model_name,
                    temperature=0.2,
                    timeout=_plan_display_i18n_timeout_s(),
                )
                response = llm.invoke([SystemMessage(content=prompt)])
            except Exception as e:
                logger.warning(
                    f"[P1-PLAN-DISPLAY-I18N] LLM invoke falló plan={plan_id} "
                    f"locale={locale}: {e!r}"
                )
                return {"enriched_meals": 0, "skipped": "llm_exception"}

            raw_content = getattr(response, "content", "") or ""
            parsed = _parse_json_response(raw_content)
            if parsed is None or not isinstance(parsed.get("meals"), list):
                return {"enriched_meals": 0, "skipped": "json_parse_error"}

            valid_by_index = {}
            for item in parsed["meals"]:
                if not isinstance(item, dict):
                    continue
                i = item.get("i")
                if not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(targets):
                    continue
                display = _validate_and_build_display(targets[i], item)
                if display is not None:
                    valid_by_index[i] = display

            if not valid_by_index:
                logger.info(
                    f"[P1-PLAN-DISPLAY-I18N] 0/{len(targets)} meals validados "
                    f"plan={plan_id} locale={locale}"
                )
                return {"enriched_meals": 0, "skipped": "no_valid_meals"}

            def _mutator(pd: dict):
                pd_days = pd.get("days") if isinstance(pd, dict) else None
                if not isinstance(pd_days, list):
                    return pd
                for i, display in valid_by_index.items():
                    t = targets[i]
                    try:
                        day = pd_days[t["day_idx"]]
                        meal = day.get("meals", [])[t["meal_idx"]]
                    except (IndexError, TypeError, KeyError, AttributeError):
                        continue
                    if not isinstance(meal, dict):
                        continue
                    disp_map = meal.get("_display")
                    if not isinstance(disp_map, dict):
                        disp_map = {}
                    disp_map[locale] = display
                    meal["_display"] = disp_map
                return pd

            try:
                update_plan_data_atomic(plan_id, _mutator, user_id=user_id)
            except Exception as e:
                logger.warning(
                    f"[P1-PLAN-DISPLAY-I18N] persist falló plan={plan_id} "
                    f"locale={locale}: {e!r}"
                )
                return {"enriched_meals": 0, "skipped": "persist_exception"}

            _emit_usage_telemetry(plan_id, user_id, model_name, response)

            logger.info(
                f"[P1-PLAN-DISPLAY-I18N] enriquecidos {len(valid_by_index)}/{len(targets)} "
                f"meals plan={plan_id} locale={locale}"
            )
            return {"enriched_meals": len(valid_by_index), "skipped": None}
        finally:
            with _INFLIGHT_LOCK:
                _INFLIGHT.discard(key)
    except Exception as e:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] enrich_plan_display excepción no controlada "
            f"(fail-open) plan={plan_id} locale={locale}: {e!r}"
        )
        return {"enriched_meals": 0, "skipped": "exception"}


def schedule_plan_display_enrichment(
    plan_id: str,
    user_id: str,
    locale: str,
    day_indices: Optional[list] = None,
) -> None:
    """Wrapper fire-and-forget: dispara `enrich_plan_display` en un thread
    background. Nunca lanza — cualquier fallo (incluso al lanzar el thread)
    queda en warning. Los 5 disparadores de la spec llaman a este helper,
    nunca al motor directo, así el caller no bloquea su propio request.
    """
    try:
        if not _plan_display_i18n_enabled():
            return
        if not isinstance(locale, str) or locale not in _COACH_LANGUAGE_NAMES:
            return

        # Pre-check barato in-process: evita levantar un thread si ya hay uno
        # en vuelo para el mismo (plan_id, locale). No sustituye el dedupe
        # real (cross-worker + re-check) que vive dentro de `enrich_plan_display`.
        with _INFLIGHT_LOCK:
            if (plan_id, locale) in _INFLIGHT:
                logger.debug(
                    f"[P1-PLAN-DISPLAY-I18N] schedule skip — ya en vuelo "
                    f"plan={plan_id} locale={locale}"
                )
                return

        def _run():
            try:
                result = enrich_plan_display(
                    plan_id, user_id, locale, day_indices=day_indices
                )
                logger.info(
                    f"[P1-PLAN-DISPLAY-I18N] background enrich plan={plan_id} "
                    f"locale={locale} result={result}"
                )
            except Exception as e:
                # enrich_plan_display ya es fail-open y no debería lanzar, pero
                # el thread background no debe morir ruidoso bajo ningún caso.
                logger.warning(
                    f"[P1-PLAN-DISPLAY-I18N] background enrich excepción "
                    f"(fail-open) plan={plan_id} locale={locale}: {e!r}"
                )

        threading.Thread(target=_run, daemon=True).start()
    except Exception as e:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] schedule_plan_display_enrichment falló "
            f"(fail-open) plan={plan_id} locale={locale}: {e!r}"
        )
