from fastapi import APIRouter, Body, Depends, HTTPException, BackgroundTasks, Request, Response
from fastapi.responses import StreamingResponse
from typing import Optional
import logging
import traceback
import os
import threading
import asyncio
import json as _json
import time as _time
from datetime import datetime, timezone, timedelta

# Importaciones relativas del entorno
from auth import get_verified_user_id, verify_api_quota
from db import (
    supabase, get_user_likes, get_active_rejections, get_or_create_session, 
    save_message, update_user_health_profile, log_api_usage, get_latest_meal_plan,
    get_latest_meal_plan_with_id, update_meal_plan_data, insert_like
)
from memory_manager import build_memory_context, summarize_and_prune
from agent import analyze_preferences_agent, swap_meal
from graph_orchestrator import (
    run_plan_pipeline,
    arun_plan_pipeline,
    _strip_untrusted_internal_keys,
    _enforce_days_to_generate_cap,
    # [P1-FORM-6] Merge defensivo de `other*` text fields al router. El merge
    # también ocurre dentro de `arun_plan_pipeline` (línea ~8430); idempotente
    # (dedup case-insensitive). Hacerlo al router blinda contra futuros
    # consumers en el handler que pudieran leer `allergies`/`medicalConditions`/
    # `dislikes`/`struggles` antes de la llamada al pipeline. HOY ningún
    # consumer del router toca esos campos, pero el contrato explícito
    # previene regresiones.
    _merge_other_text_fields,
)
from ai_helpers import expand_recipe_agent
from services import _save_plan_and_track_background, _process_swap_rejection_background, save_partial_plan_get_id
from db_inventory import restock_inventory, consume_inventory_items_completely
from rate_limiter import RateLimiter
from schemas import PUBLIC_SSE_EVENTS  # [P1-11] contrato público de eventos SSE

logger = logging.getLogger(__name__)

# [P1-6] Rate limit per-endpoint para generación de planes.
# Antes solo `LLM_MAX_PER_USER` (semáforo dentro del orquestador) limitaba
# concurrencia, pero un usuario podía disparar 50 requests al endpoint en
# paralelo y agotar `_SYNC_WRAPPER_EXECUTOR` o saturar el thread pool del
# SSE — sin contar el costo LLM (cada plan ~3-7 minutos × $0.X). 3/60s
# (1 plan cada ~20s) es suficiente para regeneraciones legítimas (un usuario
# real no genera más de 3 planes en un minuto), y bloquea spammers/bugs en
# frontend que disparan en bucle.
#
# Bucket key: `verified_user_id` para auth, `ip:<host>` para anon (vía la
# extensión P1-6 de `RateLimiter`). Singleton a nivel módulo para que todos
# los workers de un proceso compartan el contador (más estricto y eficiente
# que crear uno por request).
_PLAN_GEN_LIMITER = RateLimiter(max_calls=3, period_seconds=60)

router = APIRouter(
    prefix="/api/plans",
    tags=["plans"],
)


def _resolve_request_tz_offset(payload_value, user_id: Optional[str]) -> int:
    """[P1-1] Resuelve `tz_offset_minutes` con prioridad explícita.

    Antes los handlers (/analyze, /analyze/stream, /shift-plan) hacían
    `int(data.get("tzOffset", 0))`, colapsando "frontend no envió" con
    "frontend envió 0 (UTC)". Para usuarios genuinamente UTC el resultado es
    correcto, pero para usuarios en TZ no-UTC cuyo cliente legacy/móvil
    omitió `tzOffset`, todo el cálculo de localización y `days_since_creation`
    quedaba en UTC — produciendo off-by-one en `chunk_size_for_next_slot`
    (e.g., un plan 7d generaba el chunk 2 con count=3 cuando debía ser 4).

    Resolución:
      1. Si `payload_value` es no-None: usarlo (incluyendo `0` explícito).
         Es la fuente de verdad más fresca cuando el frontend la provee.
      2. Si es None y hay `user_id`: leer `health_profile.tz_offset_minutes`
         vivo vía `cron_tasks._get_user_tz_live` (helper que ya usan los
         flujos sensibles a TZ del worker).
      3. Si tampoco hay perfil o falla la lectura: 0 (UTC).

    Args:
        payload_value: valor crudo de `data.get("tzOffset")` — puede ser int,
            string convertible a int, o None.
        user_id: id del usuario para fallback al perfil; None para guest.

    Returns:
        Offset en minutos (formato JS: positivo para TZ negativas, e.g. +240
        para UTC-4).
    """
    if payload_value is not None:
        try:
            return int(payload_value)
        except (TypeError, ValueError):
            pass  # cae a fallback de perfil
    if user_id and user_id != "guest":
        try:
            from cron_tasks import _get_user_tz_live
            return _get_user_tz_live(user_id, fallback_minutes=0)
        except Exception as _tz_err:
            logger.debug(f"[P1-1] Fallback de TZ desde perfil falló para {user_id}: {_tz_err}")
    return 0


def _resolve_live_pantry(actual_user_id: Optional[str], data: dict) -> list:
    """[P0-1] Resuelve el inventario vivo del usuario para la validación post-LLM.

    Antes el handler `api_analyze` referenciaba `_live_pantry` sin asignarla
    (un refactor previo dejó el bloque `if _live_pantry:` huérfano), lo que
    disparaba `NameError` → HTTP 500 en cualquier request que llegara hasta la
    fase de validación. Este helper centraliza la resolución con tres fuentes,
    en orden de prioridad:

      1. Usuario autenticado → `db_inventory.get_user_inventory_net(uid)`,
         mismo pattern que ya usa `cron_tasks` para los flujos sensibles a
         stock (renewal, chunk worker). Devuelve formato heterogéneo (lista
         de strings tipo "2 unidades de Pollo") compatible con
         `_validate_and_retry_initial_chunk_against_pantry`.
      2. Fallo de DB o lectura vacía → fallback al payload del cliente:
         `data["current_pantry_ingredients"]` (frontend lo envía siempre,
         ver Plan.jsx). Para guests es la única fuente posible.
      3. Sin nada utilizable → `[]`. El caller saltará la validación
         post-LLM (corto-circuito sano: sin pantry no hay nada contra qué
         validar).

    Defensive: nunca propaga excepciones del DB. Si algo falla, loguea
    warning y degrada al payload — el endpoint no debe morir por un fallo
    transitorio de inventario.

    Args:
        actual_user_id: id verificado del usuario (None para guest).
        data: payload crudo del request, fuente de fallback.

    Returns:
        list[str]: inventario en formato compatible con el helper de
            validación. Lista vacía si ninguna fuente entrega datos.
    """
    if actual_user_id:
        try:
            from db_inventory import get_user_inventory_net
            live_inv = get_user_inventory_net(actual_user_id)
            if live_inv:
                return list(live_inv)
        except Exception as _inv_err:
            logger.warning(
                f"⚠️ [P0-1] Lectura de inventario vivo falló para user "
                f"{actual_user_id}: {_inv_err}. Cayendo a payload."
            )
    payload_pantry = data.get("current_pantry_ingredients") or []
    if isinstance(payload_pantry, list):
        return [str(x) for x in payload_pantry if x]
    return []


# [P1-5] Campos mínimos requeridos por el pipeline de generación. Su ausencia
# hace que `nutrition_calculator.get_nutrition_targets` defaultee silenciosamente
# a valores genéricos ("adulto promedio": 25 años, 154 lb, 170 cm, hombre,
# moderate) y emita un plan que no representa al usuario. Validar aquí — antes
# de invocar el pipeline — corta 30–90s de compute LLM + chunking + persistencia
# basada en datos basura, y devuelve un 422 accionable al cliente.
#
# Este set replica lo que el frontend ya valida en `Plan.jsx` (`age`,
# `mainGoal`) más los campos biométricos que el calculador no puede defaultear
# de forma segura. Si en el futuro se relaja un requisito (e.g., `gender`
# inferido por nombre o por defecto neutral), basta con quitarlo de la tupla.
_REQUIRED_FORM_FIELDS = (
    "age", "mainGoal", "weight", "height", "gender", "activityLevel",
    # [P0-FORM-4] `weightUnit` es required: sin él, `_validate_form_data_ranges`
    # y `nutrition_calculator.get_nutrition_targets` defaulteaban silenciosamente
    # a "lb". Si el usuario en realidad ingresó kg (caso real de cliente legacy
    # o hidratación incompleta desde DB), un peso de 70 (kg) se interpretaba
    # como 70/2.20462 = 31.7 kg → BMR/macros completamente errados, justo por
    # encima del mínimo de 30 kg, sin disparar el chequeo de rango. Hacerlo
    # required convierte la falla silenciosa en 422 accionable.
    "weightUnit",
    # [P0-FORM-1] `householdSize` y `groceryDuration` consumidos por el
    # pipeline (`graph_orchestrator.py:5028,5046`, `tools.py:464,467`,
    # `ai_helpers.py:387,611`, `cron_tasks.py:7756,7921,9289,9295,17465,17472`,
    # `routers/plans.py:2190,2191`) y por la lista de compras / shopping
    # calculator. El frontend (`InteractiveQuestions.jsx:804`) ya bloquea el
    # botón "Siguiente" si faltan, pero un cliente no oficial o una hidratación
    # rota desde localStorage los omitía → backend defaulteaba silenciosamente
    # a `1 persona` / `"weekly"`. Resultado: lista de compras escalada para 1
    # cuando el usuario eligió 4 → faltante crítico de comida → plan inservible.
    # Hacerlo required convierte la pérdida silenciosa en 422 accionable.
    "householdSize", "groceryDuration",
    # [P0-FORM-3] `motivation` se captura en QMotivation ("¿Por qué quieres
    # hacer esto AHORA?" — subtitle: "será tu gasolina en días difíciles").
    # ANTES era un campo huérfano: se persistía a `health_profile` y se enviaba
    # al pipeline pero ningún consumer lo leía → promesa rota al usuario, señal
    # emocional valiosa descartada. AHORA `build_motivation_context` (en
    # `prompts/plan_generator.py`) lo inyecta al planner + day generator del
    # LLM como contexto emocional para tono y descripciones de platos.
    # Validación de PRESENCIA: `_validate_form_data_min` rechaza None / "" /
    # whitespace-only via `value.strip() == ""` → 422 accionable en lugar de
    # plan generado con coaching genérico.
    "motivation",
    # [P1-2] `allergies` y `medicalConditions` añadidos como required con
    # enforcement de array no-vacío (ver `_REQUIRED_NONEMPTY_ARRAY_FIELDS`).
    # Defense-in-depth contra clientes no oficiales: el frontend ahora exige
    # señal explícita (chip / sentinel "Ninguna" / free-text) en sus QComponents,
    # pero un mobile legacy / scraper / hidratación rota podía mandar `[]` →
    # backend interpretaba "sin restricciones" → LLM podía generar plan con
    # maní / gluten / etc. para usuario realmente alérgico. Riesgo de SAFETY
    # MÉDICA directa. Ahora rechazamos con 422 si el array está vacío;
    # marcando "Ninguna" produce `["Ninguna"]` que pasa la validación
    # (length=1 con sentinel reconocido downstream por `_SENTINEL_NONE_VALUES`
    # en `graph_orchestrator.py`).
    "allergies", "medicalConditions",
)

# [P1-2] Subset de `_REQUIRED_FORM_FIELDS` donde `[]` cuenta como ausente.
# Comportamiento histórico de `_validate_form_data_min` (mantenido por compat
# para los demás fields): solo `None` y string vacío son "ausente". Para
# safety-critical fields necesitamos rechazar arrays vacíos también, porque
# downstream el LLM trata `[]` como "sin restricciones declaradas" — un modo
# de fallo silencioso de safety médica para alergias/condiciones.
_REQUIRED_NONEMPTY_ARRAY_FIELDS = frozenset({"allergies", "medicalConditions"})

# [P0-FORM-4] Valores aceptados para `weightUnit`. Comparación case-insensitive
# tras `.strip()` — frontend manda "lb"/"kg" pero un cliente legacy podría
# mandar "LB"/" lb "/etc. Cualquier otro valor (incluyendo "lbs", "kgs",
# "pounds") se rechaza con 422 — no normalizamos silenciosamente para evitar
# que un typo "lbz" se interprete como "lb" y produzca un plan basado en una
# suposición incorrecta del cálculo nutricional.
_WEIGHT_UNIT_ACCEPTED = frozenset({"lb", "kg"})

# [P1-FORM-8] Valores aceptados para `dietType`. ANTES, el backend NO
# validaba el enum: el wizard solo ofrece `balanced/vegetarian/vegan` en
# `QDietType` (`InteractiveQuestions.jsx:430-434`), pero un cliente no oficial
# (mobile legacy, scraper, integración rota, request directo a la API) podía
# enviar `dietType="keto"`, `"paleo"`, `""`, o cualquier string arbitrario.
# El orquestador lo pasaba al LLM y al filtro de catálogo
# (`constants._get_fast_filtered_catalogs`, líneas 1250+).
#
# Tras inspección de la realidad downstream:
#   - `_get_fast_filtered_catalogs` reconoce explícitamente
#     `vegano/vegan`, `vegetariano/vegetarian`, `pescetariano/pescatarian`
#     y CAE A FALLBACK SILENCIOSO (catálogo completo, sin restricciones) para
#     cualquier otro valor — incluyendo "Omnívora" (legado), "keto" (cliente
#     malintencionado), o typos.
#   - Datos históricos: tests E2E + fixtures de DB usan `"Omnívora"` como
#     equivalente a `balanced`. Posibles `health_profile.dietType="Omnívora"`
#     en producción para usuarios pre-migración del wizard.
#
# Estrategia de doble nivel para no romper legacy + bloquear payload bogus:
#   - `_DIET_TYPE_CANONICAL`: valores canónicos del wizard actual. Pasan sin
#     log.
#   - `_DIET_TYPE_LEGACY_ACCEPTED`: variantes ES + diets adicionales que el
#     catálogo downstream ya manejaba antes de este check. Pasan PERO emiten
#     `logger.info` para observabilidad — operadores pueden trackear si
#     todavía hay tráfico legacy y planificar una migración de DB.
#   - Cualquier valor fuera de la unión → 422 `invalid_biometric_range`.
#
# Mantenimiento: SSOT con `frontend/src/config/formValidation.js DIET_TYPES`
# (canónicos). Si se añade un canónico al frontend (ej. "keto"), DEBE
# añadirse acá como canónico Y al filtro de `_get_fast_filtered_catalogs`.
# Si una variante legacy nueva aparece en producción (post-mortem de un 422),
# moverla al set legacy.
#
# `dietType` NO está en `_REQUIRED_FORM_FIELDS` (presence-optional, default a
# `balanced` downstream — decisión documentada en `formValidation.js` líneas
# 74-78, "rechazarlo rompería clientes legacy sin beneficio de safety"). Por
# eso `_validate_form_data_ranges` lo trata como opcional: solo valida el
# enum si el campo está PRESENTE y no vacío. Ausente → no-op.
_DIET_TYPE_CANONICAL = frozenset({"balanced", "vegetarian", "vegan"})
_DIET_TYPE_LEGACY_ACCEPTED = frozenset({
    "omnivora",       # legacy ES, equivalente a balanced (fallthrough en catalog)
    "vegetariana",    # legacy ES de vegetarian
    "vegana",         # legacy ES de vegan
    "vegetariano",    # masc. legacy ES (catalog reconoce)
    "vegano",         # masc. legacy ES (catalog reconoce)
    "pescatarian",    # variante en catalog pero no en wizard actual
    "pescetariano",   # variante ES en catalog
})
_DIET_TYPE_ENUM = _DIET_TYPE_CANONICAL | _DIET_TYPE_LEGACY_ACCEPTED


# [P0-FORM-5] Valores aceptados para `activityLevel` y `mainGoal`. ANTES, el
# backend NO validaba estos enums. `nutrition_calculator.calculate_tdee`
# (línea ~64) hacía `ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)` →
# defaultea silenciosamente a `moderate` (×1.55) para CUALQUIER valor desconocido.
# `apply_goal_adjustment` (línea ~70) hacía `GOAL_ADJUSTMENTS.get(goal, 0.0)`
# → defaultea a `maintenance` (déficit/superávit=0). Ambos sin telemetría.
#
# Si un cliente legacy/no-oficial enviaba `activityLevel="trabajo en oficina"`
# o `mainGoal="bulking"`, el calculador producía BMR/TDEE/macros completamente
# erróneos y el plan se generaba con datos basura sin disparar warning.
# Misma clase de bug que P0-FORM-4 resolvió para `weightUnit`.
#
# Estrategia: validación estricta (sin doble nivel legacy) porque a diferencia
# de `dietType`, no existen tests/fixtures con valores legacy en español
# para estos enums — el wizard siempre envió los lower_case canónicos. Cualquier
# valor fuera del enum es señal clara de cliente no oficial o tampering →
# 422 con `accepted_range`.
#
# Comparación case-insensitive tras `.strip()` para tolerar input laxo, pero
# normalización ASCII no aplica (todos los valores son ya lower-case ASCII).
#
# Mantenimiento: SSOT con `frontend/src/config/formValidation.js`. Si se añade
# un nuevo nivel/goal al wizard, actualizar AMBOS lados Y `nutrition_calculator`
# (`ACTIVITY_MULTIPLIERS`, `GOAL_ADJUSTMENTS`, `MACRO_SPLITS`).
_ACTIVITY_LEVEL_ENUM = frozenset({
    "sedentary", "light", "moderate", "active", "athlete",
})
_MAIN_GOAL_ENUM = frozenset({
    "lose_fat", "gain_muscle", "maintenance", "performance",
})

# [P1-FORM-10] Enums no-críticos del wizard. Severidad menor que P0-FORM-5
# porque ningún downstream defaultea silenciosamente a un valor erróneo:
#   - `scheduleType` → solo lo lee `build_time_context` para hint del LLM (texto).
#   - `cookingTime` → idem, hint textual al LLM y al filtro de complejidad.
#   - `budget` → influye en `_get_fast_filtered_catalogs` (catálogo de
#      ingredientes); valor desconocido = catálogo completo (no peligroso).
#   - `groceryDuration` → en frontend rige `getTotalDaysByGroceryDuration`
#      (defaultea a 'weekly' = 7 días); en backend afecta scaling de la lista
#      de compras (defaultea a 'weekly').
#   - `sleepHours`, `stressLevel` → solo hints textuales al LLM.
#
# Por qué validar entonces:
#   1. Defense-in-depth contra cliente legacy/no-oficial que envíe basura
#      al LLM como "horario" o "estrés" arbitrario (vector de prompt-injection
#      ortogonal al ya cubierto en P1-Q8 — P1-Q8 detecta patrones, no enums).
#   2. Observabilidad: detectar drift de schema (frontend renombra un valor
#      sin actualizar backend → 422 en lugar de pase silencioso al LLM).
#   3. UX: 422 accionable mejor que un plan generado con horario "extraterrestre"
#      que el LLM intentó interpretar.
#
# `sleepHours` / `stressLevel` son strings con espacios y mayúsculas
# ("< 6 horas", "Bajo") porque el wizard los usa como labels directamente.
# Mantenemos exactly-match (sin .lower()) para preservar el match con el
# downstream existente.
_SCHEDULE_TYPE_ENUM = frozenset({"standard", "night_shift", "variable"})
_COOKING_TIME_ENUM  = frozenset({"none", "30min", "1hour", "plenty"})
_BUDGET_ENUM        = frozenset({"low", "medium", "high", "unlimited"})
_GROCERY_DURATION_ENUM = frozenset({"weekly", "biweekly", "monthly"})
_SLEEP_HOURS_ENUM   = frozenset({"< 6 horas", "6-7 horas", "7-8 horas", "> 8 horas"})
_STRESS_LEVEL_ENUM  = frozenset({"Bajo", "Moderado", "Alto", "Muy Alto"})

# Tabla de validación uniforme: field → (enum, normalizer, accepted_label).
# El normalizer aplica la transformación que el campo permite antes del match
# (lower/strip o exact). `accepted_label` es lo que se muestra en el 422.
_NON_CRITICAL_ENUM_VALIDATIONS = (
    ("scheduleType",    _SCHEDULE_TYPE_ENUM,    True,  "standard|night_shift|variable"),
    ("cookingTime",     _COOKING_TIME_ENUM,     True,  "none|30min|1hour|plenty"),
    ("budget",          _BUDGET_ENUM,           True,  "low|medium|high|unlimited"),
    ("groceryDuration", _GROCERY_DURATION_ENUM, True,  "weekly|biweekly|monthly"),
    ("sleepHours",      _SLEEP_HOURS_ENUM,      False, "< 6 horas|6-7 horas|7-8 horas|> 8 horas"),
    ("stressLevel",     _STRESS_LEVEL_ENUM,     False, "Bajo|Moderado|Alto|Muy Alto"),
)

# [P1-FORM-11] Enum de `selectedSupplements`. SSOT con `SUPPLEMENT_NAMES.keys()`
# de `constants.py` (espejado del wizard `QSupplements`). ANTES, el backend
# aceptaba CUALQUIER string en este array y `build_supplements_context` lo
# inyectaba como obligatorio al prompt del LLM via `SUPPLEMENT_NAMES.get(s, s)`
# que devuelve la key cruda al fallback → cliente no oficial podía mandar
# `selectedSupplements=["esteroides anabólicos"]` y el LLM lo veía como
# "DEBES incluir: esteroides anabólicos". Vector ortogonal al regex anti-injection
# de P1-Q8 (que detecta patrones tipo "ignore previous", no enums médicos).
# Validación case-sensitive porque el wizard manda lower_case + snake_case
# canónico; un valor con mayúsculas también se rechaza para forzar consistencia.
_SUPPLEMENT_ENUM = frozenset({
    "whey_protein", "vegan_protein", "creatine", "bcaa", "pre_workout",
    "fat_burner", "collagen", "multivitamin", "omega3", "magnesium",
    "probiotics", "electrolytes",
})


def _validate_form_data_min(data: dict) -> tuple[bool, list[str]]:
    """[P1-5] Valida la presencia mínima de campos críticos del formulario.

    El frontend valida `age` + `mainGoal` antes de navegar a /plan, pero un
    cliente no oficial (mobile legacy, scraper, request directo) puede mandar
    payload incompleto. Sin esto, el pipeline corre con defaults genéricos y
    entrega un plan que no representa al usuario — peor aún, dispara chunking
    + persistencia + costo LLM para un resultado inservible.

    Args:
        data: payload crudo del request.

    Returns:
        `(is_valid, missing_fields)`. Si `is_valid=False`, el caller debe
        rechazar la request con 422 incluyendo `missing_fields` en el detail.
    """
    missing = []
    for field in _REQUIRED_FORM_FIELDS:
        value = data.get(field)
        # `None` y string vacío cuentan como ausentes. `0` y listas vacías NO
        # se consideran ausencia POR DEFAULT — el calculador los maneja vía
        # try/except y son legítimos en algunos contextos (e.g., `weight=0`
        # para guests antes de que el helper aplique el default real).
        #
        # [P1-2] EXCEPCIÓN para safety-critical fields (`allergies`,
        # `medicalConditions`): `[]` SÍ cuenta como ausente. Ver
        # `_REQUIRED_NONEMPTY_ARRAY_FIELDS` arriba para el rationale (riesgo
        # de safety médica si el LLM lee `[]` como "sin restricciones"). El
        # frontend produce `["Ninguna"]` cuando el usuario marca el sentinel
        # explícito, así que un usuario sin alergias reales sigue pasando esta
        # validación con un solo click.
        if value is None or (isinstance(value, str) and value.strip() == ""):
            missing.append(field)
        elif (
            field in _REQUIRED_NONEMPTY_ARRAY_FIELDS
            and isinstance(value, list)
            and len(value) == 0
        ):
            missing.append(field)
    return (len(missing) == 0, missing)


# ============================================================
# [P1-3] Validación de rangos biométricos plausibles
# ------------------------------------------------------------
# `_validate_form_data_min` solo verifica PRESENCIA. Un valor presente pero
# implausible (typo "300" cuando quería "30", clientes no oficiales con bugs,
# edad infantil fuera de scope, etc.) pasaba al `nutrition_calculator` que:
#   - hace try/except y defaultea a 154/170/25 si parse falla,
#   - PERO si el valor parsea bien numéricamente (e.g., age=5, weight=10kg,
#     height=400cm), produce un BMR no fisiológico → calorías absurdas →
#     plan inservible → cuota gastada.
# Este validador corre DESPUÉS del check de presencia y rechaza con un 422
# accionable que indica `field`, `value`, `accepted_range`, y `unit`. El
# frontend muestra al usuario qué corregir sin esperar 30-90s de pipeline.
#
# Mantener alineado con `frontend/src/config/formValidation.js BIO_RANGES`.
# Backend es source of truth; el frontend solo gatea UX para feedback inmediato.
#
# Filosofía de los rangos: PERMISIVOS en sentido médico (no rechazamos BMI<18.5
# o usuarios atléticos con %BF<5), solo blindamos contra TYPOS y BOGUS payloads.
# Cubrimos extremos humanos reales (3'3" — 8'2", 30-300 kg, 12-100 años).
# ============================================================
_BIO_RANGES = {
    "age":       (12, 100),       # años; debajo = pediatría fuera de scope
    "weight_kg": (30.0, 300.0),   # kg post-conversión; el parser hace lb→kg
    "height_cm": (100, 250),      # cm; ~3'3" a 8'2", cubre extremos humanos
    "bodyFat":   (1.0, 60.0),     # %; cap para no romper Katch-McArdle (LBM>0)
    # [P1-FORM-12] Tamaño del hogar para escalar lista de compras. Wizard
    # ofrece chips 1..6 (`InteractiveQuestions.jsx:830`). Cap alto en 12 para
    # cubrir familias extendidas / hogares múltiples sin bloquear casos
    # legítimos. ANTES no se validaba: `householdSize=999` o `="abc"` pasaban
    # al `shopping_calculator` que multiplicaba cantidades x999 → lista de
    # compras absurda + posible OOM en agregación de items.
    "household": (1, 12),         # personas; cap protege contra typos / payloads bogus
}


def _coerce_numeric(raw, *, kind: str = "float"):
    """[P1-3] Convierte un valor del payload a int/float si es parseable.

    Acepta int/float directos, strings con dígitos, y strings con coma
    decimal ("70,5" → 70.5 — formato regional ES). Devuelve None para
    cualquier otra cosa (None, "", lista, dict, basura no numérica) y deja
    al caller decidir si es ausencia o bogus.

    Args:
        raw: valor crudo del payload.
        kind: "int" (entero, redondea floats hacia abajo via int(float(...)))
              o "float" (default).

    Returns:
        El número parseado, o None si no es parseable.
    """
    if raw is None or raw == "":
        return None
    try:
        if isinstance(raw, str):
            raw = raw.strip().replace(",", ".")
        if kind == "int":
            return int(float(raw))
        return float(raw)
    except (TypeError, ValueError):
        return None


def _validate_form_data_ranges(data: dict) -> tuple[bool, list[dict]]:
    """[P1-3] Valida rangos biométricos plausibles post-presencia.

    Asume que `_validate_form_data_min` ya pasó (campos críticos presentes).
    Aquí cubrimos el caso ortogonal: presente pero fuera de rango plausible.
    Si un valor vino como string-de-basura ("abc"), `_coerce_numeric` devuelve
    None y lo reportamos como out-of-range con `value` original — UX más
    clara que "weight ausente" cuando el usuario sí escribió algo inválido.

    Para `weight`, convierte LB→KG antes de comparar, leyendo `weightUnit`
    (default "lb" — coincide con `initialFormData` del frontend). Para
    `height`, asumimos cm (la UI convierte ft/in localmente en QMeasurements
    antes de enviar).

    `bodyFat` es OPCIONAL: solo se valida si está presente y no vacío.

    Returns:
        `(is_valid, errors)`. Cada error es un dict serializable con `field`,
        `value` (raw original), `accepted_range` (tupla), y `unit` (string
        descriptivo). errors=[] si todo OK.
    """
    errors: list[dict] = []

    # --- age ---
    age_raw = data.get("age")
    age = _coerce_numeric(age_raw, kind="int")
    age_min, age_max = _BIO_RANGES["age"]
    if age is None or not (age_min <= age <= age_max):
        errors.append({
            "field": "age",
            "value": age_raw,
            "accepted_range": [age_min, age_max],
            "unit": "años",
        })

    # --- height (cm) ---
    height_raw = data.get("height")
    height = _coerce_numeric(height_raw, kind="int")
    h_min, h_max = _BIO_RANGES["height_cm"]
    if height is None or not (h_min <= height <= h_max):
        errors.append({
            "field": "height",
            "value": height_raw,
            "accepted_range": [h_min, h_max],
            "unit": "cm",
        })

    # --- weight (post lb→kg conversión) ---
    weight_raw = data.get("weight")
    weight_unit_raw = data.get("weightUnit")
    # [P0-FORM-4] No defaultear silenciosamente. Si llegó vacío/None aquí, el
    # check de presencia de `_validate_form_data_min` ya debió rechazar — pero
    # si por alguna razón el caller saltó ese check, devolvemos un error
    # explícito en vez de asumir "lb" (que era el bug original).
    weight_unit = str(weight_unit_raw or "").lower().strip()
    if weight_unit not in _WEIGHT_UNIT_ACCEPTED:
        errors.append({
            "field": "weightUnit",
            "value": weight_unit_raw,
            "accepted_range": sorted(_WEIGHT_UNIT_ACCEPTED),
            "unit": "string ('lb' o 'kg')",
        })
        # Sin unidad válida no podemos validar el rango de weight; saltamos
        # al siguiente field para no emitir un segundo error confuso sobre
        # "weight fuera de rango" cuya causa real es la unidad ausente.
    else:
        weight = _coerce_numeric(weight_raw, kind="float")
        if weight is not None and weight_unit == "lb":
            # Misma constante que `nutrition_calculator.get_nutrition_targets`:
            # 1 kg = 2.20462 lb. El cálculo final del BMR usa kg.
            weight = weight / 2.20462
        w_min, w_max = _BIO_RANGES["weight_kg"]
        if weight is None or not (w_min <= weight <= w_max):
            errors.append({
                "field": "weight",
                "value": weight_raw,
                "accepted_range": [int(w_min), int(w_max)],
                "unit": f"kg (input recibido en {weight_unit})",
            })
        else:
            # [P0-FORM-4] Heurística defensiva de unidad invertida.
            # Si el peso original es "alto" (≥150 — cifra que en lb sería un
            # adulto promedio, en kg sería pesado) Y el peso post-conversión
            # cae en una zona implausiblemente baja (≤35 kg), hay sospecha de
            # que el usuario seleccionó "lb" pero ingresó la cifra en kg, o
            # viceversa. Solo loguear: el rango ya pasó (>30) así que la
            # cifra es físicamente posible (atleta extremo), pero la
            # combinación amerita observabilidad para detectar drift de UX.
            weight_raw_num = _coerce_numeric(weight_raw, kind="float")
            if (
                weight_raw_num is not None
                and weight_raw_num >= 150
                and weight <= 35
            ):
                logger.warning(
                    f"[P0-FORM-4] Sospecha de unidad invertida: weight={weight_raw_num} "
                    f"con weightUnit='{weight_unit}' → {weight:.1f} kg post-conversión. "
                    f"Validar con el usuario que la unidad es correcta. "
                    f"user_id={data.get('user_id')}, session_id={data.get('session_id')}"
                )

    # --- bodyFat (opcional) ---
    bf_raw = data.get("bodyFat")
    if bf_raw not in (None, ""):
        bf = _coerce_numeric(bf_raw, kind="float")
        bf_min, bf_max = _BIO_RANGES["bodyFat"]
        if bf is None or not (bf_min <= bf <= bf_max):
            errors.append({
                "field": "bodyFat",
                "value": bf_raw,
                "accepted_range": [bf_min, bf_max],
                "unit": "%",
            })

    # --- householdSize (P1-FORM-12) ---
    # Required (en `_REQUIRED_FORM_FIELDS`); presencia ya validada arriba.
    # Aquí validamos rango + tipo (int parseable). Sin esto, `householdSize=999`
    # o `="abc"` pasaban al `shopping_calculator` que multiplicaba cantidades
    # x999 → lista de compras absurda. El frontend ofrece chips 1..6, cap en
    # 12 cubre familias extendidas legítimas sin abrir vector de abuso.
    household_raw = data.get("householdSize")
    if household_raw is not None and household_raw != "":
        household = _coerce_numeric(household_raw, kind="int")
        h_min, h_max = _BIO_RANGES["household"]
        if household is None or not (h_min <= household <= h_max):
            errors.append({
                "field": "householdSize",
                "value": household_raw,
                "accepted_range": [h_min, h_max],
                "unit": "personas",
            })

    # --- dietType (opcional, dos niveles: canónico vs legacy) ---
    # [P1-FORM-8] Ver bloque del comentario de `_DIET_TYPE_ENUM` arriba para
    # el rationale completo. Resumen del flujo:
    #   - Ausente / vacío → no-op (default `balanced` downstream).
    #   - Canónico (balanced|vegetarian|vegan) → pasa sin log.
    #   - Legacy aceptado (omnivora, vegana, pescetariano, etc.) → pasa con
    #     `logger.info` para que operadores puedan medir tráfico legacy y
    #     decidir cuándo migrar la DB.
    #   - Cualquier otro valor → 422 con el set canónico como "accepted_range"
    #     (no exponemos legacy como "aceptado" en la respuesta del cliente —
    #     queremos que clientes nuevos usen los canónicos).
    diet_raw = data.get("dietType")
    if diet_raw not in (None, ""):
        diet_norm = (
            str(diet_raw).strip().lower().replace("í", "i").replace("ñ", "n")
            if isinstance(diet_raw, str) else None
        )
        if diet_norm in _DIET_TYPE_CANONICAL:
            pass  # path normal — sin log
        elif diet_norm in _DIET_TYPE_LEGACY_ACCEPTED:
            logger.info(
                f"[P1-FORM-8] dietType legacy aceptado: raw={diet_raw!r} "
                f"normalizado={diet_norm!r}. user_id={data.get('user_id')}, "
                f"session_id={data.get('session_id')}. Considerar migrar DB "
                f"a valor canónico (balanced|vegetarian|vegan)."
            )
        else:
            errors.append({
                "field": "dietType",
                "value": diet_raw,
                "accepted_range": sorted(_DIET_TYPE_CANONICAL),
                "unit": "string (enum: balanced|vegetarian|vegan)",
            })

    # --- activityLevel (required, enum estricto) ---
    # [P0-FORM-5] Ver `_ACTIVITY_LEVEL_ENUM` arriba para el rationale completo.
    # `_validate_form_data_min` ya garantizó presencia (está en `_REQUIRED_FORM_FIELDS`),
    # acá validamos que el VALOR esté en el enum. Sin este check, valor desconocido
    # → `nutrition_calculator.calculate_tdee` defaultea silenciosamente a `moderate`
    # (×1.55) → BMR/TDEE/macros erróneos sin telemetría. Misma clase de bug que
    # P0-FORM-4 corrigió para `weightUnit`.
    activity_raw = data.get("activityLevel")
    if activity_raw not in (None, ""):
        activity_norm = (
            str(activity_raw).strip().lower()
            if isinstance(activity_raw, str) else None
        )
        if activity_norm not in _ACTIVITY_LEVEL_ENUM:
            errors.append({
                "field": "activityLevel",
                "value": activity_raw,
                "accepted_range": sorted(_ACTIVITY_LEVEL_ENUM),
                "unit": "string (enum: sedentary|light|moderate|active|athlete)",
            })

    # --- mainGoal (required, enum estricto) ---
    # [P0-FORM-5] Análogo a activityLevel. Sin validación, `apply_goal_adjustment`
    # defaultea a `maintenance` (déficit/superávit=0%) para CUALQUIER valor
    # desconocido → un usuario que pidió "lose_fat" pero envió "perder peso"
    # (typo legacy) recibía un plan de mantenimiento sin disparar warning.
    goal_raw = data.get("mainGoal")
    if goal_raw not in (None, ""):
        goal_norm = (
            str(goal_raw).strip().lower()
            if isinstance(goal_raw, str) else None
        )
        if goal_norm not in _MAIN_GOAL_ENUM:
            errors.append({
                "field": "mainGoal",
                "value": goal_raw,
                "accepted_range": sorted(_MAIN_GOAL_ENUM),
                "unit": "string (enum: lose_fat|gain_muscle|maintenance|performance)",
            })

    # --- Enums no-críticos (P1-FORM-10) ---
    # scheduleType / cookingTime / budget / groceryDuration / sleepHours / stressLevel.
    # Defense-in-depth contra clientes legacy o no-oficiales. Ausentes pasan
    # como no-op (downstream tiene defaults seguros). Presentes con valor fuera
    # del enum → 422 accionable.
    for field, enum_set, normalize_lower, accepted_label in _NON_CRITICAL_ENUM_VALIDATIONS:
        raw = data.get(field)
        if raw in (None, ""):
            continue
        if not isinstance(raw, str):
            errors.append({
                "field": field,
                "value": raw,
                "accepted_range": sorted(enum_set),
                "unit": f"string (enum: {accepted_label})",
            })
            continue
        candidate = raw.strip().lower() if normalize_lower else raw.strip()
        if candidate not in enum_set:
            errors.append({
                "field": field,
                "value": raw,
                "accepted_range": sorted(enum_set),
                "unit": f"string (enum: {accepted_label})",
            })

    # --- selectedSupplements (P1-FORM-11) ---
    # Array de strings. Cada elemento DEBE estar en `_SUPPLEMENT_ENUM`. Vector
    # de prompt-injection si no se valida: ver bloque de _SUPPLEMENT_ENUM.
    # Solo se valida si `includeSupplements=True` (el array sin opt-in es
    # inerte downstream, ver `assemble_plan_node` línea ~4711). Array vacío
    # con opt-in → permite que la IA sugiera (comportamiento documentado en
    # `build_supplements_context`).
    if data.get("includeSupplements"):
        supps_raw = data.get("selectedSupplements")
        if supps_raw is not None:
            if not isinstance(supps_raw, list):
                errors.append({
                    "field": "selectedSupplements",
                    "value": supps_raw,
                    "accepted_range": sorted(_SUPPLEMENT_ENUM),
                    "unit": "list of strings (enum)",
                })
            else:
                invalid_supps = [
                    s for s in supps_raw
                    if not isinstance(s, str) or s not in _SUPPLEMENT_ENUM
                ]
                if invalid_supps:
                    errors.append({
                        "field": "selectedSupplements",
                        "value": invalid_supps,
                        "accepted_range": sorted(_SUPPLEMENT_ENUM),
                        "unit": "list of strings (enum)",
                    })

    return (len(errors) == 0, errors)


def _validate_total_days(data: dict) -> tuple[bool, dict | None]:
    """[P1-ORQ-8] Valida que `totalDays` sea un entero en rango razonable.

    ANTES: el router hacía `int(data.get("totalDays", 3))` directo. Para
    valores no-numéricos (ej. `"abc"`, lista, dict) lanzaba `ValueError` →
    capturado por el try/except envolvente y devuelto como 500 opaco. Para
    valores numéricos pero inválidos (`-5`, `0`, `9999`), pasaba al pipeline
    donde el orquestador corregía silenciosamente a `PLAN_CHUNK_SIZE` sin
    logging — enmascarando bugs del frontend o de clientes no oficiales.

    AHORA: rechazo explícito con 422 en API boundary. Los valores normalmente
    enviados son 7/15/30 (weekly/biweekly/monthly del wizard); cualquier
    valor fuera de [1, 30] es señal de bug o cliente fraudulento. El cap de
    30 es la duración máxima documentada del producto (plan mensual). Si en
    el futuro se introducen planes >30 días, subir este cap acompañado del
    bump correspondiente en `cron_tasks._chunked_generation`.

    Args:
        data: payload crudo del request.

    Returns:
        `(is_valid, error_or_none)`. Si invalid, el caller debe rechazar
        con 422 e incluir el dict de error en `detail.errors`.
    """
    raw = data.get("totalDays", 3)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return False, {
            "field": "totalDays",
            "value": raw,
            "reason": "no es un entero parseable",
            "accepted_range": [1, 30],
        }
    if n < 1 or n > 30:
        return False, {
            "field": "totalDays",
            "value": n,
            "reason": "fuera de rango",
            "accepted_range": [1, 30],
        }
    return True, None


def _log_dislikes_signal_loss(data: dict) -> None:
    """[P0-FORM-4] Observabilidad: detecta payloads donde `dislikes=[]` Y
    `otherDislikes=''` para usuarios autenticados (no guests).

    El frontend (`InteractiveQuestions.jsx` QDislikes) bloquea el botón
    "Siguiente" hasta que el usuario seleccione al menos un chip, marque
    "Ninguno", o llene el free-text. Si llega a este endpoint un payload con
    AMBOS campos vacíos para un user_id real, es señal de:
      - cliente no oficial (móvil legacy, scraper, integración rota), o
      - bypass del gate (manipulación de localStorage), o
      - bug de hidratación que vacía ambos campos sin que el usuario interactúe.
    Cualquier caso es un FALSO NEGATIVO de "no rechaza nada" — el backend
    procesa `dislikes=[]` como no-op (filtros vacíos) → ingredientes
    culturalmente sensibles (cilantro, hígado, etc.) cuelan al plan.
    NO bloqueamos (no rompemos clientes legacy honestos) pero emitimos warning
    para alertar a operadores. Si la métrica crece, considerar agregar
    `dislikes` a `_REQUIRED_FORM_FIELDS` (gate estricto). Guests se omiten:
    no tenemos contexto histórico para correlacionar y la categoría tiene
    ratio señal/ruido baja para anónimos.
    """
    user_id = data.get("user_id")
    if not user_id or user_id == "guest":
        return
    raw_dislikes = data.get("dislikes")
    has_chips = isinstance(raw_dislikes, list) and len(raw_dislikes) > 0
    raw_other = data.get("otherDislikes")
    has_other = isinstance(raw_other, str) and raw_other.strip() != ""
    if has_chips or has_other:
        return
    logger.warning(
        "🟠 [P0-FORM-4] dislikes signal-loss detectado: user_id=%s envió "
        "dislikes=[] y otherDislikes='' simultáneamente. Posible cliente "
        "no oficial, bypass del gate frontend, o bug de hidratación. "
        "El plan se generará SIN señal de rechazos — ingredientes "
        "culturalmente sensibles (cilantro, hígado, etc.) podrían colar.",
        user_id,
    )


def _run_pantry_validation_for_initial_chunk(
    *,
    result: dict,
    pipeline_data: dict,
    history: list,
    taste_profile: str,
    memory_ctx: str,
    background_tasks: BackgroundTasks,
    actual_user_id: Optional[str],
    pantry_ingredients: list,
    transport_label: str = "P0-1",
) -> dict:
    """[P0-1/P0-2] Valida el primer chunk contra pantry y aplica los flags
    `_initial_chunk_pantry_*` cuando degrada.

    Antes este bloque (~40 líneas) estaba duplicado entre el endpoint sync y
    el SSE; las dos copias divergieron en mensajes de log y en el manejo
    de excepciones, lo que dificultó debugging cuando la validación fallaba
    en uno de los dos paths. Centralizado: el `transport_label` se inyecta
    al log para distinguir el path en producción (`P0-1` para sync,
    `P0-2 SSE` para streaming) sin duplicar lógica.

    Si `pantry_ingredients` está vacío, retorna `result` intacto (corto-
    circuito sano: sin pantry no hay nada contra qué validar).
    """
    if not pantry_ingredients:
        return result
    try:
        from cron_tasks import _validate_and_retry_initial_chunk_against_pantry
        result, _initial_audit = _validate_and_retry_initial_chunk_against_pantry(
            pipeline_data=pipeline_data,
            history=history,
            taste_profile=taste_profile,
            memory_context=memory_ctx,
            background_tasks=background_tasks,
            pantry_ingredients=pantry_ingredients,
            initial_result=result,
            user_id=actual_user_id,
        )
        _user_label = actual_user_id or "guest"
        if _initial_audit.get("degraded"):
            result["_initial_chunk_pantry_degraded"] = True
            result["_initial_chunk_pantry_violation"] = (
                (_initial_audit.get("last_violation") or "")[:500]
            )
            logger.error(
                f"❌ [{transport_label}] Primer chunk para user={_user_label} "
                f"degradado tras {_initial_audit.get('attempts')} intento(s) "
                f"(mode={_initial_audit.get('mode')}). Plan se entrega con flag de aviso."
            )
        else:
            logger.info(
                f"✅ [{transport_label}] Primer chunk validado contra nevera para "
                f"user={_user_label} en {_initial_audit.get('attempts')} intento(s) "
                f"(mode={_initial_audit.get('mode')})."
            )
    except Exception as _err:
        # Best-effort: si la validación misma falla, no rompemos el flujo.
        # El usuario ya esperó el pipeline completo; logueamos error visible
        # para alerting y devolvemos el result original.
        logger.error(
            f"❌ [{transport_label}] Excepción inesperada en validación inicial pantry "
            f"user={actual_user_id or 'guest'}: {_err}. Continuando sin validar."
        )
    return result


def _seed_chunk1_learning(
    plan_id,
    result: dict,
    rejected_meal_names: list,
    *,
    context_label: str,
) -> None:
    """[P0-α/P0-3] Seed `_last_chunk_learning` con métricas reales del chunk 1
    para que el chunk 2 arranque con datos accionables (anti-repetición de
    bases proteicas, detección de violaciones, etc.).

    Antes esta lógica de ~80 líneas estaba duplicada entre sync y SSE — los
    mensajes de log divergieron (`[P0-3/SEED-FAILED]` vs
    `[P0-3 SSE/SEED-FAILED]`) y cualquier fix del retry/verify había que
    aplicarlo dos veces. El `context_label` (e.g., `seed_chunk1_sync`,
    `seed_chunk1_sse`) se propaga a `persist_legacy_learning_to_plan_data`
    para distinguir el path en `pipeline_metrics`.

    Retry + read-back verification: si la persistencia falla silenciosamente,
    el chunk 2 nace con dict vacío y se rompe la cadena de aprendizaje.
    Reintentamos hasta 2 veces y validamos por consulta de read-back.
    """
    try:
        from db_core import execute_sql_query
        from cron_tasks import _calculate_learning_metrics, persist_legacy_learning_to_plan_data

        chunk1_days = result.get("days", [])
        _c1_metrics = _calculate_learning_metrics(
            new_days=chunk1_days,
            prior_meals=[],
            prior_days=[],
            rejected_names=rejected_meal_names,
            allergy_keywords=[],
            fatigued_ingredients=[],
        )
        week1_lesson = {
            'repeat_pct': _c1_metrics.get('learning_repeat_pct', 0),
            'ingredient_base_repeat_pct': _c1_metrics.get('ingredient_base_repeat_pct', 0),
            'rejection_violations': _c1_metrics.get('rejection_violations', 0),
            'allergy_violations': _c1_metrics.get('allergy_violations', 0),
            'fatigued_violations': _c1_metrics.get('fatigued_violations', 0),
            'repeated_bases': _c1_metrics.get('sample_repeated_bases', []),
            'repeated_meal_names': _c1_metrics.get('sample_repeats', []),
            'rejected_meals_that_reappeared': _c1_metrics.get('sample_rejection_hits', []),
            'allergy_hits': _c1_metrics.get('sample_allergy_hits', []),
            'chunk': 1,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metrics_unavailable': False,
            'low_confidence': False,
        }
        _seed_attempts = 0
        _seed_persisted = False
        _seed_last_err = None
        while _seed_attempts < 2 and not _seed_persisted:
            _seed_attempts += 1
            try:
                ok = persist_legacy_learning_to_plan_data(
                    plan_id, week1_lesson,
                    recent_chunk_lessons=[week1_lesson],
                    context=context_label,
                )
                if not ok:
                    _seed_last_err = "persist_helper_returned_false"
                    continue
                _verify = execute_sql_query(
                    "SELECT plan_data->'_last_chunk_learning'->>'chunk' AS chunk "
                    "FROM meal_plans WHERE id = %s",
                    (plan_id,),
                    fetch_one=True,
                )
                if _verify and str(_verify.get("chunk")) == "1":
                    _seed_persisted = True
                else:
                    _seed_last_err = f"verification_failed(verify={_verify})"
            except Exception as _seed_attempt_err:
                _seed_last_err = repr(_seed_attempt_err)

        if _seed_persisted:
            logger.info(
                f"🌱 [P0-α] Seed _last_chunk_learning con métricas REALES para plan {plan_id} "
                f"(repeat_pct={_c1_metrics.get('learning_repeat_pct', 0)}% "
                f"base_repeat={_c1_metrics.get('ingredient_base_repeat_pct', 0)}% "
                f"rej_viol={_c1_metrics.get('rejection_violations', 0)} "
                f"attempts={_seed_attempts} ctx={context_label})"
            )
        else:
            logger.error(
                f"❌ [P0-3/SEED-FAILED] No se pudo persistir _last_chunk_learning "
                f"para plan {plan_id} tras {_seed_attempts} intento(s) "
                f"(ctx={context_label}). Último error: {_seed_last_err}. "
                f"El worker intentará auto-recovery desde plan_chunk_queue."
            )
    except Exception as seed_err:
        logger.error(
            f"❌ [P0-3/SEED-FAILED] Error sembrando _last_chunk_learning "
            f"(ctx={context_label}): {seed_err}"
        )


def _enqueue_remaining_chunks(
    actual_user_id: str,
    plan_id,
    result: dict,
    *,
    data: dict,
    taste_profile: str,
    memory_ctx: str,
    total_days_requested: int,
    plan_start_date: str,
    tz_offset_mins: int,
) -> None:
    """[P1-1] Encola los chunks 2..N en background. Antes el snapshot del
    path sync NO incluía `previous_meals` mientras el path SSE sí — divergencia
    silenciosa que reducía la efectividad del anti-repetición de chunk 2 en
    planes generados por el endpoint sync. Ahora `previous_meals` siempre se
    deriva de `result` y se incluye independientemente del transporte.

    [P1-12] `plan_start_date` y `tz_offset_mins` se reciben como params
    EXPLÍCITOS, no leídos de `data`. Antes los handlers mutaban
    `data["_plan_start_date"]` y `data["tz_offset_minutes"]` para que llegaran
    aquí — duplicación silenciosa que generaba dos fuentes de verdad para los
    mismos campos (`pipeline_data` y `data`). Si alguien refactorizaba el
    pipeline data sin tocar `data`, los chunks recibirían valores stale del
    payload original. Ahora `data` es el payload del cliente sin tocar; los
    valores derivados se inyectan en el snapshot vía params explícitos.
    """
    from cron_tasks import _enqueue_plan_chunk
    from db_core import execute_sql_query

    week1_meals = [
        m["name"] for d in result.get("days", [])
        for m in d.get("meals", []) if m.get("name")
    ]
    # [P1-12] form_data del snapshot = payload del cliente + valores derivados
    # (start_date, tz_offset). El chunk worker lee `form_data._plan_start_date`
    # vía cron_tasks.py:5019 y otros sitios; sin esto el worker caería al
    # fallback de TZ.
    snapshot = {
        "form_data": {
            **data,
            "_plan_start_date": plan_start_date,
            "tz_offset_minutes": tz_offset_mins,
        },
        "taste_profile": taste_profile,
        "memory_context": memory_ctx,
        "previous_meals": week1_meals,
        "totalDays": total_days_requested,
    }
    prior_plan = execute_sql_query(
        "SELECT plan_data FROM meal_plans WHERE user_id = %s "
        "ORDER BY created_at DESC OFFSET 1 LIMIT 1",
        (actual_user_id,), fetch_one=True
    )
    inherited = None
    if prior_plan:
        pd = prior_plan.get("plan_data", {})
        _hist = pd.get("_lifetime_lessons_history", [])
        _summ = pd.get("_lifetime_lessons_summary", {})
        if _hist or _summ:
            inherited = {"history": _hist, "summary": _summ}

    chunks = split_with_absorb(total_days_requested, PLAN_CHUNK_SIZE)
    offset = chunks[0]
    for wk, count in enumerate(chunks[1:], start=2):
        if count > 0:
            try:
                chunk_snapshot = dict(snapshot)
                if inherited:
                    chunk_snapshot["_inherited_lifetime_lessons"] = inherited
                _enqueue_plan_chunk(
                    actual_user_id, plan_id, wk, offset, count,
                    chunk_snapshot, chunk_kind="initial_plan",
                )
            except Exception as chunk_err:
                logger.warning(
                    f"⚠️ [CHUNK] Error encolando chunk semana {wk} "
                    f"para {actual_user_id}: {chunk_err}"
                )
        offset += count
    logger.info(
        f"🚀 [CHUNK] Plan {plan_id} creado con semana 1. "
        f"{len(chunks) - 1} chunks encolados en background."
    )


def _postprocess_pipeline_result(
    *,
    result: dict,
    actual_user_id: Optional[str],
    session_id: Optional[str],
    data: dict,
    taste_profile: str,
    memory_ctx: str,
    rejected_meal_names: list,
    total_days_requested: int,
    use_chunking: bool,
    background_tasks: BackgroundTasks,
    plan_start_date: str,
    tz_offset_mins: int,
    transport_label: str = "sync",
) -> dict:
    """[P1-1] Post-procesa el resultado del pipeline tras la validación pantry:
    persiste perfil/mensajes/audit, expurga campos internos del payload,
    encola chunking + seeding + emergency backup.

    Antes este bloque (~200 líneas) estaba duplicado en `api_analyze` y
    `api_analyze_stream` y divergió varias veces:
      - SSE no incluía `previous_meals` en el chunk snapshot, sync sí
      - Mensajes de log diferentes ([P0-3/SEED-FAILED] vs [P0-3 SSE/SEED-FAILED])
      - SSE usaba `threading.Thread(daemon=True)` para persistencia
        (rompía bajo SIGTERM), sync usaba BackgroundTasks
      - SSE corría sus DB writes inline en el coroutine → bloqueaba el loop
      - `summarize_and_prune` divergía entre BG tasks y daemon thread
    Centralizado, todas las decisiones se toman aquí en una sola pasada.

    Es SÍNCRONA — los endpoints async deben llamarla vía `asyncio.to_thread`
    para no bloquear el event loop con los DB writes auxiliares.

    [P1-12] `plan_start_date` y `tz_offset_mins` son params EXPLÍCITOS, no
    leídos de `data`. Antes el handler mutaba `data["_plan_start_date"]` y
    `data["tz_offset_minutes"]` para que llegaran al snapshot del chunk + al
    health_profile — duplicación silenciosa con `pipeline_data` que generaba
    riesgo de drift. Ahora `data` queda inmutable (representación fiel del
    payload original); los valores derivados se inyectan vía params.

    [P0-FIX-SEED] `transport_label` ("sync" | "sse") forma el `context_label`
    final que se pasa a `_seed_chunk1_learning` → `persist_legacy_learning_to_plan_data`.
    El whitelist de contextos legacy aceptados vive en
    `cron_tasks.P0_3_LEGACY_LEARNING_CONTEXTS = ('seed_chunk1_sync',
    'seed_chunk1_sse', 'rebuild_from_queue', 'synthesis_from_days')`. Antes,
    este helper hardcodeaba `context_label="seed_chunk1"` (sin sufijo) y la
    persistencia fallaba silenciosamente con `persist_helper_returned_false` →
    `_last_chunk_learning` quedaba sin sembrar → chunk 2 nacía ciego sin
    métricas reales del chunk 1, rompiendo la cadena de aprendizaje
    inter-chunk diseñada por P0-α/P0-3. Default `"sync"` por compat
    histórica; los call sites deben pasar el label explícito de su transporte.
    """
    # 1. Health profile (DB write auxiliar — fallar no rompe la respuesta).
    # [P1-12] `tz_offset_mins` se persiste como `tz_offset_minutes` en
    # health_profile para que `_resolve_request_tz_offset` (linea ~53) pueda
    # usarlo como fallback cuando un cliente legacy omite `tzOffset` en futuros
    # requests. Inyectado explícitamente — antes venía vía `data["tz_offset_minutes"]`
    # (mutación que P1-12 elimina).
    if actual_user_id:
        hp_data = {k: v for k, v in data.items() if k not in ('session_id', 'user_id')}
        hp_data["tz_offset_minutes"] = tz_offset_mins
        if hp_data:
            try:
                update_user_health_profile(actual_user_id, hp_data)
                logger.info(f"💾 health_profile guardado para user {actual_user_id}")
            except Exception as _hp_err:
                logger.warning(f"⚠️ Error guardando health_profile: {_hp_err}")

    # 2. Chat messages + summarize background (consistente con sync — antes el
    # SSE corría summarize_and_prune en threading.Thread(daemon=True) que moría
    # bajo SIGTERM). Migrado a BackgroundTasks como side-effect del refactor
    # (resuelve P1-8).
    if session_id:
        goal = data.get('mainGoal', 'Desconocido')
        try:
            save_message(session_id, "user", f"Generar plan para mi objetivo: {goal}")
            save_message(
                session_id, "model",
                "¡Aquí tienes tu estrategia nutricional personalizada generada analíticamente!",
            )
            background_tasks.add_task(summarize_and_prune, session_id)
        except Exception as _msg_err:
            logger.warning(f"⚠️ Error registrando mensajes de chat: {_msg_err}")

    # 3. API usage audit
    if actual_user_id:
        try:
            log_api_usage(actual_user_id, "gemini_analyze")
        except Exception as _audit_err:
            logger.warning(f"⚠️ Error registrando log_api_usage: {_audit_err}")

    # 4. Expurgar campos internos antes de devolver al cliente. Estos campos
    # son útiles para el pipeline interno (attribution tracker, embedding
    # storage) pero el frontend no debe verlos — riesgo de filtración de
    # internals + tamaño innecesario en el payload.
    selected_techniques = result.pop("_selected_techniques", None)
    result.pop("_profile_embedding", None)
    result.pop("_active_learning_signals", None)

    # 5. Persistencia: chunking (planes largos) o save simple (tier gratis)
    if use_chunking:
        plan_id = save_partial_plan_get_id(
            actual_user_id, result, selected_techniques, total_days_requested,
        )
        if plan_id:
            _seed_chunk1_learning(
                plan_id, result, rejected_meal_names,
                context_label=f"seed_chunk1_{transport_label}",
            )
            _enqueue_remaining_chunks(
                actual_user_id, plan_id, result,
                data=data, taste_profile=taste_profile,
                memory_ctx=memory_ctx,
                total_days_requested=total_days_requested,
                plan_start_date=plan_start_date,
                tz_offset_mins=tz_offset_mins,
            )
        if actual_user_id:
            from cron_tasks import _seed_emergency_backup_if_empty
            background_tasks.add_task(
                _seed_emergency_backup_if_empty,
                actual_user_id, result.get("days", []),
            )
        result["generation_status"] = "partial"
        result["total_days_requested"] = total_days_requested
        if plan_id:
            result["id"] = plan_id
    elif actual_user_id:
        # [P0-5] Persistencia vía BackgroundTasks (no daemon thread): sobrevive
        # SIGTERM con el grace period del worker.
        background_tasks.add_task(
            _save_plan_and_track_background,
            actual_user_id, result, selected_techniques,
        )
        from cron_tasks import _seed_emergency_backup_if_empty
        background_tasks.add_task(
            _seed_emergency_backup_if_empty,
            actual_user_id, result.get("days", []),
        )

    return result


def _attach_pantry_degraded_response_meta(response: Optional[Response], plan_data: dict) -> dict:
    """[P0-2] Calcula el resumen de degradación de pantry, adjunta los headers HTTP
    `X-Pantry-Degraded` / `X-Pantry-Degraded-Days` cuando aplica, y devuelve el dict
    resumen para inclusión en el body de la respuesta.

    El frontend puede leer la señal de DOS formas:
      1. Vía header `X-Pantry-Degraded: true` — útil para banners interceptados a
         nivel de fetch wrapper sin parsear JSON.
      2. Vía campo `_pantry_degraded_summary` dentro del body — útil para mostrar
         badges per-día y reasons específicos.

    Args:
        response: objeto FastAPI Response inyectado en el handler (None en path
            de fallback / tests directos).
        plan_data: dict con la estructura del plan. Puede contener:
            - `_initial_chunk_pantry_degraded` (P0-1)
            - `days[i]._pantry_degraded` (P0-2)
            - `_current_mode == "flexible"` (persistido por _activate_flexible_mode).

    Returns:
        El dict resumen (siempre devuelto, aunque el header no se haya adjuntado).
    """
    try:
        from cron_tasks import compute_pantry_degraded_summary
        summary = compute_pantry_degraded_summary(plan_data or {})
    except Exception as _summary_err:
        logger.debug(f"[P0-2] No se pudo computar pantry summary: {_summary_err}")
        summary = {
            "degraded": False,
            "degraded_days": [],
            "reasons": [],
            "initial_chunk_degraded": False,
            "current_mode": None,
        }
    if response is not None and summary.get("degraded"):
        try:
            response.headers["X-Pantry-Degraded"] = "true"
            if summary.get("degraded_days"):
                response.headers["X-Pantry-Degraded-Days"] = ",".join(
                    str(d) for d in summary["degraded_days"]
                )
            if summary.get("reasons"):
                # Headers HTTP no permiten coma en valores arbitrarios sin escape;
                # usamos pipe como separador (compatible con HTTP/1.1).
                response.headers["X-Pantry-Degraded-Reasons"] = "|".join(
                    summary["reasons"]
                )
        except Exception as _h_err:
            logger.debug(f"[P0-2] No se pudo setear headers pantry degraded: {_h_err}")
    return summary


# ─── TEMPORARY DEBUG ENDPOINT (REMOVE AFTER DIAGNOSIS) ───
@router.get("/debug-scaling/{user_id}")
def debug_scaling(user_id: str):
    """Temporary: compare shopping list output for household sizes 1-6."""
    from shopping_calculator import get_shopping_list_delta
    from db_plans import get_latest_meal_plan_with_id as _get_plan
    
    plan_record = _get_plan(user_id)
    
    # Fallback: try to find ANY recent plan if user_id yields nothing
    if not plan_record:
        try:
            from db_core import execute_sql_query
            row = execute_sql_query(
                "SELECT id, user_id, plan_data FROM meal_plans ORDER BY created_at DESC LIMIT 1",
                fetch_one=True
            )
            if row:
                plan_record = row
                user_id = row.get("user_id", user_id)
            else:
                return {"error": f"No plans exist in database at all"}
        except Exception as e:
            return {"error": f"No plan found for {user_id} and fallback failed: {e}"}
    
    if not plan_record:
        return {"error": f"No plan found for {user_id}"}
    
    plan_data = plan_record["plan_data"]
    days = plan_data.get("days", [])
    num_days = len(days)
    
    KEYWORDS = ['pechuga', 'pavo', 'yogurt', 'lechosa', 'aguacate', 'arroz', 'pollo', 'cebolla', 'tomate', 'melón', 'melon']
    
    comparison = {}
    for h in [1, 2, 3, 4, 5, 6]:
        scaled = get_shopping_list_delta(user_id, plan_data, is_new_plan=True, structured=True, multiplier=float(h))
        row = {}
        for item in scaled:
            name = item.get("name", "")
            if any(kw in name.lower() for kw in KEYWORDS):
                row[name] = {
                    "display_qty": item.get("display_qty"),
                    "market_qty": item.get("market_qty"),
                    "market_unit": item.get("market_unit"),
                }
        comparison[f"{h}_personas"] = row
    
    return {
        "found_user_id": user_id,
        "plan_id": plan_record.get("id"),
        "num_days_in_plan": num_days,
        "base_duration_scale": round(7.0 / max(1, num_days), 4),
        "comparison": comparison
    }
# ─── END TEMPORARY DEBUG ENDPOINT ───

from constants import PLAN_CHUNK_SIZE, split_with_absorb

def _user_has_profile(user_id: str) -> bool:
    """Devuelve True si user_id tiene fila en user_profiles. Auto-crea fila mínima si falta."""
    if not user_id or not supabase:
        return False
    try:
        res = supabase.table("user_profiles").select("id").eq("id", user_id).limit(1).execute()
        if res.data:
            return True
        # Usuario autenticado sin perfil → crear fila mínima para habilitar chunking y FK
        supabase.table("user_profiles").upsert({"id": user_id, "health_profile": {}}).execute()
        import logging as _log
        _log.getLogger(__name__).info(f"✅ [PROFILE] Fila mínima creada en user_profiles para {user_id}")
        return True
    except Exception as e:
        import logging as _log
        _log.getLogger(__name__).warning(f"⚠️ [PROFILE] No se pudo crear user_profiles para {user_id}: {e}")
        return False

def chunk_size_for_next_slot(days_since_creation: int, total_planned_days: int, base: int = 3):
    chunks = split_with_absorb(total_planned_days, base)
    consumed = 0
    for c in chunks:
        if days_since_creation < consumed + c:
            return c
        consumed += c
    return base

@router.post("/shift-plan")
def api_shift_plan(response: Response, data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    """
    On-demand endpoint to trigger an atomic shift + rolling window generation.
    Idempotent: if the plan is already up-to-date, does nothing.
    """
    from db_core import execute_sql_write, execute_sql_query
    from datetime import datetime, timezone, timedelta
    import copy, random, json
    
    try:
        user_id = data.get("user_id")
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Debes iniciar sesión."}
            
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado.")

        # P0-2 FIX: Get latest plan using FOR UPDATE to prevent race conditions with chunk workers doing blind overwrites
        from db_core import connection_pool
        from psycopg.rows import dict_row
        
        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(
                        "SELECT id, plan_data FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1 FOR UPDATE",
                        (user_id,)
                    )
                    plan_record = cursor.fetchone()
                    if not plan_record:
                        return {"success": False, "message": "No hay plan activo."}

                    plan_id = plan_record["id"]

                    # [P0-4] Advisory lock 'general' por meal_plan: serializa este shift
                    # contra el merge del worker (cron_tasks._chunk_worker T1+T2) que
                    # también adquiere el mismo lock antes de tocar plan_data. Sin esto,
                    # T2 del worker (que escribe plan_data POSTERIOR a su propio T1 con
                    # un dict en memoria) podía sobrescribir los cambios de /shift-plan
                    # ejecutados entre T1 y T2 — perdiendo el shift y la renumeración
                    # de days. El lock se libera al cerrar la transacción.
                    from db_plans import acquire_meal_plan_advisory_lock as _p04_acquire_lock
                    _p04_acquire_lock(cursor, plan_id, purpose="general")
                    plan_data = plan_record.get("plan_data", {})
                    days = plan_data.get("days", [])
                    total_planned_days = max(3, int(plan_data.get("total_days_requested", len(days))))
                    
                    if len(days) == 0:
                        return {"success": False, "message": "El plan está vacío."}

                    # Check if shift is needed
                    # [P1-1] Resolver tz con fallback al perfil cuando el frontend no
                    # envía tzOffset. Antes `data.get('tzOffset', 0)` colapsaba "no
                    # enviado" con "UTC explícito" y `days_since_creation` se calculaba
                    # en UTC para usuarios en TZ no-UTC con clientes que omiten el
                    # campo, produciendo off-by-one en chunk_size_for_next_slot.
                    _payload_tz = data.get('tzOffset')
                    tz_offset = _resolve_request_tz_offset(_payload_tz, user_id)
                    today = datetime.now(timezone.utc)
                    if _payload_tz is not None:
                        try:
                            # [P0-delta] Persistir TZ offset en health_profile silenciosamente
                            # SOLO cuando llegó del request (evitamos overwrite cuando
                            # estamos usando el valor que ya está en el perfil).
                            update_user_health_profile(user_id, {"tz_offset_minutes": int(tz_offset), "tzOffset": int(tz_offset)})
                        except Exception:
                            pass
                    if tz_offset:
                        try:
                            today -= timedelta(minutes=int(tz_offset))
                        except (ValueError, TypeError):
                            pass

                    # Parse grocery_start_date to find actual day index
                    start_date_str = plan_data.get("grocery_start_date")
                    if not start_date_str:
                        return {"success": False, "message": "Falta fecha de inicio."}
                        
                    from constants import safe_fromisoformat
                    try:
                        start_dt = safe_fromisoformat(start_date_str)
                        if start_dt.tzinfo is None:
                            start_dt = start_dt.replace(tzinfo=timezone.utc)
                        else:
                            start_dt = start_dt.astimezone(timezone.utc)
                        start_dt = start_dt - timedelta(minutes=int(tz_offset))
                        
                        # Remove time component
                        today_date = today.date()
                        start_date = start_dt.date()
                        days_since_creation = (today_date - start_date).days
                    except Exception as e:
                        return {"success": False, "message": f"Error parseando fecha: {e}"}

                    # Cuántos días del plan total quedan por vivir
                    days_remaining_in_plan = max(0, total_planned_days - days_since_creation)
                    
                    # Bloque dinámico P0-2: window_size depende de la posición en la distribución
                    window_size = chunk_size_for_next_slot(max(0, days_since_creation), total_planned_days, PLAN_CHUNK_SIZE)
                    
                    # La ventana necesita min(window_size, días restantes) días; si el plan expiró no necesita nada
                    window_needed = min(window_size, days_remaining_in_plan)

                    needs_shift = days_since_creation > 0
                    needs_fill_initial = len(days) < window_needed  # Para la guard de cortocircuito inicial

                    if not needs_shift and not needs_fill_initial:
                        # [P0-2] Resumen de pantry-degraded + headers para esta rama.
                        _p02_summary = _attach_pantry_degraded_response_meta(response, plan_data)
                        return {
                            "success": True,
                            "message": "Plan ya est\u00e1 al d\u00eda y completo.",
                            "plan_data": plan_data,
                            "_pantry_degraded_summary": _p02_summary,
                        }

                    logger.info(f"\ud83d\udd04 [API SHIFT] Shifting {days_since_creation} días. Plan total={total_planned_days}, restantes={days_remaining_in_plan}")

                    shifted_data = copy.deepcopy(plan_data)
                    shifted_days = shifted_data.get('days', [])

                    # 1. Atomic Shift (in-memory, saved at the end within transaction)
                    if needs_shift:
                        shift_amount = min(days_since_creation, len(shifted_days))
                        shifted_days = shifted_days[shift_amount:]

                    # 2. Update day names AND renumber days 1..N (requerido para continuidad del chunk worker)
                    dias_es = ["Lunes", "Martes", "Mi\u00e9rcoles", "Jueves", "Viernes", "S\u00e1bado", "Domingo"]
                    for i, day_obj in enumerate(shifted_days):
                        target_date = today + timedelta(days=i)
                        day_obj['day_name'] = dias_es[target_date.weekday()]
                        day_obj['day'] = i + 1  # Renumerar desde 1 para mantener secuencia 1..N

                    # 3. Rolling window: si el plan no ha expirado y la ventana actual tiene menos de window_size días,
                    #    y no hay ya un chunk de IA en camino, encolar generación IA real (aprendizaje continuo).
                    modified = needs_shift
                    is_partial = plan_data.get('generation_status') in ('partial', 'generating_next')
                    needs_fill = len(shifted_days) < window_needed and days_remaining_in_plan > 0

                    # [P0-4 FIX] Antes: disable_rolling_refill_for_active_7d bloqueaba TODO refill
                    # mientras el plan de 7d siguiera vivo. Eso dejaba huérfanos los planes donde
                    # el encolado síncrono inicial (chunk 2 de 4d) falló: 3 días generados y 4 vacíos
                    # sin recuperación automática. Ahora solo bloqueamos cuando hay chunks vivos
                    # en queue (estado normal); si no hay chunks pendientes y existe gap real, se
                    # permite refill para recuperar el hueco.
                    disable_rolling_refill_for_active_7d = False
                    if total_planned_days == 7 and days_remaining_in_plan > 0:
                        cursor.execute(
                            "SELECT COUNT(*) AS cnt FROM plan_chunk_queue "
                            "WHERE meal_plan_id = %s AND status IN ('pending', 'processing', 'stale')",
                            (plan_id,)
                        )
                        chunks_in_flight = int(((cursor.fetchone() or {}).get('cnt') or 0))
                        has_orphan_gap = chunks_in_flight == 0 and len(shifted_days) < total_planned_days
                        disable_rolling_refill_for_active_7d = not has_orphan_gap

                    if disable_rolling_refill_for_active_7d and needs_fill:
                        logger.info(
                            f"[P1-1] Plan de 7 días {plan_id}: rolling refill bloqueado durante vida útil "
                            f"(restantes={days_remaining_in_plan}, visibles={len(shifted_days)})."
                        )
                    elif total_planned_days == 7 and days_remaining_in_plan > 0 and needs_fill and not is_partial:
                        logger.warning(
                            f"[P0-4] Plan de 7 días {plan_id}: detectado gap huérfano "
                            f"(visibles={len(shifted_days)}/{total_planned_days}, sin chunks vivos). "
                            f"Habilitando rolling refill de recuperación."
                        )
                    elif total_planned_days in (7, 15, 30) and days_remaining_in_plan == 0 and not is_partial:
                        # [P0-1] Plan expirado: auto-renovar con señales de aprendizaje frescas.
                        # Genera un nuevo ciclo en el mismo plan_id, preservando historial.
                        try:
                            from cron_tasks import _enqueue_plan_chunk
                            cursor.execute(
                                "SELECT COUNT(*) AS cnt FROM plan_chunk_queue "
                                "WHERE meal_plan_id = %s AND status IN ('pending', 'processing', 'stale')",
                                (plan_id,)
                            )
                            if ((cursor.fetchone() or {}).get('cnt') or 0) == 0:
                                cursor.execute("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,))
                                profile_row = cursor.fetchone()
                                hp = (profile_row or {}).get("health_profile", {}) or {}
                                if hp:
                                    cursor.execute(
                                        "SELECT COALESCE(MAX(week_number), 0) AS max_week FROM plan_chunk_queue "
                                        "WHERE meal_plan_id = %s AND status <> 'cancelled'",
                                        (plan_id,)
                                    )
                                    next_week = int((cursor.fetchone() or {}).get('max_week') or 0) + 1
                                    previous_meals = [
                                        m.get('name', '') for d in shifted_days
                                        for m in d.get('meals', []) if m.get('name')
                                    ]
                                    current_offset = 0
                                    # [P0-1 FIX] Plan renovado: start = today para que el gate
                                    # de adherencia previa calcule ventanas correctas.
                                    renewal_plan_start_iso = today.isoformat()
                                    is_first_chunk = True
                                    
                                    from db_inventory import get_user_inventory_net
                                    live_inv = get_user_inventory_net(user_id)
                                    
                                    if live_inv is None:
                                        shifted_data['generation_status'] = 'expired_pending_pantry'
                                        shifted_data['pending_user_action'] = {
                                            "type": "pantry_required",
                                            "message": "Actualiza tu nevera para renovar tu plan"
                                        }
                                        shifted_days = []
                                        modified = True
                                        logger.warning(f"⚠️ [P0-1 RENEWAL] Plan {plan_id} pendiente de nevera.")
                                        try:
                                            import threading
                                            from utils_push import send_push_notification
                                            threading.Thread(
                                                target=send_push_notification,
                                                kwargs={
                                                    "user_id": user_id,
                                                    "title": "Renovación pausada",
                                                    "body": "Actualiza tu nevera para renovar tu plan.",
                                                    "url": "/dashboard"
                                                },
                                                daemon=True
                                            ).start()
                                        except Exception as e_push:
                                            logger.error(f"Error push renewal: {e_push}")
                                    else:
                                        for chunk_count in split_with_absorb(total_planned_days, PLAN_CHUNK_SIZE):
                                            snapshot = {
                                                "form_data": {
                                                    **hp,
                                                    "user_id": user_id,
                                                    "totalDays": chunk_count,
                                                    "_plan_start_date": renewal_plan_start_iso,
                                                    "current_pantry_ingredients": live_inv or [],
                                                    "_pantry_captured_at": today.isoformat(),
                                                },
                                                "taste_profile": "",
                                                "memory_context": "",
                                                "previous_meals": previous_meals,
                                                "totalDays": chunk_count,
                                                "_is_rolling_refill": True,
                                                "_is_weekly_renewal": True,
                                            }
                                            
                                            if is_first_chunk:
                                                _history = plan_data.get("_lifetime_lessons_history")
                                                _summary = plan_data.get("_lifetime_lessons_summary")
                                                if _history or _summary:
                                                    snapshot["_inherited_lifetime_lessons"] = {
                                                        "history": _history or [],
                                                        "summary": _summary or {}
                                                    }
                                                is_first_chunk = False
    
                                            _enqueue_plan_chunk(
                                                user_id, plan_id, next_week, current_offset,
                                                chunk_count, snapshot, chunk_kind="rolling_refill",
                                            )
                                            logger.info(
                                                f"🔄 [P0-1 RENEWAL] Chunk semana {next_week} encolado "
                                                f"(offset={current_offset}, count={chunk_count}) plan {plan_id}"
                                            )
                                            next_week += 1
                                            current_offset += chunk_count
                                        shifted_days = []
                                        shifted_data['grocery_start_date'] = today.isoformat()
                                        shifted_data['generation_status'] = 'generating_next'
                                        modified = True
                                        logger.info(f"🔄 [P0-1 RENEWAL] Plan semanal {plan_id} renovado.")
                                else:
                                    logger.warning(f"⚠️ [P0-1 RENEWAL] Sin health_profile para user {user_id}.")
                        except Exception as e:
                            logger.error(f"❌ [P0-1 RENEWAL] Error renovando plan semanal: {e}")
                    elif is_partial and needs_fill and total_planned_days > 7:
                        # [VISUAL CONTINUITY] Plan mensual/quincenal en estado partial:
                        # los chunks futuros ya están encolados, pero el usuario está viendo
                        # un gap porque el día pasó y el siguiente chunk aún no fue disparado.
                        # No re-encolamos (causaría días duplicados); aceleramos el siguiente
                        # chunk pendiente para que execute_after = NOW() y el cron lo tome
                        # en su próxima corrida (≤ 1 minuto).
                        try:
                            cursor.execute(
                                """
                                UPDATE plan_chunk_queue
                                SET execute_after = NOW(),
                                    updated_at = NOW()
                                WHERE id = (
                                    SELECT id FROM plan_chunk_queue
                                    WHERE meal_plan_id = %s
                                      AND status IN ('pending', 'stale')
                                      AND execute_after > NOW()
                                    ORDER BY week_number ASC
                                    LIMIT 1
                                )
                                RETURNING id, week_number
                                """,
                                (plan_id,)
                            )
                            accelerated = cursor.fetchone()
                            if accelerated:
                                logger.info(
                                    f"⚡ [VISUAL CONTINUITY] Chunk {accelerated['week_number']} "
                                    f"(id={accelerated['id']}) acelerado a NOW() para plan {plan_id} "
                                    f"(gap visible: {len(shifted_days)}/{window_needed} días)"
                                )
                        except Exception as e:
                            logger.error(f"❌ [VISUAL CONTINUITY] Error acelerando chunk pendiente: {e}")
                    elif not is_partial and needs_fill:
                        try:
                            from cron_tasks import _enqueue_plan_chunk
                            cursor.execute("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,))
                            profile_row = cursor.fetchone()
                            hp = (profile_row or {}).get("health_profile", {}) or {}
                            if hp:
                                previous_meals = [
                                    m.get('name', '') for d in shifted_days
                                    for m in d.get('meals', []) if m.get('name')
                                ]
                                days_offset = len(shifted_days)

                                # [P0-5] Catch-up: enqueue ALL missing days as sequential chunks,
                                # not just the current window. This ensures that after a long absence
                                # (e.g. 13 days missing) the entire gap gets filled, not just 3 days.
                                total_missing = days_remaining_in_plan - days_offset
                                catchup_chunks = split_with_absorb(total_missing, PLAN_CHUNK_SIZE) if total_missing > 0 else []

                                # [P1-5] Advisory lock por meal_plan para serializar catchups concurrentes.
                                # Sin esto, dos request paralelos al dashboard podían leer el mismo MAX(week_number)
                                # y duplicar chunks para la misma semana, dejando reservas dobles de inventario.
                                # pg_advisory_xact_lock se libera al cerrar la transacción.
                                # Antes este sitio invocaba la SQL directa con la hash function
                                # alternativa (no extended) y key `meal_plan_catchup_<id>`; ahora
                                # delega en el helper canónico (hashtextextended con seed 0, key
                                # namespaced por purpose) para que cualquier nuevo lock por
                                # meal_plan use el mismo espacio.
                                from db_plans import acquire_meal_plan_advisory_lock
                                acquire_meal_plan_advisory_lock(cursor, plan_id, purpose="catchup")

                                # [P1-3 / P1-5] El siguiente week_number debe seguir al máximo chunk
                                # no-cancelado (incluye completed/processing/pending/stale/failed) para
                                # no desalinearse si hubo failed/stale intermedios ni colisionar con un
                                # chunk que actualmente está en 'processing' (que aún no completó).
                                cursor.execute(
                                    """
                                    SELECT COALESCE(MAX(week_number), 1) AS max_week
                                    FROM plan_chunk_queue
                                    WHERE meal_plan_id = %s
                                      AND status <> 'cancelled'
                                    """,
                                    (plan_id,)
                                )
                                existing_chunks = cursor.fetchone()
                                next_week = int((existing_chunks or {}).get('max_week') or 1) + 1

                                current_offset = days_offset
                                chunks_enqueued = 0

                                # [P0-1 FIX] _plan_start_date vigente (post-shift) para que el
                                # gate de adherencia previa pueda calcular ventanas correctas.
                                catchup_plan_start_iso = (
                                    (start_dt + timedelta(days=days_since_creation)).isoformat()
                                    if needs_shift else start_date_str
                                )
                                _hist = plan_data.get("_lifetime_lessons_history", [])
                                _summ = plan_data.get("_lifetime_lessons_summary", {})
                                inherited = {"history": _hist, "summary": _summ} if (_hist or _summ) else None
                                is_first_catchup = True

                                for chunk_count in catchup_chunks:
                                    cursor.execute(
                                        """
                                        SELECT id, status, chunk_kind
                                        FROM plan_chunk_queue
                                        WHERE meal_plan_id = %s
                                          AND week_number = %s
                                          AND status IN ('pending', 'processing', 'stale', 'failed')
                                        ORDER BY updated_at DESC NULLS LAST, created_at DESC
                                        LIMIT 1 FOR UPDATE
                                        """,
                                        (plan_id, next_week)
                                    )
                                    conflicting_chunk = cursor.fetchone()

                                    if conflicting_chunk:
                                        logger.info(
                                            f"[P1-2] Saltando catch-up rolling refill para plan {plan_id}: "
                                            f"ya existe chunk objetivo week={next_week} "
                                            f"(id={conflicting_chunk['id']}, status={conflicting_chunk['status']}, "
                                            f"kind={conflicting_chunk.get('chunk_kind', 'unknown')})."
                                        )
                                    else:
                                        snapshot = {
                                            "form_data": {
                                                **hp,
                                                "user_id": user_id,
                                                "totalDays": chunk_count,
                                                "_plan_start_date": catchup_plan_start_iso,
                                            },
                                            "taste_profile": "",
                                            "memory_context": "",
                                            "previous_meals": previous_meals,
                                            "totalDays": chunk_count,
                                            "_is_rolling_refill": True,
                                        }
                                        if is_first_catchup and inherited:
                                            snapshot["_inherited_lifetime_lessons"] = inherited
                                            is_first_catchup = False
                                        _enqueue_plan_chunk(
                                            user_id,
                                            plan_id,
                                            next_week,
                                            current_offset,
                                            chunk_count,
                                            snapshot,
                                            chunk_kind="rolling_refill",
                                        )
                                        chunks_enqueued += 1
                                        logger.info(
                                            f"🤖 [ROLLING WINDOW] Chunk IA encolado "
                                            f"(week={next_week}, offset={current_offset}, count={chunk_count}) "
                                            f"para plan {plan_id}"
                                        )

                                    next_week += 1
                                    current_offset += chunk_count

                                if chunks_enqueued > 0:
                                    shifted_data['generation_status'] = 'generating_next'
                                    logger.info(
                                        f"🤖 [P0-5 CATCHUP] {chunks_enqueued} chunk(s) encolados "
                                        f"(total_missing={total_missing}, sizes={catchup_chunks}) "
                                        f"para plan {plan_id}"
                                    )
                                    modified = True
                            else:
                                logger.warning(f"⚠️ [ROLLING WINDOW] Sin health_profile para user {user_id}.")
                        except Exception as e:
                            logger.error(f"❌ [ROLLING WINDOW] Error encolando chunk IA: {e}")

                    # 4. Save rolling window updates
                    if modified:
                        shifted_data['days'] = shifted_days
                        new_plan_start_iso = None
                        if needs_shift and start_date_str:
                            new_start = start_dt + timedelta(days=days_since_creation)
                            new_plan_start_iso = new_start.isoformat()
                            shifted_data['grocery_start_date'] = new_plan_start_iso
                            
                            # [P0-C] Accumulate shift days
                            current_accum = int(shifted_data.get("_shift_days_accumulated", 0))
                            shifted_data["_shift_days_accumulated"] = current_accum + days_since_creation

                        # [P0-2] Sello CAS: timestamp que el worker compara para detectar
                        # si el plan fue modificado externamente durante el LLM call.
                        shifted_data['_plan_modified_at'] = datetime.now(timezone.utc).isoformat()

                        cursor.execute(
                            "UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s",
                            (json.dumps(shifted_data, ensure_ascii=False), plan_id)
                        )

                        # [P0-5 FIX] Sincronizar _plan_start_date en los snapshots de los chunks
                        # vivos del plan: tras un shift, grocery_start_date avanza pero los chunks
                        # ya encolados conservaban el origen original, desfasando los cálculos de
                        # _check_chunk_learning_ready (ventana de adherencia) y la asignación de
                        # day_name en Smart Shuffle. Ahora todos los chunks no terminales heredan
                        # el nuevo origen del plan en la misma transacción.
                        if needs_shift and new_plan_start_iso:
                            cursor.execute(
                                """
                                UPDATE plan_chunk_queue
                                SET pipeline_snapshot = jsonb_set(
                                        pipeline_snapshot,
                                        '{form_data,_plan_start_date}',
                                        %s::jsonb,
                                        true
                                    ),
                                    updated_at = NOW()
                                WHERE meal_plan_id = %s
                                  AND status IN ('pending', 'processing', 'stale', 'failed', 'pending_user_action')
                                """,
                                (json.dumps(new_plan_start_iso), plan_id)
                            )

                            # [P0-C] Shift execute_after for all pending future chunks
                            cursor.execute(
                                """
                                UPDATE plan_chunk_queue
                                SET execute_after = execute_after + (%s || ' days')::interval,
                                    updated_at = NOW()
                                WHERE meal_plan_id = %s
                                  AND status IN ('pending', 'stale')
                                  AND execute_after > NOW()
                                """,
                                (days_since_creation, plan_id)
                            )

        # [P0-2] Resumen de pantry-degraded + headers para la rama de shift exitoso.
        _p02_summary = _attach_pantry_degraded_response_meta(response, shifted_data)
        return {
            "success": True,
            "message": "Plan actualizado a la fecha.",
            "plan_data": shifted_data,
            "_pantry_degraded_summary": _p02_summary,
        }

    except Exception as e:
        logger.error(f"❌ [API SHIFT ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
def api_analyze(
    background_tasks: BackgroundTasks,
    response: Response,
    data: dict = Body(...),
    verified_user_id: Optional[str] = Depends(verify_api_quota),
    _rl: None = Depends(_PLAN_GEN_LIMITER),  # [P1-6] 429 si excede 3/60s per user|ip
):
    try:
        session_id = data.get("session_id")
        user_id = data.get("user_id")

        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")

        # [P1-5] Validación temprana de campos mínimos. Antes payloads incompletos
        # llegaban al pipeline y producían un plan basado en defaults genéricos
        # tras 30–90s de compute LLM. Ahora cortamos en <1ms con un 422 accionable.
        _ok, _missing = _validate_form_data_min(data)
        if not _ok:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "missing_required_fields",
                    "missing_fields": _missing,
                    "message": (
                        f"Faltan campos críticos para generar tu plan: {', '.join(_missing)}. "
                        f"Completa el formulario antes de continuar."
                    ),
                },
            )

        # [P0-FORM-4] Telemetría de signal-loss en `dislikes`. NO bloquea, solo
        # alerta cuando un cliente no oficial bypassa el gate frontend. Ver
        # docstring del helper para el rationale completo.
        _log_dislikes_signal_loss(data)

        # [P1-ORQ-8] Validación de `totalDays` en API boundary. Antes valores
        # negativos/cero/no-numéricos pasaban al orquestador que los corregía
        # silenciosamente, enmascarando bugs del cliente. Ahora 422 accionable.
        _ok_td, _td_err = _validate_total_days(data)
        if not _ok_td:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_total_days",
                    "errors": [_td_err],
                    "message": (
                        f"`totalDays` inválido: {_td_err.get('reason', 'rango/tipo')} "
                        f"(recibido: {_td_err.get('value')!r}, aceptado: "
                        f"enteros en {_td_err.get('accepted_range')})."
                    ),
                },
            )

        # [P1-3] Validación de rangos biométricos plausibles. Cubre el caso
        # ortogonal a P1-5: campo PRESENTE pero con valor fuera de rango (typo
        # "300" cuando quería "30", clientes legacy, scrapers). El backend es
        # source of truth; el wizard ya bloquea estos valores con `min`/`max`
        # HTML + `isFormValid`, pero validamos igual para defense-in-depth.
        _ok, _bio_errors = _validate_form_data_ranges(data)
        if not _ok:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_biometric_range",
                    "errors": _bio_errors,
                    "message": (
                        "Algunos datos biométricos están fuera del rango aceptado: "
                        + ", ".join(e["field"] for e in _bio_errors)
                        + ". Revísalos en el formulario."
                    ),
                },
            )

        history = []
        likes = []
        taste_profile = ""

        if session_id:
            get_or_create_session(session_id)
            memory = build_memory_context(session_id)
            history = memory["recent_messages"]

        actual_user_id = user_id if user_id and user_id != "guest" else None
        if actual_user_id:
            likes = get_user_likes(actual_user_id)

            # [GAP 10] → Movido a inject_learning_signals_from_profile (P0 fix)
            pass

        active_rejections = get_active_rejections(user_id=actual_user_id, session_id=session_id)
        rejected_meal_names = [r["meal_name"] for r in active_rejections] if active_rejections else []

        taste_profile = analyze_preferences_agent(likes, history, active_rejections=rejected_meal_names)

        # Detectar si es un plan de largo plazo que se beneficia del chunking
        total_days_requested = int(data.get("totalDays", 3))
        user_has_profile = actual_user_id and _user_has_profile(actual_user_id)
        use_chunking = bool(user_has_profile and total_days_requested > PLAN_CHUNK_SIZE)

        pipeline_data = dict(data)
        # [P0-A2] Strip ESTRICTO de cualquier `_key` heredada del payload del
        # cliente ANTES de inyectar las claves legítimas del backend. Garantiza
        # que un atacante no pueda spoofear `_quality_hint`, `_emotional_state`,
        # `_use_adversarial_play`, `_days_to_generate=9999`, `_plan_start_date`,
        # etc. Cliente normal no envía `_keys`; un drop ≥1 es señal de tampering
        # — se loguea WARNING dentro del helper. Después el backend inyecta las
        # legítimas (`_plan_start_date`, `_days_to_generate`, learning signals
        # vía `inject_learning_signals_from_profile`).
        _strip_untrusted_internal_keys(pipeline_data, allow_set=None, log_prefix="ROUTER /analyze")

        # [P1-FORM-6] Merge defensivo: une `otherAllergies`/`otherConditions`/
        # `otherDislikes`/`otherStruggles` (texto libre) en sus arrays
        # canónicos. HOY ningún consumer del router lee estos arrays entre
        # este punto y la llamada a `arun_plan_pipeline` (verificado con
        # grep). El merge real ocurre internamente en `arun_plan_pipeline`
        # (línea ~8430). Esta llamada es defense-in-depth: blinda contra
        # futuros consumers del router que pudieran agregar lógica que lea
        # esos arrays — sin esto, leerían pre-merge y verían sólo los
        # chips, no el free-text. Idempotente (dedup case-insensitive),
        # así que el merge interno del pipeline corre como no-op.
        _merge_other_text_fields(pipeline_data)

        from datetime import datetime, timezone, timedelta

        # SIEMPRE recalcular _plan_start_date en el backend a midnight local de HOY.
        # Nunca confiar en lo que venga del frontend/localStorage: observamos valores
        # obsoletos (día anterior) que corrompían day_name.
        # [P1-1] Resolver tz con fallback a `user_profiles.health_profile.tz_offset_minutes`
        # cuando el cliente omite `tzOffset` (móvil legacy, etc). Antes el fallback a 0
        # producía start_date_iso en UTC, desplazando el plan hasta 24h para usuarios
        # en TZ no-UTC.
        tz_offset_mins = _resolve_request_tz_offset(data.get("tzOffset"), actual_user_id)
        now_utc = datetime.now(timezone.utc)
        local_time = now_utc - timedelta(minutes=tz_offset_mins)
        local_midnight = local_time.replace(hour=0, minute=0, second=0, microsecond=0)
        start_date_iso = (local_midnight + timedelta(minutes=tz_offset_mins)).isoformat()

        # [P1-12] Single source of truth: SOLO `pipeline_data` recibe la
        # mutación. Antes mutábamos AMBOS dicts (`pipeline_data["_plan_start_date"]`
        # y `data["_plan_start_date"]`) para propagar el valor al chunk worker
        # snapshot + health_profile. Ahora `data` queda inmutable (representación
        # fiel del payload original) y los valores derivados (`start_date_iso`,
        # `tz_offset_mins`) se inyectan downstream vía params explícitos.
        # Invariante: el handler nunca confía en `data["_plan_start_date"]` ni
        # `data["tz_offset_minutes"]` — la fuente canónica es `pipeline_data`
        # durante el pipeline, y los locals (`start_date_iso`, `tz_offset_mins`)
        # post-pipeline.
        pipeline_data["_plan_start_date"] = start_date_iso

        if use_chunking:
            # Solo generar la Semana 1 ahora; las semanas 2-4 se generan en background
            pipeline_data["_days_to_generate"] = PLAN_CHUNK_SIZE
        elif total_days_requested > PLAN_CHUNK_SIZE:
            # Usuario sin perfil o guest solicitó plan largo → capear a 3 días
            pipeline_data["_days_to_generate"] = PLAN_CHUNK_SIZE

        # [P0-A2] Cap defensivo: si en algún branch futuro `pipeline_data["_days_to_generate"]`
        # se asignara con un valor distinto a `PLAN_CHUNK_SIZE`, este enforce
        # garantiza el límite. Hoy es no-op porque las dos asignaciones de arriba
        # ya usan PLAN_CHUNK_SIZE; queda como invariante explícito.
        _enforce_days_to_generate_cap(pipeline_data, log_prefix="ROUTER /analyze")

        # [P0 FIX GAP 1] Persistir update_reason global como señal de aprendizaje
        update_reason = data.get("update_reason")
        if actual_user_id and update_reason and update_reason != "dislike":
            def _persist_global_update_reason():
                try:
                    from db_core import execute_sql_write
                    execute_sql_write(
                        "INSERT INTO abandoned_meal_reasons (user_id, meal_type, reason) VALUES (%s, %s, %s)",
                        (actual_user_id, "full_plan", f"swap:{update_reason}")
                    )
                    logger.info(f"📝 [GLOBAL UPDATE LEARN] Razón persistida: reason=swap:{update_reason}")
                except Exception as e:
                    logger.warning(f"⚠️ [GLOBAL UPDATE LEARN] Error persistiendo update reason: {e}")
            background_tasks.add_task(_persist_global_update_reason)

        if actual_user_id:
            from cron_tasks import inject_learning_signals_from_profile
            inject_learning_signals_from_profile(actual_user_id, pipeline_data)

        memory_ctx = memory.get("full_context_str", "") if session_id else ""
        result = run_plan_pipeline(pipeline_data, history, taste_profile,
                                   memory_context=memory_ctx,
                                   background_tasks=background_tasks)

        # GUARD: Si el pipeline devolvió un plan de emergencia matemático
        # (`_is_fallback=True` — pasa cuando el LLM upstream se cayó: 504 de
        # Gemini, circuit breaker abierto, etc.), NO lo guardamos como plan real.
        # Antes el plan basura quedaba persistido + se encolaban N chunks futuros
        # con la misma referencia → el usuario veía "Fallback: pollo y arroz" por
        # una semana. Mejor: 503 con mensaje claro para que reintente.
        if isinstance(result, dict) and result.get("_is_fallback"):
            logger.warning(
                f"🚨 [FALLBACK-GUARD] Pipeline devolvió plan de emergencia "
                f"(LLM upstream caído). Devolviendo 503 sin persistir. user={actual_user_id or 'guest'}"
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "La IA está temporalmente saturada y no pudimos generar tu plan. "
                    "Por favor intenta de nuevo en 1-2 minutos."
                ),
            )

        # [P0-1/P1-1] Validación post-LLM contra nevera para el primer chunk.
        # Centralizada en `_run_pantry_validation_for_initial_chunk` — antes este
        # bloque (~40 líneas) estaba duplicado con `/analyze/stream` y divergió.
        # `_resolve_live_pantry` aplica la fuente correcta: live desde
        # `db_inventory.get_user_inventory_net` para auth, fallback al payload
        # (`current_pantry_ingredients`) para guest o fallo de DB.
        result = _run_pantry_validation_for_initial_chunk(
            result=result,
            pipeline_data=pipeline_data,
            history=history,
            taste_profile=taste_profile,
            memory_ctx=memory_ctx,
            background_tasks=background_tasks,
            actual_user_id=actual_user_id,
            pantry_ingredients=_resolve_live_pantry(actual_user_id, data),
            transport_label="P0-1",
        )

        # [P1-1] Post-procesamiento (perfil, mensajes, audit, persistencia,
        # chunking, seeding, emergency backup) centralizado en
        # `_postprocess_pipeline_result`. Antes este bloque (~200 líneas) estaba
        # duplicado entre sync y SSE y divergió varias veces.
        result = _postprocess_pipeline_result(
            result=result,
            actual_user_id=actual_user_id,
            session_id=session_id,
            data=data,
            taste_profile=taste_profile,
            memory_ctx=memory_ctx,
            rejected_meal_names=rejected_meal_names,
            total_days_requested=total_days_requested,
            use_chunking=use_chunking,
            background_tasks=background_tasks,
            plan_start_date=start_date_iso,  # [P1-12] explícito (antes vía data mutation)
            tz_offset_mins=tz_offset_mins,
            transport_label="sync",  # [P0-FIX-SEED] → context_label="seed_chunk1_sync"
        )

        # [P0-2] Adjuntar resumen de pantry-degraded al body + headers HTTP.
        # Cubre el caso P0-1 (initial chunk degraded) y el path futuro donde el
        # primer chunk ya viene marcado per-día — el frontend recibe la señal en
        # ambos sitios sin ambigüedad.
        result["_pantry_degraded_summary"] = _attach_pantry_degraded_response_meta(response, result)

        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"❌ [ERROR] Error en /api/analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/stream")
async def api_analyze_stream(
    request: Request,
    background_tasks: BackgroundTasks,
    data: dict = Body(...),
    verified_user_id: Optional[str] = Depends(verify_api_quota),
    _rl: None = Depends(_PLAN_GEN_LIMITER),  # [P1-6] mismo cap que sync — 429 antes de abrir el stream
):
    """
    Streaming SSE endpoint para generación de planes con progreso en tiempo real.
    Emite eventos:
      - phase: cambio de fase del pipeline (skeleton, parallel_generation, assembly, review)
      - day_started: un worker paralelo inició la generación de un día
      - day_complete: un worker paralelo terminó un día
      - complete: plan final listo (contiene el plan JSON completo)
      - error: hubo un error
    """
    try:
        session_id = data.get("session_id")
        user_id = data.get("user_id")

        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")

        # [P1-5] Misma validación temprana que el endpoint sync. Lanzar 422 ANTES
        # de abrir el StreamingResponse: si el payload es inválido, el cliente
        # recibe un JSON estándar (no un stream SSE con un único evento de error
        # — peor UX y harder to handle del lado JS).
        _ok, _missing = _validate_form_data_min(data)
        if not _ok:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "missing_required_fields",
                    "missing_fields": _missing,
                    "message": (
                        f"Faltan campos críticos para generar tu plan: {', '.join(_missing)}. "
                        f"Completa el formulario antes de continuar."
                    ),
                },
            )

        # [P0-FORM-4] Telemetría de signal-loss en `dislikes`. Mismo helper que
        # el endpoint sync — defense-in-depth para detectar bypasses del gate
        # frontend desde clientes no oficiales.
        _log_dislikes_signal_loss(data)

        # [P1-ORQ-8] Misma validación de `totalDays` que sync. Lanzar 422 ANTES
        # de abrir el StreamingResponse para que el cliente reciba un JSON
        # accionable, no un stream con error event (peor UX/handling JS).
        _ok_td, _td_err = _validate_total_days(data)
        if not _ok_td:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_total_days",
                    "errors": [_td_err],
                    "message": (
                        f"`totalDays` inválido: {_td_err.get('reason', 'rango/tipo')} "
                        f"(recibido: {_td_err.get('value')!r}, aceptado: "
                        f"enteros en {_td_err.get('accepted_range')})."
                    ),
                },
            )

        # [P1-3] Mismo check de rangos biométricos que sync. Lanzar 422 ANTES
        # de abrir el StreamingResponse para que el cliente reciba un JSON
        # accionable, no un stream con error event (cliente lo maneja peor).
        _ok, _bio_errors = _validate_form_data_ranges(data)
        if not _ok:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_biometric_range",
                    "errors": _bio_errors,
                    "message": (
                        "Algunos datos biométricos están fuera del rango aceptado: "
                        + ", ".join(e["field"] for e in _bio_errors)
                        + ". Revísalos en el formulario."
                    ),
                },
            )

        history = []
        likes = []
        taste_profile = ""

        if session_id:
            get_or_create_session(session_id)
            memory = build_memory_context(session_id)
            history = memory["recent_messages"]

        actual_user_id = user_id if user_id and user_id != "guest" else None
        if actual_user_id:
            likes = get_user_likes(actual_user_id)

            # [GAP 10] → Movido a inject_learning_signals_from_profile (P0 fix)
            pass

        active_rejections = get_active_rejections(user_id=actual_user_id, session_id=session_id)
        rejected_meal_names = [r["meal_name"] for r in active_rejections] if active_rejections else []

        taste_profile = analyze_preferences_agent(likes, history, active_rejections=rejected_meal_names)

        # [GAP 4 FIX: Detectar si es un plan de largo plazo que se beneficia del chunking]
        total_days_requested = int(data.get("totalDays", 3))
        user_has_profile = actual_user_id and _user_has_profile(actual_user_id)
        use_chunking = bool(user_has_profile and total_days_requested > PLAN_CHUNK_SIZE)

        pipeline_data = dict(data)
        # [P0-A2] Strip estricto de `_keys` heredadas del cliente — mismo
        # rationale que en `/analyze`. Capa principal de defensa contra
        # injection de claves internas (`_quality_hint`, `_emotional_state`,
        # `_use_adversarial_play`, `_days_to_generate=9999`, etc.).
        _strip_untrusted_internal_keys(pipeline_data, allow_set=None, log_prefix="ROUTER /analyze/stream")

        # [P1-FORM-6] Mismo merge defensivo que `/analyze`. Idempotente con
        # el merge interno de `arun_plan_pipeline`. Ver comentario equivalente
        # arriba para el rationale completo de defense-in-depth.
        _merge_other_text_fields(pipeline_data)

        from datetime import datetime, timezone, timedelta

        # SIEMPRE recalcular _plan_start_date en el backend a midnight local de HOY.
        # Nunca confiar en lo que venga del frontend/localStorage: observamos valores
        # obsoletos (día anterior) que corrompían day_name.
        # [P1-1] Resolver tz con fallback a `user_profiles.health_profile.tz_offset_minutes`
        # cuando el cliente omite `tzOffset` (móvil legacy, etc). Antes el fallback a 0
        # producía start_date_iso en UTC, desplazando el plan hasta 24h para usuarios
        # en TZ no-UTC.
        tz_offset_mins = _resolve_request_tz_offset(data.get("tzOffset"), actual_user_id)
        now_utc = datetime.now(timezone.utc)
        local_time = now_utc - timedelta(minutes=tz_offset_mins)
        local_midnight = local_time.replace(hour=0, minute=0, second=0, microsecond=0)
        start_date_iso = (local_midnight + timedelta(minutes=tz_offset_mins)).isoformat()

        # [P1-12] SSOT: solo `pipeline_data` se muta. Ver comentario equivalente
        # en `api_analyze` para el rationale completo.
        pipeline_data["_plan_start_date"] = start_date_iso

        if use_chunking:
            # Solo generar la Semana 1 ahora; las semanas 2-4 se generan en background
            pipeline_data["_days_to_generate"] = PLAN_CHUNK_SIZE
        elif total_days_requested > PLAN_CHUNK_SIZE:
            # Usuario sin perfil o guest solicitó plan largo → capear a 3 días
            pipeline_data["_days_to_generate"] = PLAN_CHUNK_SIZE

        # [P0-A2] Cap defensivo de `_days_to_generate` (invariante explícito).
        _enforce_days_to_generate_cap(pipeline_data, log_prefix="ROUTER /analyze/stream")

        # [P0 FIX GAP 1] Persistir update_reason global como señal de aprendizaje
        update_reason = data.get("update_reason")
        if actual_user_id and update_reason and update_reason != "dislike":
            def _persist_global_update_reason_stream():
                try:
                    from db_core import execute_sql_write
                    execute_sql_write(
                        "INSERT INTO abandoned_meal_reasons (user_id, meal_type, reason) VALUES (%s, %s, %s)",
                        (actual_user_id, "full_plan", f"swap:{update_reason}")
                    )
                    logger.info(f"📝 [GLOBAL UPDATE LEARN (STREAM)] Razón persistida: reason=swap:{update_reason}")
                except Exception as e:
                    logger.warning(f"⚠️ [GLOBAL UPDATE LEARN (STREAM)] Error persistiendo update reason: {e}")
            background_tasks.add_task(_persist_global_update_reason_stream)

        # Inyectar TODAS las señales de aprendizaje continuo
        if actual_user_id:
            from cron_tasks import inject_learning_signals_from_profile
            inject_learning_signals_from_profile(actual_user_id, pipeline_data)

        # Cola async para comunicar progreso entre el thread del pipeline y el generador SSE
        progress_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def progress_callback(event_data: dict):
            """Callback thread-safe que pone eventos en la cola async."""
            try:
                loop.call_soon_threadsafe(progress_queue.put_nowait, event_data)
            except Exception:
                pass

        # Ejecutar el pipeline en un thread separado para no bloquear el event loop
        pipeline_result = {"result": None, "error": None}

        async def run_pipeline():
            try:
                result = await arun_plan_pipeline(
                    pipeline_data, history, taste_profile,
                    memory_context=memory.get("full_context_str", "") if session_id else "",
                    progress_callback=progress_callback,
                    background_tasks=background_tasks
                )
                pipeline_result["result"] = result
            except Exception as e:
                pipeline_result["error"] = str(e)
                logger.error(f"❌ [SSE PIPELINE ERROR]: {e}")
                traceback.print_exc()
            finally:
                # Señal de fin para que el generador SSE cierre
                try:
                    loop.call_soon_threadsafe(progress_queue.put_nowait, {"event": "_done"})
                except Exception:
                    pass

        asyncio.create_task(run_pipeline())

        async def event_generator():
            """Generador SSE que consume la cola de progreso."""
            try:
                while True:
                    # Esperar eventos con timeout para detectar desconexión del cliente
                    try:
                        event_data = await asyncio.wait_for(progress_queue.get(), timeout=5.0)
                    except asyncio.TimeoutError:
                        # Heartbeat para mantener la conexión viva
                        yield f"data: {_json.dumps({'event': 'heartbeat'})}\n\n"

                        # Verificar si el cliente cerró la conexión
                        if await request.is_disconnected():
                            logger.info("🔌 [SSE] Cliente desconectado, abortando stream.")
                            return
                        continue

                    # Señal de fin del pipeline
                    if event_data.get("event") == "_done":
                        # Enviar resultado final o error
                        if pipeline_result["error"]:
                            yield f"data: {_json.dumps({'event': 'error', 'data': {'message': pipeline_result['error']}})}\n\n"
                        elif pipeline_result["result"]:
                            result = pipeline_result["result"]

                            # GUARD: Si el pipeline devolvió un plan de emergencia
                            # matemático (LLM upstream caído), NO persistir ni encolar
                            # chunks. Emitir error SSE para que el frontend muestre
                            # "intenta de nuevo" en lugar de un plan basura permanente.
                            if isinstance(result, dict) and result.get("_is_fallback"):
                                logger.warning(
                                    f"🚨 [FALLBACK-GUARD/SSE] Pipeline devolvió plan de "
                                    f"emergencia (LLM upstream caído). No se persiste. "
                                    f"user={actual_user_id or 'guest'}"
                                )
                                yield f"data: {_json.dumps({'event': 'error', 'data': {'code': 'llm_unavailable', 'message': 'La IA está temporalmente saturada y no pudimos generar tu plan. Por favor intenta de nuevo en 1-2 minutos.'}})}\n\n"
                                break

                            # [P0-3] Pre-check de desconexión ANTES de la validación pantry.
                            # La validación P0-2 puede tardar 30–60s por reintento y la
                            # persistencia que viene después es bloqueante. Si el cliente colgó
                            # (timeout, abort, navegación), cortar aquí evita persistir un plan
                            # huérfano que aparecería como "duplicado" en la próxima sesión.
                            if await request.is_disconnected():
                                logger.warning(
                                    f"🔌 [P0-3 SSE] Cliente desconectado tras pipeline (antes de "
                                    f"validación pantry). NO se persiste plan para evitar "
                                    f"duplicación silenciosa. user={actual_user_id or 'guest'}"
                                )
                                break

                            # [P0-2/P1-1] Validación post-LLM contra nevera centralizada.
                            # Antes este bloque (~50 líneas) duplicaba la lógica del sync con
                            # mensajes de log divergentes. Ahora `_run_pantry_validation_for_initial_chunk`
                            # se ejecuta vía `asyncio.to_thread` para no bloquear el event loop
                            # (el helper interno dispara `run_plan_pipeline` sync por retry).
                            _memory_ctx_sse = (
                                memory.get("full_context_str", "") if session_id else ""
                            )
                            result = await asyncio.to_thread(
                                _run_pantry_validation_for_initial_chunk,
                                result=result,
                                pipeline_data=pipeline_data,
                                history=history,
                                taste_profile=taste_profile,
                                memory_ctx=_memory_ctx_sse,
                                background_tasks=background_tasks,
                                actual_user_id=actual_user_id,
                                pantry_ingredients=_resolve_live_pantry(actual_user_id, data),
                                transport_label="P0-2 SSE",
                            )

                            # [P0-3] Re-check de desconexión DESPUÉS de la validación pantry
                            # (puede haber tardado decenas de segundos en peor caso).
                            if await request.is_disconnected():
                                logger.warning(
                                    f"🔌 [P0-3 SSE] Cliente desconectado durante validación pantry. "
                                    f"NO se persiste plan. user={actual_user_id or 'guest'}"
                                )
                                break

                            # [P0-4/P1-1] Post-procesamiento centralizado. Antes este bloque
                            # (~200 líneas) estaba duplicado e incluía DB writes inline en el
                            # coroutine + `threading.Thread(daemon=True)` para persistencia
                            # (rompía bajo SIGTERM). Centralizado en un único helper síncrono
                            # invocado vía `asyncio.to_thread` para no bloquear el event loop.
                            result = await asyncio.to_thread(
                                _postprocess_pipeline_result,
                                result=result,
                                actual_user_id=actual_user_id,
                                session_id=session_id,
                                data=data,
                                taste_profile=taste_profile,
                                memory_ctx=_memory_ctx_sse,
                                rejected_meal_names=rejected_meal_names,
                                total_days_requested=total_days_requested,
                                use_chunking=use_chunking,
                                background_tasks=background_tasks,
                                plan_start_date=start_date_iso,  # [P1-12] explícito
                                tz_offset_mins=tz_offset_mins,
                                transport_label="sse",  # [P0-FIX-SEED] → context_label="seed_chunk1_sse"
                            )

                            # [P1-2] Adjuntar `_pantry_degraded_summary` al payload del evento
                            # `complete`. Antes el SSE NO lo computaba: el sync exponía la
                            # señal vía body + headers HTTP (`X-Pantry-Degraded`), pero el
                            # SSE — path PRIMARIO del frontend — la omitía completamente, así
                            # que el banner UX que avisa "tu plan tiene ingredientes que no
                            # están en tu nevera" no aparecía cuando la generación corría por
                            # streaming. Pasamos `response=None` porque en SSE los headers HTTP
                            # se envían UNA sola vez al inicio del stream (antes de saber si
                            # hubo degradación); el frontend lee la señal del body en ambos
                            # paths igual.
                            result["_pantry_degraded_summary"] = _attach_pantry_degraded_response_meta(
                                None, result,
                            )

                            yield f"data: {_json.dumps({'event': 'complete', 'data': result}, ensure_ascii=False, default=str)}\n\n"
                        return

                    # [P1-11] Filtro de eventos públicos: solo enviar al cliente
                    # los nombres declarados en `schemas.PUBLIC_SSE_EVENTS`.
                    # Antes el endpoint reenviaba CUALQUIER evento del
                    # `progress_callback` (incluyendo `metric`, `token`,
                    # `tool_call`, `token_reset`) que el frontend silenciosamente
                    # ignoraba — ~10 KB de bandwidth desperdiciado por request +
                    # ruido en DevTools. El filtro corta ahí. Eventos de
                    # observabilidad (`metric`) siguen escribiendo a
                    # `pipeline_metrics` vía la rama interna de `_emit_progress`
                    # (no afectada por este filtro).
                    #
                    # Alias `day_completed` → `day_complete`: el orquestador
                    # emite `day_completed` (~50 call sites) pero `Plan.jsx`
                    # escucha `day_complete`. Renombramos en wire para fixear
                    # el bug latente del progress bar sin tocar el orquestador.
                    _ev_name = event_data.get("event")
                    if _ev_name not in PUBLIC_SSE_EVENTS:
                        continue
                    if _ev_name == "day_completed":
                        event_data = {**event_data, "event": "day_complete"}
                    yield f"data: {_json.dumps(event_data, ensure_ascii=False)}\n\n"

            except asyncio.CancelledError:
                logger.info("🔌 [SSE] Stream cancelado por el cliente.")
            except Exception as e:
                logger.error(f"❌ [SSE] Error en generador: {e}")
                yield f"data: {_json.dumps({'event': 'error', 'data': {'message': str(e)}})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"❌ [ERROR] Error en /api/analyze/stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recipe/expand")
def api_expand_recipe(data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        user_id = data.get("user_id")
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")
                
        if not data.get("recipe") or not data.get("name"):
            raise HTTPException(status_code=400, detail="Faltan datos de la receta para expandir.")
            
        if user_id and user_id != "guest":
            log_api_usage(user_id, "gemini_recipe_expand")
            
        expanded_steps = expand_recipe_agent(data)
        
        if user_id and user_id != "guest":
            current_plan = get_latest_meal_plan(user_id)
            if current_plan and "days" in current_plan:
                updated = False
                for day in current_plan.get("days", []):
                    for m in day.get("meals", []):
                        if m.get("name") == data.get("name"):
                            m["recipe"] = expanded_steps
                            m["isExpanded"] = True
                            updated = True
                            break
                    if updated: break
                
                if updated:
                    plan_with_id = get_latest_meal_plan_with_id(user_id)
                    if plan_with_id and "id" in plan_with_id:
                        update_meal_plan_data(plan_with_id["id"], current_plan)

        return {"success": True, "expanded_recipe": expanded_steps}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/recipe/expand: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/swap-meal")
def api_swap_meal(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        session_id = data.get("session_id")
        user_id = data.get("user_id")
        
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
                
        rejected_meal = data.get("rejected_meal")
        meal_type = data.get("meal_type", "")
        swap_reason = data.get("swap_reason", "variety")  # variety | time | dislike | similar | budget
        
        # Solo registrar rechazo cuando el usuario explícitamente dice "No me gusta"
        if rejected_meal and swap_reason == "dislike":
            logger.info(f"👎 [SWAP] Rechazo real registrado: '{rejected_meal}' (razón: {swap_reason})")
            background_tasks.add_task(_process_swap_rejection_background, session_id, user_id, rejected_meal, meal_type)
        else:
            logger.info(f"🔄 [SWAP] Cambio sin rechazo: '{rejected_meal or 'N/A'}' (razón: {swap_reason})")

        # [P1 FIX] Persistir TODAS las swap reasons como señal de aprendizaje.
        # El cron las lee de abandoned_meal_reasons para detectar patrones
        # (ej: "usuario siempre cambia desayuno por falta de tiempo → simplificar").
        if user_id and user_id != "guest" and rejected_meal and swap_reason != "dislike":
            def _persist_swap_reason():
                try:
                    from db_core import execute_sql_write
                    execute_sql_write(
                        "INSERT INTO abandoned_meal_reasons (user_id, meal_type, reason) VALUES (%s, %s, %s)",
                        (user_id, meal_type or "unknown", f"swap:{swap_reason}")
                    )
                    logger.info(f"📝 [SWAP LEARN] Razón persistida: meal_type={meal_type}, reason=swap:{swap_reason}")
                except Exception as e:
                    logger.warning(f"⚠️ [SWAP LEARN] Error persistiendo swap reason: {e}")
            background_tasks.add_task(_persist_swap_reason)
            
        if user_id and user_id != "guest":
            log_api_usage(user_id, "gemini_swap_meal")
            
            # --- HOT SIGNAL PATH (MEJORA 4) ---
            try:
                # Obtenemos likes y rechazos recientes para que el LLM no repita errores en el JIT Swap
                recent_likes = get_user_likes(user_id)
                recent_rejections = get_active_rejections(user_id=user_id, session_id=session_id)
                
                data["liked_meals"] = [like["meal_name"] for like in recent_likes] if recent_likes else []
                data["disliked_meals"] = [r["meal_name"] for r in recent_rejections] if recent_rejections else []
                logger.info(f"🔥 [HOT SIGNAL] Inyectando {len(data['liked_meals'])} likes y {len(data['disliked_meals'])} rechazos al JIT Swap.")
            except Exception as e:
                logger.warning(f"⚠️ [HOT SIGNAL] Error recuperando señales en tiempo real: {e}")
            
        result = swap_meal(data)
        return result
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/swap-meal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/like")
def api_like(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        user_id = data.get("user_id")
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
                
        insert_like(data)
        return {"success": True, "message": "Tu like/dislike ha sido guardado exitosamente."}
    except Exception as e:
        return {"error": str(e)}

@router.post("/restock")
def api_restock(data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        user_id = data.get("user_id")
        plan_id = data.get("plan_id")
        ingredients = data.get("ingredients")
        
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Debes iniciar sesión para usar la nevera virtual."}
            
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado.")
            
        if not ingredients or not isinstance(ingredients, list):
            return {"success": False, "message": "Lista de ingredientes inválida."}

        # [P0-1] Validación de Idempotencia: Verificar si el plan ya fue registrado ANTES de insertar en DB
        real_plan_id = None
        plan_data = None
        if supabase:
            try:
                if plan_id:
                    plan_res = supabase.table("meal_plans").select("id, plan_data").eq("id", plan_id).execute()
                else:
                    plan_res = supabase.table("meal_plans").select("id, plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
                
                if plan_res and plan_res.data and len(plan_res.data) > 0:
                    real_plan_id = plan_res.data[0].get("id")
                    plan_data = plan_res.data[0].get("plan_data", {})
                    
                    if plan_data.get("is_restocked") is True:
                        logger.warning(f"⚠️ [RESTOCK] El plan {real_plan_id} ya fue registrado previamente. Ignorando petición duplicada.")
                        return {"success": True, "message": "Las compras ya estaban registradas."}
            except Exception as check_err:
                logger.warning(f"⚠️ Error verificando estado is_restocked: {check_err}")

        # Validación MURO Omitida: Ahora confiamos en el Delta Shopping del frontend.
        # El frontend solo envía los ingredientes que no están en la Nevera.
        success = restock_inventory(user_id, ingredients)
        
        if success:
            log_api_usage(user_id, "restock_inventory")

            # Marcar el plan como "restocked" en BD para futuras peticiones
            if supabase and real_plan_id and plan_data is not None:
                try:
                    plan_data["is_restocked"] = True
                    supabase.table("meal_plans").update({"plan_data": plan_data}).eq("id", real_plan_id).execute()
                    logger.info(f"✅ [RESTOCK] plan_data 'is_restocked' guardado en DB para plan ID {real_plan_id}")
                except Exception as mark_err:
                    logger.warning(f"⚠️ No se pudo marcar plan como restocked: {mark_err}")

            return {"success": True, "message": "¡Despensa actualizada exitosamente!"}
        else:
            return {"success": False, "message": "Hubo un problema actualizando algunos ingredientes."}
            
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/restock: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inventory/consume")
def api_consume_inventory(data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        user_id = data.get("user_id")
        ingredients = data.get("ingredients")
        
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Debes iniciar sesión."}
            
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado.")
            
        if not ingredients or not isinstance(ingredients, list):
            return {"success": False, "message": "Lista de ingredientes inválida."}
            
        success = consume_inventory_items_completely(user_id, ingredients)
        
        if success:
            log_api_usage(user_id, "consume_inventory")
            return {"success": True, "message": "Inventario actualizado exitosamente."}
        else:
            return {"success": False, "message": "Hubo un problema vaciando algunos ingredientes."}
            
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/inventory/consume: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recalculate-shopping-list")
def api_recalculate_shopping_list(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    """
    Recalcula la lista de compras escalando las recetas por el householdSize 
    y LUEGO deduciendo el inventario físico (is_new_plan=False).
    Este acercamiento garantiza exactitud matemática.
    """
    try:
        user_id = data.get("user_id")
        household_size = max(1, int(data.get("householdSize", 1) or 1))
        grocery_duration = data.get("groceryDuration", "weekly")
        is_new_plan_flag = data.get("is_new_plan", False)
        
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Debes iniciar sesión."}
            
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado.")
        
        plan_record = get_latest_meal_plan_with_id(user_id)
        if not plan_record:
            return {"success": False, "message": "No hay plan activo."}
        
        plan_id = plan_record["id"]
        plan_data = plan_record.get("plan_data", {})
        
        if not plan_data:
            return {"success": False, "message": "Datos de plan inválidos."}
            
        from shopping_calculator import get_shopping_list_delta
        from db_plans import update_meal_plan_data
        
        # Generar las 3 variantes escaladas dinámicamente según el householdSize
        # usando el delta matemático para evitar duplicados si hay inventario (Gap 3)
        scaled_7 = get_shopping_list_delta(user_id, plan_data, is_new_plan=is_new_plan_flag, structured=True, multiplier=float(household_size))
        scaled_15 = get_shopping_list_delta(user_id, plan_data, is_new_plan=is_new_plan_flag, structured=True, multiplier=float(household_size) * 2.0)
        scaled_30 = get_shopping_list_delta(user_id, plan_data, is_new_plan=is_new_plan_flag, structured=True, multiplier=float(household_size) * 4.0)
        
        # Debug: Log DETAILED per-item comparison to diagnose scaling
        if scaled_7:
            sample = [f"{it.get('display_string','?')}" for it in scaled_7[:3]]
            has_days = bool(plan_data.get("days"))
            len_days = len(plan_data.get("days", []))
            has_perfectDay = bool(plan_data.get("perfectDay"))
            logger.info(f"🔍 [RECALC DEBUG] ×{household_size} sample (7d): {sample} | has_days={has_days} len={len_days} has_perf={has_perfectDay}")
            
            # DEBUG GRANULAR: rastear proteínas y frutas específicas
            debug_keywords = ['pechuga', 'pavo', 'yogurt', 'lechosa', 'melón', 'melon', 'aguacate', 'arroz', 'pollo']
            for it in scaled_7:
                name_lower = it.get('name', '').lower()
                if any(kw in name_lower for kw in debug_keywords):
                    logger.info(f"  📊 [{household_size}p] {it.get('name')}: display_qty={it.get('display_qty')} | market_qty={it.get('market_qty')} {it.get('market_unit')} | display_string={it.get('display_string')}")
        
        # Seleccionar lista activa para el frontend legacy
        if grocery_duration == "biweekly":
            active_list = scaled_15
        elif grocery_duration == "monthly":
            active_list = scaled_30
        else:
            active_list = scaled_7
        
        # Actualizar plan en BD
        plan_data["aggregated_shopping_list"] = active_list
        plan_data["aggregated_shopping_list_weekly"] = scaled_7
        plan_data["aggregated_shopping_list_biweekly"] = scaled_15
        plan_data["aggregated_shopping_list_monthly"] = scaled_30
        
        # Solo limpiar `is_restocked` si los parámetros cambiaron realmente
        prev_hh = plan_data.get("calc_household_size")
        prev_dur = plan_data.get("calc_grocery_duration")
        has_changed = (prev_hh != household_size) or (prev_dur != grocery_duration)
        
        plan_data["calc_household_size"] = household_size
        plan_data["calc_grocery_duration"] = grocery_duration
        
        if has_changed and plan_data.get("is_restocked"):
            plan_data.pop("is_restocked", None)
            logger.info(f"🔄 [RECALC] is_restocked limpiado — cantidades cambiaron de {prev_hh}p/{prev_dur} a {household_size}p/{grocery_duration}, requiere re-registro")
        
        # DEBUG fingerprint: allows frontend to verify it received fresh data
        import time
        plan_data["_debug_recalc"] = {
            "household_size": household_size,
            "timestamp": time.time(),
            "weekly_items_count": len(scaled_7),
            "sample_item": scaled_7[0].get("display_string", "?") if scaled_7 else "empty"
        }
        
        update_meal_plan_data(plan_id, plan_data)
        
        logger.info(f"✅ [RECALC] Listas recalculadas exitosamente ×{household_size} personas")
        
        # Devolver el plan_data actualizado directamente para evitar race conditions
        # (el frontend no necesita re-fetch de Supabase)
        return {"success": True, "plan_data": plan_data}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/recalculate-shopping-list: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{plan_id}/chunk-status")
def api_chunk_status(plan_id: str, response: Response, verified_user_id: Optional[str] = Depends(verify_api_quota)):
    from db_core import execute_sql_query
    try:
        res = execute_sql_query("SELECT user_id, plan_data FROM meal_plans WHERE id = %s", (plan_id,), fetch_one=True)
        if not res:
            raise HTTPException(status_code=404, detail="Plan no encontrado")
            
        user_id = res["user_id"]
        plan_data = res["plan_data"]
        status = plan_data.get("generation_status", "complete")
        days_generated = len(plan_data.get("days", []))
        total_days = plan_data.get("total_days_requested", days_generated)
        
        # [GAP 2] Buscar chunks fallidos si hay problemas
        failed_chunks = []
        if status in ['failed', 'complete_partial']:
            chunks_res = execute_sql_query(
                "SELECT id, week_number, status, attempts FROM plan_chunk_queue WHERE meal_plan_id = %s AND status = 'failed' ORDER BY week_number ASC",
                (plan_id,)
            )
            if chunks_res:
                failed_chunks = chunks_res
                # Si hay chunks fallidos explícitos, forzamos el status general a failed
                status = "failed"
                
        # [GAP G FIX: Enriquecer payload con ETA y learning hint]
        eta_res = execute_sql_query(
            "SELECT execute_after FROM plan_chunk_queue WHERE meal_plan_id = %s AND status IN ('pending', 'processing') ORDER BY execute_after ASC LIMIT 1",
            (plan_id,), fetch_one=True
        )
        next_chunk_eta = eta_res["execute_after"].isoformat() if eta_res and eta_res.get("execute_after") else None
        
        last_learning_hint = "Analizando tus preferencias..."
        user_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
        if user_res and user_res.get("health_profile"):
            hp = user_res["health_profile"]
            qh = hp.get("quality_history", [])
            if qh and len(qh) > 0:
                last_score = qh[-1].get("score", 0)
                last_learning_hint = f"Ajustando variedad (Quality Score: {last_score}/100)"

        # [GAP C] Exponer quality_warning y desglose de tiers
        quality_warning = bool(plan_data.get("quality_warning", False))
        quality_degraded_ratio = float(plan_data.get("quality_degraded_ratio", 0.0))
        tier_breakdown = execute_sql_query("""
            SELECT quality_tier, COUNT(*) AS cnt
            FROM plan_chunk_queue
            WHERE meal_plan_id = %s AND status = 'completed' AND quality_tier IS NOT NULL
            GROUP BY quality_tier
        """, (plan_id,)) or []
        tier_summary = {r['quality_tier']: int(r['cnt']) for r in tier_breakdown}

        # [P0-2] Resumen de pantry-degraded para el polling de chunk-status. El frontend
        # consulta este endpoint repetidamente mientras el plan está partial, así que
        # aquí es donde puede actualizar el banner cuando llegan chunks degraded.
        _p02_summary = _attach_pantry_degraded_response_meta(response, plan_data)

        return {
            "status": status,
            "days_generated": days_generated,
            "total_days_requested": total_days,  # Added requested
            "total_days": total_days,  # Mantener por retrocompatibilidad
            "failed_chunks": failed_chunks,
            "next_chunk_eta": next_chunk_eta,
            "last_learning_hint": last_learning_hint,
            "quality_warning": quality_warning,
            "quality_degraded_ratio": quality_degraded_ratio,
            "tier_breakdown": tier_summary,
            "_pantry_degraded_summary": _p02_summary,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] en chunk-status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{plan_id}/blocked_reasons")
def api_blocked_reasons(plan_id: str, verified_user_id: Optional[str] = Depends(verify_api_quota)):
    """[P1-2] Devuelve los motivos legibles por los que un plan está bloqueado.

    El frontend usa este endpoint para mostrar un banner persistente cuando un chunk
    está pausado en `pending_user_action` con su motivo (zero-log, pantry vacía, snapshot
    obsoleto), evitando que el usuario crea que el plan "se estancó" sin razón visible.

    Antes solo se mandaban hasta 2 push notifications y luego silencio durante horas.
    """
    from db_core import execute_sql_query
    try:
        plan = execute_sql_query(
            "SELECT user_id FROM meal_plans WHERE id = %s",
            (plan_id,),
            fetch_one=True,
        )
        if not plan:
            raise HTTPException(status_code=404, detail="Plan no encontrado")
        if verified_user_id and str(plan["user_id"]) != str(verified_user_id):
            raise HTTPException(status_code=403, detail="No autorizado")

        rows = execute_sql_query(
            """
            SELECT id, week_number, pipeline_snapshot, status,
                   EXTRACT(EPOCH FROM (NOW() - updated_at))::int AS paused_seconds
            FROM plan_chunk_queue
            WHERE meal_plan_id = %s AND status = 'pending_user_action'
            ORDER BY week_number ASC
            """,
            (plan_id,),
        ) or []

        # [P1-4] Lectura de logging_preference para enriquecer el motivo learning_zero_logs:
        # si el usuario aún no ha optado por auto_proxy, indicamos que la opción está disponible
        # para que el frontend ofrezca un toggle "continuar sin registrar" → PUT /preferences/logging.
        current_logging_pref = "manual"
        try:
            _pref_row = execute_sql_query(
                "SELECT logging_preference FROM user_profiles WHERE id = %s",
                (str(plan["user_id"]),),
                fetch_one=True,
            )
            if _pref_row and _pref_row.get("logging_preference"):
                current_logging_pref = str(_pref_row["logging_preference"])
        except Exception:
            pass

        reasons = []
        reason_to_text = {
            "learning_zero_logs": {
                "title": "Registra tus comidas para continuar tu plan",
                "body": "Necesitamos saber qué comiste para generar el siguiente bloque adaptado a ti.",
                "cta": "Ir al diario",
                "url": "/diary",
                # [P1-4] Acción secundaria: opt-in a auto_proxy para que el plan continúe
                # automáticamente sin requerir log explícito. Solo se ofrece si está en 'manual'.
                "secondary_action": (
                    {
                        "label": "Continuar sin registrar",
                        "endpoint": "/api/diary/preferences/logging",
                        "method": "PUT",
                        "body": {"logging_preference": "auto_proxy"},
                        "hint": "El plan continuará usando tu inventario como señal de actividad.",
                    }
                    if current_logging_pref == "manual"
                    else None
                ),
            },
            "stale_snapshot": {
                "title": "Validando tu inventario",
                "body": "Estamos refrescando tu nevera. El plan continuará en breve.",
                "cta": None,
                "url": None,
            },
            "stale_snapshot_live_unreachable": {
                "title": "Actualiza tu nevera para continuar",
                "body": "No pudimos validar tu inventario en vivo. Abre la app para refrescar y continuar el plan.",
                "cta": "Abrir nevera",
                "url": "/inventory",
            },
            "empty_pantry": {
                "title": "Tu nevera está vacía",
                "body": "Añade ingredientes a 'Mi Nevera' para que generemos el siguiente bloque del plan.",
                "cta": "Actualizar nevera",
                "url": "/inventory",
            },
        }

        import json as _json
        for row in rows:
            snap = row.get("pipeline_snapshot") or {}
            if isinstance(snap, str):
                try:
                    snap = _json.loads(snap)
                except Exception:
                    snap = {}
            reason_code = str(snap.get("_pantry_pause_reason") or "empty_pantry")
            template = reason_to_text.get(reason_code, reason_to_text["empty_pantry"])
            reasons.append({
                "chunk_id": row.get("id"),
                "week_number": row.get("week_number"),
                "reason_code": reason_code,
                "paused_seconds": int(row.get("paused_seconds") or 0),
                **template,
            })

        return {"plan_id": plan_id, "blocked": len(reasons) > 0, "reasons": reasons}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] en blocked_reasons: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{plan_id}/retry-chunk/{chunk_id}")
def api_retry_chunk(plan_id: str, chunk_id: str, verified_user_id: Optional[str] = Depends(verify_api_quota)):
    from db_core import execute_sql_write
    try:
        # Resetear el chunk fallido a 'pending'
        execute_sql_write("""
            UPDATE plan_chunk_queue
            SET status = 'pending',
                attempts = 0,
                execute_after = NOW(),
                updated_at = NOW()
            WHERE id = %s AND meal_plan_id = %s AND status = 'failed'
        """, (chunk_id, plan_id))

        # Revivir cualquier chunk que haya sido cancelado por culpa de este fallo
        execute_sql_write("""
            UPDATE plan_chunk_queue
            SET status = 'pending',
                attempts = 0,
                execute_after = NOW() + INTERVAL '1 minute',
                updated_at = NOW()
            WHERE meal_plan_id = %s AND status = 'cancelled'
        """, (plan_id,))

        # Volver a poner el plan en 'partial' para que el frontend retome el polling
        execute_sql_write("""
            UPDATE meal_plans
            SET plan_data = jsonb_set(plan_data, '{generation_status}', '"partial"'),
                updated_at = NOW()
            WHERE id = %s
        """, (plan_id,))

        return {"success": True, "message": "Chunk reenviado a la cola"}

    except Exception as e:
        logger.error(f"❌ [ERROR] en retry-chunk: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# [GAP A] Endpoint de inspección de SLA — chunks atrasados
# ============================================================
def _verify_admin_token(authorization: Optional[str]):
    """Valida Bearer token contra CRON_SECRET. Si CRON_SECRET no está seteado,
    rechaza por defecto (no exponer admin endpoints en prod sin secreto).
    """
    cron_secret = os.environ.get("CRON_SECRET")
    if not cron_secret:
        raise HTTPException(status_code=503, detail="Admin endpoints disabled: CRON_SECRET not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.replace("Bearer ", "").strip()
    if token != cron_secret:
        raise HTTPException(status_code=403, detail="Invalid admin token")


@router.get("/admin/chunks/stuck")
def api_admin_chunks_stuck(
    request: Request,
    min_lag_hours: int = 1,
    limit: int = 100,
):
    """[GAP A] Inspección operacional: lista chunks atrasados (lag > min_lag_hours).

    Usar para diagnosticar por qué planes de 15-30 días no avanzan.
    Requiere Authorization: Bearer <CRON_SECRET>.
    """
    _verify_admin_token(request.headers.get("authorization"))
    from db_core import execute_sql_query
    try:
        rows = execute_sql_query(
            """
            SELECT
                q.id,
                q.user_id,
                q.meal_plan_id,
                q.week_number,
                q.days_offset,
                q.days_count,
                q.status,
                q.attempts,
                q.quality_tier,
                q.escalated_at,
                q.execute_after,
                q.created_at,
                q.updated_at,
                EXTRACT(EPOCH FROM (NOW() - q.execute_after))::int AS lag_seconds,
                (mp.plan_data->>'total_days_requested')::int AS total_days_requested,
                jsonb_array_length(COALESCE(mp.plan_data->'days', '[]'::jsonb)) AS days_generated
            FROM plan_chunk_queue q
            LEFT JOIN meal_plans mp ON mp.id = q.meal_plan_id
            WHERE q.status IN ('pending', 'stale', 'processing', 'failed')
              AND q.execute_after < NOW() - make_interval(hours => %s)
            ORDER BY q.execute_after ASC
            LIMIT %s
            """,
            (int(min_lag_hours), int(limit)),
        ) or []

        # Resumen agregado
        by_status = {}
        for r in rows:
            s = r.get("status", "unknown")
            by_status[s] = by_status.get(s, 0) + 1

        max_lag_h = 0
        if rows:
            max_lag_h = round(max(int(r.get("lag_seconds") or 0) for r in rows) / 3600.0, 1)

        return {
            "count": len(rows),
            "max_lag_hours": max_lag_h,
            "by_status": by_status,
            "chunks": rows,
        }
    except Exception as e:
        logger.error(f"❌ [GAP A] Error en /admin/chunks/stuck: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/chunks/dead-lettered")
def api_admin_chunks_dead_lettered(
    request: Request,
    limit: int = 100,
    window_hours: Optional[int] = None,
):
    """[P1-2] Inspección operacional: lista chunks marcados con `dead_lettered_at`.

    Complementa a `/admin/chunks/stuck`: aquel filtra chunks atrasados en estados
    activos (pending/stale/processing/failed) por lag; este expone los chunks que
    YA fueron escalados a dead-letter por `_escalate_unrecoverable_chunk` y para
    los que el sistema no intentará más generaciones automáticas — el plan del
    usuario está bloqueado hasta acción manual de soporte (regenerar el plan,
    actualizar nevera, etc).

    Parámetros opcionales:
      - `limit` (default 100): cap de filas devueltas.
      - `window_hours`: si se pasa, filtra `dead_lettered_at > NOW() - INTERVAL Nh`.
        Útil para correlacionar con la alerta `dead_lettered_chunks_recent` que
        usa la misma ventana (`CHUNK_DEAD_LETTER_ALERT_WINDOW_HOURS`).

    Requiere `Authorization: Bearer <CRON_SECRET>`.
    """
    _verify_admin_token(request.headers.get("authorization"))
    from db_core import execute_sql_query
    try:
        params: list = []
        sql = """
            SELECT
                q.id,
                q.user_id,
                q.meal_plan_id,
                q.week_number,
                q.days_offset,
                q.days_count,
                q.status,
                q.attempts,
                q.dead_lettered_at,
                q.dead_letter_reason,
                q.learning_metrics,
                q.pipeline_snapshot->'_pantry_pause_reason' AS pantry_pause_reason,
                EXTRACT(EPOCH FROM (NOW() - q.dead_lettered_at))::int AS dead_seconds,
                (mp.plan_data->>'total_days_requested')::int AS total_days_requested,
                jsonb_array_length(COALESCE(mp.plan_data->'days', '[]'::jsonb)) AS days_generated
            FROM plan_chunk_queue q
            LEFT JOIN meal_plans mp ON mp.id = q.meal_plan_id
            WHERE q.dead_lettered_at IS NOT NULL
        """
        if window_hours is not None and window_hours > 0:
            sql += " AND q.dead_lettered_at > NOW() - make_interval(hours => %s)"
            params.append(int(window_hours))
        sql += " ORDER BY q.dead_lettered_at DESC LIMIT %s"
        params.append(int(limit))

        rows = execute_sql_query(sql, tuple(params)) or []

        # Resumen agregado por reason (mismo shape que la alerta de cron).
        by_reason: dict = {}
        affected_users: set = set()
        affected_plans: set = set()
        for r in rows:
            reason = str(r.get("dead_letter_reason") or "unknown")
            by_reason[reason] = by_reason.get(reason, 0) + 1
            if r.get("user_id"):
                affected_users.add(str(r["user_id"]))
            if r.get("meal_plan_id"):
                affected_plans.add(str(r["meal_plan_id"]))

        return {
            "count": len(rows),
            "window_hours": window_hours,
            "affected_users": len(affected_users),
            "affected_plans": len(affected_plans),
            "by_reason": by_reason,
            "chunks": rows,
        }
    except Exception as e:
        logger.error(f"❌ [P1-2] Error en /admin/chunks/dead-lettered: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/chunk_deferrals/{user_id}")
def api_admin_chunk_deferrals(
    user_id: str,
    request: Request,
    window_hours: int = 48,
    limit: int = 100,
):
    """[P1-3] Inspección operacional: lista deferrals recientes de un usuario.

    Cada vez que el learning gate rechaza un chunk porque su día previo aún no
    concluyó, se registra una fila en `chunk_deferrals`. Acumulación alta sobre
    el mismo (meal_plan_id, week_number) suele indicar TZ desalineada en el
    perfil del usuario o `_plan_start_date` con offset incorrecto.

    Devuelve: lista de deferrals en la ventana, conteo agregado por
    (meal_plan_id, week_number, reason) y el deferral más reciente para
    diagnóstico rápido.

    Requiere Authorization: Bearer <CRON_SECRET>.
    """
    _verify_admin_token(request.headers.get("authorization"))
    from db_core import execute_sql_query
    try:
        rows = execute_sql_query(
            """
            SELECT id, user_id::text AS user_id, meal_plan_id::text AS meal_plan_id,
                   week_number, reason, days_until_prev_end, created_at
            FROM chunk_deferrals
            WHERE user_id = %s::uuid
              AND created_at > NOW() - make_interval(hours => %s)
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, int(window_hours), int(limit)),
        ) or []

        # Agregado por (meal_plan_id, week_number, reason) para detectar patrones.
        agg: dict = {}
        for r in rows:
            key = (r.get("meal_plan_id"), r.get("week_number"), r.get("reason"))
            agg[key] = agg.get(key, 0) + 1
        by_plan_week_reason = [
            {"meal_plan_id": k[0], "week_number": k[1], "reason": k[2], "count": v}
            for k, v in sorted(agg.items(), key=lambda kv: -kv[1])
        ]

        return {
            "user_id": user_id,
            "window_hours": window_hours,
            "total": len(rows),
            "by_plan_week_reason": by_plan_week_reason,
            "deferrals": rows,
        }
    except Exception as e:
        logger.error(f"❌ [P1-3] Error en /admin/chunk_deferrals/{user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{plan_id}/regen-degraded")
def api_regen_degraded_chunks(plan_id: str, verified_user_id: Optional[str] = Depends(verify_api_quota)):
    """[GAP C] Regenera chunks completados en modo degradado (shuffle/edge/emergency)
    creando nuevos chunks pendientes que sobrescribirán los días afectados.

    Útil cuando el LLM volvió a estar disponible y el usuario quiere mejorar la calidad
    de un plan que se generó parcialmente con Smart Shuffle. Idempotente: si no hay
    chunks degradados completados, no hace nada.
    """
    from db_core import execute_sql_query, execute_sql_write
    import json
    try:
        # 1. Validar ownership del plan
        plan_row = execute_sql_query(
            "SELECT user_id, plan_data FROM meal_plans WHERE id = %s",
            (plan_id,), fetch_one=True
        )
        if not plan_row:
            raise HTTPException(status_code=404, detail="Plan no encontrado")
        if verified_user_id and str(plan_row["user_id"]) != str(verified_user_id):
            raise HTTPException(status_code=403, detail="No autorizado")

        # 2. Buscar chunks degradados completados que tengan snapshot recuperable
        degraded_chunks = execute_sql_query("""
            SELECT id, week_number, days_offset, days_count, pipeline_snapshot, user_id
            FROM plan_chunk_queue
            WHERE meal_plan_id = %s
              AND status = 'completed'
              AND quality_tier IN ('shuffle', 'edge', 'emergency')
              AND pipeline_snapshot::text != '{}'
            ORDER BY week_number ASC
        """, (plan_id,)) or []

        if not degraded_chunks:
            return {
                "success": True,
                "regenerated": 0,
                "message": "No hay chunks degradados con snapshot recuperable. Si pasaron >48h, los snapshots ya fueron purgados."
            }

        # 3. Re-encolar como pending sin _degraded para que el worker use el LLM
        regenerated = 0
        for ch in degraded_chunks:
            snap = ch["pipeline_snapshot"]
            if isinstance(snap, str):
                snap = json.loads(snap)
            snap.pop("_degraded", None)

            execute_sql_write("""
                UPDATE plan_chunk_queue
                SET status = 'pending',
                    attempts = 0,
                    quality_tier = NULL,
                    pipeline_snapshot = %s::jsonb,
                    execute_after = NOW(),
                    escalated_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
            """, (json.dumps(snap, ensure_ascii=False), ch["id"]))
            regenerated += 1

        # 4. Marcar plan como partial para que el frontend retome polling
        execute_sql_write("""
            UPDATE meal_plans
            SET plan_data = jsonb_set(plan_data, '{generation_status}', '"partial"'),
                updated_at = NOW()
            WHERE id = %s
        """, (plan_id,))

        return {
            "success": True,
            "regenerated": regenerated,
            "message": f"{regenerated} chunks degradados re-encolados. Procesarán en el próximo tick del worker."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [GAP C] Error en /regen-degraded: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/metrics")
def api_admin_metrics(
    request: Request,
    days: int = 7,
):
    """[GAP G] Agregados observacionales del pipeline de chunks.

    Responde:
      - totals por quality_tier
      - % degraded (shuffle+edge+emergency)
      - avg/p50/p95 duration_ms (usando percentile_cont)
      - learning quality: avg repeat_pct, count violations
      - top error messages (rate)
    """
    _verify_admin_token(request.headers.get("authorization"))
    from db_core import execute_sql_query
    try:
        interval_str = f"{int(days)} days"

        tier_row = execute_sql_query(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE quality_tier = 'llm') AS llm,
                COUNT(*) FILTER (WHERE quality_tier = 'shuffle') AS shuffle,
                COUNT(*) FILTER (WHERE quality_tier = 'edge') AS edge,
                COUNT(*) FILTER (WHERE quality_tier = 'emergency') AS emergency,
                COUNT(*) FILTER (WHERE quality_tier = 'error') AS errors,
                COUNT(*) FILTER (WHERE was_degraded) AS degraded
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - %s::interval
            """,
            (interval_str,), fetch_one=True,
        ) or {}

        total = int(tier_row.get("total") or 0)
        degraded = int(tier_row.get("degraded") or 0)
        errors = int(tier_row.get("errors") or 0)
        degraded_pct = round((degraded / total) * 100.0, 2) if total else 0.0
        error_pct = round((errors / total) * 100.0, 2) if total else 0.0

        perf_row = execute_sql_query(
            """
            SELECT
                ROUND(AVG(duration_ms)::numeric, 0) AS avg_ms,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms)::int AS p50_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms)::int AS p95_ms,
                ROUND(AVG(lag_seconds)::numeric, 0) AS avg_lag_s,
                ROUND(AVG(retries)::numeric, 2) AS avg_retries
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - %s::interval
              AND quality_tier != 'error'
            """,
            (interval_str,), fetch_one=True,
        ) or {}

        learning_row = execute_sql_query(
            """
            SELECT
                ROUND(AVG(learning_repeat_pct)::numeric, 2) AS avg_repeat_pct,
                SUM(rejection_violations) AS rejection_violations_total,
                SUM(allergy_violations) AS allergy_violations_total,
                COUNT(*) FILTER (WHERE rejection_violations > 0) AS chunks_with_rej_violations,
                COUNT(*) FILTER (WHERE allergy_violations > 0) AS chunks_with_alg_violations
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - %s::interval
              AND learning_repeat_pct IS NOT NULL
            """,
            (interval_str,), fetch_one=True,
        ) or {}

        top_errors = execute_sql_query(
            """
            SELECT LEFT(error_message, 200) AS error_prefix, COUNT(*) AS cnt
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - %s::interval
              AND error_message IS NOT NULL
            GROUP BY LEFT(error_message, 200)
            ORDER BY cnt DESC
            LIMIT 10
            """,
            (interval_str,),
        ) or []

        # [P1-6] Learning loss: agrega events `learning_rebuild_failed` desde
        # `chunk_lesson_telemetry`. Sin esto, una racha de chunks con
        # _last_chunk_learning corrupto degrada silenciosamente el aprendizaje
        # continuo y el equipo de operaciones se entera tarde (vía quejas de
        # usuarios sobre planes repetitivos).
        # Best-effort: si la tabla no existe (deploy híbrido) o el SELECT falla,
        # devolvemos zeros/empty en lugar de propagar la excepción.
        learning_loss_row = {}
        learning_loss_by_reason = []
        proxy_ratio_row = {}
        try:
            learning_loss_row = execute_sql_query(
                """
                SELECT
                    COUNT(*)::int AS total_events,
                    COUNT(DISTINCT meal_plan_id)::int AS plans_with_loss,
                    COUNT(DISTINCT user_id)::int AS users_with_loss
                FROM chunk_lesson_telemetry
                WHERE created_at > NOW() - %s::interval
                  AND event = 'learning_rebuild_failed'
                """,
                (interval_str,), fetch_one=True,
            ) or {}
            learning_loss_by_reason = execute_sql_query(
                """
                SELECT COALESCE(metadata->>'reason', 'unknown') AS reason,
                       COUNT(*)::int AS cnt
                FROM chunk_lesson_telemetry
                WHERE created_at > NOW() - %s::interval
                  AND event = 'learning_rebuild_failed'
                GROUP BY COALESCE(metadata->>'reason', 'unknown')
                ORDER BY cnt DESC
                """,
                (interval_str,),
            ) or []
            # [P1-7] Ratio de aprendizaje basado en proxy/synthesis vs user_logs en
            # el lifetime acumulado de planes ACTIVOS. Si crece, indica que muchos
            # usuarios no logean comidas y el sistema cae a inferir desde inventario.
            # No es un fallo (tenemos pause vía CHUNK_MAX_LIFETIME_PROXY_RATIO en el
            # gate) pero sí es señal operacional: marketing/UX podría querer
            # intervenir con onboarding sobre "registra tus comidas".
            proxy_ratio_row = execute_sql_query(
                """
                SELECT
                    AVG((plan_data->'_lifetime_lessons_summary'->>'_lifetime_proxy_ratio')::float)::float AS avg_ratio,
                    MAX((plan_data->'_lifetime_lessons_summary'->>'_lifetime_proxy_ratio')::float)::float AS max_ratio,
                    COUNT(*) FILTER (
                        WHERE (plan_data->'_lifetime_lessons_summary'->>'_lifetime_proxy_ratio')::float > 0.5
                    )::int AS plans_above_50pct,
                    COUNT(*) FILTER (
                        WHERE (plan_data->'_lifetime_lessons_summary'->>'_lifetime_proxy_ratio') IS NOT NULL
                    )::int AS plans_with_ratio,
                    COUNT(DISTINCT user_id)::int AS users_with_ratio
                FROM meal_plans
                WHERE created_at > NOW() - %s::interval
                  AND plan_data->'_lifetime_lessons_summary' IS NOT NULL
                """,
                (interval_str,), fetch_one=True,
            ) or {}
        except Exception as _learning_loss_err:
            # Telemetría es secundaria al endpoint; no fallamos el response.
            logger.warning(
                f"[P1-6/P1-7] /admin/metrics no pudo agregar learning_loss/proxy_ratio "
                f"(¿tabla chunk_lesson_telemetry missing o schema regresivo?): {_learning_loss_err}"
            )

        return {
            "window_days": int(days),
            "total_chunks": total,
            "tiers": {
                "llm": int(tier_row.get("llm") or 0),
                "shuffle": int(tier_row.get("shuffle") or 0),
                "edge": int(tier_row.get("edge") or 0),
                "emergency": int(tier_row.get("emergency") or 0),
                "error": errors,
            },
            "degraded_pct": degraded_pct,
            "error_pct": error_pct,
            "perf": {
                "avg_ms": int(perf_row.get("avg_ms") or 0),
                "p50_ms": int(perf_row.get("p50_ms") or 0),
                "p95_ms": int(perf_row.get("p95_ms") or 0),
                "avg_lag_seconds": int(perf_row.get("avg_lag_s") or 0),
                "avg_retries": float(perf_row.get("avg_retries") or 0.0),
            },
            "learning": {
                "avg_repeat_pct": float(learning_row.get("avg_repeat_pct") or 0.0),
                "rejection_violations_total": int(learning_row.get("rejection_violations_total") or 0),
                "allergy_violations_total": int(learning_row.get("allergy_violations_total") or 0),
                "chunks_with_rejection_violations": int(learning_row.get("chunks_with_rej_violations") or 0),
                "chunks_with_allergy_violations": int(learning_row.get("chunks_with_alg_violations") or 0),
            },
            # [P1-6] Pérdida de learning durante el rebuild del chunk previo —
            # señal de corrupción en plan_chunk_queue.learning_metrics o de
            # blips de DB que silentemente degradan el aprendizaje continuo.
            "learning_loss": {
                "total_events": int(learning_loss_row.get("total_events") or 0),
                "plans_with_loss": int(learning_loss_row.get("plans_with_loss") or 0),
                "users_with_loss": int(learning_loss_row.get("users_with_loss") or 0),
                "by_reason": learning_loss_by_reason,
            },
            # [P1-7] Provenance del aprendizaje agregado: cuánto del lifetime
            # depende de proxy/synthesis (low signal) vs logs reales del usuario.
            # Ratios altos (>50%) sugieren que el aprendizaje del usuario está
            # dominado por inferencias en lugar de datos concretos.
            "learning_provenance": {
                "avg_proxy_ratio": float(proxy_ratio_row.get("avg_ratio") or 0.0),
                "max_proxy_ratio": float(proxy_ratio_row.get("max_ratio") or 0.0),
                "plans_above_50pct_proxy": int(proxy_ratio_row.get("plans_above_50pct") or 0),
                "plans_with_ratio": int(proxy_ratio_row.get("plans_with_ratio") or 0),
                "users_with_ratio": int(proxy_ratio_row.get("users_with_ratio") or 0),
            },
            "top_errors": top_errors,
        }
    except Exception as e:
        logger.error(f"❌ [GAP G] Error en /admin/metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/chunks/{chunk_id}/escalate")
def api_admin_escalate_chunk(chunk_id: str, request: Request):
    """[GAP A] Forzar escalado/pickup inmediato de un chunk concreto."""
    _verify_admin_token(request.headers.get("authorization"))
    from db_core import execute_sql_write
    try:
        res = execute_sql_write(
            """
            UPDATE plan_chunk_queue
            SET status = 'pending',
                escalated_at = NOW(),
                execute_after = NOW(),
                attempts = 0,
                updated_at = NOW()
            WHERE id = %s AND status IN ('pending', 'stale', 'failed')
            RETURNING id, meal_plan_id, week_number
            """,
            (chunk_id,),
            returning=True,
        )
        if not res:
            raise HTTPException(status_code=404, detail="Chunk no encontrado o ya en processing")
        return {"success": True, "chunk": res[0]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [GAP A] Error en /admin/chunks/escalate: {e}")
        raise HTTPException(status_code=500, detail=str(e))
