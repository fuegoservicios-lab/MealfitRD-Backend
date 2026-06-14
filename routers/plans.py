from fastapi import APIRouter, Body, Depends, HTTPException, BackgroundTasks, Request, Response
from fastapi.responses import StreamingResponse
from error_utils import safe_error_detail
from typing import Optional, Any
import logging
import traceback
import os
import hmac
import threading
import asyncio
import json as _json
import math as _math
import time as _time
from datetime import datetime, timezone, timedelta

# Importaciones relativas del entorno
from auth import get_verified_user_id, verify_api_quota
from db import (
    get_user_likes, get_active_rejections, get_or_create_session,
    save_message, update_user_health_profile, update_user_health_profile_atomic, log_api_usage, get_latest_meal_plan,
    get_latest_meal_plan_with_id, update_meal_plan_data, insert_like
)
from memory_manager import build_memory_context, summarize_and_prune
from agent import analyze_preferences_agent, swap_meal
from graph_orchestrator import (
    run_plan_pipeline,
    arun_plan_pipeline,
    _strip_untrusted_internal_keys,
    _enforce_days_to_generate_cap,
    # [P1-A · 2026-05-08] `_env_int` para que `MEALFIT_PERISHABLE_*` (consumido
    # por el endpoint de restock per-item) se auto-registre en `_KNOBS_REGISTRY`
    # en lugar de bypass-ear vía `os.environ.get(...)` raw.
    _env_int,
    # [P2-WATER-RETRY-NO-JITTER · 2026-05-24] `_env_float` para los knobs
    # `MEALFIT_WATER_RETRY_BACKOFF_BASE_S` / `MEALFIT_WATER_RETRY_JITTER_MAX_S`
    # del helper `_execute_with_retry` (auto-registro en KNOBS_REGISTRY).
    _env_float,
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
from services import _save_plan_and_track_background, _process_swap_rejection_background, save_partial_plan_get_id, _persist_plan_persist_failed_alert
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

# [P3-PDF-POLISH-4-A · 2026-05-14] Rate limit per-endpoint para los dos
# endpoints del flujo PDF lista-de-compras que NO usan `verify_api_quota`:
#
#   1. `/recalculate-shopping-list`: muta `plan_data` (UPDATE jsonb bajo
#      advisory lock) + 3× `get_shopping_list_delta` (compute heavy) +
#      coherence guard. Ningún costo LLM, por lo que paywall no aplica;
#      pero un cliente autenticado podía spammear POSTs y exhaustar la
#      DB pool del mismo usuario (lost-update está cubierto por el
#      advisory lock, pero el work seguía consumiendo workers).
#
#   2. `/telemetry/pdf-stale-fallback`: INSERT a `pipeline_metrics`
#      fire-and-forget desde el frontend. Sin tope un atacante autenticado
#      podía bloat la tabla — el burst-alert cron P2-SHOPPING-3 lo
#      detecta, pero defense-in-depth previene el bloat en primer lugar.
#
# Buckets independientes (singletons a nivel módulo). Mismo bucket-key
# strategy que `_PLAN_GEN_LIMITER`: prefer `verified_user_id`, fall back a
# `ip:<host>` para anon (vía RateLimiter P1-6).
_RECALC_LIMITER = RateLimiter(max_calls=20, period_seconds=60)
_PDF_TELEMETRY_LIMITER = RateLimiter(max_calls=30, period_seconds=60)

# [P2-GUEST-LLM-RATELIMIT · 2026-05-30] `/swap-meal` y `/recipe/expand` invocan
# Gemini pero solo tenían `Depends(verify_api_quota)`. Para un GUEST (no
# autenticado) el paywall mensual NO aplica → un atacante podía martillar
# cualquiera de los dos endpoints sin tope, amplificando costo de LLM contra
# nuestra cuota. Estos limiters bucketean por `verified_user_id` o `ip:<host>`
# (RateLimiter P1-6), capando el burst per-IP de anon sin estorbar el uso
# legítimo (un usuario real no hace >12 swaps / >15 expansiones por minuto). Se
# AÑADEN a `verify_api_quota` (paywall) — mismo patrón que `/analyze` con
# `_PLAN_GEN_LIMITER`. Tooltip-anchor: P2-GUEST-LLM-RATELIMIT.
_SWAP_LIMITER = RateLimiter(max_calls=12, period_seconds=60)
_EXPAND_LIMITER = RateLimiter(max_calls=15, period_seconds=60)

# [P1-16] Registry global de session_ids cancelados durante la generación.
# Cuando el usuario clickea "Cancelar" en el frontend, el SSE se aborta
# del lado cliente — pero ANTES de P1-16 el pipeline backend seguía
# corriendo hasta terminar el día actual, persistía el plan en DB, y el
# usuario veía el plan aparecer 30s después vía Realtime UPDATE de
# `meal_plans`. UX confuso + cuota de LLM consumida innecesariamente.
#
# AHORA, `POST /api/plans/cancel` agrega el session_id a este set. Los
# nodes del pipeline (especialmente los puntos de await dentro de
# `generate_days_parallel_node` y los hedges) verifican cooperativamente
# vía `is_session_cancelled(session_id)` y abortan con
# `asyncio.CancelledError`.
#
# El set vive en memoria (in-process). Es aceptable porque:
#   - Una request de cancelación llega al MISMO worker que está corriendo
#     el pipeline (sticky por session_id en deployments con load balancer
#     hash-by-cookie/header); para deployments sin sticky, el cancel es
#     best-effort (peor caso: el otro worker sigue corriendo el pipeline,
#     mismo comportamiento que pre-P1-16).
#   - El set se limpia tras cada flush periódico para evitar leak.
#
# Thread-safety: las operaciones add/discard sobre Python set son atómicas
# bajo GIL para el caso single-element. Para iteraciones complejas usar
# el `_PLAN_CANCEL_LOCK`.
import threading as _p116_threading
_PLAN_CANCEL_REGISTRY: set = set()
_PLAN_CANCEL_LOCK = _p116_threading.Lock()


def _cancel_session(session_id: str) -> bool:
    """Marca un session_id como cancelado. Idempotente: si ya estaba
    marcado, retorna False (no hace nada)."""
    if not session_id or not isinstance(session_id, str):
        return False
    with _PLAN_CANCEL_LOCK:
        if session_id in _PLAN_CANCEL_REGISTRY:
            return False
        _PLAN_CANCEL_REGISTRY.add(session_id)
        return True


def is_session_cancelled(session_id) -> bool:
    """Chequeo cooperativo. Llamar en puntos de await del pipeline para
    abortar temprano si el frontend pidió cancelar. Retorna False para
    session_id None/empty/inválido (defensa: no abortar pipelines que
    el cliente no identificó)."""
    if not session_id or not isinstance(session_id, str):
        return False
    return session_id in _PLAN_CANCEL_REGISTRY


def _clear_cancelled_session(session_id: str) -> None:
    """Limpia el flag tras abortar el pipeline o después de un timeout
    razonable. Llamar desde el `finally` de `arun_plan_pipeline` para
    evitar que el set crezca sin límite durante la vida del proceso."""
    if not session_id:
        return
    with _PLAN_CANCEL_LOCK:
        _PLAN_CANCEL_REGISTRY.discard(session_id)


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
    # [P0-FORM-6] Cierre de drift entre `REQUIRED_FORM_FIELDS` (frontend en
    # `formValidation.js`) y este tuple. El frontend gateaba estos 7 como
    # required (botón "Siguiente" bloqueado, asterisco rojo, chips), pero el
    # backend los aceptaba como `None`/`""` sin error. Inconsistencia real:
    # un cliente legacy / hidratación rota / scraper que omitiera el campo
    # entraba al pipeline; uno que lo enviara con un valor bogus era rechazado
    # por `_NON_CRITICAL_ENUM_VALIDATIONS` con 422 — "ausente pasa silencioso,
    # presente-bogus rechaza". Cualquier path que evitara el wizard producía
    # plan degradado (señales de timing/conducta vacías → LLM defaultea
    # silenciosamente sin telemetría).
    #
    # `scheduleType`, `cookingTime`, `budget`, `sleepHours`, `stressLevel`:
    #   tienen enum estricto en `_NON_CRITICAL_ENUM_VALIDATIONS` (presente
    #   con valor inválido → 422 hoy). Subirlos a required-presence cierra el
    #   hueco "ausente → silencioso".
    #
    # `dislikes`, `struggles`:
    #   arrays multi-select con sentinel "Ninguno"/"Ninguna" (length=1 cuenta
    #   como answer válida). NO los añadimos a `_REQUIRED_NONEMPTY_ARRAY_FIELDS`
    #   porque downstream `[]` no es safety-crítico (a diferencia de allergies);
    #   sí exigimos PRESENCIA (rechazamos None/"" pero `[]` pasa).
    #
    # `dietType`: deliberadamente OUT. `_DIET_TYPE_LEGACY_ACCEPTED` documenta
    # variantes ES históricas en `health_profile.dietType` ("Omnívora",
    # "vegetariana", etc.). Hacerlo required-presence rompería rehidratación
    # de perfiles legacy sin beneficio de safety (downstream
    # `_get_fast_filtered_catalogs` defaultea a balanced = catálogo completo,
    # benigno).
    "scheduleType", "cookingTime", "budget", "sleepHours", "stressLevel",
    "dislikes", "struggles",
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
#   - `sleepHours`, `stressLevel` → [P2-AUDIT-5 · 2026-05-10] hints textuales
#      inyectados al planner + day generator vía `build_sleep_stress_context`
#      (prompts/plan_generator.py). Bloque emitido SOLO si el valor es
#      accionable (sueño 7-8h / estrés Bajo|Moderado son rangos estándar y
#      no contaminan el prompt). Antes de P2-AUDIT-5 este comment afirmaba
#      "solo hints textuales al LLM" pero NINGÚN consumer leía los campos —
#      promesa rota del wizard cerrada al inyectar ahora como hint real.
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
_BUDGET_ENUM        = frozenset({"low", "medium", "high", "unlimited", "custom"})
_GROCERY_DURATION_ENUM = frozenset({"weekly", "biweekly", "monthly"})
_SLEEP_HOURS_ENUM   = frozenset({"< 6 horas", "6-7 horas", "7-8 horas", "> 8 horas"})
_STRESS_LEVEL_ENUM  = frozenset({"Bajo", "Moderado", "Alto", "Muy Alto"})

# Tabla de validación uniforme: field → (enum, normalizer, accepted_label).
# El normalizer aplica la transformación que el campo permite antes del match
# (lower/strip o exact). `accepted_label` es lo que se muestra en el 422.
_NON_CRITICAL_ENUM_VALIDATIONS = (
    ("scheduleType",    _SCHEDULE_TYPE_ENUM,    True,  "standard|night_shift|variable"),
    ("cookingTime",     _COOKING_TIME_ENUM,     True,  "none|30min|1hour|plenty"),
    ("budget",          _BUDGET_ENUM,           True,  "low|medium|high|unlimited|custom"),
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
#
# [P1-FORM-14] CONTRATO MULTI-SITE:
#   1. `frontend/src/config/formValidation.js` → `SUPPLEMENTS` (SSOT cliente).
#   2. `frontend/src/components/assessment/questions/InteractiveQuestions.jsx`
#      → `SUPPLEMENT_META` (mapping val→{label, emoji}; invariante dev-mode
#      verifica metadata cubre 1:1 a SUPPLEMENTS).
#   3. Este `_SUPPLEMENT_ENUM` (gate API boundary).
#   4. `backend/constants.py` → `SUPPLEMENT_NAMES` (mapping val→nombre legible
#      para el prompt del LLM).
# El test `backend/test_p1_form_14_supplements_sync.py` parsea los 4 sites y
# falla en CI si detecta drift entre cualquiera de ellos. Si añades un
# suplemento nuevo (ej. "ashwagandha"), DEBE actualizarse en los 4 lugares.
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
    # [P1-FORM-12] Tamaño del hogar para escalar lista de compras. Cap alto
    # en 12 para cubrir familias extendidas / hogares múltiples sin bloquear
    # casos legítimos. ANTES no se validaba: `householdSize=999` o `="abc"`
    # pasaban al `shopping_calculator` que multiplicaba cantidades x999 →
    # lista de compras absurda + posible OOM en agregación de items.
    # [P3-5 · 2026-05-08] Paridad con `frontend/src/config/formValidation.js
    # BIO_RANGES.household` auditada por `test_p3_5_bio_ranges_parity.py`.
    # Si se bumpea aquí, subir frontend simultáneamente o el form rechazará
    # valores válidos.
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
    household_validated = None
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
        else:
            household_validated = household

    # --- [P1-15] Guard composicional householdSize × groceryDuration ---
    # El frontend (`InteractiveQuestions.jsx → QHousehold`) expone chips
    # 1..6 personas + 3 ciclos (weekly=7d / biweekly=15d / monthly=30d).
    # Cap individual de householdSize es 12 (familias extendidas legítimas)
    # pero un cliente legacy/scraper enviando `householdSize=12,
    # groceryDuration='monthly'` produce escalado de hasta 12 × 4 ciclos
    # de 7d ≈ 360× del plato base. Riesgos:
    #   - Lista de compras absurda (cientos de libras de cada ingrediente).
    #   - Posible OOM en `aggregate_and_deduct_shopping_list` con miles
    #     de líneas a humanizar/clasificar.
    #   - Chunk timeouts del LLM por tamaño desproporcionado del prompt.
    # Hasta que el wizard exponga chips 7+ explícitamente con UX adecuada
    # (warning sobre presupuesto, validación realista de almacenamiento),
    # rechazamos la combinación con 422 accionable. El usuario puede
    # mantener `householdSize=12` con `weekly`/`biweekly` o reducir a
    # `householdSize ≤ 6` con `monthly`.
    if household_validated is not None and household_validated > 6:
        grocery_raw = data.get("groceryDuration")
        if isinstance(grocery_raw, str) and grocery_raw.strip().lower() == "monthly":
            errors.append({
                "field": "householdSize",
                "value": household_validated,
                "accepted_range": [1, 6],
                "unit": "personas con groceryDuration='monthly'",
                "reason": (
                    "P1-15: householdSize > 6 + groceryDuration='monthly' "
                    "produce escalado de hasta ~360× que satura el shopping "
                    "calculator. Reducir household a ≤ 6 o usar "
                    "groceryDuration='weekly'/'biweekly'."
                ),
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
    update_reason: Optional[str] = None,
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

    [P1-PANTRY-GUARD-INITIAL-SKIP · 2026-05-18] El mismo corto-circuito aplica
    cuando la nevera tiene MENOS de `PANTRY_GUARD_MIN_ITEMS` (default 10). Razón:
    en generación inicial o regeneración manual, la lista de compras del plan
    ES LA QUE DEFINE el inventario futuro — validar contra una nevera casi vacía
    rechaza ingredientes legítimos del plan nuevo y dispara retries inútiles que
    consumen LLM quota sin valor. El guard estricto solo es útil cuando ya existe
    un ciclo de compras vivo (nevera poblada) y un swap/refill DEBE respetar lo
    que el user compró. Tooltip-anchor: P1-PANTRY-GUARD-INITIAL-SKIP.

    [P1-PANTRY-GUARD-REGEN-SKIP · 2026-05-18] Skip TOTAL cuando el request es una
    regeneración explícita (`update_reason` set). Razón arquitectónica: el
    threshold de PANTRY_GUARD_MIN_ITEMS es un proxy POOR de "intent de regen";
    cuando el user clickea "Renovar Plan Actual" / "Actualizar plan" con nevera
    llena (≥10 items), su intent es CAMBIAR la comida — validar contra la nevera
    vieja rechaza el plan nuevo en sus propios méritos. Reasons que pueden venir
    del frontend (`useRegeneratePlan.js` + Dashboard modal Actualizar): `variety`,
    `time`, `budget`, `cravings`, `weekend`, `similar`, `dislike`, o cualquier
    string truthy. La presencia del campo es la señal — el valor específico se
    persiste por separado vía `_persist_global_update_reason` para learning.
    El single-meal swap NO toca este path (va por `/swap-meal/persist`). El
    rolling refill chunks 2-4 conserva su validación pantry-aware vía Smart
    Shuffle (`_filter_days_by_fresh_pantry`) — esa SÍ es "update dishes" y
    debe respetar lo comprado. Tooltip-anchor: P1-PANTRY-GUARD-REGEN-SKIP.
    """
    if update_reason:
        _user_label_regen = actual_user_id or "guest"
        logger.info(
            f"⏭️ [{transport_label}/SKIP-REGEN] Pantry guard skip user={_user_label_regen}: "
            f"update_reason={update_reason!r} indica regen explícita (Renovar/Actualizar). "
            f"Plan nuevo DEFINE la lista de compras, no la nevera previa. "
            f"Plan se entrega sin retries pantry."
        )
        return result
    if not pantry_ingredients:
        return result
    from constants import PANTRY_GUARD_MIN_ITEMS as _PANTRY_MIN
    if len(pantry_ingredients) < _PANTRY_MIN:
        _user_label_skip = actual_user_id or "guest"
        logger.info(
            f"⏭️ [{transport_label}/SKIP] Pantry guard skip para user={_user_label_skip}: "
            f"nevera tiene {len(pantry_ingredients)} items (<{_PANTRY_MIN} threshold). "
            f"Plan inicial define lista de compras, no al revés. "
            f"Plan se entrega sin retries pantry."
        )
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
            # [P0-A] Propagar missing_list desde el audit al result para que
            # `save_partial_plan_get_id` (services.py:139, `{**plan_data, ...}`)
            # lo persista a `meal_plans.plan_data._pantry_supplement_required`.
            # `shopping_calculator.get_shopping_list_delta` lo lee desde plan_data
            # y agrega la categoría "🚨 Compra Urgente" al PDF — sin esto, el
            # primer chunk degradado entregaba un plan con flag de aviso pero la
            # lista de compras visible al usuario no incluía los items urgentes.
            _initial_missing = _initial_audit.get("missing_list") or []
            if isinstance(_initial_missing, list) and _initial_missing:
                result["_pantry_supplement_required"] = list(_initial_missing)
            logger.error(
                f"❌ [{transport_label}] Primer chunk para user={_user_label} "
                f"degradado tras {_initial_audit.get('attempts')} intento(s) "
                f"(mode={_initial_audit.get('mode')}, missing={len(_initial_missing)} items). "
                f"Plan se entrega con flag de aviso."
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
    user_id: str,
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
                    user_id=user_id,  # [P0-9] ownership check
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


# [P2-PROD-AUDIT-FOLLOWUP · 2026-05-28] Set de strong-refs a tasks de fallback
# postprocess (SSE done-callback). `asyncio.create_task` SIN guardar referencia
# puede ser garbage-collected mid-flight (gotcha conocido de asyncio) → el
# persist+schedule del fallback se perdería a la mitad, dejando un plan huérfano
# (backstopped por `_sweep_orphan_plans` a 7d, pero evitable). Guardar el ref
# aquí + discard en done-callback previene el GC y da un handle observable.
# Tooltip-anchor: P2-FALLBACK-TASK-TRACKED.
_BG_SSE_FALLBACK_TASKS: set = set()


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
                # [P1-2] Atomic write con mutator que MERGEA `hp_data` ON TOP
                # del estado actual bajo FOR UPDATE. Antes era un full-overwrite
                # destructivo del JSONB column: pasar `hp_data` (= form payload
                # + tz) al legacy `update_user_health_profile` reemplazaba el
                # JSON ENTERO, wipeando `weight_history`, `frictions`,
                # `reflection_history`, `lifetime_lessons_history`,
                # `pipeline_score_history`, `grocery_cycle`, `rejection_patterns`,
                # etc. cada vez que un usuario regeneraba un plan. Sumado al
                # lost-update bajo concurrencia (2 tabs regenerando, cron
                # rolling-refill paralelo), el meta-learning se erosionaba en
                # cada generación. La migración a atomic + mutator-merge
                # cierra AMBOS bugs: (1) preservación de campos que no son
                # del form, (2) serialización de writers concurrentes.
                def _post_pipeline_mutator(_hp):
                    _hp.update(hp_data)
                    return None

                update_user_health_profile_atomic(actual_user_id, _post_pipeline_mutator)
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
            log_api_usage(actual_user_id, "llm_analyze")
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
        # actual_user_id garantizado no-None: use_chunking solo es True cuando
        # user_has_profile (que requiere actual_user_id truthy) — ver L2413/2729.
        assert actual_user_id is not None
        plan_id = save_partial_plan_get_id(
            actual_user_id, result, selected_techniques, total_days_requested,
        )
        if plan_id:
            _seed_chunk1_learning(
                plan_id, result, rejected_meal_names,
                context_label=f"seed_chunk1_{transport_label}",
                user_id=actual_user_id,  # [P0-9] ownership check
            )
            _enqueue_remaining_chunks(
                actual_user_id, plan_id, result,
                data=data, taste_profile=taste_profile,
                memory_ctx=memory_ctx,
                total_days_requested=total_days_requested,
                plan_start_date=plan_start_date,
                tz_offset_mins=tz_offset_mins,
            )
        else:
            # [P2-PLAN-PERSIST-FAILED · 2026-05-30] `save_partial_plan_get_id`
            # devolvió None: el INSERT de meal_plans falló (pool exhaustion,
            # statement_timeout, CHECK I8 meal_plans_complete_requires_days,
            # serialization error...). Pre-fix esto se tragaba en silencio → el
            # caller (SSE generator) marcaba el KV `complete` con plan_id_final=None
            # y emitía el evento `complete`: el usuario veía éxito pero el plan NO
            # existía (historial/dashboard vacíos, chunks 2..N nunca encolados) y
            # NINGÚN system_alert se levantaba. Marcamos un flag que los 3
            # consumidores (sync L2494, SSE L3233, done-callback L2933) propagan
            # como FALLA al usuario (error event / 503 / KV failed), y emitimos el
            # system_alert para visibilidad operacional. Tooltip-anchor: P2-PLAN-PERSIST-FAILED.
            result["_persist_failed"] = True
            logger.error(
                f"🛑 [P2-PLAN-PERSIST-FAILED] save_partial_plan_get_id devolvió None "
                f"(INSERT meal_plans fallido) user={actual_user_id or 'guest'} "
                f"transport={transport_label}. El plan NO se persistió — propagando como error."
            )
            _persist_plan_persist_failed_alert(actual_user_id, f"chunk_insert_failed:{transport_label}")
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


# [P1-AUDIT-NEW-1 · 2026-05-12] `/debug-scaling/{user_id}` ELIMINADO.
# ────────────────────────────────────────────────────────────────────────────
# El endpoint vivía aquí marcado como "TEMPORARY DEBUG (REMOVE AFTER DIAGNOSIS)"
# pero quedó vivo en producción. Tenía DOS defectos críticos:
#
#   1. Sin `Depends(get_verified_user_id)` ni `_verify_admin_token`. Cualquiera
#      con la URL pública del backend podía leer el `plan_data` completo de
#      cualquier `user_id` válido (UUIDs leak vía URLs/PDFs/screenshots).
#
#   2. Fallback IDOR: si el `user_id` solicitado no tenía plan, ejecutaba
#      `SELECT id, user_id, plan_data FROM meal_plans ORDER BY created_at
#      DESC LIMIT 1` y RETORNABA el plan más reciente de OTRO usuario,
#      incluyendo `found_user_id` ajeno → enumerador trivial de UUIDs activos.
#
# Audit cross-codebase (`grep -r debug-scaling`) confirmó CERO callers: ni
# frontend, ni tests, ni scripts internos. La utilidad ad-hoc se reemplazó
# por el script local [backend/check_scaling.py](backend/check_scaling.py)
# que SRE puede ejecutar contra una DB readonly sin exponer un endpoint HTTP.
#
# Si en el futuro se necesita un endpoint admin de inspección de scaling
# (poco probable — la cobertura ya la dan los tests P3-A multiplier_e2e),
# crear handler bajo `/admin/plans/scaling-inspect/{plan_id}` gateado con
# `_verify_admin_token` (mismo patrón que `/admin/chunks/stuck`) y SIN
# fallback cross-user.
#
# Tooltip-anchor: P1-AUDIT-NEW-1-DEBUG-ENDPOINT-REMOVED

from constants import PLAN_CHUNK_SIZE, split_with_absorb

def _user_has_profile(user_id: str) -> bool:
    """Devuelve True si user_id tiene fila en user_profiles. Auto-crea fila mínima si falta."""
    if not user_id:
        return False
    try:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] PostgREST → SQL directo (pool psycopg).
        from db import execute_sql_query, execute_sql_write
        row = execute_sql_query(
            "SELECT 1 AS existe FROM public.user_profiles WHERE id = %s LIMIT 1",
            (user_id,),
            fetch_one=True,
        )
        if row:
            return True
        # Usuario autenticado sin perfil → crear fila mínima para habilitar chunking y FK.
        # ON CONFLICT DO NOTHING preserva la semántica del upsert legacy: este branch
        # solo corre cuando el SELECT no encontró fila, así que bajo race NO pisa un
        # health_profile ya existente.
        from psycopg.types.json import Jsonb
        execute_sql_write(
            "INSERT INTO public.user_profiles (id, health_profile) VALUES (%s, %s) "
            "ON CONFLICT (id) DO NOTHING",
            (user_id, Jsonb({})),
        )
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
                    # [P1-LOCK-1 · 2026-05-10] Bound lock wait + statement timeout
                    # ANTES de cualquier adquisición de lock. Sin esto, una
                    # contención con el chunk worker (T1/T2 en cron_tasks) podía
                    # dejar al caller esperando >90s por el row lock (observado en
                    # logs Postgres prod). Helper lee knobs MEALFIT_PLAN_FOR_UPDATE_*.
                    from db_plans import set_meal_plan_for_update_timeouts
                    set_meal_plan_for_update_timeouts(cursor)

                    # [P2-LOCK-2 · 2026-05-10] Resolver plan_id SIN row lock primero
                    # (cheap idempotent read). Pre-P2-LOCK-2, este SELECT incluía
                    # `FOR UPDATE` y la `acquire_meal_plan_advisory_lock` venía
                    # DESPUÉS — orden inverso al worker (advisory primero, FOR UPDATE
                    # después). Si /shift-plan y worker corrían simultáneos sobre
                    # el mismo plan, cada uno sostenía un recurso esperando el otro
                    # → Postgres detectaba deadlock dentro de `deadlock_timeout` (1s
                    # default) y abortaba uno con `deadlock_detected`. El nuevo orden
                    # (resolver id → advisory → FOR UPDATE) elimina la asimetría.
                    cursor.execute(
                        "SELECT id FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
                        (user_id,)
                    )
                    _id_row = cursor.fetchone()
                    if not _id_row:
                        return {"success": False, "message": "No hay plan activo."}
                    plan_id = _id_row["id"]

                    # [P0-4] Advisory lock 'general' por meal_plan: serializa este shift
                    # contra el merge del worker (cron_tasks._chunk_worker T1+T2) que
                    # también adquiere el mismo lock antes de tocar plan_data. Sin esto,
                    # T2 del worker (que escribe plan_data POSTERIOR a su propio T1 con
                    # un dict en memoria) podía sobrescribir los cambios de /shift-plan
                    # ejecutados entre T1 y T2 — perdiendo el shift y la renumeración
                    # de days. El lock se libera al cerrar la transacción.
                    # [P2-LOCK-2 · 2026-05-10] Movido ANTES del FOR UPDATE para
                    # unificar orden con T1/T2/bg-refill (advisory → FOR UPDATE).
                    from db_plans import acquire_meal_plan_advisory_lock as _p04_acquire_lock
                    _p04_acquire_lock(cursor, plan_id, purpose="general")

                    # Ahora SÍ tomar el row lock. Si el plan fue eliminado entre el
                    # SELECT id de arriba y este FOR UPDATE (race extremadamente rara:
                    # ~ms entre statements), la fila no existe y caemos al return.
                    cursor.execute(
                        "SELECT plan_data FROM meal_plans WHERE id = %s FOR UPDATE",
                        (plan_id,)
                    )
                    plan_record = cursor.fetchone()
                    if not plan_record:
                        return {"success": False, "message": "El plan fue eliminado durante la operación. Reintentá."}
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
                            #
                            # [P1-2] Atomic write con mutator que solo toca
                            # `tz_offset_minutes`/`tzOffset`. El legacy
                            # `update_user_health_profile({"tz_offset_minutes":..., "tzOffset":...})`
                            # hacía full-overwrite del JSONB column —
                            # pasarle ese dict de 2 keys WIPEABA TODO el resto
                            # del health_profile (weight_history, frictions,
                            # lifetime_lessons_history, etc.) cada vez que el
                            # cliente disparaba /shift-plan con tzOffset.
                            # El atomic helper preserva los demás campos bajo
                            # lock y, post-commit, dispara
                            # `_sync_chunk_queue_tz_offsets` automáticamente
                            # cuando detecta cambio de TZ — equivalente al
                            # side-effect del legacy sin la duplicación.
                            _new_tz_int = int(tz_offset)

                            def _tz_mutator(_hp):
                                _hp["tz_offset_minutes"] = _new_tz_int
                                _hp["tzOffset"] = _new_tz_int
                                return None

                            update_user_health_profile_atomic(user_id, _tz_mutator)
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
                    import re as _re_p3_shift
                    try:
                        # [P3-SHIFT-DATEONLY-LOCAL · 2026-05-18] `grocery_start_date`
                        # se persiste en DOS formatos según el path de origen:
                        #   1. "YYYY-MM-DD" date-only — backfill SQL (p0_3_backfill_plan_anchors)
                        #      y plan_data inicial cuando el LLM emite el campo sin TZ.
                        #   2. "YYYY-MM-DDTHH:MM:SS+TZ" timestamp ISO completo — fix
                        #      [GROCERY-START-DATE-TIMESTAMP-FIX 2026-05-06] en
                        #      `_ensure_grocery_start_date` (db_plans.py) cuando el
                        #      LLM NO incluyó el campo.
                        #
                        # Bug pre-fix: ambos formatos pasaban por la rama timestamp.
                        # Para date-only "2026-05-17" en TZ -4 (Santo Domingo,
                        # tz_offset=240): `replace(tzinfo=utc)` lo marca como
                        # 2026-05-17T00:00:00Z, luego `- timedelta(minutes=240)` da
                        # 2026-05-16T20:00:00Z, y `.date()` = 2026-05-16 (¡día
                        # anterior!). Eso infla `days_since_creation` en +1 → el
                        # shift elimina 1 día EXTRA del plan al cruzar la medianoche
                        # local (síntoma reportado 2026-05-18: plan [Dom,Lun,Mar]
                        # quedó como [Lun] solo — eliminó Domingo Y Martes en vez de
                        # solo Domingo).
                        #
                        # Fix: si el formato es date-only ("YYYY-MM-DD"), interpretar
                        # como fecha LOCAL del usuario (sin TZ dance) — SSOT con el
                        # `_parseStartLocal` del frontend (Dashboard.jsx:603, fix
                        # análogo desde 2026-05-06 que el backend nunca espejó).
                        # Si trae timestamp con/sin TZ, mantener la lógica legacy.
                        # Tooltip-anchor: P3-SHIFT-DATEONLY-LOCAL.
                        _is_date_only_shift = (
                            isinstance(start_date_str, str)
                            and _re_p3_shift.match(r'^\d{4}-\d{2}-\d{2}$', start_date_str.strip()) is not None
                        )
                        if _is_date_only_shift:
                            _y_p3, _m_p3, _d_p3 = (int(x) for x in start_date_str.strip().split('-'))
                            from datetime import date as _date_p3
                            start_date = _date_p3(_y_p3, _m_p3, _d_p3)
                            # `start_dt` se usa downstream para `new_start = start_dt +
                            # timedelta(days=days_since_creation)` (línea ~2051).
                            # Construirlo como aware datetime al local-midnight del user
                            # expresado en UTC: equivale a "date local + 0:00" cuando
                            # el caller hace `.isoformat()` — preserva semántica con
                            # tz_offset ya aplicado.
                            start_dt = datetime(_y_p3, _m_p3, _d_p3, tzinfo=timezone.utc) - timedelta(minutes=int(tz_offset))
                        else:
                            start_dt = safe_fromisoformat(start_date_str)
                            if start_dt.tzinfo is None:
                                start_dt = start_dt.replace(tzinfo=timezone.utc)
                            else:
                                start_dt = start_dt.astimezone(timezone.utc)
                            start_dt = start_dt - timedelta(minutes=int(tz_offset))
                            start_date = start_dt.date()

                        # Remove time component
                        today_date = today.date()
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
                        # [P1-HIST-COMPLETE-PROGRESS · 2026-05-31] Preservar los
                        # días que el shift PODA del array vivo, para que el
                        # Historial muestre el progreso COMPLETO (todos los días
                        # generados, incluidos los que ya pasaron). El Dashboard
                        # "Tu Menú" y Recetas siguen usando `days` (ventana
                        # rolling, renumerada 1..N que el chunk worker requiere);
                        # SOLO el Historial lee `_archived_days`. Es aditivo: no
                        # altera el slice ni el renumerado de abajo. Cap defensivo
                        # para no crecer sin límite (a lo sumo el plan completo).
                        if shift_amount > 0:
                            _archived = shifted_data.get("_archived_days")
                            if not isinstance(_archived, list):
                                _archived = []
                            _archived.extend(copy.deepcopy(shifted_days[:shift_amount]))
                            _arch_cap = (total_planned_days if isinstance(total_planned_days, int) and total_planned_days > 0 else 30) + 31
                            shifted_data["_archived_days"] = _archived[-_arch_cap:]
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
                        # [VISUAL CONTINUITY] Plan mensual/quincenal en estado partial.
                        #
                        # ANTES: cualquier `gap visible` (ej. 2/3 días) disparaba aceleración
                        # del siguiente chunk a NOW(). Esto VIOLABA `CHUNK_LEARNING_MODE=strict`
                        # (default): el chunk N+1 se generaba con 0 días de adherencia real
                        # del chunk N (el usuario ni siquiera había vivido el chunk actual).
                        # Reportado por usuario 2026-05-06: plan creado ayer (3 días) → hoy
                        # ve [Mié, Jue] (2 días); el sistema aceleraba chunk 2 inmediatamente
                        # robando aprendizaje del chunk 1 que aún tenía Jueves vivo.
                        #
                        # AHORA: solo aceleramos cuando ya NO quedan días vivos
                        # (`shifted_days == 0`). Mientras el chunk actual tenga al menos
                        # 1 día por consumir, respetamos strict: el chunk siguiente espera
                        # su `execute_after` natural (calculado en `_compute_chunk_delay_days`).
                        # En `safety_margin` mode, mantenemos el comportamiento previo
                        # (acelerar al menor gap) para conservar buffer ante fallos LLM.
                        from constants import CHUNK_LEARNING_MODE as _CLM
                        _vc_should_fire = (
                            _CLM == "safety_margin"
                            or len(shifted_days) == 0
                        )
                        if _vc_should_fire:
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
                                        f"(gap visible: {len(shifted_days)}/{window_needed} días, mode={_CLM})"
                                    )
                            except Exception as e:
                                logger.error(f"❌ [VISUAL CONTINUITY] Error acelerando chunk pendiente: {e}")
                        else:
                            logger.info(
                                f"⏸️ [VISUAL CONTINUITY] Skip aceleración para plan {plan_id} "
                                f"(visible={len(shifted_days)}/{window_needed}, mode=strict): "
                                f"respetando que chunk previo aún tiene días vivos."
                            )
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
                            # [P3-SHIFT-DATEONLY-LOCAL · 2026-05-18] Preservar el
                            # formato original del campo. Si entró como date-only
                            # ("YYYY-MM-DD"), persistir el shift también como
                            # date-only en lugar de promoverlo a timestamp ISO
                            # completo — evita drift entre escrituras y mantiene
                            # SSOT con el backfill SQL p0_3.
                            if _is_date_only_shift:
                                new_plan_start_iso = (start_date + timedelta(days=days_since_creation)).isoformat()
                            else:
                                new_start = start_dt + timedelta(days=days_since_creation)
                                new_plan_start_iso = new_start.isoformat()
                            shifted_data['grocery_start_date'] = new_plan_start_iso

                            # [P0-C] Accumulate shift days
                            current_accum = int(shifted_data.get("_shift_days_accumulated", 0))
                            shifted_data["_shift_days_accumulated"] = current_accum + days_since_creation

                        # [P0-2] Sello CAS: timestamp que el worker compara para detectar
                        # si el plan fue modificado externamente durante el LLM call.
                        shifted_data['_plan_modified_at'] = datetime.now(timezone.utc).isoformat()

                        # [P2-NEXT-1 · 2026-05-11] Filtro `AND user_id = %s` para
                        # cerrar drift defense-in-depth contra I2 (CLAUDE.md).
                        # `plan_id` se resolvió arriba (línea ~1585) vía SELECT
                        # filtrado por user_id, así que funcionalmente no cambia
                        # nada hoy; pero ancla la invariante para que un refactor
                        # que reordene la resolución upstream no abra IDOR silente.
                        cursor.execute(
                            "UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s AND user_id = %s",
                            (json.dumps(shifted_data, ensure_ascii=False), plan_id, user_id)
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

        # [P2-LIVE-7 · 2026-05-11] Audit api_usage. `verify_api_quota` solo LEE
        # el contador mensual — no incrementa. Si no llamamos `log_api_usage`,
        # el paywall no cuenta este shift contra el cap del usuario y un
        # cliente abusivo podría disparar shifts infinitos para forzar
        # regeneración de chunks.
        if verified_user_id:
            try:
                log_api_usage(verified_user_id, "shift_plan")
            except Exception as _audit_err:
                logger.warning(f"[P2-LIVE-7] log_api_usage shift_plan falló: {_audit_err}")

        return {
            "success": True,
            "message": "Plan actualizado a la fecha.",
            "plan_data": shifted_data,
            "_pantry_degraded_summary": _p02_summary,
        }

    except Exception as e:
        logger.error(f"❌ [API SHIFT ERROR] {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

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

        # [P1-16/CANCEL-RACE-FIX 2026-05-06] Mismo fix que en /analyze/stream:
        # limpiar registry de cancels para este session_id antes de iniciar.
        # Evita que un cancel obsoleto en vuelo aborte esta nueva pipeline.
        if session_id:
            _clear_cancelled_session(session_id)

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
            # _td_err garantizado no-None: _validate_total_days retorna (False, dict)
            # cuando inválido y (True, None) cuando válido.
            assert _td_err is not None
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
        memory: dict = {}  # default cuando no hay session_id; .get(...) abajo guardado por `if session_id`

        if session_id:
            get_or_create_session(session_id)
            memory = build_memory_context(session_id)
            history = memory["recent_messages"]

        actual_user_id = user_id if user_id and user_id != "guest" else None
        if actual_user_id:
            likes = get_user_likes(actual_user_id)

            # [GAP 10] → Movido a inject_learning_signals_from_profile (P0 fix)
            pass

        active_rejections = get_active_rejections(user_id=actual_user_id, session_id=session_id)  # pyright: ignore[reportArgumentType]
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
            # [P1-SPEND-CAP-ALERT · 2026-05-28] Distinguir spending-cap (persistente)
            # de saturación transitoria: el mensaje "intenta en 1-2 min" es FALSO
            # cuando el cap mensual de Gemini está agotado (reintentar no ayuda).
            _spend_cap = bool(result.get("_llm_spend_cap"))
            logger.warning(
                f"🚨 [FALLBACK-GUARD] Pipeline devolvió plan de emergencia "
                f"(LLM upstream caído{', spending cap' if _spend_cap else ''}). "
                f"Devolviendo 503 sin persistir. user={actual_user_id or 'guest'}"
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "El servicio de IA no está disponible en este momento. "
                    "Estamos trabajando para restablecerlo; vuelve a intentarlo más tarde."
                    if _spend_cap else
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
            # [P1-PANTRY-GUARD-REGEN-SKIP · 2026-05-18] Si el cliente envía
            # update_reason (Renovar/Actualizar), saltar el guard — el plan
            # nuevo define la nueva lista de compras.
            update_reason=data.get("update_reason"),
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

        # [P2-PLAN-PERSIST-FAILED · 2026-05-30] Si la persistencia chunking falló
        # (INSERT meal_plans → None), NO devolver un plan "exitoso" que no existe en
        # DB. Devolver 503 para que el frontend muestre reintento (el alert ya se
        # emitió en _postprocess_pipeline_result).
        if isinstance(result, dict) and result.get("_persist_failed"):
            logger.error(
                f"🛑 [P2-PLAN-PERSIST-FAILED/sync] Plan no persistido — devolviendo 503. "
                f"user={actual_user_id or 'guest'}"
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Generamos tu plan pero no pudimos guardarlo por un problema "
                    "temporal. Por favor intenta de nuevo en unos segundos."
                ),
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
        # [P3-TRACEBACK-PRINT-EXC · 2026-05-15]
        logger.exception(f"❌ [ERROR] Error en /api/analyze: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

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

        # [P1-16/CANCEL-RACE-FIX 2026-05-06] Limpiar cualquier cancel pendiente
        # del registry para este session_id ANTES de iniciar la pipeline.
        #
        # Bug observado: el cancel del frontend usa `fetch(..., keepalive: true)`
        # fire-and-forget. Si el usuario cancela una generación y de inmediato
        # regenera, el POST cancel puede llegar al backend DESPUÉS del nuevo
        # POST analyze/stream (race condition). Como el session_id es estable
        # (= guest_session_id o user_id), el cancel viejo agrega el session_id
        # al `_PLAN_CANCEL_REGISTRY` y la pipeline recién iniciada lo detecta
        # vía `is_session_cancelled(session_id)` → se aborta inmediatamente.
        # Resultado: usuario ve "no me dejó generar" tras cancelar.
        #
        # Fix: al iniciar nueva generación, limpiar el registry para garantizar
        # que cancels obsoletos no afectan esta nueva pipeline. Si el frontend
        # quiere cancelar ESTA generación, el cancel posterior (post analyze/stream)
        # se procesa normalmente porque el set vuelve a llenarse cuando llegue.
        if session_id:
            _clear_cancelled_session(session_id)

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
            # _td_err garantizado no-None: _validate_total_days retorna (False, dict)
            # cuando inválido y (True, None) cuando válido.
            assert _td_err is not None
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
        memory: dict = {}  # default cuando no hay session_id; .get(...) abajo guardado por `if session_id`

        if session_id:
            get_or_create_session(session_id)
            memory = build_memory_context(session_id)
            history = memory["recent_messages"]

        actual_user_id = user_id if user_id and user_id != "guest" else None
        if actual_user_id:
            likes = get_user_likes(actual_user_id)

            # [GAP 10] → Movido a inject_learning_signals_from_profile (P0 fix)
            pass

        active_rejections = get_active_rejections(user_id=actual_user_id, session_id=session_id)  # pyright: ignore[reportArgumentType]
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

        # [P1-DEEP-SEARCH-PIPELINE · 2026-05-15] Guardrail + tracking del
        # pipeline pendiente. Permite que el pipeline continúe corriendo
        # incluso si el cliente cierra el SSE (el pipeline corre como
        # `asyncio.create_task` independiente, más abajo — NO está shielded;
        # sobrevive porque `create_task` no se cancela al GC del generator);
        # el frontend puede volver al sitio y ver el plan listo.
        #
        # Guardrail: si el user tiene un pipeline activo < 15 min, rechazar
        # la nueva request → el frontend redirige al polling del existente.
        # Sin esto, un user que recarga durante generación dispararía un
        # 2do pipeline, pagando $0.20-$0.40 extra a Gemini.
        _deep_search_user_id = actual_user_id  # NO incluir session_id (guests no soportados aún)
        if _deep_search_user_id:
            from db_plans import check_user_has_active_pipeline, upsert_pending_pipeline
            try:
                _active = check_user_has_active_pipeline(_deep_search_user_id, max_age_min=15)
                if _active:
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "pipeline_already_running",
                            "started_at": _active.get("started_at"),
                            "message": (
                                "Ya tienes un plan generándose. Espera a que termine "
                                "(o vuelve al dashboard, te avisaremos cuando esté listo)."
                            ),
                        },
                    )
                # Reservar slot para este user. Si falla la KV, log + continuar
                # (deep-search recovery se pierde pero la generación procede).
                # [P1-DEEP-SEARCH-DEBUG · 2026-05-15] Capturar el bool de retorno
                # para que el log refleje el resultado REAL del upsert. Pre-fix
                # el log decía "registrado" aunque el upsert hubiera retornado
                # False silentemente — engañoso para diagnóstico.
                _kv_ok = upsert_pending_pipeline(_deep_search_user_id, status="generating")
                if _kv_ok:
                    logger.info(
                        f"📌 [P1-DEEP-SEARCH-PIPELINE] Pipeline registrado en KV para "
                        f"user={_deep_search_user_id[:8]} (status=generating)."
                    )
                else:
                    logger.warning(
                        f"⚠️ [P1-DEEP-SEARCH-PIPELINE] Pipeline NO se pudo registrar en KV "
                        f"para user={_deep_search_user_id[:8]} — el feature de recovery "
                        f"deep-search NO funcionará para este plan. La generación sigue."
                    )
            except HTTPException:
                raise
            except Exception as _track_err:
                logger.warning(
                    f"[P1-DEEP-SEARCH-PIPELINE] Track setup falló (best-effort): {_track_err!r}"
                )

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
        pipeline_result: dict[str, Any] = {"result": None, "error": None}

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
                # [P3-TRACEBACK-PRINT-EXC · 2026-05-15]
                logger.exception(f"❌ [SSE PIPELINE ERROR]: {e}")
                # [P1-DEEP-SEARCH-PIPELINE · 2026-05-15] Marcar pipeline como
                # failed en la KV para que el frontend que vuelve vea el error
                # en lugar de un loading infinito.
                if _deep_search_user_id:
                    try:
                        from db_plans import upsert_pending_pipeline
                        upsert_pending_pipeline(
                            _deep_search_user_id, status="failed",
                            error=str(e)[:200],
                        )
                    except Exception:
                        pass
            finally:
                # Señal de fin para que el generador SSE cierre
                try:
                    loop.call_soon_threadsafe(progress_queue.put_nowait, {"event": "_done"})
                except Exception:
                    pass

        # [P1-16] Retener el handle del task del pipeline para poder cancelarlo
        # cooperativamente si el frontend POST /api/plans/cancel llega antes
        # de que el pipeline complete. Sin handle no podemos llamar `.cancel()`
        # ni propagar `asyncio.CancelledError` al pipeline aún en vuelo.
        _pipeline_task = asyncio.create_task(run_pipeline())

        # [P2-PIPELINE-TASK-DONE-CALLBACK · 2026-05-16] Bug observado 2026-05-16
        # 02:39-02:53 (plan_id=ninguno): user inició plan, cerró pestaña a los
        # 11min, SSE generator terminó, `_pipeline_task` siguió corriendo en
        # background (asyncio.create_task independiente, NO shielded),
        # eventualmente timed-out o falló, y
        # el row `pending_pipeline:bf6f1383` quedó con status='generating'
        # FOREVER. Causa raíz: el upsert KV-complete vive DENTRO del SSE
        # generator (se ejecuta cuando llega el event 'complete'). Si el
        # generator terminó antes que el pipeline, ese path nunca corre.
        # El except interno marca 'failed' SOLO si la excepción se captura
        # dentro de `_run_pipeline_blocking` — un TimeoutError de asyncio o
        # CancelledError NO se captura ahí (CancelledError subclass BaseException
        # en Py3.8+, no Exception).
        #
        # Fix: add_done_callback que SIEMPRE marca el KV final-state cuando el
        # task termina, independiente de quién esté escuchando. Si OK → no-op
        # (el SSE generator ya marcó complete si llegó). Si fallido/cancelled
        # → marca failed para que el recovery limpie. Idempotente: si ya está
        # complete/failed, el upsert sobrescribe con el mismo status.
        # [P2-PIPELINE-TASK-DONE-COMPLETE · 2026-05-16] Bug observado 2026-05-16
        # 03:01-03:12 (Pipeline Quality=0.995, Aprobado=True): user cerró SSE
        # a las 03:12:42 (segundos ANTES de que el revisor médico terminara a
        # 03:12:50). Pipeline completó exitosamente PERO:
        #   - El `_postprocess_pipeline_result` (persist meal_plan + seed
        #     learning + schedule chunk 2)
        #   - El KV mark complete
        #   - El emit `complete` event
        # ...todo eso vive DENTRO del SSE generator. Como cerró antes, NINGUNO
        # corre → plan_id nunca persiste a `meal_plans` + KV queda `generating`.
        #
        # Fix: el done_callback ahora cubre 3 casos:
        #   (a) Task cancelled / exception → marca KV failed (anti-zombi).
        #   (b) Task OK + result poblado + SSE generator vivo → no-op (generator
        #       hace su trabajo).
        #   (c) Task OK + result poblado + SSE generator ya cerró → programa
        #       postprocess + persist + KV mark complete vía asyncio.create_task
        #       en el event loop, usando los kwargs del closure.
        # Sentinel _sse_completed_naturally se setea True dentro del SSE
        # generator JUSTO ANTES de hacer el upsert KV complete; si está False
        # cuando el callback corre, sabemos que el generator murió pre-postprocess.
        _sse_completed_naturally = {"flag": False}

        def _on_pipeline_task_done(_task):
            try:
                if _task.cancelled():
                    _err_msg = "Pipeline cancelled (timeout o cancel cooperativo)"
                    _final_status = "failed"
                    _need_fallback_postprocess = False
                else:
                    _exc = _task.exception()
                    if _exc is None:
                        # OK: chequear si el SSE generator alcanzó a hacer el
                        # postprocess. Si sí → no-op. Si NO → fallback.
                        if _sse_completed_naturally["flag"]:
                            return
                        # SSE generator cerró antes — necesitamos fallback.
                        if not pipeline_result.get("result"):
                            # Sin result: el task terminó pero no produjo plan.
                            # Marcar como failed (algo raro).
                            _err_msg = "Pipeline OK pero sin result populated"
                            _final_status = "failed"
                            _need_fallback_postprocess = False
                        else:
                            _need_fallback_postprocess = True
                            _err_msg = None
                            _final_status = "complete"
                    else:
                        _err_msg = str(_exc)[:200]
                        _final_status = "failed"
                        _need_fallback_postprocess = False

                if _need_fallback_postprocess and _deep_search_user_id:
                    # Programar el fallback postprocess en el event loop.
                    # `loop.call_soon_threadsafe` es seguro desde el callback.
                    logger.warning(
                        f"🔧 [P2-PIPELINE-TASK-DONE-COMPLETE] Pipeline OK pero SSE "
                        f"generator murió pre-postprocess. Ejecutando fallback "
                        f"persist + KV mark complete para user={_deep_search_user_id[:8]}."
                    )
                    async def _fallback_postprocess():
                        # _deep_search_user_id garantizado no-None: el closure solo se
                        # programa dentro del guard `if ... and _deep_search_user_id:` (arriba)
                        # y la variable nunca se reasigna tras L2794.
                        assert _deep_search_user_id is not None
                        try:
                            # [P2-PIPELINE-TASK-DONE-RACE-FIX · 2026-05-16] Bug
                            # observado plan_id=0844a613+5f5e5aa6 (2026-05-16 03:51):
                            # el callback dispara INMEDIATAMENTE al pipeline_task done,
                            # pero el SSE generator todavía no ha procesado el `_done`
                            # event de la queue (está en próximo tick del while loop).
                            # Cuando el callback chequea el sentinel, es False → fallback
                            # dispara. Microsegundos después, el SSE generator procesa
                            # `_done`, setea sentinel=True, y hace su propio postprocess.
                            # Resultado: AMBOS persisten → 2 planes duplicados.
                            #
                            # Fix: sleep ~3s para dar chance al SSE generator de marcar
                            # el sentinel. Re-chequear sentinel post-sleep. Si SSE
                            # generator está vivo, sentinel=True → skip fallback.
                            # Si SSE generator está realmente muerto, sentinel sigue
                            # False → fallback procede. 3s es generoso pero seguro:
                            # el postprocess full toma ~5-10s, no perdemos mucho.
                            await asyncio.sleep(3)
                            if _sse_completed_naturally["flag"]:
                                logger.info(
                                    f"✓ [P2-PIPELINE-TASK-DONE-RACE-FIX] SSE generator "
                                    f"se encargó del postprocess (sentinel=True tras sleep 3s). "
                                    f"Fallback skipping para user={_deep_search_user_id[:8]}."
                                )
                                return
                            _result = pipeline_result.get("result")
                            if not _result:
                                return
                            # [P2-PIPELINE-FALLBACK-GUARD-DONE · 2026-05-30] Mismo guard
                            # que los 2 paths reales (sync L2447 + SSE L3157): si el
                            # pipeline devolvió un plan de emergencia matemático
                            # (`_is_fallback`, LLM upstream caído), NO persistirlo via este
                            # callback. Sin el guard, cuando el SSE generator muere ANTES de
                            # procesar `_done` (la razón misma de existir de este callback) un
                            # plan "Fallback: pollo y arroz" quedaba persistido + se encolaban
                            # N chunks futuros → el usuario lo veía en su historial por una
                            # semana. Marcamos KV failed para que el frontend muestre el
                            # mensaje de reintento. Tooltip-anchor: P2-PIPELINE-FALLBACK-GUARD-DONE.
                            if isinstance(_result, dict) and _result.get("_is_fallback"):
                                logger.warning(
                                    f"🚨 [P2-PIPELINE-FALLBACK-GUARD-DONE] Pipeline devolvió "
                                    f"plan de emergencia; NO se persiste via done-callback. "
                                    f"user={_deep_search_user_id[:8]}"
                                )
                                try:
                                    from db_plans import upsert_pending_pipeline
                                    upsert_pending_pipeline(
                                        _deep_search_user_id,
                                        status="failed",
                                        error="llm_unavailable_fallback",
                                    )
                                except Exception as _kv_e:
                                    logger.warning(
                                        f"[P2-PIPELINE-FALLBACK-GUARD-DONE] KV failed update no-op: {_kv_e!r}"
                                    )
                                return
                            # Recomputar memory_ctx desde scope del endpoint
                            # (mismo cálculo que el SSE generator hace antes
                            # del postprocess original).
                            _fb_memory_ctx = (
                                memory.get("full_context_str", "") if session_id else ""
                            )
                            _result = await asyncio.to_thread(
                                _postprocess_pipeline_result,
                                result=_result,
                                actual_user_id=actual_user_id,
                                session_id=session_id,
                                data=data,
                                taste_profile=taste_profile,
                                memory_ctx=_fb_memory_ctx,
                                rejected_meal_names=rejected_meal_names,
                                total_days_requested=total_days_requested,
                                use_chunking=use_chunking,
                                background_tasks=background_tasks,
                                plan_start_date=start_date_iso,
                                tz_offset_mins=tz_offset_mins,
                                # [P2-PIPELINE-TASK-DONE-RACE-FIX · 2026-05-16]
                                # Usar "sse" (no "sse_fallback_postprocess") para
                                # que P0_3_LEGACY_LEARNING_CONTEXTS lo acepte.
                                # El context name custom rompía el seed del
                                # _last_chunk_learning. Observability se mantiene
                                # via el log "🔧 [P2-PIPELINE-TASK-DONE-COMPLETE]
                                # Ejecutando fallback..." que es único al fallback path.
                                transport_label="sse",
                            )
                            # [P2-PLAN-PERSIST-FAILED · 2026-05-30] Mismo guard que los
                            # otros 2 consumidores: si la persistencia chunking falló, marcar
                            # KV failed (no `complete` con plan_id_final=None) para que el
                            # frontend que vuelve vía /pending-status vea el error, no un
                            # phantom complete. El alert ya se emitió en el postprocess.
                            if isinstance(_result, dict) and _result.get("_persist_failed"):
                                logger.error(
                                    f"🛑 [P2-PLAN-PERSIST-FAILED/done-callback] Plan no persistido "
                                    f"— marcando KV failed. user={_deep_search_user_id[:8]}"
                                )
                                try:
                                    from db_plans import upsert_pending_pipeline
                                    upsert_pending_pipeline(
                                        _deep_search_user_id,
                                        status="failed",
                                        error="plan_persist_failed",
                                    )
                                except Exception as _kv_e:
                                    logger.warning(
                                        f"[P2-PLAN-PERSIST-FAILED/done-callback] KV failed update no-op: {_kv_e!r}"
                                    )
                                return
                            _plan_id_final = (
                                _result.get("id") or _result.get("plan_id")
                                if isinstance(_result, dict) else None
                            )
                            from db_plans import upsert_pending_pipeline
                            upsert_pending_pipeline(
                                _deep_search_user_id,
                                status="complete",
                                plan_id_final=_plan_id_final,
                            )
                            logger.info(
                                f"✅ [P2-PIPELINE-TASK-DONE-COMPLETE] Fallback postprocess "
                                f"OK user={_deep_search_user_id[:8]} plan_id={(_plan_id_final or '')[:8]}"
                            )
                        except Exception as _fb_err:
                            logger.exception(
                                f"❌ [P2-PIPELINE-TASK-DONE-COMPLETE] Fallback postprocess "
                                f"falló: {_fb_err!r}"
                            )
                            try:
                                from db_plans import upsert_pending_pipeline
                                upsert_pending_pipeline(
                                    _deep_search_user_id,
                                    status="failed",
                                    error=f"Fallback postprocess error: {str(_fb_err)[:150]}",
                                )
                            except Exception:
                                pass
                    def _schedule_fallback_task():
                        # [P2-FALLBACK-TASK-TRACKED · 2026-05-28] Guardar strong-ref
                        # para que el task no sea GC'd mid-flight; discard al terminar.
                        _t = asyncio.create_task(_fallback_postprocess())
                        _BG_SSE_FALLBACK_TASKS.add(_t)
                        _t.add_done_callback(_BG_SSE_FALLBACK_TASKS.discard)
                    try:
                        loop.call_soon_threadsafe(_schedule_fallback_task)
                    except Exception as _sched_err:
                        logger.warning(
                            f"[P2-PIPELINE-TASK-DONE-COMPLETE] No se pudo programar "
                            f"fallback postprocess: {_sched_err!r}"
                        )
                elif _final_status == "failed" and _deep_search_user_id:
                    from db_plans import upsert_pending_pipeline
                    upsert_pending_pipeline(
                        _deep_search_user_id, status=_final_status, error=_err_msg,
                    )
                    logger.warning(
                        f"⚠️ [P2-PIPELINE-TASK-DONE-COMPLETE] Pipeline task terminó "
                        f"failed; KV marcado failed para "
                        f"user={_deep_search_user_id[:8]}. Error: {_err_msg}"
                    )
            except Exception as _cb_err:
                # Best-effort: NUNCA lanzar excepción desde un done_callback
                # (Python las trata como unhandled y las loguea a stderr).
                logger.warning(
                    f"[P2-PIPELINE-TASK-DONE-COMPLETE] callback falló: {_cb_err!r}"
                )
        _pipeline_task.add_done_callback(_on_pipeline_task_done)

        # [P2-GEN-WALL-TIMEOUT · 2026-05-27] Watchdog wall-clock sobre el
        # pipeline. El pipeline corre como `asyncio.create_task` INDEPENDIENTE
        # del SSE generator (sobrevive disconnect del cliente — feature
        # P1-DEEP-SEARCH-PIPELINE). Si `arun_plan_pipeline` se cuelga de verdad
        # (LLM upstream wedged, sin progreso, sin lanzar excepción), el task
        # correría hasta el reinicio del proceso ocupando un slot y dejando el
        # KV `pending_pipeline:<user>` en `generating` para siempre. Este guard
        # lo cancela tras N segundos; el `_on_pipeline_task_done` de arriba ya
        # marca KV `failed` en su rama `_task.cancelled()` → el frontend que
        # vuelve ve el error en vez de un loading infinito.
        #
        # NO confundir con: (a) cancel cooperativo del user (`/api/plans/cancel`,
        # maneja `_should_stop`), (b) disconnect del cliente (intencional
        # keep-running). Esto es solo el techo duro anti-zombi.
        #
        # Knob `MEALFIT_GENERATION_MAX_WALL_S` (default 900s=15min, clamp
        # [60, 3600]; 0 desactiva). 15min es generoso: la generación legítima
        # (deep-search + day-gen paralelo + review médico) toma ~2-6min.
        # Tooltip-anchor: P2-GEN-WALL-TIMEOUT.
        _gen_wall_s = _env_int(
            "MEALFIT_GENERATION_MAX_WALL_S",
            900,
            validator=lambda v: v == 0 or 60 <= v <= 3600,
        )
        if _gen_wall_s > 0:
            async def _pipeline_wall_clock_guard():
                try:
                    await asyncio.sleep(_gen_wall_s)
                except asyncio.CancelledError:
                    return
                if not _pipeline_task.done():
                    logger.error(
                        f"⏱️ [P2-GEN-WALL-TIMEOUT] Pipeline excedió {_gen_wall_s}s "
                        f"sin completar — cancelando (anti-zombi). El KV se marca "
                        f"failed via done_callback. user={_deep_search_user_id or 'guest'}"
                    )
                    _pipeline_task.cancel()
            _wall_guard_task = asyncio.create_task(_pipeline_wall_clock_guard())
            # Cancelar el watchdog cuando el pipeline termine (natural, error o
            # cancel) para no dejar el sleep colgando hasta su deadline.
            _pipeline_task.add_done_callback(
                lambda _t, _g=_wall_guard_task: _g.cancel()
            )

        async def event_generator():
            """Generador SSE que consume la cola de progreso."""
            # [P6-CANCEL-PROPAGATION-FIX] Helper centralizado para chequear
            # cancel + disconnect, llamado en CADA iteración (no solo en
            # heartbeat timeout). Bug observable PDF 2026-05-06: el pipeline
            # produce eventos constantemente (<5s) → asyncio.TimeoutError
            # nunca fire → cancel check (que estaba SOLO en branch timeout)
            # nunca corre → pipeline sigue hasta terminar pese al cancel
            # cooperativo del frontend. Resultado: cuota LLM consumida +
            # plan persistido en DB ~30s después aunque user canceló.
            #
            # [P6-CANCEL-PROPAGATION-FIX-2] Pre-fix esperaba `await _pipeline_task`
            # que podía colgar 30-180s si el pipeline estaba mid-LLM call (la
            # cancelación asyncio se programa pero no se materializa hasta el
            # próximo await yield). Eso bloqueaba el SSE response → user veía
            # el browser "cargando" pese a haber cancelado. Ahora: cancelamos
            # el task pero NO esperamos — yield error inmediato + return. El
            # task se cancela en background y libera resources eventualmente.
            async def _should_stop():
                """Devuelve True SOLO si hay cancel EXPLÍCITO del usuario.

                [P1-DEEP-SEARCH-PIPELINE · 2026-05-15] Disconnect NO termina
                el generator. El generator sigue corriendo hasta procesar
                el evento `_done`, persistir el plan, y retornar. Los `yield`
                a un socket cerrado son no-ops (Starlette los maneja).
                Esto es lo que habilita el "deep-search style": aunque el
                user cierre la pestaña, el plan se persiste y queda accesible
                via `/api/plans/pending-status`.

                Solo el cancel EXPLÍCITO vía `/api/plans/cancel` aborta:
                el usuario clickeó "Cancelar Generación", quiere que pare.
                """
                if session_id and is_session_cancelled(session_id):
                    logger.info(
                        f"🛑 [P1-16] Cancel explícito recibido para session={session_id}. "
                        f"Cancelando pipeline task + limpiando KV tracker."
                    )
                    if not _pipeline_task.done():
                        _pipeline_task.cancel()
                    # Limpiar KV para que el user no vea "pending" stale.
                    if _deep_search_user_id:
                        try:
                            from db_plans import clear_pending_pipeline
                            clear_pending_pipeline(_deep_search_user_id)
                        except Exception:
                            pass
                    return True
                return False

            try:
                while True:
                    # [P6-CANCEL-PROPAGATION-FIX] Chequeo PRE-evento: si user
                    # canceló desde la iteración anterior, abortar antes de
                    # procesar el siguiente evento. Esto cubre el caso donde
                    # eventos llegan más rápido que 5s (heartbeat path nunca
                    # dispara).
                    if await _should_stop():
                        yield f"data: {_json.dumps({'event': 'error', 'data': {'code': 'user_cancelled', 'message': 'Generación cancelada por el usuario.'}})}\n\n"
                        return

                    # Esperar eventos con timeout para detectar desconexión del cliente
                    try:
                        event_data = await asyncio.wait_for(progress_queue.get(), timeout=5.0)
                    except asyncio.TimeoutError:
                        # Heartbeat para mantener la conexión viva
                        yield f"data: {_json.dumps({'event': 'heartbeat'})}\n\n"

                        # [P6-CANCEL-PROPAGATION-FIX] Mismo chequeo via helper
                        # — antes había código duplicado in-line aquí.
                        if await _should_stop():
                            yield f"data: {_json.dumps({'event': 'error', 'data': {'code': 'user_cancelled', 'message': 'Generación cancelada por el usuario.'}})}\n\n"
                            return
                        continue

                    # Señal de fin del pipeline
                    if event_data.get("event") == "_done":
                        # [P2-PIPELINE-TASK-DONE-COMPLETE · 2026-05-16] CRÍTICO:
                        # setear sentinel ANTES de empezar el postprocess. El
                        # `_pipeline_task` ya está done en este punto (por eso
                        # llegó el `_done` event); mi `add_done_callback` corre
                        # casi-síncronamente al done. Si NO marco el sentinel
                        # ahora, el callback ve `flag=False` y dispara fallback
                        # postprocess en paralelo con el SSE generator → DOBLE
                        # persist (bug observado plan_id=e910ad31+c592f7ac 2026-05-16
                        # 03:32). Setearlo aquí garantiza que el callback haga
                        # no-op cuando llegue (SSE generator está vivo y va a
                        # hacer postprocess).
                        _sse_completed_naturally["flag"] = True
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
                                # [P1-SPEND-CAP-ALERT · 2026-05-28] Mensaje honesto
                                # cuando el fallback fue por spending-cap de Gemini:
                                # "intenta en 1-2 min" es falso (reintentar no ayuda
                                # hasta subir el cap). El system_alert ya lo emitió
                                # el pipeline (graph_orchestrator).
                                _spend_cap = bool(result.get("_llm_spend_cap"))
                                logger.warning(
                                    f"🚨 [FALLBACK-GUARD/SSE] Pipeline devolvió plan de "
                                    f"emergencia (LLM upstream caído{', spending cap' if _spend_cap else ''}). "
                                    f"No se persiste. user={actual_user_id or 'guest'}"
                                )
                                _fallback_msg = (
                                    'El servicio de IA no está disponible en este momento. '
                                    'Estamos trabajando para restablecerlo; vuelve a intentarlo más tarde.'
                                    if _spend_cap else
                                    'La IA está temporalmente saturada y no pudimos generar tu plan. '
                                    'Por favor intenta de nuevo en 1-2 minutos.'
                                )
                                yield f"data: {_json.dumps({'event': 'error', 'data': {'code': 'llm_unavailable', 'message': _fallback_msg}})}\n\n"
                                break

                            # [P1-DEEP-SEARCH-PIPELINE · 2026-05-15] Comportamiento INVERTIDO
                            # respecto a P0-3 pre-fix. Pre-fix cortaba aquí si el cliente
                            # desconectaba para evitar "plan huérfano". Ahora el plan huérfano
                            # es feature: el `_save_plan_and_track_background` persiste, el
                            # KV tracker `pending_pipeline:<user_id>` marca status='complete',
                            # y el frontend lo recupera al volver via `/pending-status`.
                            # Solo loguear para observabilidad.
                            if await request.is_disconnected():
                                logger.info(
                                    f"🔌 [P1-DEEP-SEARCH-PIPELINE] Cliente desconectado tras "
                                    f"pipeline. CONTINUANDO persistencia — usuario recuperará "
                                    f"el plan via /pending-status. user={actual_user_id or 'guest'}"
                                )

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
                                # [P1-PANTRY-GUARD-REGEN-SKIP · 2026-05-18] Si el
                                # cliente envía update_reason (Renovar/Actualizar),
                                # saltar el guard — el plan nuevo define la nueva
                                # lista de compras.
                                update_reason=data.get("update_reason"),
                            )

                            # [P1-DEEP-SEARCH-PIPELINE · 2026-05-15 · re-fixed
                            # P2-PIPELINE-DISCONNECT-PERSIST · 2026-05-30] Pre-fix este
                            # re-check hacía `break` ANTES del postprocess + KV-complete.
                            # Pero el sentinel `_sse_completed_naturally` ya está True (se
                            # setea al recibir `_done`), así que el done-callback hace no-op →
                            # con el `break` el plan NUNCA se persistía y el KV
                            # `pending_pipeline:<user>` quedaba en status='generating' PARA
                            # SIEMPRE (spinner perpetuo al volver; bug clase plan bf6f1383). El
                            # comentario YA decía "ya NO cortamos persistencia" pero el `break`
                            # seguía vivo — contradicción detectada en audit 2026-05-30. Igual
                            # que el check de disconnect de arriba: el plan huérfano es FEATURE;
                            # solo logueamos y CONTINUAMOS para que el postprocess + KV-complete
                            # corran y el user recupere el plan vía /pending-status.
                            # Tooltip-anchor: P2-PIPELINE-DISCONNECT-PERSIST.
                            if await request.is_disconnected():
                                logger.info(
                                    f"🔌 [P2-PIPELINE-DISCONNECT-PERSIST] Cliente desconectado durante "
                                    f"validación pantry. CONTINUANDO persistencia — el user recuperará "
                                    f"el plan vía /pending-status. user={actual_user_id or 'guest'}"
                                )

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

                            # [P2-PLAN-PERSIST-FAILED · 2026-05-30] Si la persistencia
                            # chunking falló (INSERT meal_plans → None), NO emitir `complete`
                            # con un plan_id_final=None (phantom success). Marcar KV failed +
                            # emitir error event para que el frontend muestre reintento. El
                            # system_alert ya se emitió en _postprocess_pipeline_result.
                            if isinstance(result, dict) and result.get("_persist_failed"):
                                logger.error(
                                    f"🛑 [P2-PLAN-PERSIST-FAILED/SSE] Plan no persistido — "
                                    f"emitiendo error en vez de complete. user={actual_user_id or 'guest'}"
                                )
                                if _deep_search_user_id:
                                    try:
                                        from db_plans import upsert_pending_pipeline
                                        upsert_pending_pipeline(
                                            _deep_search_user_id,
                                            status="failed",
                                            error="plan_persist_failed",
                                        )
                                    except Exception as _kv_e:
                                        logger.warning(
                                            f"[P2-PLAN-PERSIST-FAILED/SSE] KV failed update no-op: {_kv_e!r}"
                                        )
                                yield f"data: {_json.dumps({'event': 'error', 'data': {'code': 'plan_persist_failed', 'message': 'Generamos tu plan pero no pudimos guardarlo por un problema temporal. Por favor intenta de nuevo.'}})}\n\n"
                                break

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

                            # [P1-DEEP-SEARCH-PIPELINE · 2026-05-15] Actualizar KV
                            # tracker con `plan_id_final` para que el endpoint
                            # `/api/plans/pending-status` pueda servir al user
                            # que vuelve. Best-effort: si falla, el user todavía
                            # puede recuperar el plan via /history (es solo el
                            # auto-redirect lo que se pierde).
                            if _deep_search_user_id:
                                try:
                                    _plan_id_final = (
                                        result.get("id") or result.get("plan_id")
                                        if isinstance(result, dict) else None
                                    )
                                    # [P2-PIPELINE-TASK-DONE-COMPLETE · 2026-05-16]
                                    # Marcar el sentinel ANTES del upsert para que
                                    # el done_callback NO ejecute fallback redundante.
                                    # Si el SSE generator llegó aquí, el work está hecho.
                                    _sse_completed_naturally["flag"] = True
                                    from db_plans import upsert_pending_pipeline
                                    # [P1-DEEP-SEARCH-DEBUG · 2026-05-15] log
                                    # condicional al return — antes era ciego.
                                    _kv_complete_ok = upsert_pending_pipeline(
                                        _deep_search_user_id,
                                        status="complete",
                                        plan_id_final=_plan_id_final,
                                    )
                                    if _kv_complete_ok:
                                        logger.info(
                                            f"✅ [P1-DEEP-SEARCH-PIPELINE] KV marcado complete "
                                            f"user={_deep_search_user_id[:8]} plan_id={(_plan_id_final or '')[:8]}"
                                        )
                                    else:
                                        logger.warning(
                                            f"⚠️ [P1-DEEP-SEARCH-PIPELINE] KV complete update FAILED "
                                            f"user={_deep_search_user_id[:8]} plan_id={(_plan_id_final or '')[:8]} "
                                            f"— el user no recibirá auto-redirect aunque el plan SÍ se persistió."
                                        )
                                except Exception as _kv_err:
                                    logger.warning(
                                        f"[P1-DEEP-SEARCH-PIPELINE] KV update tras complete falló: {_kv_err!r}"
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
                # [P1-DEEP-SEARCH-PIPELINE · 2026-05-15] Comportamiento INVERTIDO
                # respecto a P6-CANCEL-PROPAGATION-FIX-3. Pre-fix cancelaba
                # `_pipeline_task` para evitar "plan huérfano". Ahora el plan
                # huérfano DEJA DE SERLO: el `upsert_pending_pipeline` KV +
                # el `services._save_plan_and_track_background` al final
                # persisten el plan. El usuario lo recupera al volver via
                # `/api/plans/pending-status`.
                #
                # El pipeline corre como `asyncio.create_task` independiente del
                # generator (NO shielded — sobrevive porque create_task no se
                # cancela al GC/cancel del generator). NO lo cancelamos aquí.
                logger.info(
                    "🔌 [SSE] Stream cancelado por cliente. Pipeline sigue "
                    "corriendo (deep-search mode); el usuario lo recuperará al volver."
                )
                # Re-raise para que asyncio cierre el generator correctamente.
                raise
            except Exception as e:
                logger.error(f"❌ [SSE] Error en generador: {e}")
                yield f"data: {_json.dumps({'event': 'error', 'data': {'message': str(e)}})}\n\n"
            finally:
                # [P1-DEEP-SEARCH-PIPELINE · 2026-05-15] NO cancelamos
                # `_pipeline_task` en el finally. Pre-fix lo cancelaba como
                # "defense-in-depth contra plan huérfano". Ahora el plan
                # huérfano es feature, no bug. Si el pipeline aún corre,
                # déjalo terminar — el KV tracker maneja el lifecycle.
                if not _pipeline_task.done():
                    logger.info(
                        "📌 [P1-DEEP-SEARCH-PIPELINE] Generator terminó pero "
                        "pipeline_task sigue activo. NO se cancela — el plan "
                        "se persistirá y será recuperable via /pending-status."
                    )
                # [P1-16] Limpieza del flag de cancelación al terminar el
                # stream (por completion, error, cancel cooperativo, o
                # disconnect). Sin esto, el set `_PLAN_CANCEL_REGISTRY`
                # crecería sin límite durante la vida del proceso si los
                # usuarios cancelan repetidamente.
                if session_id:
                    _clear_cancelled_session(session_id)

                # [P2-LIVE-7 · 2026-05-11] Audit api_usage. `verify_api_quota`
                # solo lee — sin log_api_usage, streaming pipelines no contaban
                # contra el cap mensual del usuario. Log en `finally` para que
                # se cargue independiente del path de salida (success / error
                # / cancel): el `_pipeline_task` ya consumió compute LLM al
                # arrancar, así que el usuario debe pagar.
                if user_id and user_id != "guest" and user_id != session_id:
                    try:
                        log_api_usage(user_id, "llm_analyze_stream")
                    except Exception as _audit_err:
                        logger.warning(
                            f"[P2-LIVE-7] log_api_usage analyze_stream falló: {_audit_err}"
                        )

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
        # [P3-TRACEBACK-PRINT-EXC · 2026-05-15]
        logger.exception(f"❌ [ERROR] Error en /api/analyze/stream: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


# [P1-DEEP-SEARCH-PIPELINE · 2026-05-15] Endpoint para que el frontend al
# arrancar (boot hook) detecte si hay un plan generándose o ya completado
# y redirija al usuario al loading screen / dashboard automáticamente.
#
# Auth: requiere `verified_user_id` (JWT). NO usa `verify_api_quota`
# porque es polling read-only (cero costo LLM) — análogo al patrón de
# `/history-list` etc.
@router.get("/pending-status")
async def api_pending_pipeline_status(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """[P1-DEEP-SEARCH-PIPELINE · 2026-05-15] Retorna el estado del pipeline
    pendiente para el user autenticado.

    Response shapes:
      - `{"status": "none"}` — no hay pipeline activo, frontend limpia localStorage.
      - `{"status": "generating", "started_at": "<iso>"}` — pipeline corriendo;
        frontend muestra loading screen con polling cada 5-10s.
      - `{"status": "complete", "plan_id_final": "<uuid>"}` — plan listo;
        frontend muestra toast "Tu plan está listo" + redirige a /dashboard.
      - `{"status": "failed", "error": "<msg>"}` — pipeline falló; frontend
        muestra toast de error + opción de regenerar.
    """
    if not verified_user_id or verified_user_id == "guest":
        return {"status": "none"}
    try:
        from db_plans import get_pending_pipeline
        # [P1-ASYNC-SYNC-DB-BLOCKING · 2026-05-24] handler async + DB sync → to_thread.
        payload = await asyncio.to_thread(get_pending_pipeline, verified_user_id)
        if not payload:
            return {"status": "none"}
        # Filtrar a campos públicos (no leak `updated_at` interno innecesario).
        return {
            "status": payload.get("status") or "none",
            "started_at": payload.get("started_at"),
            "plan_id_final": payload.get("plan_id_final"),
            "error": payload.get("error"),
        }
    except Exception as e:
        logger.warning(f"[P1-DEEP-SEARCH-PIPELINE] /pending-status error: {e!r}")
        return {"status": "none"}


@router.post("/pending-status/ack")
async def api_pending_pipeline_ack(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """[P1-DEEP-SEARCH-PIPELINE · 2026-05-15] Limpia el row de pending
    pipeline tras el frontend acknowledge la finalización. Idempotente.

    Frontend debe llamar este endpoint después de mostrar el toast +
    redirigir a /dashboard (o tras mostrar el error de failed). Sin esto,
    el next mount detectaría el row stale y entraría en loop de redirect.
    """
    if not verified_user_id or verified_user_id == "guest":
        return {"ok": True}
    try:
        from db_plans import clear_pending_pipeline
        # [P1-ASYNC-SYNC-DB-BLOCKING · 2026-05-24] handler async + DB sync → to_thread.
        await asyncio.to_thread(clear_pending_pipeline, verified_user_id)
        return {"ok": True}
    except Exception as e:
        logger.warning(f"[P1-DEEP-SEARCH-PIPELINE] /pending-status/ack error: {e!r}")
        return {"ok": False, "error": str(e)[:200]}


@router.post("/recipe/expand")
def api_expand_recipe(data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota), _rl: None = Depends(_EXPAND_LIMITER)):  # [P2-GUEST-LLM-RATELIMIT · 2026-05-30] throttle guest LLM
    """[P1-HIST-RECIPE-1 · 2026-05-10] Expande una receta con pasos de chef
    y persiste el resultado en el plan correcto.

    Bugs originales (audit Historial 2026-05-10):
      1. **Wrong-plan persist**: el handler llamaba `get_latest_meal_plan(user_id)`
         sin recibir `plan_id` del request. Si el usuario abría `Recipes`
         desde un plan recién restaurado mientras un chunk worker insertaba
         un plan nuevo (race), el `expanded_recipe` se persistía al plan
         equivocado y el plan visible no se actualizaba.
      2. **First-match-only**: match por `m.get("name") == data.get("name")`
         con doble `break` → solo la PRIMERA ocurrencia se persistía. En
         planes de 7d donde la misma receta se repite (común), las demás
         seguían sin `recipe`/`isExpanded` y volvían a quemar cuota LLM
         en cada cook-click silenciosamente.

    Fix:
      - Aceptamos `plan_id` opcional. Si está, SELECT explícito con
        `WHERE id=%s AND user_id=%s` (ownership). Si miss → fallback a
        `get_latest_meal_plan_with_id` para no romper clientes viejos.
      - Aceptamos `day_index`/`meal_index` opcionales. Si están y son
        consistentes con `name` → targeting preciso (una sola escritura).
      - Sin índices, recorremos TODAS las ocurrencias con `name` igual
        Y `recipe` original byte-equivalente (la expansión solo aplica
        si la receta base es idéntica) → cierra el quemado de cuota.

    Tooltip-anchor: P1-HIST-RECIPE-1-START | test_p1_hist_recipe_1_expand_targeting
    """
    try:
        user_id = data.get("user_id")
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")

        if not data.get("recipe") or not data.get("name"):
            raise HTTPException(status_code=400, detail="Faltan datos de la receta para expandir.")

        # [P1-HIST-RECIPE-1] Identificadores opcionales del request.
        # Validación con isinstance porque JSON puede traer null/str/int.
        req_plan_id = data.get("plan_id") if isinstance(data.get("plan_id"), str) else None
        req_day_index = data.get("day_index") if isinstance(data.get("day_index"), int) else None
        req_meal_index = data.get("meal_index") if isinstance(data.get("meal_index"), int) else None
        req_name = data.get("name")
        req_recipe_original = data.get("recipe")

        # [P1-NEW-11 · 2026-05-11] Pre-LLM dedup check. Cierra el modo
        # de fallo "quota burn por click duplicado":
        #   1. Usuario clickea "Expandir receta" → request #1 en vuelo.
        #   2. User clickea otra vez (impaciencia / doble click → 100-300ms).
        #   3. Request #2 llega al backend. ANTES de P1-NEW-11, ambos
        #      pasaban verify_api_quota, ambos llamaban log_api_usage,
        #      ambos llamaban expand_recipe_agent → 2× tokens Gemini.
        #
        # Fix: pre-check del meal target. Si ya tiene `isExpanded=True`
        # AND su `recipe` ya no es la original (es la expansion previa),
        # retornar early con la recipe cached. NO grabar log_api_usage
        # (cuota no consumida).
        #
        # Solo activo si tenemos plan_id+day_index+meal_index del cliente
        # (cliente moderno post-P1-HIST-RECIPE-1 los manda). Si faltan,
        # caemos al path legacy (mantiene compat con clientes viejos).
        #
        # NO cierra el 5% residual de requests CONCURRENTES exactos (ambos
        # leen pre-fix antes de que el primero haga commit). Para ese
        # caso: app_kv_store advisory lock similar a P3-NEW-9 — fuera
        # del scope de este P-fix porque YAGNI bajo telemetría real.
        if (
            user_id and user_id != "guest"
            and req_plan_id
            and req_day_index is not None
            and req_meal_index is not None
        ):
            try:
                from db_core import execute_sql_query as _exec_q_dedup
                pre_row = _exec_q_dedup(
                    "SELECT plan_data->'days'->%s->'meals'->%s AS meal FROM meal_plans WHERE id = %s AND user_id = %s",
                    (req_day_index, req_meal_index, req_plan_id, user_id),
                    fetch_one=True,
                )
                if pre_row and isinstance(pre_row.get("meal"), dict):
                    existing_meal = pre_row["meal"]
                    # [P2-RECIPE-DEDUP-LIST · 2026-05-30] `recipe` se persiste
                    # SIEMPRE como `List[str]` (MealModel.recipe + el persist
                    # escribe `expanded_steps` lista en plans.py). El check previo
                    # `isinstance(..., str)` JAMÁS matcheaba → el early-return
                    # dedup era código muerto y cada cook-click duplicado (tras
                    # reload / cache-miss / cross-device) re-quemaba `log_api_usage`
                    # + una llamada Gemini. `isinstance(..., list)` + la comparación
                    # `!= req_recipe_original` (element-wise sobre listas) dispara
                    # el dedup correctamente cuando la receta ya fue expandida.
                    already_expanded = (
                        existing_meal.get("isExpanded") is True
                        and existing_meal.get("name") == req_name
                        and isinstance(existing_meal.get("recipe"), list)
                        and existing_meal["recipe"] != req_recipe_original
                    )
                    if already_expanded:
                        logger.info(
                            f"[P1-NEW-11] recipe_expand dedup: meal "
                            f"day={req_day_index} meal={req_meal_index} "
                            f"plan={req_plan_id} ya tiene isExpanded=True. "
                            f"Skip LLM call, return cached recipe."
                        )
                        return {
                            "success": True,
                            "expanded_recipe": existing_meal["recipe"],
                            "skipped_llm": True,
                            "skip_reason": "already_expanded",
                        }
            except Exception as _dedup_err:
                # Best-effort: si el dedup falla por DB, caemos al path
                # normal (mejor quemar quota una vez extra que abortar
                # el endpoint entero).
                logger.debug(
                    f"[P1-NEW-11] dedup pre-check falló (best-effort): "
                    f"{_dedup_err}"
                )

        # [P1-RECIPE-EXPAND-FAILSIGNAL · 2026-05-30] Llamar al Chef AI ANTES de
        # cobrar cuota. `expand_recipe_agent` devuelve `None` cuando la
        # expansión NO produjo contenido nuevo válido (excepción Gemini,
        # circuit-breaker, respuesta vacía/no-lista). Pre-fix: (1) `log_api_usage`
        # se invocaba ANTES de saber el resultado → un fallo cobraba un crédito
        # del paywall (free=15) sin entregar receta de chef; (2) el helper
        # devolvía la receta original en fallo y el endpoint marcaba
        # `isExpanded=True` igual → el guard del frontend jamás reintentaba.
        # Ahora: en fallo NO cobramos, NO persistimos y NO marcamos isExpanded —
        # devolvemos la original para display con `success=False` para que el
        # frontend abra el original SIN el flag (permitiendo retry posterior).
        expanded_steps = expand_recipe_agent(data)

        if not expanded_steps:
            logger.warning(
                "[P1-RECIPE-EXPAND-FAILSIGNAL] expand_recipe_agent devolvió None/vacío "
                f"para meal='{req_name}' plan={req_plan_id}. Sin cobro de cuota, sin "
                "persistencia, sin marcar isExpanded — devolviendo original para display."
            )
            return {
                "success": False,
                "expansion_failed": True,
                "expanded_recipe": req_recipe_original or [],
                "detail": "El Chef AI no pudo detallar esta receta ahora. Mostrando la versión original; intenta de nuevo en un momento.",
            }

        # Éxito real: cobrar cuota ahora (no antes — ver nota arriba).
        if user_id and user_id != "guest":
            log_api_usage(user_id, "llm_recipe_expand")

        if user_id and user_id != "guest":
            # [P1-HIST-RECIPE-1] Resolver el plan target. Si el cliente
            # envía `plan_id`, lo usamos con ownership check explícito.
            # Sin esto un atacante podría persistir contra un plan ajeno
            # (aunque user_id == verified_user_id arriba lo limita, el
            # check por id evita confusión cross-plan del mismo usuario).
            target_plan_id = None
            target_plan_data = None
            if req_plan_id:
                from db_core import execute_sql_query as _exec_q
                row = _exec_q(
                    "SELECT id, plan_data FROM meal_plans WHERE id = %s AND user_id = %s",
                    (req_plan_id, user_id),
                    fetch_one=True,
                )
                if row:
                    target_plan_id = row.get("id")
                    target_plan_data = row.get("plan_data") or {}
                else:
                    # plan_id no resoluble (no existe o no le pertenece).
                    # Fallback a latest pero loguear para detectar abuso.
                    logger.warning(
                        "[P1-HIST-RECIPE-1] plan_id=%s no resoluble para user=%s; fallback a latest",
                        req_plan_id, user_id,
                    )
            if target_plan_data is None:
                plan_with_id = get_latest_meal_plan_with_id(user_id)
                if plan_with_id and "id" in plan_with_id:
                    target_plan_id = plan_with_id.get("id")
                    target_plan_data = plan_with_id.get("plan_data") or {}

            if target_plan_data and isinstance(target_plan_data.get("days"), list) and target_plan_id:
                # [P1-AUDIT-1 · 2026-05-15] Mutación + persistencia atómica
                # bajo FOR UPDATE row lock (vía `update_plan_data_atomic`).
                # Cierre del follow-up natural documentado en
                # P1-RECALC-LOSTUPDATE (2026-05-14):
                #
                # Pre-fix flow:
                #   t=0  SELECT plan_data (línea ~3013, fuera de cualquier lock).
                #   t=1  Muta `target_plan_data["days"][...]["meals"][...]`
                #        in-memory marcando `recipe`/`isExpanded` por uno o
                #        múltiples meals (Camino 1 índices o Camino 2
                #        propagación).
                #   t=2  acquire advisory lock + UPDATE full-overwrite via
                #        `update_meal_plan_data` (P1-NEXT-1).
                #
                # Ventana lost-update entre t=0 y t=2: si un endpoint hermano
                # (`/swap-meal/persist`, `/grocery-start-date`, `/{plan_id}/name`,
                # `/recalculate-shopping-list`, `_chunk_worker` T2) mutaba
                # `plan_data` quirúrgico con su propio lock entre t=0 y t=2,
                # recipe-expand UPDATEaba full-overwrite con la copia stale →
                # la mutación del hermano se perdía silenciosamente.
                #
                # Fix: `update_plan_data_atomic` (P0-2, db_plans.py) toma
                # `SELECT … FOR UPDATE` row lock + re-SELECT fresh + callback +
                # UPDATE, todo en la misma transacción. El callback opera
                # sobre la copia POST-merge y solo toca los meals que la
                # expansión semánticamente posee — todo lo demás (días no
                # afectados, otras keys del plan_data) se preserva del fresh.
                #
                # Decisiones de producto preservadas (NO modificadas en esta
                # migración, solo movidas al callback):
                #
                # [P3-NEW-1 · 2026-05-10] DECISIÓN: NO bumpear
                # `plan_data._plan_modified_at` aquí. Argumento aplicado:
                #   1. La expansión es idempotente (`isExpanded=True` previene
                #      re-expansión sobre la misma receta).
                #   2. Cada cook-click genera una expansión. Bumpear el path
                #      reordenaría el plan al tope del Historial en cada
                #      cook-click — ruidoso y engañoso (interacción ≠ modif).
                #   3. `meal_plans.updated_at` (columna física, trigger P0-2)
                #      sí se actualiza por el UPDATE.
                #
                # [P1-NEW-7 · 2026-05-11] RESTAURADO. La persistencia se borró
                # accidentalmente en P3-NEW-1; tras restaurarla, los tests
                # `test_p1_new_7_recipe_expand_persists.py` codifican el
                # contrato (existencia del call + kwarg user_id).
                #
                # Tooltip-anchor: P1-AUDIT-1-RECIPE-EXPAND-START |
                # test_p1_audit_1_recipe_expand_lostupdate
                from db_plans import update_plan_data_atomic

                def _apply_recipe_expansion(plan_data_fresh: dict) -> dict | bool:
                    """Aplica la expansión sobre `plan_data_fresh` (copia fresh
                    re-SELECTada bajo FOR UPDATE row lock). Re-evalúa ambos
                    caminos (índices vs propagación por contenido) contra
                    `plan_data_fresh.days`, NO contra la copia leída fuera del
                    lock. Si nada se mutó (e.g. swap concurrente reemplazó la
                    receta), retorna `False` para abortar el UPDATE sin
                    persistir.
                    """
                    days_fresh = plan_data_fresh.get("days") if isinstance(plan_data_fresh, dict) else None
                    if not isinstance(days_fresh, list):
                        return False

                    updated_in_callback = False

                    # Camino 1: targeting preciso por (day_index, meal_index).
                    # Sólo se activa si los índices son consistentes con `name`
                    # — si el cliente envió índices stale (chunk worker reordenó
                    # días entre cook-click y request), caemos al match por
                    # contenido para no escribir en la posición equivocada.
                    if (
                        req_day_index is not None
                        and req_meal_index is not None
                        and 0 <= req_day_index < len(days_fresh)
                    ):
                        day = days_fresh[req_day_index]
                        meals = day.get("meals", []) if isinstance(day, dict) else []
                        if 0 <= req_meal_index < len(meals):
                            target_meal_fresh = meals[req_meal_index]
                            if isinstance(target_meal_fresh, dict) and target_meal_fresh.get("name") == req_name:
                                target_meal_fresh["recipe"] = expanded_steps
                                target_meal_fresh["isExpanded"] = True
                                updated_in_callback = True

                    # Camino 2: si no targeteamos por índices, propagar a TODAS
                    # las ocurrencias de `name` cuya receta original sea bit-
                    # equivalente a la del request. La equivalencia evita pisar
                    # un meal con nombre igual pero contenido distinto (e.g.,
                    # "Pollo guisado" del lunes vs el del jueves cuando el
                    # corrector swapea ingredientes). Sin esta propagación,
                    # cada ocurrencia repetida vuelve a quemar cuota LLM.
                    if not updated_in_callback:
                        for day in days_fresh:
                            if not isinstance(day, dict):
                                continue
                            for m in day.get("meals", []):
                                if not isinstance(m, dict):
                                    continue
                                if m.get("name") == req_name and m.get("recipe") == req_recipe_original:
                                    m["recipe"] = expanded_steps
                                    m["isExpanded"] = True
                                    updated_in_callback = True
                                    # NO break: propagamos a todas las ocurrencias.

                    if not updated_in_callback:
                        # Nada que persistir — abortar UPDATE (P0-2 contract:
                        # `return False` cancela el UPDATE; caller ve `current`
                        # sin escribir).
                        return False

                    return plan_data_fresh

                # [P2-OPEN-1] user_id ya validado al inicio del handler
                # (verified_user_id == user_id). target_plan_id viene del
                # SELECT explícito con `AND user_id = %s` (línea ~3014) o de
                # `get_latest_meal_plan_with_id(user_id)` (línea ~3029) —
                # ambos filtran por user_id. Defense-in-depth doble candado
                # al pasar user_id al helper.
                update_plan_data_atomic(
                    target_plan_id, _apply_recipe_expansion, user_id=user_id
                )
        # P1-HIST-RECIPE-1-END

        return {"success": True, "expanded_recipe": expanded_steps}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/recipe/expand: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

@router.post("/swap-meal")
def api_swap_meal(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota), _rl: None = Depends(_SWAP_LIMITER)):  # [P2-GUEST-LLM-RATELIMIT · 2026-05-30] throttle guest LLM
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
            background_tasks.add_task(_process_swap_rejection_background, session_id, user_id, rejected_meal, meal_type)  # pyright: ignore[reportArgumentType]
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
            log_api_usage(user_id, "llm_swap_meal")
            
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
    except HTTPException:
        raise
    except ValueError as ve:
        # [P3-SWAP-SOFT-FAIL-200 · 2026-05-23] Antes los swap failures
        # (`SWAP_STRICT_PANTRY_NO_INVENTORY` y `SWAP_LLM_RETRIES_EXHAUSTED`)
        # respondían HTTP 422. Eso es REST-correcto pero genera ruido
        # cosmético en DevTools del browser (`Failed to load resource:
        # 422 (Unprocessable Entity)` en rojo), confundiendo al developer
        # cuando inspecciona el comportamiento normal del fallback.
        #
        # Soft-fail pattern: retornamos HTTP 200 con `swap_failed=true` +
        # `error_code` canónico + `error_message`. Frontend checkea el
        # flag ANTES de procesar como plato exitoso. Mismo UX final
        # (toast + plato preservado) pero sin ruido visual en DevTools.
        # Knob `MEALFIT_SWAP_HARD_FAIL_HTTP_422=true` revierte al
        # comportamiento anterior si algún integrador externo dependía
        # del status 4xx.
        _msg = str(ve)
        _hard_fail_422 = os.environ.get(
            "MEALFIT_SWAP_HARD_FAIL_HTTP_422", "false"
        ).lower() == "true"

        if _msg.startswith("SWAP_STRICT_PANTRY_NO_INVENTORY"):
            logger.warning(f"⚠️ [P1-SWAP-STRICT-PANTRY] soft-fail → {_msg}")
            _payload = {
                "swap_failed": True,
                "error_code": "swap_strict_pantry_no_inventory",
                "error_message": (
                    "Tu nevera está vacía. Agrega alimentos a tu nevera "
                    "o elige otra razón de cambio (por ejemplo, 'Quiero "
                    "variedad') para que el chef pueda proponer un plato."
                ),
            }
            if _hard_fail_422:
                raise HTTPException(status_code=422, detail={
                    "code": _payload["error_code"],
                    "message": _payload["error_message"],
                })
            return _payload

        if _msg.startswith("SWAP_LLM_RETRIES_EXHAUSTED"):
            logger.warning(f"⚠️ [P3-SWAP-LLM-RETRIES] soft-fail → {_msg}")
            _payload = {
                "swap_failed": True,
                "error_code": "swap_llm_retries_exhausted",
                "error_message": (
                    "El chef IA no pudo generar una alternativa coherente "
                    "tras varios intentos. Reintenta o elige otra razón "
                    "de cambio. Tu plato actual se mantiene sin cambios."
                ),
            }
            if _hard_fail_422:
                raise HTTPException(status_code=422, detail={
                    "code": _payload["error_code"],
                    "message": _payload["error_message"],
                })
            return _payload

        logger.error(f"❌ [ERROR] Error en /api/swap-meal: {_msg}")
        raise HTTPException(status_code=500, detail=safe_error_detail(ve))
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/swap-meal: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/{plan_id}/swap-meal/persist")
def api_swap_meal_persist(
    plan_id: str,
    data: dict = Body(...),
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """[P0-NEW-A · 2026-05-11 | P1-SWAP-PERSIST-ATOMIC · 2026-05-22]
    Persistencia atómica de un meal swap.

    Reemplaza el patrón frontend-only en `AssessmentContext.jsx:1240-1243`
    que hacía ``supabase.from('meal_plans').update({plan_data}).eq('id', planId)``
    pisando el JSONB completo. Ese patrón producía lost-update:
    si `_chunk_worker` finalizaba un chunk entre que el frontend leyó
    `planData` y escribió, los `days[7-14]` recién persistidos por el
    worker (y `_chunk_lessons`, `aggregated_shopping_list`, etc.) se
    perdían porque el frontend reescribía el snapshot viejo del state
    local.

    Implementación actual (P1-SWAP-PERSIST-ATOMIC · 2026-05-22):
    `update_plan_data_atomic(plan_id, _swap_mutator, user_id=...)` —
    `SELECT … FOR UPDATE` + mutator callback que opera sobre plan_data
    fresh post-worker. El mutator muta SOLO `days[i].meals[j]`, bumpea
    `_plan_modified_at` y strippea los 4 `aggregated_shopping_list*`
    para forzar recalc downstream (el frontend invoca
    `/recalculate-shopping-list` inmediatamente después, igual que pre-fix).
    Implementación pre-fix usaba `execute_sql_write` con jsonb_set chained
    (sin row lock); ahora estructuralmente protegida por FOR UPDATE.

    Defensa-en-profundidad doble: ownership check explícito antes del
    helper atómico + el `update_plan_data_atomic` interno filtra
    `AND user_id = %s` en el SELECT FOR UPDATE y en el UPDATE final
    (espejo de `/retry-chunk` P0-HIST-IDOR-1 y `/recipe/expand`
    P1-HIST-RECIPE-1).

    Body:
      - ``day_index`` (int 0..99): índice del día.
      - ``meal_index`` (int 0..19): índice del meal dentro del día.
      - ``new_meal`` (dict): meal completo a sustituir (incluye
        ``name``/``desc``/``cals``/``prep_time``/``recipe``/``ingredients``
        — el caller construye el merge igual que el frontend pre-fix).
      - ``clear_is_restocked`` (bool, opcional): si ``true``, fuerza
        ``plan_data.is_restocked = false`` en el mismo UPDATE. Lo usa
        el frontend cuando detecta que el meal nuevo introduce
        ingredientes uncovered post-restock (lógica P0-1 conservada
        client-side; el backend solo aplica la decisión).

    Returns:
      ``{ "success": true }``

    Raises:
      401 — sin auth.
      400 — parámetros faltantes/inválidos.
      404 — plan no existe o no pertenece al usuario (no se filtra
            existencia: mismo patrón que ``/retry-chunk``).

    Tooltip-anchor: P0-NEW-A-START | test_p0_new_a_swap_persists_backend
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not plan_id or not isinstance(plan_id, str):
        raise HTTPException(status_code=400, detail="plan_id required")

    body = data or {}
    day_index = body.get("day_index")
    meal_index = body.get("meal_index")
    new_meal = body.get("new_meal")
    clear_is_restocked = bool(body.get("clear_is_restocked", False))

    # Validación estricta de bounds. Los caps (100 días, 20 meals/día)
    # cubren el peor caso legítimo (planes mensuales × 4 ciclos × 5
    # meals = 20 max). Valores fuera son síntoma de bug cliente o
    # intento de path-injection en el jsonb_set (los enteros validados
    # se interpolan al path string sin escaping).
    if not isinstance(day_index, int) or isinstance(day_index, bool) or day_index < 0 or day_index >= 100:
        raise HTTPException(status_code=400, detail="day_index must be int in [0, 99]")
    if not isinstance(meal_index, int) or isinstance(meal_index, bool) or meal_index < 0 or meal_index >= 20:
        raise HTTPException(status_code=400, detail="meal_index must be int in [0, 19]")
    if not isinstance(new_meal, dict):
        raise HTTPException(status_code=400, detail="new_meal must be a dict")
    new_meal_name = new_meal.get("name")
    if not isinstance(new_meal_name, str) or not new_meal_name.strip():
        raise HTTPException(status_code=400, detail="new_meal.name is required")
    # [P3-PROD-AUDIT-2 · 2026-05-30] Cap de tamaño del meal client-controlled antes
    # del jsonb_set (un meal legítimo ≈ pocos KB). Sin cap propio se persistía
    # verbatim hasta el cap global. Knob compartido con /restore-local.
    try:
        import json as _json_cap
        if len(_json_cap.dumps(new_meal)) > _env_int("MEALFIT_MAX_PLAN_DATA_BYTES", 2_097_152):
            raise HTTPException(status_code=413, detail="new_meal demasiado grande.")
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="new_meal no serializable.")

    from db_core import execute_sql_query
    from db_plans import update_plan_data_atomic
    try:
        owner = execute_sql_query(
            "SELECT id FROM meal_plans WHERE id = %s AND user_id = %s",
            (plan_id, verified_user_id),
            fetch_one=True,
        )
        if not owner:
            # 404 (no 403) para no filtrar existencia del plan ajeno —
            # patrón espejo de `DELETE /{plan_id}` y `/retry-chunk`.
            raise HTTPException(status_code=404, detail="Plan no encontrado")

        # [P1-SWAP-PERSIST-ATOMIC · 2026-05-22] Migrado el UPDATE plano
        # (`execute_sql_write` con `jsonb_set` chained) a
        # `update_plan_data_atomic` (db_plans.py:381, P0-2 / P2-OPEN-1).
        #
        # Pre-fix: el SQL legacy usaba `jsonb_set` quirúrgico SIN tomar
        # row lock. Si `_chunk_worker` finalizaba un chunk EN PARALELO
        # mientras el cliente swap-eaba un meal del MISMO `days[i]`, la
        # garantía era casi siempre OK (jsonb_set en server-side
        # serializa transacciones por row a nivel Postgres) pero la
        # ventana de write-after-read del lado server NO era
        # estructuralmente protegida — viola I7 en espíritu para
        # cualquier mutación que toque keys overlap (e.g., si el worker
        # también escribiera `_plan_modified_at` o si un futuro refactor
        # cambia a full-overwrite por error, sin advisory lock).
        #
        # `update_plan_data_atomic` serializa contra concurrentes vía
        # `SELECT … FOR UPDATE` + mutator callback con `plan_data` fresh
        # post-worker. El mutator muta SOLO `days[i].meals[j]` (espejo
        # semántico del jsonb_set quirúrgico) + strip de las 4 listas
        # aggregated_shopping_list* + bump de `_plan_modified_at` + flag
        # `is_restocked` opcional. Defensa-en-profundidad: el UPDATE
        # interno del helper también incluye `AND user_id = %s` (espejo
        # del SELECT inicial). Tooltip-anchor: P1-SWAP-PERSIST-ATOMIC.
        def _swap_mutator(plan_data: dict) -> dict:
            # Sanity: el plan real DEBE tener `days[day_index].meals[meal_index]`.
            # Si el plan está corrupted (días faltantes) levantamos
            # IndexError/ValueError para que el handler mapee a 400.
            days = plan_data.get("days")
            if not isinstance(days, list):
                raise ValueError(
                    f"plan_data.days corrupted (type={type(days).__name__})"
                )
            if day_index >= len(days):
                raise IndexError(
                    f"day_index={day_index} fuera de rango (plan tiene "
                    f"{len(days)} días)"
                )
            day = days[day_index]
            if not isinstance(day, dict):
                raise ValueError(
                    f"plan_data.days[{day_index}] corrupted "
                    f"(type={type(day).__name__})"
                )
            meals = day.get("meals")
            if not isinstance(meals, list):
                meals = []
                day["meals"] = meals
            # `create_missing` semantics del jsonb_set legacy: si
            # meal_index apunta justo a `len(meals)` (o más), rellenar
            # con `{}` hasta llegar. Mismo comportamiento que `jsonb_set`
            # sobre array path con `create_missing=true`.
            while len(meals) <= meal_index:
                meals.append({})
            # [P2-SWAP-RESET-ISEXPANDED · 2026-05-30] Defensa-en-profundidad: un
            # plato swapeado trae su receta base sin expandir; jamás debe heredar
            # `isExpanded:true` (cerraría el botón "Cocinar"→expand del frontend).
            # El cliente ya envía isExpanded:false (AssessmentContext); forzarlo
            # aquí cubre clientes futuros/legacy que omitan el reset.
            if isinstance(new_meal, dict):
                new_meal["isExpanded"] = False
            meals[meal_index] = new_meal

            # Strip las 4 aggregated_shopping_list* para forzar recalc
            # downstream (mismo contrato que el SQL legacy — el frontend
            # invoca `/recalculate-shopping-list` inmediatamente después).
            for _k in (
                "aggregated_shopping_list",
                "aggregated_shopping_list_weekly",
                "aggregated_shopping_list_biweekly",
                "aggregated_shopping_list_monthly",
            ):
                plan_data.pop(_k, None)

            # Sello CAS semántico (P0-3) que el sort del Historial usa.
            from datetime import datetime, timezone
            plan_data["_plan_modified_at"] = datetime.now(
                timezone.utc
            ).isoformat()

            if clear_is_restocked:
                plan_data["is_restocked"] = False

            return plan_data

        result = update_plan_data_atomic(
            plan_id,
            _swap_mutator,
            user_id=verified_user_id,
        )
        if not result:
            # Row desapareció entre el SELECT inicial y el FOR UPDATE
            # (race contra `DELETE /{plan_id}`). Tratamos como 404.
            raise HTTPException(
                status_code=404, detail="Plan no encontrado"
            )
        return {"success": True}
        # P0-NEW-A-END / P1-SWAP-PERSIST-ATOMIC-END

    except HTTPException:
        raise
    except (IndexError, ValueError) as e:
        # day_index/meal_index fuera de rango del plan real (no del bound
        # check inicial — eso es 400 más arriba con detail explícito) o
        # plan_data corrupted. Mapeo 400 con copy específico.
        raise HTTPException(
            status_code=400,
            detail=f"Inconsistencia en plan_data: {e}",
        )
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /swap-meal/persist: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/{plan_id}/grocery-start-date")
def api_set_grocery_start_date(
    plan_id: str,
    data: dict = Body(...),
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """[P0-NEW-B · 2026-05-11] Persistencia idempotente de
    `grocery_start_date` y/o `cycle_start_date` para el plan activo.

    Reemplaza el patrón frontend-only en `AssessmentContext.jsx:560`
    que hacía ``supabase.from('meal_plans').update({plan_data: latestPlan}).eq('id', planId)``
    pisando el JSONB completo al inyectar las fechas. Ese patrón
    producía el MISMO lost-update que P0-NEW-A cerró en swap: si el
    cron ``_resolve_grocery_start_date`` (cron_tasks.py:15327) o
    ``_chunk_worker`` mutaba `plan_data` en paralelo (e.g.,
    actualizando ``days[*]`` o ``_chunk_lessons``), el frontend pisaba
    todo con el snapshot stale del state local.

    Esta ruta hace `jsonb_set` quirúrgico solo sobre las dos keys de
    fecha + idempotencia `WHERE (plan_data->>'<key>') IS NULL` (mismo
    patrón del cron) + `AND user_id = %s` defense-in-depth + bump
    `updated_at = NOW()`. Si las keys ya tienen valor, el UPDATE
    afecta 0 filas y devolvemos 200 igual (idempotente, no-op).

    Body:
      - ``grocery_start_date`` (str ISO, opcional): fecha objetivo. Solo
        se persiste si ``plan_data->>'grocery_start_date'`` es NULL.
      - ``cycle_start_date`` (str ISO, opcional): fecha inmutable del
        contador `daysLeft`. Misma condición de idempotencia.

    Returns:
      ``{ "success": true, "grocery_updated": bool, "cycle_updated": bool }``
      donde los booleans reflejan si la fila fue tocada (0 rows = false).

    Raises:
      401 — sin auth.
      400 — plan_id faltante o ambas fechas vacías.
      404 — plan no existe o no pertenece al usuario.

    Tooltip-anchor: P0-NEW-B-START | test_p0_new_b_grocery_date_persists_backend
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not plan_id or not isinstance(plan_id, str):
        raise HTTPException(status_code=400, detail="plan_id required")

    body = data or {}
    grocery_iso = body.get("grocery_start_date")
    cycle_iso = body.get("cycle_start_date")

    # Validación de strings ISO. Aceptamos cualquier shape ISO razonable
    # (`YYYY-MM-DD`, `YYYY-MM-DDTHH:MM:SS...`) — el cron también pasa
    # `created_at` formateado con `isoformat()`. Lo que NO aceptamos es
    # int/None forzados a string sin sentido.
    import re as _re  # import local: el resto de plans.py no usa re.
    def _valid_iso(v) -> bool:
        if not isinstance(v, str):
            return False
        s = v.strip()
        # Floor mínimo: `YYYY-MM-DD` (10 chars). Cap razonable 64 chars.
        if len(s) < 10 or len(s) > 64:
            return False
        # Prefijo de 10 chars debe matchear el patrón YYYY-MM-DD para
        # rechazar strings arbitrarios. No parseamos con `datetime` para
        # mantener el endpoint barato y porque el cron también acepta
        # variants con timezone que `fromisoformat` no siempre tolera.
        return bool(_re.match(r"^\d{4}-\d{2}-\d{2}", s))

    grocery_valid = _valid_iso(grocery_iso)
    cycle_valid = _valid_iso(cycle_iso)

    if not grocery_valid and not cycle_valid:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least one of `grocery_start_date` or `cycle_start_date` "
                "must be provided as ISO date string (YYYY-MM-DD...)"
            ),
        )

    from db_core import execute_sql_query, execute_sql_write
    try:
        owner = execute_sql_query(
            "SELECT id FROM meal_plans WHERE id = %s AND user_id = %s",
            (plan_id, verified_user_id),
            fetch_one=True,
        )
        if not owner:
            raise HTTPException(status_code=404, detail="Plan no encontrado")

        # UPDATE por key separado. Ventajas vs. un UPDATE combinado con
        # COALESCE:
        #   - Cada UPDATE es idempotente independiente (si una key ya
        #     tiene valor, su UPDATE afecta 0 filas y no toca el resto).
        #   - Cambios futuros en la lista de keys no requieren reescribir
        #     el SQL combinado.
        # Coste: 2 round-trips a DB en lugar de 1, pero el endpoint corre
        # solo en path de hidratación inicial del plan (no hot loop) —
        # diferencia despreciable.
        grocery_updated = False
        cycle_updated = False

        if grocery_valid:
            execute_sql_write(
                """
                UPDATE meal_plans
                SET plan_data = jsonb_set(
                        COALESCE(plan_data, '{}'::jsonb),
                        '{grocery_start_date}',
                        to_jsonb(%s::text),
                        true
                    ),
                    updated_at = NOW()
                WHERE id = %s
                  AND user_id = %s
                  AND (plan_data->>'grocery_start_date') IS NULL
                """,
                (grocery_iso, plan_id, verified_user_id),
            )
            # Re-check para reportar si la fila fue tocada. Idempotente
            # ante runs paralelos (otro cron puede haber rellenado).
            after = execute_sql_query(
                "SELECT (plan_data->>'grocery_start_date') AS v "
                "FROM meal_plans WHERE id = %s AND user_id = %s",
                (plan_id, verified_user_id),
                fetch_one=True,
            )
            grocery_updated = bool(after and after.get("v"))

        if cycle_valid:
            execute_sql_write(
                """
                UPDATE meal_plans
                SET plan_data = jsonb_set(
                        COALESCE(plan_data, '{}'::jsonb),
                        '{cycle_start_date}',
                        to_jsonb(%s::text),
                        true
                    ),
                    updated_at = NOW()
                WHERE id = %s
                  AND user_id = %s
                  AND (plan_data->>'cycle_start_date') IS NULL
                """,
                (cycle_iso, plan_id, verified_user_id),
            )
            after = execute_sql_query(
                "SELECT (plan_data->>'cycle_start_date') AS v "
                "FROM meal_plans WHERE id = %s AND user_id = %s",
                (plan_id, verified_user_id),
                fetch_one=True,
            )
            cycle_updated = bool(after and after.get("v"))

        return {
            "success": True,
            "grocery_updated": grocery_updated,
            "cycle_updated": cycle_updated,
        }
        # P0-NEW-B-END

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /grocery-start-date: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/{plan_id}/restore-local")
def api_restore_plan_local(
    plan_id: str,
    data: dict = Body(...),
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """[P1-OPEN-1 · 2026-05-11] Restauración atómica de un snapshot local
    (revertir un regen rechazado) reemplazando el último direct-write del
    frontend a `meal_plans`.

    Reemplaza el patrón legacy en `AssessmentContext.jsx:1514-1517` que hacía
    ``supabase.from('meal_plans').update({plan_data, name, calories, macros})
    .eq('id', planId)`` desde el cliente. Ese era el ÚNICO callsite cliente
    que persistía full-overwrite a `meal_plans`. Modo de fallo:
        - Lost-update vs `_chunk_worker` concurrente: si el worker
          finalizaba un chunk (días 7-14, `_chunk_lessons`,
          `aggregated_shopping_list`) entre que el cliente leyó el state
          local y este UPDATE, los datos del worker se perdían bajo el
          snapshot stale del cliente.
        - Violaba invariante I6 (CLAUDE.md): "mutaciones a `plan_data` desde
          el frontend prohibidas — solo via endpoint backend".
        - Violaba I7 (CLAUDE.md): "todo full-overwrite a `plan_data` DEBE
          tomar `acquire_meal_plan_advisory_lock(purpose='general')`".

    Este handler:
      1. Verifica ownership con `SELECT WHERE id = %s AND user_id = %s`
         (404 si no resoluble — no leak de existencia cross-user, mismo
         patrón que `/swap-meal/persist`, `/retry-chunk`, `/restock`).
      2. Toma advisory lock 'general' (mismo `purpose` que T1/T2 del
         chunk worker, `/shift-plan`, `_background_shift_plan_for_user`)
         para serializar contra writers concurrentes.
      3. Bumpea `_plan_modified_at` server-side en el snapshot (semántica:
         la restauración ES un evento de modificación lógica del plan; el
         Historial usa ese path para el sort).
      4. Anota `_restored_from_local_at` para observabilidad de cuántas
         veces el usuario revierte.
      5. UPDATE atómico de `plan_data` + top-level columns derivadas
         (`name`/`calories`/`macros`) si vienen en el body. Filtro
         `AND user_id = %s` defense-in-depth (I2).
      6. Bumpea `updated_at = NOW()` explícito (redundante con trigger
         P0-2 pero anclado para refactors).

    Body:
      - ``plan_data`` (dict, requerido): snapshot completo del plan a
        restaurar (el state local del cliente).
      - ``name`` (str, opcional): si presente y no vacío, sobrescribe la
        columna top-level.
      - ``calories`` (number, opcional): si numérico finito, sobrescribe.
      - ``macros`` (dict, opcional): si dict, sobrescribe la columna jsonb.

    Returns:
      ``{ "success": true }``

    Raises:
      401 — sin auth.
      400 — `plan_data` faltante o no es dict.
      404 — plan no existe o no pertenece al usuario.

    Tooltip-anchor: P1-OPEN-1-START | test_p1_open_1_restore_local_endpoint
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not plan_id or not isinstance(plan_id, str):
        raise HTTPException(status_code=400, detail="plan_id required")

    body = data or {}
    past_plan_data = body.get("plan_data")
    if not isinstance(past_plan_data, dict):
        raise HTTPException(status_code=400, detail="plan_data must be a dict")
    # [P3-PROD-AUDIT-2 · 2026-05-30] Cap de tamaño del blob JSONB client-controlled
    # (un plan legítimo de 30 días ≈ decenas-cientos KB). Sin cap propio, se
    # persistía verbatim hasta ~25 MiB (cap global) → lectores downstream (chunk
    # worker, recalc, PDF) re-procesan un blob enorme. Blast-radius limitado al
    # propio plan, pero el cap corta abusos absurdos. Knob clampeado.
    try:
        import json as _json_cap
        if len(_json_cap.dumps(past_plan_data)) > _env_int("MEALFIT_MAX_PLAN_DATA_BYTES", 2_097_152):
            raise HTTPException(status_code=413, detail="plan_data demasiado grande.")
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="plan_data no serializable.")

    # Top-level derivados opcionales. Validación estricta para no aceptar
    # bogus que `Dashboard.jsx` interprete como header inválido.
    raw_name = body.get("name")
    new_name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None

    raw_cals = body.get("calories")
    new_calories = None
    if isinstance(raw_cals, (int, float)) and not isinstance(raw_cals, bool):
        import math as _math_p1o1
        try:
            _cf = float(raw_cals)
            if _math_p1o1.isfinite(_cf):
                new_calories = _cf
        except Exception:
            new_calories = None

    raw_macros = body.get("macros")
    new_macros = raw_macros if isinstance(raw_macros, dict) else None

    # Server-side bump del CAS marker semántico + breadcrumb de restauración.
    # Shallow copy: solo añadimos 2 keys al top-level del JSONB; los nested
    # dicts (days, meals, ingredients...) se preservan por referencia.
    from datetime import datetime as _dt_p1o1, timezone as _tz_p1o1
    plan_data_to_write = dict(past_plan_data)
    _now_iso = _dt_p1o1.now(_tz_p1o1.utc).isoformat()
    plan_data_to_write["_plan_modified_at"] = _now_iso
    plan_data_to_write["_restored_from_local_at"] = _now_iso

    from db_core import connection_pool
    from db_plans import acquire_meal_plan_advisory_lock, set_meal_plan_for_update_timeouts
    from psycopg.rows import dict_row
    try:
        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cursor:
                    # [P1-LOCK-1] Bound lock wait + statement timeout antes
                    # de adquirir locks (mismo patrón que `/shift-plan`).
                    set_meal_plan_for_update_timeouts(cursor)

                    # 1) Ownership check + existencia. 404 (no 403) para no
                    #    filtrar existencia cross-user.
                    cursor.execute(
                        "SELECT id FROM meal_plans WHERE id = %s AND user_id = %s",
                        (plan_id, verified_user_id),
                    )
                    if not cursor.fetchone():
                        raise HTTPException(status_code=404, detail="Plan no encontrado")

                    # 2) Advisory lock 'general' (I7): serializa contra T1/T2
                    #    del chunk worker, /shift-plan, /restore. El lock se
                    #    libera al cerrar la transacción (commit o rollback).
                    acquire_meal_plan_advisory_lock(cursor, plan_id, purpose="general")

                    # 3) UPDATE atómico. Columnas top-level se incluyen solo
                    #    si el caller pasó valores válidos — espejo de la
                    #    semántica legacy (no pisar header con None bogus).
                    set_clauses = ["plan_data = %s::jsonb", "updated_at = NOW()"]
                    params: list = [_json.dumps(plan_data_to_write, ensure_ascii=False)]
                    if new_name is not None:
                        set_clauses.append("name = %s")
                        params.append(new_name)
                    if new_calories is not None:
                        set_clauses.append("calories = %s")
                        params.append(new_calories)
                    if new_macros is not None:
                        set_clauses.append("macros = %s::jsonb")
                        params.append(_json.dumps(new_macros, ensure_ascii=False))

                    sql = (
                        f"UPDATE meal_plans SET {', '.join(set_clauses)} "
                        f"WHERE id = %s AND user_id = %s"
                    )
                    params.extend([plan_id, verified_user_id])
                    cursor.execute(sql, tuple(params))  # pyright: ignore[reportArgumentType]

        return {"success": True}
        # P1-OPEN-1-END

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /restore-local: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


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

@router.post("/cancel")
def api_cancel_plan_generation(data: dict = Body(...)):
    """[P1-16] Cancela una generación en vuelo identificada por `session_id`.

    [P6-CANCEL-LOG] Logging visible en terminal del backend para que el dev
    pueda verificar que el POST llega del frontend (debugging común "user
    clickeó cancel pero backend sigue generando" — necesitamos saber si el
    POST llegó O si solo SSE se cerró localmente).

    El frontend llama este endpoint desde `cancelGeneration()` en
    `Plan.jsx` ANTES (o en paralelo a) abortar el SSE del lado cliente.
    Sin este endpoint:
      - El SSE se aborta en el cliente (el reader cierra).
      - PERO el pipeline backend sigue corriendo hasta terminar el día
        actual y persistir el plan en DB.
      - El usuario ve el plan aparecer 30s después vía Realtime UPDATE
        de `meal_plans` aunque ya había cancelado.
      - Cuota de LLM consumida innecesariamente.

    Diseño:
      - Acepta cualquier session_id (no requiere auth) para reducir
        latencia y porque el peor caso es DoS de bajo impacto: registrar
        un cancel para un session_id que no existe es no-op cuando el
        pipeline checkea cooperativamente.
      - Retorna `{success: true, registered: bool}` — útil para tests.
      - Idempotente: cancelar dos veces el mismo session no falla.

    [P2-PROD-AUDIT-FOLLOWUP · 2026-05-28] DECISIÓN ACEPTADA (no gatear con auth).
    Tooltip-anchor: P2-CANCEL-NO-AUTH-ACCEPTED. El audit prod-readiness flageó
    "endpoint sin auth que cancela + limpia el KV de un session_id". Se acepta
    intencionalmente porque: (1) `session_id` es un UUID generado client-side,
    NO enumerable — el atacante necesitaría conocer el UUID exacto de una
    generación EN VUELO de la víctima (ventana de segundos); (2) el peor caso
    es DoS de un único plan (la víctima re-dispara), sin lectura ni mutación de
    datos; (3) el endpoint lo invocan tanto guests (sin JWT) como usuarios
    autenticados desde `cancelGeneration()` — exigir JWT rompería el flujo guest
    y añadiría latencia justo cuando el usuario quiere abortar. Gatear con auth
    requeriría que `clear_pending_pipeline` fuese ownership-aware (cambio más
    profundo) sin cerrar el vector para guests. Si se decide revertir esta
    decisión, leer esta nota antes de invertir esfuerzo. Análogo al patrón
    "Decisiones de producto" / "Advisors aceptados" de CLAUDE.md.
    """
    session_id = data.get("session_id") if isinstance(data, dict) else None
    if not session_id or not isinstance(session_id, str):
        logger.warning(
            f"⚠️ [P6-CANCEL-LOG] POST /api/plans/cancel sin session_id válido. "
            f"data={data!r}"
        )
        return {"success": False, "message": "session_id requerido"}
    registered = _cancel_session(session_id)
    logger.info(
        f"🛑 [P6-CANCEL-LOG] POST /api/plans/cancel recibido para "
        f"session_id={session_id} (registered_now={registered}). "
        f"event_generator detectará en próxima iteración SSE."
    )
    # [P3-CANCEL-CLEAR-KV · 2026-05-16] Limpiar el row del KV
    # `pending_pipeline:<session_id>` para que el guardrail
    # `check_user_has_active_pipeline` NO bloquee el siguiente intento del
    # mismo user. Pre-fix: el endpoint solo registraba el cancel en memoria
    # (in-process set), pero el row del KV seguía con status='generating'
    # hasta que `MAX_AGE_MIN=15min` lo dejaba pasar — el user que cancelaba y
    # disparaba otro plan inmediatamente recibía 409 `pipeline_already_running`.
    # Best-effort: si la clave no existe (session_id != user_id), no-op
    # silencioso.
    try:
        from db_plans import clear_pending_pipeline
        _cleared = clear_pending_pipeline(session_id)
        if _cleared:
            logger.info(
                f"🧹 [P3-CANCEL-CLEAR-KV] pending_pipeline row limpiado "
                f"para session_id={session_id[:8]} tras cancel explícito."
            )
    except Exception as _clear_err:
        logger.info(
            f"[P3-CANCEL-CLEAR-KV] clear best-effort skipped "
            f"session_id={session_id[:8]}: {_clear_err}"
        )
    return {
        "success": True,
        "registered": registered,
        "session_id": session_id,
    }


# ============================================================
# [P3-DEPLETED-BD · 2026-05-22] Endpoints para `user_depleted_items`.
# ============================================================
# Sustituye el localStorage.mealfit_depleted_items pre-existente que no
# sincronizaba cross-device. Pantry.jsx + AgentPage.jsx leen/escriben acá.
# Auth: `get_verified_user_id` (no `verify_api_quota` — cero costo LLM,
# patrón P1-AUDIT-3 Historial-quota-exemption).


@router.get("/depleted-items")
async def api_list_depleted_items(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """[P3-DEPLETED-BD · 2026-05-22] Lista items agotados del usuario,
    ordenados por `depleted_at DESC` (más reciente primero).

    Response: `{"items": [...]}` con shape per item:
    `{id, master_ingredient_id, ingredient_name, quantity, unit, category,
    shelf_life_days, depleted_at}`. Lista vacía si no hay agotados.
    """
    if not verified_user_id or verified_user_id == "guest":
        return {"items": []}
    try:
        from db_inventory import list_depleted_items
        # [P1-ASYNC-SYNC-DB-BLOCKING · 2026-05-24] handler async + DB sync → to_thread.
        items = await asyncio.to_thread(list_depleted_items, verified_user_id, 300)
        return {"items": items}
    except Exception as e:
        logger.warning(f"[P3-DEPLETED-BD] GET /depleted-items error: {e!r}")
        return {"items": []}


@router.post("/depleted-items")
async def api_upsert_depleted_items(
    data: dict = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    """[P3-DEPLETED-BD · 2026-05-22] Upsert items agotados. Acepta:

      - `{"items": [{ingredient_name, quantity, unit, master_ingredient_id?,
                     category?, shelf_life_days?, depleted_at?}, ...]}`

    Idempotente: usa los unique indexes partial sobre `(user_id,
    master_ingredient_id)` cuando master no es NULL y `(user_id,
    lower(trim(ingredient_name)))` cuando es NULL.

    Path principal:
      1. Frontend Pantry.jsx llama acá cuando el user agota un item desde UI.
      2. Migration one-shot: Pantry.jsx en su primer mount post-deploy
         envía aquí TODO su `localStorage.mealfit_depleted_items` legacy.

    Response: `{"success": true, "upserted": N}` o error.
    """
    if not verified_user_id or verified_user_id == "guest":
        raise HTTPException(status_code=401, detail="Auth requerida.")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise HTTPException(status_code=400, detail="`items` debe ser lista no vacía.")
    if len(items) > 500:
        raise HTTPException(status_code=400, detail="Batch demasiado grande (>500).")
    try:
        from db_inventory import bulk_upsert_depleted_items
        # [P1-ASYNC-SYNC-DB-BLOCKING · 2026-05-24] handler async + DB sync → to_thread.
        n = await asyncio.to_thread(bulk_upsert_depleted_items, verified_user_id, items)
        return {"success": True, "upserted": n}
    except Exception as e:
        logger.warning(f"[P3-DEPLETED-BD] POST /depleted-items error: {e!r}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.delete("/depleted-items")
async def api_delete_all_depleted_items(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """[P3-DELETEALL-DEPLETED · 2026-05-30] Borra TODOS los agotados del user.
    Lo invoca Pantry.jsx::confirmDeleteAll ('Vaciar Nevera') para que los
    recordatorios de "agotado" no reaparezcan al recargar — la fuente de verdad
    es `user_depleted_items` (cross-device), no el localStorage. Pre-fix el
    clear de `_persistDepleted([])` era cosmético y `_fetchAndApply` repoblaba
    los agotados desde BD al próximo mount.

    Returns `{"success": bool, "deleted": N}`. El realtime channel propaga el
    DELETE a otros tabs/devices.
    """
    if not verified_user_id or verified_user_id == "guest":
        raise HTTPException(status_code=401, detail="Auth requerida.")
    try:
        from db_inventory import delete_all_depleted_items
        # [P1-ASYNC-SYNC-DB-BLOCKING · 2026-05-24] handler async + DB sync → to_thread.
        n = await asyncio.to_thread(delete_all_depleted_items, verified_user_id)
        return {"success": True, "deleted": n}
    except Exception as e:
        logger.warning(f"[P3-DELETEALL-DEPLETED] DELETE /depleted-items (all) error: {e!r}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.delete("/depleted-items/{item_id}")
async def api_delete_depleted_item(
    item_id: int,
    verified_user_id: str = Depends(get_verified_user_id),
):
    """[P3-DEPLETED-BD · 2026-05-22] DELETE un item de `user_depleted_items`.

    Frontend lo llama cuando el user restockea (compró el item y desea
    quitar el flag de agotado) o descarta el item del listado.

    Filtro `WHERE user_id = %s` defensa-en-profundidad. Returns
    `{"success": bool, "deleted": bool}`.
    """
    if not verified_user_id or verified_user_id == "guest":
        raise HTTPException(status_code=401, detail="Auth requerida.")
    try:
        from db_inventory import delete_depleted_item
        # [P1-ASYNC-SYNC-DB-BLOCKING · 2026-05-24] handler async + DB sync → to_thread.
        ok = await asyncio.to_thread(delete_depleted_item, verified_user_id, item_id=item_id)
        return {"success": True, "deleted": ok}
    except Exception as e:
        logger.warning(f"[P3-DEPLETED-BD] DELETE /depleted-items/{item_id} error: {e!r}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


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

        # [P3-PROD-AUDIT-2 · 2026-05-30] Cap de longitud (espejo de /depleted-items:500).
        # Sin él, ~50k-500k items bajo el cap global de 25 MiB → N SELECTs + N
        # RPC/INSERT secuenciales ocupan un worker thread sync por minutos
        # (self-DoS de un worker). Knob clampeado.
        if len(ingredients) > _env_int("MEALFIT_RESTOCK_MAX_ITEMS", 500):
            return {"success": False, "message": "Lista de ingredientes demasiado grande."}

        # [P0-NEW-1 · 2026-05-10] Ownership check sobre `plan_id` user-provided.
        # Antes el SELECT en la rama `if plan_id` filtraba solo por `id` — un
        # atacante autenticado podía pasar el `plan_id` de una víctima en el
        # body y el endpoint:
        #   1. Leía plan_data ajeno (SELECT línea original sin user_id).
        #   2. Más abajo UPDATEea ese plan_data con is_restocked=True +
        #      restocked_items {keys: NOW} arbitrarios → corrupción de la
        #      lista de compras de la víctima (perecederos suprimidos por
        #      hasta 7 días).
        # Misma familia que P0-HIST-IDOR-1 (retry-chunk) y P0-HIST-IDOR-2
        # (chunk-status); auditoría inicial no cubrió `/restock`.
        #
        # Fix: ambas ramas (`if plan_id` y fallback latest) ahora filtran por
        # `user_id`. Si el caller pasa un plan_id no resoluble, 404 (mismo
        # contrato que retry-chunk:4106 / chunk-status:3474) — no leak la
        # existencia del plan ajeno ni fallback silencioso a latest (que
        # podría re-introducir wrong-plan persist como en P1-HIST-RECIPE-1).
        real_plan_id = None
        plan_data = None
        try:
            # [P1-NEON-DB-MIGRATION · 2026-06-12] PostgREST → SQL directo (Neon).
            # Ambas ramas preservan el ownership P0-NEW-1 / invariante I2.
            # `id::text` mantiene paridad de tipos con PostgREST (string, no
            # uuid.UUID) para logs y el persist atómico aguas abajo.
            from db import execute_sql_query
            if plan_id:
                plan_res = execute_sql_query(
                    "SELECT id::text AS id, plan_data FROM public.meal_plans "
                    "WHERE id = %s AND user_id = %s",
                    (plan_id, user_id),
                    fetch_one=True,
                )
                if not plan_res:
                    # plan_id no resoluble para este usuario. 404 sin
                    # filtrar si existe para otro user.
                    raise HTTPException(status_code=404, detail="Plan no encontrado")
            else:
                plan_res = execute_sql_query(
                    "SELECT id::text AS id, plan_data FROM public.meal_plans "
                    "WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
                    (user_id,),
                    fetch_one=True,
                )

            if plan_res:
                real_plan_id = plan_res.get("id")
                plan_data = plan_res.get("plan_data", {})
        except HTTPException:
            raise
        except Exception as check_err:
            logger.warning(f"⚠️ Error leyendo plan para restock: {check_err}")

        # [P1-2] Idempotencia item-level: filtrar items ya registrados dentro del
        # ciclo activo. Antes el endpoint rechazaba el request entero si
        # `is_restocked=true`, bloqueando restocks parciales (usuario compra solo
        # fresas el lunes y luego pollo el jueves dentro del mismo ciclo).
        from constants import strip_accents
        from datetime import datetime, timezone

        # [P1-A · 2026-05-08] `_env_int` registra en `_KNOBS_REGISTRY`. Los
        # clamps preservan el comportamiento previo y son locales al consumer.
        _max_cap = _env_int("MEALFIT_PERISHABLE_CYCLE_DAYS_MAX", 30)
        _max_cap = max(7, min(_max_cap, 90))
        _cycle_days = _env_int("MEALFIT_PERISHABLE_CYCLE_DAYS", 7)
        _cycle_days = max(1, min(_cycle_days, _max_cap))
        _now_utc = datetime.now(timezone.utc)
        _existing_restocked = (plan_data or {}).get("restocked_items") or {}
        if not isinstance(_existing_restocked, dict):
            _existing_restocked = {}

        # [P1-RESTOCK-LOSTUPDATE · 2026-05-30] Flag que el self-heal abajo
        # levanta cuando la nevera estaba vacía. El persist atómico (mutator
        # sobre plan_data FRESH) lo consulta para decidir si MERGE-a sobre
        # `restocked_items` existente o arranca desde cero — replicando la
        # semántica del `plan_data.pop(...)` del self-heal sobre el snapshot
        # t=0, pero aplicada al fresh re-leído bajo FOR UPDATE.
        _restock_self_heal_reset = False

        # [P3-RESTOCK-STALE-DEDUP · 2026-05-17] Self-heal del dedup `restocked_items`
        # cuando `user_inventory` quedó vacío.
        #
        # Modo de fallo cerrado: el usuario borraba TODOS los items de la nevera
        # (Pantry.jsx::confirmDeleteAll → `supabase.from('user_inventory').delete`)
        # y luego clicaba "Agregar a la Nevera" otra vez esperando el comportamiento
        # de la primera vez (re-popular toda la nevera). Pre-fix:
        #   1. confirmDeleteAll llama `_recalcShoppingListAfterPantryChange({
        #      clearRestockedFlag: true})` → frontend hace `delete
        #      result.plan_data.is_restocked` SOLO en el objeto local; el helper
        #      `/recalculate-shopping-list` solo limpia `is_restocked`+
        #      `restocked_items` cuando `householdSize`/`groceryDuration` cambian
        #      (línea 4982 `has_changed`). Como confirmDeleteAll NO cambia esos
        #      params, los flags persisten en DB stale.
        #   2. Segundo /restock: `_existing_restocked` se lee con 38 entries con
        #      timestamps recientes (<7d). `_in_cycle(prev_ts)=True` para todas
        #      → `skipped_dupes.append(name); continue` filtra 35 de 38 items.
        #      Solo los 3 items añadidos a `restocked_items` POST-restock-original
        #      (swaps + recipe expand) sobreviven el dedup → al usuario "le agrega
        #      solo 3 alimentos" en lugar de los 36-38 esperados.
        #   3. PDF render: `buildDeltaShoppingList` lee `planData?.is_restocked=true`
        #      → suprime items via `itemsRemoved++` (Dashboard.jsx:841-843) → muestra
        #      "Lista vacía, 36 ya están en nevera" pese a nevera real con 3 items.
        #
        # Defensa: si la nevera del usuario está vacía AL MOMENTO del /restock,
        # cualquier dedup previo es obsoleto por definición — los items que se
        # restockearon "antes" ya no están físicamente en el inventario. Reseteamos
        # `restocked_items`, `is_restocked` y `restocked_at_iso` antes de procesar.
        # El restock continúa normalmente y reescribe estos campos con los nuevos
        # items (línea 4300+).
        #
        # Defense-in-depth: la Pantry helper YA intenta limpiar el flag local pero
        # no persiste a DB; este self-heal cierra el gap independientemente de
        # cómo se haya vaciado la nevera (UI, agente, RPC, SQL directo, FK CASCADE).
        # Idempotente: si `restocked_items` ya estaba vacío, no-op silencioso.
        #
        # Tooltip-anchor: P3-RESTOCK-STALE-DEDUP-START | test_p3_restock_stale_dedup
        if user_id and _existing_restocked:
            try:
                # SQL raw via execute_sql_query (helper canónico del repo) en
                # vez de supabase-py `count="exact"`: más predecible (retorna
                # int directo desde la fila result), evita edge cases del
                # cliente (count=None bajo rate-limit/proxy) y consistente
                # con el resto del repo (db_plans.py:215+, db_inventory.py).
                from db_core import execute_sql_query
                _inv_count_row = execute_sql_query(
                    "SELECT COUNT(*) AS c FROM user_inventory WHERE user_id = %s",
                    (user_id,),
                    fetch_one=True,
                )
                _inv_count = int(_inv_count_row["c"]) if _inv_count_row and "c" in _inv_count_row else None
                logger.info(
                    f"🧹 [P3-RESTOCK-STALE-DEDUP] check: user={user_id[:8]}.. "
                    f"plan={real_plan_id} inv_count={_inv_count} "
                    f"restocked_items={len(_existing_restocked)}"
                )
                if isinstance(_inv_count, int) and _inv_count == 0:
                    logger.info(
                        f"🧹 [P3-RESTOCK-STALE-DEDUP] RESET: inventory vacío + "
                        f"dedup stale → limpiando is_restocked + restocked_items + "
                        f"restocked_at_iso de plan_data (in-memory, persist al UPDATE)"
                    )
                    _existing_restocked = {}
                    _restock_self_heal_reset = True
                    if isinstance(plan_data, dict):
                        plan_data.pop("is_restocked", None)
                        plan_data.pop("restocked_items", None)
                        plan_data.pop("restocked_at_iso", None)
            except Exception as _heal_err:
                logger.warning(
                    f"⚠️ [P3-RESTOCK-STALE-DEDUP] inventario count check falló: "
                    f"{_heal_err}. Procesando con dedup actual (posible undercount)."
                )
        # P3-RESTOCK-STALE-DEDUP-END

        def _name_of(it):
            if isinstance(it, dict):
                return str(it.get("name", "")).strip()
            return str(it).strip()

        def _in_cycle(iso_ts):
            try:
                ts = iso_ts.replace("Z", "+00:00") if iso_ts.endswith("Z") else iso_ts
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return (_now_utc - dt).total_seconds() / 86400.0 < _cycle_days
            except Exception:
                return False

        filtered_ingredients = []
        skipped_dupes = []
        for raw_item in ingredients:
            name = _name_of(raw_item)
            if not name:
                continue
            key = strip_accents(name.lower())
            prev_ts = _existing_restocked.get(key)
            if isinstance(prev_ts, str) and _in_cycle(prev_ts):
                skipped_dupes.append(name)
                continue
            filtered_ingredients.append(raw_item)

        if skipped_dupes:
            logger.info(f"🔁 [RESTOCK] {len(skipped_dupes)} item(s) ya registrado(s) en ciclo ({_cycle_days}d), saltando duplicados: {skipped_dupes[:5]}")

        if not filtered_ingredients:
            return {"success": True, "message": "Todos los items ya estaban registrados en este ciclo."}

        # Validación MURO Omitida: Ahora confiamos en el Delta Shopping del frontend.
        # El frontend solo envía los ingredientes que no están en la Nevera.
        # [P0-RESTOCK-DEDUP-NAME · 2026-05-20] restock_inventory ahora retorna
        # (success, persisted_names). Usamos persisted_names abajo para marcar
        # `restocked_items` SOLO con lo que efectivamente entró a DB — antes
        # marcábamos los inputs aspiracionales (filtered_ingredients) sin
        # validar que el INSERT persistió. Resultado pre-fix: ledger drift
        # (restocked_items contaba 27 keys cuando user_inventory tenía 24 rows).
        _restock_res = restock_inventory(user_id, filtered_ingredients)
        if isinstance(_restock_res, tuple):
            success, persisted_names = _restock_res
        else:
            # Defensive: callers legacy esperaban bool — backward compat.
            success, persisted_names = bool(_restock_res), []

        if success:
            log_api_usage(user_id, "restock_inventory")

            # Marcar el plan como "restocked" en BD para futuras peticiones.
            #
            # [P1-RESTOCK-LOSTUPDATE · 2026-05-30] Migrado de un UPDATE
            # full-overwrite del JSONB plan_data completo vía supabase-py
            # (que leía plan_data a t=0 sin lock, lo mutaba in-memory y reescribía
            # el JSONB ENTERO a t=2) al patrón canónico `update_plan_data_atomic`
            # (`SELECT … FOR UPDATE` + mutator sobre plan_data FRESH post-worker).
            #
            # Por qué: era la ÚLTIMA escritura full-overwrite de `plan_data` en
            # routers/plans.py y la única violación de I7 restante (todos los
            # hermanos ya migrados: /swap-meal/persist P1-SWAP-PERSIST-ATOMIC,
            # /recalculate-shopping-list P1-RECALC-LOSTUPDATE, /restore-local
            # P1-OPEN-1). Ventana lost-update real (no hipotética): entre el SELECT
            # t=0 y el UPDATE t=2, `_chunk_worker` puede persistir `days[8..14]` de
            # un plan multi-semana bajo advisory lock + el cron VISIÓN-C poda
            # `restocked_items` vencidos vía jsonb_set — el full-overwrite a t=2
            # los CLOBBEA silenciosamente (pérdida de comidas generadas). RLS filtra
            # IDOR pero NO lost-update (mismo user_id).
            #
            # El mutator aplica SOLO las 3 keys que /restock posee (is_restocked,
            # restocked_at_iso, restocked_items) MERGEANDO sobre el fresh —
            # preserva days/_chunk_lessons/aggregated_shopping_list* y prunes
            # concurrentes. `_names_to_mark` se computa FUERA del mutator (depende
            # de persisted_names, ya resuelto) porque el mutator corre dentro del
            # FOR UPDATE y DEBE ser puro CPU-only (contrato P2-MUTATOR-PURITY).
            # `user_id=` preserva el filtro AND user_id=%s en SELECT+UPDATE
            # (defensa-en-profundidad del ownership P0-NEW-1 + invariante I2).
            #
            # Tooltip-anchor: P1-RESTOCK-LOSTUPDATE-START | test_p1_restock_lostupdate
            if real_plan_id:
                try:
                    now_iso = _now_utc.isoformat()
                    # [P0-RESTOCK-DEDUP-NAME · 2026-05-20] Anotar SOLO los names
                    # que efectivamente persistieron a DB (persisted_names del
                    # retorno de restock_inventory); fallback a filtered_ingredients
                    # para callers legacy (preserva el comportamiento previo).
                    _names_to_mark = (
                        persisted_names
                        if persisted_names
                        else [_name_of(it) for it in filtered_ingredients]
                    )

                    def _restock_mutator(fresh: dict) -> dict:
                        fresh["is_restocked"] = True
                        # [RIESGO-1/3 FIX] Timestamp ISO-8601: (1) filtra
                        # perecederos del delta mid-cycle (<7d) en hybrid;
                        # (2) cron diario detecta restocks ≥7d y reactiva la lista.
                        fresh["restocked_at_iso"] = now_iso
                        # [P1-2] restocked_items: timestamp por item para
                        # supresión granular en _build_hybrid_shopping_list.
                        # Self-heal P3-RESTOCK-STALE-DEDUP: si la nevera estaba
                        # vacía, el dedup previo es obsoleto → arrancar desde cero
                        # (sobre el fresh). Si no, MERGE sobre el fresh para
                        # preservar el prune concurrente del cron.
                        if _restock_self_heal_reset:
                            ri = {}
                        else:
                            ri = fresh.get("restocked_items")
                            if not isinstance(ri, dict):
                                ri = {}
                        for nm in _names_to_mark:
                            if nm:
                                ri[strip_accents(str(nm).lower())] = now_iso
                        fresh["restocked_items"] = ri
                        return fresh

                    from db_plans import update_plan_data_atomic
                    _persisted = update_plan_data_atomic(
                        real_plan_id, _restock_mutator, user_id=user_id
                    )
                    if not _persisted:
                        logger.warning(
                            f"⚠️ [RESTOCK] plan {real_plan_id} no encontrado o no "
                            f"pertenece al user — skip persist (inventario físico "
                            f"ya restockeado; solo se omite el flag is_restocked)."
                        )
                    else:
                        logger.info(
                            f"✅ [RESTOCK] plan {real_plan_id}: input={len(filtered_ingredients)} "
                            f"persisted={len(persisted_names)} restocked_items="
                            f"{len(_persisted.get('restocked_items') or {})} entries (atomic)"
                        )
                except Exception as mark_err:
                    logger.warning(f"⚠️ No se pudo marcar plan como restocked (atomic): {mark_err}")
            # P1-RESTOCK-LOSTUPDATE-END

            # [P3-RESTOCK-DELETE-DEPLETED · 2026-05-30] Honrar el contrato que la
            # migración p3_user_depleted_items documenta ("el restock desde la
            # lista de compras DELETE-ea la fila aquí"): borrar de
            # `user_depleted_items` los ingredientes efectivamente repuestos.
            # Pre-fix /restock NUNCA tocaba la tabla → divergencia DB (badge
            # AGOTADO zombi para items que el usuario ya compró) + crecimiento
            # monótono de filas huérfanas sin GC. delete_depleted_item es
            # idempotente (no-op si nada matchea), case-insensitive por nombre
            # (ilike+trim) y filtra por user_id (defensa-en-profundidad).
            # Best-effort: cualquier fallo NO debe abortar el restock (la nevera
            # física ya se actualizó). El realtime channel propaga el DELETE a
            # otros tabs/devices.
            try:
                # [P3-BACKEND-AUDIT · 2026-06-01] Bulk DELETE en 1 round-trip
                # (antes: loop N+1 de delete_depleted_item por nombre → hasta N
                # DELETEs secuenciales en el request path). bulk_delete_depleted_items
                # preserva el match case-insensitive y el filtro user_id.
                from db_inventory import bulk_delete_depleted_items
                _names_to_clear = persisted_names or [_name_of(it) for it in filtered_ingredients]
                bulk_delete_depleted_items(user_id, _names_to_clear)
            except Exception as _depl_err:
                logger.warning(f"⚠️ [P3-RESTOCK-DELETE-DEPLETED] cleanup falló (best-effort): {_depl_err}")

            return {
                "success": True,
                "message": "¡Despensa actualizada exitosamente!",
                "persisted_count": len(persisted_names),
                "requested_count": len(filtered_ingredients),
            }
        else:
            return {"success": False, "message": "Hubo un problema actualizando algunos ingredientes."}

    except HTTPException:
        # [P0-NEW-1 · 2026-05-10] Propagar el 404 del ownership check
        # tal cual; sin esto el `except Exception` lo re-wrappea a 500 y
        # el test/cliente pierde la señal del IDOR.
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/restock: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

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

        # [P3-PROD-AUDIT-3 · 2026-05-30] Cap de tamaño — único miembro de la
        # familia inventory-batch sin él (/restock y /depleted-items ya capean a
        # 500). Sin cap, un payload gigante hace girar el loop de normalización +
        # un array bind enorme en el worker thread (self-DoS de un worker).
        if len(ingredients) > _env_int("MEALFIT_CONSUME_MAX_ITEMS", 500):
            return {"success": False, "message": "Lista de ingredientes demasiado grande."}

        success = consume_inventory_items_completely(user_id, ingredients)
        
        if success:
            log_api_usage(user_id, "consume_inventory")
            return {"success": True, "message": "Inventario actualizado exitosamente."}
        else:
            return {"success": False, "message": "Hubo un problema vaciando algunos ingredientes."}
            
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/inventory/consume: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


# [P3-WATER-TRACKER · 2026-05-16] Tracker diario de hidratacion. Reemplaza el
# card "Mi Nevera" del Dashboard que duplicaba la pagina Pantry. Cero costo
# LLM → usa `get_verified_user_id` (NO `verify_api_quota`), siguiendo el
# patron Historial-quota-exemption (CLAUDE.md). Rate limiter dedicado contra
# click-spam (60 req/min = 1/s sostenido; un usuario clickeando los 8 vasos
# en burst entra holgado, un bot/loop infinito se topa).
_WATER_TRACKER_LIMITER = RateLimiter(max_calls=60, period_seconds=60)
_WATER_DEFAULT_GOAL = 8  # P3-WATER-TRACKER: fallback cuando peso no esta disponible.
_WATER_MAX_GLASSES = 50  # mirror del CHECK de la tabla. Cap defensivo.


# [P3-NAN-INF-SANITIZE · 2026-05-16] Defensa contra `ValueError: Out of range
# float values are not JSON compliant` en `/recalculate-shopping-list`.
#
# Síntoma observado 2026-05-16 plan 4cc91584: el recalc completaba
# exitosamente (`✅ Listas recalculadas exitosamente ×1 personas`) pero al
# serializar la response, Python json.dumps rechazaba un NaN o Inf en
# `plan_data` (probablemente en una división por zero del aggregator de
# shopping calc — sample: ítems con `display_qty` formateado vs market_qty
# float). Resultado: 500 Internal Server Error + CORS error secundario.
#
# Este sanitizer reemplaza NaN/Inf con `None` (JSON-compliant + frontend
# lo trata como ausente, igual que un valor faltante). NO es el fix raíz
# (alguna fórmula divide por cero) pero impide que un NaN downstream
# crashe la API.
#
# Tooltip-anchor: P3-NAN-INF-SANITIZE-START | test_p3_nan_inf_sanitize
def _sanitize_floats_for_json(obj):
    """Reemplaza NaN/Inf con None recursivamente en dicts/lists/floats.
    Otros tipos pasan tal cual. Idempotente: aplicar dos veces == aplicar
    una vez (None se preserva como None).
    """
    if isinstance(obj, float):
        if _math.isnan(obj) or _math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_floats_for_json(v) for v in obj)
    return obj

# [P3-SUPABASE-TRANSIENT-RETRY · 2026-05-16] El endpoint /water-intake se
# dispara con frecuencia desde el listener visibilitychange del WaterTracker
# (cada vez que el usuario vuelve al tab). Cuando el tab estuvo en background,
# las conexiones HTTPS idle backend→Supabase pueden estar muertas: el primer
# uso descubre el problema y httpx levanta `RemoteProtocolError` / `ReadError`
# / `ConnectError` → pre-fix devolvíamos 500 al cliente, contaminando logs y
# disparando toast de error. Reintento cubre 99% de los blips (la conexión
# muerta se reemplaza en el reintento). Si todos fallan → 503 (transient,
# retry) en lugar de 500 (sugiere bug).
#
# [P2-WATER-RETRY-NO-JITTER · 2026-05-24] Pre-fix backoff constante 350ms
# sin jitter ni exponencial. En escenarios de mass-retry (Supabase regional
# blip → 100s de clients reintentan al mismo offset +350ms) → thundering
# herd. Otros retries del repo ya usan exponencial (`db_inventory.py:916`
# usa `0.05 * (1 << attempt)`) o knob-controlled (`shopping_calculator.py`).
# Ahora: `base * (2 ** attempt) + random.uniform(0, jitter_max)`. El jitter
# absoluto (no proporcional) preserva backoff razonable incluso con base
# muy chica. Knob `MEALFIT_WATER_RETRY_BACKOFF_BASE_S` permite ajustar
# sin redeploy si latencia Supabase cambia. Tooltip-anchor: P2-WATER-RETRY-NO-JITTER.
_WATER_RETRY_BACKOFF_BASE_S = _env_float(
    "MEALFIT_WATER_RETRY_BACKOFF_BASE_S",
    0.35,
    validator=lambda v: 0.05 <= v <= 5.0,
)
_WATER_RETRY_JITTER_MAX_S = _env_float(
    "MEALFIT_WATER_RETRY_JITTER_MAX_S",
    0.1,
    validator=lambda v: 0.0 <= v <= 1.0,
)
_WATER_RETRY_ATTEMPTS = 2


def _execute_with_retry(builder_factory, op_label: str):
    """Ejecuta una operación DB con 1 reintento. `builder_factory` es una
    lambda/callable que EJECUTA la operación completa cada vez y retorna
    el resultado.

    [P1-NEON-DB-MIGRATION · 2026-06-12] Migrado de builders supabase-py
    (PostgREST, donde el factory CONSTRUÍA el builder y aquí se llamaba
    `.execute()`) a callables sobre `execute_sql_query/_write` (pool psycopg
    → Neon). El patrón factory se conserva: cada intento re-ejecuta la
    operación contra una conexión fresca del pool, cubriendo la misma clase
    de blips transitorios (conexión idle muerta tras tab en background).

    [P2-WATER-RETRY-NO-JITTER · 2026-05-24] Backoff exponencial + jitter
    absoluto. attempt=0 → base + U[0, jitter_max]; attempt=1 → base*2 +
    U[0, jitter_max]; etc. Si subes `_WATER_RETRY_ATTEMPTS` no toques
    `base` — el exponencial escala solo.
    """
    import time as _time
    import random as _random
    last_exc = None
    for attempt in range(_WATER_RETRY_ATTEMPTS):
        try:
            return builder_factory()
        except Exception as e:
            last_exc = e
            if attempt + 1 < _WATER_RETRY_ATTEMPTS:
                sleep_s = (
                    _WATER_RETRY_BACKOFF_BASE_S * (2 ** attempt)
                    + _random.uniform(0.0, _WATER_RETRY_JITTER_MAX_S)
                )
                logger.warning(
                    f"⚠️ [P3-WATER-TRACKER] {op_label} transient en intento "
                    f"{attempt + 1}/{_WATER_RETRY_ATTEMPTS}: {type(e).__name__}: "
                    f"{(str(e) or '')[:120]}. Reintentando en {sleep_s:.3f}s."
                )
                _time.sleep(sleep_s)
    # last_exc garantizado no-None aquí: el loop (_WATER_RETRY_ATTEMPTS>=1) solo
    # alcanza este punto cuando el intento final cayó en `except` (asignó last_exc);
    # un intento exitoso retorna antes.
    assert last_exc is not None
    raise last_exc

# [P3-WATER-TRACKER · 2026-05-16] Personalizacion de la meta diaria.
# Formula: 35ml/kg de peso corporal + bonus por actividad fisica.
# - Base: 35 ml × kg (estandar nutricional, ref. Institute of Medicine).
# - Bonus actividad: 0/250/500/750 ml para sedentary/moderate/active/very_active.
# - Cada vaso = 240 ml (vaso estandar US-style, alineado con la UI de 8 vasos
#   default = 1920 ml ≈ 2 L del benchmark clasico de "8 glasses a day").
# - Clamp [6, 14] vasos: cubre 40kg (=6 vasos) a ~100kg+activo (=14 vasos)
#   sin romper la UI (la grilla wrappea a 2 filas cuando goal > 8).
_WATER_ML_PER_GLASS = 240
_WATER_GOAL_MIN = 6
_WATER_GOAL_MAX = 14
_WATER_ML_PER_KG = 35
_WATER_ACTIVITY_BONUS_ML = {
    "sedentary": 0,
    "low": 0,
    "light": 0,
    "moderate": 250,
    "mod": 250,
    "active": 500,
    "high": 500,
    "very_active": 750,
    "very_high": 750,
    "athlete": 750,
}


def _compute_water_goal(user_id: str) -> dict:
    """Calcula la meta diaria de vasos para `user_id`, derivada de
    `user_profiles.health_profile` (weight + weightUnit + activityLevel).

    Returns:
      dict con `goal` (int, [6,14]), `weight_kg` (float|None),
      `activity_level` (str|None), `computed_ml` (int|None) y `default`
      (bool — True si fallback a 8 vasos por peso ausente o invalido).

    Fail-secure: cualquier excepcion DB / parse → fallback a 8 vasos. El
    tracker debe seguir funcionando aunque el perfil este corrupto.
    """
    default = {
        "goal": _WATER_DEFAULT_GOAL,
        "weight_kg": None,
        "activity_level": None,
        "computed_ml": None,
        "default": True,
    }
    if not user_id:
        return default
    try:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] PostgREST → SQL directo (Neon).
        # health_profile es jsonb → psycopg lo devuelve como dict (paridad).
        from db import execute_sql_query
        res = execute_sql_query(
            "SELECT health_profile FROM public.user_profiles WHERE id = %s LIMIT 1",
            (user_id,),
            fetch_one=True,
        )
        if not res:
            return default
        hp = res.get("health_profile") or {}

        weight_raw = hp.get("weight")
        if weight_raw is None or weight_raw == "":
            return default
        try:
            weight_raw = float(weight_raw)
        except (ValueError, TypeError):
            return default
        if weight_raw <= 0:
            return default

        # Unit normalization — mismo patron que nutrition_calculator.py L242-265
        # (P0-FORM-4): defaultea a 'lb' por compatibilidad con perfiles legacy.
        weight_unit = str(hp.get("weightUnit") or "lb").lower().strip()
        if weight_unit not in ("lb", "kg"):
            weight_unit = "lb"
        weight_kg = weight_raw if weight_unit == "kg" else round(weight_raw / 2.20462, 1)

        # Sanity bounds: 25kg (preteen pequeño) a 250kg (extremo). Fuera de
        # este rango sospechamos data corrupta — fallback a default.
        if weight_kg < 25 or weight_kg > 250:
            return default

        activity_level = str(hp.get("activityLevel") or "moderate").lower().strip()
        bonus_ml = _WATER_ACTIVITY_BONUS_ML.get(activity_level, 250)  # default moderate

        computed_ml = int(round(weight_kg * _WATER_ML_PER_KG)) + bonus_ml
        glasses = round(computed_ml / _WATER_ML_PER_GLASS)
        glasses = max(_WATER_GOAL_MIN, min(_WATER_GOAL_MAX, glasses))

        return {
            "goal": int(glasses),
            "weight_kg": weight_kg,
            "activity_level": activity_level,
            "computed_ml": computed_ml,
            "default": False,
        }
    except Exception as e:
        logger.warning(f"[P3-WATER-TRACKER] _compute_water_goal error: {e}")
        return default


def _validate_water_date(date_str: str) -> str:
    """Acepta solo YYYY-MM-DD. Rechaza inyecciones de strings raras."""
    try:
        # `datetime.fromisoformat` acepta `YYYY-MM-DD` y formatos con hora;
        # forzamos a solo-date con un parse explicito.
        parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
        # Cap razonable: no aceptar fechas >7d en el futuro (clock skew) ni
        # >365d en el pasado (un cliente con bug podria spammear fechas viejas).
        today_utc = datetime.now(timezone.utc).date()
        if parsed > today_utc + timedelta(days=7):
            raise HTTPException(status_code=400, detail="Fecha fuera de rango (futuro).")
        if parsed < today_utc - timedelta(days=365):
            raise HTTPException(status_code=400, detail="Fecha fuera de rango (pasado).")
        return parsed.isoformat()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Formato de fecha invalido. Usa YYYY-MM-DD.")


@router.get("/water-intake")
def api_get_water_intake(
    date: str,
    verified_user_id: Optional[str] = Depends(_WATER_TRACKER_LIMITER),
):
    """[P3-WATER-TRACKER · 2026-05-16] Lee el conteo de vasos para `date`
    (fecha LOCAL del cliente, YYYY-MM-DD). Devuelve 0 si no hay row.
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="No autorizado.")
    log_date = _validate_water_date(date)

    # [P1-NEON-DB-MIGRATION · 2026-06-12] Gate equivalente al legacy `if not
    # supabase`. Fail-secure: si la DB no esta disponible, no inventamos un
    # default que el cliente confunda con "ya marcaste vasos hoy".
    # `connection_pool` se importa de db_core (NO de la fachada db): la
    # fachada bindea el valor al momento del import (None pre-init).
    from db_core import connection_pool
    if not connection_pool:
        raise HTTPException(status_code=503, detail="DB no disponible.")

    try:
        from db import execute_sql_query
        # `updated_at` se devuelve como datetime (FastAPI lo serializa a
        # ISO-8601, mismo wire format que PostgREST). El frontend no lo parsea.
        res = _execute_with_retry(
            lambda: execute_sql_query(
                "SELECT glasses, updated_at FROM public.water_intake_log "
                "WHERE user_id = %s AND log_date = %s LIMIT 1",
                (verified_user_id, log_date),
                fetch_one=True,
            ),
            op_label="GET water-intake",
        )
        glasses = 0
        updated_at = None
        if res:
            glasses = int(res.get("glasses") or 0)
            updated_at = res.get("updated_at")
        goal_meta = _compute_water_goal(verified_user_id)
        # [P3-WATER-TRACKER · 2026-05-16] `enabled` se incluye aqui para que
        # el frontend NO requiera un fetch separado a /api/user/preferences/
        # water-tracker en el mount (reduce roundtrips y elimina race entre
        # los dos fetches).
        from db_profiles import get_water_tracker_enabled
        return {
            "success": True,
            "date": log_date,
            "glasses": glasses,
            "goal": goal_meta["goal"],
            "goal_basis": {
                "weight_kg": goal_meta["weight_kg"],
                "activity_level": goal_meta["activity_level"],
                "computed_ml": goal_meta["computed_ml"],
                "default": goal_meta["default"],
            },
            "enabled": get_water_tracker_enabled(verified_user_id),
            "updated_at": updated_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        # [P3-SUPABASE-TRANSIENT-RETRY · 2026-05-16] Si llegamos aqui es porque
        # AMBOS intentos fallaron. Devolvemos 503 (transient) en lugar de 500
        # (que sugiere bug del endpoint). El cliente puede decidir reintentar.
        logger.error(
            f"❌ [P3-WATER-TRACKER] GET water-intake fallo tras {_WATER_RETRY_ATTEMPTS} "
            f"intentos: {type(e).__name__}: {(str(e) or '')[:200]}",
            exc_info=(type(e), e, e.__traceback__),
        )
        raise HTTPException(status_code=503, detail="DB temporalmente no disponible. Reintenta.")


@router.post("/water-intake")
def api_set_water_intake(
    data: dict = Body(...),
    verified_user_id: Optional[str] = Depends(_WATER_TRACKER_LIMITER),
):
    """[P3-WATER-TRACKER · 2026-05-16] Upsert del conteo para una fecha local.
    Body: `{date: "YYYY-MM-DD", glasses: int}`. Idempotente — PK
    `(user_id, log_date)` permite enviar el mismo conteo varias veces sin
    duplicar rows (no contamos eventos, contamos estado actual del dia).
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="No autorizado.")

    date_str = data.get("date")
    if not isinstance(date_str, str) or not date_str:
        raise HTTPException(status_code=400, detail="Falta campo `date`.")
    log_date = _validate_water_date(date_str)

    raw_glasses = data.get("glasses")
    if not isinstance(raw_glasses, int) or isinstance(raw_glasses, bool):
        raise HTTPException(status_code=400, detail="Campo `glasses` debe ser entero.")
    if raw_glasses < 0 or raw_glasses > _WATER_MAX_GLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"`glasses` fuera de rango [0, {_WATER_MAX_GLASSES}].",
        )

    # [P1-NEON-DB-MIGRATION · 2026-06-12] Gate equivalente al legacy `if not
    # supabase` (ver nota del GET sobre por qué db_core y no la fachada).
    from db_core import connection_pool
    if not connection_pool:
        raise HTTPException(status_code=503, detail="DB no disponible.")

    try:
        # Upsert via ON CONFLICT en PK compuesta (user_id, log_date) —
        # equivalente al upsert PostgREST con on_conflict="user_id,log_date".
        # `updated_at = NOW()` se toca explicitamente para que el cliente
        # pueda distinguir "ya actualice hoy" del default seed.
        from db import execute_sql_write
        _execute_with_retry(
            lambda: execute_sql_write(
                "INSERT INTO public.water_intake_log "
                "(user_id, log_date, glasses, updated_at) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (user_id, log_date) DO UPDATE SET "
                "glasses = EXCLUDED.glasses, updated_at = EXCLUDED.updated_at",
                (
                    verified_user_id,
                    log_date,
                    int(raw_glasses),
                    datetime.now(timezone.utc),
                ),
            ),
            op_label="POST water-intake",
        )
        # `goal` se recomputa para que el cliente pueda detectar si el usuario
        # actualizo su peso entre GET y POST (ej: edito el perfil en otra tab).
        # El roundtrip a DB para leer `health_profile` no es critico — el POST
        # ya hizo upsert, y el rate limiter (60/60s) absorbe el costo.
        goal_meta = _compute_water_goal(verified_user_id)
        return {
            "success": True,
            "date": log_date,
            "glasses": int(raw_glasses),
            "goal": goal_meta["goal"],
            "goal_basis": {
                "weight_kg": goal_meta["weight_kg"],
                "activity_level": goal_meta["activity_level"],
                "computed_ml": goal_meta["computed_ml"],
                "default": goal_meta["default"],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        # [P3-SUPABASE-TRANSIENT-RETRY · 2026-05-16] Tras 2 intentos del upsert
        # → 503. POST es idempotente (PK + on_conflict), el cliente puede
        # reintentar sin riesgo de doble escritura.
        logger.error(
            f"❌ [P3-WATER-TRACKER] POST water-intake fallo tras {_WATER_RETRY_ATTEMPTS} "
            f"intentos: {type(e).__name__}: {(str(e) or '')[:200]}",
            exc_info=(type(e), e, e.__traceback__),
        )
        raise HTTPException(status_code=503, detail="DB temporalmente no disponible. Reintenta.")


@router.post("/recalculate-shopping-list")
def api_recalculate_shopping_list(data: dict = Body(...), verified_user_id: Optional[str] = Depends(_RECALC_LIMITER)):
    """
    Recalcula la lista de compras CANÓNICA escalando las recetas por el
    householdSize + grocery_duration.

    [P3-CANONICAL-AGG-WEEKLY · 2026-05-18] La lista persistida en
    `aggregated_shopping_list_weekly` (+ biweekly/monthly) es la lista
    completa (canonical, no delta). La deducción contra inventario actual
    se hace at-render-time en el frontend vía
    `Dashboard.buildDeltaShoppingList(canonical, freshInventory)`. Este
    contrato cierra el bug "agotar/reponer rompe PDF": antes, recalcs
    sucesivos mutaban agg_weekly con deltas intermedios que dejaban al PDF
    leyendo del localStorage stale.

    [P2-NEW-B · 2026-05-11] Acepta `plan_id` opcional en el body. Si está,
    el endpoint actúa sobre ESE plan específico (con ownership AND user_id
    en el SELECT). Si no, fallback al plan más reciente (back-compat).

    Bug pre-fix: bajo race condition con `_chunk_worker` creando un plan
    nuevo, el endpoint operaba sobre el plan B (latest) cuando el usuario
    acababa de hacer swap en plan A. Espejo del mismo bug que cerró
    P1-HIST-RECIPE-1 en `/recipe/expand`.

    Body adicional opcional:
      - ``plan_id`` (str): si presente, SELECT explícito por id+user_id.
        404 si no resoluble (no leak de existencia cross-user).

    Tooltip-anchor: P2-NEW-B-START | test_p2_new_b_recalculate_accepts_plan_id
    """
    try:
        user_id = data.get("user_id")
        # [P3-PDF-POLISH-4-B-RECALC · 2026-05-14] Clamp upper bound antes de
        # llegar a `compute_household_multiplier`. SSOT del clamp vive en el
        # helper (constants.py), pero aplicarlo también aquí garantiza que
        # `calc_household_size` persistido en plan_data nunca pase el cap
        # — un POST con `householdSize=999999` se persiste como 20, no como
        # 999999. Defense-in-depth idéntica al pattern de child_mult clamp.
        try:
            _max_household = _env_int("MEALFIT_MAX_HOUSEHOLD_SIZE", 20)
        except Exception:
            _max_household = 20
        _max_household = max(1, min(_max_household, 100))
        household_size = max(1, min(int(data.get("householdSize", 1) or 1), _max_household))
        # [P2-RECALC-GROCERY-DURATION-ENUM · 2026-05-14] Clamp del enum
        # `groceryDuration` a los 3 valores válidos. Pre-fix aceptaba
        # cualquier string del body (`data.get("groceryDuration", "weekly")`)
        # sin validar. Un POST con `groceryDuration="forever"` o un typo
        # (`"weely"`, `"month"`) caía silenciosamente al else-weekly del
        # branch en línea ~4099, pero el valor inválido se persistía a
        # `plan_data.calc_grocery_duration` (línea ~4179) → un cliente
        # downstream que leyera la key persistida observaría un valor
        # nunca generado por el flujo legítimo del frontend, abriendo
        # rama de UX incoherente (banner "ciclo desconocido", export PDF
        # con label vacío, etc).
        #
        # Defense-in-depth análogo al clamp `_max_household` (P3-PDF-POLISH-4-B-RECALC)
        # arriba. Si el frontend envía un valor inválido (bug del cliente,
        # o request adversarial), normalizamos al default y loggeamos para
        # captar el caller patológico — no aborta el flujo. El test
        # parser-based ancla el enum + clamp + fallback. Tooltip-anchor:
        # P2-RECALC-GROCERY-DURATION-ENUM-START | test_p2_recalc_grocery_duration_enum
        _ALLOWED_GROCERY_DURATIONS = ("weekly", "biweekly", "monthly")
        grocery_duration = data.get("groceryDuration", "weekly")
        if grocery_duration not in _ALLOWED_GROCERY_DURATIONS:
            logger.warning(
                f"[P2-RECALC-GROCERY-DURATION-ENUM] user_id={data.get('user_id')!r} "
                f"envió groceryDuration={grocery_duration!r} (no en "
                f"{_ALLOWED_GROCERY_DURATIONS}). Normalizado a 'weekly'."
            )
            grocery_duration = "weekly"
        is_new_plan_flag = data.get("is_new_plan", False)
        # [P1-3] householdComposition opcional desde el body. Si está, sobrescribe
        # el escalar legacy. Si no, fallback al int (comportamiento previo).
        household_composition = data.get("householdComposition")
        # [P2-NEW-B · 2026-05-11] plan_id explícito opcional. Permite al
        # caller fijar el target en lugar de depender de "el plan más
        # reciente". Espejo del fix P1-HIST-RECIPE-1 sobre /recipe/expand.
        req_plan_id = data.get("plan_id") if isinstance(data.get("plan_id"), str) else None

        if not user_id or user_id == "guest":
            return {"success": False, "message": "Debes iniciar sesión."}

        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado.")

        # [P2-NEW-B · 2026-05-11] Resolución del plan target:
        #   - Si el caller pasó `plan_id`: SELECT explícito con ownership
        #     (`WHERE id = %s AND user_id = %s`). 404 si no resoluble — no
        #     leak de existencia cross-user (mismo patrón que retry-chunk
        #     P0-HIST-IDOR-1 y restock P0-NEW-1).
        #   - Si no: fallback a `get_latest_meal_plan_with_id` (back-compat
        #     con callers viejos que no pasan plan_id).
        plan_record = None
        if req_plan_id:
            from db_core import execute_sql_query
            plan_row = execute_sql_query(
                "SELECT id, plan_data FROM meal_plans WHERE id = %s AND user_id = %s",
                (req_plan_id, user_id),
                fetch_one=True,
            )
            if not plan_row:
                # 404 (no 403) para no filtrar existencia del plan ajeno.
                raise HTTPException(status_code=404, detail="Plan no encontrado")
            plan_record = {
                "id": plan_row["id"],
                "plan_data": plan_row.get("plan_data") or {},
            }
        else:
            plan_record = get_latest_meal_plan_with_id(user_id)
            if not plan_record:
                return {"success": False, "message": "No hay plan activo."}

        plan_id = plan_record["id"]
        plan_data = plan_record.get("plan_data", {})

        if not plan_data:
            return {"success": False, "message": "Datos de plan inválidos."}

        from shopping_calculator import get_shopping_list_delta, fetch_inventory_and_consumed_for_plan
        # [P1-RECALC-LOSTUPDATE · 2026-05-14] Migración del helper:
        # `update_meal_plan_data` → `update_plan_data_atomic`. Ver justificación
        # detallada en el bloque P1-RECALC-LOSTUPDATE más abajo, junto al
        # callsite. El helper toma FOR UPDATE row lock + re-SELECT fresh +
        # callback + UPDATE en la misma transacción (P0-2), cerrando la
        # ventana lost-update entre el SELECT inicial del handler y la
        # persistencia full-overwrite.
        from db_plans import update_plan_data_atomic
        from constants import compute_household_multiplier

        # [P1-3] Multiplier efectivo: si hay householdComposition lo usa,
        # si no cae a householdSize (legacy). Mismo helper en planner y cron.
        _multiplier_source = {
            "householdComposition": household_composition,
            "householdSize": household_size,
        }
        household_multiplier = compute_household_multiplier(_multiplier_source)

        # Generar las 3 variantes escaladas dinámicamente según el householdSize
        # usando el delta matemático para evitar duplicados si hay inventario (Gap 3).
        # [P1-5] Fetch inventario + consumidos UNA vez para que las 3 listas
        # weekly/biweekly/monthly se calculen contra el MISMO snapshot.
        # Sin esto, mutaciones concurrentes a `user_inventory` (Realtime
        # channel, restock, cron) entre las 3 llamadas producían deltas
        # inconsistentes según el `groceryDuration` que el usuario eligiera.
        _inv_snap, _cons_snap = fetch_inventory_and_consumed_for_plan(
            user_id, plan_data, is_new_plan_flag
        )
        # [P3-RESTOCK-STALE-RECALC-HEAL · 2026-05-18] Snapshot del tamaño de
        # user_inventory para el self-heal de is_restocked dentro de
        # _apply_recalc. Si la nevera está vacía AHORA, cualquier
        # `is_restocked=true` + `restocked_items={...}` heredado del plan
        # es stale por definición — no puede haber "ya está en nevera" si
        # la nevera no existe. Mismo invariante que /restock línea 4311.
        _inv_count_at_recalc = (
            len(_inv_snap) if isinstance(_inv_snap, list) else None
        )
        # [P3-CANONICAL-AGG-WEEKLY · 2026-05-18] REFACTOR del contrato semántico
        # de `aggregated_shopping_list_weekly` (+ biweekly/monthly).
        #
        # PRE-FIX (bug causante de "agotar+reponer rompe PDF"):
        #   El recalc computaba scaled_* con `is_new_plan=is_new_plan_flag`
        #   (default False) → resultado = DELTA contra inventario actual.
        #   Cada vez que el usuario tocaba la nevera (agotar, reponer,
        #   Borrar Todos), el helper Pantry::_recalcShoppingListAfterPantryChange
        #   disparaba este endpoint. Sucesión de recalcs mutaba agg_weekly:
        #     - Pantry con 35 items → agg_weekly = [] (delta vacío).
        #     - Pantry con 0 items → agg_weekly = [35 items].
        #     - Pantry con 34 items → agg_weekly = [1 item].
        #   El PDF leía agg_weekly del localStorage (posiblemente stale)
        #   y mostraba la lista corta del recalc intermedio.
        #
        # POST-FIX (invariante semántico):
        #   `aggregated_shopping_list_weekly` SIEMPRE representa la lista
        #   CANÓNICA (full needs del plan escalado a household_size +
        #   duration), sin deducir inventario. La deducción contra
        #   inventario se hace at-render-time en el frontend vía
        #   `buildDeltaShoppingList(canonical, freshInventory)` (Dashboard.jsx).
        #   Esto cierra la clase entera de bugs:
        #     - agotar/reponer no muta la lista en DB.
        #     - PDF muestra el delta correcto contra inventario fresco.
        #     - Restock envía el delta correcto al backend.
        #
        # is_new_plan=True le dice a get_shopping_list_delta que NO deduzca
        # inventario — devuelve la lista canónica. inventory_override sigue
        # pasándose para no romper la firma (queda ignorado downstream).
        scaled_7 = get_shopping_list_delta(
            user_id, plan_data, is_new_plan=True, structured=True,
            multiplier=household_multiplier,
            inventory_override=_inv_snap, consumed_override=_cons_snap,
        )
        scaled_15 = get_shopping_list_delta(
            user_id, plan_data, is_new_plan=True, structured=True,
            multiplier=household_multiplier * 2.0,
            inventory_override=_inv_snap, consumed_override=_cons_snap,
        )
        scaled_30 = get_shopping_list_delta(
            user_id, plan_data, is_new_plan=True, structured=True,
            multiplier=household_multiplier * 4.0,
            inventory_override=_inv_snap, consumed_override=_cons_snap,
        )
        
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
        
        # [VISIÓN-C] Híbrido: staples=periodo, perishables=semanal.
        # [RIESGO-1] Pasamos restocked_at_iso para que mid-cycle no pida
        # perecederos fraccionarios (cycle lock 7 días).
        try:
            from shopping_calculator import _build_hybrid_shopping_list as _build_hybrid
            _restocked_at = plan_data.get("restocked_at_iso") if plan_data.get("is_restocked") else None
            # [P1-2] Item-level: precedencia sobre el blanket legacy.
            _restocked_items = plan_data.get("restocked_items") if isinstance(plan_data.get("restocked_items"), dict) else None
            scaled_15_hybrid = _build_hybrid(scaled_7, scaled_15, restocked_at_iso=_restocked_at, restocked_items=_restocked_items) if scaled_15 else scaled_15  # pyright: ignore[reportArgumentType]  (structured=True ⇒ scaled_* son list)
            scaled_30_hybrid = _build_hybrid(scaled_7, scaled_30, restocked_at_iso=_restocked_at, restocked_items=_restocked_items) if scaled_30 else scaled_30  # pyright: ignore[reportArgumentType]  (structured=True ⇒ scaled_* son list)
        except Exception as _hyb_e:
            logger.warning(f"[RECALC] _build_hybrid fallo: {_hyb_e}. Usando lista extrapolada.")
            scaled_15_hybrid = scaled_15
            scaled_30_hybrid = scaled_30

        # Seleccionar lista activa para el frontend legacy
        if grocery_duration == "biweekly":
            active_list = scaled_15_hybrid
        elif grocery_duration == "monthly":
            active_list = scaled_30_hybrid
        else:
            active_list = scaled_7

        # [P1-RECALC-LOSTUPDATE · 2026-05-14] Mutación + persistencia atómica
        # bajo FOR UPDATE row lock (vía `update_plan_data_atomic`).
        #
        # Pre-fix (audit 2026-05-14): el handler hacía SELECT inicial (línea
        # ~4011 fallback latest, o ~3999 branch req_plan_id) FUERA de cualquier
        # lock, mutaba `plan_data` in-memory con aggregated_shopping_list*,
        # calc_*, _shopping_coherence_block*, etc., y persistía con
        # `update_meal_plan_data` full-overwrite. El helper P1-NEXT-1 adquiría
        # advisory lock (purpose='general') para serializar contra `_chunk_worker
        # T1/T2` y `api_shift_plan`, pero esa serialización NO cerraba el
        # window read-modify-write contra endpoints hermanos que mutan
        # `plan_data` con `jsonb_set` quirúrgico (`/swap-meal/persist`,
        # `/grocery-start-date`, `/{plan_id}/name`, `/recipe/expand`): el
        # SELECT inicial leía a t=0 sin lock; un hermano podía escribir
        # quirúrgico a t=1 con su propio lock; recalc tomaba el lock y
        # UPDATEaba full-overwrite a t=2 con la copia stale → la mutación
        # quirúrgica del hermano se perdía silenciosamente.
        #
        # Fix: `update_plan_data_atomic` toma `SELECT … FOR UPDATE` row lock
        # + re-SELECTea plan_data FRESH dentro del lock + invoca el callback +
        # UPDATEa, todo en la misma transacción. El FOR UPDATE row lock
        # conflicta con el UPDATE implícito de cualquier hermano: las
        # mutaciones quirúrgicas concurrentes completan ANTES del lock o
        # esperan DETRÁS del UPDATE. Así el callback opera sobre la copia
        # post-merge y solo toca las keys que recalc semánticamente posee
        # (aggregated_shopping_list*, calc_*, restock_*, _shopping_coherence_*,
        # _debug_recalc) — todo lo demás (days, name, plan_expires_at,
        # grocery_start_date, cycle_start_date, expanded_recipe) se preserva
        # tal cual del fresh.
        #
        # Trade-off conocido: scaled_7/15/30 + active_list se computan FUERA
        # del lock con la copia inicial de `days`. Si un swap-meal modificó
        # `days[i].meals[j]` entre el SELECT inicial y el lock, las listas
        # reflejan la versión pre-swap. Es divergencia acotada (1 ingrediente
        # añadido/removido) — el swap del usuario SE PRESERVA (no se
        # sobreescribe); el usuario puede recalcular de nuevo para sincronizar
        # listas con días si nota la discrepancia. La alternativa (recomputar
        # listas DENTRO del lock) extendería el tiempo bajo lock a
        # ~100-500ms por recalc, contendiendo con chunk_worker / shift-plan /
        # otros recalcs sobre el mismo plan.
        #
        # Tooltip-anchor: P1-RECALC-LOSTUPDATE-START | test_p1_recalc_lostupdate
        _captured_divergences: list = []

        def _apply_recalc(plan_data_fresh: dict) -> dict:
            """Callback ejecutado DENTRO del FOR UPDATE row lock con
            `plan_data_fresh` re-SELECTado bajo lock. Solo muta las keys
            que recalc posee semánticamente; cualquier mutación quirúrgica
            que un endpoint hermano haya hecho antes del lock está
            preservada en `plan_data_fresh`.
            """
            # Aggregated lists (overwrite — recalc es source-of-truth de
            # estas keys cuando se invoca explícitamente).
            plan_data_fresh["aggregated_shopping_list"] = active_list
            plan_data_fresh["aggregated_shopping_list_weekly"] = scaled_7
            plan_data_fresh["aggregated_shopping_list_biweekly"] = scaled_15_hybrid
            plan_data_fresh["aggregated_shopping_list_monthly"] = scaled_30_hybrid

            # Comparar contra calc_* del FRESH (no del initial). Un recalc
            # concurrente del mismo user pudo haber dejado is_restocked
            # consistente con los nuevos params — en ese caso has_changed
            # es False y el pop no aplica.
            prev_hh = plan_data_fresh.get("calc_household_size")
            prev_dur = plan_data_fresh.get("calc_grocery_duration")
            prev_mult = plan_data_fresh.get("calc_household_multiplier")
            has_changed = (
                (prev_hh != household_size)
                or (prev_dur != grocery_duration)
                or (prev_mult is not None and abs(float(prev_mult) - household_multiplier) > 0.01)
            )

            plan_data_fresh["calc_household_size"] = household_size
            plan_data_fresh["calc_household_multiplier"] = household_multiplier
            if isinstance(household_composition, dict):
                plan_data_fresh["calc_household_composition"] = household_composition
            plan_data_fresh["calc_grocery_duration"] = grocery_duration

            # [P3-PLAN-MODIFIED-AT-RECALC · 2026-05-18] Bumpear `_plan_modified_at`
            # CADA recalc. La drift detection en Dashboard.handleDownloadShoppingList
            # compara este campo entre local y DB para decidir si re-sincronizar.
            # Sin este bump, recalcs sucesivos (agotar/reponer/Borrar Todos)
            # actualizan agg_weekly en DB pero el frontend nunca lo nota — el PDF
            # sigue leyendo el localStorage stale con la lista de un recalc
            # intermedio (típicamente la lista corta post-Compré-todo).
            from datetime import datetime as _dt, timezone as _tz
            plan_data_fresh["_plan_modified_at"] = _dt.now(_tz.utc).isoformat()

            # [P3-RESTOCK-STALE-RECALC-HEAL · 2026-05-18] Razón adicional para
            # limpiar is_restocked: la nevera está vacía. Cuando el usuario
            # hace Borrar Todos en Pantry, el helper invoca este endpoint
            # vía `_recalcShoppingListAfterPantryChange`; el frontend ya
            # nullifica `is_restocked` en su copia local, pero la DB queda
            # stale a menos que algo más la limpie. Sin este self-heal, el
            # siguiente PDF/restock veía `is_restocked=true` +
            # `restocked_items={N entries}` + `user_inventory=[]` → dedup
            # bogus de ~27 items contra fantasmas. Mismo invariante que
            # /restock línea 4311: si inv vacío, dedup obsoleto.
            _empty_pantry_heal = (
                _inv_count_at_recalc == 0
                and plan_data_fresh.get("is_restocked")
            )
            if (has_changed or _empty_pantry_heal) and plan_data_fresh.get("is_restocked"):
                plan_data_fresh.pop("is_restocked", None)
                plan_data_fresh.pop("restocked_at_iso", None)
                plan_data_fresh.pop("restocked_items", None)
                _reason = (
                    "cantidades cambiaron"
                    f" de {prev_hh}p/{prev_dur} (mult={prev_mult}) a "
                    f"{household_size}p/{grocery_duration} "
                    f"(mult={household_multiplier:.2f})"
                    if has_changed
                    else "user_inventory vacío (flags previos stale)"
                )
                logger.info(f"🔄 [RECALC] is_restocked limpiado — {_reason}")

            # [P1-NEXT-2 · 2026-05-11] Coherence guard sobre la lista recién
            # escalada. Antes, /recalculate-shopping-list persistía
            # aggregated_shopping_list* sin invocar run_shopping_coherence_guard —
            # un recalc cliente (Pantry add/delete + Dashboard) podía dejar la
            # lista divergente vs recetas sin retry ni telemetría.
            #
            # Modo: `warn` porque el caller es síncrono y bloquear con 400
            # rompe UX cuando la divergencia viene de un edge case del
            # multiplier escalado (P3-A ya cubre escala lineal). Si
            # divergencias críticas aparecen sistémicamente, el cron diario
            # las alertará; cliente sigue viendo lista usable.
            # `action_taken="warn_only_recalc"` distingue origen post-mortem.
            # [P2-COHERENCE-1 · 2026-05-11] Capturamos `divergences` vía
            # closure list (`_captured_divergences`) para retornar
            # `_coherence_warnings` en la response.
            try:
                from shopping_calculator import run_shopping_coherence_guard_and_append_history as _coh_recalc
                divs, _ = _coh_recalc(
                    plan_data_fresh,
                    multiplier=household_multiplier,
                    mode_override="warn",
                    attempt=1,
                    action_taken="warn_only_recalc",
                    plan_id_hint=plan_id,
                )
                _captured_divergences.extend(divs or [])
            except Exception as _coh_recalc_e:
                logger.warning(f"[RECALC] coherence guard helper falló (no aborta): {_coh_recalc_e}")

            # [P3-SHOPPING-2 · 2026-05-14] Fingerprint de debug solo en non-prod.
            # Antes esto se persistía SIEMPRE a `plan_data._debug_recalc` —
            # útil para diagnóstico en dev pero ensucia jsonb en producción
            # sin valor operacional. Knob `MEALFIT_PERSIST_DEBUG_RECALC`
            # permite re-habilitar en prod si SRE necesita instrumentar un
            # incidente puntual (default False = stripped en prod, escape
            # hatch sin redeploy).
            _persist_debug = (
                os.environ.get("ENVIRONMENT", "").lower() != "production"
                or os.environ.get("MEALFIT_PERSIST_DEBUG_RECALC", "").lower() in ("1", "true", "yes", "on")
            )
            if _persist_debug:
                import time
                plan_data_fresh["_debug_recalc"] = {
                    "household_size": household_size,
                    "timestamp": time.time(),
                    "weekly_items_count": len(scaled_7),
                    "sample_item": scaled_7[0].get("display_string", "?") if scaled_7 else "empty"  # pyright: ignore[reportAttributeAccessIssue]  (structured=True ⇒ scaled_7 es list[dict])
                }

            return plan_data_fresh

        # [P1-RECALC-LOSTUPDATE · 2026-05-14] `update_plan_data_atomic`
        # adquiere FOR UPDATE row lock, re-SELECTea plan_data con
        # `WHERE id = %s AND user_id = %s`, invoca el callback y persiste
        # el resultado — todo en la misma transacción (P0-2). Si la fila
        # no existe o no pertenece al user_id → retorna `{}` → 404 explícito.
        # [P1-NEW-3 · 2026-05-10] user_id ya validado al inicio del handler
        # (verified_user_id == user_id). plan_id viene del SELECT explícito
        # o de get_latest_meal_plan_with_id (ambos filtran por user_id).
        # Defense-in-depth doble candado al pasar user_id al helper.
        merged_plan_data = update_plan_data_atomic(
            plan_id, _apply_recalc, user_id=user_id
        )
        if not merged_plan_data:
            # Plan desapareció entre el SELECT inicial y el lock (cancelado
            # por save_new_meal_plan_atomic, o filtro user_id no matched).
            # 404 explícito en lugar de retornar success con plan_data stale.
            raise HTTPException(status_code=404, detail="Plan no encontrado")

        logger.info(f"✅ [RECALC] Listas recalculadas exitosamente ×{household_size} personas")

        # [P2-COHERENCE-1 · 2026-05-11] `_coherence_warnings` para que el
        # frontend muestre toast no-bloqueante si el guard detectó drift
        # recetas↔lista. Lista vacía cuando todo OK. Cap de 5 items vía
        # `summarize_divergences_for_ui` evita payloads gigantes.
        _coherence_warnings: list = []
        if _captured_divergences:
            try:
                from shopping_calculator import summarize_divergences_for_ui
                _coherence_warnings = summarize_divergences_for_ui(_captured_divergences, max_items=5)
            except Exception as _sum_e:
                logger.warning(f"[RECALC/P2-COH-1] summarize_divergences_for_ui falló: {_sum_e}")

        # Devolver el plan_data merged directamente para evitar race conditions
        # (el frontend no necesita re-fetch de Supabase). Es exactamente lo
        # que persistimos bajo el lock — sin sorpresas downstream.
        # [P3-NAN-INF-SANITIZE · 2026-05-16] Sanitize antes de retornar:
        # algún cálculo downstream del aggregator produce NaN/Inf esporádicos
        # (ej: división por zero cuando un ingrediente tiene `package_size=0`
        # en master_ingredients). El stdlib json.dumps rechaza estos →
        # 500 Internal Server Error visible al cliente. Sanitize reemplaza
        # con None (JSON-compliant, frontend graceful). El bug raíz (de
        # dónde viene el NaN) queda como follow-up para identificar +
        # arreglar en la fórmula upstream.
        return {
            "success": True,
            "plan_data": _sanitize_floats_for_json(merged_plan_data),
            "_coherence_warnings": _sanitize_floats_for_json(_coherence_warnings),
        }

    except HTTPException:
        # [P2-NEW-B · 2026-05-11] Propagar el 404 del ownership check
        # del plan_id explícito sin envolverlo en 500. Mismo patrón que
        # /restock (P0-NEW-1) y /retry-chunk (P0-HIST-IDOR-1).
        # [P2-OPEN-2 · 2026-05-11] Cleanup: pre-fix había DOS bloques
        # `except HTTPException: raise` consecutivos — el segundo era
        # inalcanzable. Mantenemos un solo bloque con el comentario P2-NEW-B
        # que documenta la intención original.
        raise
    except Exception as e:
        # [P3-PDF-POLISH-4-D · 2026-05-14] `logger.exception` captura el
        # traceback estructurado en el mismo emit. Pre-fix duplicaba el
        # error (logger.error + dump del traceback a stderr puro) y este
        # último escapaba al pattern P2-LOGGER-MIGRATION (no era `print()`
        # bare, pero funcionalmente equivalente al stdout/stderr write
        # fuera del logger handler). Tooltip-anchor: P3-PDF-POLISH-4-D.
        logger.exception("❌ [RECALC] Error en /api/recalculate-shopping-list")
        # [P3-RECALC-503-CLASSIFICATION · 2026-05-16] Si la excepción es de
        # pool/red transitoria (free tier `couldn't get a connection`, supabase
        # `RemoteProtocolError`, etc.) escalamos a 503 "transient, retry" en
        # lugar de 500 "bug del endpoint". El cliente (Dashboard.jsx) reintenta
        # 1× tras 500ms y la mayoría de los blips se resuelven sin toast de
        # error. Bugs reales (KeyError, ValueError sobre plan_data mal formado)
        # siguen devolviendo 500 — son determinísticos y reintentar es ruido.
        from db_facts import _is_transient_db_error
        if _is_transient_db_error(e):
            raise HTTPException(
                status_code=503,
                detail="DB temporalmente saturada. Reintenta en unos segundos.",
            )
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/telemetry/pdf-stale-fallback")
def api_pdf_stale_fallback_telemetry(
    data: dict = Body(...),
    verified_user_id: Optional[str] = Depends(_PDF_TELEMETRY_LIMITER),
):
    """[P2-SHOPPING-3 · 2026-05-14] Sink backend para el evento
    `pdf_stale_inventory_fallback` emitido por el frontend cuando el fetch
    de inventario fresco falla/timeoutea y el PDF se genera con
    `liveInventory` cacheado (P1-PDF-1).

    Antes (audit 2026-05-13):
        El frontend hacía `trackEvent('pdf_stale_inventory_fallback', ...)`
        que solo enviaba a Sentry/PostHog/GA/GTM (canales externos). El
        backend NO observaba la frecuencia → imposible escalar a
        `system_alerts` cuando un blip de Supabase mantiene a TODA la flota
        en stale fallback durante horas. Operador dependía de mirar Sentry
        manualmente.

    Fix:
        Endpoint best-effort que persiste a `pipeline_metrics` con
        `node='pdf_stale_inventory_fallback'`. El cron
        `_alert_pdf_stale_inventory_fallback_burst` lee la tabla cada
        N min y emite `system_alerts.pdf_stale_inventory_fallback_burst`
        si el count supera umbral.

    NO usa `verify_api_quota` (paywall sería absurdo para telemetría).
    Solo `get_verified_user_id` — usuario autenticado al descargar PDF.

    Body esperado (defensive parse):
      - `reason`: str ∈ {timeout, error, empty_response} (cap a 64 chars).
      - `fallback_inventory_size`: int >= 0 (size del cache usado).

    Cualquier campo malformado se sustituye por default conservador —
    el endpoint NUNCA falla por payload inválido (fire-and-forget desde
    frontend; un 4xx haría que el caller catch+log y posiblemente abortar
    el PDF, lo que es peor que telemetría perdida).

    Tooltip-anchor: P2-SHOPPING-3.
    """
    try:
        if not verified_user_id:
            raise HTTPException(status_code=401, detail="No autorizado.")

        reason = data.get("reason") if isinstance(data, dict) else None
        if not isinstance(reason, str) or len(reason) == 0 or len(reason) > 64:
            reason = "unknown"

        fallback_size = data.get("fallback_inventory_size") if isinstance(data, dict) else None
        if isinstance(fallback_size, bool) or not isinstance(fallback_size, int) or fallback_size < 0:
            fallback_size = None
        # [P3-PDF-OBS-FU-B · 2026-05-14] Clamp superior defensivo. Sin cap,
        # un POST con `fallback_inventory_size=999999999` se persistía tal
        # cual en `pipeline_metrics.metadata` jsonb — pollution menor sin
        # impacto operacional pero un cliente adversarial podía inflar
        # rows con payloads grandes. 100000 es ~3 órdenes de magnitud
        # sobre el realista (inventarios típicos: 10-200 items); valor
        # mayor sugiere bug o ataque, descartar al cap.
        elif fallback_size is not None and fallback_size > 100000:
            fallback_size = 100000

        from db_core import execute_sql_write
        execute_sql_write(
            """
            INSERT INTO pipeline_metrics
                (user_id, session_id, node, duration_ms, retries,
                 tokens_estimated, confidence, metadata)
            VALUES (%s, NULL, %s, 0, 0, 0, 0, %s::jsonb)
            """,
            (
                verified_user_id,
                "pdf_stale_inventory_fallback",
                _json.dumps({
                    "reason": reason,
                    "fallback_inventory_size": fallback_size,
                }, ensure_ascii=False),
            ),
        )
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        # Best-effort: NUNCA 5xx — telemetría perdida es preferible a un
        # toast.error en el cliente confundiendo al usuario que sí ya
        # descargó el PDF.
        logger.warning(f"[P2-SHOPPING-3] telemetry insert falló (best-effort): {e}")
        return {"success": False, "error": "telemetry_insert_failed"}


@router.get("/{plan_id}/chunk-status")
def api_chunk_status(plan_id: str, response: Response, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    """[P0-HIST-IDOR-2 · 2026-05-10] Estado de generación del plan + chunks pausados/fallidos.

    Bug original (audit Historial 2026-05-10): el handler leía
    `user_id` de `meal_plans` (línea 3334) pero NUNCA lo comparaba
    contra `verified_user_id`. Cualquier user authenticated podía
    leer info de planes ajenos:
      - `last_learning_hint` con `quality_history[-1].score` del
        `user_profiles.health_profile` del DUEÑO ajeno (línea 3476-3482).
      - `failed_chunks` con `attempts` (telemetría operacional).
      - `paused_chunks` con `reason_code` resuelto desde
        `pipeline_snapshot._pause_reason`/`_pantry_pause_reason`/
        `dead_letter_reason` (filtra estado de pantry/tz/learning).
      - `tier_breakdown` (calidad de generación por tier).
      - `_user_action_required` con CTA preformateado y
        `_recovery_exhausted_chunks` (lista chunks dead-lettered).
      - `_pantry_degraded_summary` (estado de inventario).

    Polling-friendly endpoint: el frontend lo llama cada 2-5s
    durante la generación → IDOR explotable a alta frecuencia.
    Identificado como follow-up en P1-HIST-AUDIT-NEW-1.

    Fix: tras el 404 por plan inexistente, comparar el `user_id`
    leído con `verified_user_id`. Si mismatch → 404 (no 403, para
    no leak la existencia del plan ajeno; mismo contrato que
    DELETE /{plan_id}:4389 y P0-HIST-IDOR-1 retry-chunk).

    Tooltip-anchor: P0-HIST-IDOR-2-START | test_p0_hist_idor_2_chunk_status_ownership
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not plan_id or not isinstance(plan_id, str):
        raise HTTPException(status_code=400, detail="plan_id required")

    from db_core import execute_sql_query
    try:
        res = execute_sql_query("SELECT user_id, plan_data FROM meal_plans WHERE id = %s", (plan_id,), fetch_one=True)
        if not res:
            raise HTTPException(status_code=404, detail="Plan no encontrado")

        user_id = res["user_id"]
        # [P0-HIST-IDOR-2] Ownership check explícito. Sin esto, todo
        # el payload de abajo (status/learning_hint/tier_breakdown/
        # paused_chunks/_pantry_degraded_summary/_user_action_required)
        # se serializa al cliente sin importar a quién pertenezca el
        # plan. 404 (no 403) para no filtrar la existencia del plan
        # ajeno — mismo contrato que DELETE /{plan_id} y retry-chunk.
        if str(user_id) != str(verified_user_id):
            raise HTTPException(status_code=404, detail="Plan no encontrado")
        # P0-HIST-IDOR-2-END
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

        # [P0-DASH-CHIP-HONESTY · 2026-05-09] Tooltip-anchor:
        # P0-DASH-CHIP-HONESTY-START | test_p0_dash_chip_honesty
        #
        # Counters operacionales del `plan_chunk_queue` para que el
        # Dashboard pueda diferenciar 3 estados visuales en la
        # ventana del plan ACTIVO (no solo en el listado del Historial,
        # ya cubierto por P0-AUDIT-HIST-2 / P1-AUDIT-HIST-4):
        #   - in_flight_count: chunks pending/processing/stale (algo
        #     corriendo o programado para pickup).
        #   - pending_user_action_count: chunks pausados esperando
        #     intervención del usuario (nevera vacía, snapshot stale,
        #     missing prior lessons, etc).
        #   - paused_chunks: lista resumida (chunk_id, week, days_offset,
        #     days_count, reason_code, execute_after) — el frontend la
        #     usa para pintar slots faltantes con copy honesto en lugar
        #     de "en camino" que ahora miente cuando los chunks están
        #     pausados.
        #
        # Bug original (reportado en producción 2026-05-09):
        #   Plan con first_chunk completed (Sáb-Dom) + rolling_refill
        #   pending_user_action (Lun-Jue) por `empty_pantry_proactive`.
        #   `plan_data.generation_status='generating_next'` y
        #   `_user_action_required=NULL`. Dashboard.jsx:2744 calcula
        #   `_isPending = _hasActionReq && !_isGenerating` → false →
        #   pinta "Lunes - en camino" con shimmer + spinner. La queue
        #   dice que Lunes está PAUSADO esperando que el usuario
        #   actualice su nevera. UX miente.
        #
        # Fix: el endpoint ahora expone los chunks paused con reason
        # resuelto (deriva de `_pause_reason` / `_pantry_pause_reason`
        # / `dead_letter_reason` con la misma prioridad que
        # `/blocked_reasons` P2-HIST-AUDIT-9). El frontend chequea
        # `pending_user_action_count > 0` ANTES de la rama
        # "_isGenerating" y muestra "Pausado: <reason>" con CTA.
        paused_rows = execute_sql_query(
            """
            SELECT id::text AS chunk_id,
                   week_number,
                   days_offset,
                   days_count,
                   chunk_kind,
                   pipeline_snapshot,
                   dead_letter_reason,
                   execute_after,
                   updated_at,
                   EXTRACT(EPOCH FROM (NOW() - updated_at))::int AS paused_seconds
            FROM plan_chunk_queue
            WHERE meal_plan_id = %s
              AND status = 'pending_user_action'
            ORDER BY week_number ASC NULLS LAST, days_offset ASC NULLS LAST
            """,
            (plan_id,),
        ) or []

        counters_row = execute_sql_query(
            """
            SELECT
                COUNT(*) FILTER (WHERE status IN ('pending','processing','stale'))::int AS in_flight_count,
                COUNT(*) FILTER (WHERE status = 'pending_user_action')::int AS pending_user_action_count,
                COUNT(*) FILTER (WHERE status = 'failed')::int AS failed_count,
                COUNT(*) FILTER (WHERE status = 'completed')::int AS completed_count
            FROM plan_chunk_queue
            WHERE meal_plan_id = %s
            """,
            (plan_id,),
            fetch_one=True,
        ) or {}

        # Resolver reason_code con el MISMO orden de prioridad que
        # `/blocked_reasons` (plans.py:3686-3728): dead_letter_reason →
        # _pause_reason → _pantry_pause_reason → reason → _learning_zero_logs
        # → "_unknown". SSOT extraído sería ideal; por ahora duplicamos
        # la lógica con cita explícita para que un futuro refactor las
        # consolide (anchor: P0-DASH-CHIP-HONESTY).
        import json as _json
        paused_chunks = []
        for r in paused_rows:
            snap = r.get("pipeline_snapshot") or {}
            if isinstance(snap, str):
                try:
                    snap = _json.loads(snap)
                except Exception:
                    snap = {}
            dead_reason = r.get("dead_letter_reason")
            if dead_reason:
                reason_code = str(dead_reason)
            elif snap.get("_pause_reason"):
                reason_code = str(snap["_pause_reason"])
            elif snap.get("_pantry_pause_reason"):
                reason_code = str(snap["_pantry_pause_reason"])
            elif snap.get("reason"):
                reason_code = str(snap["reason"])
            elif snap.get("_learning_zero_logs"):
                reason_code = "learning_zero_logs"
            else:
                reason_code = "_unknown"

            _exec_after = r.get("execute_after")
            _updated = r.get("updated_at")
            paused_chunks.append({
                "chunk_id": r.get("chunk_id"),
                "week_number": r.get("week_number"),
                "days_offset": r.get("days_offset"),
                "days_count": r.get("days_count"),
                "chunk_kind": r.get("chunk_kind"),
                "reason_code": reason_code,
                "execute_after": (_exec_after.isoformat() if hasattr(_exec_after, "isoformat") else _exec_after),  # pyright: ignore[reportOptionalMemberAccess]  (guarded por hasattr)
                "updated_at": (_updated.isoformat() if hasattr(_updated, "isoformat") else _updated),  # pyright: ignore[reportOptionalMemberAccess]  (guarded por hasattr)
                "paused_seconds": int(r.get("paused_seconds") or 0),
            })
        # P0-DASH-CHIP-HONESTY-END
        
        last_learning_hint = "Analizando tus preferencias..."
        user_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
        if user_res and user_res.get("health_profile"):
            hp = user_res["health_profile"]
            # [A5-KEYDRIFT · 2026-05-29] La clave real es `quality_history_chunks`
            # (lista de FLOATS 0-1, cron_tasks.py:15281), NO `quality_history` (dead key,
            # cero writers). Pre-fix el hint quedaba atascado en "Analizando..." para todos.
            # Fallback al nombre viejo + guard de tipo (float nuevo vs dict legacy).
            qh = hp.get("quality_history_chunks") or hp.get("quality_history", [])
            if qh and len(qh) > 0:
                _last = qh[-1]
                if isinstance(_last, dict):
                    last_score = _last.get("score", 0)  # formato legacy dict (0-100)
                else:
                    try:
                        last_score = round(float(_last) * 100)  # float 0-1 → 0-100
                    except (TypeError, ValueError):
                        last_score = 0
                if last_score:
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

        # [P1-CHUNKS-1] Exponer `_user_action_required` cuando un chunk dead-letteró.
        # Antes el flag se persistía en plan_data pero NUNCA se exponía al frontend
        # → si el push fallaba (Firebase saturado, permisos removidos), el usuario no
        # tenía forma de enterarse que su plan se atascó hasta que abriera la app y
        # notara que no avanzaba. Ahora el frontend (Dashboard.jsx, Plan.jsx) puede
        # leer este campo del polling de chunk-status y renderizar un banner con CTA
        # ("Regenerar plan") incondicionalmente — el push deja de ser el único canal
        # de notificación. El payload completo (title/body/cta/url/chunk_id/reason)
        # viene preformateado desde `_escalate_unrecoverable_chunk` para que el
        # frontend no necesite duplicar el mapping reason→copy.
        user_action_required = plan_data.get("_user_action_required") if isinstance(plan_data, dict) else None
        recovery_exhausted_chunks = (
            plan_data.get("_recovery_exhausted_chunks") if isinstance(plan_data, dict) else None
        )

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
            # [P1-CHUNKS-1] Banner persistente para chunks dead-lettered.
            "user_action_required": user_action_required,
            "recovery_exhausted_chunks": recovery_exhausted_chunks,
            # [P0-DASH-CHIP-HONESTY · 2026-05-09] Counters operacionales
            # del plan_chunk_queue + lista de chunks pausados con reason
            # resuelto. Permite que el Dashboard distinga "en camino"
            # (in_flight > 0) de "pausado" (pending_user_action > 0).
            # Counters siempre presentes (COALESCE 0); paused_chunks
            # vacío cuando no hay paused.
            "in_flight_count": int(counters_row.get("in_flight_count") or 0),
            "pending_user_action_count": int(counters_row.get("pending_user_action_count") or 0),
            "failed_count": int(counters_row.get("failed_count") or 0),
            "completed_count": int(counters_row.get("completed_count") or 0),
            "paused_chunks": paused_chunks,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] en chunk-status: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


# [P2-HIST-AUDIT-A · 2026-05-09] Helper SSOT para aplicar
# Cache-Control no-store + Pragma no-cache a las respuestas de los
# endpoints derivados del Historial (los que el modal/listado consume
# y son sensibles a mutaciones recientes: rename, delete, restore,
# nueva generación, transición de status del cron).
#
# Bug original (audit Historial 2026-05-09):
#   Solo `/history-list` setteaba `Cache-Control: no-store` (P1-HIST-
#   AUDIT-4-FOLLOWUP). Los endpoints derivados (`/lessons-counts`,
#   `/history-status-summary`, `/{id}/lessons`, `/{id}/coherence-
#   history`, `/{id}/chunk-metrics`, `/{id}/lifetime-lessons`,
#   `/{id}/blocked_reasons`) NO. Un browser agresivo con
#   back-forward cache puede servir respuestas stale tras un restore/
#   delete — el listado se refrescaba (no-store) pero el modal abría
#   con datos viejos hasta que el TTL del singleton (P2-HIST-AUDIT-11)
#   expirara.
#
# Aplicación: cada endpoint llama `_apply_no_store(response)` justo
# después del auth check. NO usar middleware global — solo para los
# endpoints del Historial donde la freshness es contractual.
def _apply_no_store(response: Response) -> None:
    """Aplica `Cache-Control: no-store, max-age=0` + `Pragma: no-cache`
    a la respuesta. Para endpoints derivados del Historial."""
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"


@router.get("/{plan_id}/blocked_reasons")
def api_blocked_reasons(
    plan_id: str,
    response: Response,
    include_failed: bool = False,
    include_stuck: bool = False,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """[P1-2] Devuelve los motivos legibles por los que un plan está bloqueado.

    El frontend usa este endpoint para mostrar un banner persistente cuando un chunk
    está pausado en `pending_user_action` con su motivo (zero-log, pantry vacía, snapshot
    obsoleto), evitando que el usuario crea que el plan "se estancó" sin razón visible.

    Antes solo se mandaban hasta 2 push notifications y luego silencio durante horas.

    [P2-HIST-AUDIT-9 · 2026-05-09] Cobertura extendida:
      - Nuevos `reason_codes`: `tz_unresolved`, `missing_prior_lessons`,
        `missing_start_date_no_anchor`. Antes el endpoint solo conocía 4
        codes (zero-log, stale_snapshot/live_unreachable, empty_pantry);
        cualquier otro caía al fallback `empty_pantry`, mintiendo al
        usuario sobre la causa real.
      - `include_failed=true` (default False, retrocompat con Dashboard
        del plan ACTIVO que solo quiere `pending_user_action`): añade
        chunks en estado `failed` con `dead_letter_reason` poblado al
        response. El modal del Historial usa esto para enumerar los
        motivos de chunks dead-lettered (recovery_exhausted,
        unrecoverable_missing_anchor, unrecoverable_corrupted_date,
        missing_prior_lessons_unrecoverable) — antes esos eran
        invisibles excepto via `_user_action_required` agregado.

    [P1-HIST-BLOCKED-STUCK · 2026-05-09] Cobertura de chunks atascados:
      - `include_stuck=true` (default False): añade chunks en estado
        `processing` o `stale` con `execute_after < NOW() - lag_threshold`.
        Threshold configurable vía `MEALFIT_BLOCKED_REASONS_STUCK_LAG_HOURS`
        (default 3.0h, validator > 0). Cierra el gap del audit Historial
        2026-05-09 (P1-3): los crons `_alert_high_chunk_lag` detectan
        chunks atascados >1h pero el banner del modal solo veía
        `pending_user_action` y `failed`. Un chunk en `processing`
        con lag de 6h (worker zombi tras crash, lock heredado por advisory
        no liberado, etc.) no tenía surface en UI — el plan parecía
        "Generando 2/15" hasta que un cron lo escaló a `failed` o el
        usuario regeneraba. Nuevos `reason_codes`: `stuck_processing`,
        `stuck_stale` con copy informativo + CTA opcional.
    """
    # [P2-HIST-AUDIT-A · 2026-05-09] Cache-Control no-store —
    # extensión del patrón de /history-list. Sin esto, el browser BFCache
    # puede servir reasons obsoletas tras restore/delete del plan.
    _apply_no_store(response)
    from db_core import execute_sql_query
    # [P1-HIST-BLOCKED-STUCK · 2026-05-09] Knob lazy-import (mismo
    # patrón que `shopping_calculator.py:104` y otros call sites de
    # `_env_float`). Lazy evita ciclo de import al cargar el módulo
    # routers/plans antes de que el registry de knobs esté listo.
    from knobs import _env_float as _knob_env_float
    _stuck_lag_hours = _knob_env_float(
        "MEALFIT_BLOCKED_REASONS_STUCK_LAG_HOURS",
        3.0,
        validator=lambda v: v > 0,
    )
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

        # [P2-HIST-AUDIT-9 · 2026-05-09] Cuando `include_failed=True`,
        # extendemos el WHERE con `OR (status='failed' AND
        # dead_letter_reason IS NOT NULL)`. Sin la guard `dead_letter_reason
        # IS NOT NULL`, traeríamos chunks `failed` transitorios (recoverable)
        # que el cron va a reprocesar — esos no merecen banner.
        #
        # [P1-HIST-BLOCKED-STUCK · 2026-05-09] Cuando `include_stuck=True`
        # sumamos `OR (status IN ('processing','stale') AND execute_after
        # < NOW() - interval N hours)`. La guard del lag evita mostrar
        # chunks que apenas arrancaron procesamiento — solo los que
        # llevan `>= _stuck_lag_hours` colgados sin transición. El
        # threshold como parameter de query NO permite override (el
        # cliente no puede bajarlo a 0 para DOSear el endpoint listando
        # chunks healthy); el knob es la única vía operacional.
        _filters = ["status = 'pending_user_action'"]
        _params: list = [plan_id]
        if include_failed:
            _filters.append("(status = 'failed' AND dead_letter_reason IS NOT NULL)")
        if include_stuck:
            # [P0-HIST-FIX-1 · 2026-05-09] PostgreSQL
            # `make_interval(hours => ...)` SOLO acepta int; pasarle
            # un numeric/float (el knob `MEALFIT_BLOCKED_REASONS_STUCK_LAG_HOURS`
            # default `3.0`) lanza `42883 function make_interval(hours
            # => numeric) does not exist`. El endpoint devolvía 500 al
            # frontend cada vez que el modal del Historial trataba de
            # lazy-fetch reasons (siempre que hay chunks en queue),
            # rompiendo el modal entero.
            #
            # Fix: usar multiplicación de interval — `interval '1 hour'
            # * %s` acepta numeric sin cast. Preserva la precisión
            # sub-hora del knob (3.5h = 3h 30min) que make_interval
            # habría truncado a 3h.
            _filters.append(
                "(status IN ('processing', 'stale') "
                "AND execute_after < NOW() - (interval '1 hour' * %s))"
            )
            _params.append(_stuck_lag_hours)
        _status_filter = " OR ".join(_filters)
        rows = execute_sql_query(
            f"""
            SELECT id, week_number, pipeline_snapshot, status, dead_letter_reason,
                   EXTRACT(EPOCH FROM (NOW() - updated_at))::int AS paused_seconds,
                   EXTRACT(EPOCH FROM (NOW() - execute_after))::int AS lag_seconds
            FROM plan_chunk_queue
            WHERE meal_plan_id = %s AND ({_status_filter})
            ORDER BY week_number ASC
            """,
            (plan_id, *(_params[1:])),
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
            # [P2-HIST-AUDIT-9 · 2026-05-09] Reasons faltantes
            # cubiertos. Antes caían al fallback `empty_pantry` (copy
            # incorrecto). Cada uno deriva de un cron específico:
            "tz_unresolved": {
                "title": "Confirmando tu zona horaria",
                "body": "Aún no pudimos resolver tu zona horaria para programar el siguiente bloque. Abre la app desde tu dispositivo principal.",
                "cta": "Abrir Mealfit",
                "url": "/dashboard",
            },
            "missing_prior_lessons": {
                "title": "Reconstruyendo el aprendizaje del plan",
                "body": "El sistema está intentando recuperar el aprendizaje de los días previos. Si persiste, regenera el plan.",
                "cta": None,
                "url": None,
            },
            "missing_start_date_no_anchor": {
                "title": "Tu plan necesita una fecha de inicio",
                "body": "No pudimos determinar cuándo comienza el plan. Reactívalo o regéneralo desde el Dashboard.",
                "cta": "Ir al Dashboard",
                "url": "/dashboard",
            },
            # [P2-HIST-AUDIT-9 · 2026-05-09] dead_letter_reasons (chunks
            # `failed` con dead_lettered_at). Cubre el catálogo de
            # `_escalate_unrecoverable_chunk` (cron_tasks.py:7984+).
            "recovery_exhausted": {
                "title": "Tu plan necesita atención",
                "body": "No pudimos completar parte de tu plan automáticamente. Reactívalo o regéneralo para continuar.",
                "cta": "Regenerar plan",
                "url": "/dashboard?recovery_exhausted=1",
            },
            "unrecoverable_missing_anchor": {
                "title": "Tu plan necesita regenerarse",
                "body": "Detectamos un problema técnico con la fecha de inicio del plan. Tócalo para regenerarlo con tu nevera actual.",
                "cta": "Regenerar plan",
                "url": "/dashboard?action_required=missing_anchor",
            },
            "unrecoverable_corrupted_date": {
                "title": "Tu plan necesita regenerarse",
                "body": "Detectamos datos inválidos en la fecha de inicio. Regenera el plan con tu nevera actual.",
                "cta": "Regenerar plan",
                "url": "/dashboard?action_required=corrupted_date",
            },
            "missing_prior_lessons_unrecoverable": {
                "title": "Tu plan necesita regenerarse",
                "body": "No pudimos reconstruir el historial de aprendizaje de los días previos. Regenera el plan.",
                "cta": "Regenerar plan",
                "url": "/dashboard?action_required=missing_lessons",
            },
            "restore_overwrite": {
                "title": "Chunk cancelado por restore",
                "body": "Este chunk fue cancelado al reactivar otro plan archivado. No requiere acción.",
                "cta": None,
                "url": None,
            },
            "restore_source_archived": {
                "title": "Chunk cancelado al archivar",
                "body": "Este chunk fue cancelado cuando el plan se reactivó como archivado. No requiere acción.",
                "cta": None,
                "url": None,
            },
            # [P1-HIST-BLOCKED-STUCK · 2026-05-09] Reasons para chunks
            # atascados en `processing` o `stale` con lag alto. NO
            # implican que el plan esté roto — el cron `_alert_high_
            # chunk_lag` ya los reporta al ops team. Para el usuario
            # final son "se está demorando más de lo esperado". CTA
            # neutro a Dashboard donde puede regenerar si la espera
            # le incomoda; sin scare copy.
            "stuck_processing": {
                "title": "Tu plan está tardando más de lo habitual",
                "body": "Un bloque del plan lleva tiempo procesándose. Suele resolverse solo, pero puedes regenerar el plan si prefieres no esperar.",
                "cta": "Ir al Dashboard",
                "url": "/dashboard",
            },
            "stuck_stale": {
                "title": "Reanudando un bloque del plan",
                "body": "Un bloque del plan quedó marcado para reanudar tras una interrupción del worker. El cron lo retomará automáticamente.",
                "cta": None,
                "url": None,
            },
        }

        # [P2-HIST-AUDIT-9 · 2026-05-09] Fallback genérico para reason
        # codes desconocidos. Antes el código caía a `empty_pantry`
        # (mensaje específico que mentía si el reason real era otro).
        # Este fallback genérico no especula la causa.
        _UNKNOWN_REASON_TEMPLATE = {
            "title": "Bloqueo sin clasificar",
            "body": "El sistema marcó este chunk como bloqueado pero no logramos identificar la causa. Si persiste, contacta soporte.",
            "cta": None,
            "url": None,
        }

        import json as _json
        for row in rows:
            snap = row.get("pipeline_snapshot") or {}
            if isinstance(snap, str):
                try:
                    snap = _json.loads(snap)
                except Exception:
                    snap = {}
            # [P2-HIST-AUDIT-9 · 2026-05-09] Selección de reason_code
            # con prioridad por especificidad:
            #   1. Si chunk es 'failed' con dead_letter_reason → ese
            #      es el reason canónico (más definitivo que el
            #      pause_reason del snapshot, que pudo quedar stale).
            #   2. `_pause_reason` (P1-CHUNKS-3 missing_prior_lessons,
            #      seteado dentro del pipeline LangGraph).
            #   3. `_pantry_pause_reason` (los crons de pause pantry/
            #      tz/snapshot lo escriben).
            #   4. `learning_zero_logs` por compat legacy (algunos
            #      paths antiguos lo escriben directo en el snapshot).
            #   5. Fallback genérico (NO `empty_pantry` que mentía).
            row_status = row.get("status")
            dead_reason = row.get("dead_letter_reason")
            if row_status == "failed" and dead_reason:
                reason_code = str(dead_reason)
            elif row_status == "processing":
                # [P1-HIST-BLOCKED-STUCK · 2026-05-09] El SELECT ya
                # filtró por lag_seconds > _stuck_lag_hours; aquí solo
                # asignamos el reason_code canónico. Si el chunk
                # llegara con status='processing' por una rama futura
                # que no aplique el filtro (e.g. include_stuck=False
                # pero el WHERE evolucionó), la guard sigue siendo
                # consistente: status define el reason_code.
                reason_code = "stuck_processing"
            elif row_status == "stale":
                reason_code = "stuck_stale"
            else:
                reason_code = (
                    str(snap.get("_pause_reason"))
                    if snap.get("_pause_reason")
                    else (
                        str(snap.get("_pantry_pause_reason"))
                        if snap.get("_pantry_pause_reason")
                        else str(snap.get("reason") or "")
                    )
                )
                if not reason_code:
                    # Detectar zero-logs legacy desde el snapshot.
                    if snap.get("_learning_zero_logs"):
                        reason_code = "learning_zero_logs"
                    else:
                        reason_code = "_unknown"

            template = reason_to_text.get(reason_code, _UNKNOWN_REASON_TEMPLATE)
            reasons.append({
                "chunk_id": row.get("id"),
                "week_number": row.get("week_number"),
                "reason_code": reason_code,
                "status": row_status,
                "dead_letter_reason": dead_reason,
                "paused_seconds": int(row.get("paused_seconds") or 0),
                # [P1-HIST-BLOCKED-STUCK · 2026-05-09] `lag_seconds` =
                # NOW - execute_after. Para chunks `processing`/`stale`
                # del filtro stuck es la métrica diagnóstica clave (vs
                # `paused_seconds` que es NOW - updated_at — puede
                # estar fresco si el worker hizo heartbeat reciente
                # sin avanzar). Para chunks pending_user_action/failed
                # también se incluye por consistencia (puede ser útil
                # para estimar "cuánto tiempo lleva bloqueado el plan").
                "lag_seconds": int(row.get("lag_seconds") or 0),
                **template,
            })

        return {"plan_id": plan_id, "blocked": len(reasons) > 0, "reasons": reasons}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] en blocked_reasons: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/{plan_id}/retry-chunk/{chunk_id}")
def api_retry_chunk(plan_id: str, chunk_id: str, verified_user_id: Optional[str] = Depends(verify_api_quota)):
    """[P0-HIST-IDOR-1 · 2026-05-10] Reenvía un chunk fallido a la cola.

    Bug original (audit 2026-05-10): los tres `UPDATE` filtraban solo por
    `plan_id`/`chunk_id` sin chequear que el plan pertenezca al usuario
    autenticado. Cualquier user con auth podía:
      1. resetear `plan_chunk_queue` por `(chunk_id, plan_id)` en planes
         ajenos → forzar re-ejecución de chunks de otros usuarios.
      2. revivir todos los `cancelled` de un plan ajeno por `meal_plan_id`.
      3. mutar `meal_plans.generation_status='partial'` con
         `WHERE id = %s` puro → forzar polling sobre planes de víctimas.

    Adicional: `verify_api_quota` cobra cuota al ATACANTE (no al dueño)
    → DOS amplificable contra cualquier plan ajeno con el costo del
    quota propio del atacante.

    Fix: ownership check explícito (mismo patrón que `DELETE /{plan_id}`
    en plans.py:4380-4389 y `/blocked_reasons` en plans.py:3637-3645) +
    `AND user_id = %s` defense-in-depth en cada UPDATE para que un race
    en el ownership check no permita la mutación parcial.

    Tooltip-anchor: P0-HIST-IDOR-1-START | test_p0_hist_idor_1_retry_chunk_ownership
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not plan_id or not isinstance(plan_id, str):
        raise HTTPException(status_code=400, detail="plan_id required")
    if not chunk_id or not isinstance(chunk_id, str):
        raise HTTPException(status_code=400, detail="chunk_id required")

    from db_core import execute_sql_query, execute_sql_write
    try:
        # 1) Ownership check explícito. Sin esto, los UPDATE de abajo
        #    podrían tocar filas de planes ajenos cuando el filtro
        #    `AND user_id = %s` empareja una fila que originalmente
        #    pertenece al usuario pero apunta vía PK a otra tabla
        #    (defensa-en-profundidad: doble candado).
        owner = execute_sql_query(
            "SELECT id FROM meal_plans WHERE id = %s AND user_id = %s",
            (plan_id, verified_user_id),
            fetch_one=True,
        )
        if not owner:
            # Devolvemos 404 (no 403) para no filtrar la existencia del
            # plan ajeno. Mismo patrón que `DELETE /{plan_id}` línea 4389.
            raise HTTPException(status_code=404, detail="Plan no encontrado")

        # 2) Resetear el chunk fallido a 'pending'. Filtro por
        #    meal_plan_id + (subquery user_id) defense-in-depth: si
        #    una race-condition borra el plan entre el ownership check
        #    y este UPDATE, el filtro por user_id evita que reseteemos
        #    un chunk huérfano que ya no nos pertenece.
        execute_sql_write("""
            UPDATE plan_chunk_queue
            SET status = 'pending',
                attempts = 0,
                execute_after = NOW(),
                updated_at = NOW()
            WHERE id = %s
              AND meal_plan_id = %s
              AND status = 'failed'
              AND meal_plan_id IN (SELECT id FROM meal_plans WHERE user_id = %s)
        """, (chunk_id, plan_id, verified_user_id))

        # 3) Revivir cualquier chunk que haya sido cancelado por culpa de este fallo
        execute_sql_write("""
            UPDATE plan_chunk_queue
            SET status = 'pending',
                attempts = 0,
                execute_after = NOW() + INTERVAL '1 minute',
                updated_at = NOW()
            WHERE meal_plan_id = %s
              AND status = 'cancelled'
              AND meal_plan_id IN (SELECT id FROM meal_plans WHERE user_id = %s)
        """, (plan_id, verified_user_id))

        # 4) Volver a poner el plan en 'partial' para que el frontend retome el polling.
        #
        # [P0-3 · 2026-05-10] El UPDATE sella DOS timestamps:
        #   - `updated_at` (columna física, cubierta por trigger P0-2 — la
        #     línea `updated_at = NOW()` es explícita por defense-in-depth y
        #     enforced por test_p0_2 para que un futuro refactor que la quite
        #     asumiendo solo el trigger sea consciente).
        #   - `plan_data._plan_modified_at` (SSOT semántico de "última edición
        #     del contenido del plan"). Sin esto, el Historial (P1-HIST-4)
        #     ordena al usuario que retry-cheó un chunk fallido por debajo
        #     de planes intactos del mismo día, porque sort se hace por jsonb
        #     path. Retry-chunk SÍ es una modificación visible (cambia
        #     `generation_status` + dispara re-ejecución), así que merece
        #     bumpear el sello semántico.
        # Patrón espejo de cron_tasks.py:370-381 (legacy persist).
        execute_sql_write("""
            UPDATE meal_plans
            SET plan_data = jsonb_set(
                    jsonb_set(plan_data, '{generation_status}', '"partial"'),
                    '{_plan_modified_at}',
                    to_jsonb(NOW()::text),
                    true
                ),
                updated_at = NOW()
            WHERE id = %s AND user_id = %s
        """, (plan_id, verified_user_id))

        # [P2-LIVE-7 · 2026-05-11] Audit api_usage (mismo motivo que /shift-plan:
        # verify_api_quota solo lee). Sin esto, un atacante podría retry-chunkear
        # ilimitadamente para forzar re-ejecución del worker LLM sin tocar su
        # cap mensual.
        try:
            log_api_usage(verified_user_id, "retry_chunk")
        except Exception as _audit_err:
            logger.warning(f"[P2-LIVE-7] log_api_usage retry_chunk falló: {_audit_err}")

        return {"success": True, "message": "Chunk reenviado a la cola"}
        # P0-HIST-IDOR-1-END

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] en retry-chunk: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/restore")
def api_restore_plan(
    data: dict = Body(...),
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """[P0-HIST-1 · 2026-05-09] Restauración atómica de un plan archivado.

    Reemplaza el patrón legacy frontend-only que solo hacía
    ``UPDATE meal_plans SET plan_data = ? WHERE id = <latest>`` desde
    `AssessmentContext.restorePlan`. Ese patrón dejaba chunks
    pending/processing apuntando a la fila destino con
    `pipeline_snapshot` del plan anterior y los workers seguían
    mergeando días generados al estilo previo dentro del plan_data
    recién sobrescrito → contaminación silenciosa del plan restaurado.
    Adicionalmente NO actualizaba columnas top-level (name/calories/
    macros/meal_names/ingredients/techniques) → drift entre `plan_data`
    interno y el header del Dashboard que las lee directo.

    Operación (todo dentro de una sola transacción Postgres):
      1. Cancela chunks pending/processing del target con
         `dead_letter_reason='restore_overwrite'` (preserva audit trail).
      2. Libera `chunk_user_locks` asociados a chunks de ese target
         (no toca locks de otros planes activos del usuario).
      3. Sobrescribe `plan_data` Y las 6 columnas top-level con los
         valores del source — cierre acoplado de P0-HIST-2.
      4. Anota `_plan_modified_at` y `_restored_from_plan_id` en
         plan_data (post-mortem y orden por actividad reciente).

    Body:
      ``{ "source_plan_id": "<uuid>" }`` — plan archivado a restaurar.
    Target:
      Fila más reciente del usuario por `created_at` (preserva la
      semántica del frontend legacy). Si `target.id == source_plan_id`,
      el endpoint es no-op idempotente.

    Raises:
      401 — sin auth.
      400 — `source_plan_id` faltante o no string.
      404 — source plan no existe o no pertenece al usuario.
      409 — usuario sin planes activos (no hay target a sobrescribir).
      500 — error de DB.
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    source_plan_id = (data or {}).get("source_plan_id")
    if not source_plan_id or not isinstance(source_plan_id, str):
        raise HTTPException(status_code=400, detail="source_plan_id required")

    from db_core import execute_sql_query, connection_pool
    import psycopg
    from psycopg.rows import dict_row

    # 1) Source: leer plan_data + columnas top-level. La cláusula
    #    `user_id = %s` es el ownership check (RLS-safe; no confiamos
    #    en que el cliente nos pase un id de otro usuario).
    src = execute_sql_query(
        """
        SELECT id, plan_data, name, calories, macros,
               meal_names, ingredients, techniques
        FROM meal_plans
        WHERE id = %s AND user_id = %s
        """,
        (source_plan_id, verified_user_id),
        fetch_one=True,
    )
    if not src:
        raise HTTPException(status_code=404, detail="Source plan not found")

    # 2) Enriquecer plan_data antes del UPDATE. `_plan_modified_at` es
    #    consumido por crons que filtran "planes editados últimas 24h"
    #    (P3-A re-evalúa coherencia). `_restored_from_plan_id` permite
    #    correlacionar para post-mortem si un restore corrompió algo.
    enriched_pd = dict(src.get("plan_data") or {})
    enriched_pd["_plan_modified_at"] = datetime.now(timezone.utc).isoformat()
    enriched_pd["_restored_from_plan_id"] = str(source_plan_id)

    # [P1-HIST-AUDIT-3 · 2026-05-09] Limpiar flags de fallo del SOURCE.
    # `_user_action_required` (banner CTA "regenera chunks fallados") y
    # `_recovery_exhausted_chunks` (lista de chunks dead-lettered) son
    # marcas del fallo histórico del SOURCE; arrastrarlas al target tras
    # restore deja al usuario con un banner amarillo + chip "failed/action"
    # en el plan recién reactivado, aunque la decisión explícita de
    # reactivar implica que se acepta el plan tal-cual. Si la generación
    # del target vuelve a fallar tras el restore, los crons re-setearán
    # las flags con los chunks REALES del nuevo intento.
    #
    # Cobertura del consumidor:
    #   - History.jsx::getStatusInfo (líneas 340-348) clasifica como
    #     `failed` cuando `_recovery_exhausted_chunks` no está vacío;
    #     `action_required` cuando `_user_action_required` está presente.
    #   - History.jsx::actionBanner (línea 833+) renderiza banner si
    #     `_user_action_required` o `_recovery_exhausted_chunks` no
    #     vacío.
    # Limpieza simétrica con esos consumidores: pop ambos en el origen.
    enriched_pd.pop("_user_action_required", None)
    enriched_pd.pop("_recovery_exhausted_chunks", None)

    target_plan_id = None
    cancelled_chunks = 0
    cancelled_source_chunks = 0
    released_locks = 0
    is_noop = False
    try:
        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cur:
                    # [P1-HIST-AUDIT-7 · 2026-05-09] Advisory lock
                    # per-user PRIMERO en la transacción. Sin esto,
                    # dos restores concurrentes leían el mismo
                    # `target` (latest) y ambos sobrescribían — el
                    # último gana, el primero se perdía silenciosa-
                    # mente. Lock transaccional: se libera al
                    # COMMIT/ROLLBACK; users distintos NO se bloquean
                    # entre sí (key per-user, no global).
                    from db_plans import acquire_user_history_advisory_lock
                    acquire_user_history_advisory_lock(cur, verified_user_id)

                    # 3) Target dentro del lock: el plan que el USUARIO
                    #    ve como activo en su UI.
                    #    [P1-HIST-AUDIT-1 · 2026-05-08] SSOT con frontend
                    #    History.jsx::_effectiveModifiedAt → max(created_at,
                    #    _plan_modified_at). Antes este SELECT ordenaba
                    #    solo por created_at, lo que abría drift cuando
                    #    el usuario ya había restaurado un plan archivado
                    #    con created_at anterior. GREATEST(a, COALESCE(b,
                    #    a)) replica el Math.max(b, a) del frontend con
                    #    b posiblemente null. Tie-breaker secundario por
                    #    created_at DESC mantiene determinismo entre ties.
                    #
                    #    Asunción: `_plan_modified_at`, cuando existe, es
                    #    ISO timestamptz válido. Todos los call sites del
                    #    backend lo sellan con datetime.isoformat() — si
                    #    una fila corrupta lo rompe, el cast falla
                    #    ruidosamente.
                    cur.execute(
                        """
                        SELECT id
                        FROM meal_plans
                        WHERE user_id = %s
                        ORDER BY GREATEST(
                            created_at,
                            COALESCE(
                                (plan_data->>'_plan_modified_at')::timestamptz,
                                created_at
                            )
                        ) DESC,
                        created_at DESC
                        LIMIT 1
                        """,
                        (verified_user_id,),
                    )
                    tgt_row = cur.fetchone()
                    if not tgt_row:
                        # Edge: usuario sin planes. Rechazamos para
                        # evitar comportamiento sorpresa (clonar el
                        # source crearía una fila adicional y rompería
                        # el invariante "1 plan activo").
                        raise HTTPException(
                            status_code=409,
                            detail="No active plan to overwrite. Generate a new plan first.",
                        )

                    target_plan_id = tgt_row["id"]
                    if str(target_plan_id) == str(source_plan_id):
                        # No-op idempotente: el plan archivado YA es
                        # el más reciente. Salimos del bloque sin
                        # tocar plan_chunk_queue/locks/meal_plans.
                        # El COMMIT del with-transaction libera el
                        # advisory lock automáticamente.
                        is_noop = True
                    else:
                        # 3a) Cancelar TODOS los chunks "vivos" del target.
                        #     [P0-AUDIT-HIST-1 · 2026-05-09] El filtro
                        #     histórico era `('pending', 'processing')`,
                        #     pero el SSOT del resto del backend
                        #     (db_plans.py:573, services.py:222,
                        #     routers/plans.py:2059) cubre 5 estados.
                        #     Restar `stale`, `pending_user_action` y
                        #     `failed` dejaba zombis: un chunk pausado
                        #     por pantry/tz/missing-lessons o uno con
                        #     heartbeat expirado podía despertarse tras
                        #     el restore con `pipeline_snapshot` del
                        #     plan PREVIO al target y escribir días al
                        #     plan_data recién sobrescrito → exactamente
                        #     la corrupción silenciosa que P0-HIST-1 fue
                        #     diseñado a prevenir.
                        #
                        #     `failed` también se cancela porque tras
                        #     restore su `pipeline_snapshot` ya no aplica
                        #     al plan vivo; el recovery cron NO debe
                        #     re-intentarlo contra `plan_data` restaurado.
                        #
                        #     `COALESCE(...)` preserva razones previas
                        #     si un cron ya marcó la fila como
                        #     cancelled por otro motivo (defensivo,
                        #     raro pero posible).
                        cur.execute(
                            """
                            UPDATE plan_chunk_queue
                            SET status = 'cancelled',
                                dead_letter_reason = COALESCE(
                                    dead_letter_reason, %s
                                ),
                                dead_lettered_at = COALESCE(
                                    dead_lettered_at, NOW()
                                ),
                                updated_at = NOW()
                            WHERE meal_plan_id = %s
                              AND status IN (
                                  'pending', 'processing', 'stale',
                                  'pending_user_action', 'failed'
                              )
                            """,
                            ("restore_overwrite", target_plan_id),
                        )
                        cancelled_chunks = cur.rowcount or 0

                        # 3a-bis) [P2-HIST-AUDIT-5 · 2026-05-09]
                        #     Cancelar chunks vivos del SOURCE también.
                        #     Antes el restore solo cancelaba chunks
                        #     del target — workers vivos del source
                        #     seguían corriendo y escribiendo al row
                        #     source tras el copy a target → row source
                        #     con plan_data modificado post-archivo
                        #     (debería ser snapshot inmutable). UX
                        #     confusa al hacer post-mortem y consumo
                        #     desperdiciado de slots LLM. Razón
                        #     distinta `restore_source_archived` separa
                        #     el audit trail de los cancels del target.
                        #
                        #     [P0-AUDIT-HIST-1 · 2026-05-09] Mismo
                        #     5-state SSOT que el cancel del target —
                        #     un chunk del source en pending_user_action
                        #     o stale, si nunca se cancela, puede
                        #     resucitar tras el restore (e.g., el cron
                        #     de unblock pantry refresh detecta el row
                        #     source todavía "vivo") y escribir contra
                        #     un plan_data que ya no es el activo.
                        cur.execute(
                            """
                            UPDATE plan_chunk_queue
                            SET status = 'cancelled',
                                dead_letter_reason = COALESCE(
                                    dead_letter_reason, %s
                                ),
                                dead_lettered_at = COALESCE(
                                    dead_lettered_at, NOW()
                                ),
                                updated_at = NOW()
                            WHERE meal_plan_id = %s
                              AND status IN (
                                  'pending', 'processing', 'stale',
                                  'pending_user_action', 'failed'
                              )
                            """,
                            ("restore_source_archived", source_plan_id),
                        )
                        cancelled_source_chunks = cur.rowcount or 0

                        # 3b) Liberar locks asociados a chunks del
                        #     target Y del source. `chunk_user_locks`
                        #     es per-user (UNIQUE constraint sobre
                        #     user_id) → puede haber a lo más UN
                        #     lock vivo, pero si pertenece a un chunk
                        #     del source que estaba corriendo, debe
                        #     liberarse igual (sino el siguiente
                        #     pickup del worker se queda colgado).
                        #
                        #     [P0-HIST-AUDIT-1 · 2026-05-09] La PK de
                        #     plan_chunk_queue es `id` (uuid). Antes
                        #     esta subquery referenciaba `chunk_id`,
                        #     columna que NO existe → Postgres lanzaba
                        #     `UndefinedColumn` → ROLLBACK del bloque
                        #     entero → 500 silencioso desde Historial.
                        #     `chunk_user_locks.locked_by_chunk_id` se
                        #     popula con `plan_chunk_queue.id`.
                        #     [P2-HIST-AUDIT-5 · 2026-05-09] Subquery
                        #     extendida con `IN (target, source)` para
                        #     cubrir locks de cualquiera de los dos
                        #     planes afectados por el restore.
                        cur.execute(
                            """
                            DELETE FROM chunk_user_locks
                            WHERE user_id = %s
                              AND locked_by_chunk_id IN (
                                  SELECT id FROM plan_chunk_queue
                                  WHERE meal_plan_id IN (%s, %s)
                              )
                            """,
                            (verified_user_id, target_plan_id, source_plan_id),
                        )
                        released_locks = cur.rowcount or 0

                        # 3c) Sobrescribir target con plan_data Y
                        #     columnas top-level. La cláusula
                        #     `user_id = %s` es redundante con el
                        #     SELECT inicial pero no se debe omitir
                        #     (defense-in-depth contra reuso accidental
                        #     del id en otro contexto).
                        cur.execute(
                            """
                            UPDATE meal_plans
                            SET plan_data = %s::jsonb,
                                name = %s,
                                calories = %s,
                                macros = %s::jsonb,
                                meal_names = %s,
                                ingredients = %s,
                                techniques = %s
                            WHERE id = %s AND user_id = %s
                            """,
                            (
                                _json.dumps(enriched_pd),
                                src.get("name"),
                                src.get("calories"),
                                _json.dumps(src.get("macros") or {}),
                                list(src.get("meal_names") or []),
                                list(src.get("ingredients") or []),
                                list(src.get("techniques") or []),
                                target_plan_id,
                                verified_user_id,
                            ),
                        )
        logger.info(
            "[P0-HIST-1] restore plan: user=%s target=%s source=%s "
            "cancelled_chunks=%d cancelled_source_chunks=%d "
            "released_locks=%d noop=%s",
            verified_user_id, target_plan_id, source_plan_id,
            cancelled_chunks, cancelled_source_chunks,
            released_locks, is_noop,
        )
        return {
            "success": True,
            "target_plan_id": str(target_plan_id),
            "source_plan_id": str(source_plan_id),
            "cancelled_chunks": cancelled_chunks,
            # [P2-HIST-AUDIT-5 · 2026-05-09] cancelled_source_chunks
            # expone separately el conteo de chunks del SOURCE que
            # estaban pending al momento del restore — útil para
            # diagnosticar UX confusa ("¿por qué mi plan archivado
            # cambió tras restore?") y verificar que el cleanup
            # cerró los workers correctamente.
            "cancelled_source_chunks": cancelled_source_chunks,
            "released_locks": released_locks,
            "noop": is_noop,
        }
    except HTTPException:
        raise
    except psycopg.Error as e:
        logger.error(f"❌ [P0-HIST-1] DB error en restore: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))
    except Exception as e:
        logger.error(f"❌ [P0-HIST-1] error en restore: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.delete("/{plan_id}")
def api_delete_plan(
    plan_id: str,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """[P0-HIST-3 · 2026-05-09] Eliminación atómica de un plan + cleanup
    de locks de chunks asociados.

    Reemplaza el patrón legacy del frontend (`History.jsx` hacía
    ``supabase.from('meal_plans').delete().eq('id', plan_id)``
    directo). Ese flujo dejaba dos clases de basura:

      1. **Locks zombi**: `chunk_user_locks` no tiene FK a
         `meal_plans` (sólo a `user_profiles`), así que un lock
         con `locked_by_chunk_id` apuntando a un chunk del plan
         eliminado seguía vivo hasta que el sweep cron lo expirara
         por `heartbeat_at` stale (típicamente 5-15 min). Si el
         plan eliminado era el activo, la siguiente generación
         podía colisionar con ese lock zombi.
      2. **Telemetría orphan**: `chunk_lesson_telemetry` y
         `chunk_deferrals` tampoco tenían FK; sus filas con
         `meal_plan_id` no resoluble crecían monotonas. La
         migración SSOT `p0_hist_3_telemetry_orphan_fk.sql` añade
         FK con `ON DELETE SET NULL` para resolver esta clase
         desde la base de datos (defensa en profundidad).

    El endpoint hace todo en una sola transacción Postgres:
      1. Verifica ownership con `SELECT ... WHERE id=? AND user_id=?`.
      2. `DELETE FROM chunk_user_locks` con subquery que filtra
         locks asociados a chunks del plan (sin tocar locks de
         otros planes activos del usuario).
      3. `DELETE FROM meal_plans WHERE id=? AND user_id=?` —
         CASCADE en `plan_chunk_queue` borra chunks pendientes,
         SET NULL en `plan_chunk_metrics` preserva métricas
         históricas, y (post-migración SSOT P0-HIST-3) SET NULL
         en `chunk_lesson_telemetry`/`chunk_deferrals`.

    Path param:
      `plan_id` — uuid del plan a eliminar.
    Returns:
      ``{ "success": True, "released_locks": N, "deleted": True }``

    Raises:
      401 — sin auth.
      404 — plan no existe o no pertenece al usuario.
      500 — error de DB.
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not plan_id or not isinstance(plan_id, str):
        raise HTTPException(status_code=400, detail="plan_id required")

    from db_core import execute_sql_query, connection_pool
    import psycopg
    from psycopg.rows import dict_row

    # 1) Verificación de ownership. Sin esto, un usuario con auth
    #    podría borrar planes de otros (RLS lo cubriría a nivel SQL,
    #    pero el endpoint debe responder 404 explícito en vez de
    #    "delete silenciosamente afectó 0 filas").
    owner_check = execute_sql_query(
        """
        SELECT id FROM meal_plans
        WHERE id = %s AND user_id = %s
        """,
        (plan_id, verified_user_id),
        fetch_one=True,
    )
    if not owner_check:
        raise HTTPException(status_code=404, detail="Plan not found")

    released_locks = 0
    try:
        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cur:
                    # [P1-HIST-AUDIT-7 · 2026-05-09] Advisory lock
                    # per-user para serializar mutators del Historial.
                    # Sin esto, un delete concurrente con un
                    # restore/rename del mismo user podía interleavearse
                    # (e.g., delete corre entre el SELECT target y el
                    # UPDATE del restore → restore termina escribiendo
                    # en una fila que ya no existe, o el target del
                    # restore cambia entre las dos lecturas).
                    from db_plans import acquire_user_history_advisory_lock
                    acquire_user_history_advisory_lock(cur, verified_user_id)

                    # 2) Liberar chunk_user_locks asociados a chunks
                    #    del plan. La subquery debe ejecutarse ANTES
                    #    del DELETE en meal_plans, porque ON DELETE
                    #    CASCADE de plan_chunk_queue borrará los rows
                    #    cuyos ids necesitamos para filtrar.
                    #
                    #    [P0-HIST-AUDIT-1 · 2026-05-09] La PK de
                    #    plan_chunk_queue es `id` (uuid); NO existe
                    #    una columna `chunk_id`. Antes esta subquery
                    #    la referenciaba → `UndefinedColumn` →
                    #    ROLLBACK → DELETE silenciosamente abortaba.
                    #    `chunk_user_locks.locked_by_chunk_id` se
                    #    popula con `plan_chunk_queue.id`.
                    cur.execute(
                        """
                        DELETE FROM chunk_user_locks
                        WHERE user_id = %s
                          AND locked_by_chunk_id IN (
                              SELECT id FROM plan_chunk_queue
                              WHERE meal_plan_id = %s
                          )
                        """,
                        (verified_user_id, plan_id),
                    )
                    released_locks = cur.rowcount or 0

                    # 3) DELETE meal_plans. CASCADE en
                    #    plan_chunk_queue + SET NULL en
                    #    plan_chunk_metrics + (post-migración)
                    #    chunk_lesson_telemetry/chunk_deferrals.
                    cur.execute(
                        """
                        DELETE FROM meal_plans
                        WHERE id = %s AND user_id = %s
                        """,
                        (plan_id, verified_user_id),
                    )
                    deleted_count = cur.rowcount or 0
                    if deleted_count == 0:
                        # Race: alguien borró entre owner_check y este
                        # DELETE. Tratar como 404 — la intención del
                        # usuario ya se satisfizo (el plan no existe).
                        raise HTTPException(
                            status_code=404, detail="Plan not found"
                        )
        logger.info(
            "[P0-HIST-3] delete plan: user=%s plan=%s released_locks=%d",
            verified_user_id, plan_id, released_locks,
        )
        return {
            "success": True,
            "released_locks": released_locks,
            "deleted": True,
        }
    except HTTPException:
        raise
    except psycopg.Error as e:
        logger.error(f"❌ [P0-HIST-3] DB error en delete: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))
    except Exception as e:
        logger.error(f"❌ [P0-HIST-3] error en delete: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


# [P1-HIST-AUDIT-5 · 2026-05-09] Whitelist de events de
# `chunk_lesson_telemetry` que son LECCIONES semánticas (cuentan para
# el chip "X lecciones" del Historial), versus métricas mecánicas /
# de salud (no son lecciones que el sistema "aprendió").
#
# [P1-AUDIT-HIST-7 · 2026-05-09] La fuente de verdad migró a
# `constants.LESSON_COUNT_EVENT_WHITELIST` para que cualquier
# consumidor (este endpoint, admin tools, monitoring) opere sobre la
# misma tupla — drift cero por construcción. El alias privado
# `_LESSON_COUNT_EVENT_WHITELIST` se preserva como módulo-attr para
# retrocompatibilidad con tests existentes (P1-HIST-AUDIT-5,
# P2-HIST-AUDIT-2) que leen el atributo via `getattr(plans_module, ...)`.
# Catálogo + clasificación documentados en `constants.py`.
from constants import LESSON_COUNT_EVENT_WHITELIST as _LESSON_COUNT_EVENT_WHITELIST


@router.get("/lessons-counts")
def api_plans_lessons_counts(
    response: Response,
    # [P1-AUDIT-3 · 2026-05-10] Uso `get_verified_user_id` (no
    # `verify_api_quota`) intencionalmente. Es un GET de polling del
    # Historial (read-only, sin costo LLM). Aplicar el paywall acá
    # negaría al usuario ver SU PROPIO historial cuando alcanzó el cap
    # mensual — UX inaceptable. Ver CLAUDE.md "Convenciones del repo"
    # sección Historial-quota-exemption.
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """[P1-HIST-3 · 2026-05-09] Conteo de lecciones (`chunk_lesson_telemetry`)
    por plan del usuario, con whitelist de events semánticos
    (P1-HIST-AUDIT-5).

    Diseñado como SINGLE roundtrip para el listado del Historial: en
    lugar de N queries (una por card visible) cuando el usuario abre
    `/history`, devolvemos `{plan_id: count}` para todos sus planes.
    El frontend cachea el resultado en state local mientras la página
    está montada.

    [P1-HIST-AUDIT-5 · 2026-05-09] El conteo filtra por
    `_LESSON_COUNT_EVENT_WHITELIST`. Antes (P1-HIST-3 v1) contábamos
    cualquier `event`, mezclando lecciones reales con métricas
    mecánicas (`synth_schema_invalid`, `learning_rebuild_failed`,
    etc.) — el chip "X lecciones" mentía cuando un plan tenía solo
    descartes de síntesis. La whitelist explícita está documentada
    arriba del endpoint con el catálogo completo.

    Otros filtros:
      - Excluye `meal_plan_id IS NULL` (orphans post-P0-HIST-3 SET
        NULL) — esos rows pertenecen a planes ya eliminados y no
        tienen card visible.
      - Excluye implícitamente planes con count=0 vía `GROUP BY` —
        el frontend trata "sin entrada" como cero (no agrega chip).

    RLS: la tabla `chunk_lesson_telemetry` tiene RLS habilitado;
    `WHERE user_id = %s` es defense-in-depth además del filtro RLS.

    Returns:
      ``{ "counts": { "<plan_id_str>": <count_int>, ... } }``

    Raises:
      401 — sin auth.
      500 — error de DB.
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    # [P2-HIST-AUDIT-A · 2026-05-09] no-store.
    _apply_no_store(response)

    from db_core import execute_sql_query
    try:
        # `event = ANY(%s)` con array de strings es la forma idiomática
        # en psycopg para evitar generar SQL dinámico con N
        # placeholders. Postgres usa el index sobre `event` si existe
        # (`idx_chunk_lesson_telemetry_event_created_at` lo cubre).
        #
        # [P2-HIST-AUDIT-D · 2026-05-09] Agregamos `event` al SELECT +
        # GROUP BY para poder splitear por tier en Python (high /
        # partial / low). Sin esto, el chip del Historial era plano
        # — un plan con 10 lecciones high se veía igual que uno con
        # 10 low (en realidad ~10× más valioso). Mantenemos el campo
        # `counts` agregado top-level para retrocompat (frontend
        # legacy que lee body.counts directo).
        rows = execute_sql_query(
            """
            SELECT meal_plan_id::text AS pid, event, COUNT(*)::int AS cnt
            FROM chunk_lesson_telemetry
            WHERE user_id = %s
              AND meal_plan_id IS NOT NULL
              AND event = ANY(%s)
            GROUP BY meal_plan_id, event
            """,
            (verified_user_id, list(_LESSON_COUNT_EVENT_WHITELIST)),
            fetch_all=True,
        ) or []

        # Split por tier según LESSON_QUALITY_TIERS (constants.py).
        # Map event → tier para lookup O(1) por row.
        from constants import LESSON_QUALITY_TIERS
        _event_to_tier = {}
        for tier, events in LESSON_QUALITY_TIERS.items():
            for ev in events:
                _event_to_tier[ev] = tier

        counts: dict = {}
        counts_by_quality: dict = {}
        for r in rows:
            pid = r.get("pid")
            if not pid:
                continue
            cnt = int(r.get("cnt") or 0)
            ev = r.get("event") or ""
            tier = _event_to_tier.get(ev)
            # Total agregado (retrocompat).
            counts[pid] = counts.get(pid, 0) + cnt
            # Split por tier — solo si el event está clasificado en
            # alguna tier. Si LESSON_QUALITY_TIERS pierde sync con
            # LESSON_COUNT_EVENT_WHITELIST, el event cae al fallback
            # `unknown` (sin romper el response — el test de drift
            # cross-archivo lo detecta).
            if pid not in counts_by_quality:
                counts_by_quality[pid] = {"high": 0, "partial": 0, "low": 0}
            if tier in counts_by_quality[pid]:
                counts_by_quality[pid][tier] += cnt
        return {"counts": counts, "counts_by_quality": counts_by_quality}
    except Exception as e:
        logger.error(f"❌ [P1-HIST-3] error en lessons-counts: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.get("/history-status-summary")
def api_plans_history_status_summary(
    response: Response,
    # [P1-AUDIT-3 · 2026-05-10] `get_verified_user_id` intencional — ver
    # nota idéntica en `/lessons-counts` (L4779). GET polling read-only
    # del Historial, exento del paywall mensual por convención.
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """[P0-AUDIT-HIST-2 · 2026-05-09] Resumen agregado de estados de
    `plan_chunk_queue` por plan del usuario, para reconciliar con
    `plan_data` flags en el Historial.

    Bug original (audit Historial 2026-05-09):
        El Historial deriva su bucket de status (`complete` /
        `partial` / `failed` / `action_required` / `unknown`) 100% desde
        `meal_plans.plan_data` (`generation_status`,
        `_user_action_required`, `_recovery_exhausted_chunks`). Pero
        ese jsonb solo se actualiza por `_escalate_unrecoverable_chunk`
        (`cron_tasks.py:7928`). Las otras 6 rutas que setean
        `status='pending_user_action'` (pausa pantry, snapshot stale,
        TZ unresolved, missing prior lessons pre-escalation,
        `cron_tasks.py:5977 / 6038 / 6114 / 10636 / 12091 / 16798`) NO
        tocan plan_data.

        Confirmado en producción al inspeccionar la DB de MealFitRD:
        existe un chunk en `pending_user_action` cuyo plan_data dice
        `generation_status='complete'` y `_user_action_required=null` →
        el Historial muestra "Completo" (o como mucho "Parcial X/Y" si
        `daysGenerated < total`) y el banner "Acción" jamás aparece.
        El usuario nunca se entera de que hay un chunk bloqueado.

    Diseño:
        - Agregación con `FILTER (WHERE status = ...)` para devolver
          un dict por plan con cinco contadores: pending_user_action,
          failed, in_flight (pending/processing/stale), completed,
          y `total` (suma defensiva).
        - Cap defensivo de 200 planes (mismo que `/history-list`).
        - Defense-in-depth: `WHERE user_id = %s` además del RLS de
          `plan_chunk_queue` (RLS+FORCE confirmado, P0 RLS hardening).
        - Excluye `meal_plan_id IS NULL` (orphans de SET NULL post
          P0-HIST-3).

    Frontend lo consume en `History.jsx::fetchHistory` como segundo
    request (paralelo al de lessons-counts y al de history-list)
    para que `getStatusInfo` pueda elevar el bucket a
    `action_required` cuando el queue tiene
    `pending_user_action_count > 0` o `failed_count > 0` aunque
    `plan_data._user_action_required` esté null. La reconciliación
    es client-side; este endpoint solo ofrece la fuente de verdad
    operativa.

    Returns:
      ``{ "summary": {
          "<plan_id_str>": {
              "pending_user_action_count": <int>,
              "failed_count": <int>,
              "failed_unreplaced_count": <int>,
              "in_flight_count": <int>,
              "completed_count": <int>,
              "total": <int>,
              "tier_breakdown": {<tier>: <count>, ...} | null
          },
          ...
        } }``

    Raises:
      401 — sin auth.
      500 — error de DB.
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    # [P2-HIST-AUDIT-A · 2026-05-09] no-store.
    _apply_no_store(response)

    from db_core import execute_sql_query
    try:
        # `FILTER (WHERE ...)` agrega counts en una sola pasada por
        # row en lugar de N self-joins. Postgres lo planifica como
        # parcial por status sobre el HashAggregate del GROUP BY.
        # `user_id` se filtra antes de agrupar (Index Cond) — la
        # cardinalidad por usuario es baja (≤200 planes), así que
        # incluso sin índice compuesto el costo es lineal en su set.
        #
        # [P1-AUDIT-HIST-6 · 2026-05-09] tier_breakdown jsonb por
        # plan: agregamos `quality_tier` con `jsonb_object_agg` en
        # un GROUP BY anidado. Solo cuenta chunks `completed` con
        # quality_tier no-NULL (los demás estados no tienen tier
        # significativo). Resultado: `{tier: count}` o NULL si el
        # plan no tiene completed-with-tier.
        rows = execute_sql_query(
            """
            SELECT
                outer_q.meal_plan_id::text AS pid,
                outer_q.pending_user_action_count,
                outer_q.failed_count,
                outer_q.failed_unreplaced_count,
                outer_q.in_flight_count,
                outer_q.completed_count,
                outer_q.total,
                qtiers.tier_breakdown
            FROM (
                SELECT
                    meal_plan_id,
                    COUNT(*) FILTER (WHERE status = 'pending_user_action')::int AS pending_user_action_count,
                    COUNT(*) FILTER (WHERE status = 'failed')::int AS failed_count,
                    -- [P0-HIST-NEW-1 · 2026-05-09] failed sin sibling completed
                    -- para misma (plan, week). Espeja la subquery en
                    -- /history-list (~línea 5763). Sirve al frontend para
                    -- el fallback legacy cuando los counters embebidos del
                    -- listado no están disponibles (deploy lag).
                    COUNT(*) FILTER (
                        WHERE status = 'failed'
                          AND NOT EXISTS (
                              SELECT 1
                              FROM plan_chunk_queue sibling
                              WHERE sibling.meal_plan_id = plan_chunk_queue.meal_plan_id
                                AND sibling.week_number = plan_chunk_queue.week_number
                                AND sibling.id != plan_chunk_queue.id
                                AND sibling.status = 'completed'
                          )
                    )::int AS failed_unreplaced_count,
                    COUNT(*) FILTER (WHERE status IN ('pending', 'processing', 'stale'))::int AS in_flight_count,
                    COUNT(*) FILTER (WHERE status = 'completed')::int AS completed_count,
                    COUNT(*)::int AS total
                FROM plan_chunk_queue
                WHERE user_id = %s
                  AND meal_plan_id IS NOT NULL
                GROUP BY meal_plan_id
            ) outer_q
            LEFT JOIN LATERAL (
                SELECT jsonb_object_agg(quality_tier, cnt) AS tier_breakdown
                FROM (
                    SELECT quality_tier, COUNT(*)::int AS cnt
                    FROM plan_chunk_queue inner_q
                    WHERE inner_q.meal_plan_id = outer_q.meal_plan_id
                      AND inner_q.user_id = %s
                      AND inner_q.status = 'completed'
                      AND inner_q.quality_tier IS NOT NULL
                    GROUP BY quality_tier
                ) t
            ) qtiers ON TRUE
            LIMIT 200
            """,
            (verified_user_id, verified_user_id),
            fetch_all=True,
        ) or []
    except Exception as e:
        logger.error(f"❌ [P0-AUDIT-HIST-2] error en history-status-summary: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

    summary = {}
    for r in rows:
        pid = r.get("pid")
        if not pid:
            continue
        summary[pid] = {
            "pending_user_action_count": int(r.get("pending_user_action_count") or 0),
            "failed_count": int(r.get("failed_count") or 0),
            # [P0-HIST-NEW-1 · 2026-05-09] failed sin sibling completed.
            "failed_unreplaced_count": int(r.get("failed_unreplaced_count") or 0),
            "in_flight_count": int(r.get("in_flight_count") or 0),
            "completed_count": int(r.get("completed_count") or 0),
            "total": int(r.get("total") or 0),
            # [P1-AUDIT-HIST-6 · 2026-05-09] Tier breakdown solo de
            # chunks completed con quality_tier no-NULL. Misma
            # convención que `chunk_tier_breakdown` en /history-list:
            # None cuando no hay tier info (vs `{}` que confundiría
            # al frontend).
            "tier_breakdown": (
                r.get("tier_breakdown")
                if isinstance(r.get("tier_breakdown"), dict)
                and r.get("tier_breakdown")
                else None
            ),
        }

    logger.info(
        "[HISTORY-STATUS-SUMMARY] user=%s plans_with_chunks=%d",
        verified_user_id, len(summary),
    )
    return {"summary": summary}


@router.get("/{plan_id}/lessons")
def api_plan_lessons_detail(
    plan_id: str,
    response: Response,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """[P2-HIST-AUDIT-2 · 2026-05-09] Detalle por-plan de lecciones del
    aprendizaje continuo (`chunk_lesson_telemetry`).

    Complementa `/lessons-counts` (que devuelve solo el conteo
    agregado por plan_id): este endpoint expande las filas individuales
    para que el modal del Historial las pueda mostrar como tab
    "Lecciones".

    Filtros (idénticos al conteo agregado para coherencia):
      - `meal_plan_id = plan_id` y `user_id = verified_user_id`
        (defense-in-depth + ownership check vía SELECT previo).
      - `event = ANY(_LESSON_COUNT_EVENT_WHITELIST)` (P1-HIST-AUDIT-5):
        4 events semánticos. Sin esta whitelist, el detalle mostraría
        eventos mecánicos (synth_schema_invalid, etc.) que el conteo
        oculta — drift entre vistas.

    Order: `created_at DESC` (la más reciente primero).
    Cap: LIMIT 200 (defensivo; un plan rara vez genera >50 lecciones).

    Returns:
      ``{ "plan_id": "<uuid>", "lessons": [
            {
              "id": "<uuid>", "event": "<str>",
              "week_number": <int>,
              "synthesized_count": <int>, "queue_count": <int>,
              "metadata": <obj|null>,
              "created_at": "<iso>"
            },
            ...
        ] }``

    Raises:
      401 — sin auth.
      400 — plan_id missing/invalid.
      404 — plan no existe o no pertenece al usuario.
      500 — error de DB.
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not plan_id or not isinstance(plan_id, str):
        raise HTTPException(status_code=400, detail="plan_id required")
    # [P2-HIST-AUDIT-A · 2026-05-09] no-store.
    _apply_no_store(response)

    from db_core import execute_sql_query
    try:
        # 1) Ownership check explícito. Sin esto, el endpoint
        #    devolvería 200 con `lessons: []` para un plan que NO
        #    pertenece al user — confuso (200 vs 404). El SELECT
        #    extra es barato (PK lookup).
        owner = execute_sql_query(
            "SELECT id FROM meal_plans WHERE id = %s AND user_id = %s",
            (plan_id, verified_user_id),
            fetch_one=True,
        )
        if not owner:
            raise HTTPException(status_code=404, detail="Plan not found")

        rows = execute_sql_query(
            """
            SELECT id::text AS id, event, week_number,
                   synthesized_count, queue_count,
                   metadata, created_at
            FROM chunk_lesson_telemetry
            WHERE meal_plan_id = %s
              AND user_id = %s
              AND event = ANY(%s)
            ORDER BY created_at DESC
            LIMIT 200
            """,
            (plan_id, verified_user_id, list(_LESSON_COUNT_EVENT_WHITELIST)),
            fetch_all=True,
        ) or []
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [P2-HIST-AUDIT-2] error en lessons detail: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

    lessons = []
    for r in rows:
        created_at = r.get("created_at")
        if hasattr(created_at, "isoformat"):
            created_at = created_at.isoformat()  # pyright: ignore[reportOptionalMemberAccess]  (guarded por hasattr)
        # [P2-HIST-NEW-2 · 2026-05-09] Coerción defensiva del jsonb
        # `metadata`. psycopg suele entregar dicts pero un row legacy
        # con metadata=None o tipo no-dict (string raw, lista) caería
        # al frontend y rompería el render del preview de keys. None
        # explícito permite al frontend distinguir "sin metadata" de
        # "metadata vacío" — el render se omite en ambos casos pero
        # el contrato es claro.
        _meta = r.get("metadata")
        if not isinstance(_meta, dict):
            _meta = None
        lessons.append({
            "id": r.get("id"),
            "event": r.get("event"),
            "week_number": r.get("week_number"),
            "synthesized_count": r.get("synthesized_count"),
            "queue_count": r.get("queue_count"),
            "metadata": _meta,
            "created_at": created_at,
        })

    return {"plan_id": plan_id, "lessons": lessons}


@router.get("/{plan_id}/coherence-history")
def api_plan_coherence_history(
    plan_id: str,
    response: Response,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """[P2-HIST-AUDIT-2 · 2026-05-09] Detalle por-plan del historial de
    ajustes de coherencia recetas↔lista de compras.

    Complementa el chip "X ajustes" de la card (que solo muestra el
    conteo agregado de entries anomalous): este endpoint devuelve la
    lista completa para el tab "Ajustes" del modal del Historial.

    A diferencia de `/lessons` (que vive en `chunk_lesson_telemetry`),
    `_shopping_coherence_block_history` vive embebido en
    `meal_plans.plan_data` (jsonb append-only, cap 20 entries —
    P3-NEW-C). Extraemos vía operador `->`.

    Returns:
      ``{ "plan_id": "<uuid>", "history": [<entry>, ...] }``

    Cada entry tiene shape variable según `action_taken`:
      - degrade/reject_minor/reject_high: criticos + magnitudes diff
      - hydration_error: bug del consumer (block_set sin action)
      - not_applicable: warn-only (block_set=False)
      - post_swap_revalidation (P2-B): observability post-swap

    El frontend filtra/ordena según necesidad; aquí devolvemos el raw
    para que el shape backend↔frontend sea simple y el detalle sea
    extensible sin coordinación.

    Raises:
      401 — sin auth.
      400 — plan_id missing/invalid.
      404 — plan no existe o no pertenece al usuario.
      500 — error de DB.
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not plan_id or not isinstance(plan_id, str):
        raise HTTPException(status_code=400, detail="plan_id required")
    # [P2-HIST-AUDIT-A · 2026-05-09] no-store.
    _apply_no_store(response)

    from db_core import execute_sql_query
    try:
        # JOIN ownership + extract en un solo SELECT. Si la fila no
        # existe O pertenece a otro user, el SELECT devuelve None
        # → 404 sin DOS-able discovery.
        row = execute_sql_query(
            """
            SELECT plan_data->'_shopping_coherence_block_history' AS history
            FROM meal_plans
            WHERE id = %s AND user_id = %s
            """,
            (plan_id, verified_user_id),
            fetch_one=True,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Plan not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [P2-HIST-AUDIT-2] error en coherence-history: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

    # `history` viene como list[dict] (jsonb cast a Python). Si está
    # ausente o es de otro tipo (corrupción), tratamos como vacío:
    # el frontend renderiza "Sin ajustes" en lugar de crash.
    history = row.get("history")
    if not isinstance(history, list):
        history = []

    return {"plan_id": plan_id, "history": history}


# [P1-HIST-LIFETIME-LESSONS · 2026-05-09] Caps defensivos de items
# devueltos por plan. El historial puede crecer mucho en planes 30-90d
# con regen frecuente; el modal del Historial solo necesita los más
# recientes para el surface (no audit completo). Si un operador quiere
# audit total, lee directo el jsonb desde DB.
_LIFETIME_HISTORY_CAP = 50
_LIFETIME_CRITICAL_CAP = 50
# Whitelist de keys del summary que el endpoint expone. Drift detection:
# si el productor (`cron_tasks.py:~20720`) añade una key, este endpoint
# la propaga vía `whitelisted_keys` SIN filtrar por nombre — la lista
# es solo para documentar el shape público en este punto en el tiempo.
_LIFETIME_SUMMARY_NUMERIC_KEYS = (
    "total_rejection_violations",
    "total_allergy_violations",
    "_lifetime_window_days",
    "_lifetime_proxy_ratio",
    "_lifetime_user_logs_count",
    "_lifetime_proxy_count",
)
_LIFETIME_SUMMARY_LIST_KEYS = (
    "top_rejection_hits",
    "top_repeated_bases",
    "top_repeated_meal_names",
    "permanent_meal_blocklist",
)

# [P0-HIST-LEARN-1 · 2026-05-09] Whitelist de keys de
# `_last_chunk_learning` que el endpoint expone. Es el SSOT de qué
# aprendió el chunk anterior y se inyecta al PRÓXIMO chunk como
# semilla del prompt — sin esto, el modal del Historial no podía
# explicar "qué pasó en el chunk N-1 que afectó al chunk N".
#
# Productor: cron_tasks.py (`_persist_last_chunk_learning` y los
# rebuilders `_rebuild_recent_chunk_lessons_from_queue` /
# `_regenerate_recent_chunk_lessons_from_plan_days`). Cada key tiene
# semántica distinta — split por tipo para que el frontend pueda
# colorear severity sin tipear cada caso:
#
#   numeric: chunk (week N), repeat_pct (0-1), ingredient_base_repeat_pct,
#            allergy_violations, rejection_violations, fatigued_violations
#   bool   : low_confidence, metrics_unavailable, rebuilt_from_queue,
#            rebuilt_from_preflight, rebuilt_from_pipeline_failure
#   str    : timestamp (ISO), rebuilt_source_status, learning_signal_strength
#   list   : repeated_meal_names, repeated_bases, allergy_hits,
#            rejected_meals_that_reappeared
#
# Coerción defensiva por tipo: si una key llega con tipo inesperado,
# cae al default (None / []). El frontend renderiza solo lo que tiene
# valor — un plan legacy sin esta key responde sin la sub-sección.
_LAST_CHUNK_LEARNING_NUMERIC_KEYS = (
    "chunk",
    "repeat_pct",
    "ingredient_base_repeat_pct",
    "allergy_violations",
    "rejection_violations",
    "fatigued_violations",
)
_LAST_CHUNK_LEARNING_BOOL_KEYS = (
    "low_confidence",
    "metrics_unavailable",
    "rebuilt_from_queue",
    "rebuilt_from_preflight",
    "rebuilt_from_pipeline_failure",
)
_LAST_CHUNK_LEARNING_STR_KEYS = (
    "timestamp",
    "rebuilt_source_status",
    "learning_signal_strength",
)
_LAST_CHUNK_LEARNING_LIST_KEYS = (
    "repeated_meal_names",
    "repeated_bases",
    "allergy_hits",
    "rejected_meals_that_reappeared",
)


@router.get("/{plan_id}/lifetime-lessons")
def api_plan_lifetime_lessons(
    plan_id: str,
    response: Response,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """[P1-HIST-LIFETIME-LESSONS · 2026-05-09] Surface del aprendizaje
    continuo lifetime para un plan archivado.

    Bug original (audit Historial 2026-05-09 · gap P1-1):
        El tab "Lecciones" del modal del Historial solo lee
        `chunk_lesson_telemetry` (P2-HIST-AUDIT-2 + whitelist
        P1-AUDIT-HIST-7 de 4 events). Eso es **telemetría sobre el
        aprendizaje** (señales mecánicas tipo "lesson_synthesized",
        "synth_propagated_to_prompt"), no el aprendizaje en sí.

        El aprendizaje real vive en `plan_data` con 3 estructuras:

          1. ``_lifetime_lessons_summary`` (dict): agregación
             (recomputed en cada merge de chunk en `cron_tasks.py:
             ~20720`). Contiene `total_rejection_violations`,
             `total_allergy_violations`, `top_rejection_hits`,
             `top_repeated_bases`, `top_repeated_meal_names`,
             `permanent_meal_blocklist` (cap 50 — meals con ≥2
             chunks repetidos), `_lifetime_proxy_ratio` (señal de
             salud: meals aprendidos via proxy vs logs reales).

          2. ``_lifetime_lessons_history`` (list[dict]): registro
             append-only por chunk. Cada entry tiene `chunk` (week
             number), `rejection_violations`, `allergy_violations`,
             `rejected_meals_that_reappeared`, `repeated_bases`,
             `repeated_meal_names`. P1-22 filtra dead-lettered en el
             read-path antes de recomputar el summary.

          3. ``_critical_lessons_permanent`` (list[dict]): subset
             inmortal — lecciones con `allergy_violations > 0` o
             `rejection_violations >= IMMORTAL_REJ` (constants.py).
             Sobreviven al rolling window (P0-7); se podan con LRU
             priorizado (P0-6) cuando exceden el hard cap.

        Para el usuario que abre un plan archivado, todo este
        aprendizaje cross-plan era invisible. Surface en el modal
        permite al usuario ver *qué aprendió Mealfit de él* en ese
        plan — diferenciador de producto que estaba enterrado en
        jsonb sin UI.

    Diseño:
        Single roundtrip que devuelve las 3 estructuras desde
        `meal_plans.plan_data` con caps defensivos:

          - history capeado a 50 entries más recientes (el array
            puede crecer en planes largos con regen frecuente; UI
            no necesita audit completo).
          - critical_permanent capeado a 50 (consistente con el cap
            del productor en `_prune_critical_lessons_with_priority`).
          - summary se devuelve completo (es agregado, no array).

        Tolerante a corrupción: cualquier key ausente o con tipo
        incorrecto cae al default (`null` para summary, `[]` para
        listas) — un plan legacy sin estas keys responde 200 con
        valores vacíos, no 500.

    Returns:
      ``{
          "plan_id": "<uuid>",
          "summary": {
              "total_rejection_violations": <int>,
              "total_allergy_violations": <int>,
              "top_rejection_hits": [<str>, ...],
              "top_repeated_bases": [<str>, ...],
              "top_repeated_meal_names": [<str>, ...],
              "permanent_meal_blocklist": [<str>, ...],
              "_lifetime_window_days": <int>,
              "_lifetime_proxy_ratio": <float>,
              "_lifetime_user_logs_count": <int>,
              "_lifetime_proxy_count": <int>
          } | null,
          "history": [<entry>, ...],
          "critical_permanent": [<entry>, ...],
          "last_chunk_learning": {
              "chunk": <int|null>,
              "timestamp": "<iso|null>",
              "repeat_pct": <float|null>,
              "ingredient_base_repeat_pct": <float|null>,
              "allergy_violations": <int|null>,
              "rejection_violations": <int|null>,
              "fatigued_violations": <int|null>,
              "low_confidence": <bool|null>,
              "metrics_unavailable": <bool|null>,
              "rebuilt_from_queue": <bool|null>,
              "rebuilt_from_preflight": <bool|null>,
              "rebuilt_from_pipeline_failure": <bool|null>,
              "rebuilt_source_status": "<str|null>",
              "learning_signal_strength": "<str|null>",
              "repeated_meal_names": [<str>, ...],
              "repeated_bases": [<str>, ...],
              "allergy_hits": [<str>, ...],
              "rejected_meals_that_reappeared": [<str>, ...]
          } | null,
          "counts": {
              "history_total": <int>,
              "history_returned": <int>,
              "critical_permanent_total": <int>,
              "critical_permanent_returned": <int>
          }
        }``

    [P0-HIST-LEARN-1 · 2026-05-09] `last_chunk_learning` es la semilla
    literal que el cron inyecta al PRÓXIMO chunk como contexto de
    aprendizaje (ver `_persist_last_chunk_learning` en cron_tasks.py).
    Diagnóstico "por qué el chunk N+1 generó X" antes requería SQL al
    jsonb — ahora visible en el modal.

    [P0-HIST-LEARN-2 · 2026-05-09] `consecutive_zero_log_chunks` es el
    counter que dispara push alarmante (≥3) + flip de generation_status
    a 'degraded_pending_engagement'. Antes invisible al user: el sistema
    aprendía sin feedback y la única señal era el push. `generation_status`
    se devuelve junto para que el frontend pueda diferenciar
    "degradado por engagement" del status canónico del plan.

    [P3-NEW-2 · 2026-05-10] Contrato cron-dependiente — cuándo
    `last_chunk_learning` y `consecutive_zero_log_chunks` están
    populated:

      Ambos campos son escritos por el worker chunk (`_chunk_worker` en
      `cron_tasks.py`) al COMPLETAR un chunk (path T1 + T2). Específico:

        - `last_chunk_learning`: persistido en `plan_data` por
          `_persist_last_chunk_learning` invocado al final del merge T1.
          Si T1 se completa pero T2 falla, el field puede quedar
          desactualizado un cycle hasta que el siguiente chunk lo
          recompute o el cron de rescate lo reconstruya desde
          `plan_chunk_queue.learning_metrics` (ver P0-3
          `_rebuild_last_chunk_learning_from_queue`).

        - `consecutive_zero_log_chunks`: incrementado en cada chunk
          completado cuando el counter de `user_logs` está en 0; reset
          a 0 cuando hay logs. Vive solo en `plan_data` (no en columna
          dedicada).

      **Cuándo esperar `null` en estos campos** (operacional, no bug):

        1. Plan nuevo sin chunks completados aún: ambos `null`.
        2. Plan en estado `failed` o `dead_letter` antes del primer
           merge T1: `last_chunk_learning` `null`; `consecutive_zero_log_chunks`
           probablemente `0` o ausente.
        3. Plan legacy pre-P0-3 (creado antes del rebuild cron):
           `last_chunk_learning` puede ser `null` aunque el plan
           tiene chunks completados — el frontend renderiza vacío.
        4. Plan pausado indefinidamente con `_pause_reason`: el
           counter no avanza hasta que el unblock del cron lo destapa.
        5. [P3-AUDIT-1 · 2026-05-10] **Seed fallido sin rebuild
           automático**: si `_seed_last_chunk_learning` agotó sus
           `_seed_attempts=3` en T1 pre-merge (graph_orchestrator.py
           ~L1139), el campo persiste `null` Y NO hay cron de retry
           automático que lo reconstruya — `_rebuild_last_chunk_learning_from_queue`
           solo dispara cuando hay `plan_chunk_queue.learning_metrics`
           ya estampado, no para casos donde el seed nunca grabó.
           UX consecuente: frontend renderiza "0 lecciones"
           indefinidamente aunque el plan tenga chunks completados.
           **Recuperación**: trigger manual vía `/regenerate-simplified`
           sobre el chunk fallado, que re-ejecuta el seed dentro del
           merge.

      **Cuándo esperar `null` y SÍ ES BUG**: plan con N ≥1 chunks
      completados sin pausa Y `last_chunk_learning` aún `null` Y
      ningún chunk con `_seed_attempts >= 3` en logs → cron de merge
      no corrió o crasheó. Síntoma a vigilar en `pipeline_metrics`
      con `node='_chunk_worker_merge_failed'` o similar. Si se
      observa drift sistemático, ampliar este endpoint con métrica
      `_missing_last_chunk_learning_count`.

    Raises:
      401 — sin auth.
      400 — plan_id missing/invalid.
      404 — plan no existe o no pertenece al usuario.
      500 — error de DB.
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not plan_id or not isinstance(plan_id, str):
        raise HTTPException(status_code=400, detail="plan_id required")
    # [P2-HIST-AUDIT-A · 2026-05-09] no-store.
    _apply_no_store(response)

    from db_core import execute_sql_query
    try:
        # JOIN ownership + extract en un solo SELECT (mismo patrón
        # que `coherence-history` arriba). Si la fila no existe O
        # pertenece a otro user, el SELECT devuelve None → 404 sin
        # DOS-able discovery.
        row = execute_sql_query(
            """
            SELECT
                plan_data->'_lifetime_lessons_summary' AS summary,
                plan_data->'_lifetime_lessons_history' AS history,
                plan_data->'_critical_lessons_permanent' AS critical_permanent,
                plan_data->'_last_chunk_learning' AS last_chunk_learning,
                NULLIF(plan_data->>'_consecutive_zero_log_chunks', '')::int
                    AS consecutive_zero_log_chunks,
                plan_data->>'generation_status' AS generation_status
            FROM meal_plans
            WHERE id = %s AND user_id = %s
            """,
            (plan_id, verified_user_id),
            fetch_one=True,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Plan not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [P1-HIST-LIFETIME-LESSONS] error en lifetime-lessons: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

    # Summary: dict whitelisted. Cualquier key con tipo inválido cae
    # al default — el frontend renderiza "Sin datos" para esa key
    # individual, no rompe el tab entero.
    summary_raw = row.get("summary")
    summary = None
    if isinstance(summary_raw, dict):
        summary = {}
        for k in _LIFETIME_SUMMARY_NUMERIC_KEYS:
            v = summary_raw.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                summary[k] = v
            else:
                summary[k] = None
        for k in _LIFETIME_SUMMARY_LIST_KEYS:
            v = summary_raw.get(k)
            summary[k] = [str(x) for x in v if x is not None] if isinstance(v, list) else []

    # History: list[dict] append-only. Devolvemos los más recientes
    # del tail (el productor en `cron_tasks.py` añade al final).
    # Cada entry pasa una sanitización mínima — strings/ints/listas
    # se preservan; otros tipos quedan como llegan (jsonb_to_python).
    history_raw = row.get("history") or []
    if not isinstance(history_raw, list):
        history_raw = []
    history_total = len(history_raw)
    history = history_raw[-_LIFETIME_HISTORY_CAP:] if history_total > 0 else []
    # Reverse para que el más reciente quede primero (UX: lectura
    # top-down con eventos nuevos arriba).
    history = list(reversed(history))

    # Critical permanent: list[dict] (lecciones inmortales). Cap
    # defensivo simétrico con el productor.
    critical_raw = row.get("critical_permanent") or []
    if not isinstance(critical_raw, list):
        critical_raw = []
    critical_total = len(critical_raw)
    critical_permanent = (
        critical_raw[:_LIFETIME_CRITICAL_CAP]
        if critical_total > 0 else []
    )

    counts = {
        "history_total": history_total,
        "history_returned": len(history),
        "critical_permanent_total": critical_total,
        "critical_permanent_returned": len(critical_permanent),
    }

    # [P0-HIST-LEARN-1 · 2026-05-09] Surface del `_last_chunk_learning`
    # — la semilla literal que el cron inyecta al PRÓXIMO chunk. Antes
    # invisible: el modal del Historial mostraba lifetime aggregates pero
    # no "qué aprendió el último bloque y qué se transmite al siguiente".
    # Sin esto, diagnosticar "el chunk N+1 repitió X" requería SQL al
    # jsonb. Coerción por tipo (numeric/bool/str/list) protege contra
    # corrupciones legacy: keys con tipo inesperado caen al default y
    # NO rompen la sub-sección entera.
    lcl_raw = row.get("last_chunk_learning")
    last_chunk_learning = None
    if isinstance(lcl_raw, dict):
        last_chunk_learning = {}
        for k in _LAST_CHUNK_LEARNING_NUMERIC_KEYS:
            v = lcl_raw.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                last_chunk_learning[k] = v
            else:
                last_chunk_learning[k] = None
        for k in _LAST_CHUNK_LEARNING_BOOL_KEYS:
            v = lcl_raw.get(k)
            last_chunk_learning[k] = v if isinstance(v, bool) else None
        for k in _LAST_CHUNK_LEARNING_STR_KEYS:
            v = lcl_raw.get(k)
            last_chunk_learning[k] = v if isinstance(v, str) and v.strip() else None
        for k in _LAST_CHUNK_LEARNING_LIST_KEYS:
            v = lcl_raw.get(k)
            last_chunk_learning[k] = (
                [str(x) for x in v if x is not None]
                if isinstance(v, list) else []
            )

    # [P0-HIST-LEARN-2 · 2026-05-09] Surface del counter
    # `_consecutive_zero_log_chunks` — chunks consecutivos que el cron
    # generó SIN logs reales del usuario (ni consumed_meals ni
    # interacciones que cuenten como signal). Antes invisible: el counter
    # vivía solo en plan_data, dispara push notification con copy
    # alarmante a partir de ≥3 ("Tu plan se está generando sin tu
    # feedback") + flip de `generation_status` a `degraded_pending_engagement`
    # (cron_tasks.py:17487-17488) — y aún así el modal del Historial
    # NO lo mostraba. Si todos los rolling_refills corrieron sin signal,
    # el user no tenía forma de detectarlo desde la UI.
    #
    # Coerción defensiva: la key en plan_data puede ser int (camino
    # canónico) o string (legacy). NULLIF + ::int en el SELECT cubre
    # ambos. Si la conversión falla (string no-numérico), psycopg
    # lanza — preferible explícito a silenciar bug del writer.
    czl = row.get("consecutive_zero_log_chunks")
    if not isinstance(czl, int):
        czl = None
    gen_status = row.get("generation_status")
    if not isinstance(gen_status, str) or not gen_status.strip():
        gen_status = None

    return {
        "plan_id": plan_id,
        "summary": summary,
        "history": history,
        "critical_permanent": critical_permanent,
        "last_chunk_learning": last_chunk_learning,
        "consecutive_zero_log_chunks": czl,
        "generation_status": gen_status,
        "counts": counts,
    }


@router.get("/{plan_id}/chunk-metrics")
def api_plan_chunk_metrics(
    plan_id: str,
    response: Response,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """[P2-HIST-AUDIT-10 · 2026-05-09] Detalle por-chunk de métricas
    operacionales y `learning_metrics` para el modal del Historial.

    Bug original (audit Historial 2026-05-09):
        El Historial mostraba el bucket de status (P0/P1) y el
        tier_breakdown agregado (P1-AUDIT-HIST-6) pero NO exponía
        las métricas ricas por-chunk:
          - `learning_metrics` (jsonb con recovery_attempts,
            escalation_reason, shuffle_*, etc. — populated por
            `_escalate_unrecoverable_chunk` (recovery_attempts/
            escalation_reason) y el worker T1/T2. [G14-DOC-DRIFT ·
            2026-05-29] synth_quality_score/synthesized_count/queue_count
            NO viven aquí: synthesized_count/queue_count vienen de
            chunk_lesson_telemetry y synth_quality_score no se computa —
            G8 las removió del catálogo `_LM_DISPLAY_GROUPS`).
          - `lag_seconds_at_pickup` / `effective_lag_seconds_at_pickup`
            (cuánto se atrasó el sistema en agarrar el chunk).
          - `escalated_at` (timestamp de escalación si hubo).
          - `learning_persisted_at` (cuándo se commiteó learning).
          - Stats persistidas en `plan_chunk_metrics`:
            `duration_ms`, `was_degraded`, `retries`, `lag_seconds`,
            `learning_repeat_pct`, `rejection_violations`,
            `allergy_violations`, `pantry_snapshot_age_hours`,
            `error_message`.
        Para un usuario que quería diagnosticar por qué su plan se
        generó "raro", la única vista era la del Dashboard del plan
        ACTIVO (chunk-status / admin endpoints). Para planes
        archivados, el detalle era invisible.

    Diseño:
        LEFT JOIN entre `plan_chunk_queue` (estado operacional vivo)
        y `plan_chunk_metrics` (snapshot de stats al completar). Ambas
        tablas tienen RLS habilitado; el WHERE explícito por user_id
        es defense-in-depth.

        Cap LIMIT 50 (un plan típico tiene ≤30 chunks; cap defensivo).

    Ownership check vía SELECT inicial — devuelve 404 si plan no
    existe O no pertenece al usuario, sin DOS-able discovery.

    Returns:
      ``{ "plan_id": "<uuid>",
          "chunks": [
            {
              "chunk_id": "<uuid>",
              "week_number": <int>,
              "days_offset": <int>,
              "days_count": <int>,
              "status": "<str>",
              "quality_tier": "<str|null>",
              "attempts": <int>,
              "chunk_kind": "<str|null>",
              "lag_seconds_at_pickup": <int|null>,
              "effective_lag_seconds_at_pickup": <int|null>,
              "escalated_at": "<iso|null>",
              "learning_persisted_at": "<iso|null>",
              "dead_letter_reason": "<str|null>",
              "dead_lettered_at": "<iso|null>",
              "created_at": "<iso>",
              "updated_at": "<iso>",
              "learning_metrics": <obj|null>,
              "metrics": {
                "duration_ms": <int|null>,
                "was_degraded": <bool|null>,
                "retries": <int|null>,
                "lag_seconds": <int|null>,
                "learning_repeat_pct": <float|null>,
                "rejection_violations": <int|null>,
                "allergy_violations": <int|null>,
                "pantry_snapshot_age_hours": <float|null>,
                "error_message": "<str|null>",
                "metrics_created_at": "<iso|null>"
              } | null,
              "deferrals_count": <int>,
              "deferral_reasons": [<str>, ...] | null
            },
            ...
          ],
          "total_count": <int>,
          "limit": <int>
        }``

    [P1-HIST-NEW-4 · 2026-05-09] `total_count` (COUNT separado, sin
    LIMIT) y `limit` (constante actual) permiten al frontend
    renderizar "Mostrando X de N" cuando hay truncado. Antes el
    response no comunicaba cuántos chunks reales tiene el plan.

    [P1-HIST-NEW-6 · 2026-05-09] `deferrals_count` y `deferral_reasons`
    surface telemetría de `chunk_deferrals` (cada vez que un gate del
    pipeline LangGraph difirió este chunk: temporal_gate,
    learning_zero_logs, missing_prior_lessons, etc.). Antes solo
    visible vía endpoint admin.

    Raises:
      401 — sin auth.
      400 — plan_id missing/invalid.
      404 — plan no existe o no pertenece al usuario.
      500 — error de DB.
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not plan_id or not isinstance(plan_id, str):
        raise HTTPException(status_code=400, detail="plan_id required")
    # [P2-HIST-AUDIT-A · 2026-05-09] no-store.
    _apply_no_store(response)

    from db_core import execute_sql_query
    try:
        # Ownership check (PK lookup, barato).
        owner = execute_sql_query(
            "SELECT id FROM meal_plans WHERE id = %s AND user_id = %s",
            (plan_id, verified_user_id),
            fetch_one=True,
        )
        if not owner:
            raise HTTPException(status_code=404, detail="Plan not found")

        rows = execute_sql_query(
            """
            SELECT
                q.id::text AS chunk_id,
                q.week_number,
                q.days_offset,
                q.days_count,
                q.status,
                q.quality_tier,
                q.attempts,
                q.chunk_kind,
                q.lag_seconds_at_pickup,
                q.effective_lag_seconds_at_pickup,
                q.escalated_at,
                q.learning_persisted_at,
                q.dead_letter_reason,
                q.dead_lettered_at,
                q.created_at AS chunk_created_at,
                q.updated_at AS chunk_updated_at,
                q.learning_metrics,
                -- [P2-HIST-AUDIT-B · 2026-05-09] expected_preemption_seconds
                -- es el SLA esperado al pickup (derivado de chunk_kind +
                -- carga del worker pool). reservation_status indica si la
                -- reserva del worker se confirmó o cayó en fallback (ok |
                -- fallback). Útiles para diagnosticar contención: chunk
                -- atascado con lag >> expected_preemption suele ser
                -- worker pool sobrecargado o lock advisory heredado.
                q.expected_preemption_seconds,
                q.reservation_status,
                m.duration_ms,
                m.was_degraded,
                m.retries,
                m.lag_seconds AS metrics_lag_seconds,
                m.learning_repeat_pct,
                m.rejection_violations,
                m.allergy_violations,
                m.pantry_snapshot_age_hours,
                m.error_message,
                -- [P2-HIST-AUDIT-E · 2026-05-09] is_rolling_refill cross-
                -- check con q.chunk_kind. El productor de plan_chunk_metrics
                -- escribe el bool al completar (P2-NEW-G) — debe
                -- coincidir con `chunk_kind = 'rolling_refill'`. Drift
                -- entre los dos indicaría bug del writer (e.g. el chunk
                -- transicionó de kind durante recovery).
                m.is_rolling_refill,
                m.created_at AS metrics_created_at,
                -- [P2-HIST-AUDIT-F · 2026-05-09] Detección de lock
                -- zombi del usuario. `chunk_user_locks` es PK por
                -- user_id (un solo lock por usuario en cualquier
                -- momento — invariante del worker pool). Si el lock
                -- está adquirido por OTRO chunk_id (`locked_by_chunk_id
                -- IS DISTINCT FROM q.id`) Y el heartbeat es fresco
                -- (<5min), este chunk está esperando — útil para
                -- diagnosticar contención cuando un chunk lleva
                -- mucho rato en `pending` sin avanzar.
                --
                -- LEFT JOIN a chunk_user_locks por user_id. Si no hay
                -- lock activo del usuario, columnas LATERAL llegan NULL
                -- (sin lock zombi).
                CASE
                    WHEN ul.locked_by_chunk_id IS NOT NULL
                      AND ul.locked_by_chunk_id != q.id
                      AND ul.heartbeat_at > NOW() - INTERVAL '5 minutes'
                    THEN ul.locked_by_chunk_id::text
                    ELSE NULL
                END AS blocking_lock_chunk_id,
                CASE
                    WHEN ul.locked_by_chunk_id IS NOT NULL
                      AND ul.locked_by_chunk_id != q.id
                      AND ul.heartbeat_at > NOW() - INTERVAL '5 minutes'
                    THEN EXTRACT(EPOCH FROM (NOW() - ul.locked_at))::int
                    ELSE NULL
                END AS blocking_lock_age_seconds,
                -- [P1-HIST-NEW-6 · 2026-05-09] Telemetría de
                -- chunk_deferrals (cada vez que temporal_gate u otro
                -- gate del pipeline LangGraph difirió este chunk).
                -- chunk_deferrals NO tiene FK a plan_chunk_queue.id —
                -- joineamos por (meal_plan_id, week_number), que es
                -- estable porque el unique index parcial
                -- ux_plan_chunk_queue_live_week garantiza ≤1 chunk
                -- vivo por (plan, week). Para chunks completed el join
                -- puede traer deferrals de un re-enqueue posterior con
                -- mismo week_number — comportamiento aceptable: el
                -- chip muestra "este slot semanal sufrió N deferrals
                -- en su lifetime", incluso si distribuídos entre dos
                -- intentos. La cap LIMIT 5 en reasons evita inflar el
                -- payload cuando un cron loopea sobre el gate.
                COALESCE(deferrals.deferrals_count, 0) AS deferrals_count,
                deferrals.deferral_reasons AS deferral_reasons
            FROM plan_chunk_queue q
            LEFT JOIN plan_chunk_metrics m ON m.chunk_id = q.id
            -- [P2-HIST-AUDIT-F · 2026-05-09] Lock activo del usuario.
            -- chunk_user_locks tiene PK user_id (1:1) — el LEFT JOIN
            -- es 1:0..1.
            LEFT JOIN chunk_user_locks ul ON ul.user_id = q.user_id
            -- [P1-HIST-NEW-6 · 2026-05-09] LATERAL count + reasons
            -- DISTINCT — no JOIN directo porque queremos agregar
            -- (count + array) y devolver una sola fila aún cuando no
            -- hay deferrals (LEFT JOIN LATERAL preserva el chunk con
            -- NULL en deferrals_count → COALESCE 0). El índice
            -- `idx_chunk_deferrals_plan_week` (migrations) cubre el
            -- WHERE; reasons se naturalizan a un set pequeño porque
            -- el conjunto de reason codes está enumerado en el
            -- código del temporal_gate (≤10 posibles).
            LEFT JOIN LATERAL (
                SELECT COUNT(*)::int AS deferrals_count,
                       array_agg(DISTINCT reason ORDER BY reason)
                           FILTER (WHERE reason IS NOT NULL)
                           AS deferral_reasons
                FROM chunk_deferrals
                WHERE meal_plan_id = q.meal_plan_id
                  AND week_number = q.week_number
                  AND user_id = q.user_id
            ) deferrals ON TRUE
            WHERE q.meal_plan_id = %s
              AND q.user_id = %s
            ORDER BY q.week_number ASC NULLS LAST,
                     q.days_offset ASC NULLS LAST,
                     q.created_at ASC
            LIMIT 50
            """,
            (plan_id, verified_user_id),
            fetch_all=True,
        ) or []

        # [P1-HIST-NEW-4 · 2026-05-09] COUNT total separado para que el
        # frontend pueda señalizar truncado cuando supera el LIMIT 50.
        # Antes el endpoint no comunicaba si había chunks ocultos —
        # planes con 50+ rows (extreme: tier ultra 90d con swap-meal
        # re-enqueues por week que dejan completed+failed coexistentes
        # tras P0-HIST-NEW-1) renderizaban silently truncados. Ahora
        # el response incluye `total_count` y el frontend muestra
        # "Mostrando 50 de N" cuando hay diff.
        #
        # Consulta separada (no `COUNT(*) OVER ()` window function en
        # el SELECT principal) porque la window agregaría el conteo
        # AL row del LIMIT — para responses paginados eso multiplica
        # I/O sin que las filas excluidas sean leídas. El count
        # explícito sí escanea la partición pero solo con un `count(*)`
        # — más barato que arrastrar el conteo en cada row del LEFT
        # JOIN del SELECT principal.
        count_row = execute_sql_query(
            """
            SELECT COUNT(*)::int AS total_count
            FROM plan_chunk_queue
            WHERE meal_plan_id = %s
              AND user_id = %s
            """,
            (plan_id, verified_user_id),
            fetch_one=True,
        )
        # Defensivo: si el mock/driver devuelve algo distinto a dict
        # (lista, None, etc.), caemos al len(rows) como fallback —
        # peor caso, el frontend no muestra el notice de truncado pero
        # el response no rompe. Un test que no mockee el COUNT
        # explícitamente cae a esta rama.
        if isinstance(count_row, dict):
            total_count = int(count_row.get("total_count") or 0)
        else:
            total_count = len(rows)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [P2-HIST-AUDIT-10] error en chunk-metrics: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

    def _iso(value):
        if value is None:
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value

    chunks = []
    for r in rows:
        # `metrics` se setea solo si plan_chunk_metrics tenía un row
        # (chunk completed con stats commiteados). Si todos los campos
        # de m.* son NULL (LEFT JOIN no encontró match), devolvemos
        # `metrics: null` para que el frontend distinga "no commit"
        # de "stats vacíos".
        _has_metrics = any(
            r.get(k) is not None for k in (
                "duration_ms", "was_degraded", "retries",
                "metrics_lag_seconds", "learning_repeat_pct",
                "rejection_violations", "allergy_violations",
                "pantry_snapshot_age_hours", "error_message",
                # [P2-HIST-AUDIT-E · 2026-05-09] is_rolling_refill
                # también cuenta como señal de "metrics commiteados".
                "is_rolling_refill",
                "metrics_created_at",
            )
        )
        metrics_obj = None
        if _has_metrics:
            metrics_obj = {
                "duration_ms": r.get("duration_ms"),
                "was_degraded": r.get("was_degraded"),
                "retries": r.get("retries"),
                "lag_seconds": r.get("metrics_lag_seconds"),
                "learning_repeat_pct": (
                    float(r.get("learning_repeat_pct"))  # pyright: ignore[reportArgumentType]  (guarded por is not None abajo)
                    if r.get("learning_repeat_pct") is not None
                    else None
                ),
                "rejection_violations": r.get("rejection_violations"),
                "allergy_violations": r.get("allergy_violations"),
                "pantry_snapshot_age_hours": (
                    float(r.get("pantry_snapshot_age_hours"))  # pyright: ignore[reportArgumentType]  (guarded por is not None abajo)
                    if r.get("pantry_snapshot_age_hours") is not None
                    else None
                ),
                "error_message": r.get("error_message"),
                # [P2-HIST-AUDIT-E · 2026-05-09] is_rolling_refill
                # del snapshot final de plan_chunk_metrics. Ver el
                # cross-check más abajo en el dict del chunk.
                "is_rolling_refill": r.get("is_rolling_refill"),
                "metrics_created_at": _iso(r.get("metrics_created_at")),
            }

        # [P2-HIST-AUDIT-E · 2026-05-09] Cross-check de coherencia
        # entre `q.chunk_kind = 'rolling_refill'` y
        # `m.is_rolling_refill = TRUE`. Si ambas fuentes disienten,
        # el chunk pasó por recovery o el writer del snapshot tuvo
        # bug — el frontend lo señaliza como warn ("kind drift") en
        # el render. None cuando no hay metrics commiteados (sin
        # is_rolling_refill que comparar).
        _kind_is_rolling = (r.get("chunk_kind") == "rolling_refill")
        _metrics_is_rolling = r.get("is_rolling_refill")
        _kind_drift = None
        if _metrics_is_rolling is not None:
            _kind_drift = (bool(_metrics_is_rolling) != _kind_is_rolling)

        chunks.append({
            "chunk_id": r.get("chunk_id"),
            "week_number": r.get("week_number"),
            "days_offset": r.get("days_offset"),
            "days_count": r.get("days_count"),
            "status": r.get("status"),
            "quality_tier": r.get("quality_tier"),
            "attempts": r.get("attempts"),
            "chunk_kind": r.get("chunk_kind"),
            "lag_seconds_at_pickup": r.get("lag_seconds_at_pickup"),
            "effective_lag_seconds_at_pickup": r.get("effective_lag_seconds_at_pickup"),
            # [P2-HIST-AUDIT-B · 2026-05-09] Expected preemption +
            # reservation status. expected_preemption_seconds = SLA
            # esperado al pickup (None para chunks sin reserva
            # explícita). reservation_status canónico: 'ok' | 'fallback'.
            "expected_preemption_seconds": r.get("expected_preemption_seconds"),
            "reservation_status": r.get("reservation_status"),
            "escalated_at": _iso(r.get("escalated_at")),
            "learning_persisted_at": _iso(r.get("learning_persisted_at")),
            "dead_letter_reason": r.get("dead_letter_reason"),
            "dead_lettered_at": _iso(r.get("dead_lettered_at")),
            "created_at": _iso(r.get("chunk_created_at")),
            "updated_at": _iso(r.get("chunk_updated_at")),
            # `learning_metrics` viene como dict (jsonb) o None.
            "learning_metrics": (
                r.get("learning_metrics")
                if isinstance(r.get("learning_metrics"), dict)
                else None
            ),
            "metrics": metrics_obj,
            # [P2-HIST-AUDIT-E · 2026-05-09] Resultado del cross-check.
            # `True` => drift entre chunk_kind (queue, vivo) y
            # is_rolling_refill (metrics, snapshot al completar).
            # `False` => coherente. `None` => sin métricas commiteadas.
            "is_rolling_refill_drift": _kind_drift,
            # [P2-HIST-AUDIT-F · 2026-05-09] Lock zombi del usuario.
            # Si OTRO chunk_id tiene el lock con heartbeat <5min,
            # este chunk está bloqueado por contención. None cuando
            # no hay lock activo o el lock es del mismo chunk (path
            # normal: el chunk procesando legítimamente).
            "blocking_lock_chunk_id": r.get("blocking_lock_chunk_id"),
            "blocking_lock_age_seconds": r.get("blocking_lock_age_seconds"),
            # [P1-HIST-NEW-6 · 2026-05-09] Telemetría chunk_deferrals.
            # `deferrals_count` siempre presente (COALESCE 0 en SELECT).
            # `deferral_reasons` es lista DISTINCT del enum del gate
            # (max ~10 códigos: temporal_gate, learning_zero_logs,
            # missing_prior_lessons, etc.) o None cuando no hay
            # deferrals — el frontend distingue None de [] como
            # "sin info" vs "lista vacía explícita".
            "deferrals_count": int(r.get("deferrals_count") or 0),
            "deferral_reasons": (
                [str(x) for x in r.get("deferral_reasons") if x]  # pyright: ignore[reportOptionalIterable]  (guarded por isinstance list abajo)
                if isinstance(r.get("deferral_reasons"), list)
                and r.get("deferral_reasons")
                else None
            ),
        })

    # [P1-HIST-NEW-4 · 2026-05-09] `total_count` permite al frontend
    # señalizar truncado. Invariante: total_count >= len(chunks). Si
    # diverge (total_count < len(chunks)), bug del COUNT — mantenemos
    # el response sin coercer porque es una situación que el dev debe
    # diagnosticar, no un fallback silente.
    return {
        "plan_id": plan_id,
        "chunks": chunks,
        "total_count": total_count,
        "limit": 50,
    }


@router.patch("/{plan_id}/name")
def api_rename_plan(
    plan_id: str,
    data: dict = Body(...),
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """[P1-HIST-5 · 2026-05-09] Renombrado atómico: actualiza la columna
    top-level `name` Y el `plan_data->>name` (jsonb) en un solo UPDATE.

    Reemplaza el patrón legacy `History.jsx::handleEditSave` que
    hacía ``supabase.from('meal_plans').update({ name: trimmed })``
    directo. Ese flujo dejaba `plan_data.name` con el valor viejo, y
    cualquier flujo posterior que copiara `plan_data` (restore desde
    Historial pre-P0-HIST-1, swap, shift_plan que serializa
    plan_data) propagaba el nombre stale a otro contexto. P1-HIST-5
    cierra ese drift en el origen: las dos representaciones del
    nombre se mueven juntas o no se mueven.

    El UPDATE usa `jsonb_set(plan_data, '{name}', to_jsonb(?::text), true)`
    (cuarto arg `create_missing=true`) para que planes legacy sin la
    key `name` en plan_data la ganen. `meal_plans.plan_data` es
    `NOT NULL` (verificado en introspección), así que jsonb_set
    nunca recibe NULL como input.

    Body:
      ``{ "name": "<string no-vacío>" }``
    Returns:
      ``{ "success": true, "name": "<trimmed>" }``

    Raises:
      401 — sin auth.
      400 — name faltante o no-string o vacío post-trim.
      404 — plan no existe o no pertenece al usuario.
      500 — error de DB.
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not plan_id or not isinstance(plan_id, str):
        raise HTTPException(status_code=400, detail="plan_id required")

    new_name = (data or {}).get("name")
    if not isinstance(new_name, str):
        raise HTTPException(status_code=400, detail="name must be a string")
    new_name = new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="name must be non-empty")
    # Cap defensivo. La columna `name` es text sin length limit en el
    # schema, pero un nombre de 10K caracteres es síntoma de bug
    # cliente, no caso legítimo. 200 cubre cualquier nombre razonable
    # ("Plan Sintético 30 días para Juan Pérez con Restricciones X" ~80).
    if len(new_name) > 200:
        raise HTTPException(status_code=400, detail="name too long (max 200 chars)")

    # [P2-HIST-AUDIT-6 · 2026-05-09] Rechazar control chars (0x00-0x1F
    # + DEL 0x7F). Casos cubiertos:
    #   - NUL byte (\x00): Postgres jsonb lanza
    #     `unsupported Unicode escape sequence` al hacer
    #     `to_jsonb('foo\x00bar'::text)` → ROLLBACK del UPDATE.
    #   - CR/LF/TAB y otros 0x01-0x1F: rompen el render del UI
    #     (la card del Historial corta el nombre a una línea, pero
    #     la primera línea queda con el control char invisible —
    #     copy/paste a otro contexto puede arrastrarlo).
    #   - DEL (0x7F): legacy unicode artifact, no es printable.
    # Permitimos espacios estándar (0x20+) y todos los unicodes
    # printables (caracteres no-ASCII como ñ, á, emoji 🍳 son OK).
    # Whitespace común al borde ya fue trimmed; control chars
    # interiores son siempre input malicioso o bug cliente.
    for ch in new_name:
        codepoint = ord(ch)
        if codepoint < 0x20 or codepoint == 0x7F:
            raise HTTPException(
                status_code=400,
                detail="name contains invalid control characters",
            )

    # [P1-HIST-AUDIT-2 · 2026-05-09] Sello de `_plan_modified_at` para
    # mantener SSOT con el sort del Historial. `History.jsx::_effectiveModifiedAt`
    # ordena por max(created_at, _plan_modified_at); el SELECT target del
    # restore (P1-HIST-AUDIT-1) hace lo mismo en SQL. Si rename NO sella
    # `_plan_modified_at`, el plan recién renombrado NO sube en el listado
    # (la card optimistic-updated visualmente vuelve abajo en el próximo
    # fetchHistory). Sello aquí cierra el drift en el origen.
    #
    # Coherente con los otros ~6 paths del backend que sellan
    # `_plan_modified_at` en cada mutación de plan_data:
    #   - cron_tasks.py:284-377 (post-swap learning persist).
    #   - cron_tasks.py:13337 (chunk merge).
    #   - cron_tasks.py:16636, :16742 (rolling refill).
    #   - cron_tasks.py:20278, :20759 (revisión simplified).
    #   - routers/plans.py:3661 (api_restore_plan enriched_pd).
    _modified_at_iso = datetime.now(timezone.utc).isoformat()

    # [P1-HIST-AUDIT-7 · 2026-05-09] Transacción explícita + advisory
    # lock per-user para serializar con restore/delete del mismo user.
    # Antes este endpoint usaba `execute_sql_write` (transacción
    # implícita single-statement). Reescribimos a `connection_pool +
    # conn.transaction()` para tomar `pg_advisory_xact_lock` antes del
    # UPDATE — coherente con los otros dos mutators.
    from db_core import connection_pool
    import psycopg
    from psycopg.rows import dict_row
    try:
        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cur:
                    from db_plans import acquire_user_history_advisory_lock
                    acquire_user_history_advisory_lock(cur, verified_user_id)

                    # Atomic rename + ownership check vía RETURNING.
                    # Si la fila no existe O pertenece a otro user,
                    # RETURNING devuelve [] y respondemos 404 sin
                    # DOS-able discovery.
                    #
                    # `jsonb_set` anidado: la capa interior actualiza
                    # `name`, la exterior actualiza `_plan_modified_at`.
                    # Ambas con `create_missing=true` (4º arg) para que
                    # planes legacy sin alguna de las keys la ganen.
                    # `meal_plans.plan_data` es NOT NULL (verificado en
                    # introspección) → jsonb_set nunca recibe NULL como
                    # input base.
                    cur.execute(
                        """
                        UPDATE meal_plans
                        SET name = %s,
                            plan_data = jsonb_set(
                                jsonb_set(
                                    plan_data, '{name}', to_jsonb(%s::text), true
                                ),
                                '{_plan_modified_at}', to_jsonb(%s::text), true
                            )
                        WHERE id = %s AND user_id = %s
                        RETURNING id
                        """,
                        (new_name, new_name, _modified_at_iso, plan_id, verified_user_id),
                    )
                    result = cur.fetchall()
                    if not result:
                        raise HTTPException(status_code=404, detail="Plan not found")

        logger.info(
            "[P1-HIST-5] rename plan: user=%s plan=%s",
            verified_user_id, plan_id,
        )
        return {"success": True, "name": new_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [P1-HIST-5] error en rename: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.get("/history-list")
def api_plans_history_list(
    response: Response,
    # [P1-AUDIT-3 · 2026-05-10] `get_verified_user_id` intencional — ver
    # nota idéntica en `/lessons-counts` (L4779). GET polling read-only
    # del Historial, exento del paywall mensual por convención.
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
):
    """[P1-HIST-AUDIT-4 · 2026-05-09] Listado del Historial con projection
    mínima — reemplaza el ``select('*')`` del frontend que arrastraba el
    ``plan_data`` jsonb completo (típicamente 30-80KB por plan).

    Bug original (audit historial 2026-05-08):
        ``History.jsx::fetchHistory`` hacía
        ``supabase.from('meal_plans').select('*')``. Para un usuario
        tier ultra con 50+ planes archivados, eso descarga 2-5MB en
        cada apertura del Historial — la mayoría como
        ``plan_data._lifetime_lessons_history``,
        ``plan_data._recent_chunk_lessons``, ``pipeline_snapshot`` y
        otros internals que la card NO consume. El modal sí necesita
        ``days/meals``, pero solo del plan que el usuario abra; no
        upfront para todos.

    Diseño:
        - Projection vía operadores jsonb (``->``, ``->>``,
          ``jsonb_array_length``) → Postgres extrae solo los keys que
          la card del Historial consume; el blob ``plan_data`` no
          viaja al backend Python ni al cliente.
        - Sort SSOT con ``History.jsx::_effectiveModifiedAt`` y con
          ``api_restore_plan`` (P1-HIST-AUDIT-1).
        - Filter ``name IS NOT NULL`` espeja la convención post-
          P2-HIST-1 del frontend (filas sin name son garbage no-
          actionable).
        - ``coherence_adjusts_count`` calculado server-side (anomalous
          actions: degrade/reject_minor/reject_high/hydration_error)
          espeja ``History.jsx::getCoherenceAdjustsCount`` para que el
          chip aparezca con el mismo conteo.
        - Cap defensivo de 200 filas — el tier ultra actual permite
          ~100, así que 200 es 2× margen sin bandwidth ilimitado.

    Lazy-load del modal:
        Cuando el usuario abre una card, el frontend hace una
        request adicional a Supabase RPC/select por ``id`` para traer
        ``plan_data.days`` (necesario para el menú del modal). Eso
        concentra el bandwidth pesado en el plan que sí se mira, no
        en la lista upfront.

    Returns:
        ``{ "plans": [
            {
              "id": "<uuid>", "name": "...", "created_at": "<iso>",
              "calories": <int|null>, "macros": <obj|null>,
              "plan_modified_at": "<iso>|null",
              "generation_status": "...|null",
              "total_days_requested": <int|null>,
              "days_generated": <int>,
              "user_action_required": <obj|null>,
              "recovery_exhausted_count": <int>,
              "user_forced_simplified_weeks": <obj|null>,
              "shift_days_accumulated": <int|null>,
              "consecutive_zero_log_chunks": <int|null>,
              "grocery_start_date": "<iso|null>",
              "cycle_start_date": "<iso|null>",
              "coherence_adjusts_count": <int>,
              "coherence_last_hypotheses": [<str>, ...] (max 5),
              "preview_meals": [{"name", "meal"}, ...] (max 4),
              "goal": "<str|null>",
              "diet_preference": "<str|null>",
              "allergies": [<str>, ...],
              "chunk_pending_user_action_count": <int>,
              "chunk_failed_count": <int>,
              "chunk_failed_unreplaced_count": <int>,
              "chunk_in_flight_count": <int>,
              "chunk_scheduled_count": <int>,
              "chunk_running_now_count": <int>,
              "chunk_completed_count": <int>,
              "chunk_tier_breakdown": {<tier>: <count>, ...} | null,
              "chunk_pantry_degraded_count": <int>,
              "chunk_pantry_degraded_reasons": [<str>, ...] | null,
              "primary_action_reason": <str|null>
            },
            ...
        ] }``

    [P1-HIST-PANTRY-DEGRADED · 2026-05-09]
        ``chunk_pantry_degraded_count`` cuenta chunks cuyo
        ``learning_metrics->>'pantry_degraded_reason'`` está poblado
        (typical values: ``stale_snapshot``, ``empty_pantry_proxy``,
        ``inventory_unreachable``). Permite al frontend renderizar
        un chip retroactivo "Pantry degradada" en la card del
        Historial cuando el plan se generó con señal de pantry
        comprometida — diferenciador de calidad enterrado en el
        jsonb hasta ahora.

        ``chunk_pantry_degraded_reasons`` es la lista DISTINCT de
        reasons agregadas via ``array_agg(DISTINCT ...) FILTER``;
        sirve para el tooltip del chip (lista las causas reales
        sin duplicar). NULL si count = 0.

    Raises:
        401 — sin auth.
        500 — error de DB.
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # [P1-HIST-AUDIT-4-FOLLOWUP · 2026-05-09 → P3-A · 2026-05-10]
    # Cache-Control no-store evita que el browser sirva una respuesta
    # cacheada después de operaciones que mutan el set de planes
    # (restore, delete, rename, nueva generación). Sin esto, un usuario
    # que abrió /history antes de generar su primer plan veía empty
    # state cacheado incluso después de generar.
    #
    # P3-A · 2026-05-10: migrado al helper SSOT `_apply_no_store`
    # (definido en este mismo módulo línea 3706, P2-HIST-AUDIT-A). Antes
    # estos 2 headers vivían inline; con el helper, cualquier cambio
    # futuro al contrato de "no-store del Historial" se aplica desde un
    # único site. El test `test_p3_a_apply_no_store_helper_usage` previene
    # que vuelvan headers inline a este endpoint o a sus pares derivados.
    _apply_no_store(response)

    from db_core import execute_sql_query
    try:
        # Sort idéntico al SSOT de api_restore_plan:3623+ y al helper
        # JS `_effectiveModifiedAt`. Tie-breaker secundario por
        # created_at DESC para determinismo.
        #
        # [P1-AUDIT-HIST-4 · 2026-05-09] LEFT JOIN agregado a
        # `plan_chunk_queue` para integrar los counters del queue
        # (pending_user_action / failed / in_flight) en el mismo
        # response. Antes el frontend hacía DOS roundtrips (este
        # endpoint + `/history-status-summary` de P0-AUDIT-HIST-2)
        # y reconciliaba client-side; con el JOIN, los counters
        # llegan por plan en una sola request, eliminando:
        #   - Roundtrip extra al cargar la página.
        #   - Race condition entre los dos endpoints (un restore/
        #     delete entre las 2 requests podía dejar el bucket
        #     desincronizado entre el listado y el summary).
        # El endpoint `/history-status-summary` se preserva para
        # consumidores externos (admin tools, monitoring).
        #
        # Subquery pre-filtrada por `user_id = %s` y `meal_plan_id
        # IS NOT NULL` para que la GROUP BY agregue solo chunks del
        # usuario actual — Postgres usa el FK index sobre meal_plan_id
        # más el filtro user_id (defense-in-depth + RLS). El
        # `LEFT JOIN` garantiza que planes sin chunks aparecen con
        # counters = 0 (`COALESCE` arriba).
        rows = execute_sql_query(
            """
            SELECT
                mp.id::text AS id,
                mp.name,
                mp.created_at,
                mp.calories,
                mp.macros,
                mp.plan_data->>'_plan_modified_at' AS plan_modified_at,
                mp.plan_data->>'generation_status' AS generation_status,
                COALESCE(
                    NULLIF(mp.plan_data->>'total_days_requested', '')::int,
                    NULLIF(mp.plan_data->>'totalDays', '')::int
                ) AS total_days_requested,
                jsonb_array_length(
                    COALESCE(mp.plan_data->'days', '[]'::jsonb)
                ) AS days_generated,
                mp.plan_data->'_user_action_required' AS user_action_required,
                jsonb_array_length(
                    COALESCE(mp.plan_data->'_recovery_exhausted_chunks', '[]'::jsonb)
                ) AS recovery_exhausted_count,
                mp.plan_data->'_user_forced_simplified_weeks' AS user_forced_simplified_weeks,
                -- [P2-HIST-AUDIT-C · 2026-05-09] Días acumulados de shift_plan
                -- (TZ resync, rollover por inventario, etc). Útil como tag
                -- diagnóstico cuando un plan generado para semana X aparece
                -- corrido N días — el usuario puede entender que NO es bug
                -- visual, fue un ajuste deliberado del backend.
                NULLIF(mp.plan_data->>'_shift_days_accumulated', '')::int
                    AS shift_days_accumulated,
                -- [P0-HIST-LEARN-2 · 2026-05-09] Counter de chunks
                -- consecutivos generados sin feedback del usuario. ≥3
                -- dispara push notification + flip de generation_status
                -- a 'degraded_pending_engagement' (cron_tasks.py:17487).
                -- Surface en card del listado para que el chip "Sin tu
                -- feedback: N" aparezca SIN abrir el modal.
                NULLIF(mp.plan_data->>'_consecutive_zero_log_chunks', '')::int
                    AS consecutive_zero_log_chunks,
                COALESCE(mp.plan_data->'_shopping_coherence_block_history', '[]'::jsonb) AS coherence_history,
                -- [P3-HIST-ACTIVE-CHIP · 2026-05-18] Fechas de inicio del
                -- plan para que el frontend pueda derivar el bucket
                -- temporal (active / past / future) sin descargar todo
                -- `plan_data`. `grocery_start_date` es la fecha real del
                -- ciclo de compras del usuario (preferred); `cycle_start_date`
                -- es la fecha inmutable del primer día del plan original
                -- (fallback cuando `grocery_start_date` no se resolvió aún
                -- en el cron `_resolve_grocery_start_date`).
                mp.plan_data->>'grocery_start_date' AS grocery_start_date,
                mp.plan_data->>'cycle_start_date' AS cycle_start_date,
                mp.plan_data->'days'->0->'meals' AS preview_meals_raw,
                mp.plan_data->>'goal' AS goal_root,
                mp.plan_data->'assessment'->>'mainGoal' AS goal_assessment,
                mp.plan_data->>'diet_preference' AS diet_root,
                mp.plan_data->'assessment'->>'diet_preference' AS diet_assessment_snake,
                mp.plan_data->'assessment'->>'dietPreference' AS diet_assessment_camel,
                mp.plan_data->'assessment'->>'dietType' AS diet_assessment_type,
                COALESCE(mp.plan_data->'allergies', mp.plan_data->'assessment'->'allergies', mp.plan_data->'assessment'->'intolerances') AS allergies,
                COALESCE(qstats.pending_user_action_count, 0)::int AS chunk_pending_user_action_count,
                COALESCE(qstats.failed_count, 0)::int AS chunk_failed_count,
                COALESCE(qstats.failed_unreplaced_count, 0)::int AS chunk_failed_unreplaced_count,
                COALESCE(qstats.in_flight_count, 0)::int AS chunk_in_flight_count,
                -- [P3-HIST-CHUNK-SCHEDULED · 2026-05-18] Ver comentario
                -- en el LATERAL `qstats` (líneas ~8460). Permite al
                -- frontend distinguir "estos chunks se generarán cuando
                -- llegue su momento" (scheduled) de "el worker los está
                -- procesando ya" (running_now).
                COALESCE(qstats.scheduled_count, 0)::int AS chunk_scheduled_count,
                COALESCE(qstats.running_now_count, 0)::int AS chunk_running_now_count,
                COALESCE(qstats.completed_count, 0)::int AS chunk_completed_count,
                COALESCE(qstats.pantry_degraded_count, 0)::int AS chunk_pantry_degraded_count,
                qstats.pantry_degraded_reasons AS chunk_pantry_degraded_reasons,
                qtiers.tier_breakdown AS chunk_tier_breakdown,
                qaction.reason_code AS primary_action_reason_code
            FROM meal_plans mp
            LEFT JOIN (
                SELECT
                    meal_plan_id,
                    COUNT(*) FILTER (WHERE status = 'pending_user_action') AS pending_user_action_count,
                    COUNT(*) FILTER (WHERE status = 'failed') AS failed_count,
                    -- [P0-HIST-NEW-1 · 2026-05-09] failed chunks SIN sibling
                    -- completed para la misma (meal_plan_id, week_number).
                    -- El índice parcial `ux_plan_chunk_queue_live_week`
                    -- (migrations/p2_new_e:171) impide dos filas vivas
                    -- (pending/processing/stale/failed) por (plan, week)
                    -- pero PERMITE coexistencia `completed` + `failed` —
                    -- típicamente cuando un chunk completó días, fue
                    -- re-encolado (post-swap revalidation, manual retry) y
                    -- el segundo intento dead-letteró. La fila vieja sigue
                    -- contribuyendo a `failed_count` pero los días YA
                    -- están en plan_data.days vía la fila completed
                    -- hermana → bucket `complete`/`partial` no debería
                    -- elevarse a `action_required` por estos residuos.
                    --
                    -- Se hace via correlated subquery en lugar de window
                    -- function para mantener el GROUP BY plano y porque
                    -- la cardinalidad esperada es baja (<100 chunks/plan).
                    COUNT(*) FILTER (
                        WHERE status = 'failed'
                          AND NOT EXISTS (
                              SELECT 1
                              FROM plan_chunk_queue sibling
                              WHERE sibling.meal_plan_id = plan_chunk_queue.meal_plan_id
                                AND sibling.week_number = plan_chunk_queue.week_number
                                AND sibling.id != plan_chunk_queue.id
                                AND sibling.status = 'completed'
                          )
                    ) AS failed_unreplaced_count,
                    COUNT(*) FILTER (WHERE status IN ('pending', 'processing', 'stale')) AS in_flight_count,
                    -- [P3-HIST-CHUNK-SCHEDULED · 2026-05-18] Split del
                    -- in_flight_count en 2 dimensiones según el reloj
                    -- de scheduling (`execute_after`):
                    --   - scheduled_count: chunks pending/stale con
                    --     `execute_after > NOW()` → DORMIDOS esperando
                    --     su turno. NO se están "generando ahora" — el
                    --     worker filtra por `WHERE execute_after <= NOW()`
                    --     antes del pickup (cron_tasks.py:20376). Para un
                    --     plan de 7 días con `CHUNK_PROACTIVE_MARGIN_DAYS=0`,
                    --     el chunk-2 (días 4-7) tiene execute_after =
                    --     grocery_start_date + 3 días. Si hoy es día 1,
                    --     ese chunk vive 3 días dormido.
                    --   - running_now_count: chunks elegibles AHORA
                    --     (`execute_after <= NOW()`). Cubre:
                    --       · `status='processing'`: worker corriendo.
                    --       · `status='pending'/'stale'` con execute_after
                    --         vencido: en cola, será pickeado en el
                    --         próximo tick del scheduler (≤1 min).
                    --     El frontend usa este split para mostrar copy
                    --     preciso: "se generarán cuando llegue su momento"
                    --     vs "Mealfit los está generando ahora".
                    --
                    -- Edge case: si execute_after es NULL (chunk legacy
                    -- pre-migration con DEFAULT NOW() en INSERT que cayó
                    -- a NULL por bug), lo contamos como running_now —
                    -- conservador: el worker lo pickearía si pasa el
                    -- filtro (NULL <= NOW() es UNKNOWN ≈ false en
                    -- WHERE, pero el worker tiene fallback explícito).
                    COUNT(*) FILTER (
                        WHERE status IN ('pending', 'stale')
                          AND execute_after IS NOT NULL
                          AND execute_after > NOW()
                    ) AS scheduled_count,
                    COUNT(*) FILTER (
                        WHERE status IN ('pending', 'processing', 'stale')
                          AND (execute_after IS NULL OR execute_after <= NOW())
                    ) AS running_now_count,
                    COUNT(*) FILTER (WHERE status = 'completed') AS completed_count,
                    -- [P1-HIST-PANTRY-DEGRADED · 2026-05-09] Conteo de
                    -- chunks con `pantry_degraded_reason` no-NULL en
                    -- learning_metrics jsonb. Cubre planes generados
                    -- con señal de pantry comprometida (stale_snapshot,
                    -- empty_pantry_proxy, inventory_unreachable, etc).
                    -- El productor (cron_tasks.py:19631) escribe la
                    -- key vía `learning_metrics["pantry_degraded_reason"]
                    -- = form_data["_pantry_degraded_reason"]` cuando el
                    -- pipeline detectó el flag al pickup del chunk.
                    --
                    -- DISTINCT array de reasons sirve al tooltip del
                    -- chip — sin DISTINCT, planes con 5 chunks degraded
                    -- por la misma razón mostrarían "stale_snapshot,
                    -- stale_snapshot, ..." (5 veces). El FILTER excluye
                    -- chunks healthy (NULL no es elemento del array).
                    COUNT(*) FILTER (
                        WHERE learning_metrics ? 'pantry_degraded_reason'
                          AND learning_metrics->>'pantry_degraded_reason' IS NOT NULL
                          AND learning_metrics->>'pantry_degraded_reason' <> ''
                    ) AS pantry_degraded_count,
                    array_agg(
                        DISTINCT learning_metrics->>'pantry_degraded_reason'
                    ) FILTER (
                        WHERE learning_metrics ? 'pantry_degraded_reason'
                          AND learning_metrics->>'pantry_degraded_reason' IS NOT NULL
                          AND learning_metrics->>'pantry_degraded_reason' <> ''
                    ) AS pantry_degraded_reasons
                FROM plan_chunk_queue
                WHERE user_id = %s
                  AND meal_plan_id IS NOT NULL
                GROUP BY meal_plan_id
            ) qstats ON qstats.meal_plan_id = mp.id
            -- [P1-AUDIT-HIST-6 · 2026-05-09] Tier breakdown por plan
            -- vía LATERAL anidado: cada plan recibe su jsonb
            -- {tier: count} solo de chunks `completed` con
            -- quality_tier no-NULL. NULL si el plan no tiene chunks
            -- completed (Postgres LATERAL deja qtiers.tier_breakdown
            -- NULL si la subquery interna devuelve 0 rows). Tiers
            -- canónicos: llm/shuffle/edge/emergency/failed/paused/error.
            LEFT JOIN LATERAL (
                SELECT jsonb_object_agg(quality_tier, cnt) AS tier_breakdown
                FROM (
                    SELECT quality_tier, COUNT(*)::int AS cnt
                    FROM plan_chunk_queue inner_q
                    WHERE inner_q.meal_plan_id = mp.id
                      AND inner_q.user_id = %s
                      AND inner_q.status = 'completed'
                      AND inner_q.quality_tier IS NOT NULL
                    GROUP BY quality_tier
                ) t
            ) qtiers ON TRUE
            -- [P2-HIST-NEW-1 · 2026-05-09] Primary action reason del
            -- chunk bloqueante más temprano. Permite al frontend
            -- promover el chip "Acción" a "Acción: empty_pantry" en
            -- la card del listado — antes el reason solo era visible
            -- al abrir el modal (vía /blocked_reasons lazy fetch).
            -- Inconsistencia con Dashboard que ya muestra el reason
            -- en el slot del plan ACTIVO desde P0-DASH-CHIP-HONESTY.
            --
            -- Priority chain igual a /blocked_reasons (plans.py:3823+):
            --   dead_letter_reason → _pause_reason → _pantry_pause_reason
            --   → reason. SSOT extraído sería ideal; por ahora
            --   duplicamos con cita explícita.
            -- Order: week_number ASC para que el reason del chunk
            -- bloqueante MÁS TEMPRANO domine — un plan con varios
            -- chunks bloqueados típicamente solo necesita resolver el
            -- primero (los demás se desbloquean en cascada).
            LEFT JOIN LATERAL (
                SELECT
                    COALESCE(
                        qa.dead_letter_reason,
                        qa.pipeline_snapshot->>'_pause_reason',
                        qa.pipeline_snapshot->>'_pantry_pause_reason',
                        qa.pipeline_snapshot->>'reason'
                    ) AS reason_code
                FROM plan_chunk_queue qa
                WHERE qa.meal_plan_id = mp.id
                  AND qa.user_id = %s
                  AND (
                    qa.status = 'pending_user_action'
                    OR (qa.status = 'failed' AND qa.dead_letter_reason IS NOT NULL)
                  )
                ORDER BY qa.week_number ASC NULLS LAST,
                         qa.created_at ASC
                LIMIT 1
            ) qaction ON TRUE
            WHERE mp.user_id = %s
              AND mp.name IS NOT NULL
            ORDER BY GREATEST(
                mp.created_at,
                COALESCE(
                    (mp.plan_data->>'_plan_modified_at')::timestamptz,
                    mp.created_at
                )
            ) DESC,
            mp.created_at DESC
            LIMIT 200
            """,
            (verified_user_id, verified_user_id, verified_user_id, verified_user_id),
            fetch_all=True,
        ) or []
    except Exception as e:
        logger.error(f"❌ [P1-HIST-AUDIT-4] error cargando history-list: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

    # Anomalous coherence actions — coherente con
    # `History.jsx::getCoherenceAdjustsCount`.
    # [P2-HIST-AUDIT-13 · 2026-05-09] SSOT migró a
    # `constants.COHERENCE_ANOMALOUS_ACTIONS` (mismo patrón que
    # P1-AUDIT-HIST-7 lesson whitelist). El alias local
    # `_ANOMALOUS_COHERENCE_ACTIONS` se preserva como variable local
    # del bloque para retrocompat con tests inline. La conversión a
    # `frozenset` permite el `in` lookup O(1) (la constante es tuple
    # para que Python intern la deje como singleton estable).
    from constants import COHERENCE_ANOMALOUS_ACTIONS as _COHERENCE_ANOMALOUS_ACTIONS_TUPLE
    _ANOMALOUS_COHERENCE_ACTIONS = frozenset(_COHERENCE_ANOMALOUS_ACTIONS_TUPLE)

    plans = []
    for row in rows:
        # `coherence_history` viene como list[dict] (jsonb cast por
        # psycopg). Si parsea distinto, tratamos como vacío para no
        # romper render del listado por una card corrupta.
        history = row.get("coherence_history") or []
        coherence_adjusts_count = 0
        # [P1-4 · 2026-05-10] Extraer hipótesis de la ÚLTIMA entry anomalous
        # del history. El frontend las usa para construir un tooltip rico en
        # el chip "N ajustes" (humanizando vía `getCoherenceHypothesisLabel`).
        # Recorrer al revés es O(N) en peor caso pero el cap del history es 20
        # (P3-NEW-C/MEALFIT_COHERENCE_BLOCK_HISTORY_CAP), N pequeño.
        # Cap 5 hipótesis distintas: los tooltips quedan legibles y el caso
        # frecuente es 1-3 divergencias dominantes por entry.
        coherence_last_hypotheses: list[str] = []
        if isinstance(history, list):
            for entry in history:
                if not isinstance(entry, dict):
                    continue
                if entry.get("action_taken") in _ANOMALOUS_COHERENCE_ACTIONS:
                    coherence_adjusts_count += 1
            # Walk reverse para encontrar la última entry anomalous.
            for entry in reversed(history):
                if not isinstance(entry, dict):
                    continue
                if entry.get("action_taken") not in _ANOMALOUS_COHERENCE_ACTIONS:
                    continue
                divs = entry.get("divergences")
                if not isinstance(divs, list):
                    break
                seen: set[str] = set()
                for d in divs:
                    if not isinstance(d, dict):
                        continue
                    h = d.get("hypothesis")
                    if not isinstance(h, str) or not h or h in seen:
                        continue
                    seen.add(h)
                    coherence_last_hypotheses.append(h)
                    if len(coherence_last_hypotheses) >= 5:
                        break
                break

        # Preview meals: solo (name, meal) por meal del primer día,
        # cap 4. El frontend espera (string short + emoji) — los
        # demás keys (cals, recipe, etc.) son ruido para preview.
        # Filter ANTES de slice (espeja `activeMeals.slice(0, 3)` del
        # JS legacy en `renderMealPreview`): el cap aplica a meals
        # VÁLIDOS, ignorando entries corruptos en el array.
        preview_meals_raw = row.get("preview_meals_raw") or []
        preview_meals = []
        if isinstance(preview_meals_raw, list):
            for m in preview_meals_raw:
                if not isinstance(m, dict) or not m.get("name"):
                    continue
                # [P2-HIST-AUDIT-12 · 2026-05-09] Filtrar meals con
                # `isSkipped=true` ANTES del slice de 4. Sin esto, si
                # los primeros 4 meals del primer día incluyen 3
                # skipped, el response devolvía esos 4; el frontend
                # (`renderMealPreview`) filtraba `!m.isSkipped` post-
                # slice y terminaba renderizando solo 1 chip cuando
                # debía mostrar 3 válidos del SIGUIENTE meal del día.
                # El filter aquí garantiza que el cap 4 cuente meals
                # VÁLIDOS, no posiciones del array. El frontend
                # mantiene su filter como defense-in-depth (planes
                # legacy o response cacheado pueden traer skipped).
                if m.get("isSkipped"):
                    continue
                preview_meals.append({
                    "name": m.get("name"),
                    "meal": m.get("meal") or "",
                })
                if len(preview_meals) >= 4:
                    break

        # Goal/diet con fallback root → assessment (espeja
        # `getSmartTags` JS líneas 426-437).
        goal = row.get("goal_root") or row.get("goal_assessment")
        diet = (
            row.get("diet_root")
            or row.get("diet_assessment_snake")
            or row.get("diet_assessment_camel")
            or row.get("diet_assessment_type")
        )

        # Allergies: si el jsonb era list[str] llega como list; si era
        # objeto u otra cosa, tratamos como vacío.
        allergies_raw = row.get("allergies")
        allergies = allergies_raw if isinstance(allergies_raw, list) else []

        # `created_at` es datetime → isoformat. `plan_modified_at`
        # ya es text (extraído via ->>).
        created_at = row.get("created_at")
        if hasattr(created_at, "isoformat"):
            created_at = created_at.isoformat()  # pyright: ignore[reportOptionalMemberAccess]  (guarded por hasattr)

        plans.append({
            "id": row.get("id"),
            "name": row.get("name"),
            "created_at": created_at,
            "calories": row.get("calories"),
            "macros": row.get("macros"),
            "plan_modified_at": row.get("plan_modified_at"),
            "generation_status": row.get("generation_status"),
            "total_days_requested": row.get("total_days_requested"),
            "days_generated": row.get("days_generated") or 0,
            "user_action_required": row.get("user_action_required"),
            "recovery_exhausted_count": row.get("recovery_exhausted_count") or 0,
            "user_forced_simplified_weeks": row.get("user_forced_simplified_weeks"),
            # [P2-HIST-AUDIT-C · 2026-05-09] Shift días acumulados.
            # int o None (None cuando el plan no sufrió shift_plan).
            "shift_days_accumulated": (
                int(row["shift_days_accumulated"])
                if isinstance(row.get("shift_days_accumulated"), int)
                else None
            ),
            # [P0-HIST-LEARN-2 · 2026-05-09] None si la key no existe
            # (plan legacy pre-engagement-tracking) O el valor no es int.
            "consecutive_zero_log_chunks": (
                int(row["consecutive_zero_log_chunks"])
                if isinstance(row.get("consecutive_zero_log_chunks"), int)
                else None
            ),
            # [P3-HIST-ACTIVE-CHIP · 2026-05-18] Fechas de inicio para el
            # chip temporal "Activo" del listado. Strings ISO (`YYYY-MM-DD`
            # o `YYYY-MM-DDTHH:MM:SS+TZ`) ya que vienen del jsonb via `->>`;
            # `None` si la key no existe en plan_data (plan legacy o
            # generado antes de que el cron resuelva grocery_start_date).
            # El frontend tiene fallback chain: grocery_start_date →
            # cycle_start_date → plan.created_at.
            "grocery_start_date": (
                row.get("grocery_start_date")
                if isinstance(row.get("grocery_start_date"), str)
                and row.get("grocery_start_date").strip()  # pyright: ignore[reportOptionalMemberAccess]  (guarded por isinstance str)
                else None
            ),
            "cycle_start_date": (
                row.get("cycle_start_date")
                if isinstance(row.get("cycle_start_date"), str)
                and row.get("cycle_start_date").strip()  # pyright: ignore[reportOptionalMemberAccess]  (guarded por isinstance str)
                else None
            ),
            "coherence_adjusts_count": coherence_adjusts_count,
            # [P1-4 · 2026-05-10] Hipótesis (max 5 distintas) de la última
            # entry anomalous del history. Vacía si no hay anomalous o si
            # las divergencias no traen `hypothesis` (cap_swallowed_modifier
            # / unit_mismatch / yield_uncovered / pantry_overdeduct / unknown).
            "coherence_last_hypotheses": coherence_last_hypotheses,
            "preview_meals": preview_meals,
            "goal": goal,
            "diet_preference": diet,
            "allergies": allergies,
            # [P1-AUDIT-HIST-4 · 2026-05-09] Counters embebidos del
            # `plan_chunk_queue` (vía LEFT JOIN). Mismo set que
            # devuelve `/history-status-summary` (P0-AUDIT-HIST-2),
            # ahora en el mismo response del listado para que el
            # frontend reconcilie sin un segundo roundtrip y sin
            # race condition. Counters siempre presentes (0 cuando
            # el plan no tiene chunks — `COALESCE` en el SELECT).
            "chunk_pending_user_action_count": int(row.get("chunk_pending_user_action_count") or 0),
            "chunk_failed_count": int(row.get("chunk_failed_count") or 0),
            # [P0-HIST-NEW-1 · 2026-05-09] failed sin sibling completed.
            # `getStatusInfo` (frontend) usa este counter para la regla de
            # reconciliación que eleva el bucket — `chunk_failed_count`
            # incluye residuos post-recompletion (mismo (plan, week) tiene
            # fila completed + fila failed) que NO ameritan banner de
            # acción porque los días ya están en plan_data.
            "chunk_failed_unreplaced_count": int(row.get("chunk_failed_unreplaced_count") or 0),
            "chunk_in_flight_count": int(row.get("chunk_in_flight_count") or 0),
            # [P3-HIST-CHUNK-SCHEDULED · 2026-05-18] Split de chunk_in_flight_count
            # en 2 dimensiones según el reloj de scheduling:
            # - chunk_scheduled_count: dormidos esperando su execute_after.
            # - chunk_running_now_count: elegibles AHORA (processing o pending
            #   con execute_after <= NOW()). El frontend usa este split para
            #   mostrar copy preciso vs el mensaje genérico previo que decía
            #   "generando ahora" incluso para chunks que duermen 3-7 días.
            "chunk_scheduled_count": int(row.get("chunk_scheduled_count") or 0),
            "chunk_running_now_count": int(row.get("chunk_running_now_count") or 0),
            "chunk_completed_count": int(row.get("chunk_completed_count") or 0),
            # [P1-AUDIT-HIST-6 · 2026-05-09] Tier breakdown — dict
            # `{tier: count}` solo de chunks completed. Útil para
            # que el modal del Historial muestre la distribución de
            # calidad de un plan archivado (vs el chunk-status del
            # plan activo que ya lo expone). `None` cuando el plan
            # no tiene chunks completed o todos tienen quality_tier
            # NULL — el frontend trata None como "no info" y omite
            # el render. Si llega como dict vacío `{}` (LATERAL no
            # encontró rows con tier no-NULL) lo coercemos a None
            # para cero render del bloque.
            "chunk_tier_breakdown": (
                row.get("chunk_tier_breakdown")
                if isinstance(row.get("chunk_tier_breakdown"), dict)
                and row.get("chunk_tier_breakdown")
                else None
            ),
            # [P1-HIST-PANTRY-DEGRADED · 2026-05-09] Surface
            # retroactivo del flag `learning_metrics.pantry_degraded_reason`.
            # Count > 0 => al menos un chunk del plan se generó con
            # señal de pantry comprometida. El frontend usa esto para
            # un chip ámbar "Pantry degradada" en la card del listado;
            # el array de reasons distintas alimenta el tooltip.
            # Count siempre presente (COALESCE 0 en SELECT); reasons
            # NULL/array vacío se sanitizan a None para que el
            # frontend distinga "sin info" de "lista vacía".
            "chunk_pantry_degraded_count": int(row.get("chunk_pantry_degraded_count") or 0),
            "chunk_pantry_degraded_reasons": (
                [str(r) for r in row.get("chunk_pantry_degraded_reasons") if r]  # pyright: ignore[reportOptionalIterable]  (guarded por isinstance list abajo)
                if isinstance(row.get("chunk_pantry_degraded_reasons"), list)
                and row.get("chunk_pantry_degraded_reasons")
                else None
            ),
            # [P2-HIST-NEW-1 · 2026-05-09] Primary action reason — un
            # solo string canónico extraído del chunk bloqueante más
            # temprano. None cuando no hay chunks bloqueados (plan
            # healthy o ya resuelto). Frontend usa esta key para
            # promover el chip "Acción" a "Acción: empty_pantry".
            "primary_action_reason": (
                str(row.get("primary_action_reason_code")).strip()
                if isinstance(row.get("primary_action_reason_code"), str)
                and row.get("primary_action_reason_code").strip()  # pyright: ignore[reportOptionalMemberAccess]  (guarded por isinstance str)
                else None
            ),
        })

    # [P1-HIST-AUDIT-4-FOLLOWUP · 2026-05-09] Logger de diagnóstico
    # para debug del flujo Historial. Visible en logs del backend; el
    # operador puede grep `[HISTORY-LIST]` para ver:
    #   - Qué user_id resolvió el JWT (`verified_user_id`).
    #   - Cuántos planes devolvió la query (`count`).
    # Si count=0 pero el plan existe en DB, el `verified_user_id`
    # NO matchea el `meal_plans.user_id` del plan → mismatch de
    # cuentas (typical: 2 emails similares con typo en login).
    logger.info(
        "[HISTORY-LIST] user=%s count=%d",
        verified_user_id, len(plans),
    )
    return {"plans": plans}


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
    # [P1-PROD-AUDIT-BUNDLE · 2026-05-28] Comparación constant-time del admin
    # token. Este gate protege TODOS los `/admin/*` (plans/system/notifications)
    # con el mismo `CRON_SECRET`; un `!=` plano corta en el primer byte que
    # difiere → side-channel de timing que permite recuperar el secreto
    # byte-a-byte. `hmac.compare_digest` es el mismo patrón que ya usa el
    # webhook handler (app.py). Tooltip-anchor: P1-ADMIN-TOKEN-CONSTTIME.
    if not hmac.compare_digest(token, cron_secret):
        raise HTTPException(status_code=403, detail="Invalid admin token")


# ---------------------------------------------------------------------------
# [P2-ADMIN-RATE-LIMIT · 2026-05-15] Rate limiter compartido para todos los
# endpoints `/admin/*`.
#
# Pre-fix: los ~12 endpoints `/admin/*` (8 en plans.py + 3 en system.py)
# estaban auth-gated por `CRON_SECRET` pero sin rate limiter. Un operador
# descuidado con un script en loop (mala `crontab` `* * * * *` en vez de
# `*/30 * * * *`) o un atacante que obtuvo el CRON_SECRET vía leak puede
# saturar el pool DB — `/admin/health-snapshot` hace 6 queries paralelas.
#
# Diseño: instancia única de `RateLimiter` con key por IP (admin endpoints
# no usan `verified_user_id` — se autentican con CRON_SECRET, no JWT). La
# `RateLimiter.__call__` retorna 429 con header `Retry-After: <N>` cuando
# se excede; el helper la invoca con `verified_user_id=None` para forzar
# el fallback a `ip:<client_ip>`.
#
# Defaults conservadores: 60 req / 60s por IP. Un dashboard SRE pollando
# 1 req/s queda bajo el cap; un script roto en loop 100x/s se cae rápido.
# Knobs `MEALFIT_ADMIN_RATE_LIMIT_PER_MIN` y `MEALFIT_ADMIN_RATE_LIMIT_PERIOD_S`
# permiten override sin redeploy.
#
# Tooltip-anchor: P2-ADMIN-RATE-LIMIT.
# Test: `tests/test_p2_admin_rate_limit.py`.
# ---------------------------------------------------------------------------
from rate_limiter import RateLimiter as _AdminRateLimiterCls
from knobs import _env_int as _admin_env_int


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


_ADMIN_RATE_LIMIT_MAX_CALLS = _clamp_int(
    _admin_env_int("MEALFIT_ADMIN_RATE_LIMIT_PER_MIN", 60), 10, 600
)
_ADMIN_RATE_LIMIT_PERIOD_S = _clamp_int(
    _admin_env_int("MEALFIT_ADMIN_RATE_LIMIT_PERIOD_S", 60), 10, 3600
)
_ADMIN_RATE_LIMITER = _AdminRateLimiterCls(
    max_calls=_ADMIN_RATE_LIMIT_MAX_CALLS,
    period_seconds=_ADMIN_RATE_LIMIT_PERIOD_S,
)


def _check_admin_rate_limit(request: Request) -> None:
    """Aplica rate limiting global a un admin endpoint. Llama justo después
    de `_verify_admin_token(...)`. Key por IP (`verified_user_id=None`
    fuerza el fallback IP del RateLimiter).

    Levanta `HTTPException(429)` con header `Retry-After` cuando se excede.
    """
    _ADMIN_RATE_LIMITER(request, verified_user_id=None)


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
    _check_admin_rate_limit(request)  # [P2-ADMIN-RATE-LIMIT]
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
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


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
    _check_admin_rate_limit(request)  # [P2-ADMIN-RATE-LIMIT]
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
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


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
    _check_admin_rate_limit(request)  # [P2-ADMIN-RATE-LIMIT]
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
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/{plan_id}/chunks/{chunk_id}/regenerate-simplified")
def api_regenerate_dead_lettered_simplified(
    plan_id: str,
    chunk_id: str,
    verified_user_id: Optional[str] = Depends(verify_api_quota),
):
    """[P1-ζ] Permite al usuario re-encolar un chunk dead-lettered forzando
    `flexible_mode + advisory_only` para que el siguiente intento NO falle por
    los gates que lo dead-letearon (lecciones perdidas, anchor irresoluble,
    pantry insuficiente).

    Cubre el último escalón de la cascada de recovery: cuando el sistema agotó
    sus reintentos automáticos y el chunk quedó marcado con `dead_lettered_at`,
    el banner del frontend ofrece este endpoint como CTA. El endpoint NO
    requiere admin token: el usuario es dueño del plan y puede decidir aceptar
    una versión simplificada antes que esperar intervención manual.

    Flujo:
      1. Validar ownership del plan.
      2. Validar que el chunk efectivamente está dead_lettered.
      3. Limpiar `dead_lettered_at`, `dead_letter_reason`, attempts.
      4. Inyectar `_pantry_flexible_mode=True`, `_pantry_advisory_only=True`,
         `_learning_flexible_mode=True`, `_user_forced_simplified=True` en
         pipeline_snapshot (y en form_data dentro del snapshot, donde el day-
         tagging del merge los lee).
      5. Marcar `_user_action_required=None` en plan_data para retirar el
         banner.
      6. Re-encolar status='pending' execute_after=NOW().
    """
    from db_core import execute_sql_query, execute_sql_write
    import json
    try:
        plan_row = execute_sql_query(
            "SELECT user_id, plan_data FROM meal_plans WHERE id = %s",
            (plan_id,), fetch_one=True
        )
        if not plan_row:
            raise HTTPException(status_code=404, detail="Plan no encontrado")
        if verified_user_id and str(plan_row["user_id"]) != str(verified_user_id):
            raise HTTPException(status_code=403, detail="No autorizado")

        chunk_row = execute_sql_query(
            """
            SELECT id, status, dead_lettered_at, pipeline_snapshot, week_number
            FROM plan_chunk_queue
            WHERE id = %s AND meal_plan_id = %s
            """,
            (chunk_id, plan_id),
            fetch_one=True,
        )
        if not chunk_row:
            raise HTTPException(status_code=404, detail="Chunk no encontrado")
        if not chunk_row.get("dead_lettered_at"):
            raise HTTPException(
                status_code=409,
                detail="El chunk no está dead-lettered. Solo chunks dead-lettered pueden regenerarse en modo simplificado.",
            )

        snap = chunk_row["pipeline_snapshot"]
        if isinstance(snap, str):
            snap = json.loads(snap)
        if not isinstance(snap, dict):
            snap = {}

        snap["_pantry_flexible_mode"] = True
        snap["_pantry_advisory_only"] = True
        snap["_learning_flexible_mode"] = True
        snap["_user_forced_simplified"] = True
        snap["_user_forced_simplified_at"] = datetime.now(timezone.utc).isoformat()
        # Limpiar gates que dead-letearon el chunk para que no re-disparen.
        snap.pop("_pantry_pause_reason", None)
        snap.pop("_pantry_pause_started_at", None)
        snap.pop("_pantry_pause_reminders", None)
        snap.pop("_force_unblock_attempted_at", None)
        snap["_learning_ready_deferrals"] = 0

        fd = snap.get("form_data") or {}
        if isinstance(fd, dict):
            fd["_pantry_flexible_mode"] = True
            fd["_pantry_advisory_only"] = True
            fd["_pantry_degraded_reason"] = "user_forced_simplified"
            snap["form_data"] = fd

        # [P2-HIST-MODALS-A11Y · 2026-05-30] Defense-in-depth: WHERE filtra
        # también por `meal_plan_id` (el ownership ya se validó arriba en el
        # SELECT inicial + el lookup `chunk_row` con `meal_plan_id = %s`).
        # Mismo patrón belt-and-suspenders que el UPDATE de `meal_plans` de
        # abajo (P1-NEW-4) y que el hermano `retry-chunk` (P0-HIST-IDOR-1):
        # un futuro refactor que rompa el check upstream sin tocar este
        # UPDATE no debe re-introducir un re-enqueue cross-plan.
        execute_sql_write(
            """
            UPDATE plan_chunk_queue
            SET status = 'pending',
                dead_lettered_at = NULL,
                dead_letter_reason = NULL,
                attempts = 0,
                escalated_at = NOW(),
                execute_after = NOW(),
                pipeline_snapshot = %s::jsonb,
                updated_at = NOW()
            WHERE id = %s AND meal_plan_id = %s
            """,
            (json.dumps(snap, ensure_ascii=False), chunk_id, plan_id),
        )

        # Limpiar banner del frontend + [P3-2] mirror del flag por semana.
        # `_user_forced_simplified_weeks` es un dict {week_number: iso_ts}
        # que el frontend lee desde plan_data (vía supabase direct) para
        # mostrar un badge sutil en los días de esa semana. Sin este mirror,
        # el flag solo vivía en plan_chunk_queue.pipeline_snapshot — que el
        # frontend nunca consulta — y el toggle quedaba write-only desde la
        # perspectiva UX.
        _wn_str = str(int(chunk_row["week_number"])) if chunk_row.get("week_number") is not None else "0"
        _ts_iso = datetime.now(timezone.utc).isoformat()
        # [P1-NEW-4 · 2026-05-10] Defense-in-depth: WHERE filtra también
        # por user_id. El handler hace ownership check más arriba en el
        # SELECT inicial, pero un futuro refactor que rompa ese check sin
        # tocar este UPDATE re-introduce IDOR. Mismo patrón que cierra
        # P0-HIST-IDOR-1 retry-chunk:4119-4123.
        execute_sql_write(
            """
            UPDATE meal_plans
            SET plan_data = jsonb_set(
                    jsonb_set(
                        jsonb_set(
                            COALESCE(plan_data, '{}'::jsonb),
                            '{_user_action_required}',
                            'null'::jsonb,
                            true
                        ),
                        '{generation_status}',
                        '"partial"'::jsonb,
                        true
                    ),
                    ARRAY['_user_forced_simplified_weeks', %s],
                    to_jsonb(%s::text),
                    true
                ),
                updated_at = NOW()
            WHERE id = %s AND user_id = %s
            """,
            (_wn_str, _ts_iso, plan_id, verified_user_id),
        )

        # Resolver alertas system_alerts asociadas a este chunk dead-lettered.
        # [P2-CHUNK-8] Dos fixes:
        #   (a) Typo: el alert_key canónico es `dead_lettered_chunks_recent`
        #       (cron_tasks.py:18103); el código tenía las dos palabras invertidas,
        #       así que el IN nunca matcheaba la agregada y ésta quedaba viva tras
        #       un regenerate-simplified exitoso.
        #   (b) Faltaba la alerta per-chunk `dead_lettered_chunk:<plan>:<week>`
        #       (cron_tasks.py:13515). La doc (system_alerts_resolution_table.md L21)
        #       declara que `regenerate-simplified` la resuelve, pero no estaba en el IN
        #       → quedaba huérfana hasta que el sweep por edad la cerrara.
        try:
            execute_sql_write(
                """
                UPDATE system_alerts
                SET resolved_at = NOW()
                WHERE resolved_at IS NULL
                  AND alert_key IN (
                    'chunk_paused_indefinitely:' || %s || ':' || %s,
                    'dead_lettered_chunk:' || %s || ':' || %s,
                    'dead_lettered_chunks_recent'
                  )
                """,
                (
                    plan_id, str(chunk_row["week_number"]),
                    plan_id, str(chunk_row["week_number"]),
                ),
            )
        except Exception:
            pass

        logger.warning(
            f"[P1-ζ/USER-FORCED-SIMPLIFIED] user={verified_user_id} plan={plan_id} "
            f"chunk={chunk_id} week={chunk_row['week_number']} re-encolado en flexible_mode."
        )

        # [P2-LIVE-7 · 2026-05-11] Audit api_usage. Mismo razonamiento que
        # /retry-chunk: el endpoint cobra cuota al validar pero no
        # incrementaba el contador. Re-encolar chunk = futuro LLM call.
        try:
            log_api_usage(verified_user_id, "regenerate_simplified")  # pyright: ignore[reportArgumentType]  (log_api_usage maneja None/guest internamente)
        except Exception as _audit_err:
            logger.warning(f"[P2-LIVE-7] log_api_usage regenerate_simplified falló: {_audit_err}")

        return {
            "success": True,
            "chunk_id": chunk_id,
            "week_number": chunk_row["week_number"],
            "message": (
                "Chunk re-encolado en modo simplificado. Procesará en el próximo tick "
                "del worker y los días generados se marcarán como advisory."
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [P1-ζ] Error en regenerate-simplified: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


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
        # [P1-NEW-4 · 2026-05-10] Defense-in-depth: WHERE filtra también
        # por user_id (ownership check explícito ya ocurrió arriba en
        # `plan_row` lookup; este es doble candado para futuros refactors).
        execute_sql_write("""
            UPDATE meal_plans
            SET plan_data = jsonb_set(plan_data, '{generation_status}', '"partial"'),
                updated_at = NOW()
            WHERE id = %s AND user_id = %s
        """, (plan_id, verified_user_id))

        # [P0-REGEN-BILLING · 2026-05-24] Audit api_usage 1×N (no 1×1).
        # Cada chunk re-encolado consume un LLM call independiente cuando el
        # worker lo procese; loggear 1 sola vez subcontaba billing y permitía
        # quota bypass (user regeneraba N chunks pagando 1 cargo y consumiendo
        # N LLM calls downstream). El loop emite N rows en api_usage para que
        # el cap mensual (gratis=15/basic=50/plus=200) refleje el costo real.
        # Ancla del test: `test_p0_regen_billing.py`.
        if regenerated > 0 and verified_user_id:
            for _ in range(regenerated):
                try:
                    log_api_usage(verified_user_id, "regen_degraded")
                except Exception as _audit_err:
                    logger.warning(f"[P0-REGEN-BILLING] log_api_usage regen_degraded falló: {_audit_err}")

        return {
            "success": True,
            "regenerated": regenerated,
            "message": f"{regenerated} chunks degradados re-encolados. Procesarán en el próximo tick del worker."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [GAP C] Error en /regen-degraded: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


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
    _check_admin_rate_limit(request)  # [P2-ADMIN-RATE-LIMIT]
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
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/admin/chunks/{chunk_id}/escalate")
def api_admin_escalate_chunk(chunk_id: str, request: Request):
    """[GAP A] Forzar escalado/pickup inmediato de un chunk concreto."""
    _verify_admin_token(request.headers.get("authorization"))
    _check_admin_rate_limit(request)  # [P2-ADMIN-RATE-LIMIT]
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
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.get("/admin/guest-metrics")
def api_admin_guest_metrics_status(request: Request):
    """[P1-33] Inspección operacional del flag `_GUEST_METRICS_ENABLED`.

    Antes este flag se decidía 1× al startup vía probe contra
    `pipeline_metrics`; si fallaba, las métricas de guest se perdían
    DURANTE TODA la vida del proceso sin que SRE lo viera (solo log
    CRITICAL en startup, fácil de perder en restarts agitados).

    Este endpoint permite:
      - Ver el estado actual sin SSH al pod ni redeploy.
      - Verificar el resultado del último probe + timestamp + error.
      - Detectar drift entre estado y schema actual de la DB.

    Requires Authorization: Bearer <CRON_SECRET>.
    """
    _verify_admin_token(request.headers.get("authorization"))
    _check_admin_rate_limit(request)  # [P2-ADMIN-RATE-LIMIT]
    from graph_orchestrator import get_guest_metrics_status
    return get_guest_metrics_status()


@router.post("/admin/guest-metrics/probe")
def api_admin_guest_metrics_probe(request: Request):
    """[P1-33] Re-ejecuta el probe de schema sin restart.

    Caso de uso: SRE acaba de aplicar `ALTER TABLE pipeline_metrics ALTER
    COLUMN user_id DROP NOT NULL;` manualmente sobre una DB que estaba
    drifted. Sin este endpoint, había que reiniciar el worker para
    re-evaluar el flag (perdiendo conexiones cacheadas + pipelines en
    vuelo). Con este endpoint, el probe corre on-demand y el flag se
    actualiza en memoria.

    Devuelve el snapshot post-probe (mismo formato que GET).
    """
    _verify_admin_token(request.headers.get("authorization"))
    _check_admin_rate_limit(request)  # [P2-ADMIN-RATE-LIMIT]
    from graph_orchestrator import (
        verify_pipeline_metrics_guest_insert,
        get_guest_metrics_status,
    )
    try:
        result = verify_pipeline_metrics_guest_insert()
        snapshot = get_guest_metrics_status()
        snapshot["probe_executed"] = True
        snapshot["probe_result"] = result
        return snapshot
    except Exception as e:
        logger.error(f"❌ [P1-33] Error ejecutando probe on-demand: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/admin/guest-metrics/force")
def api_admin_guest_metrics_force(request: Request, data: dict = Body(...)):
    """[P1-33] Override manual del flag (sin re-probe).

    Body: `{"enabled": bool, "reason": "string opcional"}`.

    Casos de uso:
      - SRE detecta inserts fallidos en logs y quiere desactivar
        proactivamente antes de la próxima ventana de probe.
      - SRE acaba de aplicar la migración manualmente y quiere habilitar
        sin esperar al probe (que requiere INSERT/DELETE round-trip).
      - Disable temporal durante incident response (DB stress, alertas
        falsas) para silenciar telemetría sin migración de schema.

    El override queda en memoria hasta el próximo probe (que puede
    re-flippearlo si el schema sigue mal). El reason queda en
    `last_reason` para auditoría.
    """
    _verify_admin_token(request.headers.get("authorization"))
    _check_admin_rate_limit(request)  # [P2-ADMIN-RATE-LIMIT]
    from graph_orchestrator import force_set_guest_metrics_enabled
    enabled = data.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(
            status_code=400,
            detail="Body debe incluir 'enabled': true|false",
        )
    reason = data.get("reason") if isinstance(data.get("reason"), str) else None
    return force_set_guest_metrics_enabled(enabled=enabled, reason=reason)
