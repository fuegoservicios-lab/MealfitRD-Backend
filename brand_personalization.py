"""[P1-SUPERMARKET-PERSONALIZATION · 2026-07-03] (audit v6 · P1-2) Conexión supermarket → generación.

Los ~2,000 `supermarket_products` alimentaban SOLO la lista de compras (overlay de costeo
P1-SUPERMARKET-COSTING) y las sugerencias de presupuesto — cero influencia en la CREACIÓN de
platos. Este módulo cierra la fase 1 del roadmap (docs/supermarket_db.md): las marcas que el
usuario eligió en "Marcas del súper" (`user_brand_preferences` JOIN `supermarket_products` vía
`master food_name`) se convierten en una señal de preferencia POSITIVA para el planner y el
day-generator, por el MISMO canal que el taste aprendido (append a `taste_profile` en
`_build_shared_context` — fluye a esqueleto + day-gen sin tocar sus prompts).

Claves del diseño (espejo de taste_model):
  - Señal SUAVE: "prefiérelos cuando encajen" — jamás fuerza un alimento en cada plato ni
    rompe variedad/clínica (los gates deterministas corren después e ignoran este bloque).
  - Solo usuarios reales (UUID ≥30 chars); guests jamás consultan la tabla.
  - Usuarios sin preferencias → '' (byte-equivalente → prompt-cache preservado).
  - Fail-open TOTAL: cualquier error → '' (la generación jamás se bloquea por esto).
  - Los food_name provienen del match contra la lista de compras del plan (derivada del
    catálogo verificado) → inyectar estos nombres NO abre la puerta a alimentos off-catalog.

Rollback sin redeploy: MEALFIT_BRAND_PREF_PERSONALIZATION=false.
Test ancla: test_p1_supermarket_personalization.py. tooltip-anchor: P1-SUPERMARKET-PERSONALIZATION
"""
from __future__ import annotations

import logging

from knobs import _env_bool, _env_int

logger = logging.getLogger(__name__)

BRAND_PREF_PERSONALIZATION_ENABLED = _env_bool("MEALFIT_BRAND_PREF_PERSONALIZATION", True)
BRAND_PREF_MAX_ITEMS = _env_int("MEALFIT_BRAND_PREF_MAX_ITEMS", 8, validator=lambda v: 1 <= v <= 20)


def _is_real_user(user_id) -> bool:
    u = str(user_id or "").strip().lower()
    return bool(u) and u != "guest" and not u.startswith("guest") and len(u) >= 30  # UUID-ish


def fetch_user_brand_foods(user_id) -> list:
    """Alimentos (food_name master) con marca preferida del usuario, dedupeados por alimento.
    [{"food": str, "brand": str}] — solo productos activos. Fail-open: []."""
    if not _is_real_user(user_id):
        return []
    try:
        from db import execute_sql_query
        rows = execute_sql_query(
            """
            SELECT DISTINCT ON (lower(p.food_key)) p.food_key AS food, sp.brand
            FROM public.user_brand_preferences p
            JOIN public.supermarket_products sp ON sp.id = p.product_id
            WHERE p.user_id = %s AND sp.active
            ORDER BY lower(p.food_key), p.updated_at DESC NULLS LAST
            """,
            (str(user_id),),
            fetch_all=True,
        ) or []
    except Exception as _e:
        logger.debug(f"[P1-SUPERMARKET-PERSONALIZATION] fetch no-op: {type(_e).__name__}: {_e}")
        return []
    out = []
    for r in rows:
        food = str(r.get("food") or "").strip()
        if not food:
            continue
        out.append({"food": food, "brand": str(r.get("brand") or "").strip()})
    return out[: int(BRAND_PREF_MAX_ITEMS)]


def build_brand_pref_context(user_id) -> str:
    """Bloque de prompt con los alimentos que el usuario eligió del súper (señal positiva).
    '' si está OFF / guest / sin preferencias / error (byte-equivalente = prompt-cache ok)."""
    if not BRAND_PREF_PERSONALIZATION_ENABLED:
        return ""
    try:
        foods = fetch_user_brand_foods(user_id)
        if not foods:
            return ""
        lineas = ", ".join(
            (f"{f['food']} ({f['brand']})" if f.get("brand") else f["food"]) for f in foods
        )
        return (
            "=== ALIMENTOS QUE EL USUARIO YA COMPRA EN EL SÚPER (marcas preferidas) ===\n"
            f"El usuario eligió estas presentaciones para su lista de compras: {lineas}.\n"
            "Son señal de preferencia POSITIVA: le gustan y los tiene a mano. Cuando un plato "
            "admita uno de estos alimentos con naturalidad, PREFIÉRELO sobre un equivalente "
            "(ej. si el desayuno lleva yogurt y eligió yogurt griego, usa yogurt griego). "
            "NO los fuerces en cada plato ni sacrifiques variedad, horarios ni reglas clínicas."
        )
    except Exception as _e:
        logger.debug(f"[P1-SUPERMARKET-PERSONALIZATION] contexto no-op: {type(_e).__name__}: {_e}")
        return ""
