"""[P1-NEON-DB-MIGRATION · 2026-06-12] Endpoints de datos user-scoped que
reemplazan los accesos directos del frontend a Postgres via PostgREST
(cliente JS legacy). Post-migración a Neon, el frontend NO tiene acceso a la DB:
PostgREST apuntaba al Postgres anterior (datos stale tras el cutover) —
leerlo es servir data vieja; escribirlo es split-brain.

Reemplazos cubiertos (audit 2026-06-12, sección frontend):
- RPC `increment_inventory_quantity` (Pantry velocímetro) → POST /api/inventory/increment
- RPC `update_health_profile_merge` + UPDATEs escalares → PATCH /api/profile
- SELECTs de `user_inventory` con embed master_ingredients → GET /api/inventory
- INSERTs de Pantry (add/undo/restore-depleted, semántica 23505) → POST /api/inventory/items
- DELETEs de Pantry (item/all) → DELETE /api/inventory/items[/{id}]
- Catálogo `master_ingredients` → GET /api/catalog
- SELECT del último plan (restoreSessionData/regenerate/restore/recalc-precheck)
  → GET /api/plans-data/latest
- SELECT lazy de plan_data por id (History modal, PDF sync) → GET /api/plans-data/{plan_id}

Convenciones:
- Auth: `get_verified_user_id` SIN `verify_api_quota` — cero costo LLM; el
  paywall 402 bloquearía la nevera/perfil del usuario (misma razón que la
  historial-quota-exemption, CLAUDE.md).
- Invariante I2: TODA query filtra `AND user_id = %s` server-side (el delete
  legacy de Pantry confiaba en RLS; aquí el filtro es explícito).
- Paridad de tipos PostgREST: uuid→::text, numeric→::float8, timestamptz→
  `to_jsonb(col)#>>'{}'` (ISO-8601 con 'T' — `::text` daría separador espacio
  que Safari `new Date()` no parsea).
- [P1-ASYNC-SYNC-DB-BLOCKING] handlers async + asyncio.to_thread para no
  bloquear el event loop con los roundtrips sync del pool.
"""

from typing import Any, Dict, Optional

import asyncio
import logging
import os

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from pydantic import BaseModel

from auth import get_verified_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["user-data"])


def _require_user(verified_user_id: Optional[str]) -> str:
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Autenticación requerida.")
    return verified_user_id


# ---------------------------------------------------------------------------
# Inventario (user_inventory + embed master_ingredients)
# ---------------------------------------------------------------------------

# Shape de embed idéntico al de PostgREST `select('*, master_ingredients(...)')`:
# cada row lleva un dict anidado bajo la key 'master_ingredients' (o None si
# no hay FK). Los consumers (Pantry/Dashboard/useRegeneratePlan) no cambian
# su lectura — solo el transporte.
_INVENTORY_SELECT = """
    SELECT
        ui.id,
        ui.user_id::text AS user_id,
        ui.ingredient_name,
        ui.quantity::float8 AS quantity,
        ui.unit,
        to_jsonb(ui.created_at)#>>'{}' AS created_at,
        to_jsonb(ui.updated_at)#>>'{}' AS updated_at,
        ui.master_ingredient_id::text AS master_ingredient_id,
        ui.source,
        ui.category,
        ui.brand,
        CASE WHEN mi.id IS NULL THEN NULL ELSE jsonb_build_object(
            'name', mi.name,
            'category', mi.category,
            'default_unit', mi.default_unit,
            'market_container', mi.market_container,
            'shelf_life_days', mi.shelf_life_days
        ) END AS master_ingredients
    FROM user_inventory ui
    LEFT JOIN master_ingredients mi ON mi.id = ui.master_ingredient_id
"""


def _fetch_inventory(user_id: str, only_positive: bool = True):
    from db import execute_sql_query
    where = "WHERE ui.user_id = %s"
    if only_positive:
        where += " AND ui.quantity > 0"
    return execute_sql_query(
        f"{_INVENTORY_SELECT} {where} ORDER BY ui.ingredient_name ASC",
        (user_id,),
        fetch_all=True,
    ) or []


@router.get("/inventory")
async def api_get_inventory(
    include_zero: bool = False,
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Inventario del usuario con embed master_ingredients (shape PostgREST).
    Reemplaza los SELECTs directos de Pantry.fetchData, Dashboard
    fetchLiveInventory/refetch/PDF/restock y useRegeneratePlan."""
    uid = _require_user(verified_user_id)
    items = await asyncio.to_thread(_fetch_inventory, uid, not include_zero)
    return {"items": items}


class InventoryItemBody(BaseModel):
    ingredient_name: str
    quantity: float
    unit: Optional[str] = None
    master_ingredient_id: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None


@router.post("/inventory/items", status_code=201)
async def api_add_inventory_item(
    body: InventoryItemBody = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    """INSERT plano (NO upsert — P3-PANTRY-ADD-UX-INSERT). En conflicto con
    UNIQUE (user_id, ingredient_name, unit) responde 409 — el frontend
    refetchea e incrementa el row existente (misma semántica que el manejo
    23505 legacy de handleAddNewItem/handleRestoreDepleted)."""
    uid = _require_user(verified_user_id)

    def _insert():
        import psycopg
        from db import execute_sql_query
        try:
            return execute_sql_query(
                """
                WITH ins AS (
                    INSERT INTO user_inventory
                        (user_id, ingredient_name, quantity, unit,
                         master_ingredient_id, source, category)
                    VALUES (%s, %s, %s, %s, %s,
                            COALESCE(%s, 'shopping_list'), %s)
                    RETURNING *
                )
                SELECT
                    ins.id,
                    ins.user_id::text AS user_id,
                    ins.ingredient_name,
                    ins.quantity::float8 AS quantity,
                    ins.unit,
                    to_jsonb(ins.created_at)#>>'{}' AS created_at,
                    to_jsonb(ins.updated_at)#>>'{}' AS updated_at,
                    ins.master_ingredient_id::text AS master_ingredient_id,
                    ins.source,
                    ins.category,
                    CASE WHEN mi.id IS NULL THEN NULL ELSE jsonb_build_object(
                        'name', mi.name,
                        'category', mi.category,
                        'default_unit', mi.default_unit,
                        'market_container', mi.market_container,
                        'shelf_life_days', mi.shelf_life_days
                    ) END AS master_ingredients
                FROM ins
                LEFT JOIN master_ingredients mi ON mi.id = ins.master_ingredient_id
                """,
                (uid, body.ingredient_name, body.quantity, body.unit,
                 body.master_ingredient_id, body.source, body.category),
                fetch_one=True,
            )
        except psycopg.errors.UniqueViolation:
            return "__duplicate__"

    row = await asyncio.to_thread(_insert)
    if row == "__duplicate__":
        raise HTTPException(
            status_code=409,
            detail="duplicate: ya existe un item con ese nombre y unidad.",
        )
    if not row:
        raise HTTPException(status_code=500, detail="INSERT no retornó fila.")
    return {"item": row}


class InventoryIncrementBody(BaseModel):
    item_id: int
    delta: float


@router.post("/inventory/increment")
async def api_increment_inventory(
    body: InventoryIncrementBody = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Incremento atómico de quantity. Reemplaza la RPC SECURITY DEFINER
    `increment_inventory_quantity` (usaba auth.uid() interno — sin contexto
    JWT en Neon). Misma semántica: UPDATE ... SET quantity = quantity + delta
    WHERE id AND user_id, RETURNING quantity. 404 si el row no es del usuario."""
    uid = _require_user(verified_user_id)

    def _inc():
        from db import execute_sql_write
        # [P1-NEON-DB-MIGRATION] GREATEST(0, ...) replica el clamp server-side
        # de la RPC SECURITY DEFINER `increment_inventory_quantity` (P2-4) que
        # este endpoint reemplaza. Sin él, dos tabs decrementando en paralelo
        # dejan quantity negativa (no hay CHECK constraint en user_inventory):
        # el row desaparece de GET /api/inventory (filtra quantity > 0) pero
        # sigue bloqueando el INSERT 409-dedup por el UNIQUE.
        return execute_sql_write(
            """
            UPDATE user_inventory
            SET quantity = GREATEST(0, quantity + %s::numeric), updated_at = NOW()
            WHERE id = %s AND user_id = %s
            RETURNING quantity::float8 AS quantity
            """,
            (body.delta, body.item_id, uid),
            returning=True,
        )

    rows = await asyncio.to_thread(_inc)
    if not rows:
        raise HTTPException(status_code=404, detail="Item no encontrado.")
    return {"quantity": rows[0]["quantity"]}


@router.delete("/inventory/items/{item_id}")
async def api_delete_inventory_item(
    item_id: int,
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Delete de un item. I2 explícito: el delete legacy de Pantry confiaba
    en RLS (sin .eq(user_id)); aquí el filtro es obligatorio."""
    uid = _require_user(verified_user_id)

    def _del():
        from db import execute_sql_write
        return execute_sql_write(
            "DELETE FROM user_inventory WHERE id = %s AND user_id = %s RETURNING id",
            (item_id, uid),
            returning=True,
        )

    rows = await asyncio.to_thread(_del)
    if not rows:
        raise HTTPException(status_code=404, detail="Item no encontrado.")
    return {"deleted": True}


@router.delete("/inventory/items")
async def api_delete_all_inventory(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Vaciar nevera completa ('Borrar Todos' de Pantry)."""
    uid = _require_user(verified_user_id)

    def _del_all():
        from db import execute_sql_write
        rows = execute_sql_write(
            "DELETE FROM user_inventory WHERE user_id = %s RETURNING id",
            (uid,),
            returning=True,
        )
        return len(rows or [])

    deleted = await asyncio.to_thread(_del_all)
    return {"deleted_count": deleted}


@router.get("/catalog")
async def api_get_catalog(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Catálogo master_ingredients (cuasi-inmutable, ~20KB). El frontend
    mantiene su cache singleton de 24h — este endpoint solo cambia el
    transporte. Auth requerida (paridad con el acceso RLS previo)."""
    _require_user(verified_user_id)

    def _catalog():
        from db import execute_sql_query
        return execute_sql_query(
            """
            SELECT id::text AS id, slug, name, category, aliases,
                   density_g_per_cup::float8 AS density_g_per_cup,
                   density_g_per_unit::float8 AS density_g_per_unit,
                   shelf_life_days,
                   price_per_lb::float8 AS price_per_lb,
                   price_per_unit::float8 AS price_per_unit,
                   market_container, container_weight_g::float8 AS container_weight_g,
                   available_sizes_g, default_unit
            FROM master_ingredients ORDER BY name ASC
            """,
            fetch_all=True,
        ) or []

    return {"items": await asyncio.to_thread(_catalog)}


# ---------------------------------------------------------------------------
# Perfil (user_profiles)
# ---------------------------------------------------------------------------

# Whitelist ESTRICTA de columnas escalares actualizables por el cliente.
# NUNCA añadir columnas de entitlement (plan_tier, subscription_status,
# subscription_end_date, paypal_subscription_id, ...) — el tier es
# server-derived desde PayPal (I-Billing-1, P0-BILLING-1); aceptar
# plan_tier del cliente reabriría el upgrade gratis via DevTools.
# Tooltip-anchor: P1-NEON-PROFILE-SCALAR-WHITELIST.
_PROFILE_SCALAR_WHITELIST = frozenset({"full_name"})


class ProfilePatchBody(BaseModel):
    health_profile: Optional[Dict[str, Any]] = None
    fields: Optional[Dict[str, Any]] = None


@router.get("/profile")
async def api_get_profile(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Perfil completo (incluye health_profile y el middleware de graceful
    degradation de get_user_profile). Reemplaza los .select('*').single()
    de fetchProfile/refreshProfileAndPlan."""
    uid = _require_user(verified_user_id)
    from db import get_user_profile
    profile = await asyncio.to_thread(get_user_profile, uid)
    if profile is None:
        raise HTTPException(status_code=404, detail="Perfil no encontrado.")
    return {"profile": profile}


@router.patch("/profile")
async def api_patch_profile(
    body: ProfilePatchBody = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Reemplaza la RPC `update_health_profile_merge` (merge jsonb ||) y el
    UPDATE escalar de updateUserProfile. El merge ocurre server-side en un
    solo UPDATE — misma garantía anti-race que la RPC (P1-FORM-9)."""
    uid = _require_user(verified_user_id)

    fields = dict(body.fields or {})
    rejected = sorted(set(fields) - _PROFILE_SCALAR_WHITELIST)
    if rejected:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Campos no permitidos: {rejected}. Permitidos: "
                f"{sorted(_PROFILE_SCALAR_WHITELIST)}. Las columnas de "
                "entitlement son server-derived (I-Billing-1)."
            ),
        )
    if not body.health_profile and not fields:
        raise HTTPException(status_code=400, detail="Nada que actualizar.")

    def _patch():
        from db import execute_sql_write
        from psycopg.types.json import Jsonb
        updated = False
        if body.health_profile:
            rows = execute_sql_write(
                """
                UPDATE user_profiles
                SET health_profile = COALESCE(health_profile, '{}'::jsonb) || %s::jsonb
                WHERE id = %s RETURNING id
                """,
                (Jsonb(body.health_profile), uid),
                returning=True,
            )
            updated = updated or bool(rows)
        if fields:
            # Whitelist ya validada — SET dinámico seguro (keys controladas).
            set_clause = ", ".join(f"{k} = %s" for k in fields)
            rows = execute_sql_write(
                f"UPDATE user_profiles SET {set_clause} WHERE id = %s RETURNING id",
                (*fields.values(), uid),
                returning=True,
            )
            updated = updated or bool(rows)
        return updated

    updated = await asyncio.to_thread(_patch)
    if not updated:
        raise HTTPException(status_code=404, detail="Perfil no encontrado.")
    return {"success": True}


# ---------------------------------------------------------------------------
# Planes (lecturas que el frontend hacía directo a meal_plans)
# ---------------------------------------------------------------------------

@router.get("/plans-data/latest")
async def api_get_latest_plan(
    include_plan_data: bool = True,
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Último plan del usuario. Reemplaza los SELECT .order(created_at desc)
    .limit(1) de restoreSessionData / regenerateSingleMeal / restorePlan /
    _recalcShoppingListAfterPantryChange. `include_plan_data=false` para los
    callers que solo resuelven el plan_id activo (payload liviano)."""
    uid = _require_user(verified_user_id)

    def _latest():
        from db import execute_sql_query
        cols = "id::text AS id, to_jsonb(created_at)#>>'{}' AS created_at, " \
               "to_jsonb(updated_at)#>>'{}' AS updated_at"
        if include_plan_data:
            cols += ", plan_data"
        return execute_sql_query(
            f"SELECT {cols} FROM meal_plans WHERE user_id = %s "
            "ORDER BY created_at DESC LIMIT 1",
            (uid,),
            fetch_one=True,
        )

    row = await asyncio.to_thread(_latest)
    return {"plan": row}  # null si no hay planes — el frontend ya maneja ausencia


@router.get("/plans-data/{plan_id}")
async def api_get_plan_data(
    plan_id: str,
    verified_user_id: str = Depends(get_verified_user_id),
):
    """plan_data de un plan específico con ownership (I2). Reemplaza el
    lazy-load del modal de History y el sync pre-PDF del Dashboard
    (P3-PDF-ALWAYS-SYNC)."""
    uid = _require_user(verified_user_id)

    def _by_id():
        from db import execute_sql_query
        return execute_sql_query(
            "SELECT id::text AS id, plan_data, "
            "to_jsonb(updated_at)#>>'{}' AS updated_at "
            "FROM meal_plans WHERE id = %s AND user_id = %s",
            (plan_id, uid),
            fetch_one=True,
        )

    row = await asyncio.to_thread(_by_id)
    if not row:
        raise HTTPException(status_code=404, detail="Plan no encontrado.")
    return {"plan": row}


# ---------------------------------------------------------------------------
# Súper Personalización (health_profile.super_personalization)
# [P1-SUPERPERSONALIZATION-1 · 2026-06-19]
# ---------------------------------------------------------------------------
# Panel opt-in (Ajustes) con dimensiones de PREFERENCIA que el wizard no captura:
# gustos positivos, cocina/cultura, restricción religiosa, equipo de cocina,
# perfil de sabor, nivel de cocina + un texto libre. Persiste como sub-key JSONB
# de health_profile (sin migración). Se inyecta a plan-gen y chat vía
# `build_super_personalization_context`. ADITIVO: NO toca alergias/condiciones/
# medicamentos (esas viven en sus campos estructurados validados).

_SUPERPERS_RELIGION_VALUES = {"", "none", "halal", "kosher", "sin_cerdo", "sin_res", "sin_mariscos", "sin_alcohol", "otra"}
_SUPERPERS_SKILL_VALUES = {"", "principiante", "intermedio", "avanzado"}
_SUPERPERS_MAX_OTHER = 80  # restricción cultural/religiosa "otra" (texto libre acotado)
_SUPERPERS_FLAVOR_KEYS = ("picante", "dulce", "salado")
_SUPERPERS_FLAVOR_LEVELS = {"", "bajo", "medio", "alto"}
_SUPERPERS_LIST_KEYS = ("foodLikes", "cuisines", "kitchenEquipment")
_SUPERPERS_MAX_LIST = 30
_SUPERPERS_MAX_ITEM_LEN = 60
_SUPERPERS_MAX_FREETEXT = 1500

# [P1-SUPERPERSONALIZATION-1 · Fase 3 · 2026-06-19] Kill-switch del enriquecimiento
# del RAG: al guardar con un freeText NUEVO, se extraen facts (DeepSeek-flash) →
# user_facts (Cohere embed) en background, para que el RAG los recupere en
# plan-gen Y chat automáticamente. Default ON; flip a "false" en el .env del VPS
# para apagar sin redeploy (cuesta ~1 call LLM + 1 embedding por guardado con
# texto cambiado). Reusa el MISMO pipeline que el chat (dedup/contradicción/lock).
_SUPERPERS_EXTRACT_FACTS = os.getenv(
    "MEALFIT_SUPERPERS_EXTRACT_FACTS", "true"
).strip().lower() in ("1", "true", "yes", "on")


def _clean_super_personalization(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Valida + normaliza el payload (defensivo: listas acotadas, enums
    validados, freeText capado). 422 ante shapes inválidos."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Payload inválido.")
    out: Dict[str, Any] = {}
    for key in _SUPERPERS_LIST_KEYS:
        raw = payload.get(key) or []
        if not isinstance(raw, list):
            raise HTTPException(status_code=422, detail=f"'{key}' debe ser una lista.")
        items = []
        seen = set()
        for x in raw[:_SUPERPERS_MAX_LIST]:
            s = str(x).strip()[:_SUPERPERS_MAX_ITEM_LEN]
            if s and s.lower() not in seen:
                seen.add(s.lower())
                items.append(s)
        out[key] = items
    rel = str(payload.get("religiousRestriction") or "").strip().lower()
    if rel not in _SUPERPERS_RELIGION_VALUES:
        raise HTTPException(status_code=422, detail="religiousRestriction inválida.")
    out["religiousRestriction"] = "" if rel == "none" else rel
    # Texto libre de la restricción "otra" (solo relevante si rel == "otra").
    other = str(payload.get("religiousRestrictionOther") or "").strip()[:_SUPERPERS_MAX_OTHER]
    out["religiousRestrictionOther"] = other if out["religiousRestriction"] == "otra" else ""
    skill = str(payload.get("cookingSkill") or "").strip().lower()
    if skill not in _SUPERPERS_SKILL_VALUES:
        raise HTTPException(status_code=422, detail="cookingSkill inválido.")
    out["cookingSkill"] = skill
    flavor_in = payload.get("flavorProfile") or {}
    flavor_out: Dict[str, str] = {}
    if not isinstance(flavor_in, dict):
        raise HTTPException(status_code=422, detail="flavorProfile debe ser un objeto.")
    for k in _SUPERPERS_FLAVOR_KEYS:
        lvl = str(flavor_in.get(k) or "").strip().lower()
        if lvl not in _SUPERPERS_FLAVOR_LEVELS:
            raise HTTPException(status_code=422, detail=f"flavorProfile.{k} inválido.")
        if lvl:
            flavor_out[k] = lvl
    out["flavorProfile"] = flavor_out
    free = payload.get("freeText") or ""
    if not isinstance(free, str):
        raise HTTPException(status_code=422, detail="freeText debe ser texto.")
    out["freeText"] = free.strip()[:_SUPERPERS_MAX_FREETEXT]
    return out


@router.get("/user/preferences/super-personalization")
async def api_get_super_personalization(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Devuelve el payload de súper personalización del usuario (o {} si no lo
    ha llenado). Read-only, cero costo LLM (misma exención que el resto de
    /user/preferences)."""
    uid = _require_user(verified_user_id)
    from db import get_user_profile

    profile = await asyncio.to_thread(get_user_profile, uid)
    hp = (profile or {}).get("health_profile") or {}
    sp = hp.get("super_personalization") if isinstance(hp, dict) else None
    return {"super_personalization": sp if isinstance(sp, dict) else {}}


class SuperPersonalizationBody(BaseModel):
    foodLikes: Optional[list] = None
    cuisines: Optional[list] = None
    kitchenEquipment: Optional[list] = None
    religiousRestriction: Optional[str] = None
    religiousRestrictionOther: Optional[str] = None
    cookingSkill: Optional[str] = None
    flavorProfile: Optional[Dict[str, Any]] = None
    freeText: Optional[str] = None


@router.put("/user/preferences/super-personalization")
async def api_put_super_personalization(
    background_tasks: BackgroundTasks,
    body: SuperPersonalizationBody = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Persiste el payload validado en health_profile.super_personalization vía
    update_user_health_profile_atomic (SELECT…FOR UPDATE + callback, I7 — sin
    lost-update bajo concurrencia). Filtra por user_id autenticado (I2).

    [Fase 3] Si el `freeText` CAMBIÓ, extrae facts del texto en background →
    user_facts (Cohere embed) para que el RAG los recupere en plan-gen Y chat."""
    uid = _require_user(verified_user_id)
    cleaned = _clean_super_personalization(body.model_dump())

    from datetime import datetime, timezone
    cleaned["updatedAt"] = datetime.now(timezone.utc).isoformat()

    from db import update_user_health_profile_atomic

    # [Fase 3] Detecta si el freeText cambió (vs el guardado previo) para NO
    # re-extraer facts en cada guardado. Se computa dentro del mutator (que ve
    # el hp actual bajo el lock) y se lee fuera vía closure.
    _state = {"freetext_changed": False}

    def _mutator(hp):
        if not isinstance(hp, dict):
            hp = {}
        prev = hp.get("super_personalization")
        prev_free = prev.get("freeText") if isinstance(prev, dict) else ""
        new_free = cleaned.get("freeText") or ""
        _state["freetext_changed"] = bool(new_free) and new_free != (prev_free or "")
        hp["super_personalization"] = cleaned
        return hp

    new_hp = await asyncio.to_thread(update_user_health_profile_atomic, uid, _mutator)
    if new_hp is None:
        raise HTTPException(status_code=404, detail="Perfil no encontrado.")

    # [Fase 3] Enriquecer el RAG: extraer facts del texto libre en background.
    # `async_extract_and_save_facts` es síncrona (router lite + lock + LLM +
    # embed + dedup); BackgroundTasks la corre en el threadpool TRAS enviar la
    # respuesta → no bloquea el PUT. Reusa el MISMO pipeline que el chat, así
    # que los facts heredan dedup/contradicción/embedding asimétrico.
    if _SUPERPERS_EXTRACT_FACTS and _state["freetext_changed"]:
        try:
            from fact_extractor import async_extract_and_save_facts
            background_tasks.add_task(async_extract_and_save_facts, uid, cleaned["freeText"])
            logger.info(
                f"[P1-SUPERPERSONALIZATION-1/Fase3] Extracción de facts encolada "
                f"para user {uid} (freeText {len(cleaned['freeText'])} chars)."
            )
        except Exception as _fx_err:  # noqa: BLE001 — best-effort, no rompe el guardado
            logger.warning(
                f"[P1-SUPERPERSONALIZATION-1/Fase3] No se pudo encolar extracción "
                f"de facts: {_fx_err}"
            )

    return {"super_personalization": cleaned}


# ---------------------------------------------------------------------------
# Perfil Clínico Avanzado (health_profile.clinical_profile)
# [P1-CLINICAL-PANEL · 2026-07-03]
# ---------------------------------------------------------------------------
# Panel opt-in (Ajustes) con las dimensiones clínicas que el wizard NO captura
# (P1 restantes del audit clínico 2026-07-03): laboratorios recientes, historia
# ponderal, síntomas digestivos y entrenamiento (tipo/hora/frecuencia) + texto
# libre. Persiste como sub-key JSONB de health_profile (sin migración, patrón
# P1-SUPERPERSONALIZATION-1). Se inyecta a plan-gen (planner + day-gen vía
# clinical_directives) y al revisor médico vía `build_clinical_profile_context`
# (prompts/plan_generator.py). ADITIVO: NO reemplaza condiciones/alergias/
# medicamentos del wizard — los labs generan GUÍA de prompt (flags honestos
# "compatible con X, requiere confirmación profesional"), nunca diagnóstico.

_CLINPROF_LAB_RANGES: Dict[str, tuple] = {
    # key → (min, max) permisivo-pero-sano; fuera de rango = 422 (typo probable).
    "glucosa_ayunas":   (40.0, 500.0),    # mg/dL
    "hba1c":            (3.0, 15.0),      # %
    "colesterol_total": (80.0, 500.0),    # mg/dL
    "ldl":              (30.0, 400.0),    # mg/dL
    "hdl":              (10.0, 150.0),    # mg/dL
    "trigliceridos":    (30.0, 2000.0),   # mg/dL
    "creatinina":       (0.2, 15.0),      # mg/dL
    "tfg":              (5.0, 150.0),     # mL/min/1.73m²
    "tsh":              (0.01, 100.0),    # µUI/mL
    "acido_urico":      (1.0, 15.0),      # mg/dL
    "hemoglobina":      (5.0, 22.0),      # g/dL
    "vitamina_d":       (4.0, 150.0),     # ng/mL
}
_CLINPROF_GI_VALUES = {"estrenimiento", "diarrea", "reflujo", "distension", "ninguno"}
_CLINPROF_TRAINING_TYPES = {"", "fuerza", "cardio", "mixto", "crossfit", "calistenia", "deporte"}
_CLINPROF_TRAINING_TIMES = {"", "manana", "mediodia", "tarde", "noche"}
_CLINPROF_WEIGHT_UNITS = {"", "lb", "kg"}
_CLINPROF_WEIGHT_RANGE = (20.0, 700.0)  # genérico lb/kg — solo anti-typo
_CLINPROF_MAX_FREETEXT = 1500
_CLINPROF_MAX_LABS_DATE = 20


def _clinprof_num(raw: Any, key: str, lo: float, hi: float) -> Optional[float]:
    """Parsea un numérico opcional ('' / None → None). Coma decimal es-DO
    normalizada. Fuera de rango → 422 accionable con el nombre del campo."""
    if raw is None or raw == "":
        return None
    try:
        v = float(str(raw).replace(",", "."))
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail=f"'{key}' debe ser numérico.")
    if not (lo <= v <= hi):
        raise HTTPException(
            status_code=422,
            detail=f"'{key}' fuera de rango plausible ({lo}-{hi}). ¿Typo?",
        )
    return v


def _clean_clinical_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Valida + normaliza el payload del panel clínico. 422 ante shapes/rangos
    inválidos. Campos vacíos se OMITEN (el builder de contexto es no-op sin
    datos accionables)."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Payload inválido.")
    out: Dict[str, Any] = {}

    labs_in = payload.get("labs") or {}
    if not isinstance(labs_in, dict):
        raise HTTPException(status_code=422, detail="'labs' debe ser un objeto.")
    labs_out: Dict[str, Any] = {}
    for key, (lo, hi) in _CLINPROF_LAB_RANGES.items():
        v = _clinprof_num(labs_in.get(key), key, lo, hi)
        if v is not None:
            labs_out[key] = v
    labs_date = str(labs_in.get("labsDate") or "").strip()[:_CLINPROF_MAX_LABS_DATE]
    if labs_date:
        labs_out["labsDate"] = labs_date
    out["labs"] = labs_out

    wh_in = payload.get("weightHistory") or {}
    if not isinstance(wh_in, dict):
        raise HTTPException(status_code=422, detail="'weightHistory' debe ser un objeto.")
    wh_out: Dict[str, Any] = {}
    unit = str(wh_in.get("unit") or "").strip().lower()
    if unit not in _CLINPROF_WEIGHT_UNITS:
        raise HTTPException(status_code=422, detail="weightHistory.unit inválida (lb|kg).")
    _wlo, _whi = _CLINPROF_WEIGHT_RANGE
    for wkey in ("maxWeight", "minWeight", "weight6mAgo"):
        v = _clinprof_num(wh_in.get(wkey), f"weightHistory.{wkey}", _wlo, _whi)
        if v is not None:
            wh_out[wkey] = v
    if wh_out and not unit:
        raise HTTPException(status_code=422, detail="weightHistory.unit requerida si das pesos.")
    if unit:
        wh_out["unit"] = unit
    wh_out["unintentionalLoss"] = bool(wh_in.get("unintentionalLoss"))
    out["weightHistory"] = wh_out

    gi_in = payload.get("giSymptoms") or []
    if not isinstance(gi_in, list):
        raise HTTPException(status_code=422, detail="'giSymptoms' debe ser una lista.")
    gi_out = []
    for x in gi_in[:8]:
        s = str(x).strip().lower()
        if s and s not in _CLINPROF_GI_VALUES:
            raise HTTPException(status_code=422, detail=f"giSymptoms '{s}' inválido.")
        if s and s not in gi_out:
            gi_out.append(s)
    # Sentinel 'ninguno' exclusivo (misma regla que los multi-select del wizard).
    if "ninguno" in gi_out and len(gi_out) > 1:
        gi_out = [s for s in gi_out if s != "ninguno"]
    out["giSymptoms"] = gi_out

    tr_in = payload.get("training") or {}
    if not isinstance(tr_in, dict):
        raise HTTPException(status_code=422, detail="'training' debe ser un objeto.")
    tr_type = str(tr_in.get("type") or "").strip().lower()
    if tr_type not in _CLINPROF_TRAINING_TYPES:
        raise HTTPException(status_code=422, detail="training.type inválido.")
    tr_time = str(tr_in.get("timeOfDay") or "").strip().lower()
    if tr_time not in _CLINPROF_TRAINING_TIMES:
        raise HTTPException(status_code=422, detail="training.timeOfDay inválido.")
    days_raw = tr_in.get("daysPerWeek")
    tr_days = 0
    if days_raw not in (None, ""):
        try:
            tr_days = int(float(str(days_raw)))
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail="training.daysPerWeek debe ser 0-7.")
        if not (0 <= tr_days <= 7):
            raise HTTPException(status_code=422, detail="training.daysPerWeek debe ser 0-7.")
    out["training"] = {"type": tr_type, "timeOfDay": tr_time, "daysPerWeek": tr_days}

    free = payload.get("freeText") or ""
    if not isinstance(free, str):
        raise HTTPException(status_code=422, detail="freeText debe ser texto.")
    out["freeText"] = free.strip()[:_CLINPROF_MAX_FREETEXT]
    return out


@router.get("/user/preferences/clinical-profile")
async def api_get_clinical_profile(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Devuelve el perfil clínico avanzado del usuario (o {} si no lo llenó).
    Read-only, cero costo LLM (misma exención que /user/preferences)."""
    uid = _require_user(verified_user_id)
    from db import get_user_profile

    profile = await asyncio.to_thread(get_user_profile, uid)
    hp = (profile or {}).get("health_profile") or {}
    cp = hp.get("clinical_profile") if isinstance(hp, dict) else None
    return {"clinical_profile": cp if isinstance(cp, dict) else {}}


class ClinicalProfileBody(BaseModel):
    labs: Optional[Dict[str, Any]] = None
    weightHistory: Optional[Dict[str, Any]] = None
    giSymptoms: Optional[list] = None
    training: Optional[Dict[str, Any]] = None
    freeText: Optional[str] = None


@router.put("/user/preferences/clinical-profile")
async def api_put_clinical_profile(
    background_tasks: BackgroundTasks,
    body: ClinicalProfileBody = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Persiste el payload validado en health_profile.clinical_profile vía
    update_user_health_profile_atomic (FOR UPDATE + callback, I7). Filtra por
    user_id autenticado (I2). freeText nuevo → extracción de facts en
    background (mismo pipeline/knob que súper personalización)."""
    uid = _require_user(verified_user_id)
    cleaned = _clean_clinical_profile(body.model_dump())

    from datetime import datetime, timezone
    cleaned["updatedAt"] = datetime.now(timezone.utc).isoformat()

    from db import update_user_health_profile_atomic

    _state = {"freetext_changed": False}

    def _mutator(hp):
        if not isinstance(hp, dict):
            hp = {}
        prev = hp.get("clinical_profile")
        prev_free = prev.get("freeText") if isinstance(prev, dict) else ""
        new_free = cleaned.get("freeText") or ""
        _state["freetext_changed"] = bool(new_free) and new_free != (prev_free or "")
        hp["clinical_profile"] = cleaned
        return hp

    new_hp = await asyncio.to_thread(update_user_health_profile_atomic, uid, _mutator)
    if new_hp is None:
        raise HTTPException(status_code=404, detail="Perfil no encontrado.")

    if _SUPERPERS_EXTRACT_FACTS and _state["freetext_changed"]:
        try:
            from fact_extractor import async_extract_and_save_facts
            background_tasks.add_task(async_extract_and_save_facts, uid, cleaned["freeText"])
            logger.info(
                f"[P1-CLINICAL-PANEL] Extracción de facts encolada para user {uid} "
                f"(freeText clínico {len(cleaned['freeText'])} chars)."
            )
        except Exception as _fx_err:  # noqa: BLE001 — best-effort, no rompe el guardado
            logger.warning(f"[P1-CLINICAL-PANEL] No se pudo encolar extracción de facts: {_fx_err}")

    return {"clinical_profile": cleaned}
