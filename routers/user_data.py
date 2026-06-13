"""[P1-NEON-DB-MIGRATION · 2026-06-12] Endpoints de datos user-scoped que
reemplazan los accesos directos del frontend a Postgres via PostgREST
(supabase-js). Post-migración a Neon, el frontend NO tiene acceso a la DB:
PostgREST apunta al Postgres de SUPABASE (datos stale tras el cutover) —
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

from fastapi import APIRouter, Body, Depends, HTTPException
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
