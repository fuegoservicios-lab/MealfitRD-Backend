"""[P1-SUPERMARKET-DB · 2026-07-02] Supermercado RD artificial.

Endpoints de la base de datos `supermarket_products` (Neon): el "supermercado
dominicano" de MealfitRD. Cada fila es UNA presentación comprable de un alimento
verificado (alimento + marca opcional + presentación + porción + duración +
precio RD$). Se navega públicamente desde el landing (/supermercado) y se edita
desde la misma página con el gate admin (Bearer CRON_SECRET).

Contrato:
- GET  /api/supermarket/products      → público, read-only, RateLimiter per-IP
  (60/60s). Solo filas `active` salvo `include_inactive=1` con token admin.
  NO usa `verify_api_quota` (cero costo LLM, página pública de marketing —
  misma razón que la historial-quota-exemption de CLAUDE.md).
- POST /api/supermarket/products      → admin (`_verify_admin_token`).
- PATCH /api/supermarket/products/{id}  → admin.
- DELETE /api/supermarket/products/{id} → admin (hard delete; para "ocultar"
  preferir PATCH active=false).

Seguridad (simétrica a I6): el frontend JAMÁS escribe directo a la tabla —
todas las mutaciones pasan por aquí con token admin verificado constant-time.
Tipos para JSON: uuid→::text, numeric→::float8, timestamptz→to_jsonb(...)#>>'{}'
(convención de routers/user_data.py).
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from db import execute_sql_query, execute_sql_write
from rate_limiter import RateLimiter
from routers.plans import _check_admin_rate_limit, _verify_admin_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/supermarket", tags=["supermarket"])

# Público per-IP (el landing no exige sesión). 60/60s es generoso para navegación
# humana y frena scraping/hammering básico.
_PUBLIC_LIST_LIMITER = RateLimiter(max_calls=60, period_seconds=60)

_MAX_LIMIT = 1000

# Columnas expuestas/mutables — whitelist explícita (nunca interpolar keys del cliente).
_SELECT_COLS = """
    id::text AS id,
    food_name,
    brand,
    presentation,
    portion_label,
    duration_label,
    price_rd::float8 AS price_rd,
    notes,
    category,
    master_food_name,
    image_url,
    description,
    is_verified,
    active,
    to_jsonb(created_at)#>>'{}' AS created_at,
    to_jsonb(updated_at)#>>'{}' AS updated_at
"""

_MUTABLE_FIELDS = (
    "food_name", "brand", "presentation", "portion_label", "duration_label",
    "price_rd", "notes", "category", "master_food_name", "image_url",
    "description", "is_verified", "active",
)


class SupermarketProductIn(BaseModel):
    food_name: str = Field(min_length=1, max_length=120)
    brand: Optional[str] = Field(default=None, max_length=120)
    presentation: Optional[str] = Field(default=None, max_length=120)
    portion_label: Optional[str] = Field(default=None, max_length=60)
    duration_label: Optional[str] = Field(default=None, max_length=60)
    price_rd: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    notes: Optional[str] = Field(default=None, max_length=500)
    category: Optional[str] = Field(default=None, max_length=80)
    master_food_name: Optional[str] = Field(default=None, max_length=120)
    image_url: Optional[str] = Field(default=None, max_length=800)
    description: Optional[str] = Field(default=None, max_length=800)
    is_verified: bool = True
    active: bool = True


class SupermarketProductPatch(BaseModel):
    food_name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    brand: Optional[str] = Field(default=None, max_length=120)
    presentation: Optional[str] = Field(default=None, max_length=120)
    portion_label: Optional[str] = Field(default=None, max_length=60)
    duration_label: Optional[str] = Field(default=None, max_length=60)
    price_rd: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    notes: Optional[str] = Field(default=None, max_length=500)
    category: Optional[str] = Field(default=None, max_length=80)
    master_food_name: Optional[str] = Field(default=None, max_length=120)
    image_url: Optional[str] = Field(default=None, max_length=800)
    description: Optional[str] = Field(default=None, max_length=800)
    is_verified: Optional[bool] = None
    active: Optional[bool] = None


def _clean(value: Optional[str]) -> Optional[str]:
    """Trimea y colapsa strings vacíos a NULL (evita '' vs NULL en el unique index)."""
    if value is None:
        return None
    value = value.strip()
    return value or None


@router.get("/products")
async def api_supermarket_list(
    request: Request,
    q: Optional[str] = None,
    category: Optional[str] = None,
    include_inactive: bool = False,
    limit: int = 1000,
    offset: int = 0,
    _rl: Any = Depends(_PUBLIC_LIST_LIMITER),
):
    """Listado público del supermercado. `include_inactive=1` requiere token admin
    (los productos ocultos solo son visibles en modo edición)."""
    if include_inactive:
        _verify_admin_token(request.headers.get("authorization"))

    limit = max(1, min(int(limit), _MAX_LIMIT))
    offset = max(0, int(offset))

    where: List[str] = []
    params: List[Any] = []
    if not include_inactive:
        where.append("active")
    if _clean(q):
        where.append("(food_name ILIKE %s OR coalesce(brand,'') ILIKE %s OR coalesce(category,'') ILIKE %s)")
        like = f"%{q.strip()}%"
        params.extend([like, like, like])
    if _clean(category):
        where.append("category = %s")
        params.append(category.strip())

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    def _fetch() -> Dict[str, Any]:
        rows = execute_sql_query(
            f"""
            SELECT {_SELECT_COLS}
            FROM public.supermarket_products
            {where_sql}
            ORDER BY category NULLS LAST, lower(food_name), lower(coalesce(brand,'')), lower(coalesce(presentation,''))
            LIMIT %s OFFSET %s
            """,
            tuple(params) + (limit, offset),
            fetch_all=True,
        )
        total_row = execute_sql_query(
            f"SELECT count(*)::float8 AS total FROM public.supermarket_products {where_sql}",
            tuple(params),
            fetch_one=True,
        )
        cats = execute_sql_query(
            """
            SELECT category, count(*)::float8 AS n
            FROM public.supermarket_products
            WHERE active AND category IS NOT NULL
            GROUP BY category
            ORDER BY category
            """,
            fetch_all=True,
        )
        return {
            "products": rows or [],
            "total": int((total_row or {}).get("total") or 0),
            "categories": cats or [],
        }

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as exc:
        logger.error(f"❌ [P1-SUPERMARKET-DB] list falló: {exc}")
        raise HTTPException(status_code=500, detail="No se pudo cargar el supermercado.")


@router.post("/products")
async def api_supermarket_create(request: Request, body: SupermarketProductIn):
    """Crea un producto/variante. Admin only (Bearer CRON_SECRET)."""
    _verify_admin_token(request.headers.get("authorization"))
    _check_admin_rate_limit(request)

    def _insert():
        return execute_sql_write(
            f"""
            INSERT INTO public.supermarket_products
                (food_name, brand, presentation, portion_label, duration_label,
                 price_rd, notes, category, master_food_name, image_url,
                 description, is_verified, active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (lower(food_name), lower(coalesce(brand,'')), lower(coalesce(presentation,'')))
            DO NOTHING
            RETURNING {_SELECT_COLS}
            """,
            (
                _clean(body.food_name), _clean(body.brand), _clean(body.presentation),
                _clean(body.portion_label), _clean(body.duration_label), body.price_rd,
                _clean(body.notes), _clean(body.category), _clean(body.master_food_name),
                _clean(body.image_url), _clean(body.description),
                body.is_verified, body.active,
            ),
            returning=True,
        )

    try:
        rows = await asyncio.to_thread(_insert)
    except Exception as exc:
        logger.error(f"❌ [P1-SUPERMARKET-DB] create falló: {exc}")
        raise HTTPException(status_code=500, detail="No se pudo crear el producto.")
    if not rows:
        raise HTTPException(
            status_code=409,
            detail="Ya existe esa combinación de alimento + marca + presentación.",
        )
    return {"product": rows[0]}


@router.patch("/products/{product_id}")
async def api_supermarket_update(request: Request, product_id: str, body: SupermarketProductPatch):
    """Actualiza campos de un producto (parcial). Admin only."""
    _verify_admin_token(request.headers.get("authorization"))
    _check_admin_rate_limit(request)

    # Solo campos presentes en el payload (exclude_unset) y whitelisted.
    changes = {
        k: v for k, v in body.model_dump(exclude_unset=True).items()
        if k in _MUTABLE_FIELDS
    }
    if not changes:
        raise HTTPException(status_code=422, detail="Nada que actualizar.")

    sets: List[str] = []
    params: List[Any] = []
    for key, value in changes.items():
        sets.append(f"{key} = %s")
        params.append(_clean(value) if isinstance(value, str) else value)
    sets.append("updated_at = now()")
    params.append(product_id)

    def _update():
        return execute_sql_write(
            f"""
            UPDATE public.supermarket_products
            SET {', '.join(sets)}
            WHERE id = %s::uuid
            RETURNING {_SELECT_COLS}
            """,
            tuple(params),
            returning=True,
        )

    try:
        rows = await asyncio.to_thread(_update)
    except Exception as exc:
        # 23505 = colisión con el unique index de variante tras el rename.
        if "uq_supermarket_products_variant" in str(exc):
            raise HTTPException(
                status_code=409,
                detail="Ya existe esa combinación de alimento + marca + presentación.",
            )
        logger.error(f"❌ [P1-SUPERMARKET-DB] update falló: {exc}")
        raise HTTPException(status_code=500, detail="No se pudo actualizar el producto.")
    if not rows:
        raise HTTPException(status_code=404, detail="Producto no encontrado.")
    return {"product": rows[0]}


@router.delete("/products/{product_id}")
async def api_supermarket_delete(request: Request, product_id: str):
    """Elimina un producto (hard delete). Admin only. Para ocultar sin borrar,
    usar PATCH active=false."""
    _verify_admin_token(request.headers.get("authorization"))
    _check_admin_rate_limit(request)

    def _delete():
        return execute_sql_write(
            "DELETE FROM public.supermarket_products WHERE id = %s::uuid RETURNING id::text AS id",
            (product_id,),
            returning=True,
        )

    try:
        rows = await asyncio.to_thread(_delete)
    except Exception as exc:
        logger.error(f"❌ [P1-SUPERMARKET-DB] delete falló: {exc}")
        raise HTTPException(status_code=500, detail="No se pudo eliminar el producto.")
    if not rows:
        raise HTTPException(status_code=404, detail="Producto no encontrado.")
    return {"deleted": True, "id": rows[0]["id"]}
