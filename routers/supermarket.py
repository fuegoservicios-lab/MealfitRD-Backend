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

from auth import get_verified_user_id
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


# ── [P1-SUPERMARKET-MATCH · 2026-07-02] lista de compras → variantes del súper ──
# Dado el set de nombres de la `aggregated_shopping_list`, devuelve los alimentos
# del catálogo que calzan (con TODAS sus variantes de marca/presentación activas)
# para que el Dashboard muestre marcas y precios reales por ítem. Público sin
# paywall (cero costo LLM — misma razón que GET /products); RateLimiter propio.
# Matching insensible a acentos/mayúsculas contra food_name Y master_food_name
# (el link suave a master_ingredients), con fallback singular/plural y por
# prefijo ("arroz" → "Arroz blanco", "Arroz integral", …).
_MATCH_LIMITER = RateLimiter(max_calls=30, period_seconds=60)
_MATCH_MAX_NAMES = 200
_MATCH_MAX_FOODS_PER_NAME = 4


class SupermarketMatchIn(BaseModel):
    names: List[str] = Field(min_length=1, max_length=_MATCH_MAX_NAMES)


def _norm_food(value: Optional[str]) -> str:
    """minúsculas + sin acentos + espacios colapsados (simétrica al frontend)."""
    import unicodedata
    s = unicodedata.normalize("NFD", (value or "").strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.split())


def _singular(s: str) -> str:
    """Heurística ligera es-DO: 'zanahorias'→'zanahoria', 'coles'→'col'."""
    if len(s) > 4 and s.endswith("es"):
        return s[:-2]
    if len(s) > 3 and s.endswith("s"):
        return s[:-1]
    return s


@router.post("/match")
async def api_supermarket_match(body: SupermarketMatchIn, _rl: Any = Depends(_MATCH_LIMITER)):
    """Matching de nombres de la lista de compras contra el catálogo del súper."""

    def _match() -> Dict[str, Any]:
        rows = execute_sql_query(
            """
            SELECT id::text AS id, food_name, brand, presentation,
                   price_rd::float8 AS price_rd, category, is_verified
            FROM public.supermarket_products
            WHERE active
            ORDER BY lower(food_name), (brand IS NOT NULL), price_rd NULLS LAST
            """,
            fetch_all=True,
        ) or []

        # Índice food normalizado → {food_name, category, variants[]}. master_food_name
        # (si difiere) apunta al MISMO grupo — alias de resolución, no grupo aparte.
        foods: Dict[str, Dict[str, Any]] = {}
        alias: Dict[str, str] = {}
        for r in rows:
            key = _norm_food(r["food_name"])
            g = foods.setdefault(key, {"food_name": r["food_name"], "category": r.get("category"), "variants": []})
            g["variants"].append({
                "id": r["id"], "brand": r.get("brand"), "presentation": r.get("presentation"),
                "price_rd": r.get("price_rd"), "is_verified": bool(r.get("is_verified")),
            })

        master_rows = execute_sql_query(
            """
            SELECT DISTINCT master_food_name, food_name
            FROM public.supermarket_products
            WHERE active AND master_food_name IS NOT NULL
            """,
            fetch_all=True,
        ) or []
        for r in master_rows:
            mk, fk = _norm_food(r["master_food_name"]), _norm_food(r["food_name"])
            if mk and mk not in foods and fk in foods:
                alias[mk] = fk

        def _resolve(raw: str) -> List[Dict[str, Any]]:
            name = _norm_food(raw)
            if not name:
                return []
            candidates: List[str] = []
            for probe in (name, _singular(name)):
                if probe in foods and probe not in candidates:
                    candidates.append(probe)
                elif probe in alias and alias[probe] not in candidates:
                    candidates.append(alias[probe])
            if not candidates and len(name) >= 4:
                # Contención con límite de palabra, en ambas direcciones:
                #   food ⊇ nombre: "pechuga de pollo" → "Filete pechuga de pollo",
                #                  "arroz" → "Arroz blanco", "Arroz integral", …
                #   nombre ⊇ food: "filete de salmon fresco" → "Salmón".
                # El padding con espacios evita falsos positivos por substring
                # ("sal" NO matchea "salsa de tomate").
                scored = []
                for probe in dict.fromkeys((name, _singular(name))):
                    padded_probe = f" {probe} "
                    for k in foods:
                        padded_food = f" {k} "
                        if padded_probe in padded_food:
                            scored.append((0, k))
                        elif len(k) >= 4 and padded_food in padded_probe:
                            scored.append((1, k))
                    if scored:
                        break
                for _, k in sorted(scored):
                    if k not in candidates:
                        candidates.append(k)
            return [foods[k] for k in candidates[:_MATCH_MAX_FOODS_PER_NAME]]

        seen: set = set()
        matches: Dict[str, Any] = {}
        for raw in body.names:
            raw = (raw or "").strip()
            if not raw or raw.lower() in seen:
                continue
            seen.add(raw.lower())
            found = _resolve(raw)
            if found:
                matches[raw] = found
        return {"matches": matches, "catalog_size": len(rows)}

    try:
        return await asyncio.to_thread(_match)
    except Exception as exc:
        logger.error(f"❌ [P1-SUPERMARKET-MATCH] match falló: {exc}")
        raise HTTPException(status_code=500, detail="No se pudo consultar el supermercado.")


# ── [P1-SUPERMARKET-PREFS · 2026-07-02] marca preferida por usuario (fase 2) ──
# Tabla `user_brand_preferences` (migración p1_supermarket_prefs_2026_07_02.sql):
# una fila por (user_id, food_key normalizado) → producto del súper elegido.
# Auth con `get_verified_user_id` (guests usan localStorage en el cliente, sin
# persistencia server-side). NO usa `verify_api_quota` (cero costo LLM — misma
# razón que la historial-quota-exemption); anti-spam via RateLimiter propio.
# Invariante I2: toda query filtra `WHERE user_id = %s`.
_PREFS_LIMITER = RateLimiter(max_calls=40, period_seconds=60)


class BrandPreferenceIn(BaseModel):
    food_key: str = Field(min_length=1, max_length=120)
    # None = borrar la preferencia de ese alimento.
    product_id: Optional[str] = Field(default=None, max_length=64)


@router.get("/preferences")
async def api_get_brand_preferences(
    user_id: str = Depends(get_verified_user_id),
    _rl: Any = Depends(_PREFS_LIMITER),
):
    """Preferencias de marca del usuario autenticado, con el producto hidratado."""

    def _fetch() -> Dict[str, Any]:
        rows = execute_sql_query(
            """
            SELECT p.food_key,
                   sp.id::text AS product_id,
                   sp.food_name, sp.brand, sp.presentation,
                   sp.price_rd::float8 AS price_rd, sp.active
            FROM public.user_brand_preferences p
            JOIN public.supermarket_products sp ON sp.id = p.product_id
            WHERE p.user_id = %s
            ORDER BY p.food_key
            """,
            (user_id,),
            fetch_all=True,
        ) or []
        return {"preferences": {r["food_key"]: r for r in rows}}

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as exc:
        logger.error(f"❌ [P1-SUPERMARKET-PREFS] get falló: {exc}")
        raise HTTPException(status_code=500, detail="No se pudieron cargar tus preferencias.")


@router.put("/preferences")
async def api_put_brand_preference(
    body: BrandPreferenceIn,
    user_id: str = Depends(get_verified_user_id),
    _rl: Any = Depends(_PREFS_LIMITER),
):
    """Upsert (o borrado con product_id=null) de la marca preferida de UN alimento."""
    food_key = _norm_food(body.food_key)
    if not food_key:
        raise HTTPException(status_code=422, detail="food_key inválido.")

    def _write() -> Dict[str, Any]:
        if body.product_id is None:
            execute_sql_write(
                "DELETE FROM public.user_brand_preferences WHERE user_id = %s AND food_key = %s",
                (user_id, food_key),
            )
            return {"ok": True, "food_key": food_key, "product_id": None}
        # El producto debe existir y estar visible al público — un id inventado
        # o un producto oculto por la admin UI no puede quedar como preferencia.
        product = execute_sql_query(
            "SELECT id::text AS id FROM public.supermarket_products WHERE id = %s::uuid AND active",
            (body.product_id,),
            fetch_one=True,
        )
        if not product:
            raise HTTPException(status_code=404, detail="Producto no encontrado en el supermercado.")
        execute_sql_write(
            """
            INSERT INTO public.user_brand_preferences (user_id, food_key, product_id)
            VALUES (%s, %s, %s::uuid)
            ON CONFLICT (user_id, food_key)
            DO UPDATE SET product_id = EXCLUDED.product_id, updated_at = now()
            """,
            (user_id, food_key, body.product_id),
        )
        return {"ok": True, "food_key": food_key, "product_id": body.product_id}

    try:
        return await asyncio.to_thread(_write)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ [P1-SUPERMARKET-PREFS] put falló: {exc}")
        raise HTTPException(status_code=500, detail="No se pudo guardar tu preferencia.")


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
