"""[P1-SUPERMARKET-DB · 2026-07-02] Supermercado RD artificial.

Endpoints de la base de datos `supermarket_products` (Neon): el "supermercado
dominicano" de Bioboros. Cada fila es UNA presentación comprable de un alimento
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
import os
import threading
import time
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

# ─────────────────────────────────────────────────────────────────────────────
# [P1-SUPERMARKET-CATALOG-CACHE · 2026-08-05] Cache in-process del catalogo.
#
# `/match` cargaba la tabla ENTERA en cada llamada -dos SELECT completos, sin
# cache- y el Dashboard lo pide en cada refresco de pagina. Medido el 2026-08-05
# contra Neon: 1.739 filas activas (251 alimentos distintos), 236-382 ms solo de
# DB por llamada. El trabajo en Python es irrelevante al lado (el parser de
# presentaciones: 2 ms sobre las 597 filas que lo necesitan), asi que el tiempo
# que el usuario ve esperando "Buscando marcas en el supermercado..." es
# basicamente latencia de dos consultas que devuelven SIEMPRE lo mismo.
#
# El catalogo solo cambia cuando un admin lo edita, y esas tres rutas invalidan
# la cache explicitamente -asi un cambio se ve al instante y el TTL es solo una
# red de seguridad, no el mecanismo-.
#
# ⚠️ LA INVALIDACION ES POR PROCESO. Hoy es completa porque el servicio corre con
# `uvicorn --workers 1` (verificado en el systemd del VPS el 2026-08-05). Si algun
# dia se sube ese numero, un PATCH atendido por el worker A NO limpiaria la cache
# del worker B, que serviria precios viejos hasta que expire su TTL (<=5 min por
# defecto). Si eso llega a pasar: o se baja el TTL, o la invalidacion pasa a un
# canal compartido (Redis pub/sub o una fila en `app_kv_store` que los workers
# consulten). No es hipotetico-lejano: subir workers es lo primero que se toca
# cuando llega trafico.
# ─────────────────────────────────────────────────────────────────────────────
def _catalog_cache_ttl_s() -> int:
    """TTL en segundos. Knob `MEALFIT_SUPERMARKET_CATALOG_CACHE_TTL_S`; 0 = sin
    cache (rollback sin redeploy). Clamp [0, 3600]."""
    try:
        v = int(os.environ.get("MEALFIT_SUPERMARKET_CATALOG_CACHE_TTL_S", "300"))
    except (TypeError, ValueError):
        v = 300
    return max(0, min(3600, v))


# [P2-BACKEND-SUPERMARKET-CACHE · 2026-08-14] `gen` es un contador de generacion.
#
# EL DEFECTO QUE CIERRA. Los tres handlers admin invalidan la cache ANTES de
# ejecutar la escritura:  `_invalidate_catalog_cache()` … `await
# asyncio.to_thread(_insert)`. Ese `await` es un punto de cesion GARANTIZADO, asi
# que `/match` puede correr entera en medio: encuentra la cache vacia, relee las
# filas PRE-escritura y las repuebla con `at = time.time()` fresco. Resultado:
# hasta 5 minutos (el TTL) de precios obsoletos alimentando el costeo de marca
# del Dashboard y de la Nevera, justo despues de que un admin los corrigiera.
#
# ⚠️ Mover la invalidacion a DESPUES de la escritura NO cierra la carrera: deja
# exactamente la misma ventana, solo que desplazada. Lo que la cierra es que el
# rellenador compruebe si la generacion cambio mientras el leia; si cambio, tira
# lo que trajo en vez de publicarlo.
_CATALOG_CACHE: Dict[str, Any] = {"at": 0.0, "rows": None, "master": None, "gen": 0}
_CATALOG_LOCK = threading.Lock()


def _catalog_generation() -> int:
    """La generacion vigente. Se captura ANTES de leer la DB."""
    with _CATALOG_LOCK:
        return int(_CATALOG_CACHE.get("gen") or 0)


def _invalidate_catalog_cache() -> None:
    """Tras cualquier mutacion admin. Sin esto un precio editado tardaria hasta
    el TTL en verse, y el editor vive en la MISMA pagina que lo consume."""
    with _CATALOG_LOCK:
        _CATALOG_CACHE["rows"] = None
        _CATALOG_CACHE["master"] = None
        _CATALOG_CACHE["at"] = 0.0
        _CATALOG_CACHE["gen"] = int(_CATALOG_CACHE.get("gen") or 0) + 1


def _publish_catalog_cache(rows, master_rows, gen_al_empezar: int) -> bool:
    """Publica el relleno SOLO si nadie invalido mientras se leia la DB.

    Devuelve False cuando descarta: quien lea eso sabra que su lectura era de
    antes de una escritura y que la siguiente peticion volvera a la DB.
    """
    with _CATALOG_LOCK:
        if int(_CATALOG_CACHE.get("gen") or 0) != gen_al_empezar:
            return False
        _CATALOG_CACHE["rows"] = rows
        _CATALOG_CACHE["master"] = master_rows
        _CATALOG_CACHE["at"] = time.time()
        return True


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
    size_grams::float8 AS size_grams,
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

# [P2-BRANDPREF-SIZE-COLUMN · 2026-07-02] `size_grams` = tamaño EXPLÍCITO del envase en gramos
# (líquidos ≈ ml). Cierra el fail-open del parser de presentaciones: la "L" suelta es ambigua
# (libra/litro en el PDF) → esos productos PERDÍAN el overlay de costeo de marca preferida
# (P1-SUPERMARKET-COSTING). Con size_grams poblado (admin UI), el costeo lo usa DIRECTO y el
# parser queda como fallback. Migración: p2_supermarket_size_grams_2026_07_02.sql (ambos dirs).
_MUTABLE_FIELDS = (
    "food_name", "brand", "presentation", "portion_label", "duration_label",
    "price_rd", "size_grams", "notes", "category", "master_food_name", "image_url",
    "description", "is_verified", "active",
)


class SupermarketProductIn(BaseModel):
    food_name: str = Field(min_length=1, max_length=120)
    brand: Optional[str] = Field(default=None, max_length=120)
    presentation: Optional[str] = Field(default=None, max_length=120)
    portion_label: Optional[str] = Field(default=None, max_length=60)
    duration_label: Optional[str] = Field(default=None, max_length=60)
    price_rd: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    size_grams: Optional[float] = Field(default=None, gt=0, le=50_000)  # [P2-BRANDPREF-SIZE-COLUMN]
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
    size_grams: Optional[float] = Field(default=None, gt=0, le=50_000)  # [P2-BRANDPREF-SIZE-COLUMN]
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
        # [P1-SUPERMARKET-CATALOG-CACHE · 2026-08-05] Las dos consultas del catalogo
        # se sirven de cache cuando esta caliente. Ver el bloque del helper arriba.
        _ttl = _catalog_cache_ttl_s()
        with _CATALOG_LOCK:
            _hot = (
                _ttl > 0
                and _CATALOG_CACHE["rows"] is not None
                and (time.time() - float(_CATALOG_CACHE["at"] or 0)) < _ttl
            )
            if _hot:
                rows = _CATALOG_CACHE["rows"]
                master_rows_cached = _CATALOG_CACHE["master"]
            else:
                rows = None
                master_rows_cached = None

        # [P2-BACKEND-SUPERMARKET-CACHE · 2026-08-14] La generacion se captura AQUI,
        # antes de tocar la DB. Si una mutacion admin la bumpea mientras leemos, el
        # relleno se descarta al final en vez de publicar filas pre-escritura.
        _gen_al_empezar = _catalog_generation()

        if rows is None:
            rows = execute_sql_query(
                """
            SELECT id::text AS id, food_name, brand, presentation,
                   price_rd::float8 AS price_rd,
                   size_grams::float8 AS size_grams,
                   category, is_verified
            FROM public.supermarket_products
            WHERE active
            -- [P1-BRAND-BUDGET-COHERENCE · 2026-07-29] price_rd ANTES del tiebreak de
            -- marca: con `(brand IS NOT NULL)` primero, TODO genérico ordenaba antes
            -- que TODO branded sin importar precio. Para ítems sin `package_grams`
            -- (presentación no parseable, ~21% del catálogo) el frontend no puede
            -- re-ordenar (sizeFilteredVariants/stableSortedVariants requieren targetG)
            -- y usa este array crudo — "N opciones · desde RD$X" podía mostrar el
            -- genérico como "desde" aunque una marca fuera más barata (test:
            -- test_p1_brand_budget_coherence_match_order.py).
            ORDER BY lower(food_name), price_rd NULLS LAST, (brand IS NOT NULL)
            """,
                fetch_all=True,
            ) or []

        # [P1-BRAND-SIZE-FILTER · 2026-07-06] `size_g` efectivo por variante:
        # size_grams (admin UI) autoritativo; fallback al parser SSOT del costeo
        # (`_parse_presentation_grams`, misma lógica que el overlay de marcas).
        # El frontend lo usa para filtrar variantes al tamaño del envase que la
        # lista de compras ya eligió (`package_grams` del ítem).
        from shopping_calculator import _parse_presentation_grams

        def _size_g(r: Dict[str, Any]) -> Optional[float]:
            try:
                g = float(r.get("size_grams") or 0) or None
            except (TypeError, ValueError):
                g = None
            if g is not None and not (1.0 <= g <= 50000.0):
                g = None
            if g is None:
                g = _parse_presentation_grams(r.get("presentation"))
            return g

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
                "size_g": _size_g(r),
            })

        if master_rows_cached is not None:
            master_rows = master_rows_cached
        else:
            master_rows = execute_sql_query(
                """
                SELECT DISTINCT master_food_name, food_name
                FROM public.supermarket_products
                WHERE active AND master_food_name IS NOT NULL
                """,
                fetch_all=True,
            ) or []
            if _catalog_cache_ttl_s() > 0:
                # [P2-BACKEND-SUPERMARKET-CACHE · 2026-08-14] Publica SOLO si nadie
                # invalido mientras leiamos. Si un admin escribio en medio, estas
                # filas son de antes de su cambio: cachearlas las dejaria vivas
                # hasta el TTL, que es justo el bug. Descartar cuesta una consulta
                # mas en la siguiente peticion.
                if not _publish_catalog_cache(rows, master_rows, _gen_al_empezar):
                    logger.info(
                        "[P2-BACKEND-SUPERMARKET-CACHE] relleno descartado: hubo una "
                        "mutacion admin mientras se leia el catalogo"
                    )
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
    user_id: Optional[str] = Depends(get_verified_user_id),
    _rl: Any = Depends(_PREFS_LIMITER),
):
    """Preferencias de marca del usuario autenticado, con el producto hidratado.

    [P1-SUPERMARKET-PREFS-DISCONTINUED · 2026-07-29] (fix round, finding 2) Antes
    devolvía la fila aunque `sp.active = false` (el producto elegido fue dado de
    baja por la admin UI) y dejaba que el CALLER decidiera filtrar por `active` —
    `SupermarketBrands.jsx` no lo hacía, así que un pin muerto revertía en silencio
    al default/más-barato en /match y en `fetch_brand_pref_packages` (costeo), sin
    avisar al usuario, y la fila quedaba huérfana para siempre (sin cron que limpie
    filas fuera del filtro `user_id = %s` de I2). Ahora: `preferences` SOLO trae
    filas activas (mismo contrato de forma para el caller — nada se rompe si no
    había inactivas); las inactivas se reportan aparte en `discontinued` (para que
    el frontend avise) Y se auto-borran aquí mismo (self-healing — este GET es el
    único lugar donde el usuario y su lista de pins muertos coinciden). Best-effort:
    si el DELETE falla, la fila sigue apareciendo en `discontinued` en la próxima
    carga — no bloquea."""
    # get_verified_user_id retorna None para anónimos (contrato P0-AUDIT-1) —
    # el caller DEBE rechazar. Sin este guard, el GET respondía 200 vacío y el
    # cliente creería que el guest tiene persistencia server-side.
    if not user_id:
        raise HTTPException(status_code=403, detail="Inicia sesión para guardar tus marcas preferidas.")

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
        active_rows = [r for r in rows if r.get("active")]
        stale_rows = [r for r in rows if not r.get("active")]
        if stale_rows:
            stale_keys = [r["food_key"] for r in stale_rows]
            try:
                execute_sql_write(
                    "DELETE FROM public.user_brand_preferences "
                    "WHERE user_id = %s AND food_key = ANY(%s)",
                    (user_id, stale_keys),
                )
                logger.info(
                    f"🧹 [P1-SUPERMARKET-PREFS-DISCONTINUED] {len(stale_keys)} preferencia(s) "
                    f"con producto discontinuado limpiadas para user={user_id}: {stale_keys}"
                )
            except Exception as exc_del:
                # Best-effort: no bloquea la respuesta — la fila muerta se reintenta
                # en la próxima carga (sigue excluida de `preferences` igual).
                logger.warning(
                    f"⚠️ [P1-SUPERMARKET-PREFS-DISCONTINUED] no se pudo limpiar "
                    f"{len(stale_keys)} preferencia(s) obsoleta(s): {exc_del}"
                )
        return {
            "preferences": {r["food_key"]: r for r in active_rows},
            "discontinued": [
                {
                    "food_key": r["food_key"],
                    "food_name": r.get("food_name"),
                    "brand": r.get("brand"),
                    "presentation": r.get("presentation"),
                }
                for r in stale_rows
            ],
        }

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as exc:
        logger.error(f"❌ [P1-SUPERMARKET-PREFS] get falló: {exc}")
        raise HTTPException(status_code=500, detail="No se pudieron cargar tus preferencias.")


@router.put("/preferences")
async def api_put_brand_preference(
    body: BrandPreferenceIn,
    user_id: Optional[str] = Depends(get_verified_user_id),
    _rl: Any = Depends(_PREFS_LIMITER),
):
    """Upsert (o borrado con product_id=null) de la marca preferida de UN alimento."""
    # Fail-secure (contrato P0-AUDIT-1): None = anónimo → rechazar. Los guests
    # persisten en localStorage del cliente, jamás server-side.
    if not user_id:
        raise HTTPException(status_code=403, detail="Inicia sesión para guardar tus marcas preferidas.")
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

    # [P1-SUPERMARKET-CATALOG-CACHE · 2026-08-05] El editor vive en la MISMA pagina
    # que consume el catalogo: sin invalidar, el admin editaria un precio y no lo
    # veria hasta que expirase el TTL.
    _invalidate_catalog_cache()

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

    # [P1-SUPERMARKET-CATALOG-CACHE · 2026-08-05] El editor vive en la MISMA pagina
    # que consume el catalogo: sin invalidar, el admin editaria un precio y no lo
    # veria hasta que expirase el TTL.
    _invalidate_catalog_cache()

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

    # [P1-SUPERMARKET-CATALOG-CACHE · 2026-08-05] El editor vive en la MISMA pagina
    # que consume el catalogo: sin invalidar, el admin editaria un precio y no lo
    # veria hasta que expirase el TTL.
    _invalidate_catalog_cache()

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
