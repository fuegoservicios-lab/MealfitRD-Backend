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

from typing import Any, Dict, List, Optional

import asyncio
import base64
import logging
import os

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import get_verified_user_id
# [P1-GUEST-CATALOG · 2026-08-11] El catálogo responde también sin sesión; el limitador
# es su contrapeso. Mismo singleton de módulo que usan los de `routers/plans.py`, que
# ya sabe agrupar por `ip:<host>` cuando no hay usuario (extensión P1-6).
from rate_limiter import RateLimiter
# [P1-SCAN-CATALOG-MATCH · 2026-08-10] La resolución «texto libre → alimento del
# catálogo» es SSOT en constants, junto a `pantry_names_match`.
from constants import pantry_names_match, resolve_scanned_food

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
    # [P2-NEVERA-BRANDS-MANUAL · 2026-07-07] Marca elegida al añadir a mano
    # (del picker del súper). NULL = sin marca (no pinta chip en la Nevera).
    brand: Optional[str] = None


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

    # [P2-NEVERA-BRANDS-MANUAL · 2026-07-07] Marca del picker: trim → NULL si vacío.
    _brand = (body.brand or "").strip() or None

    def _insert():
        import psycopg
        from db import execute_sql_query
        try:
            return execute_sql_query(
                """
                WITH ins AS (
                    INSERT INTO user_inventory
                        (user_id, ingredient_name, quantity, unit,
                         master_ingredient_id, source, category, brand)
                    VALUES (%s, %s, %s, %s, %s,
                            COALESCE(%s, 'shopping_list'), %s, %s)
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
                    ins.brand,
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
                 body.master_ingredient_id, body.source, body.category, _brand),
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


class InventoryUnitBody(BaseModel):
    unit: str


@router.patch("/inventory/items/{item_id}/unit")
async def api_change_inventory_unit(
    item_id: int,
    body: InventoryUnitBody = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    """[P1-PANTRY-SCAN-V0 · 2026-07-11] Cambiar el envase/unidad de un item
    (feedback owner: "no quiero una lata, quiero un paquete de habichuelas").
    Atómico en UN statement (CTEs): si ya existe otro row del usuario con el
    mismo nombre y la unidad destino (UNIQUE user+nombre+unidad), MERGEA las
    cantidades en el destino y borra el origen; si no, actualiza la unidad in
    place. I2: todo filtra user_id."""
    uid = _require_user(verified_user_id)
    new_unit = (body.unit or "").strip()
    if not new_unit or len(new_unit) > 40:
        raise HTTPException(status_code=422, detail="Unidad inválida.")

    def _change():
        from db import execute_sql_write
        return execute_sql_write(
            """
            WITH src AS (
                SELECT id, user_id, ingredient_name, quantity
                FROM user_inventory WHERE id = %s AND user_id = %s
            ), dup AS (
                SELECT ui.id AS dup_id, src.id AS src_id, src.quantity AS src_qty
                FROM user_inventory ui
                JOIN src ON ui.user_id = src.user_id
                    AND ui.ingredient_name = src.ingredient_name
                    AND ui.unit = %s AND ui.id <> src.id
            ), merged AS (
                UPDATE user_inventory SET quantity = user_inventory.quantity + dup.src_qty,
                    updated_at = NOW()
                FROM dup WHERE user_inventory.id = dup.dup_id
                RETURNING dup.src_id
            ), removed AS (
                DELETE FROM user_inventory
                WHERE id IN (SELECT src_id FROM merged)
                RETURNING id
            ), switched AS (
                UPDATE user_inventory SET unit = %s, updated_at = NOW()
                WHERE id = %s AND user_id = %s
                    AND NOT EXISTS (SELECT 1 FROM dup)
                RETURNING id
            )
            SELECT (SELECT COUNT(*) FROM removed) AS merged_count,
                   (SELECT COUNT(*) FROM switched) AS switched_count,
                   (SELECT COUNT(*) FROM src) AS found
            """,
            (item_id, uid, new_unit, new_unit, item_id, uid),
            returning=True,
        )

    rows = await asyncio.to_thread(_change)
    r = (rows or [{}])[0]
    if not r.get("found"):
        raise HTTPException(status_code=404, detail="Item no encontrado.")
    return {"merged": bool(r.get("merged_count")), "unit": new_unit}


class InventoryItemPatchBody(BaseModel):
    # [P1-PANTRY-ROW-EDIT · 2026-07-11] Edición directa de una fila del paso 21
    # (feedback owner: "si quiero escribir 200 gramos no tendría que darle al '+'
    # 200 veces" + "modificar las marcas"). quantity = valor ABSOLUTO (no delta);
    # brand: string = setear, "" = limpiar a NULL, ausente = no tocar.
    quantity: Optional[float] = None
    brand: Optional[str] = None


@router.patch("/inventory/items/{item_id}")
async def api_patch_inventory_item(
    item_id: int,
    body: InventoryItemPatchBody = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Set absoluto de cantidad y/o marca de un item. I2: filtra user_id."""
    uid = _require_user(verified_user_id)
    sets, params = [], []
    if body.quantity is not None:
        q = max(0.0, min(9999.0, float(body.quantity)))
        sets.append("quantity = %s::numeric")
        params.append(round(q, 2))
    if body.brand is not None:
        _b = body.brand.strip()[:60] or None
        sets.append("brand = %s")
        params.append(_b)
    if not sets:
        raise HTTPException(status_code=422, detail="Nada que actualizar (quantity y/o brand).")

    def _patch():
        from db import execute_sql_write
        return execute_sql_write(
            f"""
            UPDATE user_inventory SET {', '.join(sets)}, updated_at = NOW()
            WHERE id = %s AND user_id = %s
            RETURNING quantity::float8 AS quantity, brand
            """,
            (*params, item_id, uid),
            returning=True,
        )

    rows = await asyncio.to_thread(_patch)
    if not rows:
        raise HTTPException(status_code=404, detail="Item no encontrado.")
    return {"quantity": rows[0]["quantity"], "brand": rows[0]["brand"]}


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


# ---------------------------------------------------------------------------
# [P1-PANTRY-SCAN-V0 · 2026-07-11 → P1-VISION-NO-LOCAL · 2026-07-28] Escáner
# de nevera por foto (vision → items)
# ---------------------------------------------------------------------------
# Feature del owner: botón "Escanear mi nevera" — foto de la nevera física →
# modelo con visión detecta alimentos + cantidades → match contra el catálogo
# verificado → el usuario CONFIRMA la lista antes de que toque user_inventory
# (el endpoint NO escribe nada; los adds van por POST /inventory/items).
#
# [P1-VISION-NO-LOCAL · 2026-07-28] Provider CLOUD (el mismo transporte que
# "Escanear comida" del Dashboard) via el knob compartido de vision_agent:
#   MEALFIT_VISION_PROVIDER = off | openai_compatible
#   MEALFIT_VISION_MODEL    (modelo con visión, p.ej. gpt-5.6-luna)
#   MEALFIT_VISION_BASE_URL / VISION_API_KEY
# El provider LOCAL (Ollama/gemma, P1-MEAL-SCAN-GEMMA) y el single-flight que
# lo protegía (una sola GPU de 4GB, sin concurrencia) fueron ELIMINADOS junto
# con él — el laptop del owner no podía sostener el servicio. El transporte
# cloud no tiene ese límite: dos scans simultáneos ya no compiten por la
# misma GPU, así que no hay lock que mantener (ni el 409 "escáner ocupado"
# que producía).

_VISION_PROMPT = (
    "Eres un asistente de nutricion dominicano. Mira la foto de una nevera/despensa "
    "y lista TODOS los alimentos visibles e identificables con certeza razonable. "
    "Para cada uno estima la cantidad visible y su unidad de compra tipica en "
    "Republica Dominicana. Unidades permitidas: unidad, lb, g, paquete, botella, "
    "lata, taza, funda. En 'quantity' pon el NUMERO DE ENVASES O PIEZAS que se ven "
    "(1 paquete, 2 latas, 6 huevos) — NUNCA el peso o los gramos impresos en el "
    "empaque. En 'name' usa nombres genericos en espanol dominicano (ej: 'pechuga "
    "de pollo', 'arroz blanco', 'platano verde', 'huevos', 'leche') — la marca NO "
    "va en el nombre. [P1-PANTRY-SCAN-BRAND] Si el empaque muestra una MARCA "
    "legible (ej: Quaker, Rica, La Famosa), ponla en 'brand'; si no se lee o el "
    "alimento no tiene empaque, pon null. NO inventes "
    "alimentos que no se vean claramente; si dudas, omitelo. "
    # [P2-VISION-GUINEO-PLATANO · 2026-07-12] Misma leccion que el meal-scan:
    # banana dulce = 'guineo' en RD, no 'platano' (alimento distinto).
    "OJO en RD: el GUINEO es la banana dulce (delgada, curva, cascara fina, se "
    "come cruda) - NO lo llames platano; el PLATANO es mas grande y grueso y se "
    "cocina. "
    # [P1-SCAN-BREAD-GENERIC · 2026-08-10] El dueño fotografio pan de agua y salio
    # «pan de hot dog». El modelo no tiene por que acertar el tipo de pan desde una
    # foto en una funda, y aqui equivocarse SI cuesta: el catalogo no tiene ni «pan
    # de agua» ni «pan de hot dog», asi que un tipo inventado se queda sin match y
    # el usuario no puede agregar su pan — mientras que «pan» a secas es alias de
    # «Pan blanco familiar» y entra. Preferir el generico no es rendirse: es que el
    # nombre generico SI existe en el catalogo y el especifico inventado no.
    "OJO tambien con el PAN: en RD el pan de agua y el pan sobao son panes de mesa "
    "(sueltos, redondos u ovalados, NO vienen rebanados en funda) y no son pan de "
    "hot dog ni de hamburguesa (esos vienen ya partidos para rellenar). Si ves pan "
    "y no distingues el tipo con CERTEZA, escribe simplemente 'pan' - vale mas el "
    "nombre generico que acertar el tipo por suerte. "
    "Responde SOLO el JSON."
)


def _sane_scan_qty(qty, unit) -> float:
    """[P1-PANTRY-SCAN-QTY · 2026-07-11] Cantidad sanitizada por clase de unidad.
    Bug vivo: foto de UN paquete de avena → el modelo devolvió el peso impreso
    (500) → el clamp plano min(99) mostró "99 paquete". Envases discretos con
    qty absurda (>12) casi siempre son un peso mal leído → colapsar a 1 (mejor
    subestimar: el usuario ajusta con +). 'unidad' tolera más (30 huevos); pesos
    reales (lb/g) conservan rango amplio."""
    try:
        q = float(qty or 1)
    except (TypeError, ValueError):
        q = 1.0
    u = str(unit or "").strip().lower()
    if u in ("g", "gramo", "gramos"):
        return max(10.0, min(5000.0, q))
    if u in ("lb", "libra", "libras"):
        return max(0.25, min(10.0, q))
    if u in ("unidad", "unidades"):
        return float(max(1, min(30, round(q))))
    # paquete / lata / botella / funda / taza — envase discreto
    q = round(q)
    if q > 12:
        return 1.0
    return float(max(1, q))

# [P1-VISION-NO-LOCAL · 2026-07-28] Schema de salida estructurada para el
# cliente LangChain (reemplaza el JSON-schema crudo `_VISION_SCHEMA` que se
# le pasaba a Ollama como `format`) — mismo patrón que `ImageDescription` de
# vision_agent.py, ya probado contra la API real de Luna.
class _PantryScanItem(BaseModel):
    name: str = Field(description="Nombre generico del alimento en espanol dominicano, sin marca.")
    quantity: float = Field(description="Numero de envases o piezas visibles (NUNCA el peso o los gramos impresos en el empaque).")
    unit: str = Field(description="Unidad de compra: unidad, lb, g, paquete, botella, lata, taza o funda.")
    confidence: float = Field(description="Confianza 0-1 en la deteccion del item.")
    brand: Optional[str] = Field(default=None, description="Marca legible en el empaque (ej: Quaker, Rica), o null si no se lee o no tiene empaque.")


class _PantryScanResult(BaseModel):
    items: List[_PantryScanItem] = Field(default_factory=list, description="Alimentos detectados en la foto de la nevera/despensa.")


def vision_scan_provider() -> str:
    """Provider activo del escáner ('off' apaga el feature — el frontend oculta
    el botón vía `photo_scan_enabled` en /pantry-feasibility)."""
    return (os.environ.get("MEALFIT_VISION_PROVIDER") or "off").strip().lower()


def _match_catalog(detected_name: str, catalog: list) -> Optional[Dict[str, Any]]:
    """Row de `master_ingredients` al que corresponde lo que leyó la visión, o None.

    [P1-SCAN-CATALOG-MATCH · 2026-08-10] La regla vive en
    `constants.resolve_scanned_food` (SSOT, junto a `pantry_names_match`) — aquí
    solo se traduce nombre → row.

    LO QUE HABÍA ANTES: contención de substring en ambas direcciones y, si eso
    fallaba, UN token en común bastaba. Como "de" es un token y 36 de los 204
    alimentos lo llevan, cualquier detección con esa palabra caía en el primer
    alimento del catálogo que la tuviera — «pan de hamburguesa» → «Polvo de
    hornear», que es lo que reportó el dueño. Y el substring en la dirección
    catálogo⊆detectado producía la familia clásica de este repo: «salami» → «Sal».

    Medido con 34 detecciones etiquetadas contra el catálogo real: 19 aciertos y
    15 mapeos al alimento equivocado antes; 34 aciertos y cero después."""
    match_name = resolve_scanned_food(
        detected_name,
        [row["name"] for row in catalog],
        {row["name"]: (row.get("aliases") or []) for row in catalog},
    )
    if not match_name:
        return None
    for row in catalog:
        if row["name"] == match_name:
            return row
    return None


async def _cloud_vision_scan(image_bytes: bytes) -> list:
    """[P1-VISION-NO-LOCAL · 2026-07-28] Reemplaza `_ollama_vision_scan`:
    MISMO prompt de negocio (`_VISION_PROMPT`, incluye la lección
    guineo/plátano) y MISMO contrato de salida (items con name/quantity/
    unit/confidence/brand), pero el transporte ahora es el cliente cloud
    compartido con el meal-scan (`vision_agent.analyze_image_structured` —
    mismo `_resolve_vision_client`/resize/telemetría que "Escanear comida"),
    en vez de un roundtrip httpx propio contra Ollama. Lanza en cualquier
    error (red, parseo, provider caído) — el caller (el endpoint) lo mapea
    a HTTP 502, el mismo contrato que tenía con Ollama."""
    from vision_agent import analyze_image_structured
    result = await analyze_image_structured(image_bytes, _VISION_PROMPT, _PantryScanResult)
    items = result.items if isinstance(result, _PantryScanResult) else []
    return [
        {
            "name": it.name,
            "quantity": it.quantity,
            "unit": it.unit,
            "confidence": it.confidence,
            "brand": it.brand,
        }
        for it in items
    ]


@router.post("/inventory/photo-scan")
async def api_inventory_photo_scan(
    body: Dict[str, Any] = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Foto (base64) → items detectados con match al catálogo. READ-ONLY:
    no escribe user_inventory — el cliente confirma y agrega vía /inventory/items."""
    _require_user(verified_user_id)

    provider = vision_scan_provider()
    if provider == "off":
        raise HTTPException(status_code=503, detail="El escáner de nevera no está disponible por ahora.")
    if provider != "openai_compatible":
        raise HTTPException(status_code=503, detail=f"Provider de visión desconocido: {provider}")

    image_b64 = str(body.get("image_b64") or "")
    # ~6MB de imagen (8MB b64). El cliente ya reescala a ≤1024px, pero el
    # server NO confía en eso: `analyze_image_structured` corre el mismo
    # guard `prepare_image_for_vision` que el meal-scan antes de despachar.
    if not image_b64 or len(image_b64) > 8_000_000:
        raise HTTPException(status_code=422, detail="Imagen ausente o demasiado grande.")

    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except Exception:
        raise HTTPException(status_code=422, detail="Imagen inválida (base64 corrupto).")

    try:
        items = await _cloud_vision_scan(image_bytes)
    except Exception as e:
        logger.warning(f"[P1-PANTRY-SCAN-V0] photo-scan falló ({type(e).__name__}): {e}")
        raise HTTPException(
            status_code=502,
            detail="No pudimos analizar la foto (el modelo de visión no respondió). Intenta de nuevo.",
        )

    def _match_against_catalog():
        from db import execute_sql_query
        # [P1-SCAN-ALIASES · 2026-08-10] `aliases` entra al SELECT: son los 816
        # sinónimos curados del catálogo y esta consulta ni los pedía, así que el
        # escáner resolvía nombres a ciegas teniendo la respuesta al lado.
        catalog = execute_sql_query(
            "SELECT id::text AS id, name, aliases, market_container, default_unit FROM master_ingredients",
            fetch_all=True,
        ) or []
        out = []
        for it in items[:40]:
            match = _match_catalog(it.get("name"), catalog)
            _unit = str(it.get("unit") or "unidad")[:20]
            # [P1-PANTRY-SCAN-BRAND] Marca leída del empaque — etiqueta el item al
            # confirmar. NO toca user_brand_preferences: la preferencia "para
            # siempre" es SOLO elección manual del usuario (un OCR equivocado no
            # debe contaminar sus marcas preferidas globales).
            _brand = (str(it.get("brand") or "").strip() or None)
            # [P1-SCAN-CATALOG-MATCH · 2026-08-10] ¿El mapeo RENOMBRA lo que leyó
            # la visión? La lista de confirmación mostraba `catalog_name` a secas,
            # así que un mapeo equivocado borraba de la pantalla lo que el modelo
            # había dicho: el dueño vio «Polvo de hornear» sobre la foto de un pan
            # y concluyó, razonablemente, que el modelo había confundido las dos
            # cosas — cuando el modelo había leído bien y falló el mapeo. Con este
            # flag el cliente puede enseñar ambas y el error deja de ser invisible.
            # Solo se marca cuando el nombre cambia DE VERDAD (no por plural ni
            # acento), para no ensuciar «huevos» → «Huevo».
            _renamed = bool(match) and not pantry_names_match(
                str(it.get("name") or ""), match["name"]
            )
            out.append({
                "detected_name": str(it.get("name") or "")[:80],
                "detected_brand": _brand[:40] if _brand else None,
                "quantity": _sane_scan_qty(it.get("quantity"), _unit),
                "unit": _unit,
                "confidence": max(0.0, min(1.0, float(it.get("confidence") or 0))),
                "master_ingredient_id": match["id"] if match else None,
                "catalog_name": match["name"] if match else None,
                "catalog_renamed": _renamed,
                "catalog_unit": (match.get("market_container") or match.get("default_unit")) if match else None,
            })
        return out

    # READ-ONLY: el match contra el catálogo NUNCA escribe user_inventory —
    # los adds van por POST /inventory/items una vez el usuario confirma.
    results = await asyncio.to_thread(_match_against_catalog)
    return {"items": results, "provider": provider}


# [P1-GUEST-CATALOG · 2026-08-11] Campos que un SELECTOR de alimentos necesita, y ni uno
# más. El resto de columnas del catálogo —precios por libra y por unidad, densidades,
# envase de mercado, tamaños disponibles— no las usa ningún buscador: las usa la Nevera
# para calcular costos y conversiones, y son el trabajo curado de este producto. Un
# invitado no las necesita para escribir «arroz».
_CATALOG_CAMPOS_INVITADO = ("id", "slug", "name", "category", "aliases", "default_unit", "staple_gate_label")

# Anti-abuso del camino sin sesión: el catálogo es ~20KB y el cliente lo cachea 24h, así
# que un uso legítimo pide esto UNA vez por wizard. El límite no molesta a nadie real y
# evita que un endpoint sin auth se convierta en un grifo de scraping barato.
_CATALOG_LIMITER = RateLimiter(max_calls=12, period_seconds=60)

# [P1-PLAN-MODE · 2026-08-11] El interruptor y la puerta de metas. Quota-exempt por
# la doctrina de /restock: aplicar el paywall al botón de APAGAR el gasto es
# exactamente al revés — un usuario topado en 402 que no puede apagar deja al chunk
# worker gastándole dinero al negocio, porque los crons no cobran.
_PLAN_MODE_LIMITER = RateLimiter(max_calls=15, period_seconds=60)
_TARGETS_LIMITER = RateLimiter(max_calls=30, period_seconds=60)

# [P2-I18N-LOCALE-DISPARA-LLM · 2026-08-21] `PATCH /profile` dejó de ser un UPDATE
# escalar el 2026-08-19: desde P1-PLAN-DISPLAY-I18N, un `locale` distinto de 'es-DO'
# despacha `schedule_plan_display_enrichment` — la traducción LLM del plan entero, por
# lotes. Era el write más caro del router y el único sin limitador.
#
# 10/60s y no menos: el uso legítimo es tocar un ajuste, no diez por minuto, pero el
# wizard guarda el perfil por pasos y una ráfaga corta es normal. El paywall NO es la
# herramienta aquí (doctrina de /restock: al llegar al cap el usuario quedaría atrapado
# en su idioma y cada cambio le quemaría crédito de PLANES, porque
# `get_monthly_api_usage` cuenta toda fila de `api_usage` sin filtrar endpoint).
_PROFILE_PATCH_LIMITER = RateLimiter(max_calls=10, period_seconds=60)


@router.get("/catalog")
async def api_get_catalog(
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    _rl: None = Depends(_CATALOG_LIMITER),
):
    """Catálogo master_ingredients (cuasi-inmutable, ~20KB). El frontend
    mantiene su cache singleton de 24h — este endpoint solo cambia el transporte.

    [P1-GUEST-CATALOG · 2026-08-11] SIN SESIÓN TAMBIÉN RESPONDE, con una proyección
    reducida. Antes exigía auth «(paridad con el acceso RLS previo)» — o sea, la
    restricción venía del TRANSPORTE anterior, no de una necesidad de privacidad:
    `master_ingredients` es una tabla global de referencia, no datos de nadie.

    La consecuencia era un paso del formulario que un invitado no podía completar. El
    wizard es público (`/assessment` sin login) y su paso 15 —«Tus básicos de siempre»—
    busca contra este catálogo: sin sesión la lista llegaba vacía, así que el buscador
    no autocompletaba nada y no se podía añadir ningún alimento. No fallaba de forma
    visible; simplemente no encontraba nada, que es peor, porque parece que el alimento
    no existe.

    Lo que NO se abre: precios, densidades y datos de envase. Un buscador necesita
    nombres; el resto es el trabajo curado del producto y sigue detrás de la sesión.

    [P2-I18N-CATALOGO-BUSCADOR-SIN-PUENTE · 2026-08-22] Se envía también `name_en`.

    Está poblado al **347/347** y en **329** filas difiere del nombre español (medido contra
    Neon, no supuesto), y el endpoint no lo mandaba: un usuario en inglés que escribía
    «chicken» o «rice» en el buscador de básicos obtenía CERO resultados. No fallaba de
    forma visible; simplemente no encontraba nada, que es peor, porque parece que el
    alimento no existe — el mismo modo de fallo que `P1-GUEST-CATALOG` cerró para el
    invitado.

    ⚠️ **Esto cubre UN idioma de los cuatro.** `name_en` es un gloss inglés, no un catálogo
    multilingüe: para fr/it/pt no hay columna que enviar y el buscador sigue exigiendo el
    nombre español. Decirlo importa — un puente a medias presentado como puente completo es
    peor que ninguno, porque nadie vuelve a mirarlo.

    Y `name_en` es un NOMBRE, así que va en la proyección abierta al invitado por el mismo
    argumento que `name`: lo que sigue detrás de la sesión son los precios y el trabajo
    curado, no cómo se llama un alimento.
    """

    def _catalog():
        from db import execute_sql_query
        return execute_sql_query(
            """
            SELECT id::text AS id, slug, name, name_en,
                   to_jsonb(mi)->>'gloss_es' AS gloss_es, category, aliases,
                   density_g_per_cup::float8 AS density_g_per_cup,
                   density_g_per_unit::float8 AS density_g_per_unit,
                   shelf_life_days,
                   price_per_lb::float8 AS price_per_lb,
                   price_per_unit::float8 AS price_per_unit,
                   market_container, container_weight_g::float8 AS container_weight_g,
                   available_sizes_g, default_unit,
                   kcal_per_100g::float8 AS kcal_per_100g,
                   protein_g_per_100g::float8 AS protein_g_per_100g,
                   carbs_g_per_100g::float8 AS carbs_g_per_100g,
                   fats_g_per_100g::float8 AS fats_g_per_100g,
                   fiber_g_per_100g::float8 AS fiber_g_per_100g,
                   sodium_mg_per_100g::float8 AS sodium_mg_per_100g
            FROM master_ingredients AS mi ORDER BY name ASC
            """,
            fetch_all=True,
        ) or []

    items = await asyncio.to_thread(_catalog)

    # [P1-STAPLE-SEARCH-RANK · 2026-08-09] Rótulo del gate same-day-protein por
    # alimento, calculado AQUÍ desde el SSOT (`_MAIN_PROTEIN_ALIASES` +
    # `_SAME_DAY_PROTEIN_GATE_LABELS`) y servido al cliente.
    #
    # El motivo de servirlo en vez de que el frontend lo deduzca: dos alimentos
    # distintos del catálogo pueden colapsar al MISMO rótulo (clara de huevo y
    # huevo → "huevo"), así que declarar ambos como básicos gasta dos de los
    # ocho cupos para un solo efecto. Para avisarlo, el cliente necesita conocer
    # el rótulo — y la única forma de que no se desincronice con el motor es que
    # NO tenga su propia copia de la tabla de alias. Este repo ya pagó ese
    # precio: la canonicalización de dieta vivía en tres tablas a mano, driftaron,
    # y la del filtro servía pollo a vegetarianas.
    #
    # `None` cuando el alimento no participa del gate (legumbres, vegetales,
    # cereales): esos ya pueden repetirse libremente, así que no hay nada que
    # avisar. Fail-safe: cualquier error deja el campo ausente y el cliente
    # degrada a no mostrar el aviso.
    try:
        from graph_orchestrator import _protein_gate_labels_in_text
        for _it in items:
            _labels = _protein_gate_labels_in_text(str(_it.get("name") or ""))
            _it["staple_gate_label"] = "+".join(sorted(_labels)) if _labels else None
    except Exception:
        logger.warning("[P1-STAPLE-SEARCH-RANK] no se pudo anotar el catálogo con el rótulo del gate", exc_info=True)

    # [P1-MANUAL-FOOD-LOG · 2026-08-11] Porciones PRECOMPUTADAS server-side. El
    # componedor del diario las multiplica (`qty × grams_per_qty`) y eso es aritmética;
    # decidir cuántos gramos tiene «1 taza de arroz» es del catálogo y de nadie más. Si
    # el cliente llevara su propia tabla de conversión, sería otra copia del motor
    # esperando a driftar — el precio que este repo ya pagó con la dieta.
    # La resolución REAL al enviar vuelve a correr server-side (`food_search`); esto
    # existe solo para que la vista previa del cliente enseñe los mismos números.
    for _it in items:
        _p = [{"unit": "g", "grams_per_qty": 1.0, "label": "g"}]
        if _it.get("density_g_per_cup"):
            _p.append({"unit": "taza", "grams_per_qty": float(_it["density_g_per_cup"]), "label": "taza"})
        if _it.get("density_g_per_unit"):
            _p.append({"unit": "unidad", "grams_per_qty": float(_it["density_g_per_unit"]), "label": "unidad"})
        _du = str(_it.get("default_unit") or "").strip().lower()
        _def = "unidad" if (_du in ("unidad", "unit") and len(_p) > 2) else ("taza" if any(x["unit"] == "taza" for x in _p) and _du not in ("lb", "unidad") else _p[-1]["unit"])
        for _x in _p:
            _x["default"] = (_x["unit"] == _def)
        _it["portions"] = _p

    # [P1-GUEST-CATALOG · 2026-08-11] La poda va DESPUÉS de anotar el rótulo del gate: ese
    # campo lo calcula el backend a propósito (ver la nota de P1-STAPLE-SEARCH-RANK justo
    # arriba) y el buscador de básicos lo necesita para avisar de que dos alimentos gastan
    # un solo cupo. Podar antes lo dejaría fuera y el invitado perdería ese aviso.
    if not verified_user_id:
        items = [
            {k: it.get(k) for k in _CATALOG_CAMPOS_INVITADO}
            for it in items
        ]

    return {"items": items}


@router.get("/catalog/dishes")
async def api_get_catalog_dishes(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """[P1-MANUAL-FOOD-LOG · 2026-08-11] Los 60 platos criollos curados
    (`data/dominican_dishes.json`), en su vista pública: label + ración
    (`finished_g`) + `per_100g`. SIN `constituents` — la expansión a Nevera es
    server-side y el cliente no necesita saber de qué está hecho el moro.

    Cuasi-inmutable como el catálogo (~4 KB gzip): el frontend lo cachea 24 h en
    `pantryCache` junto al resto. Auth por paridad con `/catalog`."""
    _require_user(verified_user_id)
    import food_search
    return {"items": await asyncio.to_thread(food_search.dishes_for_client)}


@router.get("/profile/plan-mode")
async def api_get_plan_mode(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """[P1-PLAN-MODE · 2026-08-11] La postura del usuario sobre la generación."""
    _require_user(verified_user_id)
    from plan_mode import get_plan_mode
    return await asyncio.to_thread(get_plan_mode, verified_user_id)


class PlanModeBody(BaseModel):
    plan_mode: str = Field(..., pattern="^(plan|tracking)$")


@router.put("/profile/plan-mode")
async def api_put_plan_mode(
    body: PlanModeBody,
    verified_user_id: Optional[str] = Depends(_PLAN_MODE_LIMITER),
):
    """[P1-PLAN-MODE · 2026-08-11] Apagar/encender la generación de planes.

    NO pasa por PATCH /api/profile ni por `_PROFILE_SCALAR_WHITELIST` a propósito:
    cambiar el modo no es editar un escalar — es una transacción (cancelar cola,
    liberar locks, estampar el plan) y una regla que vive en dos puertas se cumple
    en una. La respuesta dice LO QUE PASÓ (chunks cancelados, estado restaurado,
    si el plan venció) para que la UI no adivine ni refetchee el perfil entero.

    Apagar es GRATIS y no es negociable: cobrar por el freno es cobrar por dejar
    de cobrar. Reanudar dentro de la ventana también (los chunks ya se pagaron);
    tras vencer, el camino es un /analyze nuevo, que ya se cobra solo."""
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Inicia sesión.")
    from plan_mode import pause_plan_generation, resume_plan_generation
    if body.plan_mode == "tracking":
        out = await asyncio.to_thread(pause_plan_generation, verified_user_id)
    else:
        out = await asyncio.to_thread(resume_plan_generation, verified_user_id)
    # Espejo client-side para el arranque en frío del dashboard (ver el wrapper de
    # Dashboard.jsx): si el perfil llega lento, «no sé» no puede leerse como «plan».
    return {"success": True, **out}


@router.get("/nutrition/targets")
async def api_nutrition_targets(
    verified_user_id: Optional[str] = Depends(_TARGETS_LIMITER),
):
    """[P1-PLAN-MODE · 2026-08-11] Las metas del contador SIN plan.

    `get_nutrition_targets` es pura (sin DB, sin LLM) y hoy solo corre dentro del
    pipeline: sin esta puerta, la tarjeta de progreso pinta 2000/150/200/60 — los
    cuatro `||` de TrackingProgress son un plan genérico disfrazado de meta
    personal, y una barra que miente es peor que una barra ausente.

    Contrato: la FORMA es idéntica a plan_data (`calories` numérico +
    `macros.{protein,carbs,fats}` strings con 'g') para que la tarjeta consuma una
    sola forma venga de donde venga. `missing_fields` es lo que la hace honesta:
    fail-CLOSED — sin `ok:true` el cliente no pinta barras."""
    if not verified_user_id:
        return {"ok": False, "missing_fields": ["session"], "reason": "guest"}

    # [P1-TARGETS-NAMEERROR · 2026-08-12] TODO el camino con DB va dentro del try:
    # este endpoint es fail-closed POR CONTRATO ({ok:false} honesto, jamás 500 crudo
    # al cliente) y la primera versión lo violó del modo más tonto — el fetch del
    # perfil vivía FUERA del try y usaba `get_user_profile` sin importarlo (el
    # idioma del archivo es import lazy por-endpoint). NameError es runtime-only:
    # el import-check del módulo pasa, y el test estructural también pasaba. El
    # test nuevo llama el happy path DE VERDAD.
    try:
        from db import get_user_profile
        profile = await asyncio.to_thread(get_user_profile, verified_user_id)
        hp = (profile or {}).get("health_profile") or {}

        # [P1-TRACKING-FINISH-SENSITIVE-GUARD · 2026-08-12] medicalConditions es
        # requerida TAMBIÉN aquí: el gate de embarazo/lactancia del calculador lee
        # esa lista, y sin exigirla un perfil con condiciones vacías (hidratación
        # rota, PATCH viejo destructivo) devolvía metas CON DÉFICIT y ok:true —
        # fail-open silencioso del gate. `[]` es falsy ⇒ cuenta como faltante.
        # En la rama plan el contrato equivalente vive en _REQUIRED_FORM_FIELDS.
        _requeridos = ("gender", "age", "height", "weight", "weightUnit", "activityLevel", "mainGoal", "medicalConditions")
        faltan = [c for c in _requeridos if not hp.get(c)]
        if faltan:
            return {"ok": False, "missing_fields": faltan}

        from nutrition_calculator import get_nutrition_targets
        t = await asyncio.to_thread(get_nutrition_targets, hp)
        m = t.get("macros") or {}
        return {
            "ok": True,
            "calories": int(t.get("target_calories") or 0),
            "macros": {
                "protein": m.get("protein_str") or f"{m.get('protein_g', 0)}g",
                "carbs": m.get("carbs_str") or f"{m.get('carbs_g', 0)}g",
                "fats": m.get("fats_str") or f"{m.get('fats_g', 0)}g",
            },
            "goal_label": t.get("goal_label"),
            "kinematics": t.get("kinematics"),
        }
    except Exception as e:
        logger.error(f"[P1-PLAN-MODE] /nutrition/targets falló: {e}")
        return {"ok": False, "missing_fields": [], "reason": "calc_error"}


# ---------------------------------------------------------------------------
# Perfil (user_profiles)
# ---------------------------------------------------------------------------

# Whitelist ESTRICTA de columnas escalares actualizables por el cliente.
# NUNCA añadir columnas de entitlement (plan_tier, subscription_status,
# subscription_end_date, paypal_subscription_id, ...) — el tier es
# server-derived desde PayPal (I-Billing-1, P0-BILLING-1); aceptar
# plan_tier del cliente reabriría el upgrade gratis via DevTools.
# Tooltip-anchor: P1-NEON-PROFILE-SCALAR-WHITELIST.
#
# [P1-I18N-DASHBOARD · 2026-08-15] `locale` entra aquí y NO en un endpoint
# propio. La razón está escrita al revés en el docstring de PUT
# /profile/plan-mode: aquel quedó FUERA del whitelist porque cambiar el modo es
# una TRANSACCIÓN (cancela la cola, libera locks, estampa el plan). Elegir
# idioma es literalmente escribir un escalar y no tiene efectos laterales, así
# que un endpoint propio solo añadiría un limitador más, una fila más en la
# tabla de exención de cuota y una segunda puerta al mismo UPDATE.
_PROFILE_SCALAR_WHITELIST = frozenset({"full_name", "locale"})

# [P1-I18N-DASHBOARD · 2026-08-15] Valores admitidos de `locale`.
#
# El whitelist de arriba valida CLAVES, no VALORES — nunca necesitó mirarlos
# porque `full_name` es texto libre. `locale` sí: es un enum, y la columna lleva
# un CHECK en la DB (migración p1_i18n_dashboard_locale_2026_08_15.sql). Sin
# esta validación el CHECK haría el trabajo, pero devolviendo un 500 crudo de
# psycopg en vez de un 400 que explica qué pasó. Fail-closed y legible.
#
# SSOT de la lista: `frontend/src/i18n/locales.js`. Este frozenset, el CHECK de
# la migración y el boot de `index.html` son espejos suyos —
# `test_p1_i18n_dashboard.py` falla si divergen.
# Tooltip-anchor: P1-I18N-DASHBOARD-LOCALE-VALUES.
_LOCALE_VALUES = frozenset({"es-DO", "en-US", "pt-BR", "fr-FR", "it-IT"})


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
    _rl: None = Depends(_PROFILE_PATCH_LIMITER),
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
    # [P1-I18N-DASHBOARD · 2026-08-15] Validación de VALOR para los escalares
    # que son enum. El whitelist de arriba solo mira claves; sin esto, el 400
    # honesto lo daría el CHECK de la DB como un 500 de psycopg.
    if "locale" in fields and fields["locale"] not in _LOCALE_VALUES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Idioma no soportado: {fields['locale']!r}. "
                f"Permitidos: {sorted(_LOCALE_VALUES)}."
            ),
        )

    # [P3-COUNTRY-DB-CHECK · 2026-08-22] La simétrica de arriba para la ÚNICA clave enum que vive
    # dentro del JSONB. `health_profile` es texto libre por diseño (super_personalization, perfil
    # clínico, alergias) y sigue siéndolo: esto valida UNA clave, no convierte el merge en un
    # whitelist — ensancharlo rompería tres superficies que no tienen que ver con el país.
    #
    # Rechaza, NO canoniza. `canonicalize_country` cae a 'DO' ante cualquier valor no canónico, así
    # que coercer aquí dejaría al usuario recibiendo planes dominicanos sin nada en pantalla que se
    # lo dijera — el «default sembrado» que este repo ya pagó. Un 400 le dice qué pasó.
    #
    # SSOT de la lista: `constants.COUNTRY_PROFILES`. El CHECK de
    # `p3_country_db_check_2026_08_22.sql` es la red de abajo, no el sustituto: sin este 400, el
    # CHECK haría el trabajo devolviendo un 500 crudo de psycopg.
    # Tooltip-anchor: P3-COUNTRY-DB-CHECK-VALUE.
    if body.health_profile and body.health_profile.get("country") is not None:
        from constants import UnsupportedCountryError, assert_supported_country
        try:
            assert_supported_country(body.health_profile["country"])
        except UnsupportedCountryError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

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

    # [P1-PLAN-DISPLAY-I18N · 2026-08-19 · fix-round 1 F3/F4/F5] tooltip-anchor:
    # P1-PLAN-DISPLAY-I18N-TRIGGER-4 (spec: 3 = mutadores/Task 3, 4 = cambio de
    # idioma — este PATCH). El UPDATE de arriba (locale != es-DO) completó —
    # despachar enrich del plan ACTIVO (el más reciente) SOLO si le falta
    # `_display[locale]` en el PRIMER o el ÚLTIMO día. Mirar ambos extremos
    # (no solo el primero) cierra el freeze de un enriquecimiento parcial: el
    # motor trocea por lotes y permite recuperación parcial (un lote que falla
    # no tumba a los demás) — si solo mirásemos el primer día, un plan cuyo
    # último lote nunca corrió quedaría "ya enriquecido" para siempre, sin
    # ningún disparador que lo complete (el 5º disparador de la spec es "no
    # hay backfill masivo"). Proyección jsonb O(1): NO se baja `plan_data`
    # completo (puede ser cientos de KB-MB con 30 días de recetas expandidas)
    # solo para mirar dos claves. `->-1` en el índice de array jsonb cuenta
    # desde el final (soportado por Postgres/Neon) — con 1 solo día, primer y
    # último son el mismo elemento (redundante pero inofensivo). Best-effort:
    # el PATCH de perfil JAMÁS puede fallar por esto.
    _p1_i18n_new_locale = fields.get("locale")
    if _p1_i18n_new_locale and _p1_i18n_new_locale != "es-DO":
        try:
            # [P3-I18N-DISPLAY-BLANKET-CIEGO-AL-SQL · 2026-08-23] La consulta que miraba
            # `_display` en los dos extremos vive ahora en el SSOT (`plan_display_i18n`):
            # este router solo despacha. Era una lectura de `_display` para decidir, por
            # SQL, en un fichero sin permiso — invisible para el blanket.
            from plan_display_i18n import schedule_plan_display_enrichment as _p1_i18n_schedule
            from plan_display_i18n import active_plan_missing_locale as _p1_i18n_missing
            _p1_i18n_plan_id = await asyncio.to_thread(_p1_i18n_missing, uid, _p1_i18n_new_locale)
            if _p1_i18n_plan_id:
                _p1_i18n_schedule(_p1_i18n_plan_id, uid, _p1_i18n_new_locale)
        except Exception as _p1_i18n_e:
            logger.warning(
                f"[P1-PLAN-DISPLAY-I18N] dispatch PATCH /profile locale falló "
                f"user={uid} locale={_p1_i18n_new_locale!r}: {_p1_i18n_e!r}"
            )

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
        # [P1-ARQ25-F1-LIFECYCLE · 2026-09-02] `revision` (I12): el frontend adopta el plan
        # entero cuando la del servidor es mayor que la local (hydrateLatestPlan).
        cols = "id::text AS id, to_jsonb(created_at)#>>'{}' AS created_at, " \
               "to_jsonb(updated_at)#>>'{}' AS updated_at, revision"
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
            "SELECT id::text AS id, plan_data, revision, "
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
# del RAG: al guardar con un freeText NUEVO, se extraen facts (GLM-flash) →
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


# ---------------------------------------------------------------------------
# "Mis básicos" (health_profile.staple_foods)
# [P1-STAPLE-FOODS · 2026-08-02]
# ---------------------------------------------------------------------------
# Alimentos que el usuario declara que come "de siempre" (feature aprobada por el owner, ver
# CLAUDE.md). Máx 8, SOLO nombres del catálogo verificado (`master_ingredients`) — a propósito
# chips-únicamente, NO texto libre como dislikes/allergies: los básicos alimentan gates
# deterministas (graph_orchestrator.build_variety_report, agent.py swap gate) que matchean por
# alias exacto; texto libre arbitrario no matchearía nada y frustraría al usuario en silencio.
# Persiste como sub-key JSONB de health_profile (sin migración — mismo patrón que
# super_personalization/clinical_profile). Consumido vía `form_data['staple_foods']`: en la
# generación inicial/renovación llega directo del cliente (mismo mecanismo que dislikes/allergies,
# formData.stapleFoods → localStorage cifrado); en swap/regen-day se hidrata SERVER-SIDE desde
# aquí en `_enrich_clinical_from_profile` (routers/plans.py) para que una edición reciente en
# Ajustes no quede invisible por una ventana de cliente stale. Knob global MEALFIT_STAPLE_FOODS
# (default ON) apaga el CONSUMO en graph_orchestrator/agent.py — este endpoint sigue disponible
# incluso con el knob OFF (persistencia y consumo son kill-switches independientes a propósito).

_STAPLE_FOODS_MAX = 8


def _validate_staple_foods(names: list) -> list:
    """Valida `names` contra `master_ingredients` (case-insensitive, catálogo verificado — NO texto
    libre). 422 si excede el máximo o si algún nombre no matchea el catálogo (lista los inválidos
    para que el frontend pueda señalarlos). Dedup case-insensitive preservando el primer casing
    recibido. Vacío es válido (el usuario puede no tener básicos declarados)."""
    if not isinstance(names, list):
        raise HTTPException(status_code=422, detail="'staple_foods' debe ser una lista.")
    cleaned: list = []
    seen_lower: set = set()
    for n in names:
        s = str(n or "").strip()
        if not s:
            continue
        sl = s.lower()
        if sl in seen_lower:
            continue
        seen_lower.add(sl)
        cleaned.append(s)
    if len(cleaned) > _STAPLE_FOODS_MAX:
        raise HTTPException(
            status_code=422,
            detail=f"Máximo {_STAPLE_FOODS_MAX} básicos (recibidos {len(cleaned)}).",
        )
    if not cleaned:
        return []
    from db import execute_sql_query
    rows = execute_sql_query(
        "SELECT name FROM master_ingredients WHERE lower(name) = ANY(%s)",
        (list(seen_lower),),
        fetch_all=True,
    ) or []
    _valid_lower = {str(r["name"]).strip().lower() for r in rows}
    _invalid = [c for c in cleaned if c.lower() not in _valid_lower]
    if _invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Alimento(s) no encontrados en el catálogo verificado: {', '.join(_invalid)}.",
        )
    return cleaned


@router.get("/user/preferences/staple-foods")
async def api_get_staple_foods(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Devuelve los básicos declarados del usuario (o [] si no ha declarado ninguno). Read-only,
    cero costo LLM (misma exención que el resto de /user/preferences)."""
    uid = _require_user(verified_user_id)
    from db import get_user_profile

    profile = await asyncio.to_thread(get_user_profile, uid)
    hp = (profile or {}).get("health_profile") or {}
    sf = hp.get("staple_foods") if isinstance(hp, dict) else None
    return {"staple_foods": sf if isinstance(sf, list) else []}


class StapleFoodsBody(BaseModel):
    staple_foods: List[str] = Field(default_factory=list)


@router.put("/user/preferences/staple-foods")
async def api_put_staple_foods(
    body: StapleFoodsBody = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Persiste `health_profile.staple_foods` vía update_user_health_profile_atomic (SELECT…FOR
    UPDATE + callback, I7 — sin lost-update bajo concurrencia). Filtra por user_id autenticado
    (I2). Valida máx 8 + catálogo real (422 si no) — es un widget de CHIPS, no texto libre."""
    uid = _require_user(verified_user_id)
    cleaned = await asyncio.to_thread(_validate_staple_foods, body.staple_foods)

    from db import update_user_health_profile_atomic

    def _mutator(hp):
        if not isinstance(hp, dict):
            hp = {}
        hp["staple_foods"] = cleaned
        return hp

    new_hp = await asyncio.to_thread(update_user_health_profile_atomic, uid, _mutator)
    if new_hp is None:
        raise HTTPException(status_code=404, detail="Perfil no encontrado.")

    return {"staple_foods": cleaned}
