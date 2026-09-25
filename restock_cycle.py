"""[P1-CHAT-TOOLS-AUDIT · 2026-09-14] Lo que `POST /api/plans/restock` hace ALREDEDOR de
`restock_inventory`, en un módulo que el chat puede llamar.

EL DEFECTO QUE CIERRA: `tools.mark_shopping_list_purchased` («ya fui al súper», dicho en el chat)
solo llamaba a `restock_inventory`. Las marcas del plan (`is_restocked`, `restocked_at_iso`,
`restocked_items`) las escribía ÚNICAMENTE el endpoint del botón «Ya compré la lista». Tres
consecuencias medibles:

  1. `cron_tasks._first_purchase_pause_applies` lee `is_restocked`: quien compró por chat seguía
     «sin haber comprado nunca» y la generación se le pausaba (`awaiting_first_purchase`).
  2. Sin `restocked_items` no había dedupe por ciclo: una re-emisión del LLM (o decirlo dos veces)
     SUMABA la compra entera otra vez a la Nevera.
  3. La tool pasaba strings (`display_string`) ⇒ ruta legacy de `restock_inventory`, que pierde
     `package_grams` (la fila nace en «paquetes» y ninguna receta la puede descontar).

Por qué un módulo nuevo y no una llamada al endpoint: `routers/plans.py` es un handler HTTP
(`Depends`, `HTTPException`) y no se edita desde este frente. Esto REPLICA su semántica paso a paso
—mismos knobs, misma clave del ledger, mismo mutator atómico con `user_id` (I2 + I7)— y deja la
invocación de `restock_inventory` en el caller (el ancla de `test_p0_restock_dedup_name.py` vive
en `tools.py`). Deuda anotada: `/restock` podría delegar aquí y dejar de ser la otra copia.

Contrato de las funciones: best-effort donde `/restock` es best-effort (marcas, marcas del súper,
limpieza de agotados, descongelado, reproyección) y NUNCA tragándose en silencio lo que decide qué
se compra: si no se puede leer la Nevera, el dedupe cae a «solo por ciclo», igual que el endpoint.

tooltip-anchor: P1-CHAT-TOOLS-AUDIT-RESTOCK-CYCLE
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from constants import strip_accents
from knobs import _env_int

logger = logging.getLogger(__name__)


def purchase_cycle_days() -> int:
    """Mismos knobs y clamps que `/restock` (P1-A · 2026-05-08)."""
    _max_cap = max(7, min(_env_int("MEALFIT_PERISHABLE_CYCLE_DAYS_MAX", 30), 90))
    return max(1, min(_env_int("MEALFIT_PERISHABLE_CYCLE_DAYS", 7), _max_cap))


def purchase_item_name(item) -> str:
    """Nombre del alimento de un ítem de compra: el `name` de un ítem estructurado o, para un
    string («3 lbs de Pollo»), el nombre que `restock_inventory` persistirá por la ruta legacy.
    Así la clave con la que se DEDUPLICA es la misma con la que se MARCA (`persisted_names`)."""
    if isinstance(item, dict):
        return str(item.get("name") or "").strip()
    raw = str(item or "").strip()
    if not raw:
        return ""
    try:
        from shopping_calculator import _parse_quantity
        _q, _u, _n = _parse_quantity(raw)
        return str(_n or raw).strip()
    except Exception:
        return raw


def _ledger_key(name: str) -> str:
    return strip_accents(str(name or "").lower())


def _in_cycle(iso_ts, now_utc: datetime, cycle_days: int) -> bool:
    try:
        ts = iso_ts.replace("Z", "+00:00") if iso_ts.endswith("Z") else iso_ts
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (now_utc - dt).total_seconds() / 86400.0 < cycle_days
    except Exception:
        return False


def filter_purchase_for_cycle(user_id: str, plan_data: Optional[dict], items: list) -> Dict[str, Any]:
    """Dedupe por ciclo de `/restock` (P1-2 + P3-RESTOCK-STALE-DEDUP + P2-RESTOCK-DEDUP-RESPECTS-INVENTORY).

    Devuelve `{"filtered", "skipped", "rebought", "self_heal_reset"}`. Un ítem se SALTA si ya se
    marcó dentro del ciclo Y sigue en la Nevera; si ya no está (se consumió o se botó), volver a
    comprarlo es legítimo. La presencia se resuelve con `constants.pantry_names_match` (SSOT de
    identidad de filas): «Huevos» comprado contra la fila «Huevo» ES el mismo alimento.
    """
    from db import execute_sql_query

    out: Dict[str, Any] = {"filtered": [], "skipped": [], "rebought": [], "self_heal_reset": False}
    existing = (plan_data or {}).get("restocked_items") or {}
    if not isinstance(existing, dict):
        existing = {}

    inv_rows = None
    try:
        inv_rows = execute_sql_query(
            "SELECT ingredient_name, quantity::float8 AS quantity FROM user_inventory WHERE user_id = %s AND kind = 'food'",
            (user_id,), fetch_all=True,
        ) or []
    except Exception as e:
        logger.warning(
            f"[P1-CHAT-TOOLS-AUDIT] lectura de la Nevera falló ({type(e).__name__}); "
            f"dedupe solo por ciclo, sin self-heal."
        )

    # P3-RESTOCK-STALE-DEDUP: Nevera vacía ⇒ el dedupe previo es obsoleto por definición.
    if existing and inv_rows is not None and len(inv_rows) == 0:
        logger.info(f"🧹 [P1-CHAT-TOOLS-AUDIT] Nevera vacía + dedupe previo → reset (user={str(user_id)[:8]})")
        existing = {}
        out["self_heal_reset"] = True

    present_names = None
    if inv_rows is not None:
        present_names = [
            str(r.get("ingredient_name") or "") for r in inv_rows
            if r.get("ingredient_name") and float(r.get("quantity") or 0) > 0
        ]

    def _present(name: str) -> bool:
        if present_names is None:
            return True  # sin lectura: conducta de /restock (dedupe solo por ciclo)
        from constants import pantry_names_match
        return any(pantry_names_match(name, p) for p in present_names)

    now_utc = datetime.now(timezone.utc)
    cycle_days = purchase_cycle_days()
    for it in items or []:
        name = purchase_item_name(it)
        if not name:
            continue
        prev_ts = existing.get(_ledger_key(name))
        if isinstance(prev_ts, str) and _in_cycle(prev_ts, now_utc, cycle_days):
            if _present(name):
                out["skipped"].append(name)
                continue
            out["rebought"].append(name)
        out["filtered"].append(it)
    if out["skipped"]:
        logger.info(
            f"🔁 [P1-CHAT-TOOLS-AUDIT] {len(out['skipped'])} ítem(s) ya registrado(s) en el ciclo "
            f"({cycle_days}d), se saltan: {out['skipped'][:5]}"
        )
    return out


def resolve_purchase_brands(items: list) -> None:
    """P2-NEVERA-BRANDS: `brand_product_id` → `brand` (un SELECT). Fail-open total."""
    try:
        ids = {str(it.get("brand_product_id")) for it in items
               if isinstance(it, dict) and it.get("brand_product_id")}
        if not ids:
            return
        from db import execute_sql_query
        rows = execute_sql_query(
            "SELECT id::text AS id, brand FROM public.supermarket_products WHERE id = ANY(%s::uuid[])",
            (list(ids),), fetch_all=True,
        ) or []
        by_id = {r["id"]: (str(r.get("brand") or "").strip() or "Genérico") for r in rows}
        for it in items:
            if isinstance(it, dict) and it.get("brand_product_id"):
                b = by_id.get(str(it["brand_product_id"]))
                if b:
                    it["brand"] = b
    except Exception as e:
        logger.warning(f"⚠️ [P1-CHAT-TOOLS-AUDIT] marcas no resueltas (fail-open): {e}")


def mark_plan_restocked(user_id: str, plan_id: str, names: List[str], *, self_heal_reset: bool = False) -> bool:
    """Las 3 claves que `/restock` posee, vía `update_plan_data_atomic` (FOR UPDATE + mutator puro
    sobre el plan FRESCO, `user_id=` ⇒ `AND user_id = %s` en SELECT y UPDATE: I2 + I7).
    Marca SOLO lo que persistió (P0-RESTOCK-DEDUP-NAME). True si quedó escrito."""
    if not plan_id or not names:
        return False
    now_iso = datetime.now(timezone.utc).isoformat()
    _names = [n for n in names if n]

    def _mutator(fresh: dict) -> dict:
        fresh["is_restocked"] = True
        fresh["restocked_at_iso"] = now_iso
        ri = {} if self_heal_reset else fresh.get("restocked_items")
        if not isinstance(ri, dict):
            ri = {}
        for nm in _names:
            ri[_ledger_key(nm)] = now_iso
        fresh["restocked_items"] = ri
        return fresh

    try:
        from db import update_plan_data_atomic
        persisted = update_plan_data_atomic(str(plan_id), _mutator, user_id=user_id)
    except Exception as e:
        logger.warning(f"⚠️ [P1-CHAT-TOOLS-AUDIT] no se pudo marcar el plan {plan_id} como comprado: {e}")
        return False
    if not persisted:
        logger.warning(f"⚠️ [P1-CHAT-TOOLS-AUDIT] plan {plan_id} no encontrado para user={str(user_id)[:8]} — sin marca")
        return False
    return True


def after_purchase_side_effects(user_id: str, plan_id: Optional[str], names: List[str]) -> Dict[str, Any]:
    """Los tres efectos best-effort de `/restock` tras una compra que persistió: limpiar los
    «Agotados» repuestos (P3-RESTOCK-DELETE-DEPLETED), descongelar el plan (P1-PLAN-FREEZE) y
    reencolar la proyección de compras (ARQ25-F5). Ninguno puede deshacer la compra."""
    out = {"plan_unfrozen": False}
    try:
        from db import bulk_delete_depleted_items
        bulk_delete_depleted_items(user_id, [n for n in names if n])
    except Exception as e:
        logger.warning(f"⚠️ [P1-CHAT-TOOLS-AUDIT] limpieza de agotados falló (best-effort): {e}")
    try:
        from cron_tasks import try_unfreeze_plan_for_user
        out["plan_unfrozen"] = bool(try_unfreeze_plan_for_user(user_id))
    except Exception as e:
        logger.debug(f"[P1-CHAT-TOOLS-AUDIT] descongelado no-op: {type(e).__name__}: {e}")
    if plan_id:
        try:
            from plan_jobs import enqueue_shopping_reprojection
            enqueue_shopping_reprojection(str(plan_id), user_id, reason="restock")
        except Exception as e:
            logger.debug(f"[P1-CHAT-TOOLS-AUDIT] reproyección no encolada: {e!r}")
    return out
