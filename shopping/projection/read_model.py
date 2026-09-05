"""[P1-ARQ25-F5-SHOPPING-PROJECTION · 2026-09-04 · extraído P3-SHOPPING-PROJECTION-PKG · 2026-09-05]
Read model de la proyección de compras: una lista por ventana (principal 7/15/30 + top-ups de frescos) con
el MISMO agregador que `/recalculate-shopping-list` (Nevera y consumidos descontados, multiplicador del
hogar persistido en `plan_data`). Determinista, cero LLM. El resultado vive en `plan_jobs.payload.result`
(no en `plan_data`: escribir ahí bumpea `revision` y haría stale a la propia proyección).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

_PROJECTION_ROW_KEYS = ("name", "display_string", "display_name_en", "category", "display_category", "market_qty",
                        "market_unit", "display_qty", "sku_size_label", "base_qty", "base_unit", "is_perishable",
                        "is_staple", "package_grams", "shelf_life_days")
_PROJECTION_COST_KEYS = ("estimated_cost_rd", "estimated_cost", "cost_rd")  # la fila del agregador usa la primera
_PROJECTION_MAX_ITEMS = 150


def _duration_for_days(days: int) -> str:
    d = int(days or 0)
    return "weekly" if d <= 7 else ("biweekly" if d <= 15 else "monthly")


def _row_cost(row: dict) -> Optional[float]:
    for k in _PROJECTION_COST_KEYS:
        v = row.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) > 0:
            return round(float(v), 2)
    return None


def _compact_row(row: dict) -> dict:
    out = {k: row.get(k) for k in _PROJECTION_ROW_KEYS if row.get(k) is not None}
    cost = _row_cost(row)
    if cost is not None:
        out["cost_rd"] = cost
    return out


def build_shopping_projection(plan_data: dict, user_id: str, windows: list, *, revision: Optional[int],
                              policy_hash: Optional[str] = None, max_items: int = _PROJECTION_MAX_ITEMS) -> dict:
    """Read model: una lista por ventana. Principal = ciclo completo (multiplicador de ciclo, como el
    recálculo); `fresh_only` = solo perecederos de ese top-up. Determinista, cero LLM."""
    import shopping_calculator as sc
    try:
        hm = float(plan_data.get("calc_household_multiplier") or 1.0)
    except (TypeError, ValueError):
        hm = 1.0
    hm = max(0.5, min(10.0, hm))
    inv, cons = sc.fetch_inventory_and_consumed_for_plan(user_id, plan_data, is_new_plan=True)
    try:
        trip = sc.active_trip_window_days(plan_data)
    except Exception:
        trip = None
    out_windows = []
    for w in windows or []:
        if not isinstance(w, dict):
            continue
        days = int(w.get("days") or 0)
        if days <= 0:
            continue
        fresh_only = bool(w.get("fresh_only"))
        mult = hm if fresh_only else hm * float(sc.cycle_qty_multiplier(_duration_for_days(days)) or 1.0)
        rows = sc.get_shopping_list_delta(
            user_id, plan_data, is_new_plan=True, structured=True, multiplier=mult,
            inventory_override=inv, consumed_override=cons, cycle_days=days, window_days=trip,
        ) or []
        rows = [r for r in rows if isinstance(r, dict)]
        if fresh_only:
            rows = [r for r in rows if r.get("is_perishable")]
        items = [_compact_row(r) for r in rows][:max_items]
        costs = [c for c in (_row_cost(r) for r in rows) if c is not None]
        out_windows.append({
            "kind": str(w.get("kind") or ("fresh_topup" if fresh_only else "main")),
            "start_day": int(w.get("start_day") or 0), "end_day": int(w.get("end_day") or days),
            "days": days, "cycle_days": int(w.get("cycle_days") or days), "fresh_only": fresh_only,
            "item_count": len(items), "priced_count": len(costs),
            "cost_rd": round(sum(costs), 2) if costs else None,  # None = sin precios, no «gratis»
            "items": items,
        })
    return {
        "schema_version": 1, "revision": revision, "policy_hash": policy_hash,
        "household_multiplier": hm, "computed_at": datetime.now(timezone.utc).isoformat(),
        "windows": out_windows,
    }


__all__ = ["build_shopping_projection", "_compact_row", "_row_cost", "_duration_for_days",
           "_PROJECTION_ROW_KEYS", "_PROJECTION_COST_KEYS", "_PROJECTION_MAX_ITEMS"]
