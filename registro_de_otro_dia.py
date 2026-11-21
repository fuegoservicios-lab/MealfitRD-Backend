# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1145 · 2026-11-21] «Ayer me comí otra galleta de Koko» sin hora.

Antes, un registro con `days_ago` y sin hora se guardaba a la hora de AHORA menos N días: a las 4:22 del 21, «ayer»
quedó a las 4:22 del 20 — la primera fila del viernes, antes del almuerzo, y el dueño no la encontró. Ahora:
  · va a una hora TÍPICA de su franja en ese día (`hora_tipica`), y
  · si es picoteo (merienda/snack) y ese día ya tiene uno, se SUMA a él («otra galleta» es más de la misma merienda).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

HORA_TIPICA = {
    "desayuno": (8, 0),
    "almuerzo": (13, 0),
    "merienda": (17, 0),
    "snack": (17, 0),
    "cena": (20, 0),
    "complemento": (10, 0),
    "suplemento": (9, 0),
}
PICOTEO = ("merienda", "snack")


def _corte_h() -> int:
    try:
        from hora_de_la_franja import CORTE_BIOLOGICO_H
        return int(CORTE_BIOLOGICO_H)
    except Exception:  # noqa: BLE001
        return 4


def hora_tipica(meal_type: Optional[str], dia_iso: str, tz_offset_min: int) -> str:
    """ISO UTC de la hora típica de la franja en el día `dia_iso` (hora local del usuario)."""
    h, m = HORA_TIPICA.get(str(meal_type or "").strip().lower(), (13, 0))
    d = date.fromisoformat(str(dia_iso)[:10])
    local = datetime(d.year, d.month, d.day, h, m)
    return (local + timedelta(minutes=int(tz_offset_min))).replace(tzinfo=timezone.utc).isoformat()


def ventana_del_dia(dia_iso: str, tz_offset_min: int) -> tuple:
    """[inicio, fin) UTC del día biológico `dia_iso` (corte a las 4:00 local)."""
    d = date.fromisoformat(str(dia_iso)[:10])
    ini = datetime(d.year, d.month, d.day, _corte_h()) + timedelta(minutes=int(tz_offset_min))
    return ini.replace(tzinfo=timezone.utc), (ini + timedelta(days=1)).replace(tzinfo=timezone.utc)


def picoteo_del_dia(user_id: str, meal_type: Optional[str], dia_iso: str, tz_offset_min: int,
                    *, execute_sql_query: Any = None) -> Optional[dict]:
    """La última merienda/snack de ese día (la fila), o None. Solo para franjas de picoteo."""
    if str(meal_type or "").strip().lower() not in PICOTEO:
        return None
    q = execute_sql_query
    if q is None:
        from db_core import execute_sql_query as q
    ini, fin = ventana_del_dia(dia_iso, tz_offset_min)
    return q(
        "SELECT id, meal_type, meal_name, calories, protein, carbs, healthy_fats, ingredients, consumed_at "
        "FROM consumed_meals WHERE user_id = %s::uuid AND meal_type IN ('merienda', 'snack') "
        "AND consumed_at >= %s AND consumed_at < %s ORDER BY consumed_at DESC LIMIT 1",
        (str(user_id), ini.isoformat(), fin.isoformat()), fetch_one=True) or None


def sumar(fila: dict, *, meal_name: str, calories, protein, carbs, healthy_fats, ingredients) -> dict:
    """Los campos de la fila con lo nuevo sumado (para `db_update_consumed_meal`)."""
    import json
    ing = fila.get("ingredients") or []
    if isinstance(ing, str):
        try:
            ing = json.loads(ing)
        except ValueError:
            ing = [ing]
    if not isinstance(ing, list):
        ing = []
    nuevos = [i for i in (ingredients or [meal_name]) if i]
    nombre = str(fila.get("meal_name") or "")
    return {
        "meal_name": nombre if str(meal_name).lower() in nombre.lower() else f"{nombre} + {meal_name}",
        "calories": int(fila.get("calories") or 0) + int(calories or 0),
        "protein": int(fila.get("protein") or 0) + int(protein or 0),
        "carbs": int(fila.get("carbs") or 0) + int(carbs or 0),
        "healthy_fats": int(fila.get("healthy_fats") or 0) + int(healthy_fats or 0),
        "ingredients": list(dict.fromkeys(list(ing) + nuevos)) or None,
    }
