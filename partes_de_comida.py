# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1170 · 2026-11-23] Las PARTES de una comida que se registró junta.

Un usuario diario (captura del 22-nov): su batido «Mass gainer GNC Bulk 1340» y su plato de bistec se fusionaron en
UNA cena (misma franja a <30 min, lote 982) y la app lo pintaba todo pegado: «Mass gainer … y Bistec de res y pechuga
…», un solo total, y en «Ingredientes» el batido ni aparecía. Pidió «que diga su total y cada comida separada».

La fusión YA guardaba cada parte en `consumed_meals.nutrition` (`base_meal` + `addons`); nadie la leía. Esto la lee:
`partes_de_comida(nutrition)` devuelve las partes (nombre + kcal + P/C/G) SOLO si hay más de una. La lista del día
lleva solo los nombres (`nombres_de_partes`); la ficha, las partes completas.
"""
from __future__ import annotations

from typing import Any


def _num(v: Any) -> int:
    try:
        return int(round(float(v or 0)))
    except (TypeError, ValueError):
        return 0


def _parte(d: Any) -> dict | None:
    if not isinstance(d, dict):
        return None
    nombre = str(d.get("meal_name") or "").strip()
    if not nombre:
        return None
    return {
        "meal_name": nombre,
        "calories": _num(d.get("calories")),
        "protein": _num(d.get("protein")),
        "carbs": _num(d.get("carbs")),
        "healthy_fats": _num(d.get("healthy_fats")),
    }


def partes_de_comida(nutrition: Any) -> list[dict]:
    """Las partes de una comida fusionada (la base primero, luego lo añadido), o [] si es una comida sola."""
    if not isinstance(nutrition, dict):
        return []
    addons = nutrition.get("addons")
    if not isinstance(addons, list) or not addons:
        return []
    partes = [p for p in [_parte(nutrition.get("base_meal")), *(_parte(a) for a in addons)] if p]
    return partes if len(partes) > 1 else []


def nombres_de_partes(nutrition: Any) -> list[str]:
    return [p["meal_name"] for p in partes_de_comida(nutrition)]
