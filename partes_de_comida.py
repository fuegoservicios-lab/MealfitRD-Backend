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


def nutrition_con_parte_sumada(fila: Any, meal_name: Any, calories: Any, protein: Any, carbs: Any,
                               healthy_fats: Any) -> dict | None:
    """[P1-PLAN-LOTE-1171 · 2026-11-23] Cuando el coach SUMA algo a una comida ya registrada.

    «Add it on yesterday»: el coach corrigió la pechuga de ayer a «Pechuga de pollo a la parrilla + Yogur Ratio
    Protein…» con el total nuevo, y la corrección borraba `nutrition` — la ficha no podía separar el yogur. Si el nombre
    nuevo es el viejo + « + algo» y las cifras solo suben, esto guarda la fila vieja como base (o conserva sus partes) y
    lo sumado como una parte más, con la diferencia de cada macro. Si no es una suma clara, None (conducta previa)."""
    if not isinstance(fila, dict) or not isinstance(meal_name, str):
        return None
    viejo = str(fila.get("meal_name") or "").strip()
    nuevo = meal_name.strip()
    if not viejo or not nuevo.lower().startswith(viejo.lower()):
        return None
    resto = nuevo[len(viejo):]
    if not resto.startswith(" + ") or not resto[3:].strip():
        return None
    claves = (("calories", calories), ("protein", protein), ("carbs", carbs), ("healthy_fats", healthy_fats))
    if calories is None:
        return None
    deltas = {}
    for k, v in claves:
        antes = _num(fila.get(k))
        despues = antes if v is None else _num(v)
        if despues < antes:
            return None
        deltas[k] = despues - antes
    if deltas["calories"] <= 0:
        return None
    previa = fila.get("nutrition") if isinstance(fila.get("nutrition"), dict) else {}
    base = previa.get("base_meal") if partes_de_comida(previa) else None
    addons = list(previa.get("addons") or []) if base else []
    if not base:
        ings = fila.get("ingredients")
        base = {"meal_name": viejo, **{k: _num(fila.get(k)) for k, _ in claves},
                "ingredients": ings if isinstance(ings, list) else []}
    addons.append({"meal_name": resto[3:].strip(), **deltas, "ingredients": []})
    out = {k: v for k, v in previa.items() if k == "attachment_ids"}
    out.update({"base_meal": base, "addons": addons})
    return out
