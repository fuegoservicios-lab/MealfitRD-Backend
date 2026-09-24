# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-201 · 2026-09-24] Las anotaciones de la Nevera que lee la IA dejan de mentir.

`db_inventory.get_user_inventory[_net]` le pasa al generador cada alimento de la Nevera con dos avisos:
  · caducidad inferida («[⚠️ URGENTE: Caducado - IA: Prioriza su uso…]») = `shelf_life_days` del catálogo − días desde
    que se registró la compra;
  · agotamiento previsto («[⚠️ PREDICCIÓN: Se agotará en ~N días…]») = cantidad en gramos / consumo diario.

Medido en la Nevera REAL del único usuario con relleno ligado a ella (41 alimentos, compra del 4-sep): 33 salían
«Caducado» —canela, comino, laurel, pimienta, sal, avena, pasta integral, habichuelas secas, soya texturizada, aceite—
porque `master_ingredients.shelf_life_days` vale 14 en 333 de 349 filas (relleno: `pantry_durability.py` ya lo dice), y
el catálogo manda antes que la inferencia por nombre que sí sabía que un seco dura 180 días. Y el huevo salía «se agotará
en ~0 días» con un cartón de 20: la predicción divide la cantidad CRUDA («1» cartón) entre 150 g/día.

  · `plazo`: para lo que el SSOT de durabilidad (`pantry_durability.classify`) clasifica como duradero (`pantry`, `cold`),
    el MAYOR entre el catálogo y el SSOT. Nunca acorta; lo fresco y lo congelable siguen como estaban.
  · `masa_conocida`: la predicción sólo con unidades de masa/volumen (o fruta con su consumo por categoría, que ya es
    por unidad). Con conteos y envases no hay gramos: no se predice nada, en vez de predecir «~0 días».
Knob `MEALFIT_PANTRY_ANNOTATIONS_SSOT` (True).
"""
from __future__ import annotations

_MASA_VOLUMEN = {"g", "gr", "gramo", "gramos", "kg", "kilo", "kilos", "lb", "lbs", "libra", "libras", "oz", "onza",
                 "onzas", "ml", "l", "litro", "litros"}


def activo() -> bool:
    """tooltip-anchor: MEALFIT_PANTRY_ANNOTATIONS_SSOT"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PANTRY_ANNOTATIONS_SSOT", True)
    except Exception:
        return True


def plazo(nombre, categoria, plazo_catalogo):
    """Días de vida para la anotación de caducidad: el del catálogo, salvo que el SSOT diga que dura más (duraderos)."""
    if not activo():
        return plazo_catalogo
    try:
        from pantry_durability import classify
        c = classify(nombre, categoria)
        if c.get("cls") in ("pantry", "cold"):
            return max(int(plazo_catalogo or 0), int(c.get("days_fresh") or 0))
    except Exception:
        pass
    return plazo_catalogo


def masa_conocida(unidad, categoria, tasa_dinamica=None) -> bool:
    """¿Tiene sentido dividir esta cantidad entre el consumo diario? Con masa/volumen, sí. Con conteos o envases, sólo la
    fruta y sólo con la tasa por categoría (1 unidad/día): la tasa dinámica del plan va en GRAMOS/día («3 Limón» entre
    ~12 g/día daba «se agota en ~0 días»)."""
    if not activo():
        return True
    u = str(unidad or "").strip().lower().split(" ")[0].rstrip(".")
    if u in _MASA_VOLUMEN:
        return True
    return "fruta" in str(categoria or "").lower() and not (tasa_dinamica and tasa_dinamica > 0)
