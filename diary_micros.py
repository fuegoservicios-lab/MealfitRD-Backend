# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-103 · 2026-09-18] El contador de MICROS del día («Micros de hoy»).

Hermano del contador de macros (`TrackingProgress`), con la misma doctrina: la aritmética corre en el servidor
y el cliente pinta. La fuente es lo que YA guarda cada comida registrada: `consumed_meals.ingredients` (lista de
cadenas «150 g de Pechuga de pollo») — la escriben el componedor (líneas del catálogo), «me lo comí» (los
ingredientes del plato del plan) y el coach. Cada cadena se resuelve contra el catálogo con el MISMO resolutor que
usa el informe de micros del plan (`IngredientNutritionDB.micros_from_ingredient_string`), así el contador y el
informe no pueden discrepar sobre el mismo ingrediente.

Cobertura honesta, por diseño: una comida registrada con foto o con macros propias no trae ingredientes → no trae
micros, y el contador lo dice (`micros=None` en esa comida; `micros_coverage` en el total). Mostrar 0 mg de
calcio por un plato escaneado sería mentir con número.

Los OCHO del contador (el dueño eligió esta lista): fibra, sodio, potasio, calcio, hierro, vitamina C, vitamina A
y vitamina D. Las metas salen de `micronutrients.dri_targets` (DRI/IOM + OMS, por sexo/edad/embarazo): el sodio
es TECHO (OMS <2000 mg), el resto SUELO.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from knobs import _env_float

logger = logging.getLogger(__name__)

# [P1-DIARY-MICROS-PLAUSIBLE · ola final · 2026-09-23] `GET /consumed` vuelve a juzgar las comidas del día en CADA
# lectura (el panel refresca al volver a la pestaña, tras registrar…): sin memoria, la MISMA comida inflada dejaba un
# warning por lectura. Se avisa UNA vez por par (kcal de los renglones, kcal de la comida), redondeado, por proceso.
# Con tope: al llenarse se vacía — en el peor caso un aviso se repite, nunca crece sin fin.
_AVISADOS: set = set()
_AVISADOS_MAX = 512
_AVISADOS_LOCK = threading.Lock()


def _avisar_una_vez(kcal_res: float, kcal_m: float) -> None:
    clave = (round(kcal_res), round(kcal_m))
    with _AVISADOS_LOCK:
        if clave in _AVISADOS:
            return
        if len(_AVISADOS) >= _AVISADOS_MAX:
            _AVISADOS.clear()
        _AVISADOS.add(clave)
    logger.warning("[P1-DIARY-MICROS-PLAUSIBLE] micros descartados: renglones %s kcal vs comida %s kcal", *clave)


def _ratio_max() -> float:
    """[P1-DIARY-MICROS-PLAUSIBLE · 2026-09-23] Cuántas veces pueden superar las kcal de los renglones resueltos a las
    kcal guardadas de la comida antes de descartar sus micros. Clamp [1.05, 10]."""
    return _env_float("MEALFIT_DIARY_MICROS_KCAL_RATIO_MAX", 1.5, validator=lambda v: 1.05 <= v <= 10.0)


# Orden de pantalla. Clave del contador → clave del dict que devuelve `micros_from_ingredient_string`.
MICROS_CONTADOR: tuple[tuple[str, str], ...] = (
    ("fiber_g", "fiber"),
    ("sodium_mg", "sodium_mg"),
    ("potassium_mg", "potassium_mg"),
    ("calcium_mg", "calcium_mg"),
    ("iron_mg", "iron_mg"),
    ("vit_c_mg", "vit_c_mg"),
    ("vit_a_mcg", "vit_a_mcg"),
    ("vit_d_mcg", "vit_d_mcg"),
)

CLAVES = tuple(k for k, _ in MICROS_CONTADOR)


def micros_de_ingredientes(ingredients: Any, db) -> Optional[dict]:
    """Micros de UNA comida a partir de sus cadenas de ingredientes.

    Devuelve `None` si la comida no trae ingredientes (foto, macros propias, chat sin desglose): «sin datos»,
    que no es lo mismo que cero. Con ingredientes devuelve `{"values": {clave: total}, "resolved": n, "total": m}`;
    un ingrediente que no resuelve nombre o gramos cuenta en `total` y no en `resolved`."""
    if not isinstance(ingredients, list) or not ingredients:
        return None
    acc = {k: 0.0 for k in CLAVES}
    resolved = 0
    total = 0
    kcal = 0.0
    con_macros = hasattr(db, "macros_from_ingredient_string")
    for ing in ingredients:
        if not isinstance(ing, str) or not ing.strip():
            continue
        total += 1
        try:
            m = db.micros_from_ingredient_string(ing)
        except Exception:
            m = None
        if not m:
            continue
        resolved += 1
        if con_macros:
            try:
                mac = db.macros_from_ingredient_string(ing)
                if mac and mac.get("kcal") is not None:
                    kcal += float(mac["kcal"])
            except Exception:
                pass
        for clave, src in MICROS_CONTADOR:
            v = m.get(src)
            if v is not None:
                acc[clave] += float(v)
    if total == 0:
        return None
    out = {"values": {k: round(v, 1) for k, v in acc.items()}, "resolved": resolved, "total": total}
    if con_macros:
        out["kcal"] = round(kcal, 1)
    return out


def micros_plausibles(micros: Optional[dict], kcal_comida) -> Optional[dict]:
    """[P1-DIARY-MICROS-PLAUSIBLE · 2026-09-23] Descarta los micros de UNA comida cuando sus renglones resueltos pesan
    más energía de la que la comida dice tener. Caso real: «8 rodajas de plátano maduro hervido» = 8 plátanos
    enteros (2.240 g, ~3.000 kcal) dentro de un almuerzo de 695 kcal ⇒ potasio 9,8 g en la tarjeta. Sin kcal de la
    comida o sin kcal resueltas no se juzga (no hay con qué comparar). `None` = «sin datos», que la cobertura ya
    cuenta honestamente («con datos de 1 de 2 comidas»)."""
    if not micros:
        return micros
    try:
        kcal_res = float(micros.get("kcal") or 0.0)
        kcal_m = float(kcal_comida or 0.0)
    except (TypeError, ValueError):
        return micros
    if kcal_m <= 0 or kcal_res <= 0:
        return micros
    if kcal_res > _ratio_max() * kcal_m:
        _avisar_una_vez(kcal_res, kcal_m)
        return None
    return micros


def resumen_micros(meals: list[dict]) -> dict:
    """Totales del día + cobertura: cuántas comidas aportaron datos de las registradas."""
    tot = {k: 0.0 for k in CLAVES}
    con_datos = 0
    for m in meals or []:
        mi = m.get("micros")
        if not mi:
            continue
        con_datos += 1
        for k in CLAVES:
            tot[k] += float((mi.get("values") or {}).get(k) or 0.0)
    return {
        "micros": {k: round(v, 1) for k, v in tot.items()},
        "micros_coverage": {"con_datos": con_datos, "total": len(meals or [])},
    }


def metas_micros(sex: Optional[str], age: Optional[int], pregnant: bool = False) -> dict:
    """Las metas de los ocho, en la forma que pinta el cliente: `{clave: {target, kind, unit}}`.
    `kind` es `floor` (llegar) o `ceiling` (no pasar: el sodio)."""
    from micronutrients import dri_targets
    dri = dri_targets(sex=sex, age=age, pregnant=pregnant)
    out = {}
    for k in CLAVES:
        d = dri.get(k) or {}
        if "ceiling" in d:
            out[k] = {"target": float(d["ceiling"]), "kind": "ceiling", "unit": d.get("unit", "")}
        elif "floor" in d:
            out[k] = {"target": float(d["floor"]), "kind": "floor", "unit": d.get("unit", "")}
    return out
