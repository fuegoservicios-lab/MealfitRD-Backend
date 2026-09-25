# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-257 · 2026-09-25] El techo renal de proteína se cumple en lo que se ENTREGA.

Batería rd252, perfil renal + gota (techo KDIGO 0,8 g/kg = 60 g/día): el plan del modelo pasaba el techo, el recorte por
comida «no convergía» y se entregaba el plan de EMERGENCIA… que también lo pasaba (96/75/75 g). Réplica local pase a pase:

  · el recorte (`_trim_day_protein_to_ceiling`) solo escala las líneas de proteína DOMINANTE — carne y pescado. El huevo
    (su grasa pesa más en kcal) y los lácteos no cuentan, así que 4 huevos seguían enteros y la pechuga bajaba a 35 g
    («Pollo y Arroz» con 35 g de pollo);
  · después, `identidad_plato` devolvía el alimento del nombre a su ración («↑35→90 g de Pechuga de pollo») mirando solo
    el margen de kcal y de grasa del día: el techo de proteína no existía para él.

Aquí: qué es recortable bajo un techo renal (lo dominante + huevo, claras y lácteos) y cuánto margen de proteína le queda
a un día con techo. tooltip-anchor: P1-PLAN-LOTE-257-TECHO-RENAL
"""
from __future__ import annotations

import re
import unicodedata

_HUEVO_LACTEO_RX = re.compile(
    r"\b(?:huevos?|claras?|yemas?|queso|quesos|yogur|yogurt|yogures|requeson|cottage|kefir|ricotta|"
    r"leche(?! de (?:coco|almendra|avena|soya|soja|arroz|marañon|maranon)))\b")


def _sa(s) -> str:
    return unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()


def recortable(linea, db, incluye_huevo_lacteo: bool = False) -> bool:
    """Lo que el recorte de proteína puede escalar: lo proteína-dominante y, bajo un techo renal, huevo y lácteos."""
    import graph_orchestrator as go
    if go._ingredient_is_protein_dominant(str(linea), db):
        return True
    return bool(incluye_huevo_lacteo and _HUEVO_LACTEO_RX.search(_sa(linea)))


def retrim_dia(meals, plan_data, db, antes=None) -> bool:
    """[P1-PLAN-LOTE-259 · 2026-09-25] Tras escalar un día hacia su banda de kcal, la proteína vuelve al techo renal.

    Traza dentro del proceso de la batería (rd257, renal + gota): el cierre final de banda del escudo
    (`reconcile_all_macros_band_post_finalize` → `apply_update_macro_engine`) ya no apunta a la proteína con techo renal,
    pero al escalar los carbohidratos para llevar el día de 983 a 1.469 kcal arrastra la proteína que traen dentro (la
    avena de 1 a 1⅔ tazas, el arroz): 60 → 73/87/70 g sobre un techo de 60. El recorte (con huevo y lácteos, lote 257)
    devuelve el día al techo sin deshacer las kcal de los carbohidratos. Devuelve True si recortó.

    `antes` (la proteína del día ANTES del motor): nunca peor que antes. El motor solo deshace lo que ÉL sube; si el día
    ya llegaba sobre el techo, recortarlo es de los recortes de aguas arriba y del gate duro, y el contrato del motor en
    renal es no tocar las líneas proteicas (`test_engine_renal_protein_preserving`, que el gate de 258-261 tumbó).
    tooltip-anchor: P1-PLAN-LOTE-259-MOTOR-RENAL"""
    techo = techo_renal(plan_data)
    if techo <= 0 or not isinstance(meals, list):
        return False
    import graph_orchestrator as go
    if antes is not None:
        techo = max(techo, float(antes or 0.0))
    p = sum(go._meal_macro_num(m.get("protein")) for m in meals if isinstance(m, dict))
    if p <= techo * 1.05:
        return False
    return bool(go._trim_day_protein_to_ceiling(meals, techo, db, ceiling_pct=1.0, incluye_huevo_lacteo=True))


def reenforzar(plan, pg, db=None) -> bool:
    """[P1-PLAN-LOTE-260 · 2026-09-25] Antes de dar por perdido un plan renal que un redondeo, el truth-up o el rebalanceo
    subieron sobre el techo, se vuelve a recortar (el recorte per-comida con huevo y lácteos y, si su reajuste de kcal lo
    re-sube, un recorte sin reajuste) y se verifica. Devuelve True si el plan queda dentro (y deja `meals_enforced`).

    Batería rd257: el plan de la IA para renal + gota quedaba sobre el techo tras la cuantización (Guard 4d), el truth-up
    (8z) y el rebalanceo; esos tres puntos solo marcaban `meals_enforced=False` y el gate duro lo cambiaba por el plan de
    EMERGENCIA, que pasa por los mismos recortes (lotes 257/259) — tirar el plan del modelo ya no compraba seguridad.
    Si aun así no cabe, el marcado y la escalada siguen como estaban. tooltip-anchor: P1-PLAN-LOTE-260-REENFORZAR"""
    try:
        import graph_orchestrator as go
        cap = plan.get("renal_protein_cap") if isinstance(plan, dict) else None
        if not isinstance(cap, dict) or not cap.get("applied") or not pg or float(pg) <= 0:
            return False
        if db is None:
            from nutrition_db import IngredientNutritionDB
            db = IngredientNutritionDB()
        go._enforce_renal_per_meal(plan, float(pg), go._meal_macro_num(plan.get("calories")), db)

        def _dentro():
            return all(sum(go._meal_macro_num(m.get("protein")) for m in (d.get("meals") or []) if isinstance(m, dict))
                       <= float(pg) * 1.05 for d in (plan.get("days") or []) if isinstance(d, dict))
        if not _dentro():
            for d in plan.get("days") or []:
                if isinstance(d, dict):
                    go._trim_day_protein_to_ceiling([m for m in (d.get("meals") or []) if isinstance(m, dict)],
                                                    float(pg), db, ceiling_pct=1.0, incluye_huevo_lacteo=True)
        ok = _dentro()
        cap["meals_enforced"] = ok
        if ok:
            go.logger.info("🫘 [P1-PLAN-LOTE-260] plan renal re-recortado y verificado bajo el techo (no se tira).")
        return ok
    except Exception:
        return False


def techo_renal(plan_data) -> float:
    """Gramos del techo renal de proteína del plan; 0 si no hay techo aplicado."""
    cap = (plan_data or {}).get("renal_protein_cap") if isinstance(plan_data, dict) else None
    if not isinstance(cap, dict) or not cap.get("applied"):
        return 0.0
    try:
        return float(cap.get("protein_g") or 0)
    except (TypeError, ValueError):
        return 0.0
