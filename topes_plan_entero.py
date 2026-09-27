# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-558 · 2026-09-27] Los topes que abarcan el plan ENTERO, también después de cambiar un plato.

Auditoría del formulario: «Cambiar plato», «Actualizar platos» y el coach trabajan sobre un mini-plan de un plato o de un
día, así que los topes que sólo se pueden medir sobre el plan completo no corrían: el pescado de embarazo/lactancia
(≤340 g por semana, `embarazo_pescado`), el casabe en DM2 (`dm2_seguro`) y las yemas con colesterol alto
(`yemas_colesterol`). Una embarazada con ~300 g de pescado en la semana que cambiaba un plato a tilapia pasaba el tope.

Aquí corren los MISMOS topes del generador, en el mismo orden que el escudo previo al guardado (yemas → casabe → etiquetas,
que incluyen el pescado del embarazo), sobre el plan completo con el plato nuevo ya dentro. Son idempotentes y puros
(sin pool: corren dentro del FOR UPDATE). Si algo cambió, los macros de las comidas se re-miden de sus líneas.
tooltip-anchor: P1-PLAN-LOTE-558
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def aplicar(plan, form_data, db=None) -> int:
    """Nº de comidas/líneas tocadas; 0 ante cualquier error o sin contexto clínico (fail-open)."""
    if not isinstance(plan, dict) or not isinstance(form_data, dict) or not form_data:
        return 0
    n = 0
    try:
        if db is None:
            from nutrition_db import IngredientNutritionDB
            db = IngredientNutritionDB()
    except Exception:
        db = None
    for nombre, paso in (
        ("yemas", lambda: __import__("yemas_colesterol").topar_yemas(plan, form_data, db=db)),
        ("casabe", lambda: __import__("dm2_seguro").limitar_casabe(plan, form_data, db=db)),
        ("pescado", lambda: __import__("embarazo_pescado").limitar_pescado(plan, form_data, db=db)),
    ):
        try:
            n += int(paso() or 0)
        except Exception as e:
            logger.debug(f"[P1-PLAN-LOTE-558] tope {nombre} no-op: {type(e).__name__}: {e}")
    if n:
        try:
            from graph_orchestrator import _truth_up_meal_macros_from_strings as _tu
            for d in plan.get("days") or []:
                for m in (d.get("meals") or []) if isinstance(d, dict) else []:
                    if isinstance(m, dict):
                        _tu(m, db)
        except Exception as e:
            logger.debug(f"[P1-PLAN-LOTE-558] truth-up no-op: {type(e).__name__}: {e}")
        logger.info(f"🧮 [P1-PLAN-LOTE-558] topes de plan entero tras un cambio: {n} cambio(s)")
    return n


__all__ = ["aplicar"]
