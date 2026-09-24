# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-196 · 2026-09-24] Sin lácteos, el huevo es un básico: puede repetirse el mismo día con otra técnica.

La causa nº 1 de reintentos que quedaba en las baterías reales: en 4 de 4 corridas de «alergia a lácteos y mariscos»
(rd13, rd15, rd18 y la del 23-sep) el revisor rechazó «MISMA PROTEÍNA REPETIDA EL MISMO DÍA» por el HUEVO — revoltillo
en el desayuno y huevo cocido en la merienda. Sin yogur ni queso, la merienda sólo puede llevar huevo; quitarle la
proteína se probó en el lote 184 y abrió un déficit de proteína (revertido).

El mecanismo ya existe y lo aprobó el dueño («Decisión B», P1-STAPLE-FOODS): un básico declarado puede repetirse el
mismo día si CADA aparición usa una técnica distinta (revoltillo ≠ cocido ≠ tortilla), y nunca si la técnica no se
puede determinar. Aquí sólo se decide que, para quien no puede tomar lácteos (alergia o intolerancia), el huevo cuenta
como básico aunque no lo haya marcado. No aplica si el huevo está vetado (alergia o rechazo) ni en dieta vegana.
Knob `MEALFIT_EGG_STAPLE_WITHOUT_DAIRY` (True). tooltip-anchor: P1-PLAN-LOTE-196-HUEVO-SIN-LACTEOS
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_LACTEOS = ("lacteo", "leche", "lactosa", "dairy", "milk", "caseina")


def enabled() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_EGG_STAPLE_WITHOUT_DAIRY", True)
    except Exception:                                                          # noqa: BLE001
        return True


def etiquetas_extra(form_data) -> set:
    """{'huevo'} si el usuario no puede tomar lácteos y sí huevo; set() en cualquier otro caso (o ante la duda)."""
    try:
        if not (enabled() and isinstance(form_data, dict)):
            return set()
        from constants import strip_accents, alergias_y_rechazos, canonicalize_diet_type
        declarado = [strip_accents(str(a).lower()) for a in alergias_y_rechazos(form_data)]
        otras = strip_accents(str(form_data.get("otherAllergies") or "").lower())
        if not any(t in a for a in declarado + [otras] for t in _LACTEOS):
            return set()
        if canonicalize_diet_type(form_data.get("dietType")) == "vegan":
            return set()
        import graph_orchestrator as go
        if go._allergen_pool_item_banned("Huevo", alergias_y_rechazos(form_data)):
            return set()
        return {"huevo"}
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-196] no-op: {type(e).__name__}: {e}")
        return set()
