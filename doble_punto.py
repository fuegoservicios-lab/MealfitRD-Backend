# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-634 · 2026-09-28] Un paso no termina en «..».

Validación del 592 (estudiante, día 3): «…el brócoli al lado; toma agua con la comida. Acompaña con pechuga de pollo..».
Corpus de las corridas recientes: 6 de 4.231 comidas con el punto doble, siempre al pegar un «Acompaña con X.» a un
Montaje, y ya presente en `pipeline_result` (antes del shield): varias pasadas pegan frases con su punto y ninguna mira
el de la anterior. Aquí, al final del contrato de receta (que corre en la cola REAL del shield), dos puntos seguidos
pasan a uno; los puntos suspensivos («...») no se tocan. tooltip-anchor: P1-PLAN-LOTE-634
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_DOBLE_634 = re.compile(r"(?<!\.)\.\s*\.(?!\.)")


def limpiar(meal) -> int:
    """Nº de pasos corregidos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list):
            return 0
        n = 0
        for i, p in enumerate(rec):
            if isinstance(p, str) and ".." in p.replace(" ", ""):
                q = _DOBLE_634.sub(".", p)
                if q != p:
                    rec[i] = q
                    n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-634] no-op: {type(e).__name__}: {e}")
        return 0


__all__ = ["limpiar"]
