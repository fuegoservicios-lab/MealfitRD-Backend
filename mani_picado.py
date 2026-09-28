# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-781 · 2026-09-28] El maní no se filetea: «maní fileteado» → «maní picado».

Auditoría de la sesión 6d sobre la cola 744 (5.157 comidas del corpus): 79 (1,5 %) dicen «mide 10 g de maní fileteado»
o «Guineo fresco con maní fileteadas y queso cottage». Viene de la sustitución por presupuesto: «almendras fileteadas»
pasa a «maní» y el participio se queda (hasta en femenino). El maní no se lamina: se pica o va entero.

Aquí, en la cola del contrato: «maní fileteado/a(s)» → «maní picado» en el nombre, los pasos y la lista (y en
`ingredients_raw`, cada línea por su texto, nunca por índice). tooltip-anchor: P1-PLAN-LOTE-781
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_MANI_FILETEADO = re.compile(r"\b(?P<m>[Mm]an[ií])\s+fileteado?s?\b|\b(?P<m2>[Mm]an[ií])\s+fileteadas?\b")


def _cambia(t):
    if not isinstance(t, str):
        return t
    return _MANI_FILETEADO.sub(lambda mm: (mm.group("m") or mm.group("m2")) + " picado", t)


def limpiar(meal) -> int:
    """Nº de textos cambiados; 0 ante cualquier error."""
    try:
        if not isinstance(meal, dict):
            return 0
        n = 0
        nombre = meal.get("name")
        nuevo = _cambia(nombre)
        if nuevo != nombre:
            meal["name"] = nuevo
            n += 1
        for campo in ("recipe", "ingredients", "ingredients_raw"):
            lista = meal.get(campo)
            if not isinstance(lista, list):
                continue
            for i, t in enumerate(lista):
                nuevo = _cambia(t)
                if nuevo != t:
                    lista[i] = nuevo
                    n += 1
        if n:
            meal.pop("_display", None)
            logger.info(f"🥜 [P1-PLAN-LOTE-781] «{str(meal.get('name'))[:50]}»: {n} «maní fileteado» → «maní picado»")
        return n
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-781] no-op: {type(e).__name__}: {e}")
        return 0


__all__ = ["limpiar"]
