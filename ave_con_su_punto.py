# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-737 · 2026-09-28] El ave o el cerdo crudos de la lista siempre dicen cuándo están hechos.

Replay de la cola (665) sobre el corpus: en los planes de los últimos 4 días, 88 de 698 comidas con pollo/pavo CRUDO en
la lista (12,6 %) no dicen 74 °C ni «sin partes rosadas» en ningún paso ni nota. Dos causas: (1) el cerrador de proteína
añade «💪 Cocina pechuga de pollo a la plancha o hervida y sírvela como proteína del plato.» DESPUÉS de que se inyectan
las notas de seguridad, y la frase no lleva ni tiempo ni punto; (2) P2-UNDERCOOK-TIME-NOTE sólo pone la nota si el paso
dice un tiempo menor de 5 min, y deja fuera «dora»/«sella» — «dora la pechuga… 3-4 min, volteando para calentar de manera
uniforme» pasaba sin nada (batería real del 636 y corpus).

Aquí, en la cola del contrato, con ave o cerdo CRUDOS en la lista (no cocido, lata, ahumado, rostizado, desmenuzado,
embutido) y ningún criterio en todo el plato, va la nota estándar de P2-UNDERCOOK-TIME-NOTE (el mismo texto: su
idempotencia es la nuestra). La frase del cerrador NO se reescribe a propósito: una docena de sitios la reconocen por su
texto exacto (la fusión de dos cerradores, el 405, el 426, el 449…) y dejarían de deduplicarla en la siguiente pasada.
tooltip-anchor: P1-PLAN-LOTE-737
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

#: el MISMO texto que `graph_orchestrator` (P2-UNDERCOOK-TIME-NOTE): «debe cocinarse por completo» es la idempotencia
NOTA_AVE = ("⚠️ Seguridad alimentaria: el pollo/cerdo debe cocinarse por completo (interior sin partes rosadas, ~74°C). "
            "Si el tiempo indicado no basta para tu corte, extiéndelo hasta que esté bien cocido.")
_AVE_RE = re.compile(r"\b(?:pollo|pechugas?|pavo|muslos?|contramuslos?)\b")
_CERDO_RE = re.compile(r"\b(?:cerdo|chuletas?)\b")
_YA_HECHA_RE = re.compile(r"cocid|\blatas?\b|enlatad|ahumad|rostizad|desmenuzad|jamon|salchich|embutid|longaniza|chorizo|"
                          r"tocino|bacon|caldo|consome|cubito")
_CRITERIO_RE = re.compile(r"\b7[0-9]\s*°|\b7[0-9]\s*grados|\b74\b|165\s*°?\s*f\b|sin\s+partes\s+rosadas|"
                          r"jugos?\s+(?:salgan\s+)?claros|no\s+(?:quede|este|tenga)n?\s+rosad|"
                          r"debe\s+cocinarse\s+por\s+completo")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def asegurar(meal) -> int:
    """1 si añade la nota; 0 si no hace falta o ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lineas = [_sa(x) for x in (meal.get("ingredients") or []) if isinstance(x, str)]
        if not any((_AVE_RE.search(l) or _CERDO_RE.search(l)) and not _YA_HECHA_RE.search(l) for l in lineas):
            return 0
        if _CRITERIO_RE.search(" . ".join(_sa(p) for p in rec if isinstance(p, str))):
            return 0
        rec.append(NOTA_AVE)
        meal["recipe"] = rec
        meal["_food_safety_undercook_time"] = True
        meal.pop("_display", None)
        logger.info(f"🍗 [P1-PLAN-LOTE-737] «{str(meal.get('name'))[:50]}»: el ave/cerdo crudo lleva su punto de cocción")
        return 1
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-737] no-op: {type(e).__name__}: {e}")
        return 0


__all__ = ["asegurar", "NOTA_AVE"]
