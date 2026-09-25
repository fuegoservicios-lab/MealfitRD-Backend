# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-230 · 2026-09-25 · fuera del grafo en P1-PLAN-LOTE-240] G18 para toda condición.

`docs/clinical_enforcement_decisions.md` (G18): SOLO los críticos AGUDOS van al fallback matemático; lo demás se entrega
con banner ámbar. Pero el revisor LLM etiqueta «critical» preocupaciones crónicas que su propio reporte de fact-check
extrapola, y el gate que las degradaba existía solo para DM2 (glucémico) y bariátrico. Batería real del 25-sep: colesterol
alto («6½ yemas en el día 2»), gastritis («ajo, cebolla, vinagre, piña»), hipotiroidismo («yuca y casabe repetidos») →
dos críticos seguidos y el usuario recibía «Pollo y Arroz / Pescado y Batata», 3 comidas, sin sus básicos ni suplementos.
Un crítico SIN ninguna marca aguda (alérgeno, dieta, crudo, embarazo/lactancia, mercurio, celíaca, interacción
farmacológica, renal/potasio/fósforo, hipoglucemia, tiramina) baja a «high»: reintenta con la directiva y, agotados los
intentos, se entrega el plan REAL con banner. El knob `MEALFIT_NON_ACUTE_CRITICAL_SOFT_REJECT` vive en el grafo.
tooltip-anchor: P1-PLAN-LOTE-230-CRITICO-NO-AGUDO
"""
from __future__ import annotations

import unicodedata

# Marcas agudas que se SUMAN a `graph_orchestrator._NON_GLYCEMIC_SAFETY_MARKERS` (el SSOT de los otros dos gates).
_ACUTE_EXTRA_MARKERS = (
    "tiramina", "imao", "crisis hipertensiva", "hipoglucem", "insulina", "sulfonilurea", "litio",
    "anticoagul", "sangrado", "hiperkalem", "hiperpotasem", "atragant", "asfixia",
)


def _sin_acentos(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def _critical_is_non_acute(issues) -> bool:
    """True si NINGUNA razón del crítico lleva una marca de peligro agudo (acentos fuera). Vacío → False."""
    if not issues:
        return False
    try:
        import graph_orchestrator as _go
        marcas = tuple(_go._NON_GLYCEMIC_SAFETY_MARKERS) + _ACUTE_EXTRA_MARKERS
    except Exception:
        return False
    for raw in issues:
        t = _sin_acentos(str(raw).lower())
        if any(m in t for m in marcas):
            return False
    return True
