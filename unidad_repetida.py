# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-921 · 2026-09-29] «sella el filete de filete de pescado blanco»: la unidad repetida tras una sustitución
se colapsa en los PASOS al final de la cola.

Batería real (el formulario del dueño, 29-sep, cena D1): el ajuste de presupuesto cambió «salmón» por «Filete de pescado
blanco» y los pasos quedaron «mide 211 g de filete de filete de pescado blanco…» y «sella el filete de filete de pescado
blanco…». El colapso existe (`_dedup_unit_noun_collision`, P2-SUBST-UNIT-DEDUP) pero sólo lo llaman algunos caminos de
sustitución. Aquí, como última palabra sobre los pasos (no notas; nunca el nombre ni la lista). Revisor del 857 (6d):
también «1 pechuga de pechuga de pollo». Knob `MEALFIT_UNIT_REPEAT_COLLAPSE` (True). tooltip-anchor: P1-PLAN-LOTE-921
"""
from __future__ import annotations

import re

_PECHUGA_RE = re.compile(r"\b(pechuga|muslo|lomo)(s)?\s+de\s+\1s?\s+de\s+", re.IGNORECASE)
_NOTA_RE = re.compile(r"^\s*(?:⚠|🤰|⚕|🌱|🛡|nota\b)", re.IGNORECASE)


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_UNIT_REPEAT_COLLAPSE", True)
    except Exception:                                                          # noqa: BLE001
        return True


def colapsar(texto: str) -> str:
    """«filete de filete de pescado» → «filete de pescado»; «pechuga de pechuga de pollo» → «pechuga de pollo»."""
    t = str(texto)
    try:
        import graph_orchestrator as go
        t = go._dedup_unit_noun_collision(t)
    except Exception:                                                          # noqa: BLE001
        pass
    return _PECHUGA_RE.sub(lambda m: f"{m.group(1)}{m.group(2) or ''} de ", t)


def limpiar(meal) -> int:
    """Nº de pasos corregidos; 0 ante cualquier error."""
    try:
        if not on() or not isinstance(meal, dict) or not isinstance(meal.get("recipe"), list):
            return 0
        n = 0
        for i, p in enumerate(meal["recipe"]):
            if isinstance(p, str) and not _NOTA_RE.search(p):
                q = colapsar(p)
                if q != p:
                    meal["recipe"][i] = q
                    n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:                                                          # noqa: BLE001
        return 0
