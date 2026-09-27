# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-565 · 2026-09-27] El paso no pide yema líquida cuando la nota del plato exige yema firme.

Batería real (perfil tipo dueño, día 3): «plancha 2 huevos 2-3 min hasta que la clara cuaje (yema líquida)» y, en el
mismo plato, «⚠️ Seguridad alimentaria: cocina el huevo por completo (≥71°C, yema y clara firmes…)». En el corpus
(4.710 comidas) 13 pasos piden yema líquida o huevo poché; en 6 la nota del plato dice lo contrario («cocina 3 minutos
para una yema líquida» en una avena con huevo poché). Una receta que se contradice obliga al usuario a elegir; la nota es
la política (embarazo y lactancia incluidos), así que el paso se alinea: «(yema líquida)» → «(yema firme)», «hasta que
la clara cuaje» → «hasta que la clara y la yema cuajen» y «cocina 3 minutos para una yema líquida» → «cocina 4-5
minutos, hasta que la yema esté firme». Sin esa nota en el plato no se toca nada. tooltip-anchor: P1-PLAN-LOTE-565
"""
from __future__ import annotations

import re

_NOTA_RE = re.compile(r"yema\s+y\s+clara\s+firmes|yema\s+y\s+clara\s+firme|cocina\s+el\s+huevo\s+por\s+completo", re.IGNORECASE)
_NOTAS = ("⚠", "🤰", "⚕", "💡", "🌱")


def _cambia(p: str) -> str:
    q = re.sub(r"(\b\d+)\s*(?:[-–]\s*\d+\s*)?min(?:utos)?\s+para\s+una\s+yema\s+l[ií]quida",
               "4-5 minutos, hasta que la yema esté firme", p, flags=re.IGNORECASE)
    q = re.sub(r"para\s+una\s+yema\s+l[ií]quida", "hasta que la yema esté firme", q, flags=re.IGNORECASE)
    q = re.sub(r"hasta\s+que\s+la\s+clara\s+cuaje(\s*\(yema\s+l[ií]quida\))",
               "hasta que la clara y la yema cuajen", q, flags=re.IGNORECASE)
    q = re.sub(r"\(yema\s+l[ií]quida\)", "(yema firme)", q, flags=re.IGNORECASE)
    q = re.sub(r"\byema\s+l[ií]quida\b(?!\s*,\s*pero)", "yema firme", q, flags=re.IGNORECASE)
    return q


def ajustar(meal) -> int:
    """Nº de pasos alineados; 0 ante cualquier error o sin la nota de yema firme."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not any(isinstance(p, str) and any(e in p for e in _NOTAS) and _NOTA_RE.search(p)
                                                for p in rec):
            return 0
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or any(e in p for e in _NOTAS):
                continue
            if re.search(r"sin\s+yema\s+l[ií]quida", p, re.IGNORECASE):
                continue                                   # «revuelve… sin yema líquida» ya dice lo correcto
            q = _cambia(p)
            if q != p:
                rec[i] = q
                n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


__all__ = ["ajustar"]
