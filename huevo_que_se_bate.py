# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-562 · 2026-09-27] El huevo que se BATE no está «bien cocido» todavía.

El prompt de embarazo/lactancia pide «huevo bien cocido» y el modelo lo escribe también en el huevo CRUDO de una masa o un
revoltillo: «bate 1 huevo bien cocido con 5 ml de leche» en unos panqueques (batería real rd559, embarazo), «bate 2 huevos
bien cocidos con sal al gusto» (corpus; el revisor rechazó uno: «batir huevos ya cocidos para hacer…», un reintento
entero). Un huevo cocido no se bate. Sólo se toca la mención que el paso BATE («bate 3 huevos y 2 claras de huevo bien
cocidos» → «bate 3 huevos y 2 claras de huevo») y, si el plato bate sus huevos o los hace masa, la LÍNEA del huevo. El
punto de cocción del Toque («hasta que cuajen… huevo BIEN cocido para el embarazo») y las notas ⚠️/🤰 no se tocan: la
primera versión los borraba (replay del corpus). Huevos en mitades, rodajas o pelados son duros: no se tocan.
tooltip-anchor: P1-PLAN-LOTE-562
"""
from __future__ import annotations

import re

_NOTAS = ("⚠", "💡", "🤰", "⚕")
_Q = r"(?:(?:los|las|el|la)\s+)?(?:[\d½¼¾⅓⅔]+\s+)?"
_EGG = r"(?:huevos?|claras?(?:\s+de\s+huevo)?|yemas?)"
_BATE_RE = re.compile(r"\b(?P<v>bate|batir|b[aá]telos?)\s+(?P<obj>" + _Q + _EGG + r"(?:\s+y\s+" + _Q + _EGG + r")?)"
                      r"\s+bien\s+cocid[oa]s?\b", re.IGNORECASE)
_MASA_RE = re.compile(r"\bhuevos?\b[^.;:]{0,80}\bmasa\b|\bmasa\b[^.;:]{0,80}\bhuevos?\b", re.IGNORECASE)
_DURO_RE = re.compile(r"\bhuevos?\b[^.;:]{0,40}\b(?:mitad(?:es)?|rodajas?|pelad[oa]s?|cuartos|dur[oa]s?)\b", re.IGNORECASE)
_LINEA_RE = re.compile(r"\b(huevos?|claras?(?:\s+de\s+huevo)?|yemas?)\s+bien\s+cocid[oa]s?\b", re.IGNORECASE)


def limpiar(meal) -> int:
    """Nº de textos corregidos; 0 ante cualquier error (fail-open)."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list):
            return 0
        pasos = [(i, p) for i, p in enumerate(rec) if isinstance(p, str) and not any(e in p for e in _NOTAS)]
        if any(_DURO_RE.search(p) for _i, p in pasos):
            return 0                                           # huevos duros en el plato: la palabra es cierta
        n = 0
        batido = False
        for i, p in pasos:
            q = _BATE_RE.sub(lambda m: f"{m.group('v')} {m.group('obj')}", p)
            if q != p:
                rec[i] = q
                n += 1
                batido = True
            elif _MASA_RE.search(p) or re.search(r"\b(?:bate|batir|b[aá]telos?)\s+" + _Q + _EGG, p, re.IGNORECASE):
                batido = True
        if batido:
            for k in ("ingredients", "ingredients_raw"):
                lineas = meal.get(k)
                if not isinstance(lineas, list):
                    continue
                for j, x in enumerate(lineas):
                    if isinstance(x, str):
                        y = _LINEA_RE.sub(lambda m: m.group(1), x)
                        if y != x:
                            lineas[j] = y
                            n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


__all__ = ["limpiar"]
