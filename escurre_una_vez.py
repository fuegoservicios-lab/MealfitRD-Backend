# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-911 · 2026-09-29] Lo que el paso escurre no viene «escurrido».

El paso copia el nombre de la LISTA con su descriptor y queda una orden que se contradice: «escurre 1 taza de lentejas de
lata escurridas (192 g)» (batería real de embarazo rdv801), «escurre ⅔ taza de habichuelas rojas de lata, escurridas y
enjuagadas (≈120 g)» (rdv864), «enjuaga y escurre 1½ tazas de garbanzos cocidos (de lata, enjuagados y escurridos)». Corpus
del VPS (5.336 comidas, salida de la cola con el 888): 9 cláusulas.

Aquí, en la cláusula cuyo verbo es «escurre» (con o sin «enjuaga»), el participio «escurrid-/enjuagad-» del objeto se va
(con su coma o su «y»); si el descriptor decía «enjuagadas» y el verbo no enjuagaba, el verbo pasa a «escurre y enjuaga».
Con otro verbo («mide ½ taza de habichuelas cocidas (de lata, enjuagadas y escurridas)») el descriptor dice cómo viene y
se queda. Las notas no se tocan. Knob `MEALFIT_DRAIN_ONCE` (True). tooltip-anchor: P1-PLAN-LOTE-911
"""
from __future__ import annotations

import re

_NOTAS = ("⚠", "💡", "🌱", "⚕", "🤰")
_VERBO_RE = re.compile(r"\b(?P<v>enjuaga\s+y\s+escurre|escurre\s+y\s+enjuaga|escurre)\b", re.IGNORECASE)
_PART = r"(?:escurrid[oa]s?|enjuagad[oa]s?)"
_GRUPO_RE = re.compile(r",?\s*(?:\by\s+)?\b" + _PART + r"\b(?:\s+y\s+" + _PART + r"\b)?", re.IGNORECASE)
#: otra orden dentro de la misma cláusula: el objeto de «escurre» termina ahí
_OTRA_ORDEN_RE = re.compile(r"\s+y\s+(?:luego\s+)?(?:mezcla|corta|pica|mide|lava|pela|añade|agrega|incorpora|sirve|ten|ralla|"
                            r"exprime|reserva|coloca|cocina|calienta|sofr[ií]e|saltea|maja|tritura|bate|sazona)\w*\b",
                            re.IGNORECASE)


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_DRAIN_ONCE", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _clausula(cl: str) -> str:
    """La cláusula sin el participio repetido; la misma si no hay nada que quitar."""
    v = _VERBO_RE.search(cl)
    if not v:
        return cl
    fin = _OTRA_ORDEN_RE.search(cl, v.end())
    tope = fin.start() if fin else len(cl)
    objeto = cl[v.end():tope]
    enjuaga = False
    for _ in range(3):
        g = _GRUPO_RE.search(objeto)
        if not g:
            break
        enjuaga = enjuaga or "enjuagad" in g.group(0).lower()
        objeto = objeto[:g.start()] + objeto[g.end():]
    if objeto == cl[v.end():tope]:
        return cl
    verbo = v.group("v")
    if enjuaga and "enjuaga" not in verbo.lower():
        verbo = verbo + " y enjuaga"
    return cl[:v.start()] + verbo + re.sub(r"\s{2,}", " ", objeto) + cl[tope:]


def limpiar(meal) -> int:
    """Nº de pasos corregidos; 0 ante cualquier error."""
    try:
        if not on() or not isinstance(meal, dict):
            return 0
        rec = meal.get("recipe")
        if not isinstance(rec, list) or not rec:
            return 0
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or any(e in p for e in _NOTAS):
                continue
            trozos = re.split(r"((?<=[.;])\s+)", p)
            for k in range(0, len(trozos), 2):
                trozos[k] = _clausula(trozos[k])
            s = "".join(trozos)
            if s != p:
                rec[i] = s
                n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:                                                          # noqa: BLE001
        return 0
