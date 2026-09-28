# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-786 · 2026-09-28] La cuenta de una pieza vaga sigue a su peso: «¼ pedazo mediano de yuca (≈173 g)».

Bloque 3 real del plan 6594aae1 (28-sep, 14:53 UTC): «¼ pedazo mediano de yuca (≈173 g)» en un almuerzo y «¼ pedazo mediano
de yuca (≈104 g)» en la cena; y en la lista del motor, «0.25 pedazo mediano de yuca (≈173 g)». El resolvedor lee el
PARÉNTESIS (`grams_from_ingredient_string` → 173 g: macros y compra salen de ahí), así que lo que miente es la fracción:
con el pedazo de la tabla casera (400 g) son 0,43 → «½». Un cuarto de pedazo se ve como 100 g y el paso decía 150.

Aquí, al final de la cola: en la lista visible y en la del motor, las líneas «<cantidad> <pieza VAGA de la tabla casera>
(≈N g)» cuya cantidad se aleja un cuarto o más de N / peso de la pieza pasan a esa cantidad (en cuartos, con su singular o
plural); la línea del motor conserva su forma decimal. Sólo unidades vagas (pedazo, porción, pieza, trozo, lonja): las
de tamaño natural no llevan este paréntesis. Nunca fuera de 0,25-10 piezas (el humanizador tampoco las contaría).
Knob `MEALFIT_PIEZA_SIGUE_AL_PESO`. tooltip-anchor: P1-PLAN-LOTE-786
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_FRAC = {"¼": 0.25, "½": 0.5, "¾": 0.75, "⅓": 1.0 / 3.0, "⅔": 2.0 / 3.0}
_TOLERANCIA = 0.35
_LINEA_RE = re.compile(r"^(?P<pre>\s*)(?P<ent>\d+(?:[.,]\d+)?)?\s*(?P<fr>[¼½¾⅓⅔])?\s+(?P<label>[^()]+?)\s*"
                       r"\((?P<aprox>≈\s*)?(?P<g>\d+(?:[.,]\d+)?)\s*g\)(?P<cola>\s*)$")


def _activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PIEZA_SIGUE_AL_PESO", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _cantidad(ent, fr):
    if not ent and not fr:
        return None
    return float((ent or "0").replace(",", ".")) + _FRAC.get(fr or "", 0.0)


def _decimal(q: float) -> str:
    return str(int(q)) if abs(q - int(q)) < 1e-9 else f"{q:g}"


def corregir_linea(linea):
    """La línea con su cantidad corregida, o la misma si no aplica."""
    try:
        s = str(linea)
        m = _LINEA_RE.match(s)
        if not m:
            return linea
        from humanize_ingredients import _LABEL_TO_ENTRY, number_to_fraction_str, strip_accents
        entrada = _LABEL_TO_ENTRY.get(strip_accents(m.group("label").strip().lower()))
        peso = float((entrada or {}).get("weight") or 0.0)
        q = _cantidad(m.group("ent"), m.group("fr"))
        if not entrada or peso <= 0 or not q:
            return linea
        piezas = float(m.group("g").replace(",", ".")) / peso
        if not (0.25 <= piezas <= 10):
            return linea
        nueva = max(0.25, round(piezas * 4) / 4.0)
        # sólo si la cuenta mostrada se equivoca de verdad (>35 % del peso): ¾ para 0,62 piezas es tan buena como ½
        if abs(nueva - q) < 1e-9 or abs(q - piezas) / piezas <= _TOLERANCIA:
            return linea
        etiqueta = entrada["singular"] if nueva <= 1.0 else entrada["plural"]
        decimal = m.group("ent") and not m.group("fr") and re.search(r"[.,]", m.group("ent") or "")
        cifra = _decimal(nueva) if decimal else number_to_fraction_str(nueva)
        return f"{m.group('pre')}{cifra} {etiqueta} ({m.group('aprox') or ''}{m.group('g')} g){m.group('cola')}"
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-786] no-op: {type(e).__name__}: {e}")
        return linea


def ajustar(meal) -> int:
    """Nº de líneas cambiadas (lista visible y del motor); 0 ante cualquier error."""
    try:
        if not isinstance(meal, dict) or not _activo():
            return 0
        n = 0
        for campo in ("ingredients", "ingredients_raw"):
            lineas = meal.get(campo)
            if not isinstance(lineas, list):
                continue
            for i, x in enumerate(lineas):
                if not isinstance(x, str) or "(" not in x:
                    continue
                y = corregir_linea(x)
                if y != x:
                    lineas[i] = y
                    n += 1
        if n:
            meal.pop("_display", None)
            logger.info(f"🥔 [P1-PLAN-LOTE-786] «{str(meal.get('name'))[:50]}»: {n} pieza(s) con la cuenta de su peso")
        return n
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-786] no-op: {type(e).__name__}: {e}")
        return 0


__all__ = ["ajustar", "corregir_linea"]
