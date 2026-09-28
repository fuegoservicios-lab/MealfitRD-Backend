# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-664 · 2026-09-28] Los gramos entre paréntesis de un paso no contradicen la línea casera de la lista.

Batería real del 28-sep sobre el 636 (estudiante): «prepara ⅓ taza de yogurt natural sin azúcar (155 g)» con «⅓ taza de
yogurt natural sin azúcar» en la lista (y en `ingredients_raw`: los macros cuentan ⅓ de taza, ~80 g), y «corta ¼ plátano
maduro (100 g)» con «¼ plátano maduro» (~70 g). El sincronizador de pistas (P1-STEP-GRAM-HINT-STALE) sólo conoce los
gramos de las líneas que los traen escritos («80 g de avena», «¾ manzana (≈120 g)», lote 310); una línea casera sin
peso no tiene con qué compararse y la pista vieja se queda. Corpus: 32 de 207 pistas así, >40 % fuera («1 taza de yogurt
(125 g)» con ~245, «6 claras de huevo (300 g)» con ~200, «¼ taza de quinoa seca (110 g)» con ~46).

Aquí, cuando un paso repite EXACTAMENTE una línea casera de la lista (misma cantidad y alimento, plurales y acentos
aparte) y le pone unos gramos a más de un 40 % y a más de 15 g de lo que el motor cuenta para esa línea
(`graph_orchestrator._resolve_line_food_grams`, el mismo resolvedor del reconciliador), el paréntesis pasa a «(≈N g)» con
los del motor, redondeados a 5 g. Menos de eso es ruido de densidades y no se toca. tooltip-anchor: P1-PLAN-LOTE-664
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_CON_PESO = re.compile(r"\(\s*[≈~]?\s*\d+(?:[.,]\d+)?\s*(?:g|ml)\s*\)|^\s*\d+(?:[.,]\d+)?\s*(?:g|gr|gramos?|ml)\b",
                       re.IGNORECASE)
_CASERA = re.compile(r"^\s*(?:\d+(?:[.,]\d+)?\s*[½¼¾⅓⅔]?|[½¼¾⅓⅔])\s+\S")
_NOTAS = ("⚠", "💡", "🌱", "⚕", "🤰", "🛒")


def sincronizar(meal) -> int:
    """Nº de pasos reescritos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        import graph_orchestrator as go
        import pasos_cantidades as pq
        lineas = []
        for ln in meal.get("ingredients") or []:
            s = str(ln).strip()
            if not s or _CON_PESO.search(s) or not _CASERA.match(s):
                continue
            g = go._resolve_line_food_grams(s)[1]
            rx = pq._patron_lead(s)
            if g and g > 0 and rx is not None:
                lineas.append((rx, float(g)))
        if not lineas:
            return 0
        n = 0
        for i, paso in enumerate(rec):
            if not isinstance(paso, str) or paso.lstrip().startswith(_NOTAS):
                continue
            base = pq._sa(paso.lower())
            if len(base) != len(paso):
                continue                                   # sin índices alineados no se reescribe
            cambios = []
            for rx, g in lineas:
                for mm in rx.finditer(base):
                    viejo = mm.group("g")
                    if viejo is None:
                        continue
                    v = float(viejo.replace(",", "."))
                    if abs(v - g) <= 15.0 or abs(v - g) <= 0.40 * g:
                        continue
                    nuevo = int(round(g / 5.0) * 5) or int(round(g))
                    cambios.append((mm.start("cola"), mm.end("cola"), f" (≈{nuevo} g)"))
            if not cambios:
                continue
            s = paso
            for ini, fin, txt in sorted(set(cambios), reverse=True):
                s = s[:ini] + txt + s[fin:]
            rec[i] = s
            n += 1
        if n:
            meal.pop("_display", None)
            logger.info(f"⚖️ [P1-PLAN-LOTE-664] «{str(meal.get('name'))[:50]}»: {n} paso(s) con los gramos de su línea")
        return n
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-664] no-op: {type(e).__name__}: {e}")
        return 0


__all__ = ["sincronizar"]
