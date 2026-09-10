# -*- coding: utf-8 -*-
"""[P1-DESC-CATEGORIA-FALSA · 2026-09-09] El plátano no es un tubérculo, y la ficha lo llamaba así.

## De dónde sale

El dueño miró la tarjeta de su plan vivo `0871ea93` y lo dijo de pasada, después de preguntar por
el mango: *«El plátano maduro no es un tubérculo, eso es otro detalle»*. Tenía razón dos veces —
el plato era de plátano **verde**, y ni el verde ni el maduro son tubérculos: son el fruto de una
musácea. En RD entran en «víveres», que es una categoría **culinaria** y mezcla tubérculos de
verdad (yautía, ñame, batata, yuca) con cosas que no lo son (plátano, guineo, panapén, auyama).

## Lo medido, que fija el tamaño del arreglo

```
comidas con `desc` en planes vivos ......... 72
dicen «tubérculo» .......................... 1
y el plato no lleva ninguno ................ 1   ← el suyo
```

**Uno de 72.** Por eso esto es una lima, no una taxonomía botánica: el módulo mira UNA afirmación
—«tubérculo»— y sólo la borra cuando el plato no lleva ninguno. Ampliarlo a un diccionario de
categorías sería construir maquinaria para un caso, y un guard ruidoso acaba apagado; es la misma
frontera que se midió en [P1-CLASH-HUEVO-Y-VIVERES].

Y una nota sobre el instrumento: la primera sonda buscó la clave `description` —que no existe— y
devolvió «0 casos» de un defecto que sí estaba. El campo es `desc`. *Una sonda que pregunta por
una clave inexistente no mide cero: no mide.*

## Por qué determinista y no una regla del prompt

Hoy quedó medido dos veces que **pedir no es imponer**: «elige EXACTAMENTE uno del catálogo» dio
0 de 12, y «evita mariscos caros» dio cangrejo a RD$958. Añadir una regla más al prompt sería
repetir el experimento esperando otro resultado.

El día determinista **no escribe `desc`**, así que por ese camino el defecto no puede nacer. Esto
cubre el camino del LLM, que es donde nace.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: Lo que la palabra «tubérculo» afirma. Si el plato lleva alguno de éstos, la ficha dice la verdad
#: y no se toca. `papa` incluida: es un tubérculo aunque no sea criollo.
#: tooltip-anchor: _SI_ES_TUBERCULO (test_p1_desc_categoria_falsa.py)
_SI_ES_TUBERCULO = ("yautia", "yautía", "ñame", "name", "mapuey", "batata", "papa", "patata",
                    "yuca", "casabe")

#: La afirmación vigilada, con sus flexiones. Se recorta la frase entera («de tubérculo»,
#: «con tubérculos») porque borrar sólo el sustantivo deja un muñón peor que el error.
_RE_TUBERCULO = re.compile(
    r"\s*(?:,\s*)?\b(?:de|con|a\s+base\s+de|rico\s+en)\s+tub[eé]rculos?\b|\btub[eé]rculos?\b",
    re.IGNORECASE)


def _norm(s: Any) -> str:
    t = str(s or "").lower()
    for a, b in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u")):
        t = t.replace(a, b)
    return t


def lleva_tuberculo(meal: Any) -> bool:
    """¿El plato lleva de verdad algún tubérculo? Mira nombre e ingredientes."""
    if not isinstance(meal, dict):
        return False
    texto = _norm(meal.get("name")) + " " + " ".join(
        _norm(i) for i in (meal.get("ingredients") or meal.get("ingredients_raw") or []))
    return any(_norm(t) in texto for t in _SI_ES_TUBERCULO)


def limpia_desc(meal: Any) -> bool:
    """Quita de `desc` la categoría que el plato no tiene. `True` si tocó algo.

    Sólo borra: no reescribe la ficha ni inventa una categoría de repuesto. Decir de menos es
    recuperable; afirmar de más es lo que el dueño pilló.
    """
    if not isinstance(meal, dict):
        return False
    desc = meal.get("desc")
    if not isinstance(desc, str) or not desc.strip():
        return False
    if not _RE_TUBERCULO.search(desc):
        return False
    if lleva_tuberculo(meal):
        return False                      # la ficha dice la verdad
    nuevo = _RE_TUBERCULO.sub("", desc)
    nuevo = re.sub(r"\s{2,}", " ", nuevo)
    nuevo = re.sub(r"\s+([,.;])", r"\1", nuevo).strip()
    nuevo = re.sub(r"^[,;\s]+", "", nuevo)
    if not nuevo or nuevo == desc:
        return False
    meal["desc"] = nuevo[0].upper() + nuevo[1:] if nuevo[0].islower() else nuevo
    return True


def limpia_plan(plan_data: Any) -> int:
    """Pasa por todas las comidas del plan. Devuelve cuántas fichas se corrigieron. Fail-open."""
    n = 0
    try:
        for d in ((plan_data or {}).get("days") or []):
            for m in ((d or {}).get("meals") or []):
                if limpia_desc(m):
                    n += 1
    except Exception as e:                                             # noqa: BLE001
        logger.debug(f"[P1-DESC-CATEGORIA-FALSA] no-op ({e!r})")
    if n:
        logger.info(f"[P1-DESC-CATEGORIA-FALSA] {n} ficha(s) afirmaban «tubérculo» sin llevarlo")
    return n


__all__ = ["lleva_tuberculo", "limpia_desc", "limpia_plan"]
