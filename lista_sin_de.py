# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-912 · 2026-09-29] «½ de pimiento morrón» en la LISTA → «½ pimiento morrón».

El 429 lo arregló en los PASOS y su comentario daba la lista por arreglada; no lo está: batería real rdv868 (mujer que
pierde grasa), desayuno del día 1, lista «½ de pimiento morrón». Corpus del VPS (5.336 comidas): 21 listas («½ de chile
poblano», «½ de puerro en rodajas finas», «½ de pimentón»…). «½» se lee «media»: «media de pimiento» no es español; «¼ de
cebolla» («un cuarto de») sí, y se queda. Sólo el «½», el entero y el entero con «½» ante un alimento (no ante un
artículo: «½ de la cebolla» es otra frase); el decimal de máquina no es de este lote. Sólo display: el crudo no se toca.
Knob `MEALFIT_LIST_HALF_WITHOUT_DE` (True). tooltip-anchor: P1-PLAN-LOTE-912
"""
from __future__ import annotations

import re

_RE = re.compile(r"^(?P<n>\s*(?:\d+\s?½|½|\d+))\s+de\s+(?P<resto>(?!(?:la|el|los|las|un|una|unos|unas)\b)[a-záéíóúñü].*)$",
                 re.IGNORECASE | re.DOTALL)


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_LIST_HALF_WITHOUT_DE", True)
    except Exception:                                                          # noqa: BLE001
        return True


def quitar(linea):
    """La línea sin el «de» que sobra; la misma si no es el caso o ante cualquier error."""
    try:
        if not isinstance(linea, str) or not on():
            return linea
        m = _RE.match(linea)
        if not m:
            return linea
        return f"{m.group('n')} {m.group('resto')}"
    except Exception:                                                          # noqa: BLE001
        return linea
