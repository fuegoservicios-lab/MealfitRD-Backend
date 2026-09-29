# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-931 · 2026-09-29] El cerrador de proteína deja para el final lo que el catálogo llama SECO.

Corpus del VPS (566 planes): 15 platos con «N g de guisantes secos cocidos» añadidos por el cerrador y un único texto que
los nombra, «Acompaña con guisantes secos.» — guisantes partidos sin cocer de guarnición, también en una merienda
(«Casabe tostado con mantequilla de maní, canela, nueces y guisantes secos», 80 g). Todos del perfil de compra mensual
con alergia al pescado: desde el día 4 el cerrador sólo puede añadir lo que aguanta el mes (`compra_unica`, lote 521), y
«Guisantes secos» es la legumbre más magra de esa lista, así que gana el desempate de «la más magra».

Las otras legumbres del pool («Lentejas», «Habichuelas blancas») el cerrador las escribe «… cocidas»: se compran cocidas
o de lata y el paso del lote 425 las calienta. La fila que ya se llama «secos» sale «secos cocidos», la base la mide en
seco y nadie la cuece. Aquí va al final del pool: sólo se elige si no hay otro candidato (el piso de proteína gana), y
entonces la nota de cocción previa del lote 375 —que desde este lote conoce el guisante— dice cómo cocerla.
Reorden, no filtro. Knob `MEALFIT_CLOSER_DRY_LAST` (True). tooltip-anchor: P1-PLAN-LOTE-931
"""
from __future__ import annotations

import re

_SECO_RE = re.compile(r"\bsec[oa]s?\b")


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_CLOSER_DRY_LAST", True)
    except Exception:                                                          # noqa: BLE001
        return True


def ordenar(pool):
    """`pool`: [(info, nombre sin acentos en minúscula)]. Devuelve el mismo pool con lo seco al final, en su orden."""
    try:
        if not pool or len(pool) < 2 or not activo():
            return pool
        secos = [c for c in pool if _SECO_RE.search(str(c[1]))]
        if not secos or len(secos) == len(pool):
            return pool
        return [c for c in pool if not _SECO_RE.search(str(c[1]))] + secos
    except Exception:                                                          # noqa: BLE001
        return pool


__all__ = ["activo", "ordenar"]
