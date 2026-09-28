# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-796 · 2026-09-28] «… sin gluten» sólo absuelve al gluten.

La excusa FORWARD del escáner de alérgenos (`graph_orchestrator._GLUTEN_FORWARD_EXCUSE_RX`: «avena certificada sin
gluten», «pan sin gluten») absuelve un término de la clase gluten cuando lo sigue la negación. El escáner trabaja con un
conjunto PLANO de términos prohibidos, así que la excusa no sabía POR QUÉ clase se buscaba el término: si también es de
otra clase declarada —el mole (maní, ajonjolí, frutos secos), la granola (frutos secos), el biscuit y la pizza (lácteos),
el empanizado (huevo)— «2 cdas de mole sin gluten» pasaba limpio para un alérgico al maní. Medido antes del arreglo con
`_scan_allergen_violations`: mole/Mani, granola/Frutos Secos, biscuit/Lácteos, pizza/Lácteos, croquetas/Huevo → [].

Una clase está declarada cuando TODOS sus sinónimos están en el conjunto prohibido (`_expand_allergy_declarations` los
añade enteros al resolverla). Lo consume el escáner en la condición de la excusa forward.
tooltip-anchor: P1-PLAN-LOTE-796-SIN-GLUTEN-SOLO-GLUTEN
"""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def _clases_de_cada_termino() -> dict:
    """término normalizado → sinónimos (normalizados) de cada clase NO gluten que lo busca (sinónimo u oculto)."""
    import graph_orchestrator as go
    from constants import strip_accents
    from vocabulario_alergenos import OCULTOS

    def _n(x) -> str:
        return strip_accents(str(x)).lower().strip()

    out: dict = {}
    for clase, syns in go._ALLERGEN_SYNONYMS.items():
        if clase == "gluten" or not syns:
            continue
        base = frozenset(_n(s) for s in syns)
        for t in (*base, *(_n(x) for x in (OCULTOS.get(clase) or ()))):
            out.setdefault(t, []).append(base)
    return out


def otra_clase_lo_prohibe(termino, prohibidos) -> bool:
    """¿`termino` lo prohíbe también una clase distinta del gluten que las alergias declaradas prohíben entera?

    Ante un fallo del vocabulario responde True: la excusa no se aplica y el término queda marcado (fail-secure)."""
    try:
        prohibidos = prohibidos if isinstance(prohibidos, (set, frozenset)) else set(prohibidos or ())
        return any(base <= prohibidos for base in _clases_de_cada_termino().get(str(termino or ""), ()))
    except Exception:                                                           # noqa: BLE001
        return True
