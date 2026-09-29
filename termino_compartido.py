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

[P1-PLAN-LOTE-796 · ronda 3 · 2026-09-28] La misma idea para las demás excusas que sólo valen para UNA clase
(`excusa_de_su_clase`): «pancakes veganos» o «sin huevo» no llevan huevo, pero sí trigo; el muffin inglés es pan sin
huevo; la bechamel con leche de avena no lleva leche de vaca, pero sí harina. Se excusa el término sólo si TODAS las
clases declaradas que lo buscan son de las que la excusa habla. tooltip-anchor: P1-PLAN-LOTE-796-EXCUSA-DE-SU-CLASE
"""
from __future__ import annotations

import functools
import re


def _n(x) -> str:
    from constants import strip_accents
    return strip_accents(str(x or "")).lower().strip()


@functools.lru_cache(maxsize=1)
def _indice() -> dict:
    """término normalizado → [(clase, sinónimos normalizados)] de cada clase que lo busca (sinónimo u oculto)."""
    import graph_orchestrator as go
    from vocabulario_alergenos import OCULTOS

    out: dict = {}
    for clase, syns in go._ALLERGEN_SYNONYMS.items():
        if not syns:
            continue
        base = frozenset(_n(s) for s in syns)
        for t in {*base, *(_n(x) for x in (OCULTOS.get(clase) or ()))}:
            out.setdefault(t, []).append((clase, base))
    return out


def _clases_declaradas(termino, prohibidos) -> set:
    prohibidos = {_n(p) for p in (prohibidos or ())}
    return {c for c, base in _indice().get(_n(termino), ()) if base <= prohibidos}


def otra_clase_lo_prohibe(termino, prohibidos) -> bool:
    """¿`termino` lo prohíbe también una clase distinta del gluten que las alergias declaradas prohíben entera?

    Ante un fallo del vocabulario responde True: la excusa no se aplica y el término queda marcado (fail-secure).
    [P1-PLAN-LOTE-796 · ronda 3] Término y prohibidos se normalizan igual que las clases (sin acentos, minúsculas): el
    escáner los pasa sólo sin acentos y un sinónimo con mayúsculas haría que la excusa volviera a absolver."""
    try:
        return bool(_clases_declaradas(termino, prohibidos) - {"gluten"})
    except Exception:                                                           # noqa: BLE001
        return True


_HUEVO = frozenset({"huevo", "huevos"})
_LACTEO = frozenset({"lacteos", "lactosa"})
_ANIMAL = _HUEVO | _LACTEO | {"pescado", "mariscos"}
_VEGANO_TRAS_RX = re.compile(r"^\s*vegan[oa]s?\b")
_SIN_HUEVO_TRAS_RX = re.compile(r"^\s*(?:\(\s*)?sin\s+huevos?\b")
_MUFFIN_INGLES_RX = re.compile(r"^\s*ingles(?:es|as?)?\b")
_ENGLISH_ANTES_RX = re.compile(r"\benglish\s+$")


def _clases_de_la_excusa(t: str, linea: str, ini: int, fin: int) -> frozenset:
    """Las clases para las que el CONTEXTO de esta aparición excusa el término (vacío: ninguna)."""
    from vocabulario_alergenos import PLATOS
    tras, antes = linea[fin:], linea[:ini]
    if t == "muffin" and (_MUFFIN_INGLES_RX.match(tras) or _ENGLISH_ANTES_RX.search(antes)):
        return _HUEVO                                           # el muffin inglés es pan sin huevo
    if t == "bechamel":
        from excusas_vegetales import leche_vegetal_en
        if leche_vegetal_en(linea):
            return _LACTEO                                      # sin leche de vaca, pero con harina
    if t in PLATOS:
        if _VEGANO_TRAS_RX.match(tras):
            return _ANIMAL                                      # «pizza vegana»: sin queso, con trigo
        if _SIN_HUEVO_TRAS_RX.match(tras):
            return _HUEVO                                       # «pancakes sin huevo»: con trigo
    return frozenset()


def excusa_de_su_clase(termino, linea, ini, fin, prohibidos) -> bool:
    """¿El contexto de esta aparición excusa `termino` para TODAS las clases declaradas que lo buscan?

    Sin una clase declarada que lo busque (alergia escrita literal), no se excusa. Ante un fallo, tampoco (fail-secure)."""
    try:
        t = _n(termino)
        validas = _clases_de_la_excusa(t, str(linea or ""), ini, fin)
        if not validas:
            return False
        clases = _clases_declaradas(t, prohibidos)
        return bool(clases) and clases <= validas
    except Exception:                                                           # noqa: BLE001
        return False
