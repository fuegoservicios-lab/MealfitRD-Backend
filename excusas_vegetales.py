# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-247 · 2026-09-25] «Maní molido hasta obtener una crema» no es un lácteo.

Batería real del 25-sep (vegana, México): la guarda de dieta rechazó CRÍTICO «25 g de maní molido hasta obtener una
crema» como LÁCTEO por la palabra «crema» y quemó un intento (y si reincide, el siguiente paso es el plan de
emergencia). Los escáneres de dieta y de alérgenos solo perdonaban el análogo vegetal DETRÁS del término («crema de
maní», `_PLANT_ADJ_EXCUSE_RX`); aquí el vegetal va DELANTE y la «crema» es el RESULTADO de procesarlo.

Acotado a propósito: solo «crema»/«mantequilla», solo si lo inmediatamente anterior dice que es el resultado
(«hasta obtener/formar/hacer una…», «convertido en…») y solo si antes, en la MISMA línea, hay un fruto seco, semilla o
legumbre. «Fresas con almendras y crema batida» sigue siendo lácteo. tooltip-anchor: P1-PLAN-LOTE-247-CREMA-VEGETAL
"""
from __future__ import annotations

import re

_TERMINOS = ("crema", "mantequilla")
_VEGETAL_RX = re.compile(
    r"\b(?:mani|cacahuate|cacahuete|almendras?|nuez|nueces|avellanas?|maranon|anacardos?|cajuil|pistachos?|"
    r"semillas?|ajonjoli|sesamo|girasol|calabaza|chia|linaza|coco|soya|soja|garbanzos?|avena|aguacate|tahini|"
    r"habichuelas?|frijol(?:es)?|lentejas?|guandules?|gandules?|arvejas?|auyama|batata|yuca|papas?)\b")
_RESULTADO_RX = re.compile(
    r"(?:obtener|formar|hacer|lograr|conseguir|quede|queden|quedar|convertid[oa]s?\s+en|convertir(?:lo|la|los|las)?"
    r"\s+en|reducid[oa]s?\s+a|hecho\s+una?|hecha\s+una?)\s+(?:una?\s+)?$")


def prefijo_vegetal_excusa(termino: str, antes: str) -> bool:
    """True si `termino` (sin acentos) es la crema/mantequilla que RESULTA de moler un vegetal nombrado en `antes`."""
    try:
        t = str(termino or "").strip().lower()
        if not any(t == x or t.startswith(x) for x in _TERMINOS):
            return False
        a = str(antes or "").lower()
        return bool(_RESULTADO_RX.search(a) and _VEGETAL_RX.search(a))
    except Exception:
        return False
