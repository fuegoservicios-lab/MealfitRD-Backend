# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-665 · 2026-09-28] Dos líneas del mismo alimento en FORMAS distintas no se funden en una.

Plan VIVO de producción (6594aae1, leído el 28-sep): «Avena cremosa de guineo y leche» con «440 ml de leche descremada»
en la lista y, en los pasos, «250 ml de leche descremada, 100 g de leche descremada en polvo»; lo mismo en su «Jugo de
chinola». `_merge_duplicate_food_lines` agrupa por el alimento canónico del catálogo y los dos resuelven a «Leche
descremada»: 100 g de leche EN POLVO (~360 kcal, ~36 g de proteína) pasaban a contar como ~100 ml de leche líquida
(~35 kcal). Igual «30 g de habichuelas rojas secas» + «90 g de habichuelas rojas cocidas» → «120 g de Habichuelas
rojas»: gramos de dos bases distintas sumados (el seco rinde ~2,5× al cocerse). Telemetría del corpus: 35 de 231
fusiones juntaban formas; las que falsean macros son las de base (legumbre/grano seco con cocido, leche con leche en
polvo); huevo cocido + huevo es el mismo peso y se sigue fundiendo.

`clave(linea)` es la forma que cambia la base del número: «polvo», «evaporada», «condensada» y, en legumbres y granos,
«cocido» (cocido/hervido/de lata) frente a la base del catálogo, que es la seca. La fusión agrupa por (alimento, clave). tooltip-anchor:
P1-PLAN-LOTE-665
"""
from __future__ import annotations

import re
import unicodedata

_POLVO = re.compile(r"\ben\s+polvo\b")
_EVAPORADA = re.compile(r"\bevaporad[ao]s?\b")
_CONDENSADA = re.compile(r"\bcondensad[ao]s?\b")
_GRANO = re.compile(r"\b(?:habichuelas?|frijol(?:es)?|lentejas?|garbanzos?|gandul(?:es)?|guandul(?:es)?|arvejas?|"
                    r"chicharos?|arroz|quinoa|bulgur|cebada|pasta|fideos?|espaguetis?|macarrones|cuscus|couscous|"
                    r"trigo)\b")
_SECO = re.compile(r"\b(?:sec[oa]s?|crud[oa]s?|en seco|sin cocinar|deshidratad[oa]s?)\b")
_COCIDO = re.compile(r"\b(?:cocid[oa]s?|hervid[oa]s?|precocid[oa]s?|en lata|enlatad[oa]s?|de lata)\b")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def clave(linea) -> str:
    """La forma de `linea` que cambia la base del número; "" si no la dice (o no importa: el huevo cocido pesa lo que el
    crudo)."""
    t = _sa(linea)
    if _POLVO.search(t):
        return "polvo"
    if _EVAPORADA.search(t):
        return "evaporada"
    if _CONDENSADA.search(t):
        return "condensada"
    if _GRANO.search(t) and _COCIDO.search(t) and not _SECO.search(t):
        # la fila del catálogo de granos y legumbres está en SECO (criterio de los lotes 343/542): «½ taza de arroz
        # blanco» y «20 g de arroz blanco crudo» son la misma base y se funden (test_p1_finalize_tail_parity)
        return "cocido"
    return ""


__all__ = ["clave"]
