# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-782 · 2026-09-28] La congruencia del cerrador de proteína, por PALABRA y por lo que IDENTIFICA al alimento.

Corpus de la cola 744 (1.244 desayunos distintos): 86 (6,9 %) llevaban 35-300 g de atún en agua que el modelo no
escribió —su descripción no lo menciona en 84— y 85 se llamaban «…, aguacate y atún en agua». Reproducido con el cerrador
real sobre esos mismos desayunos, sin el atún: `_close_protein_gap_for_meal` declara CONGRUENTE al candidato «Atún en
agua» («esa proteína ya está en el plato: escálala») porque uno de sus tokens es «agua» y `"agua" in meal_text` es verdad
dentro de «AGUAcate». La congruencia decide ANTES que la categoría (huevo o lácteo en el desayuno), así que todo plato con
aguacate recibía atún: 300 g a una embarazada en un mangú.

Es la subcadena que ya se cerró en `_scale_congruent_protein_line` (P2-STEM-BOUNDED, «agua⊄aguacate») y la que el repo
pagó con «pollo»⊂«repollo» y «sal»⊂«salsa». Aquí:
  · por PALABRA completa (singular/plural), nunca por subcadena;
  · lo que DESCRIBE (el líquido de la lata, el color de la habichuela, «secos», «molido», «de hoja», «de freír»…) no
    identifica un alimento por sí solo: sólo cuenta dentro del nombre completo del candidato («queso de hoja» sí; la
    «hoja» de laurel, no; «frutos secos» no son «guisantes secos»).
Knob `MEALFIT_CLOSER_CONGRUENCE_WORD` (True; False = la subcadena de antes). tooltip-anchor: P1-PLAN-LOTE-782
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Palabras que acompañan al nombre de una proteína del catálogo sin decir QUÉ alimento es.
_DESCRIPTORES = frozenset((
    "agua", "aceite", "lata", "latas", "enlatado", "enlatada", "salmuera", "natural",
    "entero", "entera", "enteros", "enteras",
    "rojo", "roja", "rojos", "rojas", "negro", "negra", "negros", "negras",
    "blanco", "blanca", "blancos", "blancas", "verde", "verdes", "pinto", "pintos",
    "seco", "seca", "secos", "secas", "molido", "molida", "texturizado", "texturizada",
    "cocido", "cocida", "cocidos", "cocidas", "claro", "light", "bajo", "sodio", "grasa",
    "descremado", "descremada", "desnatado", "desnatada", "magro", "magra", "ahumado", "ahumada",
    "tierno", "tierna", "fresco", "fresca", "frescos", "frescas",
    "hoja", "hojas", "freir", "griego", "griega", "dominicano", "dominicana", "criollo", "criolla",
))


def _activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_CLOSER_CONGRUENCE_WORD", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _palabras(texto) -> list:
    return [w for w in re.split(r"[^a-z0-9]+", str(texto or "")) if w]


def _patron(tok: str):
    raices = {tok}
    if tok.endswith("es") and len(tok) > 5:
        raices.add(tok[:-2])
    if tok.endswith("s") and len(tok) > 4:
        raices.add(tok[:-1])
    alt = "|".join(sorted((re.escape(r) for r in raices), key=len, reverse=True))
    return re.compile(r"\b(?:" + alt + r")(?:s|es)?\b")


def congruente(nombre_candidato, texto_plato, genericas=()) -> bool:
    """¿El candidato YA está en el plato? Ambos textos llegan en minúsculas y sin acentos (`nlow` y `meal_text` del
    cerrador). El nombre completo cuenta como frase; una palabra suelta, sólo si identifica (no genérica, no
    descriptiva). Ante cualquier error, False: el cerrador sigue por la categoría de la franja."""
    try:
        nombre = str(nombre_candidato or "")
        texto = str(texto_plato or "")
        if not nombre or not texto:
            return False
        if not _activo():
            toks = [t for t in nombre.split() if len(t) >= 4 and t not in genericas]
            return any(t in texto for t in toks)
        palabras = _palabras(nombre)
        # las mismas palabras que miraba la regla de antes; sin ninguna («queso blanco»: todo genérico), nunca congruente
        propias = [t for t in palabras if len(t) >= 4 and t not in genericas]
        if not propias:
            return False
        frase = r"\s+".join(re.escape(p) for p in palabras)
        if re.search(r"\b" + frase + r"\b", texto):
            return True
        return any(_patron(t).search(texto) for t in propias if t not in _DESCRIPTORES)
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-782] no-op: {type(e).__name__}: {e}")
        return False


__all__ = ["congruente"]
