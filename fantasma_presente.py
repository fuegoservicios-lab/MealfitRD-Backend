# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-802 · 2026-09-28] El pan fantasma no se materializa si la lista ya trae pan, con la unidad que sea.

Batería real (embarazo, 28-sep, con el 801 desplegado): «Tostadas integrales con queso fresco, mango y yogurt griego
entero» salió con «1 rebanada de pan integral familiar» + «2 rebanadas de pan integral» + «2 rebanadas de pan integral»
(5 rebanadas en la lista y en la compra; los pasos tuestan 2). El pase de carbohidratos fantasma
(`_add_missing_recipe_step_carbs`, P2-STEP-CARB-GHOST) busca en la lista el TOKEN del paso —«lonjas de pan»— y la lista
dice «rebanada»: materializaba «2 lonjas de pan integral» en el grafo y OTRA vez en el escudo del guardado, porque el
humanizador la muestra como «2 rebanadas…» y la siguiente pasada tampoco la encuentra (no era idempotente). Corpus: 11 de
las 13 comidas con una línea repetida (de 5.346), una del plan vivo 92328ff7. El pan ya está si la lista trae CUALQUIER
pan. Knob `MEALFIT_GHOST_BREAD_ANY_UNIT` (True). tooltip-anchor: P1-PLAN-LOTE-802
"""
from __future__ import annotations

import re

#: los tokens de la tabla del fantasma que materializan pan
_PAN_TOKENS = frozenset(("lonjas de pan", "pan integral"))
#: cualquier pan de la lista («1 rebanada de pan integral familiar», «2 lonjas de pan integral», «pan de agua»)
_PAN_EN_LISTA = re.compile(r"\bpan(?:es)?\b")


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_GHOST_BREAD_ANY_UNIT", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _singular(token: str) -> "str | None":
    """«uvas» → «uva», «nueces» → «nuez», «fresas» → «fresa»; None si el token no está en plural."""
    if " " in token or len(token) < 4:
        return None
    if token.endswith("ces"):
        return token[:-3] + "z"
    if token.endswith("s"):
        return token[:-1]
    return None


def presente(token: str, ing_hay: str) -> bool:
    """¿La lista ya trae el alimento de `token` aunque con otra unidad? `ing_hay` va en minúsculas y sin acentos."""
    if not on():
        return False
    if token in _PAN_TOKENS and bool(_PAN_EN_LISTA.search(ing_hay or "")):
        return True
    # [P1-PLAN-LOTE-885 · 2026-09-29] el token de la tabla va en plural («uvas») y la lista en singular («60 g de uva»):
    # el guard de frutas fantasma añadía «¼ taza de uvas (43g)» a comidas que ya llevaban uva (15 en el corpus, planes
    # ES y PR de la batería de 6d: «395 g de uva» + «¼ taza de uvas»). El singular también cuenta.
    sing = _singular(token)
    return bool(sing and on_singular() and re.search(r"\b" + re.escape(sing) + r"\b", ing_hay or ""))


def on_singular() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_GHOST_TOKEN_SINGULAR", True)
    except Exception:                                                          # noqa: BLE001
        return True
