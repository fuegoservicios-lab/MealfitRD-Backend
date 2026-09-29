# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-918 · 2026-09-29] En un plato de base DULCE la fruta está en su sitio, aunque lleve huevo al lado.

Baterías reales del 28/29-sep: «Avena cremosa con aguacate, huevo bien cocido y yogurt griego entero» (rdv809, con la ficha
«endulzada con aguacate maduro en cubos y coronada con aguacate fresco») y «Avena cremosa con aguacate y huevos
sancochados» (rdv868). La cadena que lo produce: el cerrador de proteína le pone huevo a la avena, el nombre pasa a decir
«…guayaba y huevo», el detector de pareo fruta + salado (`_meal_has_sweet_savory_clash`, que sólo lee el NOMBRE) ve «huevo
+ guayaba» y el autocorrector cambia la fruta por aguacate. «huevo» entró al vocabulario del detector el 09-sep
(P1-CLASH-HUEVO-Y-VIVERES) por «huevo + mango» en un revoltillo; en una avena el huevo va AL LADO.

Corpus del VPS (8.574 comidas únicas): 680 con la marca `fruit_savory`; 120 empiezan por una base dulce y en 114 de ellas
el único salado del nombre es el huevo; 23 en baterías desde el 27-sep.

Aquí: si el nombre EMPIEZA por una base dulce (avena, yogur, batido, panqueques, porridge, parfait, granola, chía, bowl de
yogur/avena/frutas) y lo único salado que nombra es el huevo, la fruta no choca: ni se rechaza ni se cambia. La «avena
salada» es un plato salado. Con arroz, pasta, coliflor o mangú en el nombre sigue siendo choque. Knob
`MEALFIT_SWEET_BASE_KEEPS_FRUIT` (True). tooltip-anchor: P1-PLAN-LOTE-918
"""
from __future__ import annotations

import re

_BASE_RE = re.compile(r"^\s*(?:avena|yogur|yogurt|batid[oa]s?|smoothie|licuado|panqueques?|pancakes?|porridge|parfait|vasito|"
                      r"granola|pudin de chia|chia|crema de avena|gachas|tostadas? francesas?|"
                      r"bowl(?: [a-z]+){0,2} de (?:yogur|yogurt|avena|frutas?))\b")
_SALADA_RE = re.compile(r"\bsalad[oa]s?\b")
_HUEVO = frozenset({"huevo", "revoltillo", "revuelto"})


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_SWEET_BASE_KEEPS_FRUIT", True)
    except Exception:                                                          # noqa: BLE001
        return True


def fruta_en_su_sitio(name_low: str, salados) -> bool:
    """`name_low`: el nombre en minúscula y sin acentos; `salados`: las bases saladas del detector que el nombre menciona.
    True si la fruta pertenece a la base dulce y lo salado es sólo el huevo de al lado. Nunca lanza."""
    try:
        if not on() or not salados:
            return False
        nombre = str(name_low or "")
        if not _BASE_RE.match(nombre) or _SALADA_RE.search(nombre):
            return False
        return all(str(t) in _HUEVO for t in salados)
    except Exception:                                                          # noqa: BLE001
        return False
