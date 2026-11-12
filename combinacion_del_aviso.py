# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-984 · 2026-11-11] El aviso de comida no propone mezclas que nadie come.

Caso vivo del dueño (11-nov, 8:30): «Una avena cremosa con pavo y aguacate te caería buenísima para desayunar». El
bloque de su Nevera (lote 960) le pide al modelo proponer con la proteína que tiene (pavo) y la plantilla le sugería
«avena» como variedad: los pegó en UN plato. Una base dulce (avena, yogur, batido, granola, panqueques…) no lleva una
proteína salada de charcutería o de plato fuerte (pavo, pollo, salami, jamón, atún, res…).

`mezcla_rara(texto)` lo detecta en la MISMA frase sin separador de opciones («o», «;», «y de postre»…); el cron
regenera UNA vez con la corrección y, si vuelve a salir, manda el aviso fijo. Knob `MEALFIT_NUDGE_COMBO_GUARD` (True).
tooltip-anchor: P1-PLAN-LOTE-984
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

_DULCES = r"avena|yogur|yogurt|batido|batida|smoothie|granola|panqueque|pancake|cereal|compota|arroz con leche|chia"
_SALADAS = (r"pavo|pollo|salami|jamon|atun|sardina|res|cerdo|chuleta|longaniza|salchicha|tocineta|bacon|camaron|"
            r"pescado|bacalao|carne|chorizo|mortadela")
_RE_DULCE = re.compile(rf"\b(?:{_DULCES})s?\b")
_RE_SALADA = re.compile(rf"\b(?:{_SALADAS})(?:es|s)?\b")
# Lo que separa DOS opciones o dos platos dentro de una frase: ahí ya no es una mezcla.
_ARTICULOS = {"un", "una", "unos", "unas", "su", "tu"}
_SEPARADOR = re.compile(r"\bo\b|\bu\b|;|\bo bien\b|\by de postre\b|\bacompanad[oa] de\b|\bal lado\b|\baparte\b|\bluego\b")

REGLA = ("\nCOMBINACIÓN QUE SE COME DE VERDAD [P1-PLAN-LOTE-984]: la idea es UN plato real de esa comida. Una base "
         "dulce (avena, yogur, batido, granola, panqueques) va con fruta, leche, canela o frutos secos, NUNCA con una "
         "proteína salada (pavo, pollo, salami, jamón, atún, res); si la proteína de su Nevera es salada, ponla con pan, "
         "casabe, tostada, huevos, mangú, arepa o víveres.\n")
CORRECCION = ("\n\n⚠️ CORRECCIÓN OBLIGATORIA: tu propuesta mezcla una base dulce (avena, yogur, batido, granola…) con una "
              "proteína salada (pavo, pollo, salami, jamón, atún, res…) en el mismo plato, y eso no se come así. Propón "
              "UNA combinación que de verdad se come en esa comida: la proteína salada con pan, casabe, tostada, huevos, "
              "mangú, arepa o víveres; o la base dulce con fruta, leche, canela o frutos secos. Escribe el mensaje "
              "completo de nuevo, con el mismo estilo.")


def _norm(t) -> str:
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t).strip()


def mezcla_rara(texto: str) -> Optional[str]:
    """El tramo con la mezcla («avena cremosa con pavo»), o None."""
    for frase in re.split(r"[.!?\n:,]+", _norm(texto)):
        for tramo in _SEPARADOR.split(frase or ""):
            for d in _RE_DULCE.finditer(tramo or ""):
                for s in _RE_SALADA.finditer(tramo):
                    # La base dulce PRIMERO y la salada añadida a ella («avena cremosa con pavo»), pegadas. Al revés
                    # es otro plato («sándwich de pavo con batido», «pollo empanizado con avena»); con un artículo en
                    # medio son dos cosas («… con un batido»).
                    if s.start() <= d.end():
                        continue
                    entre = tramo[d.end():s.start()].split()
                    if len(entre) <= 4 and "con" in entre and not (_ARTICULOS & set(entre)):
                        return tramo[d.start():s.end()].strip()
    return None


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_NUDGE_COMBO_GUARD", True)
    except Exception:                                                          # noqa: BLE001
        return True
