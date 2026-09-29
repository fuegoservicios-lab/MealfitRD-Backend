# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-861 · 2026-09-29] El sustituto de la fruta de un plato salado no es un alimento que el plato ya tiene.

`_fruit_savory_autofix` (P1-APPETIT-AUTOFIX) cambia la fruta dulce de un desayuno salado por el primer sustituto admitido
de («Aguacate», «Tomate», «Batata»), elegido UNA vez para todo el plan. En un plato que ya llevaba aguacate salía el
aguacate dos veces: «corta ¼ aguacate en gajos y ¼ aguacate maduro en mitades… sirve con el aguacate en gajos y aguacate
fresco al lado», «Tostadas integrales con huevo y aguacate, con aguacate fresco» (batería real DO de 6d, 29-sep, D3
desayuno; corpus: ≈30 de 666 sustituciones, 22 que la fusión de duplicados tuvo que sumar). Aquí el sustituto se elige
por plato: el primero admitido que el plato NO lleva ya. Si los lleva todos, None (el llamador pone la fruta al lado).
Knob `MEALFIT_FRUIT_SAVORY_DISTINCT_SUB` (True). tooltip-anchor: P1-PLAN-LOTE-861
"""
from __future__ import annotations

import re
import unicodedata


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_FRUIT_SAVORY_DISTINCT_SUB", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def elegir(meal: dict, admitidos) -> "str | None":
    """El primer sustituto admitido que el plato no lleva ya (en su lista); None si los lleva todos."""
    cands = [c for c in (admitidos or []) if c]
    if not cands:
        return None
    if not on() or not isinstance(meal, dict):
        return cands[0]
    lista = " ; ".join(_sa(x) for x in (meal.get("ingredients") or []) if isinstance(x, str))
    for c in cands:
        if not re.search(r"\b" + re.escape(_sa(c)) + r"s?\b", lista):
            return c
    return None
