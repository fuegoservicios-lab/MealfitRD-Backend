# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-450 · 2026-09-27] Un ingrediente que el paso nombra por su CLASE o por su nombre regional ya está usado.

`graph_orchestrator._ensure_ingredients_used_in_recipe` (P2-RECIPE-REVERSE-COHERENCE) busca los tokens del nombre de la
lista en los pasos; si no los ve, añade «Incorpora también X durante la preparación». Replay de 322 planes: el paso decía
«Cocina el pescado a la plancha…» con «filete de tilapia» en la lista (6 comidas), o «añade el jitomate» con «tomate» en
la lista del perfil mexicano (7): el reparador añadía «Incorpora también filete de tilapia durante la preparación» a un
plato que ya lo cocinaba —el usuario lee un segundo pescado—. Aquí, la clase del pescado («el pescado», «el filete») y el
regionalismo cuentan como uso. tooltip-anchor: P1-PLAN-LOTE-450
"""
from __future__ import annotations

import re

_PESCADOS_450 = ("tilapia", "mero", "chillo", "salmon", "bacalao", "merluza", "dorado", "corvina", "pargo", "lubina",
                 "carite", "robalo", "basa", "pangasius")
_ALIAS_450 = {p: ("pescado", "filete") for p in _PESCADOS_450}
_ALIAS_450.update({"pescado": ("filete",), "tomate": ("jitomate",), "jitomate": ("tomate",),
                   "habichuela": ("frijol",), "habichuelas": ("frijoles",), "frijol": ("habichuela",),
                   "frijoles": ("habichuelas",), "guineo": ("banana", "banano"), "cambur": ("guineo", "banana")})


def usado_por_alias(stems, recipe_low: str) -> bool:
    """¿Nombra el texto (sin acentos, en minúscula) la clase o el sinónimo regional de alguno de `stems`?"""
    for st in stems or ():
        for a in _ALIAS_450.get(st, ()):
            if re.search(r"\b" + re.escape(a) + r"(?:s|es)?\b", recipe_low or ""):
                return True
    return False
