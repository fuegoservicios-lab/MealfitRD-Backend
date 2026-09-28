# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-734 · 2026-09-28] «queso fresco batido» no hace de un bowl un batido.

Batería real sobre el 659 (adulto mayor con HTA, día 3, desayuno): «Bowl fresco de mango, lechosa y queso fresco batido
con yogurt griego entero» salió con «El Toque de Fuego: coloca todos los ingredientes en la licuadora y licúa…» y un
Montaje que acomoda el mango y la lechosa por separado — la receta se contradice. P1-BATIDO-ADJETIVO (26-jul) enseñó a
`_name_suggests_blended` a descontar «queso crema batido», «yogurt batido»… por lista EXACTA, y P1-BLEND-STEP-REQUIRED
(el que inserta el paso de licuadora) nunca la usó: su regex veía «batido» a secas. Corpus del VPS: 20 nombres con un
«batido» adjetivo que la lista no conocía («queso fresco batido», «queso blanco batido», «yogurt natural batido», «queso
ricotta batido»); 16 llevaban paso de licuadora — incluida una «Tostada integral con queso fresco batido».

Aquí el adjetivo se reconoce por su FORMA: un lácteo o huevo seguido (hasta tres palabras, sin cruzar «y», «con», «al»…)
de «batido/a(s)». `es_batido` es el predicado del paso de licuadora; `sin_adjetivo` lo usa también
`_name_suggests_blended`. «Batido de lechosa», «Licuado de guineo» o «Batido verde con huevo batido» siguen siendo
batidos. tooltip-anchor: P1-PLAN-LOTE-734
"""
from __future__ import annotations

import re

_ADJETIVO_RE = re.compile(
    r"\b(?:queso|quesos|requeson|ricotta|cottage|crema|nata|mantequilla|yogur|yogurt|claras?|huevos?|yemas?)\b"
    r"(?:\s+(?!(?:y|e|o|con|al|a|en|sin|para|sobre)\b)[a-z]+){0,3}?\s+batid[oa]s?\b")
_SUSTANTIVO_RE = re.compile(r"\b(?:batido|licuado|smoothie|frappe|malteada)s?\b")


def sin_adjetivo(nombre_norm) -> str:
    """El nombre (ya en minúscula y sin acentos) sin sus «<lácteo/huevo> … batido»."""
    try:
        return _ADJETIVO_RE.sub(" ", str(nombre_norm or ""))
    except Exception:
        return str(nombre_norm or "")


def es_batido(nombre_norm) -> bool:
    """¿El nombre (en minúscula y sin acentos) dice que el plato ES un batido/licuado?"""
    try:
        return bool(_SUSTANTIVO_RE.search(sin_adjetivo(nombre_norm)))
    except Exception:
        return bool(_SUSTANTIVO_RE.search(str(nombre_norm or "")))


__all__ = ["es_batido", "sin_adjetivo"]
