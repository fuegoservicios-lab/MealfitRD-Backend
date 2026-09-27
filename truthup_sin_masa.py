# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-525 · 2026-09-27] Una hierba o especia escrita SIN cantidad no pesa: no aborta el truth-up del plato.

`graph_orchestrator._truth_up_meal_macros_from_strings` recalcula los macros de una comida sumando sus líneas, y aborta
(deja los números viejos) si una línea cuyo NOMBRE está en el catálogo trae una cantidad que no convierte: así no deja
masa real sin contar. «al gusto»/«una pizca» ya se saltaban (P1-TRUTHUP-ALGUSTO-SKIP), pero «Orégano dominicano» o
«Ajo» —sin ninguna cantidad— abortaban el recálculo de TODA la comida: el orégano seco tiene 265 kcal/100 g y no pasa el
umbral de «despreciable». En el replay de la compra única un almuerzo mostraba 938 kcal y 98 g de proteína cuando sus
líneas sumaban 762 y 68: los números no se recalculaban desde la sustitución.

Sólo hierbas y especias (lo que se echa por pizcas o dientes); el aceite, la cebolla o el limón sin cantidad siguen
abortando, porque sí pueden pesar. tooltip-anchor: P1-PLAN-LOTE-525
"""
from __future__ import annotations

import re

_HIERBAS = frozenset({
    "oregano", "ajo", "sal", "pimienta", "comino", "canela", "cilantro", "perejil", "laurel", "tomillo", "curcuma",
    "curry", "vainilla", "romero", "albahaca", "hierbabuena", "menta", "jengibre", "paprika", "pimenton", "clavo",
    "anis", "sazon", "adobo", "cebollin", "recao", "culantro",
})
# cualquier cifra o medida es una cantidad: «Ajo (2 dientes)» ya no es «sin cantidad»
_CANTIDAD = re.compile(r"[\d½¼¾⅓⅔]|\b(?:tazas?|cdas?|cdtas?|cucharad\w*|cucharadit\w*|g|gr|gramos?|kg|ml|pizcas?|dientes?"
                       r"|ramas?|ramitas?|hojas?|lonjas?|rebanadas?|piezas?|unidad(?:es)?|puñad\w*|manojos?)\b",
                       re.IGNORECASE)


def _sa(t) -> str:
    try:
        from constants import strip_accents
        return strip_accents(str(t or "")).lower()
    except Exception:
        return str(t or "").lower()


def sin_masa(linea) -> bool:
    """«Orégano dominicano», «Ajo», «Pimienta negra» → True; «Aceite de oliva», «Cebolla», «2 dientes de ajo»,
    «Pechuga de pollo al ajo» → False (la cabeza de la línea tiene que ser la hierba)."""
    t = _sa(linea).strip()
    if not t or _CANTIDAD.search(t) or re.match(r"ajo\s*porro", t):      # el ajo porro (puerro) sí pesa
        return False
    cabeza = re.split(r"[\s,;(/]+", t, maxsplit=1)[0]
    return cabeza in _HIERBAS or t.startswith("nuez moscada")


__all__ = ["sin_masa"]
