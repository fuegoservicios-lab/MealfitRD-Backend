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


# [P1-PLAN-LOTE-590 · 2026-09-27] Tres formas más que no pesan y abortaban el recálculo de TODA la comida: «1 pizca de sal
# y pimienta» (el pulido de la lista escribe «1 pizca», y el guard sólo conocía «una pizca»), «jugo de ½ limón» (la cantidad
# va dentro: el lector no la convierte) y «Limón» sin cantidad — el lote 525 lo dejaba abortar «porque puede pesar», pero un
# limón son ~30 kcal y el aborto dejaba los números VIEJOS de la comida: en 11 de 808 comidas de las baterías de esta semana
# los macros mostrados no cuadraban con sus líneas por ≥10 g de proteína (98 g mostrados con 26 g reales; 806 kcal con
# 420). tooltip-anchor: P1-PLAN-LOTE-590
_PIZCA_590 = re.compile(r"^(?:\d+(?:[.,]\d+)?\s*[½¼¾⅓⅔]?|[½¼¾⅓⅔])\s*pizcas?\b")
_JUGO_590 = re.compile(r"^(?:el\s+|un\s+chorrito\s+de\s+)?jugo\s+de\s+(?:(?:\d+(?:[.,/]\d+)?\s*[½¼¾⅓⅔]?|[½¼¾⅓⅔]|un|una|medio|media)\s+)?"
                       r"(?:lim[oó]n(?:es)?|limas?|naranjas?\s+agrias?)\b")
_CITRICO_SOLO_590 = re.compile(r"^(?:lim[oó]n(?:es)?|limas?)\s*$")


def legible(linea):
    """La línea con la cantidad donde el lector la entiende («Limón, 1 unidad» → «1 limón», «Aguacate (¼ unidad)» →
    «¼ aguacate», «1–2 ciruelas» → «1½ ciruelas»), o None si ya estaba bien o no se sabe enderezar. [P1-PLAN-LOTE-590]"""
    try:
        import linea_invertida as li
        return li.enderezar(linea) or li.rango(linea)
    except Exception:
        return None


def sin_masa(linea) -> bool:
    """«Orégano dominicano», «Ajo», «Pimienta negra» → True; «Aceite de oliva», «Cebolla», «2 dientes de ajo»,
    «Pechuga de pollo al ajo» → False (la cabeza de la línea tiene que ser la hierba)."""
    t = _sa(linea).strip()
    if _PIZCA_590.match(t) or _JUGO_590.match(t) or _CITRICO_SOLO_590.match(t):      # [P1-PLAN-LOTE-590]
        return True
    if not t or _CANTIDAD.search(t) or re.match(r"ajo\s*porro", t):      # el ajo porro (puerro) sí pesa
        return False
    cabeza = re.split(r"[\s,;(/]+", t, maxsplit=1)[0]
    return cabeza in _HIERBAS or t.startswith("nuez moscada")


__all__ = ["sin_masa"]
