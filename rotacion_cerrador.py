# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-919 · 2026-09-29] El cerrador de proteína no repite en el día el lácteo dulce que otra comida ya lleva,
si tiene otro equivalente.

Baterías reales del 29-sep: «Avena cremosa de guayaba con chía, leche y yogurt griego entero» al desayuno y «Casabe tostado
con queso fresco, lechosa y yogurt griego entero» en la merienda del mismo día, las dos con el yogurt que añadió el
cerrador. Corpus del VPS (92 planes recientes, 1.170 comidas): de los 135 lácteos dulces que añadió el cerrador, 100 caían
en un día que ya llevaba ese mismo lácteo en otra comida (queso cottage 67 de 84, yogurt griego entero 33 de 51); el yogurt
sale dos veces el mismo día en 76 de 283 días.

La causa: la regla de «no repetir en el día» del cerrador (P1-CLOSER-DAY-AWARE-PROTEIN, 10-jul) compara ETIQUETAS del
gate de proteína repetida, y ese gate sólo conoce carnes, pescados, mariscos y huevo. Para el yogurt, el cottage y la
ricotta —lo que el cerrador pone en el desayuno y la merienda— no hay etiqueta, así que el primero del orden gana siempre.

Aquí esas etiquetas, con cuatro condiciones que salieron de simular la regla sobre el corpus y leer cada cambio (la
versión sin ellas ponía queso gouda sobre la lechosa y huevo en un vasito de yogur):
  1. sólo entre LÁCTEOS DULCES (`graph_orchestrator._SWEET_DAIRY_TOKENS`): el sustituto del yogurt es cottage o ricotta,
     nunca huevo ni un queso de sal;
  2. sólo si el pool del plato —ya filtrado por el guard dulce, la pasta de untar, el embarazo y la Nevera— tiene otro
     lácteo dulce que el día no lleva; sin equivalente libre, el orden de siempre (el piso de proteína gana);
  3. nunca contra lo que el PLATO ya lleva: en un vasito de yogur, más yogur no es una repetición;
  4. en un plato que ya tiene QUESO, el equivalente no puede ser otro queso (cottage, ricotta): la regla del segundo
     queso del cerrador (P1-CLOSER-NO-DUP-CHEESE) lo cambiaría por huevo cocido —73 de 154 cambios sobre 566 planes eran
     «…queso blanco, lechosa y HUEVO»— y, con el huevo en el nombre, el detector de pareo le cambia la fruta al plato.
Es un reorden, no un filtro. Knob `MEALFIT_CLOSER_DAY_FOOD_ROTATION` (True). tooltip-anchor: P1-PLAN-LOTE-919
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_PREFIJO = "alimento:"
# (familia, formas en singular). Paridad con `_SWEET_DAIRY_TOKENS` anclada en el test del lote.
_FAMILIAS = (("yogurt", ("yogur", "yogurt")), ("cottage", ("cottage",)), ("ricotta", ("ricotta", "requeson")))
_PATRONES = tuple((fam, re.compile(r"\b(?:" + "|".join(formas) + r")(?:s|es)?\b")) for fam, formas in _FAMILIAS)


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_CLOSER_DAY_FOOD_ROTATION", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(texto) -> str:
    try:
        from constants import strip_accents
        return strip_accents(str(texto or "")).lower()
    except Exception:                                                          # noqa: BLE001
        return str(texto or "").lower()


def etiquetas(texto) -> set:
    """Las etiquetas `alimento:<familia>` de los lácteos dulces que nombra `texto` (el nombre de un candidato o las líneas
    de un plato), por palabra completa. Vacío con el knob apagado o ante cualquier error."""
    try:
        if not texto or not activo():
            return set()
        limpio = _sa(texto)
        return {_PREFIJO + fam for fam, patron in _PATRONES if patron.search(limpio)}
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-919] no-op: {type(e).__name__}: {e}")
        return set()


def del_plato(meal) -> set:
    """Las etiquetas de un plato: su nombre y sus ingredientes, el mismo texto que lee el gate de proteína repetida."""
    try:
        if not isinstance(meal, dict):
            return set()
        lineas = [str(x) for x in (meal.get("ingredients") or []) if isinstance(x, str)]
        return etiquetas(" . ".join([str(meal.get("name") or "")] + lineas))
    except Exception:                                                          # noqa: BLE001
        return set()


def _texto_del_plato(meal) -> str:
    """Nombre e ingredientes, el `meal_text` con el que el cerrador decide si el plato ya tiene queso."""
    if not isinstance(meal, dict):
        return ""
    return _sa(str(meal.get("name", "")) + " " + " ".join(str(i) for i in (meal.get("ingredients") or [])))


def _tiene_queso(texto) -> bool:
    """Con las MISMAS pistas que la regla del segundo queso del cerrador; sin ella (knob apagado), nunca."""
    try:
        go = __import__("graph_orchestrator")
        return bool(go.CLOSER_NO_DUP_PROTEIN) and any(h in str(texto) for h in go._CLOSER_CHEESE_HINT)
    except Exception:                                                          # noqa: BLE001
        return False


def choca(candidato, meal, pool, usados) -> set:
    """Las etiquetas por las que `candidato` choca con el día: las suyas que otra comida ya lleva (`usados`), salvo que el
    plato también lo lleve o que el `pool` —[(info, nombre sin acentos)], ya filtrado para este plato— no tenga otro
    lácteo dulce libre. Vacío = no choca por esta regla. Nunca lanza."""
    try:
        mias = etiquetas(candidato)
        dia = set(usados or ())
        if not mias or not (mias & dia) or (mias & del_plato(meal)):
            return set()
        con_queso = _tiene_queso(_texto_del_plato(meal))
        for _info, nombre in (pool or ()):
            otras = etiquetas(nombre)
            if not otras or (otras & dia) or (otras & mias):
                continue
            if con_queso and _tiene_queso(_sa(nombre)):
                continue                                   # sería el segundo queso del plato: acabaría en huevo
            return mias
        return set()
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-919] no-op: {type(e).__name__}: {e}")
        return set()


__all__ = ["activo", "etiquetas", "del_plato", "choca"]
