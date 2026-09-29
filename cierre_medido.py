# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-933 · 2026-09-29] El cerrador de proteína cuenta la línea que ESCRIBE, no la fila del catálogo.

El cerrador (`graph_orchestrator._close_protein_gap_for_meal`) calcula los gramos con la densidad de la FILA del catálogo
(«Lentejas»: 24,6 g de proteína por 100 g, en SECO) y escribe la línea con su participio, «124 g de lentejas cocidas»,
que la base mide cocidas: 100 g de lentejas cocidas son 35 g secas y 8,6 g de proteína. Cree que añadió 31 g y añadió
10,7; el truth-up de después re-mide el plato y el día se queda corto.

Simulado sobre 566 planes del VPS (1.340 cierres, `/tmp/g59/sim_conta.py`): 64 difieren en más de 3 g. En la dirección que
abre déficit: legumbres (habichuelas blancas 10, lentejas 4, gandules 2, habichuelas negras 1), soya texturizada (2) y el
yogurt natural del plato escalado con la densidad del griego (17). Con el lote 932 el cerrador de la compra única vuelve
a tener legumbres entre sus candidatos.

Aquí, antes de calcular los gramos, la ficha del candidato pasa a llevar los macros de lo que se va a ESCRIBIR, por 100 g
de línea: la línea congruente que el plato ya tiene y el cerrador va a escalar (se averigua con el propio
`_scale_congruent_protein_line` sobre una copia: no hay una segunda detección), o la línea nueva «{nombre}{participio}».
Todo lo demás del cerrador —topes, sitio de kcal, mínimos cocinables, los macros que suma al plato— sigue igual y sale
coherente con lo que la base medirá.

Sólo la dirección que entrega MENOS de lo contado (medido < 90 % de la fila). Las carnes y los mariscos van al revés
(100 g de pechuga cocida son 135 g crudos: cuenta 22,5 g y entrega 30,4): corregirlo baja un 26 % los gramos de casi todos
los cierres con carne y pide una batería real; queda anotado, no hecho.
Knob `MEALFIT_CLOSER_COUNTS_WRITTEN_LINE` (True). tooltip-anchor: P1-PLAN-LOTE-933
"""
from __future__ import annotations

import copy
import logging

logger = logging.getLogger(__name__)

_UMBRAL = 0.90
_SONDA_G = 100.0


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_CLOSER_COUNTS_WRITTEN_LINE", True)
    except Exception:                                                          # noqa: BLE001
        return True


def sufijo(nm, no_cook) -> str:
    """El participio con el que el cerrador escribe la línea nueva (« cocidas») o «». La MISMA regla del cerrador, con sus
    mismas pistas; la paridad la vigila `test_el_sufijo_es_el_que_escribe_el_cerrador`."""
    go = __import__("graph_orchestrator")
    try:
        from constants import strip_accents
        limpio = strip_accents(str(nm))
    except Exception:                                                          # noqa: BLE001
        limpio = str(nm)
    ya_cocido = any(h in limpio for h in go._PRECOOKED_PROTEIN_HINT)
    lacteo = any(h in limpio for h in go._NO_COOK_SAFE_PROTEIN_HINT + go._CHEESE_WORDING_HINT)
    if no_cook or ya_cocido or lacteo or "cocid" in limpio:
        return ""
    return " " + go.participio_concordado(str(nm))


class _Medida:
    """La ficha del candidato con los macros de la línea que se escribe; lo demás, de la ficha."""

    def __init__(self, ficha, protein, kcal, carbs, fats):
        self._ficha = ficha
        self.name = ficha.name
        self.protein, self.kcal, self.carbs, self.fats = protein, kcal, carbs, fats

    def __getattr__(self, nombre):
        return getattr(self._ficha, nombre)


def _macros(db, linea) -> dict:
    mc = db.macros_from_ingredient_string(str(linea))
    return mc if isinstance(mc, dict) else {}


def _por_100_g_de_linea(meal, nm, cook, db):
    """(protein, kcal, carbs, fats) que suman 100 g MÁS de la línea que el cerrador va a tocar; `None` si no se puede medir."""
    go = __import__("graph_orchestrator")
    ings = [x for x in (meal.get("ingredients") or [])]
    sonda = {"ingredients": list(ings)}
    raw = meal.get("ingredients_raw")
    if isinstance(raw, list):
        sonda["ingredients_raw"] = copy.deepcopy(raw)
    if go._scale_congruent_protein_line(sonda, nm, _SONDA_G, db):
        tocadas = [(a, b) for a, b in zip(ings, sonda["ingredients"]) if a != b]
        if len(tocadas) != 1:
            return None
        antes, despues = _macros(db, tocadas[0][0]), _macros(db, tocadas[0][1])
        if not antes or not despues:
            return None
        return tuple(float(despues.get(k) or 0) - float(antes.get(k) or 0) for k in ("protein", "kcal", "carbs", "fats"))
    nueva = _macros(db, f"{int(_SONDA_G)} g de {nm}{cook}")
    if not nueva:
        return None
    return tuple(float(nueva.get(k) or 0) for k in ("protein", "kcal", "carbs", "fats"))


def como_se_mide(meal, chosen, no_cook, db):
    """`chosen` tal cual, o su ficha con los macros de la línea que se va a escribir cuando esa línea entrega menos de lo
    que la fila cuenta. Nunca lanza: ante la duda, la cuenta de siempre."""
    try:
        if chosen is None or db is None or not isinstance(meal, dict) or not activo():
            return chosen
        fila_p = float(getattr(chosen, "protein", 0) or 0)
        if fila_p <= 0:
            return chosen
        nm = str(chosen.name).lower()
        medido = _por_100_g_de_linea(meal, nm, sufijo(nm, no_cook), db)
        if not medido or medido[0] <= 0 or medido[1] <= 0 or medido[0] >= _UMBRAL * fila_p:
            return chosen
        logger.info(f"⚖️ [P1-PLAN-LOTE-933] {chosen.name}: la línea que se escribe mide {medido[0]:.1f} g de proteína por "
                    f"100 g y la fila cuenta {fila_p:.1f}: el cerrador cuenta la línea | meal={str(meal.get('name'))[:40]}")
        return _Medida(chosen, *medido)
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-933] no-op: {type(e).__name__}: {e}")
        return chosen


__all__ = ["activo", "sufijo", "como_se_mide"]
