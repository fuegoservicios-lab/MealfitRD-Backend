# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-193 · 2026-09-24] Topes por LÍNEA que faltaban: semillas/frutos secos ≤ 40 g, pescado en lata ≤ 170 g.

Medido sobre las corridas guardadas (rd2–rd20), sin IA:
  · Semillas y frutos secos: de 270 líneas, 8 pasaban de 40 g y 6 eran del PLAN DE EMERGENCIA — «14,5 cdas de semillas
    de chía» (≈145 g, desayuno de 1.231 kcal) tres veces, «¾ taza de semillas de chía» (≈105 g) otras tres. La plantilla
    de emergencia pone a cada comida sus macros y el cerrador de banda escala lo único denso que encuentra.
  · Pescado en lata: 5 líneas de 280–300 g de atún en una comida — dos en EMBARAZO. El revisor de rd20 (DM2): «la
    porción de atún de 331 g en una sola comida es excesiva… mercurio y sodio». Lo infla el cerrador de proteína.
Los topes de realismo cubrían proteínas en general (techo alto), hierbas, víveres y huevos; no esto. 40 g es el techo
físico que el micro-cerrador ya usaba para semillas/frutos secos (`_SEED_NUT_TOKENS`, SSOT); 170 g, lata y media (la
ración de referencia de la FDA para el atún es 113 g).

Corre en el bucle de topes del escudo pre-INSERT (tras el cierre de banda): lista y raw a la vez, cuantizado, macros
re-medidos. Quedarse corto de proteína o de calorías en ESE día es el lado seguro; 300 g de atún o 145 g de chía no.
Leches, harinas, aceites, yogures y quesos vegetales no son frutos secos. Knobs `MEALFIT_SEED_NUT_LINE_CAP_G` (40) y
`MEALFIT_CANNED_FISH_LINE_CAP_G` (170); 0 = apagado. tooltip-anchor: P1-PLAN-LOTE-193-TOPES-POR-LINEA
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_NO_ES_FRUTO_SECO = ("leche", "bebida", "harina", "aceite", "yogur", "yogurt", "queso", "salsa", "nuez moscada")
_PESCADO_LATA = re.compile(r"\b(?:at[uú]n|sardinas?|caballa|arenque)\b", re.IGNORECASE)
_EN_LATA = re.compile(r"\ben\s+(?:agua|lata|aceite)\b|\benlatad[oa]s?\b|\blata\b|\bescurrid[oa]s?\b", re.IGNORECASE)


def _knob(nombre: str, defecto: int) -> int:
    try:
        from knobs import _env_int
        return max(0, _env_int(nombre, defecto))
    except Exception:                                                          # noqa: BLE001
        return defecto


def _tope_de(linea: str, tokens_semillas, sa):
    """(tope_g, clase) de la línea, o (0, None) si no le toca ninguno."""
    s = sa(str(linea).lower())
    if "al gusto" in s:
        return 0, None
    if _PESCADO_LATA.search(s) and _EN_LATA.search(s):
        return _knob("MEALFIT_CANNED_FISH_LINE_CAP_G", 170), "pescado en lata"
    if not any(t in s for t in _NO_ES_FRUTO_SECO) and any(sa(t) in s for t in tokens_semillas):
        return _knob("MEALFIT_SEED_NUT_LINE_CAP_G", 40), "semillas/frutos secos"
    return 0, None


def cap(days, db=None) -> int:
    """Muta `days`. Devuelve cuántas líneas recortó. Nunca lanza."""
    if not days:
        return 0
    try:
        import graph_orchestrator as go
        from constants import strip_accents as sa
        from nutrition_db import rescale_ingredient_string as resc, quantize_ingredient_string as quant
        if db is None:
            from nutrition_db import IngredientNutritionDB
            db = IngredientNutritionDB()
        tokens = tuple(go._SEED_NUT_TOKENS)
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-193] no-op: {type(e).__name__}: {e}")
        return 0
    n = 0
    for day in days:
        for meal in (day.get("meals") or []) if isinstance(day, dict) else []:
            try:
                ings = meal.get("ingredients") if isinstance(meal, dict) else None
                if not isinstance(ings, list):
                    continue
                tocada = False
                for i, linea in enumerate(list(ings)):
                    if not isinstance(linea, str):
                        continue
                    tope, clase = _tope_de(linea, tokens, sa)
                    if not tope:
                        continue
                    g = db.grams_from_ingredient_string(linea)
                    if not g or g <= tope + 0.5:
                        continue
                    f = tope / float(g)
                    nueva = resc(linea, f)
                    if not nueva or nueva == linea:
                        continue
                    try:
                        nueva = quant(nueva)[0] or nueva          # «3.37 cdas» → «3¼ cdas» (incremento medible)
                    except Exception:                                  # noqa: BLE001
                        pass
                    if isinstance(meal.get("ingredients_raw"), list):   # raw por ALIMENTO (SSOT), antes de tocar la lista
                        go._sync_one_raw_line(meal, i, linea, f)
                    ings[i] = nueva
                    tocada = True
                    n += 1
                    logger.info(f"📏 [P1-PLAN-LOTE-193] '{str(meal.get('name'))[:40]}': «{linea}» → «{nueva}» "
                                f"({clase}, tope {tope} g)")
                if tocada:
                    meal["_line_cap_193"] = True
                    meal.pop("_display", None)
                    try:
                        go._truth_up_meal_macros_from_strings(meal, db)
                    except Exception:                                          # noqa: BLE001
                        pass
            except Exception as e:                                             # noqa: BLE001
                logger.debug(f"[P1-PLAN-LOTE-193] comida no-op: {type(e).__name__}: {e}")
    return n
