# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-178 · 2026-09-23] DM2: un casabe por bloque de días, no uno por comida.

Batería real del generador (DM2 + insulina): el revisor médico marcó el casabe en LAS CUATRO corridas —«casabe en dos
días», «en el desayuno y nuevamente en la merienda PM»— y cada plan se entregó con el aviso ámbar al usuario («Tu plan
está listo, con un detalle por revisar»). El prompt ya pide «el casabe, como mucho una vez cada 3 días», pero cada día se
genera por separado y el modelo no ve los otros: la frecuencia entre días sólo se puede cumplir después.

Este pase deja el PRIMER casabe del bloque y cambia los demás por una rebanada de pan integral (la misma función en el
plato, más fibra), en la lista, la compra, el nombre, la descripción y los pasos; los macros los re-mide el truth-up.
Si el pan choca con una alergia o un rechazo declarado (gluten, trigo, pan), el casabe se queda. Sólo con la regla
`dm2` activa. En un plan de una sola comida (swap, regenerar un plato) no hay «otro» casabe: no toca nada.
Knob `MEALFIT_DM2_CASABE_CAP` (True). tooltip-anchor: P1-PLAN-LOTE-178-DM2-CASABE
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_CASABE = re.compile(r"\bcasabes?\b", re.IGNORECASE)
_REEMPLAZO = "1 rebanada de pan integral"
_NOMBRE = "Pan integral"


def enabled() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_DM2_CASABE_CAP", True)
    except Exception:                                                          # noqa: BLE001
        return True


def aplica(form_data) -> bool:
    try:
        from condition_rules import detect_active_rules
        return any(getattr(r, "id", "") == "dm2" for r in detect_active_rules(form_data or {}))
    except Exception:                                                          # noqa: BLE001
        return False


def _pan_prohibido(form_data) -> bool:
    try:
        from constants import alergias_y_rechazos
        declarado = alergias_y_rechazos(form_data)
        if not declarado:
            return False
        import graph_orchestrator as go
        return bool(go._scan_allergen_violations({"days": [{"meals": [{"ingredients": [_REEMPLAZO]}]}]}, declarado))
    except Exception:                                                          # noqa: BLE001
        return True      # ante la duda, el casabe se queda


def _tiene_casabe(meal: dict) -> bool:
    return any(isinstance(x, str) and _CASABE.search(x) for x in (meal.get("ingredients") or []))


def limitar_casabe(plan, form_data, db=None) -> int:
    """Devuelve cuántas comidas cambió. Muta `plan`."""
    if not (enabled() and isinstance(plan, dict) and aplica(form_data)):
        return 0
    comidas = [m for d in (plan.get("days") or []) if isinstance(d, dict)
               for m in (d.get("meals") or []) if isinstance(m, dict)]
    con_casabe = [m for m in comidas if _tiene_casabe(m)]
    if len(con_casabe) < 2 or _pan_prohibido(form_data):
        return 0
    try:
        import dish_naming
        import graph_orchestrator as go
        if db is None:
            from nutrition_db import IngredientNutritionDB
            db = IngredientNutritionDB()
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-178] tope de casabe no-op: {type(e).__name__}: {e}")
        return 0
    tocadas = 0
    for m in con_casabe[1:]:
        try:
            for campo in ("ingredients", "ingredients_raw"):
                lineas = m.get(campo)
                if isinstance(lineas, list):
                    m[campo] = [(_REEMPLAZO if isinstance(x, str) and _CASABE.search(x) else x) for x in lineas]
            m["name"] = dish_naming.sustituir_alimento(m, _CASABE, _NOMBRE)
            go._rewrite_recipe_steps_after_subs(m, [(["casabe"], "pan integral")])
            m.pop("_display", None)
            m["_dm2_casabe_cap"] = True
            try:
                go._truth_up_meal_macros_from_strings(m, db)
            except Exception:                                                  # noqa: BLE001
                pass
            tocadas += 1
        except Exception as e:                                                 # noqa: BLE001
            logger.debug(f"[P1-PLAN-LOTE-178] casabe no-op en {str(m.get('name'))[:40]}: {e!r}")
    if tocadas:
        logger.info(f"🩸 [P1-PLAN-LOTE-178] DM2: {tocadas} casabe(s) de más en el bloque → pan integral (queda el primero)")
    return tocadas
