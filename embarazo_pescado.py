# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-187 · 2026-09-23] Embarazo y lactancia: el pescado y el marisco, 227–340 g por semana (FDA/EPA, 2-3 raciones).

Batería real: el revisor rechazó CRÍTICO «545 g de pescado y mariscos en 3 días» (rd5 → plan de EMERGENCIA) y «460 g de
pescado (120 g de atún claro y 340 g de tilapia) en tres días» (rd15). La regla está en el prompt desde el lote 175 y el
modelo la incumple: cada día se genera por separado y no ve los otros, así que el total sólo se puede cumplir DESPUÉS.

Se recorren los días en orden y, cuando el pescado acumulado pasaría del tope de la ventana (340 g por cada 7 días), el
pescado de ESE plato se cambia por la misma cantidad de una proteína que ese día no se repita (pechuga de pollo,
pechuga de pavo, carne de res, cerdo; ninguna que choque con alergias o rechazos declarados). Sustituir y no recortar:
recortar abre DÉFICIT de proteína (medido en el lote 184). Sólo dieta omnívora — una pescetariana no tiene a qué
cambiar — y sin sustituto limpio el plato se queda como está. Lista, compra, nombre y pasos; los macros los re-mide el
truth-up. Knob `MEALFIT_PREGNANCY_FISH_CAP_G` (340; 0 = apagado). tooltip-anchor: P1-PLAN-LOTE-187-PESCADO-EMBARAZO
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# «dorado» sólo como «filete de dorado»: suelto es también adjetivo («plátano dorado», «pollo dorado»); «bonito», fuera.
_PESCADO = re.compile(r"\b(?:filetes?\s+de\s+dorado|(?:filetes?\s+de\s+)?(?:pescado(?:\s+blanco)?|tilapia|at[uú]n(?:\s+claro)?|"
                      r"salm[oó]n|mero|chillo|pargo|merluza|bacalao|arenque|sardinas?|caballa|corvina|camarones|camar[oó]n|"
                      r"langostinos?|calamar(?:es)?|pulpo|mejillones|langosta|cangrejo))\b", re.IGNORECASE)
_SUSTITUTOS = ("Pechuga de pollo", "Pechuga de pavo", "Carne de res", "Cerdo")


def tope_g() -> int:
    try:
        from knobs import _env_int
        return max(0, _env_int("MEALFIT_PREGNANCY_FISH_CAP_G", 340))
    except Exception:                                                          # noqa: BLE001
        return 340


def aplica(form_data) -> bool:
    try:
        from nutrition_calculator import _is_pregnancy_or_lactation
        from constants import canonicalize_diet_type
        fd = form_data or {}
        return bool(_is_pregnancy_or_lactation(fd)) and canonicalize_diet_type(fd.get("dietType")) == "balanced"
    except Exception:                                                          # noqa: BLE001
        return False


def _gramos(linea: str, db) -> float:
    try:
        return float(db.grams_from_ingredient_string(linea) or 0.0)
    except Exception:                                                          # noqa: BLE001
        return 0.0


def _lineas_pescado(meal: dict) -> list:
    return [i for i, x in enumerate(meal.get("ingredients") or []) if isinstance(x, str) and _PESCADO.search(x)]


def _sustituto(dia: dict, meal: dict, vetos: list, go):
    """El primer sustituto cuya proteína no aparece en OTRA comida del día (mismo SSOT que la puerta de variedad) y que
    no choca con alergias ni rechazos. `None` si no hay ninguno limpio."""
    from constants import strip_accents
    usados: set = set()
    for m in dia.get("meals") or []:
        if m is meal or not isinstance(m, dict):
            continue
        blob = strip_accents((str(m.get("name") or "") + " " + " ".join(str(x) for x in m.get("ingredients") or [])).lower())
        usados |= go._protein_gate_labels_in_text(blob)
    for nombre in _SUSTITUTOS:
        if go._protein_gate_labels_in_text(strip_accents(nombre.lower())) & usados:
            continue
        if vetos and go._allergen_pool_item_banned(nombre, vetos):
            continue
        return nombre
    return None


def limitar_pescado(plan, form_data, db=None) -> int:
    """Devuelve cuántas comidas cambió. Muta `plan`."""
    cap = tope_g()
    if not (cap and isinstance(plan, dict) and aplica(form_data)):
        return 0
    dias = [d for d in (plan.get("days") or []) if isinstance(d, dict) and d.get("meals")]
    if not dias:
        return 0
    try:
        import dish_naming
        import graph_orchestrator as go
        from constants import alergias_y_rechazos
        if db is None:
            from nutrition_db import IngredientNutritionDB
            db = IngredientNutritionDB()
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-187] tope de pescado no-op: {type(e).__name__}: {e}")
        return 0
    tope = cap * max(1.0, len(dias) / 7.0)
    vetos = alergias_y_rechazos(form_data)
    acumulado, tocadas = 0.0, 0
    for dia in sorted(dias, key=lambda d: int(d.get("day") or 0)):
        for meal in dia.get("meals") or []:
            if not isinstance(meal, dict):
                continue
            idx = _lineas_pescado(meal)
            if not idx:
                continue
            gramos = sum(_gramos(meal["ingredients"][i], db) for i in idx)
            if acumulado + gramos <= tope:
                acumulado += gramos
                continue
            nombre = _sustituto(dia, meal, vetos, go)
            if nombre is None:
                acumulado += gramos                  # sin sustituto limpio el plato se queda como está
                continue
            try:
                tokens: set = set()            # los pasos dicen «la tilapia», no «filete de tilapia»: los tres
                for i in idx:
                    g = _PESCADO.search(meal["ingredients"][i]).group(0).lower()
                    base = re.sub(r"^filetes?\s+de\s+", "", g)
                    tokens |= {g, base, base.split()[0]}
                for campo in ("ingredients", "ingredients_raw"):
                    lineas = meal.get(campo)
                    if not isinstance(lineas, list):
                        continue
                    meal[campo] = [(f"{round(_gramos(x, db)) or 100} g de {nombre.lower()}"
                                    if isinstance(x, str) and _PESCADO.search(x) else x) for x in lineas]
                meal["name"] = dish_naming.sustituir_alimento(meal, _PESCADO, nombre.split(" de ")[-1].capitalize()
                                                              if nombre.startswith("Pechuga") else nombre)
                go._rewrite_recipe_steps_after_subs(meal, [(sorted(tokens, key=len, reverse=True), nombre.lower())])
                meal.pop("_display", None)
                meal["_embarazo_pescado_cap"] = nombre
                try:
                    go._truth_up_meal_macros_from_strings(meal, db)
                except Exception:                                              # noqa: BLE001
                    pass
                tocadas += 1
            except Exception as e:                                             # noqa: BLE001
                logger.debug(f"[P1-PLAN-LOTE-187] pescado no-op en {str(meal.get('name'))[:40]}: {e!r}")
    if tocadas:
        logger.info(f"🐟 [P1-PLAN-LOTE-187] embarazo: {tocadas} plato(s) de pescado de más → otra proteína "
                    f"(tope {round(tope)} g en {len(dias)} día(s))")
    return tocadas
