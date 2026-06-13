"""[P2-MDDA-PORTION-SOLVER · 2026-06-13] Solver determinista de porciones —
el "lado matemático" del cerebro dividido (MDDA).

Problema: el LLM elige los alimentos de cada comida y porciona "a ojo", lo que
produce drift de macros (sub-entrega de proteína sistemática — ver
`project_plan_quality_degraded_finding_2026_06_13`). Este solver toma los
ingredientes que el LLM eligió + el target de macros del slot, computa los macros
REALES por ingrediente (vía `nutrition_db`), y RE-ESCALA las porciones para clavar
el target — sin tocar QUÉ alimentos eligió el LLM (preserva la creatividad).

Algoritmo v1 — escalado proporcional por grupo de macro dominante (determinista,
sin scipy):
  1. Por ingrediente resoluble, clasificar por su macro DOMINANTE (mayor aporte
     calórico: 4·P vs 4·C vs 9·F).
  2. Por cada macro {protein, carbs, fats}: factor = target_macro / Σ(macro en su
     grupo), clamp a [min_scale, max_scale]. Escalar la cantidad de cada
     ingrediente del grupo por ese factor (los macros escalan lineal con la
     cantidad, sea cual sea la unidad → escalamos `quantity` directo).
  3. Ingredientes no-resolubles (sin macros / sin gramos) se dejan intactos.

Por qué proporcional y no LP/scipy: el 80% del valor (cerrar el déficit de
proteína 110g→154g) se logra con escalado por grupo, sin dependencia pesada ni
soluciones no-deterministas. Si la telemetría muestra que grupos acoplados
(un ingrediente que es P y C a la vez) necesitan optimización conjunta, se añade
LP entonces — no antes (convención del repo: no diseñar para requisitos hipotéticos).
"""
from __future__ import annotations

from typing import Optional


# Aportes calóricos Atwater (kcal/g) por macro — para decidir el macro dominante.
_KCAL_PER_G = {"protein": 4.0, "carbs": 4.0, "fats": 9.0}


def _get(d: dict, *keys, default=0.0):
    if not isinstance(d, dict):
        return default
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return default


def _coerce_line(ing) -> tuple:
    """ing dict o string → (quantity, unit, name). Tolerante a aliases de key."""
    if isinstance(ing, dict):
        qty = ing.get("quantity", ing.get("qty", ing.get("amount")))
        unit = ing.get("unit", "unidad")
        name = ing.get("name") or ing.get("ingredient_name") or ing.get("item_name") or ""
        return qty, unit, name
    # string "150 g pechuga de pollo" → delega el parseo al shopping_calculator
    try:
        from shopping_calculator import _parse_quantity
        q, u, n = _parse_quantity(ing, apply_yield_multiplier=False)
        return q, u, n
    except Exception:
        return None, "unidad", str(ing)


def _target_macros(target: dict) -> dict:
    return {
        "kcal": _get(target, "kcal", "cals", "calories", "target_calories"),
        "protein": _get(target, "protein", "protein_g", "proteina"),
        "carbs": _get(target, "carbs", "carbs_g", "carbohidratos"),
        "fats": _get(target, "fats", "fat", "fats_g", "grasas"),
    }


def solve_portion_macros(
    ingredients: list,
    target: dict,
    db=None,
    *,
    min_scale: float = 0.3,
    max_scale: float = 3.5,
    tolerance_pct: float = 0.10,
) -> dict:
    """Re-escala porciones para clavar el target de macros del slot.

    Args:
        ingredients: lista de dicts {name, quantity, unit} (o strings parseables).
        target: macros objetivo del slot {kcal, protein, carbs, fats} (acepta aliases).
        db: IngredientNutritionDB; si None se instancia uno (carga master_ingredients).
        min_scale/max_scale: clamp del factor por grupo (evita porciones absurdas).
        tolerance_pct: para reportar `converged` (|achieved-target|/target ≤ tol).

    Returns:
        dict con:
          - ingredients: lista re-escalada (mismas keys de entrada, `quantity` ajustada).
          - achieved: {kcal,protein,carbs,fats} reales tras el escalado (solo resolubles).
          - target: macros objetivo normalizados.
          - report: por macro {current, target, factor, applied}.
          - resolved_count / unresolved: cuántos ingredientes se pudieron computar.
          - converged: bool (todos los macros con target>0 dentro de tolerancia).
    """
    if db is None:
        from nutrition_db import IngredientNutritionDB
        db = IngredientNutritionDB()
    tgt = _target_macros(target)

    # 1) computar macros por ingrediente + clasificar por macro dominante.
    entries = []  # cada uno: {idx, qty, unit, name, macros|None, group|None}
    for idx, ing in enumerate(ingredients):
        qty, unit, name = _coerce_line(ing)
        macros = db.macros_for_line(qty, unit, name) if name else None
        group = None
        if macros:
            contrib = {m: macros[m] * _KCAL_PER_G[m] for m in _KCAL_PER_G}
            if any(contrib.values()):
                group = max(contrib, key=contrib.get)
        entries.append({"idx": idx, "qty": _get({"q": qty}, "q") if qty is not None else qty,
                        "raw_qty": qty, "unit": unit, "name": name,
                        "macros": macros, "group": group})

    # 2) factor de escalado por grupo de macro.
    report = {}
    factors = {}
    for macro in _KCAL_PER_G:  # protein, carbs, fats
        current = sum(e["macros"][macro] for e in entries
                      if e["macros"] and e["group"] == macro)
        target_v = tgt[macro]
        factor = 1.0
        applied = False
        if current > 0 and target_v > 0:
            factor = max(min_scale, min(max_scale, target_v / current))
            applied = abs(factor - 1.0) > 1e-9
        factors[macro] = factor
        report[macro] = {"current": round(current, 2), "target": round(target_v, 2),
                         "factor": round(factor, 4), "applied": applied}

    # 3) aplicar el factor a la cantidad de cada ingrediente de su grupo.
    out_ingredients = []
    achieved = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
    resolved = 0
    for e, ing in zip(entries, ingredients):
        new_ing = dict(ing) if isinstance(ing, dict) else {"name": e["name"],
                                                            "quantity": e["raw_qty"], "unit": e["unit"]}
        if e["macros"] and e["group"]:
            f = factors[e["group"]]
            base_q = e["raw_qty"]
            try:
                scaled_q = float(base_q) * f
                new_ing["quantity"] = round(scaled_q, 2)
            except (TypeError, ValueError):
                scaled_q = base_q
            for m in achieved:
                achieved[m] += e["macros"][m] * f
            resolved += 1
        elif e["macros"]:  # resoluble pero sin grupo (macros todos 0, e.g. agua)
            for m in achieved:
                achieved[m] += e["macros"][m]
            resolved += 1
        out_ingredients.append(new_ing)

    achieved = {m: round(v, 1) for m, v in achieved.items()}

    converged = True
    for macro in ("protein", "carbs", "fats"):
        t = tgt[macro]
        if t > 0:
            if abs(achieved[macro] - t) / t > tolerance_pct:
                converged = False
                break

    return {
        "ingredients": out_ingredients,
        "achieved": achieved,
        "target": tgt,
        "report": report,
        "resolved_count": resolved,
        "unresolved": len(ingredients) - resolved,
        "converged": converged,
    }


def solve_meal_macros(
    ingredient_strings: list,
    target: dict,
    db=None,
    *,
    min_scale: float = 0.3,
    max_scale: float = 3.5,
    tolerance_pct: float = 0.10,
) -> dict:
    """Variante para los ingredientes-STRING de un meal del plan ("0.5 taza de avena
    (50g)"). Mismo algoritmo que `solve_portion_macros` pero re-escribe los strings
    (cantidad líder + hint de gramos) en vez de un campo `quantity`, preservando el
    formato que consumen el coherence guard + shopping aggregator + frontend.

    Returns dict con `ingredients` (lista de strings re-escalados), `achieved`,
    `target`, `report`, `resolved_count`, `unresolved`, `converged`.
    """
    if db is None:
        from nutrition_db import IngredientNutritionDB
        db = IngredientNutritionDB()
    from nutrition_db import rescale_ingredient_string
    tgt = _target_macros(target)

    entries = []
    for s in ingredient_strings:
        macros = db.macros_from_ingredient_string(s)
        group = None
        if macros:
            contrib = {m: macros[m] * _KCAL_PER_G[m] for m in _KCAL_PER_G}
            if any(contrib.values()):
                group = max(contrib, key=contrib.get)
        entries.append({"s": s, "macros": macros, "group": group})

    report, factors = {}, {}
    for macro in _KCAL_PER_G:
        current = sum(e["macros"][macro] for e in entries
                      if e["macros"] and e["group"] == macro)
        target_v = tgt[macro]
        factor, applied = 1.0, False
        if current > 0 and target_v > 0:
            factor = max(min_scale, min(max_scale, target_v / current))
            applied = abs(factor - 1.0) > 1e-9
        factors[macro] = factor
        report[macro] = {"current": round(current, 2), "target": round(target_v, 2),
                         "factor": round(factor, 4), "applied": applied}

    out_strings = []
    factors_applied = []  # factor por-ingrediente (1.0 = intacto), alineado con input
    achieved = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
    resolved = 0
    for e in entries:
        if e["macros"] and e["group"]:
            f = factors[e["group"]]
            out_strings.append(rescale_ingredient_string(e["s"], f))
            factors_applied.append(f)
            for m in achieved:
                achieved[m] += e["macros"][m] * f
            resolved += 1
        elif e["macros"]:
            out_strings.append(e["s"])
            factors_applied.append(1.0)
            for m in achieved:
                achieved[m] += e["macros"][m]
            resolved += 1
        else:
            out_strings.append(e["s"])
            factors_applied.append(1.0)

    achieved = {m: round(v, 1) for m, v in achieved.items()}
    converged = True
    for macro in ("protein", "carbs", "fats"):
        t = tgt[macro]
        if t > 0 and abs(achieved[macro] - t) / t > tolerance_pct:
            converged = False
            break

    return {
        "ingredients": out_strings,
        "factors_applied": factors_applied,
        "achieved": achieved,
        "target": tgt,
        "report": report,
        "resolved_count": resolved,
        "unresolved": len(ingredient_strings) - resolved,
        "converged": converged,
    }
