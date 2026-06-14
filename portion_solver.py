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

import os
from typing import Optional


# Aportes calóricos Atwater (kcal/g) por macro — para decidir el macro dominante.
_KCAL_PER_G = {"protein": 4.0, "carbs": 4.0, "fats": 9.0}


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _envb(name: str, default: bool) -> bool:
    return str(os.environ.get(name, str(default))).strip().lower() in ("1", "true", "yes", "on")


# [M2-SOLVER-NNLS · 2026-06-14] Solver multi-restricción: reemplaza el escalado GREEDY por-grupo
# (que con ingredientes acoplados —pollo=P+grasa, arroz=C+P— no clava los 4 macros a la vez) por
# mínimos cuadrados ACOTADOS con regularización hacia x≈1, resuelto por descenso por coordenadas
# (box-QP convexo, exacto por coordenada, determinista, SIN dependencias — scipy no está instalado).
# El benchmark M2 midió la fuga: proteína 16% MAPE / solo 48% en ±10%. Este es el fix. Fallback al
# greedy si falla o se desactiva. Pesos: kcal+proteína priorizados (lo clínicamente crítico).
SOLVER_LSQ = _envb("MEALFIT_SOLVER_LSQ", True)
# Pesos TUNED por el benchmark M2 (A/B 2026-06-14): proteína 2.0 sobre-priorizaba y regresaba la
# grasa (12.2→15.8% MAPE). Rebalanceados → all-4-en-±10% subió de 24%→50% y la grasa volvió a 12.1%.
SOLVER_W_KCAL = _envf("MEALFIT_SOLVER_W_KCAL", 1.2)
SOLVER_W_PROTEIN = _envf("MEALFIT_SOLVER_W_PROTEIN", 1.5)
SOLVER_W_CARBS = _envf("MEALFIT_SOLVER_W_CARBS", 1.1)
SOLVER_W_FATS = _envf("MEALFIT_SOLVER_W_FATS", 1.4)
# Regularización hacia el porcionado original del LLM (x=1): evita porciones absurdas (un
# ingrediente a min_scale y otro a max_scale solo para clavar macros). Más alto = más fiel al LLM.
SOLVER_LSQ_REG = _envf("MEALFIT_SOLVER_LSQ_REG", 0.10)

# [P3-PROTEIN-FLOOR · 2026-06-14] El LSQ es SIMÉTRICO + regulariza hacia x=1 (el porcionado bajo del
# LLM) → con ingredientes acoplados RETIENE la fuente de proteína para no pasarse de kcal/grasa →
# sub-entrega de proteína sistemática (medido: ~50% en ±10%, déficit 30-70% en planes reales). La
# proteína es el macro clínicamente crítico (muscle gain + saciedad). Este piso, POST-solve, escala
# SOLO los ingredientes proteína-dominantes para cerrar el gap si la proteína quedó bajo target
# (clamp max_scale), aceptando cierto overshoot de kcal como trade-off clínico. Solo actúa en
# UNDERSHOOT; jamás recorta (overshoot intacto) ni toca meals sin fuente de proteína escalable
# (proteína vegetal difusa → el fix correcto es upstream, no inflar carbos). Knob para A/B + rollback.
SOLVER_PROTEIN_FLOOR = _envb("MEALFIT_SOLVER_PROTEIN_FLOOR", True)
SOLVER_PROTEIN_FLOOR_TOL = _envf("MEALFIT_SOLVER_PROTEIN_FLOOR_TOL", 0.05)
# Fracción calórica MÍNIMA de proteína para considerar un ingrediente "fuente de proteína" escalable.
# 0.25 capta huevo (36%), queso, carne, pescado, leguminosas (31%); excluye arroz (~8%), aceite,
# fruta. MÁS robusto que el macro DOMINANTE: huevo/queso son grasa-dominantes por kcal pero SON
# fuentes de proteína — escalarlos cierra el déficit (el dominante los excluía → el piso no disparaba).
SOLVER_PROTEIN_SOURCE_FRAC = _envf("MEALFIT_SOLVER_PROTEIN_SOURCE_FRAC", 0.25)


def _is_protein_source(macros: dict) -> bool:
    """True si el ingrediente es una FUENTE de proteína ESCALABLE: la proteína aporta
    >= SOLVER_PROTEIN_SOURCE_FRAC de sus kcal Y NO es carbo-dominante (la proteína aporta al menos
    tanto como los carbos). La 2ª condición excluye leguminosas/granos (lentejas/habichuelas/arroz):
    escalarlos para clavar la proteína dispararía los carbos a porciones absurdas (3 tazas de
    lentejas) → ese déficit de proteína vegetal difusa se corrige UPSTREAM, no inflando un carbo.
    Capta huevo/queso (grasa-dominantes por kcal pero proteína-ricos y bajos en carbo), carne, pescado."""
    p = macros.get("protein") or 0.0
    if p <= 0:
        return False
    c = macros.get("carbs") or 0.0
    kcal = macros.get("kcal") or (4.0 * p + 4.0 * c + 9.0 * (macros.get("fats") or 0))
    if kcal <= 0:
        return False
    return (4.0 * p) / kcal >= SOLVER_PROTEIN_SOURCE_FRAC and (4.0 * p) >= (4.0 * c)


def _apply_protein_floor(entries: list, sc: list, factors: list, tgt: dict, max_scale: float) -> bool:
    """[P3-PROTEIN-FLOOR] Si la proteína achieved quedó bajo target, sube SOLO los factores de las
    FUENTES de proteína (proteína >= SOLVER_PROTEIN_SOURCE_FRAC de sus kcal) para cerrar el gap, clamp
    a max_scale. Muta `factors`. Retorna True si aplicó. Determinista; no-op si no hay fuente de
    proteína escalable (proteína vegetal difusa → fix upstream) o si la proteína ya está en/sobre el
    piso. Trade-off: cierta sobre-entrega de kcal/grasa (proteína es el macro clínicamente crítico)."""
    tp = tgt.get("protein", 0) or 0
    if tp <= 0:
        return False
    pg = [i for i in sc if _is_protein_source(entries[i]["macros"])]
    if not pg:
        return False  # sin fuente de proteína escalable → no inflar carbos/grasa para fingir proteína
    ach_p = sum(entries[i]["macros"]["protein"] * factors[i] for i in sc)
    if ach_p >= tp * (1.0 - SOLVER_PROTEIN_FLOOR_TOL):
        return False  # ya en banda o por encima → no tocar (jamás recorta)
    pg_p = sum(entries[i]["macros"]["protein"] * factors[i] for i in pg)
    if pg_p <= 0:
        return False
    non_pg_p = ach_p - pg_p
    extra = (tp - non_pg_p) / pg_p           # factor extra sobre las fuentes para clavar el target
    if extra <= 1.0:
        return False                          # solo subir
    applied = False
    for i in pg:
        new_f = min(max_scale, factors[i] * extra)
        if new_f > factors[i] + 1e-9:
            factors[i] = new_f
            applied = True
    return applied


def _box_lsq(A_rows: list, b: list, weights: list, lo: float, hi: float,
             reg: float, iters: int = 150) -> list:
    """Mínimos cuadrados ACOTADOS con regularización hacia x=1, por descenso por coordenadas.
    Minimiza  Σ_r w_r (Σ_j A[r][j]·x_j − b[r])²  +  reg·Σ_j (x_j − 1)²  s.a. x_j ∈ [lo, hi].
    Problema convexo pequeño (≤~15 vars, ≤4 filas) → CD con minimización 1D exacta por coordenada
    converge al óptimo global. Determinista, pure-python (sin numpy/scipy). Retorna x (factores)."""
    nrows = len(A_rows)
    n = len(A_rows[0]) if nrows else 0
    x = [1.0] * n
    if n == 0:
        return x
    denom = [reg + sum(weights[r] * A_rows[r][i] ** 2 for r in range(nrows)) for i in range(n)]
    res = [sum(A_rows[r][i] * x[i] for i in range(n)) - b[r] for r in range(nrows)]  # Σ A·x − b
    for _ in range(iters):
        max_delta = 0.0
        for i in range(n):
            if denom[i] <= 0:
                continue
            num = reg  # = reg·1 (target del prior)
            for r in range(nrows):
                a = A_rows[r][i]
                if a != 0.0:
                    num -= weights[r] * a * (res[r] - a * x[i])  # c_r = res_r − a·x_i
            xi = num / denom[i]
            xi = lo if xi < lo else (hi if xi > hi else xi)
            d = xi - x[i]
            if d != 0.0:
                for r in range(nrows):
                    res[r] += A_rows[r][i] * d
                x[i] = xi
                if abs(d) > max_delta:
                    max_delta = abs(d)
        if max_delta < 1e-7:
            break
    return x


def _compute_scale_factors(entries: list, tgt: dict, min_scale: float, max_scale: float) -> tuple:
    """Factor de escalado POR-INGREDIENTE (alineado con `entries`). Usa el solver LSQ multi-macro
    si está habilitado; si no (o si falla), cae al greedy por-grupo. `entries[i]` debe tener
    `macros` ({kcal,protein,carbs,fats}|None) y `group` (macro dominante|None). Retorna (factors, method)."""
    factors = [1.0] * len(entries)
    sc = [i for i, e in enumerate(entries) if e.get("macros") and e.get("group")]
    if not sc:
        return factors, "none"
    method = None
    if SOLVER_LSQ:
        try:
            _w = {"kcal": SOLVER_W_KCAL, "protein": SOLVER_W_PROTEIN,
                  "carbs": SOLVER_W_CARBS, "fats": SOLVER_W_FATS}
            A_rows, brow, wrow = [], [], []
            for m in ("kcal", "protein", "carbs", "fats"):
                if tgt.get(m, 0) > 0:  # solo ecuaciones con target real (evita forzar macro→0)
                    A_rows.append([entries[i]["macros"][m] for i in sc])
                    brow.append(float(tgt[m]))
                    wrow.append(_w[m])
            if A_rows:
                xs = _box_lsq(A_rows, brow, wrow, min_scale, max_scale, SOLVER_LSQ_REG)
                for j, i in enumerate(sc):
                    factors[i] = xs[j]
                method = "lsq"
        except Exception:
            pass
    if method is None:
        # Fallback GREEDY por grupo de macro dominante (algoritmo v1).
        gf = {}
        for macro in _KCAL_PER_G:
            current = sum(entries[i]["macros"][macro] for i in sc if entries[i]["group"] == macro)
            tv = tgt.get(macro, 0)
            gf[macro] = max(min_scale, min(max_scale, tv / current)) if (current > 0 and tv > 0) else 1.0
        for i in sc:
            factors[i] = gf[entries[i]["group"]]
        method = "greedy"
    # [P3-PROTEIN-FLOOR · 2026-06-14] Piso de proteína post-solve (cierra el déficit sistémico de
    # sub-entrega que el LSQ simétrico introduce). Solo undershoot; clamp max_scale. No-op si no hay
    # fuente de proteína escalable o si ya está en banda. Aplica a ambos métodos.
    if SOLVER_PROTEIN_FLOOR and _apply_protein_floor(entries, sc, factors, tgt, max_scale):
        method += "+pfloor"
    return factors, method


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

    # 2) factor de escalado POR-INGREDIENTE (LSQ multi-macro; greedy fallback). El `report`
    #    greedy por-macro se conserva como telemetría.
    ing_factors, method = _compute_scale_factors(entries, tgt, min_scale, max_scale)
    report = {}
    for macro in _KCAL_PER_G:  # protein, carbs, fats
        current = sum(e["macros"][macro] for e in entries
                      if e["macros"] and e["group"] == macro)
        target_v = tgt[macro]
        gfactor = max(min_scale, min(max_scale, target_v / current)) if (current > 0 and target_v > 0) else 1.0
        report[macro] = {"current": round(current, 2), "target": round(target_v, 2),
                         "factor": round(gfactor, 4), "applied": abs(gfactor - 1.0) > 1e-9}

    # 3) aplicar el factor por-ingrediente a la cantidad.
    out_ingredients = []
    achieved = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
    resolved = 0
    for idx, (e, ing) in enumerate(zip(entries, ingredients)):
        new_ing = dict(ing) if isinstance(ing, dict) else {"name": e["name"],
                                                            "quantity": e["raw_qty"], "unit": e["unit"]}
        if e["macros"] and e["group"]:
            f = ing_factors[idx]
            base_q = e["raw_qty"]
            try:
                new_ing["quantity"] = round(float(base_q) * f, 2)
            except (TypeError, ValueError):
                pass
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
        "method": method,
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

    # [M2-SOLVER-NNLS] Factor POR-INGREDIENTE (LSQ multi-macro; greedy fallback). Reemplaza el
    # factor único por-grupo. El `report` greedy se conserva como telemetría por-macro.
    ing_factors, method = _compute_scale_factors(entries, tgt, min_scale, max_scale)
    report = {}
    for macro in _KCAL_PER_G:
        current = sum(e["macros"][macro] for e in entries
                      if e["macros"] and e["group"] == macro)
        target_v = tgt[macro]
        gfactor = max(min_scale, min(max_scale, target_v / current)) if (current > 0 and target_v > 0) else 1.0
        report[macro] = {"current": round(current, 2), "target": round(target_v, 2),
                         "factor": round(gfactor, 4), "applied": abs(gfactor - 1.0) > 1e-9}

    out_strings = []
    factors_applied = []  # factor por-ingrediente (1.0 = intacto), alineado con input
    achieved = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
    resolved = 0
    for idx, e in enumerate(entries):
        if e["macros"] and e["group"]:
            f = ing_factors[idx]
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
        "method": method,
        "resolved_count": resolved,
        "unresolved": len(ingredient_strings) - resolved,
        "converged": converged,
    }
