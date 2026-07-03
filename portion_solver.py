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


# [P2-SOLVER-KNOBS-REGISTRY · 2026-06-18] (audit fresco P2) Delegamos a los helpers de knobs.py para que
# los 6 knobs MEALFIT_SOLVER_* se auto-registren en _KNOBS_REGISTRY → visibles en /health/version. Antes
# leían os.environ crudo y eludían el registry: un override de los pesos del solver (núcleo de precisión)
# era invisible al operador. Fail-safe: si knobs no importa, helpers locales equivalentes (raw os.environ).
try:
    from knobs import _env_float as _envf, _env_bool as _envb  # auto-registran en _KNOBS_REGISTRY
except Exception:  # pragma: no cover - knobs siempre disponible en prod
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
                return factors, "lsq"
        except Exception:
            pass
    # Fallback GREEDY por grupo de macro dominante (algoritmo v1).
    gf = {}
    for macro in _KCAL_PER_G:
        current = sum(entries[i]["macros"][macro] for i in sc if entries[i]["group"] == macro)
        tv = tgt.get(macro, 0)
        gf[macro] = max(min_scale, min(max_scale, tv / current)) if (current > 0 and tv > 0) else 1.0
    for i in sc:
        factors[i] = gf[entries[i]["group"]]
    return factors, "greedy"


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


# ============================================================
# [P1-NEXT-LEVEL-BATCH · 2026-07-02] Refinador GLOBAL entero del día.
# ------------------------------------------------------------
# La precisión era una cadena SECUENCIAL (solver per-meal → closers → caps →
# quantize → recheck) donde cada pasada des-hace un poco a la anterior, y el
# rebalance del recheck es CONTINUO + re-quantize (el redondeo re-abre drift).
# Este refinador opera DIRECTO sobre el estado post-quantize en PASOS ENTEROS
# de 5g (las porciones siguen humanas — cero re-quantize) optimizando el DÍA
# COMPLETO de forma conjunta: local search greedy que en cada iteración aplica
# el movimiento ±step de UNA línea que más reduce el error ponderado de banda
# (kcal+P+C+F simultáneos). Determinista, sin dependencias, ~O(iters × líneas).
#
# Respeta el plato: bounds por línea [max(floor_g, 0.5×orig), min(cap_g, 2×orig)]
# — jamás convierte una guarnición en plato ni un plato en migaja. Las líneas
# exentas (condimentos/aceites vía exempt_tokens del caller) no se tocan.
# tooltip-anchor: P1-NEXT-LEVEL-SOLVER. Test: test_p1_next_level_batch.py.
# ============================================================

_REFINE_WEIGHTS = {"kcal": 1.0, "protein": 1.5, "carbs": 1.0, "fats": 1.2}


def _refine_error(delivered: dict, targets: dict) -> float:
    err = 0.0
    for k, w in _REFINE_WEIGHTS.items():
        t = float(targets.get(k) or 0.0)
        if t <= 0:
            continue
        err += w * ((float(delivered.get(k) or 0.0) - t) / t) ** 2
    return err


def refine_day_portions_integer(
    meals: list,
    targets: dict,
    db,
    step_g: float = 5.0,
    floor_g: float = 15.0,
    cap_g: float = 300.0,
    exempt_tokens: tuple = (),
    max_iters: int = 250,
) -> int:
    """Refina las porciones del DÍA en pasos enteros de `step_g` para clavar la banda
    all-4 (kcal/P/C/F conjuntos). Muta `ingredients` (+`ingredients_raw` lockstep) y
    NO toca los macros del meal (el caller hace truth-up por meal tocado — mismo
    contrato que _cap_unrealistic_portions). Devuelve nº de movimientos aplicados.

    `targets`: {"kcal","protein","carbs","fats"} en unidades absolutas del día.
    Fail-safe: cualquier error → 0 movimientos (día intacto)."""
    import re as _re
    try:
        from nutrition_db import rescale_ingredient_string as _resc
        try:
            from constants import strip_accents as _sa
        except Exception:
            def _sa(s):
                return s

        # 1) Censo de líneas móviles: gram-based, resolubles, no exentas.
        lines = []  # dicts: meal, idx, grams, per_g {kcal,p,c,f}, orig_grams
        delivered = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
        for meal in meals or []:
            if not isinstance(meal, dict):
                continue
            ings = meal.get("ingredients")
            if not isinstance(ings, list):
                continue
            for idx, ing in enumerate(ings):
                s = str(ing)
                mc = None
                try:
                    mc = db.macros_from_ingredient_string(s)
                except Exception:
                    mc = None
                if mc:
                    delivered["kcal"] += float(mc.get("kcal") or 0.0)
                    delivered["protein"] += float(mc.get("protein") or 0.0)
                    delivered["carbs"] += float(mc.get("carbs") or 0.0)
                    delivered["fats"] += float(mc.get("fats") or 0.0)
                il = _sa(s.lower())
                if "al gusto" in il or "opcional" in il:
                    continue
                if any(tok and tok in il for tok in exempt_tokens):
                    continue
                m_g = _re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\b", il)
                if not m_g or not mc:
                    continue
                grams = float(m_g.group(1).replace(",", "."))
                if grams <= 0:
                    continue
                per_g = {k: float(mc.get(k2) or 0.0) / grams
                         for k, k2 in (("kcal", "kcal"), ("protein", "protein"),
                                       ("carbs", "carbs"), ("fats", "fats"))}
                if all(abs(v) < 1e-9 for v in per_g.values()):
                    continue
                lines.append({"meal": meal, "idx": idx, "grams": grams,
                              "orig": grams, "per_g": per_g})
        if not lines:
            return 0
        tg = {k: float(targets.get(k) or 0.0) for k in ("kcal", "protein", "carbs", "fats")}
        if not any(v > 0 for v in tg.values()):
            return 0

        # 2) Greedy: el mejor movimiento ±step por iteración hasta converger.
        moves = 0
        err = _refine_error(delivered, tg)
        for _ in range(int(max_iters)):
            best = None  # (new_err, line, direction)
            for ln in lines:
                lo = max(float(floor_g), 0.5 * ln["orig"])
                hi = min(float(cap_g), 2.0 * ln["orig"])
                for direction in (+1.0, -1.0):
                    ng = ln["grams"] + direction * float(step_g)
                    if ng < lo - 1e-9 or ng > hi + 1e-9:
                        continue
                    cand = {k: delivered[k] + direction * float(step_g) * ln["per_g"][k]
                            for k in delivered}
                    ne = _refine_error(cand, tg)
                    if ne < err - 1e-9 and (best is None or ne < best[0]):
                        best = (ne, ln, direction)
            if best is None:
                break
            ne, ln, direction = best
            ln["grams"] += direction * float(step_g)
            for k in delivered:
                delivered[k] += direction * float(step_g) * ln["per_g"][k]
            err = ne
            moves += 1

        if not moves:
            return 0

        # 3) Aplicar los cambios a los strings (lockstep raw) por línea tocada.
        touched_meals = set()
        for ln in lines:
            if abs(ln["grams"] - ln["orig"]) < 1e-9:
                continue
            meal = ln["meal"]
            idx = ln["idx"]
            factor = ln["grams"] / ln["orig"]
            try:
                s = str(meal["ingredients"][idx])
                new_s = _resc(s, factor)
                if new_s and new_s != s:
                    meal["ingredients"][idx] = new_s
                    raw = meal.get("ingredients_raw")
                    if isinstance(raw, list) and idx < len(raw):
                        try:
                            raw[idx] = _resc(str(raw[idx]), factor)
                        except Exception:
                            pass
                    touched_meals.add(id(meal))
                    meal["_global_refine_applied"] = True
            except Exception:
                continue
        return moves if touched_meals else 0
    except Exception:
        return 0
