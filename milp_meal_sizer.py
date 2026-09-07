"""[BIOBOROS-3.0-MILP-MEAL-SIZER] Optimizador de porciones y macros con MILP (HiGHS).

Reemplaza el descenso por coordenadas heurístico de `portion_solver.py` por
un Programa Lineal Entero Mixto (MILP) resuelto con el motor HiGHS (vía SciPy).

Formulación matemática:
  Minimizar la desviación absoluta ponderada respecto a los objetivos nutricionales
  (L1-norm) + regularización suave hacia las porciones base culinarias:
    min  Σ w_m · (slack_pos_m + slack_neg_m) + λ · Σ (scale_pos_i + scale_neg_i)
  Sujeto a:
    Σ A_mi · x_i + slack_pos_m - slack_neg_m = target_m   ∀ macro m ∈ {kcal, P, C, F}
    x_i - base_x_i - scale_pos_i + scale_neg_i = 0         ∀ ingrediente i
    lower_bound_i <= x_i <= upper_bound_i                  ∀ ingrediente i
    x_j ∈ Z (enteros)                                      ∀ ingrediente discreto j (ej. huevos)
    slack_pos, slack_neg, scale_pos, scale_neg >= 0
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

_log = logging.getLogger(__name__)

# Intento de importar scipy.optimize.milp / linprog
_HAS_SCIPY = False
try:
    import numpy as np
    from scipy.optimize import LinearConstraint, milp
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass
class SizerIngredient:
    name: str
    food_id: str
    base_qty: float  # Gramos o unidades base propuestas
    unit: str = "g"
    # Nutrientes por gramo (o por unidad si is_discrete=True)
    kcal_per_unit: float = 0.0
    protein_per_unit: float = 0.0
    carbs_per_unit: float = 0.0
    fats_per_unit: float = 0.0
    # Cotas culinarias de seguridad
    min_qty: float = 0.0
    max_qty: float = 1000.0
    is_discrete: bool = False  # True para huevos u otras piezas indivisibles
    is_frozen: bool = False    # True si el usuario o la regla fijaron la cantidad


@dataclass
class MacroTarget:
    kcal: float
    protein: float
    carbs: float
    fats: float
    tolerance_pct: float = 0.05  # 5% tolerancia estándar


@dataclass
class SizerResult:
    status: Literal["optimal", "feasible_relaxed", "infeasible", "no_solver"]
    ingredients: List[Dict[str, Any]]
    solved_macros: Dict[str, float]
    target_macros: Dict[str, float]
    macro_errors: Dict[str, float]
    converged: bool
    execution_time_ms: float = 0.0
    details: str = ""


class MilpMealSizer:
    """Optimizador MILP para el dimensionamiento exacto de una comida."""

    def __init__(
        self,
        weight_kcal: float = 0.05,
        weight_protein: float = 4.0,
        weight_carbs: float = 1.0,
        weight_fats: float = 3.0,
        regularization_lambda: float = 0.01,
    ):
        self.w_kcal = weight_kcal
        self.w_protein = weight_protein
        self.w_carbs = weight_carbs
        self.w_fats = weight_fats
        self.reg_lambda = regularization_lambda

    def solve(
        self,
        ingredients: List[SizerIngredient],
        target: MacroTarget,
    ) -> SizerResult:
        import time
        t0 = time.perf_counter()

        if not ingredients:
            return SizerResult(
                status="infeasible",
                ingredients=[],
                solved_macros={"kcal": 0, "protein": 0, "carbs": 0, "fats": 0},
                target_macros={"kcal": target.kcal, "protein": target.protein, "carbs": target.carbs, "fats": target.fats},
                macro_errors={"kcal": 1.0, "protein": 1.0, "carbs": 1.0, "fats": 1.0},
                converged=False,
                details="No ingredients provided",
            )

        if not _HAS_SCIPY:
            return self._fallback_solve(ingredients, target, t0)

        n = len(ingredients)
        # Variables de decisión:
        # x_0..x_{n-1} : cantidades finales de cada ingrediente
        # s_pos, s_neg para {kcal, P, C, F} (4 pares = 8 variables)
        # r_pos, r_neg para cada ingrediente (2*n variables para desvío de base_qty)
        # Total variables: n + 8 + 2*n = 3*n + 8
        n_vars = 3 * n + 8
        c = np.zeros(n_vars)

        # Pesos en la función objetivo
        idx_s_pos = n
        idx_s_neg = n + 4
        idx_r_pos = n + 8
        idx_r_neg = 2 * n + 8

        weights = [self.w_kcal, self.w_protein, self.w_carbs, self.w_fats]
        for m_idx, w in enumerate(weights):
            c[idx_s_pos + m_idx] = w
            c[idx_s_neg + m_idx] = w

        for i in range(n):
            c[idx_r_pos + i] = self.reg_lambda
            c[idx_r_neg + i] = self.reg_lambda

        # Cotas de las variables
        lb = np.zeros(n_vars)
        ub = np.full(n_vars, np.inf)

        integrality = np.zeros(n_vars)

        for i, ing in enumerate(ingredients):
            if ing.is_frozen:
                lb[i] = ing.base_qty
                ub[i] = ing.base_qty
            else:
                lb[i] = max(0.0, ing.min_qty)
                ub[i] = max(lb[i], ing.max_qty)

            if ing.is_discrete:
                integrality[i] = 1  # Variable entera

        # Restricciones de igualdad:
        # 1. Cuatro ecuaciones para los macros:
        #    Σ A_mi · x_i + s_pos_m - s_neg_m = target_m
        # 2. n ecuaciones para la regularización de la cantidad base:
        #    x_i - r_pos_i + r_neg_i = base_qty_i
        num_constraints = 4 + n
        A_eq = np.zeros((num_constraints, n_vars))
        b_l = np.zeros(num_constraints)
        b_u = np.zeros(num_constraints)

        # Macro equations
        macros_target = [target.kcal, target.protein, target.carbs, target.fats]
        for m_idx in range(4):
            for i, ing in enumerate(ingredients):
                if m_idx == 0:
                    val = ing.kcal_per_unit
                elif m_idx == 1:
                    val = ing.protein_per_unit
                elif m_idx == 2:
                    val = ing.carbs_per_unit
                else:
                    val = ing.fats_per_unit
                A_eq[m_idx, i] = val

            A_eq[m_idx, idx_s_pos + m_idx] = 1.0
            A_eq[m_idx, idx_s_neg + m_idx] = -1.0
            b_l[m_idx] = macros_target[m_idx]
            b_u[m_idx] = macros_target[m_idx]

        # Base qty regularization equations
        for i, ing in enumerate(ingredients):
            row_idx = 4 + i
            A_eq[row_idx, i] = 1.0
            A_eq[row_idx, idx_r_pos + i] = -1.0
            A_eq[row_idx, idx_r_neg + i] = 1.0
            b_l[row_idx] = ing.base_qty
            b_u[row_idx] = ing.base_qty

        constraints = LinearConstraint(A_eq, b_l, b_u)

        from scipy.optimize import Bounds
        bounds = Bounds(lb, ub)

        res = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0

        if not res.success:
            return SizerResult(
                status="infeasible",
                ingredients=[{"name": ing.name, "qty": ing.base_qty, "unit": ing.unit} for ing in ingredients],
                solved_macros={"kcal": 0, "protein": 0, "carbs": 0, "fats": 0},
                target_macros={"kcal": target.kcal, "protein": target.protein, "carbs": target.carbs, "fats": target.fats},
                macro_errors={"kcal": 1.0, "protein": 1.0, "carbs": 1.0, "fats": 1.0},
                converged=False,
                execution_time_ms=dt_ms,
                details=f"HiGHS failed: {res.status} - {res.message}",
            )

        sol_x = res.x[:n]
        solved_kcal = sum(x * ing.kcal_per_unit for x, ing in zip(sol_x, ingredients))
        solved_prot = sum(x * ing.protein_per_unit for x, ing in zip(sol_x, ingredients))
        solved_carbs = sum(x * ing.carbs_per_unit for x, ing in zip(sol_x, ingredients))
        solved_fats = sum(x * ing.fats_per_unit for x, ing in zip(sol_x, ingredients))

        solved_macros = {
            "kcal": round(float(solved_kcal), 1),
            "protein": round(float(solved_prot), 1),
            "carbs": round(float(solved_carbs), 1),
            "fats": round(float(solved_fats), 1),
        }

        # Calcular errores relativos
        errs = {}
        for m_key, s_val, t_val in [
            ("kcal", solved_kcal, target.kcal),
            ("protein", solved_prot, target.protein),
            ("carbs", solved_carbs, target.carbs),
            ("fats", solved_fats, target.fats),
        ]:
            if t_val > 0:
                errs[m_key] = abs(s_val - t_val) / t_val
            else:
                errs[m_key] = 0.0

        # Criterio de convergencia estricto (dentro de tolerancia configurada)
        is_converged = all(err <= target.tolerance_pct for err in errs.values())

        out_ingredients = []
        for x, ing in zip(sol_x, ingredients):
            final_qty = round(float(x), 1 if not ing.is_discrete else 0)
            out_ingredients.append({
                "name": ing.name,
                "food_id": ing.food_id,
                "qty": final_qty,
                "unit": ing.unit,
                "is_discrete": ing.is_discrete,
                "kcal": round(final_qty * ing.kcal_per_unit, 1),
                "protein": round(final_qty * ing.protein_per_unit, 1),
                "carbs": round(final_qty * ing.carbs_per_unit, 1),
                "fats": round(final_qty * ing.fats_per_unit, 1),
            })

        return SizerResult(
            status="optimal" if is_converged else "feasible_relaxed",
            ingredients=out_ingredients,
            solved_macros=solved_macros,
            target_macros={"kcal": target.kcal, "protein": target.protein, "carbs": target.carbs, "fats": target.fats},
            macro_errors={k: round(v, 4) for k, v in errs.items()},
            converged=is_converged,
            execution_time_ms=dt_ms,
            details="Optimal solution found by HiGHS",
        )

    def _fallback_solve(
        self,
        ingredients: List[SizerIngredient],
        target: MacroTarget,
        t0: float,
    ) -> SizerResult:
        """Fallback determinista si scipy no estuviera disponible en el entorno."""
        import time
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return SizerResult(
            status="no_solver",
            ingredients=[{"name": ing.name, "qty": ing.base_qty, "unit": ing.unit} for ing in ingredients],
            solved_macros={"kcal": 0, "protein": 0, "carbs": 0, "fats": 0},
            target_macros={"kcal": target.kcal, "protein": target.protein, "carbs": target.carbs, "fats": target.fats},
            macro_errors={"kcal": 1.0, "protein": 1.0, "carbs": 1.0, "fats": 1.0},
            converged=False,
            execution_time_ms=dt_ms,
            details="SciPy not installed in runtime",
        )
