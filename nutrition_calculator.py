# backend/nutrition_calculator.py
"""
Agente Calculador: Cálculos nutricionales exactos con la ecuación de Mifflin-St Jeor.
Elimina la carga matemática del LLM para evitar alucinaciones numéricas.
"""

import logging

logger = logging.getLogger(__name__)

def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str, body_fat_pct: float = None) -> int:
    """
    Calcula el BMR (Tasa Metabólica Basal).
    Si se proporciona el % de grasa corporal, utiliza la ecuación más precisa de Katch-McArdle.
    De lo contrario, utiliza Mifflin-St Jeor.
    
    Katch-McArdle: BMR = 370 + (21.6 * LBM)
    Mifflin-St Jeor:
      Hombres: BMR = 10×peso(kg) + 6.25×altura(cm) − 5×edad + 5
      Mujeres: BMR = 10×peso(kg) + 6.25×altura(cm) − 5×edad − 161
    """
    if body_fat_pct is not None and body_fat_pct > 0:
        # Katch-McArdle
        lean_body_mass = weight_kg * (1 - (body_fat_pct / 100.0))
        bmr = 370 + (21.6 * lean_body_mass)
    else:
        # Mifflin-St Jeor
        if gender.lower() in ["male", "masculino", "m", "hombre"]:
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
        else:
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    
    return int(round(bmr))


# Multiplicadores de actividad física estándar (Katch-McArdle)
ACTIVITY_MULTIPLIERS = {
    "sedentary":  1.2,    # Oficina, poco o ningún ejercicio
    "light":      1.375,  # Ejercicio ligero 1-3 días/semana
    "moderate":   1.55,   # Ejercicio moderado 3-5 días/semana
    "active":     1.725,  # Ejercicio intenso 6-7 días/semana
    "athlete":    1.9,    # Entrenamientos dobles / trabajo físico
}

# Ajustes calóricos según el objetivo del usuario
GOAL_ADJUSTMENTS = {
    "lose_fat":     -0.20,   # Déficit del 20% para pérdida de grasa
    "gain_muscle":  +0.15,   # Superávit del 15% para ganancia muscular
    "maintenance":   0.00,   # Sin ajuste (mantenimiento)
    "performance":  +0.10,   # Superávit ligero del 10% para rendimiento
}

# Distribución de macros según objetivo (porcentaje de calorías totales)
MACRO_SPLITS = {
    "lose_fat":     {"protein_pct": 0.35, "carbs_pct": 0.35, "fats_pct": 0.30},
    "gain_muscle":  {"protein_pct": 0.30, "carbs_pct": 0.45, "fats_pct": 0.25},
    "maintenance":  {"protein_pct": 0.25, "carbs_pct": 0.45, "fats_pct": 0.30},
    "performance":  {"protein_pct": 0.25, "carbs_pct": 0.50, "fats_pct": 0.25},
}


def calculate_tdee(bmr: float, activity_level: str) -> int:
    """Calcula el TDEE (Gasto Energético Total Diario) = BMR × multiplicador de actividad.

    [P0-FORM-5] Si `activity_level` no está en `ACTIVITY_MULTIPLIERS`, se aplica
    el default ×1.55 (moderate) PERO se loguea WARNING. El path normal pasa
    `_validate_form_data_ranges` en el router y rechaza con 422 antes de llegar
    aquí, así que cualquier warning aquí es señal de:
      - caller no-router (cron, proactive_agent, scripts) que saltó la validación
      - bug upstream que mutó el value entre router y calculator
      - test/fixture con valor desconocido
    Sin este warning el plan se generaba con TDEE incorrecto sin telemetría.
    """
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level)
    if multiplier is None:
        logger.warning(
            f"[P0-FORM-5] activity_level={activity_level!r} no está en "
            f"ACTIVITY_MULTIPLIERS={list(ACTIVITY_MULTIPLIERS.keys())}. "
            f"Aplicando default 1.55 (moderate). Caller no pasó por "
            f"`_validate_form_data_ranges` o hay drift de schema."
        )
        multiplier = 1.55
    return int(round(bmr * multiplier))


def apply_goal_adjustment(tdee: float, goal: str) -> int:
    """Aplica el ajuste calórico según el objetivo (déficit/superávit).

    [P0-FORM-5] Mismo patrón que `calculate_tdee`: warning si `goal` no está
    en `GOAL_ADJUSTMENTS`. Default `0.0` (maintenance) puede ser razonable
    para "no sé qué meta" pero debe alertarse — un usuario que pidió `lose_fat`
    pero envió un valor desconocido recibiría un plan de mantenimiento.
    """
    adjustment = GOAL_ADJUSTMENTS.get(goal)
    if adjustment is None:
        logger.warning(
            f"[P0-FORM-5] goal={goal!r} no está en "
            f"GOAL_ADJUSTMENTS={list(GOAL_ADJUSTMENTS.keys())}. "
            f"Aplicando default 0.0 (maintenance). Caller no pasó por "
            f"`_validate_form_data_ranges` o hay drift de schema."
        )
        adjustment = 0.0
    target_calories = tdee * (1 + adjustment)
    # Redondear a múltiplos de 50 para números más limpios
    return int(round(target_calories / 50) * 50)


def calculate_macros(target_calories: int, goal: str) -> dict:
    """
    Calcula los gramos exactos de cada macronutriente basándose en:
    - Proteína: 4 cal/g
    - Carbohidratos: 4 cal/g
    - Grasas: 9 cal/g
    """
    split = MACRO_SPLITS.get(goal, MACRO_SPLITS["maintenance"])

    protein_cals = target_calories * split["protein_pct"]
    carbs_cals = target_calories * split["carbs_pct"]
    fats_cals = target_calories * split["fats_pct"]

    return {
        "protein_g": round(protein_cals / 4),
        "carbs_g": round(carbs_cals / 4),
        "fats_g": round(fats_cals / 9),
        "protein_str": f"{round(protein_cals / 4)}g",
        "carbs_str": f"{round(carbs_cals / 4)}g",
        "fats_str": f"{round(fats_cals / 9)}g",
    }


# ============================================================
# [P2-MDDA-SLOT-ALLOCATION · 2026-06-13] Reparte los macros diarios objetivo
# por slot de comida. Es el TARGET que consume `portion_solver.solve_portion_macros`
# (lado determinista del cerebro dividido): cada comida recibe su cuota de
# kcal/proteína/carbos/grasas, y el solver re-escala porciones para clavarla.
# ============================================================

# Splits por # de comidas (fracción del total diario por slot). Suman 1.0.
# 4 comidas es el default del producto (desayuno/almuerzo/merienda/cena).
MEAL_SLOT_SPLITS: dict = {
    3: {"desayuno": 0.30, "almuerzo": 0.40, "cena": 0.30},
    4: {"desayuno": 0.20, "almuerzo": 0.35, "merienda": 0.15, "cena": 0.30},
    5: {"desayuno": 0.20, "merienda_am": 0.10, "almuerzo": 0.35,
        "merienda_pm": 0.10, "cena": 0.25},
}


def allocate_macros_per_slot(
    daily_targets: dict,
    num_meals: int = 4,
    splits: dict | None = None,
) -> dict:
    """[P2-MDDA-SLOT-ALLOCATION] Distribuye los macros diarios por slot de comida.

    Args:
        daily_targets: macros del día. Acepta aliases comunes
            (``calories``/``target_calories``/``kcal``; ``protein``/``protein_g``;
            ``carbs``/``carbs_g``; ``fats``/``fat``/``fats_g``).
        num_meals: 3/4/5 → usa el split correspondiente de ``MEAL_SLOT_SPLITS``
            (default 4). Valores fuera del set caen a 4.
        splits: override explícito ``{slot: fracción}`` (debe sumar ~1.0); si se
            pasa, ignora ``num_meals``. No se re-normaliza si no suma 1.0 — el
            caller es responsable (permite repartos parciales intencionales).

    Returns:
        ``{slot: {kcal, protein, carbs, fats}}`` con cada macro = fracción × diario,
        redondeado. Tooltip-anchor: P2-MDDA-SLOT-ALLOCATION.
    """
    def _g(*keys):
        for k in keys:
            v = daily_targets.get(k) if isinstance(daily_targets, dict) else None
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return 0.0

    day = {
        "kcal": _g("kcal", "calories", "target_calories", "cals"),
        "protein": _g("protein", "protein_g", "proteina"),
        "carbs": _g("carbs", "carbs_g", "carbohidratos"),
        "fats": _g("fats", "fat", "fats_g", "grasas"),
    }
    if splits is None:
        splits = MEAL_SLOT_SPLITS.get(num_meals, MEAL_SLOT_SPLITS[4])

    out = {}
    for slot, frac in splits.items():
        out[slot] = {
            "kcal": round(day["kcal"] * frac, 1),
            "protein": round(day["protein"] * frac, 1),
            "carbs": round(day["carbs"] * frac, 1),
            "fats": round(day["fats"] * frac, 1),
        }
    return out


# ============================================================
# [P1-SWAP-MACROS · 2026-05-22] Validador post-gen de macros del meal
# generado por LLM en swap/modify vs los targets del slot original.
# Cierra el gap del audit "Cambiar Plato": pre-fix el prompt solo inyectaba
# `target_calories` como hint soft, sin re-validación → drift arbitrario
# permitido (ej. target 350kcal/15g protein → LLM emite 450kcal/8g sin queja).
# ============================================================

def _meal_macros_tolerance_pct() -> float:
    """Knob `MEALFIT_SWAP_MACROS_TOLERANCE_PCT` (default 0.15, clamp [0.05, 0.50]).
    Tolerancia per-meal en swap/modify. 15% es sweet-spot empírico:
    - Más estricto que plan-completo (que tiene compensación entre meals).
    - Más laxo que zero-drift (LLMs varían naturalmente ±5-15% en estimaciones
      cuando porcionan a ojo sin pesos exactos).
    Tooltip-anchor: P1-SWAP-MACROS-TOLERANCE-KNOB.
    """
    import os
    try:
        v = float(os.environ.get("MEALFIT_SWAP_MACROS_TOLERANCE_PCT", "0.15"))
    except (TypeError, ValueError):
        return 0.15
    return max(0.05, min(0.50, v))


def _meal_macros_validate_enabled() -> bool:
    """Kill switch `MEALFIT_SWAP_MACROS_VALIDATE` (default True). Si False, el
    swap/modify retorna lo que el LLM emita sin validar macros (legacy pre-fix).
    Flip a False sin redeploy si la validación introduce demasiados retries.
    Tooltip-anchor: P1-SWAP-MACROS-KILL-SWITCH.
    """
    import os
    return os.environ.get("MEALFIT_SWAP_MACROS_VALIDATE", "true").lower() != "false"


def validate_meal_macros_against_targets(
    meal_macros: dict,
    target_macros: dict,
    tolerance_pct: float | None = None,
) -> tuple:
    """[P1-SWAP-MACROS · 2026-05-22] Compara los macros del meal generado por
    LLM contra los targets del slot original.

    Llamado en `agent.py::swap_meal` y `tools.py::execute_modify_single_meal`
    dentro del retry loop tenacity (idéntico patrón a
    `validate_ingredients_against_pantry`): si falla, inyectamos el `summary`
    al próximo prompt + raise ValueError para forzar tenacity retry.

    Args:
        meal_macros: salida del LLM (MealModel.model_dump()) con keys
            ``cals``/``protein``/``carbs``/``fats``. Acepta aliases comunes
            (``calories``/``protein_g``/``fat``/etc.).
        target_macros: shape similar. Si una key falta o es 0 → SKIP enforce
            sobre ese macro (no se puede validar drift sobre target indefinido).
        tolerance_pct: 0..1; ``None`` → knob (default 0.15).

    Returns:
        ``(passed, drifts, summary)``:
            * passed (bool): True si TODOS los macros con target válido están
              dentro de la tolerancia.
            * drifts (dict): ``{macro_key: {actual, target, delta_pct}}`` por
              cada macro evaluado (incluso los que pasaron — útil para
              telemetría).
            * summary (str): mensaje legible para inyectar al retry prompt
              del LLM (vacío string si passed=True).

    Reglas:
        - Target ``None``/``0`` → key se omite (no se puede validar drift sobre 0).
        - Actual ``None`` → se trata como 0 (drift 100% del target).
        - ``cals`` tiene tolerancia 1.5× la base (kcal varían más por side
          dishes y porciones; protein/carbs/fats responden a ingredientes).

    Tooltip-anchor: P1-SWAP-MACROS-VALIDATOR | test_p1_swap_macros_validate
    """
    if tolerance_pct is None:
        tolerance_pct = _meal_macros_tolerance_pct()
    try:
        tolerance_pct = max(0.0, min(1.0, float(tolerance_pct or 0.0)))
    except (TypeError, ValueError):
        tolerance_pct = 0.15

    def _get(d: dict, *keys, default=0):
        if not isinstance(d, dict):
            return default
        for k in keys:
            v = d.get(k)
            if v is not None:
                return v
        return default

    actual = {
        "cals":    _get(meal_macros, "cals", "calories"),
        "protein": _get(meal_macros, "protein", "protein_g"),
        "carbs":   _get(meal_macros, "carbs", "carbs_g"),
        "fats":    _get(meal_macros, "fats", "fat", "fats_g"),
    }
    target = {
        "cals":    _get(target_macros, "cals", "calories", "target_calories"),
        "protein": _get(target_macros, "protein", "protein_g", "target_protein"),
        "carbs":   _get(target_macros, "carbs", "carbs_g", "target_carbs"),
        "fats":    _get(target_macros, "fats", "fat", "fats_g", "target_fats"),
    }

    drifts: dict = {}
    failures: list = []
    cals_tolerance = min(1.0, tolerance_pct * 1.5)

    for key in ("cals", "protein", "carbs", "fats"):
        try:
            t = float(target.get(key) or 0)
            a = float(actual.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if t <= 0:
            continue  # target indefinido → no se enforce
        delta_pct = abs(a - t) / t
        drifts[key] = {
            "actual": round(a, 1),
            "target": round(t, 1),
            "delta_pct": round(delta_pct, 3),
        }
        tol_for_key = cals_tolerance if key == "cals" else tolerance_pct
        if delta_pct > tol_for_key:
            unit = "kcal" if key == "cals" else "g"
            failures.append(
                f"{key}={int(round(a))}{unit} (target ~{int(round(t))}{unit}, "
                f"drift {int(round(delta_pct * 100))}% > {int(round(tol_for_key * 100))}%)"
            )

    if not failures:
        return (True, drifts, "")
    summary = (
        "⚠️ MACROS FUERA DE OBJETIVO: " + " | ".join(failures) +
        ". Reformula el plato preservando ingredientes pero ajusta porciones "
        "para acercarte a los objetivos."
    )
    return (False, drifts, summary)


# ============================================================
# [P1-SWAP-RECIPE-COHERENCE · 2026-05-22] Mini-coherence check per-meal
# sobre la salida LLM del swap/modify. Cierra el gap user-facing dejado
# documentado en el bundle P1-SWAP-MACROS — la escalación coherence
# inline (P1-SWAP-COHERENCE-ESCALATE) cubre el surface #2 (plan-gen
# swap-to-best) PERO el "Cambiar Plato" del UI usa surface #4
# (/recalculate-shopping-list) que es síncrono y NO bloquea. Este check
# corre ANTES del retorno de swap_meal/execute_modify_single_meal, así
# que un cap_swallowed_modifier severo (receta dice "el pollo" pero
# ingredients=["pavo"]) gatea el retry tenacity sin que el plato roto
# salga jamás del backend.
# ============================================================

def _swap_recipe_coherence_enabled() -> bool:
    """Kill switch `MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE` (default True).
    Mirror de `_meal_macros_validate_enabled`. Flip a False si introduce
    demasiados retries en prod o falsos positivos por sinónimos no
    canonicalizados. Tooltip-anchor: P1-SWAP-RECIPE-COHERENCE-KILL-SWITCH.
    """
    import os
    return os.environ.get("MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE", "true").lower() != "false"


# [P3-SWAP-COHERENCE-MULTI-CAT · 2026-05-22] Pre-fix el validator solo
# cubría proteínas. Caso real verificado log 2026-05-22 23:04: el LLM
# fallaba mencionando un alias proteína no listado. Pero el mismo modo
# de fallo aplica a CARBS (recipe dice "papa" + ingredients=["arroz"])
# y VEGGIES (recipe dice "aguacate" + ingredients=["lechuga"]). Frutas
# se mantienen opt-in (default False) — su impacto user-facing es menor
# y tienen mayor potencial de FP en recetas que las mencionan como
# decoración o salsa sin querer convertirlas en compra obligatoria.
def _recipe_coherence_active_categories() -> list:
    """Devuelve lista de tuples ``(category_name_es, synonyms_dict)``
    para las categorías habilitadas vía knobs. Si ``constants`` no es
    importable, solo protein cae al fallback minimalista; las otras
    categorías se skipean con WARN (no romper hot path por dep faltante).

    Knobs (todos opt-out / opt-in independientes):
      - ``MEALFIT_SWAP_RECIPE_COHERENCE_PROTEIN`` default True
      - ``MEALFIT_SWAP_RECIPE_COHERENCE_CARB`` default True
      - ``MEALFIT_SWAP_RECIPE_COHERENCE_VEGGIE`` default True
      - ``MEALFIT_SWAP_RECIPE_COHERENCE_FRUIT`` default False (opt-in)

    El master kill switch ``MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE`` se
    sigue gateando arriba (en agent.py / tools.py callsites) — si OFF,
    este helper no se invoca.
    """
    import os

    def _enabled(env_key: str, default: str) -> bool:
        return os.environ.get(env_key, default).lower() != "false"

    categories: list = []

    # --- Proteínas (legacy, default True) ---
    if _enabled("MEALFIT_SWAP_RECIPE_COHERENCE_PROTEIN", "true"):
        try:
            from constants import PROTEIN_SYNONYMS
            categories.append(("proteína", PROTEIN_SYNONYMS))
        except Exception as _e:
            logger.warning(
                f"[P3-SWAP-COHERENCE-MULTI-CAT] constants.PROTEIN_SYNONYMS "
                f"no disponible ({type(_e).__name__}); usando fallback "
                f"minimalista. Si esto sucede en prod, revisar constants."
            )
            categories.append(("proteína", _FALLBACK_PROTEIN_SYNONYMS))

    # --- Carbohidratos (NUEVO, default True) ---
    if _enabled("MEALFIT_SWAP_RECIPE_COHERENCE_CARB", "true"):
        try:
            from constants import CARB_SYNONYMS
            categories.append(("carbohidrato", CARB_SYNONYMS))
        except Exception as _e:
            logger.warning(
                f"[P3-SWAP-COHERENCE-MULTI-CAT] constants.CARB_SYNONYMS no "
                f"disponible ({type(_e).__name__}); categoría carb skip en "
                f"este runtime. Set MEALFIT_SWAP_RECIPE_COHERENCE_CARB=false "
                f"para silenciar este warning."
            )

    # --- Vegetales/grasas (NUEVO, default True) ---
    if _enabled("MEALFIT_SWAP_RECIPE_COHERENCE_VEGGIE", "true"):
        try:
            from constants import VEGGIE_FAT_SYNONYMS
            categories.append(("vegetal", VEGGIE_FAT_SYNONYMS))
        except Exception as _e:
            logger.warning(
                f"[P3-SWAP-COHERENCE-MULTI-CAT] constants.VEGGIE_FAT_SYNONYMS "
                f"no disponible ({type(_e).__name__}); categoría veggie skip."
            )

    # --- Frutas (NUEVO, default False — opt-in) ---
    if _enabled("MEALFIT_SWAP_RECIPE_COHERENCE_FRUIT", "false"):
        try:
            from constants import FRUIT_SYNONYMS
            categories.append(("fruta", FRUIT_SYNONYMS))
        except Exception as _e:
            logger.warning(
                f"[P3-SWAP-COHERENCE-MULTI-CAT] constants.FRUIT_SYNONYMS "
                f"no disponible ({type(_e).__name__}); categoría fruit skip."
            )

    return categories


# [P1-SWAP-RECIPE-COHERENCE · 2026-05-22] Fallback minimalista para entornos
# donde `constants.PROTEIN_SYNONYMS` no es importable (dep langchain no
# instalada, tests aislados). Subset de las proteínas más comunes en planes
# RD para que el validator siga teniendo SEÑAL aunque sea reducida.
# Sincronizado a mano contra `constants.PROTEIN_SYNONYMS`; si los aliases
# del module canónico crecen, esta lista NO se actualiza automáticamente
# — el test `test_p1_swap_recipe_coherence_fallback_subset_of_canonical`
# detecta drift cuando ambos son importables.
_FALLBACK_PROTEIN_SYNONYMS = {
    "pollo": ["pollo", "pechuga", "muslo", "alitas"],
    "pavo": ["pavo", "pechuga de pavo"],
    "res": ["res", "carne molida", "bistec", "filete de res"],
    "cerdo": ["cerdo", "lomo", "pernil", "costilla", "chuleta"],
    "pescado": ["pescado", "salmon", "tilapia", "mero", "chillo", "atun"],
    "huevos": ["huevos", "huevo", "tortilla", "revoltillo"],
    "camarones": ["camarones", "camaron"],
    "queso": ["queso blanco", "queso de freir", "queso mozzarella", "ricotta", "queso fresco"],
    "yogurt": ["yogurt", "yogur"],
    "habichuelas": ["habichuelas", "habichuela", "frijoles"],
    "lentejas": ["lentejas", "lenteja"],
    "garbanzos": ["garbanzos", "garbanzo"],
    "tofu": ["tofu", "soya"],
}


def _strip_accents_lower(s: str) -> str:
    """Normaliza string para matching tolerante (acentos + case + espacios
    múltiples). Inline para no requerir import de unicodedata en hot path
    si el kill switch está OFF."""
    if not isinstance(s, str):
        return ""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", s)
    out = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return " ".join(out.split())


def validate_meal_recipe_ingredients_coherence(meal: dict) -> tuple:
    """[P1-SWAP-RECIPE-COHERENCE · 2026-05-22] Detecta el modo de fallo
    `cap_swallowed_modifier` a nivel meal-output: la receta menciona una
    proteína canónica (cualquier alias de `PROTEIN_SYNONYMS`) pero la lista
    de ingredientes estructurados NO contiene ninguna versión de esa
    proteína. Resultado pre-fix: el shopping aggregator construía lista de
    compras sin la proteína mencionada → user veía receta con "el pollo"
    pero su lista decía solo "200g pavo" → frustrado al cocinar.

    Args:
        meal: dict con keys `ingredients: List[str]` y `recipe: List[str]`.
            Acepta `MealModel.model_dump()` directo.

    Returns:
        ``(passed, divergences, summary)``:
            * passed (bool): True si NO hay divergencias críticas detectadas.
            * divergences (dict): ``{canonical_protein: {mentioned_alias,
              listed: False}}`` para cada proteína en receta sin alias en
              ingredients. Vacío si passed.
            * summary (str): mensaje para inyectar al retry prompt (vacío
              si passed).

    Scope V1:
        - Solo PROTEÍNAS canónicas (subset de mayor impacto nutricional).
          Carbs/veggies quedan para V2 si se observan recurrencias.
        - Detección unidireccional: receta menciona → ingredientes no la
          tienen. La dirección inversa (ingredients lista X pero recipe
          no la usa) NO se valida acá — es ineficiencia menor (ingrediente
          comprado sin usar) vs el cap-swallowed que es BUG funcional.
        - Match por substring tras `_strip_accents_lower`. No usa stemming
          ni lemmatización — suficiente para Spanish noun forms en recetas.

    Tooltip-anchor: P1-SWAP-RECIPE-COHERENCE-VALIDATOR | test_p1_swap_recipe_coherence
    """
    if not isinstance(meal, dict):
        return (True, {}, "")

    ingredients = meal.get("ingredients") or []
    recipe = meal.get("recipe") or []
    if not ingredients or not recipe:
        # Sin data suficiente para comparar — no falsos positivos.
        return (True, {}, "")

    # [P3-SWAP-COHERENCE-MULTI-CAT · 2026-05-22] Helper centralizado que
    # importa lazy los 4 catálogos (PROTEIN/CARB/VEGGIE/FRUIT_SYNONYMS)
    # gateados por sus respectivos knobs. Si no hay categorías activas
    # (knobs todos OFF o constants no importable), retornamos passed.
    categories = _recipe_coherence_active_categories()
    if not categories:
        return (True, {}, "")

    recipe_text = _strip_accents_lower(" ".join(str(r) for r in recipe if r))
    if not recipe_text:
        return (True, {}, "")

    ingredients_text = _strip_accents_lower(" | ".join(str(i) for i in ingredients if i))

    divergences: dict = {}

    for category_name, syn_dict in categories:
        for canonical, aliases in syn_dict.items():
            # Si otra categoría ya reportó este canónico (raro pero posible
            # con catálogos solapados en futuro), respetamos el primer hit.
            if canonical in divergences:
                continue

            # Normalizamos cada alias una sola vez; las claves del dict
            # también pueden ser multi-palabra (e.g. "queso de freír",
            # "plátano verde").
            norm_aliases = [_strip_accents_lower(a) for a in aliases if a]
            norm_aliases = [a for a in norm_aliases if len(a) >= 3]
            if not norm_aliases:
                continue

            # ¿Algún alias aparece en la receta?
            mentioned_alias = None
            for a in norm_aliases:
                # Boundary check ligero — evitamos match parcial como "res" en
                # "estresante". Usamos espacios/puntuación a los costados.
                if _alias_appears_as_word(a, recipe_text):
                    mentioned_alias = a
                    break
            if not mentioned_alias:
                continue

            # ¿Algún alias aparece en los ingredientes estructurados?
            listed = any(a in ingredients_text for a in norm_aliases)
            if listed:
                continue

            divergences[canonical] = {
                "mentioned_alias": mentioned_alias,
                "listed": False,
                # [P3-SWAP-COHERENCE-MULTI-CAT · 2026-05-22] Categoría
                # añadida al sub-dict para observabilidad (log + métricas
                # downstream). NO rompe back-compat: callers existentes
                # (`agent.py`, `tools.py`) solo leen `mentioned_alias` y
                # el summary. Test P3-SWAP-RETRY-COHERENCE-HINT verifica
                # que mentioned_alias/listed sigan presentes.
                "category": category_name,
            }

    if not divergences:
        return (True, {}, "")

    # [P3-SWAP-RETRY-COHERENCE-HINT · 2026-05-22] El summary debe incluir
    # el ALIAS específico que el LLM mencionó en la receta (no solo el
    # canónico). Pre-fix solo decía "la receta describe el uso de pescado"
    # — pero el LLM había escrito "dorado", no "pescado", así que al buscar
    # qué corregir, NO encontraba "pescado" en su receta y reintentaba con
    # el mismo alias. Incidente verificado log 2026-05-22 23:04-23:05:
    # 3 intentos consecutivos mencionando "dorado", todos fallidos.
    # Post-fix: cita el alias verbatim + estructura "elige UNA opción" con
    # rutas (a)/(b) explícitas para reducir ambigüedad.
    # [P3-SWAP-COHERENCE-MULTI-CAT · 2026-05-22] Solo añadir el qualifier
    # "(que cuenta como X)" cuando el alias ≠ canónico — evita ruido como
    # "`papa` (que cuenta como papas)" donde el lector entiende que son lo
    # mismo. El qualifier sigue load-bearing cuando alias es lejano
    # (ej: "dorado" → "pescado").
    def _mention_phrase(canonical: str, info: dict) -> str:
        alias = info.get("mentioned_alias") or canonical
        alias_norm = _strip_accents_lower(alias)
        canon_norm = _strip_accents_lower(canonical)
        if (
            alias_norm == canon_norm
            or alias_norm in canon_norm
            or canon_norm in alias_norm
        ):
            return f"`{alias}`"
        return f"`{alias}` (que cuenta como {canonical})"

    mention_phrases = [
        _mention_phrase(canonical, info)
        for canonical, info in sorted(divergences.items())
    ]
    mentions_str = ", ".join(mention_phrases)
    summary = (
        f"⚠️ RECETA MENCIONA INGREDIENTES NO LISTADOS: en los pasos de la receta "
        f"escribiste {mentions_str}, pero el array `ingredients` NO los contiene "
        f"(el usuario no podrá comprarlos).\n"
        f"CORRECCIÓN OBLIGATORIA — elige UNA opción:\n"
        f"  (a) Añade al array `ingredients` cada alimento mencionado con cantidad "
        f"medible (ej: '180g de pescado fresco').\n"
        f"  (b) Reescribe los pasos de la receta SIN mencionar esos alimentos — usa "
        f"ÚNICAMENTE los ingredientes que ya listaste."
    )
    return (False, divergences, summary)


# ============================================================
# [P2-SWAP-CONSISTENCY · 2026-05-22] Validador post-gen de prep_time del
# meal generado por LLM cuando swap_reason='time' ("No tengo tiempo hoy").
# Pre-fix: el prompt inyectaba el hint "< 20 min" pero NO había validador
# que rechazara una receta de 40 min — el LLM podía ignorar el hint sin
# consecuencias. Cierra el gap "soft-only" detectado en el audit del modal.
# Espejo del patrón macros validator: validate → si falla, inyecta summary
# al retry prompt + raise ValueError para que tenacity reintente.
# ============================================================

def _swap_prep_time_target_minutes() -> int:
    """Knob `MEALFIT_SWAP_PREP_TIME_TARGET_MIN` (default 20, clamp [5, 120]).
    Cuando swap_reason='time', el LLM debe emitir un meal con
    ``prep_time <= N`` minutos. Si excede → tenacity retry con feedback.
    Tooltip-anchor: P2-SWAP-PREP-TIME-TARGET-KNOB.
    """
    import os
    try:
        v = int(os.environ.get("MEALFIT_SWAP_PREP_TIME_TARGET_MIN", "20"))
    except (TypeError, ValueError):
        return 20
    return max(5, min(120, v))


def _swap_prep_time_validate_enabled() -> bool:
    """Kill switch `MEALFIT_SWAP_PREP_TIME_VALIDATE` (default True). Flip a
    False sin redeploy si introduce demasiados retries en prod o el parser
    de prep_time falla en formatos inesperados. Tooltip-anchor:
    P2-SWAP-PREP-TIME-KILL-SWITCH.
    """
    import os
    return os.environ.get("MEALFIT_SWAP_PREP_TIME_VALIDATE", "true").lower() != "false"


def _parse_prep_time_minutes(value) -> float | None:
    """Extrae minutos del field ``prep_time`` (MealModel: str ~"15 min").
    Acepta int/float directos (legacy). Retorna None si no se puede parsear
    — el caller debe tratarlo como passthrough (no fallar el guard si no
    podemos leer el valor, mejor permitir que abortar falso-positivo).
    Tooltip-anchor: P2-SWAP-PREP-TIME-PARSE.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool antes que int (issubclass bool de int)
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    try:
        import re as _re
        s = str(value).lower().strip()
        # Match primer número entero o decimal. Casos: "15 min", "15-20 min",
        # "aprox 30 minutos", "1 hora" (heurística: si dice "hora" multiplica).
        if "hora" in s or "hr" in s or "hour" in s:
            m = _re.search(r"(\d+(?:\.\d+)?)", s)
            if m:
                return float(m.group(1)) * 60.0
        m = _re.search(r"(\d+)", s)
        if m:
            return float(m.group(1))
    except Exception:
        return None
    return None


def validate_meal_prep_time_against_target(
    meal: dict,
    target_minutes: int | None = None,
) -> tuple:
    """[P2-SWAP-CONSISTENCY · 2026-05-22] Compara ``meal.prep_time`` contra
    el target (default knob 20 min). Solo se invoca cuando
    ``swap_reason='time'``; los demás reasons NO consultan este validator.

    Args:
        meal: ``MealModel.model_dump()`` del LLM con key ``prep_time``
            (str format "15 min" según schemas.py:14).
        target_minutes: tope. None → knob.

    Returns:
        ``(passed, actual_min, summary)``:
            * passed (bool): True si actual <= target, o si no se puede
              parsear el valor (passthrough — preferimos un meal posiblemente
              lento a un falso positivo que aborte el swap).
            * actual_min (float | None): minutos parseados, o None.
            * summary (str): mensaje para inyectar al retry prompt (vacío
              si passed).

    Tooltip-anchor: P2-SWAP-PREP-TIME-VALIDATOR | test_p2_swap_consistency_prep_time
    """
    if target_minutes is None:
        target_minutes = _swap_prep_time_target_minutes()
    raw = meal.get("prep_time") if isinstance(meal, dict) else None
    actual = _parse_prep_time_minutes(raw)
    if actual is None:
        # Passthrough: si el LLM no emitió prep_time o el formato es exótico,
        # NO abortamos el swap (legacy behavior). Solo enforce cuando podemos
        # leer un número concreto.
        return (True, None, "")
    if actual <= float(target_minutes):
        return (True, actual, "")
    summary = (
        f"⏱️ PREP_TIME FUERA DE OBJETIVO: la receta toma ~{int(round(actual))} min "
        f"pero el usuario pidió ≤{target_minutes} min ('No tengo tiempo hoy'). "
        f"Reformula con técnica más rápida (microondas, salteado directo, "
        f"corte fino, sin marinar largo, ingredientes pre-cocidos)."
    )
    return (False, actual, summary)


def _alias_appears_as_word(alias: str, text: str) -> bool:
    """Substring con boundary check ligero — alias rodeado por non-letter
    o inicio/fin de texto. Evita falsos positivos como "res" en
    "estresante" o "ave" en "lavar". No usa regex \\b porque acentos
    Unicode no cuentan como word-char en \\b clásico de Python."""
    if not alias or not text:
        return False
    idx = 0
    L = len(alias)
    while True:
        found = text.find(alias, idx)
        if found < 0:
            return False
        # Verificar boundary izquierda
        left_ok = found == 0 or not text[found - 1].isalpha()
        # Verificar boundary derecha
        right_idx = found + L
        right_ok = right_idx >= len(text) or not text[right_idx].isalpha()
        if left_ok and right_ok:
            return True
        idx = found + 1


def _get_smoothed_weight_history(weight_history):
    """
    Calcula una media móvil de 7 días para suavizar fluctuaciones diarias (retención de líquidos, etc.).
    """
    if not weight_history:
        return []
    
    from datetime import datetime, timedelta
    
    def get_w_kg(entry):
        w = float(entry['weight'])
        if entry.get('unit', 'lb') == 'lb':
            w /= 2.20462
        return w

    smoothed = []
    # weight_history assumes to be already sorted by date
    for i, entry in enumerate(weight_history):
        current_date = datetime.strptime(entry["date"], "%Y-%m-%d")
        window_start = current_date - timedelta(days=7)
        
        # Encontrar todas las entradas dentro de los últimos 7 días
        window_entries = [
            get_w_kg(e) for e in weight_history[:i+1] 
            if datetime.strptime(e["date"], "%Y-%m-%d") >= window_start
        ]
        
        avg_weight_kg = sum(window_entries) / len(window_entries)
        
        # Re-empaquetar con el peso suavizado (siempre en kg para el cálculo interno)
        smoothed_entry = dict(entry)
        smoothed_entry['weight'] = avg_weight_kg
        smoothed_entry['unit'] = 'kg' # Normalizado a kg
        smoothed.append(smoothed_entry)
        
    return smoothed


def _calculate_weight_velocity(smoothed_history):
    """Calcula velocidad y aceleración del cambio de peso (en kg por día) usando datos suavizados."""
    if len(smoothed_history) < 3:
        return None
    
    mid = len(smoothed_history) // 2
    first_half = smoothed_history[:mid]
    second_half = smoothed_history[mid:]
    
    from datetime import datetime
    
    def get_w_kg(entry):
        # Ya viene normalizado a kg desde el smoothing
        return float(entry['weight'])
        
    def get_days(entries):
        if len(entries) < 2: return 1
        d1 = datetime.strptime(entries[0]["date"], "%Y-%m-%d")
        d2 = datetime.strptime(entries[-1]["date"], "%Y-%m-%d")
        days = (d2 - d1).days
        return max(days, 1)
        
    v1 = (get_w_kg(first_half[-1]) - get_w_kg(first_half[0])) / get_days(first_half)
    v2 = (get_w_kg(second_half[-1]) - get_w_kg(second_half[0])) / get_days(second_half)
    
    acceleration = v2 - v1
    
    body_fat_trend = 0.0
    # Try to calculate body fat trend between the first and last entry
    try:
        if 'bodyFat' in smoothed_history[0] and 'bodyFat' in smoothed_history[-1]:
            bf_start = float(smoothed_history[0]['bodyFat'])
            bf_end = float(smoothed_history[-1]['bodyFat'])
            body_fat_trend = bf_end - bf_start
    except (ValueError, TypeError):
        pass
    
    return {
        'velocity_current': v2,
        'velocity_previous': v1,
        'acceleration': acceleration,
        'body_fat_trend': body_fat_trend,
        'is_losing_decelerating': acceleration > 0 and v1 < 0,  # Perdía peso pero ya pierde más lento
        'is_losing_accelerating': acceleration < 0 and v2 < 0,  # Perdiendo peso cada vez más rápido
        'is_gaining_decelerating': acceleration < 0 and v1 > 0, # Ganaba peso pero ya gana más lento
        'is_gaining_accelerating': acceleration > 0 and v2 > 0  # Ganando peso cada vez más rápido
    }


def get_nutrition_targets(form_data: dict) -> dict:
    """
    Función principal: Orquesta todos los cálculos nutricionales.
    Recibe los datos del formulario del usuario y devuelve un dict completo
    con BMR, TDEE, calorías objetivo y macros exactos.
    """
    # Extraer datos biométricos del formulario
    # NOTA: El frontend envía el peso en la unidad que el usuario seleccionó (LB o KG)
    # junto con el campo 'weightUnit' que indica la unidad.
    # La altura siempre se almacena en CM.
    try:
        weight_raw = float(form_data.get("weight", 154))
        height = float(form_data.get("height", 170))
        age = int(form_data.get("age", 25))
    except (ValueError, TypeError):
        weight_raw, height, age = 154, 170, 25  # Defaults seguros
    
    # [P0-FORM-4] El path normal pasa por `_validate_form_data_min` (router) que
    # ya rechaza payloads sin `weightUnit`. Pero esta función también la invocan
    # callers internos (cron tasks, agent.py modify_meal, etc.) que leen perfiles
    # almacenados en DB. Si un perfil legacy no tiene la key, antes asumíamos
    # silenciosamente "lb" y producíamos un BMR incorrecto cuando el usuario
    # originalmente había ingresado kg. Ahora WARNEAMOS para tener observabilidad
    # del drift y aplicamos el default solo como fallback explícito.
    weight_unit_raw = form_data.get("weightUnit")
    if not weight_unit_raw:
        logger.warning(
            f"[P0-FORM-4] nutrition_calculator: weightUnit ausente en form_data "
            f"(user_id={form_data.get('user_id')}). Asumiendo 'lb' por compatibilidad "
            f"con perfiles legacy. Si el usuario es nuevo este es un bug del caller."
        )
        weight_unit = "lb"
    else:
        weight_unit = str(weight_unit_raw).lower().strip()
        if weight_unit not in ("lb", "kg"):
            logger.warning(
                f"[P0-FORM-4] nutrition_calculator: weightUnit={weight_unit_raw!r} "
                f"inválido (esperado 'lb' o 'kg'). Asumiendo 'lb'."
            )
            weight_unit = "lb"

    # Convertir a kilogramos si está en libras
    if weight_unit == "kg":
        weight = weight_raw
        weight_display = f"{weight_raw}kg"
    else:
        weight = round(float(weight_raw / 2.20462), 1)
        weight_display = f"{weight_raw}lbs → {weight}kg"
    
    try:
        body_fat = float(form_data.get("bodyFat")) if form_data.get("bodyFat") else None
    except (ValueError, TypeError):
        body_fat = None

    gender = form_data.get("gender", "male")
    activity_level = form_data.get("activityLevel", "moderate")
    goal = form_data.get("mainGoal") or form_data.get("goal") or "maintenance"

    # 1. BMR (Tasa Metabólica Basal)
    bmr = calculate_bmr(weight, height, age, gender, body_fat)
    
    # 2. TDEE (Gasto Energético Total)
    tdee = calculate_tdee(bmr, activity_level)
    
    # 3. Calorías objetivo (con ajuste por meta)
    target_calories = apply_goal_adjustment(tdee, goal)
    
    # --- MEJORA 4: METABOLISMO EVOLUTIVO (AJUSTE DINÁMICO DE MACROS) ---
    weight_history = form_data.get("weight_history", [])
    metabolism_notes = ""
    dynamic_deficit_bonus = 0.0
    
    if weight_history and len(weight_history) >= 2:
        try:
            from datetime import datetime
            history_sorted = sorted(weight_history, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))
            
            # 1. Aplicar Smoothing Metabólico
            smoothed_history = _get_smoothed_weight_history(history_sorted)
            
            first_entry = smoothed_history[0]
            last_entry = smoothed_history[-1]
            days_diff = (datetime.strptime(last_entry["date"], "%Y-%m-%d") - datetime.strptime(first_entry["date"], "%Y-%m-%d")).days
            
            if days_diff >= 14:  # MEJORA: Cambio de 7 a 14 días para mayor fiabilidad
                start_weight = float(first_entry["weight"])
                current_weight = float(last_entry["weight"])
                
                weight_diff = current_weight - start_weight
                pct_change = (weight_diff / start_weight) * 100
                
                velocity_data = _calculate_weight_velocity(smoothed_history)
                body_fat_trend = velocity_data.get('body_fat_trend', 0.0) if velocity_data else 0.0
                
                # 2. Detección de Fluctuación Cíclica (Fase Lútea / Retención Femenina)
                is_female = gender.lower() in ["female", "femenina", "f", "mujer"]
                raw_last_weight = float(history_sorted[-1]["weight"])
                if history_sorted[-1].get("unit", "lb") == "lb":
                    raw_last_weight /= 2.20462
                    
                raw_diff = raw_last_weight - current_weight  # current_weight es el suavizado
                is_cyclical_spike = is_female and raw_diff > 0.5 and pct_change >= 0  # Subida brusca vs la media
                
                if goal == "lose_fat":
                    if is_cyclical_spike:
                        dynamic_deficit_bonus = 0.0
                        metabolism_notes = f"✅ [METABOLISMO EVOLUTIVO]: Posible retención hídrica temporal detectada (fluctuación hormonal). Se ignora el estancamiento temporal ({pct_change:.1f}%) para evitar un déficit agresivo erróneo."
                    elif body_fat_trend < 0 and pct_change >= -0.5:
                        dynamic_deficit_bonus = 0.0
                        metabolism_notes = f"✅ [METABOLISMO EVOLUTIVO]: Recomposición corporal detectada. El peso se ha estancado ({pct_change:.1f}%) pero la grasa corporal disminuyó ({body_fat_trend:.1f}%). Manteniendo el déficit actual para proteger la ganancia muscular."
                    elif pct_change >= -0.5:
                        # Gradación proporcional al estancamiento
                        weeks_stalled = days_diff / 7
                        if weeks_stalled >= 3:
                            dynamic_deficit_bonus = -0.10 # 10% extra después de 3 semanas
                        elif weeks_stalled >= 2:
                            dynamic_deficit_bonus = -0.07
                        else:
                            dynamic_deficit_bonus = -0.05
                            
                        metabolism_notes = f"⚠️ [METABOLISMO EVOLUTIVO]: El peso del usuario se ha estancado o subido en los últimos {days_diff} días. He aplicado un déficit extra dinámico del {abs(int(dynamic_deficit_bonus*100))}% para reactivar la pérdida de grasa."
                    elif velocity_data and velocity_data['is_losing_decelerating']:
                        dynamic_deficit_bonus = -0.07
                        metabolism_notes = f"⚠️ [METABOLISMO EVOLUTIVO]: Detecté una DESACELERACIÓN en la pérdida de peso (pre-plateau). He aplicado un déficit dinámico extra del 7% proactivamente."
                    elif pct_change < -1.5 * (days_diff / 7) or (velocity_data and velocity_data['is_losing_accelerating']):
                        dynamic_deficit_bonus = +0.05
                        metabolism_notes = f"⚠️ [METABOLISMO EVOLUTIVO]: La pérdida de peso está acelerando muy rápido. He reducido el déficit un 5% para proteger la masa muscular (Anti-Rebound)."
                elif goal == "gain_muscle":
                    if pct_change <= 0.5:
                        weeks_stalled = days_diff / 7
                        if weeks_stalled >= 3:
                            dynamic_deficit_bonus = +0.10
                        elif weeks_stalled >= 2:
                            dynamic_deficit_bonus = +0.07
                        else:
                            dynamic_deficit_bonus = +0.05
                        metabolism_notes = f"⚠️ [METABOLISMO EVOLUTIVO]: El usuario busca ganar masa pero su peso está estancado. He añadido un superávit dinámico extra del {int(dynamic_deficit_bonus*100)}% para impulsar el anabolismo."
                    elif velocity_data and velocity_data['is_gaining_decelerating']:
                        dynamic_deficit_bonus = +0.07
                        metabolism_notes = f"⚠️ [METABOLISMO EVOLUTIVO]: Detecté una DESACELERACIÓN en la ganancia de masa (pre-plateau). He aplicado un superávit dinámico extra del 7% proactivamente."
                    elif pct_change > 1.0 * (days_diff / 7) or (velocity_data and velocity_data['is_gaining_accelerating']):
                        dynamic_deficit_bonus = -0.05
                        metabolism_notes = f"⚠️ [METABOLISMO EVOLUTIVO]: El peso está subiendo demasiado rápido (riesgo de grasa excesiva). He reducido el superávit un 5% (Anti-Rebound)."
                
                if dynamic_deficit_bonus != 0.0:
                    target_calories = target_calories + (tdee * dynamic_deficit_bonus)
                    target_calories = int(round(target_calories / 50) * 50)
        except Exception as e:
            logger.info(f"Error en metabolismo evolutivo: {e}")
    # -------------------------------------------------------------------

    calculation_details_str = (
        f"BMR (Mifflin-St Jeor): {bmr} kcal | "
        f"TDEE ({activity_level}, ×{ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)}): {tdee} kcal | "
        f"Objetivo ({goal}): {target_calories} kcal"
    )
    if metabolism_notes:
        calculation_details_str += f"\n{metabolism_notes}"

    # Preservar el cálculo original para el Dashboard
    original_target_calories = target_calories
    original_macros = calculate_macros(original_target_calories, goal)

    # 4. Macronutrientes exactos distribuidos en base al objetivo y calorías REVISADAS para la IA
    macros = calculate_macros(target_calories, goal)
    
    # Descripción legible del objetivo
    goal_labels = {
        "lose_fat": "Pérdida de Grasa (Déficit 20%)",
        "gain_muscle": "Ganancia Muscular (Superávit 15%)",
        "maintenance": "Mantenimiento",
        "performance": "Rendimiento Deportivo (Superávit 10%)",
    }
    
    result = {
        "bmr": bmr,
        "tdee": tdee,
        "target_calories": target_calories,
        "total_daily_calories": original_target_calories,
        "total_daily_macros": original_macros,
        "goal_label": goal_labels.get(goal, goal),
        "macros": macros,
        "calculation_details": calculation_details_str,
        "kinematics": velocity_data if 'velocity_data' in locals() else None
    }
    
    logger.info(f"\n🔢 [CALCULADORA NUTRICIONAL] Resultados exactos:")
    logger.info(f"   📊 BMR: {bmr} kcal (Peso: {weight_display}, Altura: {height}cm, Edad: {age}, Género: {gender})")
    logger.info(f"   🏃 TDEE: {tdee} kcal (Actividad: {activity_level})")
    logger.info(f"   🎯 Calorías Objetivo: {target_calories} kcal ({goal_labels.get(goal, goal)})")
    logger.info(f"   🥩 Proteína: {macros['protein_g']}g | 🍚 Carbos: {macros['carbs_g']}g | 🥑 Grasas: {macros['fats_g']}g")
    
    return result
