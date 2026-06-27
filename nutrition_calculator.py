# backend/nutrition_calculator.py
"""
Agente Calculador: Cálculos nutricionales exactos con la ecuación de Mifflin-St Jeor.
Elimina la carga matemática del LLM para evitar alucinaciones numéricas.
"""

import logging
import os  # [P2-BUDGET-FLOOR · 2026-06-21] usado por los knobs del piso de presupuesto (otras funciones lo importan local)

logger = logging.getLogger(__name__)

# [P1-PREGNANCY-DEFICIT-GATE · 2026-06-14] Gate de SEGURIDAD fail-hard: una persona embarazada o
# lactando NUNCA debe recibir un déficit calórico (riesgo fetal/de lactancia). Antes el goal genérico
# (lose_fat) aplicaba un déficit del 20% sin ninguna salvaguarda. Knob de rollback con default ON
# (auto-registrado en _KNOBS_REGISTRY). Anchor: P1-PREGNANCY-DEFICIT-GATE.
try:
    from knobs import _env_bool as _nc_env_bool
    PREGNANCY_DEFICIT_GATE_ENABLED = _nc_env_bool("MEALFIT_PREGNANCY_DEFICIT_GATE", True)
except Exception:  # pragma: no cover - knobs siempre disponible en prod; fail-safe a ON (seguro)
    PREGNANCY_DEFICIT_GATE_ENABLED = True

# [P1-MIN-CALORIE-FLOOR · 2026-06-15] (gap-audit P1-2) Piso mínimo de calorías GENERAL (no solo embarazo).
# Hasta ahora el único floor calórico era el de embarazo/lactancia; un perfil válido pero extremo (mujer
# pequeña/mayor en déficit → ~850 kcal) caía bajo el mínimo clínico sin floor ni flag. Estándar clínico
# conservador: ~1200 kcal/día mujer, ~1500 kcal/día hombre (umbral bajo el cual un déficit sostenido es
# riesgoso: pérdida muscular, déficit de micronutrientes). Floor de seguridad + flag `low_calorie_floored`
# → gate de revisión profesional (FS9) aguas abajo. Auto-registrado en _KNOBS_REGISTRY. Anchor: P1-MIN-CALORIE-FLOOR.
try:
    from knobs import _env_int as _nc_env_int
    # [P2-MIN-KCAL-KNOB-CLAMP · 2026-06-18] (audit fresco P2) `validator=` rango [800, 4000]: un valor
    # inválido (0, typo "120", negativo) NO debe desactivar/degradar SILENCIOSAMENTE el piso clínico de
    # calorías — cae al default seguro + marca parse_failed (visible en /health/version). Es el único guard
    # de seguridad de esta dimensión que sin clamp podía romperse por un número fuera de rango sin parse-error.
    _MIN_KCAL_RANGE = lambda v: 800 <= v <= 4000  # noqa: E731 (lambda inline, consistente con otros knobs)
    MIN_TARGET_KCAL_FEMALE = _nc_env_int("MEALFIT_MIN_TARGET_KCAL_FEMALE", 1200, validator=_MIN_KCAL_RANGE)
    MIN_TARGET_KCAL_MALE = _nc_env_int("MEALFIT_MIN_TARGET_KCAL_MALE", 1500, validator=_MIN_KCAL_RANGE)
except Exception:  # pragma: no cover - knobs siempre disponible en prod; fail-safe a defaults clínicos
    MIN_TARGET_KCAL_FEMALE = 1200
    MIN_TARGET_KCAL_MALE = 1500

# [P1-MINOR-SAFETY-GATE · 2026-06-18] (audit fresco P1-A) Gate de SEGURIDAD para menores de edad (<18).
# El formulario acepta edades 12-17 (router `_BIO_RANGES["age"]=(12,100)`) pero el pipeline los trataba como
# adultos: BMR Mifflin (no validada en adolescentes), déficit -20% permitido sobre un cuerpo en crecimiento,
# piso de kcal de adulto, y CERO gate de revisión profesional por edad. Es la simétrica del gate de embarazo
# (P1-PREGNANCY-DEFICIT-GATE). Por seguridad: un menor NUNCA recibe déficit calórico (se fuerza al menos
# mantenimiento) y el plan SIEMPRE queda con `requires_professional_review` (FS9). NO inventamos un piso
# pediátrico de kcal ni ecuaciones pediátricas (Schofield/IOM) — eso requiere validación clínica; el flag FS9
# deriva esos ajustes a un profesional. Knob de rollback default ON (auto-registrado). Anchor: P1-MINOR-SAFETY-GATE.
try:
    from knobs import _env_bool as _nc_env_bool_minor
    MINOR_SAFETY_GATE_ENABLED = _nc_env_bool_minor("MEALFIT_MINOR_SAFETY_GATE", True)
except Exception:  # pragma: no cover - knobs siempre disponible en prod; fail-safe a ON (seguro)
    MINOR_SAFETY_GATE_ENABLED = True


def _min_target_kcal(gender) -> int:
    """[P1-MIN-CALORIE-FLOOR · 2026-06-15] Piso clínico de calorías por sexo (mujer 1200 / hombre 1500).
    Sexo explícito no-male (female/'') → piso de mujer (más bajo, no sobre-floorea). NOTA: el caller
    `get_nutrition_targets` ya defaultea `gender='male'` cuando la key falta, así que un perfil SIN género
    resuelve al piso de hombre (1500) — consistente con `calculate_bmr`, que usa el mismo default male.
    Anchor: P1-MIN-CALORIE-FLOOR."""
    g = str(gender or "").strip().lower()
    if g in ("male", "masculino", "masculina", "hombre", "m"):  # 'masculina' por simetría
        return MIN_TARGET_KCAL_MALE
    return MIN_TARGET_KCAL_FEMALE


def _is_pregnancy_or_lactation(form_data) -> bool:
    """True si el perfil declara embarazo/lactancia (en `medicalConditions` o en un campo dedicado).
    Sobre-inclusivo a propósito: un falso positivo solo evita un déficit (seguro); un falso negativo
    aplicaría un déficit a una embarazada (peligroso). Anchor: P1-PREGNANCY-DEFICIT-GATE."""
    if not isinstance(form_data, dict):
        return False
    # Campo dedicado opcional (algunos forms lo envían aparte de medicalConditions).
    for _k in ("isPregnant", "pregnant", "embarazada", "isLactating", "lactating", "breastfeeding"):
        _v = form_data.get(_k)
        if _v is True:
            return True
        if isinstance(_v, str) and _v.strip().lower() in ("1", "true", "yes", "si", "sí", "on"):
            return True
    try:
        from constants import PREGNANCY_CONDITION_TERMS, strip_accents
    except Exception:
        return False
    raw = form_data.get("medicalConditions") or form_data.get("medical_conditions") or []
    if isinstance(raw, str):
        raw = [raw]
    for _c in raw:
        try:
            _cl = strip_accents(str(_c).strip().lower())
        except Exception:
            _cl = str(_c).strip().lower()
        if _cl and any(_t in _cl for _t in PREGNANCY_CONDITION_TERMS):
            return True
    return False


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
        _g = str(gender or "").strip().lower()  # None-safe (un caller interno podría pasar gender=None)
        if _g in ("male", "masculino", "masculina", "m", "hombre"):  # 'masculina' por simetría con femenino/femenina
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
        else:
            # [P2-GENDER-ENUM-WARN · 2026-06-18] (audit fresco P2) El else aplica la ecuación FEMENINA; un
            # `gender` no reconocido (other/no-binario/typo/'') cae aquí en SILENCIO con un sesgo de ~-166 kcal.
            # Warneamos para observabilidad (mismo patrón P0-FORM-5 de activity/goal): el path normal pasa el
            # enum del router; un warning aquí señala caller interno (cron/agent/perfil legacy) o drift de schema.
            if _g not in ("female", "femenino", "femenina", "f", "mujer"):
                logger.warning(f"[P2-GENDER-ENUM-WARN] gender={gender!r} no reconocido (ni male ni female) — "
                               f"se aplica la ecuación/piso femenino por default.")
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


# [P2-SOLVER-KNOBS-REGISTRY · 2026-06-18] (audit fresco P2) Lectura module-level vía `_env_float`
# (auto-registra en _KNOBS_REGISTRY → visible en /health/version); antes os.environ crudo per-call eludía
# el registry. El clamp [1.6, 3.0] se preserva en la función. Fail-safe a 2.2.
try:
    from knobs import _env_float as _nc_env_float
    _PROTEIN_CEILING_RAW = _nc_env_float("MEALFIT_PROTEIN_CEILING_G_PER_KG", 2.2)
except Exception:  # pragma: no cover - knobs siempre disponible en prod
    _PROTEIN_CEILING_RAW = 2.2


def _protein_ceiling_g_per_kg() -> float:
    """[C1-PROTEIN-CEILING · 2026-06-13] Techo clínico de proteína por kg de peso corporal.
    Knob `MEALFIT_PROTEIN_CEILING_G_PER_KG` (default 2.2, clamp [1.6, 3.0]). Posición ISSN:
    1.6-2.2 g/kg cubre ganancia/preservación de músculo; >2.4 no aporta beneficio adicional
    y es difícil de cumplir dentro del presupuesto calórico. El split por % de calorías
    (30% para gain_muscle) puede dar 2.8+ g/kg en personas livianas con TDEE alto."""
    try:
        return max(1.6, min(3.0, float(_PROTEIN_CEILING_RAW)))
    except (TypeError, ValueError):
        return 2.2


def calculate_macros(target_calories: int, goal: str, weight_kg: float = None,
                     body_fat_pct: float = None) -> dict:
    """
    Calcula los gramos exactos de cada macronutriente basándose en:
    - Proteína: 4 cal/g
    - Carbohidratos: 4 cal/g
    - Grasas: 9 cal/g

    [C1-PROTEIN-CEILING · 2026-06-13] Si `weight_kg` se provee, la proteína se CAPEA a un
    techo clínico (`_protein_ceiling_g_per_kg()` × peso) y las calorías liberadas se
    redistribuyen a carbohidratos (el macro flexible). Sin esto, el % de calorías producía
    targets de 2.8+ g/kg (inalcanzables sin sobre-cargar proteína en cada comida y reñidos
    con el presupuesto calórico → el plan quedaba sistemáticamente corto). Anchor: C1-PROTEIN-CEILING.

    [P2-PROTEIN-CEILING-ADJ-WEIGHT · 2026-06-18] (audit fresco P2) Si `body_fat_pct` > 30 (obesidad), el
    techo se calcula sobre el PESO AJUSTADO (LBM + 0.25×(peso−LBM)), no el peso total: 2.2 g/kg × peso total
    sobre-prescribe en obesidad (ej. 150 kg/45% grasa → 330 g/día). La práctica clínica usa peso ajustado/masa
    magra. Sin `body_fat_pct` o ≤30% se mantiene el peso total (comportamiento previo, cero regresión).
    """
    split = MACRO_SPLITS.get(goal, MACRO_SPLITS["maintenance"])

    protein_cals = target_calories * split["protein_pct"]
    carbs_cals = target_calories * split["carbs_pct"]
    fats_cals = target_calories * split["fats_pct"]

    protein_g = protein_cals / 4.0
    if weight_kg and weight_kg > 0:
        _ceiling_wkg = float(weight_kg)
        if body_fat_pct and body_fat_pct > 30:
            _lbm = float(weight_kg) * (1 - (float(body_fat_pct) / 100.0))
            _ceiling_wkg = _lbm + 0.25 * (float(weight_kg) - _lbm)  # peso ajustado (obesidad)
        ceiling_g = _protein_ceiling_g_per_kg() * _ceiling_wkg
        if protein_g > ceiling_g:
            freed_cals = (protein_g - ceiling_g) * 4.0
            protein_g = ceiling_g
            protein_cals = protein_g * 4.0
            carbs_cals += freed_cals  # redistribuir a carbos (macro flexible)
            _wlabel = "ajustado" if abs(_ceiling_wkg - float(weight_kg)) > 0.05 else "total"
            logger.info(
                f"🩺 [C1-PROTEIN-CEILING] Proteína capeada a {_protein_ceiling_g_per_kg()} g/kg "
                f"× {round(_ceiling_wkg, 1)}kg ({_wlabel}) = {round(protein_g)}g "
                f"(era {round(target_calories * split['protein_pct'] / 4)}g); {round(freed_cals)} kcal → carbos."
            )

    return {
        "protein_g": round(protein_g),
        "carbs_g": round(carbs_cals / 4),
        "fats_g": round(fats_cals / 9),
        "protein_str": f"{round(protein_g)}g",
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
    # [P1-CLINICAL-MEAL-COUNT · 2026-06-27] 2 y 6 añadidos para cubrir el rango clínico 2-6
    # (pocas e intensas ↔ pequeñas y frecuentes). 2 es override-only (raro); 6 = bariátrica/hipoglucemia.
    2: {"almuerzo": 0.50, "cena": 0.50},
    3: {"desayuno": 0.30, "almuerzo": 0.40, "cena": 0.30},
    4: {"desayuno": 0.20, "almuerzo": 0.35, "merienda": 0.15, "cena": 0.30},
    5: {"desayuno": 0.20, "merienda_am": 0.10, "almuerzo": 0.35,
        "merienda_pm": 0.10, "cena": 0.25},
    6: {"desayuno": 0.18, "merienda_am": 0.12, "almuerzo": 0.28,
        "merienda_pm": 0.12, "cena": 0.22, "merienda_noche": 0.08},
}

# [P1-CLINICAL-MEAL-COUNT · 2026-06-27] meal_types canónicos por cantidad de comidas. El planner produce
# `meal_types` (DaySkeletonModel) y el day_generator genera EXACTAMENTE esos slots (build_day_assignment_context
# línea ~374). Orden alineado con el reparto de meriendas de MEAL_SLOT_SPLITS (AM→PM→Nocturna).
MEAL_TYPES_BY_COUNT: dict = {
    2: ["Almuerzo", "Cena"],
    3: ["Desayuno", "Almuerzo", "Cena"],
    4: ["Desayuno", "Almuerzo", "Merienda", "Cena"],
    5: ["Desayuno", "Merienda AM", "Almuerzo", "Merienda PM", "Cena"],
    6: ["Desayuno", "Merienda AM", "Almuerzo", "Merienda PM", "Cena", "Merienda Nocturna"],
}


def meal_types_for_count(n: int) -> list:
    """[P1-CLINICAL-MEAL-COUNT · 2026-06-27] Lista de `meal_types` es-DO para `n` comidas (cae a 4 si
    `n` no está en el set). tooltip-anchor: P1-CLINICAL-MEAL-COUNT"""
    return list(MEAL_TYPES_BY_COUNT.get(n, MEAL_TYPES_BY_COUNT[4]))


def _mc_norm_text(value) -> str:
    """Normaliza condiciones/medicamentos a un blob lower sin acentos para matching. Aplana listas
    ANIDADAS y strings (acepta `[medicalConditions_list, otherConditions_str, ...]`) → así el decisor
    ve también el texto libre del formulario (otherConditions/otherMedications)."""
    import unicodedata
    parts: list = []

    def _walk(v):
        if v is None:
            return
        if isinstance(v, (list, tuple)):
            for x in v:
                _walk(x)
        else:
            parts.append(str(v))

    _walk(value)
    raw = " ".join(parts)
    return unicodedata.normalize("NFD", raw.lower()).encode("ascii", "ignore").decode("ascii")


def decide_meals_per_day(form_data: dict, daily_kcal: float | None = None) -> dict:
    """[P1-CLINICAL-MEAL-COUNT · 2026-06-27] Decide CUÁNTAS comidas/día debe tener el plan, mapeando la
    evidencia clínica (tabla del owner): 2-3 'pocas e intensas' (mejoran sensibilidad a la insulina) ↔
    4-6 'pequeñas y frecuentes' (evitan hipoglucemia, facilitan altas calorías sin saturar). Es el lado
    "nivel clínico" del feature: la frecuencia la deriva la IA de los datos médicos del formulario, NO el
    usuario (salvo override explícito `num_meals`/`mealsPerDay`).

    Prioridad (la SEGURIDAD manda — el riesgo de hipoglucemia SIEMPRE gana sobre el reduce de DM2):
      1. Override explícito del usuario (num_meals/mealsPerDay/meals_per_day), clamp [2,6].
      2. Cirugía bariátrica → 6 (porciones pequeñas y frecuentes).
      3. Riesgo de hipoglucemia — condición (hipoglucemia) O medicación (insulina/sulfonilureas) → 5
         (comidas frecuentes evitan caídas de glucosa). DEBE ir ANTES de DM2: un DM2 en insulina NO
         debe reducir comidas (sería iatrogénico).
      4. DM2 / resistencia a la insulina / prediabetes / síndrome metabólico (SIN riesgo de hipo) → 3
         (pocas e intensas mejoran la sensibilidad a la insulina).
      5. Gasto calórico muy alto (≥2900 kcal/día, atleta) → 5 (repartir evita saturar).
      6. Default → 4 (estructura estándar del producto).

    Devuelve {"num_meals": int, "reason": str, "source": "override"|"clinical"|"default"}. Fail-safe →
    4/default ante cualquier error. tooltip-anchor: P1-CLINICAL-MEAL-COUNT"""
    try:
        fd = form_data or {}
        # 1. Override explícito del usuario (campo opcional del formulario / data hook).
        raw = fd.get("num_meals") or fd.get("mealsPerDay") or fd.get("meals_per_day")
        if raw not in (None, "", 0):
            try:
                n = int(raw)
                if 2 <= n <= 6:
                    return {"num_meals": n, "reason": "preferencia explícita del usuario", "source": "override"}
            except (TypeError, ValueError):
                pass
        # Incluye el TEXTO LIBRE del formulario (otherConditions/otherMedications) además de los chips,
        # para que "hipoglucemia"/"cirugía bariátrica" escritos a mano también disparen la decisión.
        conds = _mc_norm_text([
            fd.get("medicalConditions"), fd.get("medical_conditions"), fd.get("conditions"),
            fd.get("otherConditions"), fd.get("other_conditions"),
        ])
        meds = _mc_norm_text([
            fd.get("medications"), fd.get("medicamentos"),
            fd.get("otherMedications"), fd.get("other_medications"),
        ])
        # 2. Bariátrica. [P1-BARIATRIC-CLINICAL-RULES] tokens SSOT en constants (compartidos con la ConditionRule).
        try:
            from constants import BARIATRIC_CONDITION_TERMS as _BARIA_TERMS
        except Exception:
            _BARIA_TERMS = ("bariatr", "bypass gastric", "manga gastric", "gastrectom", "sleeve gastric")
        if any(k in conds for k in _BARIA_TERMS):
            return {"num_meals": 6, "reason": "post-cirugía bariátrica: porciones pequeñas y frecuentes", "source": "clinical"}
        # 3. Riesgo de hipoglucemia (condición o medicación) — SEGURIDAD, antes de DM2.
        hypo_cond = any(k in conds for k in ("hipoglucemia", "hipoglicemia", "hypoglycemia"))
        hypo_med = any(k in meds for k in (
            "insulina", "insulin", "glibenclamida", "gliburida", "glimepirida", "glipizida",
            "gliclazida", "glicazida", "sulfonilurea", "sulfonylurea"))
        if hypo_cond or hypo_med:
            return {"num_meals": 5,
                    "reason": "riesgo de hipoglucemia (condición o medicación): comidas frecuentes evitan caídas de glucosa",
                    "source": "clinical"}
        # 4. DM2 / resistencia a la insulina / prediabetes (sin riesgo de hipo).
        if any(k in conds for k in (
                "diabetes tipo 2", "diabetes mellitus tipo 2", "diabetes t2", "diabetes tipo ii",
                "dm2", "dm 2", "dm-2",
                "resistencia a la insulina", "insulin resistance", "prediabetes", "prediabetico",
                "sindrome metabolico", "metabolic syndrome")):
            return {"num_meals": 3,
                    "reason": "DM2/resistencia a la insulina: pocas comidas e intensas mejoran la sensibilidad a la insulina",
                    "source": "clinical"}
        # 5. Gasto calórico muy alto.
        try:
            _k = float(daily_kcal or 0)
        except (TypeError, ValueError):
            _k = 0.0
        if _k >= 2900:
            return {"num_meals": 5,
                    "reason": f"gasto calórico alto (~{round(_k)} kcal): repartir en más comidas evita saturar la digestión",
                    "source": "clinical"}
        # 6. Default.
        return {"num_meals": 4, "reason": "estructura estándar (4 comidas)", "source": "default"}
    except Exception:
        return {"num_meals": 4, "reason": "fallback (error en el decisor)", "source": "default"}


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


def _swap_cals_tolerance_mult() -> float:
    """[P2-SWAP-CALS-TOLERANCE-KNOB · 2026-06-23] (audit inteligencia P2-13) Multiplicador de la
    tolerancia calórica vs la base. Knob `MEALFIT_SWAP_CALS_TOLERANCE_MULT` (default 1.5, clamp
    [1.0, 2.0]). Las kcal varían más que protein/carbs/fats por side dishes → la base se relaja ×N
    SOLO para kcal. El default 1.5 (→ ±22.5% con base 0.15) es decisión testeada; un swap puede
    persistir ese drift sin re-escalar (P1-2 rebalancer lo cierra al re-apuntar al target). Bajarlo
    a ~1.2 (→ ±18%) estrecha el drift kcal por-comida SIN redeploy — medir tasa de retries/422 antes.
    Tooltip-anchor: P2-SWAP-CALS-TOLERANCE-KNOB.
    """
    import os
    try:
        v = float(os.environ.get("MEALFIT_SWAP_CALS_TOLERANCE_MULT", "1.5"))
    except (TypeError, ValueError):
        return 1.5
    return max(1.0, min(2.0, v))


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
    cals_tolerance = min(1.0, tolerance_pct * _swap_cals_tolerance_mult())  # [P2-SWAP-CALS-TOLERANCE-KNOB]

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
    _age_defaulted = False
    try:
        weight_raw = float(form_data.get("weight", 154))
        height = float(form_data.get("height", 170))
        # [P1-MINOR-SAFETY-GATE review] `int(float(...))` (no `int(...)`) para igualar al router
        # (`_coerce_numeric kind='int'`): un age string-decimal ("17.5") NO debe caer al default 25
        # (adulto) y eludir el gate de menores — int("17.5") lanza, int(float("17.5"))=17.
        _age_raw = form_data.get("age")
        age = int(float(_age_raw)) if _age_raw is not None else 25
        _age_defaulted = _age_raw is None
    except (ValueError, TypeError):
        weight_raw, height, age = 154, 170, 25  # Defaults seguros
        _age_defaulted = True
    # [P2-MINOR-GATE-SILENT-DEFAULT · 2026-06-22] (audit fresco P2-7) age ausente/no-parseable cae al default
    # 25 (adulto) → un menor real de un caller interno legacy eludiría el gate de menores (BMR adulto + déficit
    # permitido, sin FS9) SIN rastro. El path normal (router `_validate_form_data_min`) siempre envía age válido
    # → esto solo afecta callers internos que leen perfiles legacy. WARNEAMOS para observabilidad (mismo patrón
    # que weightUnit P0-FORM-4): un operador que vea esta línea para un usuario real investiga el caller.
    # tooltip-anchor: P2-MINOR-GATE-SILENT-DEFAULT
    if _age_defaulted:
        logger.warning(
            f"[P2-MINOR-GATE-SILENT-DEFAULT] nutrition_calculator: age ausente/no-parseable en form_data "
            f"(user_id={form_data.get('user_id')}, raw={form_data.get('age')!r}). Asumiendo 25 (adulto) — si "
            f"el usuario fuera MENOR, el gate de menores (FS9) NO se aplicaría. Bug del caller si es un perfil "
            f"real; el path por router valida age."
        )
    
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

    # [P1-PREGNANCY-DEFICIT-GATE · 2026-06-14] SEGURIDAD fail-hard ANTES de calcular calorías: si se
    # declaró embarazo/lactancia y la meta produciría un déficit (lose_fat), se fuerza mantenimiento.
    # El piso de TDEE más abajo es el cinturón-y-tirantes (cubre cualquier déficit dinámico residual).
    # El plan además queda con requires_professional_review (FS9) vía las condiciones médicas.
    _pregnancy_safety = None
    if PREGNANCY_DEFICIT_GATE_ENABLED and _is_pregnancy_or_lactation(form_data):
        _orig_goal = goal
        # [P2-PREGNANCY-GATE-ADJUSTMENT · 2026-06-19] (audit fresco P2-16) Basado en el AJUSTE numérico, no en el
        # string 'lose_fat' — espejo del gate de menores (línea ~1086). Hoy son equivalentes (lose_fat es la única
        # meta deficitaria en GOAL_ADJUSTMENTS), pero si se añade una meta deficitaria nueva, el gate de embarazo
        # la neutralizaría igual (el piso TDEE de abajo ya protege las kcal; esto cierra la asimetría de telemetría).
        if GOAL_ADJUSTMENTS.get(goal, 0.0) < 0:
            goal = "maintenance"
        _pregnancy_safety = {
            "applied": True,
            "original_goal": _orig_goal,
            "effective_goal": goal,
            "note": ("🤰 EMBARAZO/LACTANCIA DETECTADO — por seguridad NO se aplica déficit calórico "
                     "(se usa al menos mantenimiento). El requerimiento energético sube en el 2º/3º "
                     "trimestre y durante la lactancia; este plan es ORIENTATIVO y DEBE ser supervisado "
                     "por tu obstetra/nutricionista: prioriza folato y hierro, y evita pescados altos en "
                     "mercurio, embutidos y quesos/lácteos no pasteurizados (riesgo de listeria)."),
        }
        logger.warning(f"🤰 [P1-PREGNANCY-DEFICIT-GATE] Embarazo/lactancia → meta '{_orig_goal}' "
                       f"forzada a '{goal}' (sin déficit).")

    # [P1-MINOR-SAFETY-GATE · 2026-06-18] (audit fresco P1-A) SEGURIDAD para menores: simétrica del gate de
    # embarazo. Un menor (<18) NUNCA recibe déficit calórico (se fuerza al menos mantenimiento); el piso de
    # TDEE más abajo es el cinturón-y-tirantes. El plan queda con `requires_professional_review` (FS9) vía
    # `minor_safety` aguas abajo (Guard 8c). NO se cambian las ecuaciones (BMR/floor de adulto) — eso lo
    # deriva el flag FS9 a un profesional, sin inventar valores pediátricos no validados.
    _minor_safety = None
    if MINOR_SAFETY_GATE_ENABLED and 0 < age < 18:  # `age` ya es int (parseado arriba con int(float(...)))
        _orig_goal_minor = form_data.get("mainGoal") or form_data.get("goal") or "maintenance"
        # Basado en el AJUSTE, no en el string 'lose_fat': cualquier meta deficitaria (ahora o futura en
        # GOAL_ADJUSTMENTS) se neutraliza a mantenimiento para un menor. El piso TDEE de abajo es el respaldo.
        if GOAL_ADJUSTMENTS.get(goal, 0.0) < 0:
            goal = "maintenance"
        _minor_safety = {
            "applied": True,
            "age": age,
            "original_goal": _orig_goal_minor,
            "effective_goal": goal,
            "note": ("🧒 MENOR DE EDAD DETECTADO (edad " + str(age) + ") — por seguridad NO se aplica déficit "
                     "calórico (un cuerpo en crecimiento no debe restringir energía) y este plan es ORIENTATIVO: "
                     "las necesidades de un adolescente las define un pediatra/nutricionista (las ecuaciones de "
                     "este plan están calibradas para adultos). Consúltalo antes de seguir este plan."),
        }
        logger.warning(f"🧒 [P1-MINOR-SAFETY-GATE] Menor de edad (edad={age}) → meta '{_orig_goal_minor}' "
                       f"efectiva '{goal}' (sin déficit); requiere revisión profesional (FS9).")

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

    # [P1-PREGNANCY-DEFICIT-GATE · 2026-06-14] Piso de seguridad cinturón-y-tirantes: tras CUALQUIER
    # ajuste (meta + déficit dinámico), una embarazada/lactante jamás queda por debajo de mantenimiento.
    if _pregnancy_safety and target_calories < tdee:
        target_calories = int(round(tdee / 50) * 50)
        _pregnancy_safety["floored_to_tdee"] = target_calories

    # [P1-MINOR-SAFETY-GATE · 2026-06-18] Piso cinturón-y-tirantes: tras CUALQUIER ajuste (meta + déficit
    # dinámico), un menor jamás queda por debajo de mantenimiento (TDEE).
    if _minor_safety and target_calories < tdee:
        target_calories = int(round(tdee / 50) * 50)
        _minor_safety["floored_to_tdee"] = target_calories

    # [P1-MIN-CALORIE-FLOOR · 2026-06-15] (gap-audit P1-2) Piso clínico GENERAL tras TODOS los ajustes
    # (meta + déficit dinámico + piso de embarazo): ningún objetivo cae bajo el mínimo seguro por sexo.
    # Si floorea, se registra `_low_calorie_floored` para cablear el gate de revisión profesional (FS9).
    _low_calorie_floored = None
    _min_kcal = _min_target_kcal(gender)
    if target_calories < _min_kcal:
        _pre_floor_calories = target_calories
        target_calories = _min_kcal
        _low_calorie_floored = {
            "applied": True,
            "pre_floor_calories": _pre_floor_calories,
            "floored_to": _min_kcal,
            "gender": gender,
        }
        logger.warning(f"⚠️ [P1-MIN-CALORIE-FLOOR] Objetivo {_pre_floor_calories} kcal < piso clínico "
                       f"{_min_kcal} kcal ({gender}) → elevado al piso. Requiere revisión profesional (FS9).")

    calculation_details_str = (
        f"BMR (Mifflin-St Jeor): {bmr} kcal | "
        f"TDEE ({activity_level}, ×{ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)}): {tdee} kcal | "
        f"Objetivo ({goal}): {target_calories} kcal"
    )
    if metabolism_notes:
        calculation_details_str += f"\n{metabolism_notes}"

    # Preservar el cálculo original para el Dashboard
    original_target_calories = target_calories
    # [C1-PROTEIN-CEILING] `weight` (kg) → techo clínico de proteína por peso corporal.
    # [P2-PROTEIN-CEILING-ADJ-WEIGHT] `body_fat` → peso ajustado para el techo en obesidad (>30% grasa).
    original_macros = calculate_macros(original_target_calories, goal, weight_kg=weight, body_fat_pct=body_fat)

    # 4. Macronutrientes exactos distribuidos en base al objetivo y calorías REVISADAS para la IA
    macros = calculate_macros(target_calories, goal, weight_kg=weight, body_fat_pct=body_fat)

    # [P1-BARIATRIC-PROTEIN-TARGET · 2026-06-27] El pouch post-bariátrico no tolera el volumen de proteína
    # de un target estándar por peso (visto en vivo corr=5b30b71f: target 100g → la comida pequeña no lo
    # alcanza → el gate de piso de proteína + el revisor rechazan por DÉFICIT). La guía bariátrica es 60-90g/día
    # de alta calidad, NO el target por peso corporal. Capeamos la proteína a un máximo bariátrico-apropiado
    # (knob, default 90g) y redistribuimos las kcal liberadas a GRASA (no carbos → evita carga glucémica/dumping)
    # para mantener el total calórico. Hace el piso alcanzable en volumen pequeño. tooltip-anchor: P1-BARIATRIC-PROTEIN-TARGET
    try:
        from constants import BARIATRIC_CONDITION_TERMS as _BARIA_T, strip_accents as _sa_b
        _cond_blob_b = _sa_b(
            " ".join(str(x) for x in (form_data.get("medicalConditions") or []))
            + " " + str(form_data.get("otherConditions") or "")
        ).lower()
        _is_baria = any(t in _cond_blob_b for t in _BARIA_T)
    except Exception:
        _is_baria = False
    if _is_baria:
        _baria_cap = _nc_env_float("MEALFIT_BARIATRIC_PROTEIN_MAX_G", 90.0)
        for _mac in (macros, original_macros):
            try:
                _p = float(_mac.get("protein_g") or 0)
                if _p > _baria_cap:
                    _freed = (_p - _baria_cap) * 4.0
                    _mac["protein_g"] = round(_baria_cap)
                    _mac["protein_str"] = f"{round(_baria_cap)}g"
                    _new_fats = round(float(_mac.get("fats_g") or 0) + _freed / 9.0)
                    _mac["fats_g"] = _new_fats
                    _mac["fats_str"] = f"{_new_fats}g"
            except Exception:
                pass
        logger.info(f"🔻 [P1-BARIATRIC-PROTEIN-TARGET] proteína bariátrica capeada a ≤{round(_baria_cap)}g/día "
                    f"(volumen del pouch); kcal liberadas → grasa.")

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
    if _pregnancy_safety:
        result["pregnancy_lactation_safety"] = _pregnancy_safety
    if _low_calorie_floored:
        result["low_calorie_floored"] = _low_calorie_floored
    if _minor_safety:
        result["minor_safety"] = _minor_safety

    logger.info(f"\n🔢 [CALCULADORA NUTRICIONAL] Resultados exactos:")
    logger.info(f"   📊 BMR: {bmr} kcal (Peso: {weight_display}, Altura: {height}cm, Edad: {age}, Género: {gender})")
    logger.info(f"   🏃 TDEE: {tdee} kcal (Actividad: {activity_level})")
    logger.info(f"   🎯 Calorías Objetivo: {target_calories} kcal ({goal_labels.get(goal, goal)})")
    logger.info(f"   🥩 Proteína: {macros['protein_g']}g | 🍚 Carbos: {macros['carbs_g']}g | 🥑 Grasas: {macros['fats_g']}g")

    return result


# ============================================================
# [P2-BUDGET-FLOOR · 2026-06-21] Piso de presupuesto escalado por metas (Fase 3 del build
# "todo terreno" pedido por el owner). Decisión de producto del owner: BLOQUEAR si el
# presupuesto declarado (modo 'custom') es físicamente insuficiente para alimentar al usuario
# según SUS metas, y pedir ajuste — NUNCA bajar la calidad nutricional para encajar en un
# precio demasiado bajo. El presupuesto NO toca los pisos clínicos (proteína/calorías son
# budget-blind, ver Fase 2); este floor sólo evita prometer un plan profesional con un monto
# imposible. Tooltip-anchor: P2-BUDGET-FLOOR.
# ============================================================
_GROCERY_DURATION_DAYS = {"weekly": 7, "biweekly": 15, "monthly": 30}


def _budget_floor_enabled() -> bool:
    return os.environ.get("MEALFIT_BUDGET_FLOOR_ENABLED", "true").lower() != "false"


# [BUDGET-MIN-NONLINEAR · 2026-06-23] Piso TOTAL por ciclo (DOP) a la caloría de referencia,
# household 1. NO lineal: ciclos largos tienen descuento por compra grande (pedido del owner:
# 7d=4000, 15d=7000, 30d=13000; antes era lineal 571.43 RD$/día → 15d=8571, 30d=17143). DEBE
# quedar consistente con BUDGET_MIN_TOTAL del frontend (formValidation.js). cal_scale (calorías)
# y household se aplican ENCIMA en min_budget_for_goals. Supersede el knob per-día
# MEALFIT_BUDGET_FLOOR_PER_DAY_DOP (ahora inerte). Knob por ciclo para tunear con la inflación.
_BUDGET_CYCLE_FLOOR_DEFAULTS_DOP = {7: "4000", 15: "7000", 30: "13000"}


def _budget_cycle_floor_dop(days: int) -> float:
    """Piso de presupuesto TOTAL del ciclo en DOP a la caloría de referencia (household 1),
    NO lineal. Knob por ciclo: MEALFIT_BUDGET_FLOOR_TOTAL_{7,15,30}D_DOP."""
    default = _BUDGET_CYCLE_FLOOR_DEFAULTS_DOP.get(int(days))
    if default is None:
        # Ciclo no estándar (no debería ocurrir): interpola desde el piso de 7 días (conservador).
        per_day_7 = float(_BUDGET_CYCLE_FLOOR_DEFAULTS_DOP[7]) / 7.0
        return max(0.0, per_day_7 * max(1, int(days)))
    try:
        return max(0.0, float(os.environ.get(f"MEALFIT_BUDGET_FLOOR_TOTAL_{int(days)}D_DOP", default)))
    except (TypeError, ValueError):
        return float(default)


def _budget_floor_kcal_ref() -> float:
    try:
        return max(800.0, float(os.environ.get("MEALFIT_BUDGET_FLOOR_KCAL_REF", "2000")))
    except (TypeError, ValueError):
        return 2000.0


def _budget_usd_to_dop() -> float:
    try:
        return max(1.0, float(os.environ.get("MEALFIT_BUDGET_USD_TO_DOP", "60")))
    except (TypeError, ValueError):
        return 60.0


def _budget_floor_tolerance_pct() -> float:
    try:
        v = float(os.environ.get("MEALFIT_BUDGET_FLOOR_TOLERANCE_PCT", "0.05"))
        return min(0.5, max(0.0, v))
    except (TypeError, ValueError):
        return 0.05


def min_budget_for_goals(form_data: dict) -> dict:
    """Estima el presupuesto MÍNIMO (en DOP) para alimentar al usuario según sus metas
    (calorías objetivo), su ciclo de compras (días) y su hogar (personas). [BUDGET-MIN-NONLINEAR ·
    2026-06-23] Parte de un piso TOTAL por ciclo NO lineal (7d=4000, 15d=7000, 30d=13000;
    descuento por compra grande) calibrado al frontend, y lo escala por calorías (un usuario de
    3500 kcal necesita más comida → más presupuesto) y por hogar. Conservador (lower bound) para
    NO sobre-bloquear. Devuelve {min_budget_dop, min_per_day_dop, days, household, target_calories}."""
    days = _GROCERY_DURATION_DAYS.get(str(form_data.get("groceryDuration") or "weekly").lower(), 7)
    try:
        household = int(float(form_data.get("householdSize") or 1))
    except (TypeError, ValueError):
        household = 1
    household = min(12, max(1, household))
    try:
        nutr = get_nutrition_targets(form_data)
        target_calories = int(nutr.get("target_calories") or 0)
    except Exception as e:
        logger.warning(f"[P2-BUDGET-FLOOR] get_nutrition_targets falló ({type(e).__name__}); usando ref")
        target_calories = 0
    if target_calories <= 0:
        target_calories = int(_budget_floor_kcal_ref())
    # [BUDGET-MIN-NONLINEAR · 2026-06-23] Piso por ciclo (no lineal) × escalado por calorías ×
    # hogar. min_per_day se deriva sólo para el display/back-compat del dict.
    cycle_base = _budget_cycle_floor_dop(days)
    cal_scale = max(1.0, target_calories / _budget_floor_kcal_ref())
    min_budget_dop = cycle_base * cal_scale * household
    min_per_day = min_budget_dop / max(1, days)
    return {
        "min_budget_dop": round(min_budget_dop),
        "min_per_day_dop": round(min_per_day),
        "days": days,
        "household": household,
        "target_calories": target_calories,
    }


def validate_budget_sufficient(form_data: dict) -> tuple:
    """Bloqueo pre-generación: si el presupuesto 'custom' declarado es insuficiente para las
    metas, retorna (False, detail) con los números para el mensaje accionable. Solo aplica a
    budget='custom' con monto explícito (las opciones categóricas son cualitativas). Fail-open:
    ante cualquier error, NO bloquea (mejor generar que romper el flujo)."""
    try:
        if not _budget_floor_enabled():
            return True, None
        if str(form_data.get("budget") or "").lower() != "custom":
            return True, None
        try:
            declared = float(str(form_data.get("budgetAmount")).replace(",", "").strip())
        except (TypeError, ValueError):
            declared = 0.0
        if declared <= 0:
            # custom sin monto válido: build_budget_context cae a 'medium', no es nuestro bloqueo.
            return True, None
        currency = str(form_data.get("budgetCurrency") or "DOP").upper()
        usd_dop = _budget_usd_to_dop()
        declared_dop = declared * usd_dop if currency == "USD" else declared
        info = min_budget_for_goals(form_data)
        threshold = info["min_budget_dop"] * (1.0 - _budget_floor_tolerance_pct())
        if declared_dop >= threshold:
            return True, None
        min_in_currency = (info["min_budget_dop"] / usd_dop) if currency == "USD" else info["min_budget_dop"]
        sym = "US$" if currency == "USD" else "RD$"
        msg = (
            f"Tu presupuesto de {sym}{round(declared):,} es insuficiente para tus metas "
            f"({info['target_calories']} kcal/día × {info['days']} días"
            + (f" × {info['household']} personas" if info["household"] > 1 else "")
            + f"). El mínimo para un plan profesional es ~{sym}{round(min_in_currency):,}. "
            "Sube tu presupuesto o ajusta tus metas (menos días, menos personas, o una meta "
            "calórica menor). No bajamos la calidad nutricional para encajar en un presupuesto "
            "demasiado bajo."
        )
        return False, {
            "error_code": "budget_below_goal_floor",
            "min_budget": round(min_in_currency),
            "declared": round(declared),
            "currency": currency,
            "days": info["days"],
            "household": info["household"],
            "target_calories": info["target_calories"],
            "message": msg,
        }
    except Exception as e:
        logger.warning(f"[P2-BUDGET-FLOOR] validate_budget_sufficient falló ({type(e).__name__}: {e}); fail-open")
        return True, None
