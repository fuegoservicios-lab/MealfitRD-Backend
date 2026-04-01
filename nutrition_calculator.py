# backend/nutrition_calculator.py
"""
Agente Calculador: Cálculos nutricionales exactos con la ecuación de Mifflin-St Jeor.
Elimina la carga matemática del LLM para evitar alucinaciones numéricas.
"""

def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> int:
    """
    Calcula el BMR (Tasa Metabólica Basal) usando la ecuación de Mifflin-St Jeor.
    
    Hombres: BMR = 10×peso(kg) + 6.25×altura(cm) − 5×edad + 5
    Mujeres: BMR = 10×peso(kg) + 6.25×altura(cm) − 5×edad − 161
    """
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
    """Calcula el TDEE (Gasto Energético Total Diario) = BMR × multiplicador de actividad."""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)  # Default: moderado
    return int(round(bmr * multiplier))


def apply_goal_adjustment(tdee: float, goal: str) -> int:
    """Aplica el ajuste calórico según el objetivo (déficit/superávit)."""
    adjustment = GOAL_ADJUSTMENTS.get(goal, 0.0)
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
    
    weight_unit = form_data.get("weightUnit", "lb")  # 'lb' o 'kg'
    
    # Convertir a kilogramos si está en libras
    if weight_unit == "kg":
        weight = weight_raw
        weight_display = f"{weight_raw}kg"
    else:
        weight = round(float(weight_raw / 2.20462), 1)
        weight_display = f"{weight_raw}lbs → {weight}kg"
    
    gender = form_data.get("gender", "male")
    activity_level = form_data.get("activityLevel", "moderate")
    goal = form_data.get("mainGoal") or form_data.get("goal") or "maintenance"
    skip_lunch = form_data.get("skipLunch", False)
    
    # 1. BMR (Tasa Metabólica Basal)
    bmr = calculate_bmr(weight, height, age, gender)
    
    # 2. TDEE (Gasto Energético Total)
    tdee = calculate_tdee(bmr, activity_level)
    
    # 3. Calorías objetivo (con ajuste por meta)
    target_calories = apply_goal_adjustment(tdee, goal)
    
    calculation_details_str = (
        f"BMR (Mifflin-St Jeor): {bmr} kcal | "
        f"TDEE ({activity_level}, ×{ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)}): {tdee} kcal | "
        f"Objetivo ({goal}): {target_calories} kcal"
    )

    # Preservar el cálculo original para el Dashboard
    original_target_calories = target_calories
    original_macros = calculate_macros(original_target_calories, goal)

    # Ajuste por 'Almuerzo Familiar / Ya resuelto'
    if skip_lunch:
        # Reservar ~35% de las calorías para el almuerzo que el usuario comerá por su cuenta
        reserved_cals = int(round(target_calories * 0.35 / 50) * 50)
        target_calories = target_calories - reserved_cals
        calculation_details_str += f" | ⚠️ skipLunch ACTIVO: Se reservaron {reserved_cals} kcal para el Almuerzo Familiar. Calorías restantes asignadas a la IA: {target_calories} kcal."
    
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
        "calculation_details": calculation_details_str
    }
    
    print(f"\n🔢 [CALCULADORA NUTRICIONAL] Resultados exactos:")
    print(f"   📊 BMR: {bmr} kcal (Peso: {weight_display}, Altura: {height}cm, Edad: {age}, Género: {gender})")
    print(f"   🏃 TDEE: {tdee} kcal (Actividad: {activity_level})")
    print(f"   🎯 Calorías Objetivo: {target_calories} kcal ({goal_labels.get(goal, goal)})")
    print(f"   🥩 Proteína: {macros['protein_g']}g | 🍚 Carbos: {macros['carbs_g']}g | 🥑 Grasas: {macros['fats_g']}g")
    
    return result
