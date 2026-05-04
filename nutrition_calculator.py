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
    skip_lunch = form_data.get("skipLunch", False)
    
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
            print(f"Error en metabolismo evolutivo: {e}")
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
        "calculation_details": calculation_details_str,
        "kinematics": velocity_data if 'velocity_data' in locals() else None
    }
    
    print(f"\n🔢 [CALCULADORA NUTRICIONAL] Resultados exactos:")
    print(f"   📊 BMR: {bmr} kcal (Peso: {weight_display}, Altura: {height}cm, Edad: {age}, Género: {gender})")
    print(f"   🏃 TDEE: {tdee} kcal (Actividad: {activity_level})")
    print(f"   🎯 Calorías Objetivo: {target_calories} kcal ({goal_labels.get(goal, goal)})")
    print(f"   🥩 Proteína: {macros['protein_g']}g | 🍚 Carbos: {macros['carbs_g']}g | 🥑 Grasas: {macros['fats_g']}g")
    
    return result
