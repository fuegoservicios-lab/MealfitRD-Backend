import sys

with open('backend/cron_tasks.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """    # MEJORA 3: Aprendizaje de Patrones de Éxito y Temporalidad
    try:
        succ_techs, aban_techs, freq_meals = calculate_meal_success_scores(user_id, days_back=14)
        day_adherence = calculate_day_of_week_adherence(user_id, days_back=30)"""

replacement = """    # MEJORA 3: Aprendizaje de Patrones de Éxito y Temporalidad
    try:
        # [GAP 7] Ventana dinámica de aprendizaje basada en el offset del plan
        days_offset = int(pipeline_data.get('_days_offset', 0))
        dynamic_days_back = max(14, days_offset + 7)
        
        succ_techs, aban_techs, freq_meals = calculate_meal_success_scores(user_id, days_back=dynamic_days_back)
        day_adherence = calculate_day_of_week_adherence(user_id, days_back=max(30, dynamic_days_back))"""

if target in content:
    content = content.replace(target, replacement)
    print("Change applied successfully.")
else:
    print("Error: Target not found.")

with open('backend/cron_tasks.py', 'w', encoding='utf-8') as f:
    f.write(content)
