import re

with open('backend/cron_tasks.py', 'r', encoding='utf-8') as f:
    content = f.read()

target_pattern = r'(# MEJORA 3: Aprendizaje de Patrones de [^\n]+\n\s+try:\n\s+)(succ_techs, aban_techs, freq_meals = calculate_meal_success_scores\(user_id, days_back=14\)\n\s+day_adherence = calculate_day_of_week_adherence\(user_id, days_back=30\))'

replacement = r'''\1# [GAP 7] Ventana dinámica de aprendizaje basada en el offset del plan
        days_offset = int(pipeline_data.get('_days_offset', 0))
        dynamic_days_back = max(14, days_offset + 7)
        
        succ_techs, aban_techs, freq_meals = calculate_meal_success_scores(user_id, days_back=dynamic_days_back)
        day_adherence = calculate_day_of_week_adherence(user_id, days_back=max(30, dynamic_days_back))'''

content_new, count = re.subn(target_pattern, replacement, content, count=1)

if count > 0:
    print("Change applied successfully.")
    with open('backend/cron_tasks.py', 'w', encoding='utf-8') as f:
        f.write(content_new)
else:
    print("Error: Target not found.")
