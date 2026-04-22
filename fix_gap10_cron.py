import re

with open('backend/cron_tasks.py', 'r', encoding='utf-8') as f:
    content = f.read()

target_pattern = r'(logger\.error\(f"â \x8c \[CRON\] Error resetting totalDays for \{user_id\}: \{e\}"\)\n\s+)(# \[GAP 4 IMPLEMENTATION\]: Pre-cargar el emergency backup cache por si la regeneraciÃ³n IA de hoy falla\n\s+backup_plan = health_profile\.get\(\'emergency_backup_plan\', \[\]\))'

# Let's try matching a broader substring since unicode can be tricky
def replace_chunk(s):
    lines = s.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "Error resetting totalDays for {user_id}: {e}" in line:
            out.append(line)
            # Inject GAP 10 right after
            inject = """                
            # [GAP 10] Guardar resumen de adherencia cruzada para el próximo plan
            try:
                from db_core import execute_sql_write
                succ_techs, aban_techs, freq_meals = calculate_meal_success_scores(user_id, days_back=max(14, total_days_requested))
                if freq_meals:
                    import json
                    execute_sql_write("UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{previous_plan_frequent_meals}', %s::jsonb) WHERE id = %s", (json.dumps(freq_meals), user_id))
                    logger.info(f"🔄 [GAP 10] Resumen de comidas frecuentes guardado para seed del próximo plan de {user_id}")
            except Exception as gap10_e:
                logger.warning(f"⚠️ [GAP 10] Error calculando resumen cruzado al expirar: {gap10_e}")"""
            out.extend(inject.split('\n'))
            i += 1
            continue
        out.append(line)
        i += 1
    return '\n'.join(out)

c_new = replace_chunk(content)
if content != c_new:
    print("Change 1 (cron_tasks.py) applied successfully via lines split.")
    with open('backend/cron_tasks.py', 'w', encoding='utf-8') as f:
        f.write(c_new)
else:
    print("Error: Target 1 not found via lines split.")
