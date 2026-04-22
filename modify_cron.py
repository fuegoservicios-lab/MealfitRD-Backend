import sys

with open('cron_tasks.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "Error procesando chunk {week_number}" in l and "logger.error" in l:
        lines[i] = '            import traceback; tb_str = traceback.format_exc(); logger.error(f"❌ [CHUNK] Error procesando chunk {week_number} para plan {meal_plan_id}: {e}\\n{tb_str}")\n'

with open('cron_tasks.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
