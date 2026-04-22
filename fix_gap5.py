import sys

with open('backend/cron_tasks.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Change 1: Add the guard for len(consumed_records) < 3
target_1 = """        if expected_meals_count > 0:
            # 1. Calcular adherencia granular del ciclo actual"""

replacement_1 = """        if not consumed_records or len(consumed_records) < 3:
            logger.info(f"📊 [FEEDBACK LOOP] Insuficientes datos reales ({len(consumed_records) if consumed_records else 0} < 3). Omitiendo _meal_level_adherence para evitar falsos positivos EMA.")
        elif expected_meals_count > 0:
            # 1. Calcular adherencia granular del ciclo actual"""

if target_1 in content:
    content = content.replace(target_1, replacement_1)
    print("Change 1 applied successfully.")
else:
    print("Error: Target 1 not found.")

# Change 2: Delay chunk 2 by changing offset - 2 to offset - 1
target_2 = """    days_offset_int = max(0, int(days_offset))
    delay_days = max(0, days_offset_int - 2)"""

replacement_2 = """    days_offset_int = max(0, int(days_offset))
    # [GAP 5] Delay = max(0, days_offset - 1): el chunk se genera 1 día antes para recabar más adherencia.
    delay_days = max(0, days_offset_int - 1)"""

if target_2 in content:
    content = content.replace(target_2, replacement_2)
    print("Change 2 applied successfully.")
else:
    print("Error: Target 2 not found.")

# Update the docstring in _enqueue_plan_chunk as well
target_3 = "Delay = max(0, days_offset - 2): el chunk se genera 2"
replacement_3 = "Delay = max(0, days_offset - 1): el chunk se genera 1"
if target_3 in content:
    content = content.replace(target_3, replacement_3)
    print("Change 3 (docstring) applied successfully.")
else:
    print("Target 3 (docstring) not found, continuing anyway.")

with open('backend/cron_tasks.py', 'w', encoding='utf-8') as f:
    f.write(content)
