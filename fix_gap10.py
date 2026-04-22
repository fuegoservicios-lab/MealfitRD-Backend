import sys

with open('backend/cron_tasks.py', 'r', encoding='utf-8') as f:
    content = f.read()

target_cron = """            # [GAP 4 IMPLEMENTATION]: Pre-cargar el emergency backup cache por si la regeneración IA de hoy falla
            backup_plan = health_profile.get('emergency_backup_plan', [])"""

replacement_cron = """            # [GAP 10] Guardar resumen de adherencia cruzada para el próximo plan
            try:
                from db_core import execute_sql_write
                succ_techs, aban_techs, freq_meals = calculate_meal_success_scores(user_id, days_back=max(14, total_days_requested))
                if freq_meals:
                    import json
                    execute_sql_write("UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{previous_plan_frequent_meals}', %s::jsonb) WHERE id = %s", (json.dumps(freq_meals), user_id))
                    logger.info(f"🔄 [GAP 10] Resumen de comidas frecuentes guardado para seed del próximo plan de {user_id}")
            except Exception as gap10_e:
                logger.warning(f"⚠️ [GAP 10] Error calculando resumen cruzado al expirar: {gap10_e}")

            # [GAP 4 IMPLEMENTATION]: Pre-cargar el emergency backup cache por si la regeneración IA de hoy falla
            backup_plan = health_profile.get('emergency_backup_plan', [])"""

if target_cron in content:
    content = content.replace(target_cron, replacement_cron)
    print("Change 1 (cron_tasks.py) applied successfully.")
else:
    print("Error: Target 1 not found in cron_tasks.py.")

with open('backend/cron_tasks.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Now, modify routers/plans.py
with open('backend/routers/plans.py', 'r', encoding='utf-8') as f:
    content_plans = f.read()

target_plans_1 = """        if actual_user_id:
            likes = get_user_likes(actual_user_id)"""

replacement_plans_1 = """        if actual_user_id:
            likes = get_user_likes(actual_user_id)
            
            # [GAP 10] Leer previous_plan_frequent_meals para seed cold-start
            try:
                from backend.db_core import execute_sql_query
                profile_row = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (actual_user_id,), fetch_all=False)
                if profile_row and profile_row.get("health_profile"):
                    hp = profile_row["health_profile"]
                    if hp.get("previous_plan_frequent_meals"):
                        pipeline_data["frequent_meals"] = hp["previous_plan_frequent_meals"]
            except Exception as e:
                import logging
                logging.warning(f"Error cargando previous_plan_frequent_meals: {e}")"""

# Apply to api_analyze
if target_plans_1 in content_plans:
    # Use replace to replace both occurrences (in api_analyze and api_analyze_stream)
    content_plans = content_plans.replace(target_plans_1, replacement_plans_1)
    print("Change 2 (routers/plans.py) applied successfully.")
else:
    print("Error: Target 2 not found in routers/plans.py.")

with open('backend/routers/plans.py', 'w', encoding='utf-8') as f:
    f.write(content_plans)
