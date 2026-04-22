import sys

with open('backend/cron_tasks.py', 'r', encoding='utf-8') as f:
    content = f.read()

target_2 = """                        # 2. Rescatar este chunk y los futuros en degraded mode
                        execute_sql_write(\"\"\"
                            UPDATE plan_chunk_queue 
                            SET status = 'pending', 
                                attempts = 0,
                                pipeline_snapshot = jsonb_set(pipeline_snapshot, '{_degraded}', 'true'::jsonb),
                                updated_at = NOW() 
                            WHERE meal_plan_id = %s AND status IN ('pending', 'failed')
                        \"\"\", (meal_plan_id,))"""

replacement_2 = """                        # 2. Rescatar este chunk y los futuros en degraded mode
                        execute_sql_write(\"\"\"
                            UPDATE plan_chunk_queue 
                            SET status = 'pending', 
                                attempts = 0,
                                pipeline_snapshot = jsonb_set(pipeline_snapshot, '{_degraded}', 'true'::jsonb),
                                updated_at = NOW() 
                            WHERE meal_plan_id = %s AND status IN ('pending', 'failed')
                        \"\"\", (meal_plan_id,))
                        
                        # [GAP 6] Guardar timestamp del downgrade para evitar flapping
                        from datetime import datetime, timezone
                        execute_sql_write(
                            \"\"\"
                            UPDATE user_profiles 
                            SET health_profile = jsonb_set(
                                COALESCE(health_profile, '{}'::jsonb), 
                                '{_last_downgrade_time}', 
                                %s::jsonb
                            ) WHERE id = %s
                            \"\"\",
                            (f'"{datetime.now(timezone.utc).isoformat()}"', str(user_id))
                        )"""

if target_2 in content:
    content = content.replace(target_2, replacement_2)
    print("Change 2 (Downgrade timestamp) applied successfully.")
else:
    print("Error: Target 2 not found.")

with open('backend/cron_tasks.py', 'w', encoding='utf-8') as f:
    f.write(content)
