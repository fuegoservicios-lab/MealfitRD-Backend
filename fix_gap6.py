import re

with open('backend/cron_tasks.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Change 1: Insert Probe LLM block before the Smart Shuffle logger
target_pattern_1 = r'(is_degraded = snap\.get\("_degraded", False\)\s+if is_degraded:\s+)(logger\.warning\(f"[^"]*\[CHUNK DEGRADED\] Generando chunk \{week_number\} en modo degraded)'

replacement_1 = r'''\1# [GAP 6 FIX: Probe LLM para auto-recovery]
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    import os, json
                    from datetime import datetime, timezone
                    
                    # Evitar flapping: revisar si hicimos downgrade hace menos de 10 minutos
                    user_res_flap = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_all=False)
                    hp_flap = user_res_flap.get("health_profile", {}) if user_res_flap else {}
                    last_downgrade = hp_flap.get('_last_downgrade_time')
                    can_probe = True
                    
                    if last_downgrade:
                        from constants import safe_fromisoformat
                        ld_dt = safe_fromisoformat(last_downgrade)
                        if (datetime.now(timezone.utc) - ld_dt).total_seconds() < 600:
                            can_probe = False
                            logger.info(f"⏳ [GAP 6] Downgrade reciente ({last_downgrade}), saltando probe para evitar flapping.")

                    if can_probe:
                        logger.info(f"🔍 [GAP 6] Iniciando Probe LLM para auto-recovery del chunk {week_number}...")
                        probe_llm = ChatGoogleGenerativeAI(
                            model="gemini-3.1-flash-lite-preview",
                            temperature=0.0,
                            google_api_key=os.environ.get("GEMINI_API_KEY"),
                            max_retries=0
                        )
                        import concurrent.futures as _cf
                        def _do_probe():
                            return probe_llm.invoke("ping")
                        
                        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                            ex.submit(_do_probe).result(timeout=10)
                            
                        logger.info(f"🟢 [GAP 6] LLM Probe exitoso. Sistema estabilizado, restaurando a modo AI.")
                        is_degraded = False
                        snap.pop('_degraded', None)
                        
                        # Actualizar en BD para el actual y todos los futuros chunks
                        execute_sql_write("UPDATE plan_chunk_queue SET pipeline_snapshot = pipeline_snapshot - '_degraded' WHERE meal_plan_id = %s", (meal_plan_id,))
                        
                        # Limpiar historial de downgrade
                        execute_sql_write("UPDATE user_profiles SET health_profile = health_profile - '_last_downgrade_time' WHERE id = %s", (user_id,))
                        
                except Exception as probe_e:
                    logger.warning(f"⚠️ [CHUNK DEGRADED] Probe LLM falló o no pudo ejecutar ({probe_e}). Modo Smart Shuffle activo.")
            
            if is_degraded:
                \2'''

content_new, count_1 = re.subn(target_pattern_1, replacement_1, content, count=1)
if count_1 > 0:
    print("Change 1 (Probe LLM) applied successfully.")
else:
    print("Error: Target 1 not found.")

# Change 2: Inject _last_downgrade_time after setting _degraded
target_pattern_2 = r'(\s+WHERE meal_plan_id = %s AND status IN \(\'pending\', \'failed\'\)\n\s+""",\s*\(meal_plan_id,\)\n\s*\))'
replacement_2 = r'''\1
                        
                        # [GAP 6] Guardar timestamp del downgrade para evitar flapping
                        from datetime import datetime, timezone
                        execute_sql_write(
                            """
                            UPDATE user_profiles 
                            SET health_profile = jsonb_set(
                                COALESCE(health_profile, '{}'::jsonb), 
                                '{_last_downgrade_time}', 
                                %s::jsonb
                            ) WHERE id = %s
                            """,
                            (f'"{datetime.now(timezone.utc).isoformat()}"', str(user_id))
                        )'''

content_final, count_2 = re.subn(target_pattern_2, replacement_2, content_new, count=1)
if count_2 > 0:
    print("Change 2 (Downgrade timestamp) applied successfully.")
else:
    print("Error: Target 2 not found.")

with open('backend/cron_tasks.py', 'w', encoding='utf-8') as f:
    f.write(content_final)
