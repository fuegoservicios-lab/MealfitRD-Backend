import re

with open('backend/cron_tasks.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Change 1: _persist_nightly_learning_signals
target_1 = r'''(        quality_score = calculate_plan_quality_score\(user_id, \{'days': days\}, consumed_records\)\n\s+)(quality_history = validated_profile\.quality_history\n\s+if not isinstance\(quality_history, list\):\n\s+quality_history = \[\]\n\s+quality_history\.append\(quality_score\)\n\s+quality_history = quality_history\[-5:\]\n\s+execute_sql_write\(\n\s+"UPDATE user_profiles SET health_profile = jsonb_set\(jsonb_set\(health_profile, '\{last_plan_quality\}', %s::jsonb\), '\{quality_history\}', %s::jsonb\) WHERE id = %s",\n\s+\(json\.dumps\(quality_score\), json\.dumps\(quality_history\), user_id\)\n\s+\))'''

replacement_1 = r'''\1# [GAP 8] Diferenciar fuente de rotaciones
        quality_history_rotations = getattr(validated_profile, 'quality_history_rotations', [])
        if not isinstance(quality_history_rotations, list):
            quality_history_rotations = []
            
        quality_history_rotations.append(quality_score)
        quality_history_rotations = quality_history_rotations[-5:]
        
        execute_sql_write(
            "UPDATE user_profiles SET health_profile = jsonb_set(jsonb_set(health_profile, '{last_plan_quality}', %s::jsonb), '{quality_history_rotations}', %s::jsonb) WHERE id = %s",
            (json.dumps(quality_score), json.dumps(quality_history_rotations), user_id)
        )'''

# Change 2: _inject_advanced_learning_signals
target_2 = r'''(        pipeline_data\['_previous_plan_quality'\] = quality_score\n\s+logger\.info\(f"[^"]*\[SELF-EVALUATION\] Calidad del Plan Anterior para \{user_id\}: \{quality_score:\.2f\}"\)\n\s+)(quality_history = health_profile\.get\('quality_history', \[\]\)\n\s+if not isinstance\(quality_history, list\):\n\s+quality_history = \[\]\n\s+quality_history\.append\(quality_score\)\n\s+quality_history = quality_history\[-5:\] # Mantener los últimos 5 ciclos\n\s+pipeline_data\['quality_history'\] = quality_history\n\s+# Analizar tendencias para inyectar un hint al LLM\n\s+if len\(quality_history\) >= 3:\n\s+last_3 = quality_history\[-3:\]\n\s+if all\(score < 0\.3 for score in last_3\):\n\s+pipeline_data\['_quality_hint'\] = 'drastic_change'\n\s+logger\.warning\(f"[^"]*\[FEEDBACK LOOP\] Quality Score muy bajo por 3 ciclos consecutivos\. Se activará un CAMBIO RADICAL en la estrategia\."\)\n\s+elif all\(score > 0\.8 for score in last_3\):\n\s+pipeline_data\['_quality_hint'\] = 'increase_complexity'\n\s+logger\.info\(f"[^"]*\[FEEDBACK LOOP\] Quality Score muy alto por 3 ciclos consecutivos\. Se permitirá MAYOR COMPLEJIDAD\."\)\n\s+# Detectar Plateau Silencioso \(GAP 6\)\n\s+if len\(quality_history\) >= 4 and not pipeline_data\.get\('_quality_hint'\):\n\s+mean_q = sum\(quality_history\) / len\(quality_history\)\n\s+variance = sum\(\(q - mean_q\)\*\*2 for q in quality_history\) / len\(quality_history\)\n\s+if variance < 0\.01 and mean_q < 0\.6:\n\s+pipeline_data\['_quality_hint'\] = 'break_plateau'\n\s+logger\.warning\(f"[^"]*\[FEEDBACK LOOP\] Plateau Silencioso detectado\. Quality score estancado en \{mean_q:\.2f\}\. Se forzará una ruptura de monotonía\."\)\n\s+# Guardamos el score y el historial en el health_profile\n\s+execute_sql_write\(\n\s+"UPDATE user_profiles SET health_profile = jsonb_set\(jsonb_set\(health_profile, '\{last_plan_quality\}', %s::jsonb\), '\{quality_history\}', %s::jsonb\) WHERE id = %s",\n\s+\(json\.dumps\(quality_score\), json\.dumps\(quality_history\), user_id\)\n\s+\))'''

replacement_2 = r'''\1# [GAP 8] Diferenciar fuente de chunks (para evitar saturar con shifts diarios)
        quality_history_chunks = health_profile.get('quality_history_chunks', [])
        if not isinstance(quality_history_chunks, list):
            quality_history_chunks = []
            
        quality_history_chunks.append(quality_score)
        quality_history_chunks = quality_history_chunks[-5:] # Mantener los últimos 5 ciclos de chunks
        
        pipeline_data['quality_history_chunks'] = quality_history_chunks
        
        # Analizar tendencias para inyectar un hint al LLM
        if len(quality_history_chunks) >= 3:
            last_3 = quality_history_chunks[-3:]
            if all(score < 0.3 for score in last_3):
                pipeline_data['_quality_hint'] = 'drastic_change'
                logger.warning(f"🚨 [FEEDBACK LOOP] Quality Score muy bajo por 3 ciclos consecutivos. Se activará un CAMBIO RADICAL en la estrategia.")
            elif all(score > 0.8 for score in last_3):
                pipeline_data['_quality_hint'] = 'increase_complexity'
                logger.info(f"🌟 [FEEDBACK LOOP] Quality Score muy alto por 3 ciclos consecutivos. Se permitirá MAYOR COMPLEJIDAD.")
        
        # Detectar Plateau Silencioso (GAP 6 / GAP 8)
        if len(quality_history_chunks) >= 4 and not pipeline_data.get('_quality_hint'):
            mean_q = sum(quality_history_chunks) / len(quality_history_chunks)
            variance = sum((q - mean_q)**2 for q in quality_history_chunks) / len(quality_history_chunks)
            if variance < 0.01 and mean_q < 0.6:
                pipeline_data['_quality_hint'] = 'break_plateau'
                logger.warning(f"⚠️ [FEEDBACK LOOP] Plateau Silencioso detectado. Quality score estancado en {mean_q:.2f}. Se forzará una ruptura de monotonía.")
        
        # Guardamos el score y el historial en el health_profile
        execute_sql_write(
            "UPDATE user_profiles SET health_profile = jsonb_set(jsonb_set(health_profile, '{last_plan_quality}', %s::jsonb), '{quality_history_chunks}', %s::jsonb) WHERE id = %s",
            (json.dumps(quality_score), json.dumps(quality_history_chunks), user_id)
        )'''

# Using string matching instead of regex for safety where possible
import string
def replace_chunk_1(s):
    look_for = """        quality_history = validated_profile.quality_history
        if not isinstance(quality_history, list):
            quality_history = []
            
        quality_history.append(quality_score)
        quality_history = quality_history[-5:]
        
        execute_sql_write(
            "UPDATE user_profiles SET health_profile = jsonb_set(jsonb_set(health_profile, '{last_plan_quality}', %s::jsonb), '{quality_history}', %s::jsonb) WHERE id = %s",
            (json.dumps(quality_score), json.dumps(quality_history), user_id)
        )"""
    
    replace_with = """        # [GAP 8] Diferenciar fuente de rotaciones
        quality_history_rotations = getattr(validated_profile, 'quality_history_rotations', [])
        if not isinstance(quality_history_rotations, list):
            quality_history_rotations = []
            
        quality_history_rotations.append(quality_score)
        quality_history_rotations = quality_history_rotations[-5:]
        
        execute_sql_write(
            "UPDATE user_profiles SET health_profile = jsonb_set(jsonb_set(health_profile, '{last_plan_quality}', %s::jsonb), '{quality_history_rotations}', %s::jsonb) WHERE id = %s",
            (json.dumps(quality_score), json.dumps(quality_history_rotations), user_id)
        )"""
    return s.replace(look_for, replace_with)

def replace_chunk_2(s):
    # This might have unicode emojis so regex or substring is tricky
    # Let's search by lines
    lines = s.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "quality_history = health_profile.get('quality_history', [])" in line:
            # Skip lines until execute_sql_write
            # Start injecting replacement
            inject = """        # [GAP 8] Diferenciar fuente de chunks (para evitar saturar con shifts diarios)
        quality_history_chunks = health_profile.get('quality_history_chunks', [])
        if not isinstance(quality_history_chunks, list):
            quality_history_chunks = []
            
        quality_history_chunks.append(quality_score)
        quality_history_chunks = quality_history_chunks[-5:] # Mantener los últimos 5 ciclos de chunks
        
        pipeline_data['quality_history_chunks'] = quality_history_chunks
        
        # Analizar tendencias para inyectar un hint al LLM
        if len(quality_history_chunks) >= 3:
            last_3 = quality_history_chunks[-3:]
            if all(score < 0.3 for score in last_3):
                pipeline_data['_quality_hint'] = 'drastic_change'
                logger.warning(f"🚨 [FEEDBACK LOOP] Quality Score muy bajo por 3 ciclos consecutivos. Se activará un CAMBIO RADICAL en la estrategia.")
            elif all(score > 0.8 for score in last_3):
                pipeline_data['_quality_hint'] = 'increase_complexity'
                logger.info(f"🌟 [FEEDBACK LOOP] Quality Score muy alto por 3 ciclos consecutivos. Se permitirá MAYOR COMPLEJIDAD.")
        
        # Detectar Plateau Silencioso (GAP 6 / 8)
        if len(quality_history_chunks) >= 4 and not pipeline_data.get('_quality_hint'):
            mean_q = sum(quality_history_chunks) / len(quality_history_chunks)
            variance = sum((q - mean_q)**2 for q in quality_history_chunks) / len(quality_history_chunks)
            if variance < 0.01 and mean_q < 0.6:
                pipeline_data['_quality_hint'] = 'break_plateau'
                logger.warning(f"⚠️ [FEEDBACK LOOP] Plateau Silencioso detectado. Quality score estancado en {mean_q:.2f}. Se forzará una ruptura de monotonía.")
        
        # Guardamos el score y el historial en el health_profile
        execute_sql_write(
            "UPDATE user_profiles SET health_profile = jsonb_set(jsonb_set(health_profile, '{last_plan_quality}', %s::jsonb), '{quality_history_chunks}', %s::jsonb) WHERE id = %s",
            (json.dumps(quality_score), json.dumps(quality_history_chunks), user_id)
        )"""
            out.extend(inject.split('\n'))
            
            # Skip until we consume the original block
            while i < len(lines):
                if "UPDATE user_profiles SET health_profile = jsonb_set" in lines[i]:
                    i += 2 # skip the query and the parameters
                    break
                i += 1
            i += 1
            continue
        out.append(line)
        i += 1
    return '\n'.join(out)

c1 = replace_chunk_1(content)
c2 = replace_chunk_2(c1)

with open('backend/cron_tasks.py', 'w', encoding='utf-8') as f:
    f.write(c2)
    
if content != c2:
    print("Changes applied via line replacements.")
else:
    print("No changes were made. Check targets.")
