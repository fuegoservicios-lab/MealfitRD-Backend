import sys
with open('backend/cron_tasks.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''                                            # Deferred / limit reached
                                            if not needs_swap:
                                                shifted_days[d_idx]['meals'][m_idx]['_needs_swap'] = True
                                                s                        # --- [GAP 2 FIX: Recalcular Macros y Lista de Compras] ---
                        total_cals = 0'''

replacement = '''                                            # Deferred / limit reached
                                            if not needs_swap:
                                                shifted_days[d_idx]['meals'][m_idx]['_needs_swap'] = True
                                                shifted_days[d_idx]['meals'][m_idx]['_swap_reason'] = matched_trigger
                                                modified = True
                                                if in_critical_range:
                                                    logger.info(f"⚡ [CRON] JIT Swap limit reached ({severity}). Deferred meal '{m_name}'.")
                                                else:
                                                    logger.info(f"🍽️ [CRON] Día {d_idx+1} meal '{m_name}' flagged for deferred swap (trigger: '{matched_trigger}')")
                        
                        # --- [GAP 2 FIX: Recalcular Macros y Lista de Compras] ---
                        total_cals = 0'''

if target in content:
    new_content = content.replace(target, replacement)
    with open('backend/cron_tasks.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Fixed successfully')
else:
    print('Target not found, trying fallback')
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 's                        # --- [GAP 2 FIX: Recalcular Macros y Lista de Compras] ---' in line:
            lines[i] = '                        # --- [GAP 2 FIX: Recalcular Macros y Lista de Compras] ---'
            lines.insert(i, "                                                shifted_days[d_idx]['meals'][m_idx]['_swap_reason'] = matched_trigger\n                                                modified = True\n                                                if in_critical_range:\n                                                    logger.info(f\"⚡ [CRON] JIT Swap limit reached ({severity}). Deferred meal '{m_name}'.\")\n                                                else:\n                                                    logger.info(f\"🍽️ [CRON] Día {d_idx+1} meal '{m_name}' flagged for deferred swap (trigger: '{matched_trigger}')\")")
            with open('backend/cron_tasks.py', 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print('Fixed via fallback')
            break
