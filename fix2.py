import sys
import re

with open('backend/cron_tasks.py', 'r', encoding='utf-8') as f:
    content = f.read()

# I will find the exact block using regex
start_marker = "                        # --- [GAP 2 FIX: Recalcular Macros y Lista de Compras] ---"
end_marker = "                        except Exception as tdee_e:"

if start_marker in content and end_marker in content:
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker) + len(end_marker)
    
    correct_block = """                        # --- [GAP 2 FIX: Recalcular Macros y Lista de Compras] ---
                        total_cals = 0
                        total_p, total_c, total_f = 0, 0, 0
                        valid_days = len(shifted_days)
                        
                        if valid_days > 0:
                            for d in shifted_days:
                                for m in d.get('meals', []):
                                    total_cals += m.get('calories', 0)
                                    total_p += m.get('protein', 0)
                                    total_c += m.get('carbs', 0)
                                    total_f += m.get('fats', 0)
                            
                            avg_cals = int(total_cals / valid_days)
                            avg_p = int(total_p / valid_days)
                            avg_c = int(total_c / valid_days)
                            avg_f = int(total_f / valid_days)
                        else:
                            avg_cals, avg_p, avg_c, avg_f = 0, 0, 0, 0
                            
                        # --- [GAP 3: TDEE Drift Detection] ---
                        # Si el usuario cambió de peso significativamente, escalar las calorías de los días restantes
                        force_chunk_invalidation = False
                        current_tdee_target = None
                        current_tdee_val = None
                        try:
                            from nutrition_calculator import calculate_nutrition
                            
                            # Recalcular TDEE con datos actuales del health_profile
                            current_nutrition = calculate_nutrition(health_profile)
                            current_target = current_nutrition.get('total_daily_calories', 0) or current_nutrition.get('target_calories', 0)
                            current_tdee_target = current_target
                            current_tdee_val = current_nutrition.get('tdee')
                            
                            # Comparar con las calorías promedio de los días restantes
                            if current_target and avg_cals:
                                drift_pct = abs(current_target - avg_cals) / max(avg_cals, 1)
                                
                                if drift_pct > 0.08:  # >8% de diferencia
                                    force_chunk_invalidation = True
                                    scale_factor = current_target / max(avg_cals, 1)
                                    logger.info(f"⚖️ [TDEE DRIFT] Detected {drift_pct*100:.1f}% calorie drift for {user_id}. Scaling {avg_cals}→{current_target} kcal (factor: {scale_factor:.2f})")
                                    
                                    for d in shifted_days:
                                        for m in d.get('meals', []):
                                            m['cals'] = int(m.get('cals', m.get('calories', 0)) * scale_factor)
                                            m['calories'] = m['cals']
                                            m['protein'] = int(m.get('protein', 0) * scale_factor)
                                            m['carbs'] = int(m.get('carbs', 0) * scale_factor)
                                            m['fats'] = int(m.get('fats', 0) * scale_factor)
                                    
                                    # Recalcular promedios con los nuevos valores
                                    avg_cals = current_target
                                    avg_p = int(sum(m.get('protein',0) for d in shifted_days for m in d.get('meals',[])) / max(valid_days,1))
                                    avg_c = int(sum(m.get('carbs',0) for d in shifted_days for m in d.get('meals',[])) / max(valid_days,1))
                                    avg_f = int(sum(m.get('fats',0) for d in shifted_days for m in d.get('meals',[])) / max(valid_days,1))
                        except Exception as tdee_e:"""
    
    new_content = content[:start_idx] + correct_block + content[end_idx:]
    with open('backend/cron_tasks.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Fix successfully applied!")
else:
    print("Markers not found.")
