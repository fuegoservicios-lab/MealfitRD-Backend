import json

ingredients = [
    {"raw": "2 plátanos maduros", "meal_slot": "Desayuno"},
    {"raw": "3 huevos", "meal_slot": "Desayuno"},
    {"raw": "1 libra de pollo", "meal_slot": "Almuerzo"},
    {"raw": "2 plátanos maduros", "meal_slot": "Merienda"}
]

def _pre_consolidate_ingredients_multiday(ingredients_list, base_days=3):
    import re, unicodedata
    def _normalize(text: str) -> str:
        if not text: return ""
        nfkd = unicodedata.normalize('NFKD', text.lower().strip())
        return re.sub(r'\s+', ' ', ''.join(c for c in nfkd if not unicodedata.combining(c)))

    from constants import parse_ingredient_qty

    groups = {}
    order = []
    
    for item in ingredients_list:
        ing_str = item.get("raw", "") if isinstance(item, dict) else item
        meal_slot = item.get("meal_slot", "Despensa General") if isinstance(item, dict) else "Despensa General"
        
        if not isinstance(ing_str, str) or not ing_str.strip():
            continue
            
        num, unit, name = parse_ingredient_qty(ing_str, to_metric=False)
        key = (_normalize(name), _normalize(unit))
        
        if key not in groups:
            groups[key] = {"total": num, "unit": unit, "name": name, "raw": ing_str, "meal_slots": [meal_slot], "can_sum": num is not None}
            order.append(key)
        else:
            entry = groups[key]
            if meal_slot not in entry["meal_slots"]:
                entry["meal_slots"].append(meal_slot)
                
            if entry["can_sum"] and num is not None:
                entry["total"] = (entry["total"] or 0) + num
            else:
                entry["can_sum"] = False
    
    result = []
    
    def format_qty(qty):
        return str(int(qty)) if qty == int(qty) else f"{qty:.2f}"
            
    for key in order:
        entry = groups[key]
        slot = entry["meal_slots"][0] if entry["meal_slots"] else "Despensa General"
        
        if entry["can_sum"] and entry["total"] is not None:
            base_total = entry["total"]
            div = base_days if base_days > 0 else 1
            
            t7 = base_total * (7 / div)
            t15 = base_total * (15 / div)
            t30 = base_total * (30 / div)
            
            unit_part = f" {entry['unit']}" if entry['unit'] else ""
            
            result.append({
                "name": entry['name'],
                "meal_slot": slot,
                "raw_qty_7_days": f"{format_qty(t7)}{unit_part}".strip(),
                "raw_qty_15_days": f"{format_qty(t15)}{unit_part}".strip(),
                "raw_qty_30_days": f"{format_qty(t30)}{unit_part}".strip()
            })
        else:
            result.append({
                "name": entry["raw"],
                "meal_slot": slot,
                "raw_qty_7_days": "Al gusto / Variable",
                "raw_qty_15_days": "Al gusto / Variable",
                "raw_qty_30_days": "Al gusto / Variable"
            })
    
    return result

print(json.dumps(_pre_consolidate_ingredients_multiday(ingredients), indent=2, ensure_ascii=False))
