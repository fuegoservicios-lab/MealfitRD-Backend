import json
from db_core import supabase
from agent import _pre_consolidate_ingredients_multiday

def main():
    if not supabase:
        print("No Supabase connection")
        return
        
    res = supabase.table("meal_plans").select("id, plan_data").order("created_at", desc=True).limit(1).execute()
    if not res.data:
        print("No plans")
        return
        
    plan = res.data[0]["plan_data"]
    
    ingredients = []
    days = plan.get("days", [])
    print(f"Number of options (days array): {len(days)}")
    for day_data in days:
        meals = day_data.get("meals", [])
        print(f"Option has {len(meals)} meals")
        for m in meals:
            ing = m.get("ingredients", [])
            meal_name = m.get("name", m.get("meal", "Despensa General")) # <====== LOOK AT THIS
            meal_only = m.get("meal", "Despensa General")
            if ing:
                for idx in ing:
                    ingredients.append({
                        "raw": idx,
                        "meal_slot": meal_only,
                        "meal_slot_with_name": meal_name
                    })
                    
    print(f"\nExtracted {len(ingredients)} ingredients. Sample:")
    for i in ingredients[:3]:
        print(i)
        
    if ingredients:
        json_output = _pre_consolidate_ingredients_multiday(ingredients)
        print(f"\nPre-consolidated output sample: {json.dumps(json_output[:3], indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    main()
