import sys
import json
from db_core import supabase

def main():
    if not supabase:
        print("No Supabase connection")
        sys.exit(1)
        
    res = supabase.table("meal_plans").select("id, plan_data").order("created_at", desc=True).limit(1).execute()
    if not res.data:
        print("No paths")
        return
        
    plan = res.data[0]["plan_data"]
    
    meals_found = set()
    first_few_ings = []
    
    for day in plan.get("days", []):
        for meal in day.get("meals", []):
            m_name = meal.get("meal", "NOT_FOUND")
            meals_found.add(m_name)
            for ing in meal.get("ingredients", []):
                if len(first_few_ings) < 5:
                    first_few_ings.append({"raw": ing, "meal": m_name})
                    
    print(f"Meals found in latest plan: {meals_found}")
    print(f"First few ingredients: {json.dumps(first_few_ings, indent=2)}")

if __name__ == "__main__":
    main()
