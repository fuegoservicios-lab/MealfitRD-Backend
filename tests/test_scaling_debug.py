import json
import asyncio
from shopping_calculator import get_shopping_list_delta

def run():
    from db_plans import get_latest_meal_plan_with_id
    from shopping_calculator import _parse_quantity
    
    user_id = "ed4e4554-1da8-451c-9758-1676ab15b889"
    plan_record = get_latest_meal_plan_with_id(user_id)
    plan_data = plan_record["plan_data"]
    
    print("--- RAW PLAN DATA (meal 1) ---")
    d1 = plan_data.get('days', [])[0] if plan_data.get('days') else {}
    m1 = d1.get('meals', [])[0] if d1.get('meals') else {}
    print("Example ingredients original:", m1.get("ingredients", []))
    
    for h_size in [1, 2, 3, 4, 5, 6]:
        print(f"\n--- SCALED 7 DAYS FOR {h_size} PERSON ---")
        scaled_7 = get_shopping_list_delta(user_id, plan_data, is_new_plan=True, structured=True, multiplier=float(h_size))
        
        for item in scaled_7:
            name = item.get("name", "").lower()
            if "pechuga" in name or "aceite" in name or "pollo" in name:
                print(f"[{h_size} personas] {item.get('display_string')}")

if __name__ == "__main__":
    run()
