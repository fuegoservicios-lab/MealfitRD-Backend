import json
import sys
from shopping_calculator import get_shopping_list_delta
from db_core import supabase

def main():
    user_id = "ed4e4554-1da8-451c-9758-1676ab15b889"
    res = supabase.table("meal_plans").select("plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
    if not res.data:
        print("No plan found")
        return
        
    plan_data = res.data[0]["plan_data"]
    
    with open("test_output.txt", "w", encoding="utf-8") as f:
        scaled_4 = get_shopping_list_delta(user_id, plan_data, is_new_plan=True, structured=True, multiplier=4.0)
        scaled_6 = get_shopping_list_delta(user_id, plan_data, is_new_plan=True, structured=True, multiplier=6.0)
        
        f.write("=== MULTIPLIER 4 ===\n")
        f.write(json.dumps([x for x in scaled_4 if "pavo" in str(x).lower()], indent=2) + "\n")
        f.write("=== MULTIPLIER 6 ===\n")
        f.write(json.dumps([x for x in scaled_6 if "pavo" in str(x).lower()], indent=2) + "\n")
        
        # Test also what is currently inside plan_data
        f.write("=== RAW DB AGGREGATED WEEKLY ===\n")
        f.write(json.dumps([x for x in plan_data.get("aggregated_shopping_list_weekly", []) if "pavo" in str(x).lower()], indent=2) + "\n")

if __name__ == "__main__":
    main()
