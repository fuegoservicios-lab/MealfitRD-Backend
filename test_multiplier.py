import json
from db_plans import get_latest_meal_plan_with_id
from shopping_calculator import get_shopping_list_delta

plan = get_latest_meal_plan_with_id("ed4e4554-1da8-451c-9758-1676ab15b889")
if plan:
    plan_data = plan["plan_data"]
    list_1 = get_shopping_list_delta("guest", plan_data, is_new_plan=True, structured=True, multiplier=1.0)
    list_3 = get_shopping_list_delta("guest", plan_data, is_new_plan=True, structured=True, multiplier=3.0)
    
    # filter for yuca
    yuca_1 = [i for i in list_1 if "yuca" in i["display_name"].lower()]
    yuca_3 = [i for i in list_3 if "yuca" in i["display_name"].lower()]
    
    print("--- MULTIPLIER 1 ---")
    for y in yuca_1:
        print(f"Name: {y['display_name']}, Qty: {y['display_qty']}")
        
    print("--- MULTIPLIER 3 ---")
    for y in yuca_3:
        print(f"Name: {y['display_name']}, Qty: {y['display_qty']}")
else:
    print("No plan found")
