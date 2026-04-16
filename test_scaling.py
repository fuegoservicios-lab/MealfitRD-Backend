import json
from shopping_calculator import get_shopping_list_delta

plan_result_mock = {
    "days": [
        {
            "day": 1,
            "meals": [
                {"ingredients": ["0.4 lbs de Pechuga de pavo", "0.5 lbs de Yuca"]}
            ]
        },
        {
            "day": 2,
            "meals": [
                {"ingredients": ["0.4 lbs de Pechuga de pavo", "0.5 lbs de Yuca"]}
            ]
        },
        {
            "day": 3,
            "meals": [
                {"ingredients": ["0.4 lbs de Pechuga de pavo", "0.5 lbs de Yuca"]}
            ]
        }
    ]
}

def print_scaled(mult):
    res = get_shopping_list_delta("test", plan_result_mock, is_new_plan=True, structured=True, multiplier=float(mult))
    print(f"\n--- MULTIPLIER {mult} ---")
    for r in res:
        print(f"{r['name']}: {r['display_qty']}")

print_scaled(3)
print_scaled(4)
print_scaled(5)
