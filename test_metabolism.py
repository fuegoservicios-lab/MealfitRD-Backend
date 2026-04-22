from nutrition_calculator import get_nutrition_targets
import json

# Simulated form data with stagnant weight history
form_data = {
    "weight": 180,
    "weightUnit": "lb",
    "height": 175,
    "age": 30,
    "gender": "male",
    "activityLevel": "moderate",
    "mainGoal": "lose_fat",
    "weight_history": [
        {"date": "2026-04-01", "weight": 180.5, "unit": "lb"},
        {"date": "2026-04-08", "weight": 180.0, "unit": "lb"},
        {"date": "2026-04-16", "weight": 180.2, "unit": "lb"}
    ]
}

print("=== TEST ESTANCAMIENTO (Debe bajar calorías extra) ===")
res = get_nutrition_targets(form_data)
print(res.get("calculation_details", ""))

print("\n=== TEST PÉRDIDA RÁPIDA (Debe subir calorías para proteger) ===")
form_data["weight_history"] = [
    {"date": "2026-04-01", "weight": 180.5, "unit": "lb"},
    {"date": "2026-04-16", "weight": 170.0, "unit": "lb"}  # > 10 lbs en 15 dias (muy rapido)
]
res2 = get_nutrition_targets(form_data)
print(res2.get("calculation_details", ""))
