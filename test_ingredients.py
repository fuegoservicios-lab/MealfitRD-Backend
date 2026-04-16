import sys
import os
sys.path.insert(0, os.path.abspath('backend'))

from shopping_calculator import _parse_quantity, apply_smart_market_units, get_master_ingredients, aggregate_shopping_list

# mock plan
test_ingredients = [
    "150g Pavo molido",
    "3 uds Tortillas de maíz",
    "100g Repollo fresco",
    "10ml Aceite vegetal",
    "5ml Jugo de limón",
    "2g Sal",
    "1g Orégano",
    "50g Longaniza dominicana",
    "350g Yautía",
    "150g Camarones pelados",
    "100g Zanahoria",
    "15ml Jugo de limón",
    "5g Cilantro fresco",
    "2g Sal",
    "200g Yogurt griego natural descremado",
    "200g Fresas frescas",
    "15g Semillas de chía",
    "30g Casabe"
]

print("--- Parsed Quantities ---")
for raw in test_ingredients:
    qty, unit, name = _parse_quantity(raw)
    print(f"'{raw}' -> {qty} {unit} de {name}")

print("\n--- Aggregated Shopping List ---")
res = aggregate_shopping_list(test_ingredients)
for r in res:
    print(r)
