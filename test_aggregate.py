import sys
from shopping_calculator import aggregate_and_deduct_shopping_list

all_ingredients = [
    "1 1/4 lbs de Papas y",
    "2 1/2 lbs de Papas",
]

consumed_ingredients = []

result = aggregate_and_deduct_shopping_list(all_ingredients, consumed_ingredients)

import json
print(json.dumps(result, indent=2, ensure_ascii=False))
