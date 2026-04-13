import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from shopping_calculator import _parse_quantity

tests = [
    "200g de arrosito crudo",
    "1 taza de carne de pechuga molida",
    "2 raciones de polló frito",
    "3 cucharadas de aceite oliva"
]

for t in tests:
    print(f"Original: {t}")
    qty, unit, name = _parse_quantity(t)
    print(f"Parsed: qty={qty}, unit={unit}, name='{name}'")
    print("-" * 40)
