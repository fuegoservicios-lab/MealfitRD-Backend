import os
import sys
from dotenv import load_dotenv

# Asegurar que el entorno pueda leer nuestras variables
load_dotenv(".env")

from db_plans import get_latest_meal_plan

# UID hardcodeado para debug, o puedes buscar el más reciente
plan = get_latest_meal_plan("c3cc97d2-7c39-4467-bc1a-63d1a81c7ff2")
print("IS RESTOCKED:", plan.get("is_restocked"))
print("KEYS:", plan.keys())
