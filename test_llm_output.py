import asyncio
import os
import json
from dotenv import load_dotenv

load_dotenv("backend/.env")

from agent import generate_auto_shopping_list

plan_data = {
    "days": [
        {
            "meals": [
                {
                    "meal": "Desayuno",
                    "name": "Huevos Revueltos",
                    "ingredients": ["2 huevos", "1 pan integral"]
                },
                {
                    "meal": "Almuerzo",
                    "name": "Pollo a la plancha",
                    "ingredients": ["1 pechuga de pollo", "1/2 taza de arroz"]
                }
            ]
        }
    ]
}

def main():
    items = generate_auto_shopping_list(plan_data)
    out = []
    for i, item in enumerate(items):
        d = {}
        if hasattr(item, 'model_dump'):
            d = item.model_dump()
        elif hasattr(item, 'dict'):
            d = item.dict()
        else:
            d = dict(item)
        out.append(d)
    
    with open("backend/test_llm_output.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
