from schemas import ShoppingListModel
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import AUTO_SHOPPING_LIST_PROMPT
import json
import os
from dotenv import load_dotenv

load_dotenv()

shopping_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.0,
    google_api_key=os.environ.get("GEMINI_API_KEY")
).with_structured_output(ShoppingListModel)

ingredients_json = [
  {
    "name": "plátanos maduros",
    "meal_slot": "Desayuno",
    "raw_qty_7_days": "10",
    "raw_qty_15_days": "20",
    "raw_qty_30_days": "40"
  },
  {
    "name": "pollo",
    "meal_slot": "Almuerzo",
    "raw_qty_7_days": "2 lb",
    "raw_qty_15_days": "4 lb",
    "raw_qty_30_days": "8 lb"
  }
]

prompt = AUTO_SHOPPING_LIST_PROMPT.format(ingredients_json=json.dumps(ingredients_json, ensure_ascii=False))

res = shopping_llm.invoke(prompt)
print(res.model_dump_json(indent=2))
