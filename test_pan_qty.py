import os
from supabase import create_client
from dotenv import load_dotenv
import json

load_dotenv("backend/.env")
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

res = supabase.table("user_meal_plans").select("plan_data").eq("user_id", "de7003de-683b-46e1-93a1-c02a28dd7478").order("created_at", desc=True).limit(1).execute()

if res.data:
    plan = res.data[0]["plan_data"]
    for day in plan.get("days", []):
        for meal in day.get("meals", []):
            for ing in meal.get("ingredients", []):
                if "pan" in ing.lower() or "integral" in ing.lower():
                    print(f"Encontrado PAN en plan: {ing}")
else:
    print("No plan found.")
