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
    print("Plan snippet:")
    print(json.dumps(plan["days"][0]["meals"][0], indent=2))
else:
    print("No plan found.")
