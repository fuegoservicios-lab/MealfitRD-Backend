from dotenv import load_dotenv
load_dotenv()
from db_core import supabase

res = supabase.table("user_inventory").select("*").limit(2).execute()
for row in res.data:
    print(row.get("ingredient_name"), row.get("created_at"))
