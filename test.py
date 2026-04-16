import os
from supabase import create_client

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(url, key)

res = supabase.table("user_inventory").select("ingredient_name", "quantity").eq("user_id", "f3b6214e-8efe-4e1d-bf31-d3e45d3de745").execute()
print(f"Total items in DB: {len(res.data)}")
for r in res.data[:5]:
    print(r)
