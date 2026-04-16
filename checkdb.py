import os
from supabase import create_client

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not url or not key:
    print("NO ENV")
    exit(1)

supabase = create_client(url, key)
response = supabase.table("user_inventory").select("ingredient_name, quantity, unit").eq("user_id", "f3b6214e-8efe-4e1d-bf31-d3e45d3de745").execute()

for r in response.data:
    print(r)
