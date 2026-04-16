import sys
import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

res = supabase.table("master_ingredients").select("name, density_g_per_unit, market_container, container_weight_g").in_("name", ["Tortillas de trigo integral", "Agua"]).execute()
for r in res.data:
    print(r)
