import os
from dotenv import load_dotenv
load_dotenv()

from supabase import create_client, Client

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

from datetime import datetime
now = datetime.now()
start_date = datetime(now.year, now.month, 1).isoformat()
print(f"start_date: {start_date}")

user_id = "ed4e4554-1da8-451c-9758-1676ab15b889"
res = supabase.table("api_usage").select("*", count="exact").eq("user_id", user_id).gte("created_at", start_date).execute()

print("Count:", res.count)
print("Data length:", len(res.data))
