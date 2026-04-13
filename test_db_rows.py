# NOTE: db_shopping module was removed (legacy shopping list architecture)
from db import get_session_messages
from dotenv import load_dotenv
import json

load_dotenv()

# We need a valid user_id to test, I will just pick the first user id that has shopping items
# actually I will just query supabase directly using REST or Python client

from supabase import create_client, Client
import os

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

res = supabase.table("custom_shopping_items").select("*").order("created_at", desc=True).limit(5).execute()
print(json.dumps(res.data, indent=2, ensure_ascii=False))

