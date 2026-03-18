import os
from supabase import create_client, Client
from dotenv import load_dotenv
import uuid

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

random_uuid = str(uuid.uuid4())
session_uuid = str(uuid.uuid4())

try:
    res = supabase.table("agent_sessions").insert({"id": session_uuid, "user_id": random_uuid, "locked_at": None}).execute()
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
