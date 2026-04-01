import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

db_uri = os.environ.get("SUPABASE_DB_URL")
if not db_uri:
    print("Error: SUPABASE_DB_URL not found in environment")
    exit(1)

commands = [
    "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS paypal_subscription_id TEXT;",
    "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS paypal_plan_id TEXT;",
    "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS subscription_status TEXT;"
]

try:
    with psycopg.connect(db_uri, autocommit=True) as conn:
        with conn.cursor() as cur:
            for cmd in commands:
                print(f"Executing: {cmd}")
                cur.execute(cmd)
    print("✅ Migration successful: Columns added to user_profiles.")
except Exception as e:
    print(f"❌ Error during migration: {e}")
