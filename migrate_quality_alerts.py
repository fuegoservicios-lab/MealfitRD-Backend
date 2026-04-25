import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

db_uri = os.environ.get("SUPABASE_DB_URL")
if not db_uri:
    print("Error: SUPABASE_DB_URL not found in environment")
    exit(1)

commands = [
    """
    CREATE TABLE IF NOT EXISTS system_alerts (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        alert_key TEXT NOT NULL UNIQUE,
        alert_type TEXT NOT NULL,
        severity TEXT NOT NULL DEFAULT 'warning',
        title TEXT NOT NULL,
        message TEXT NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        affected_user_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        triggered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        resolved_at TIMESTAMP WITH TIME ZONE NULL
    );
    """,
    "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS quality_alert_at TIMESTAMP WITH TIME ZONE;",
]

try:
    with psycopg.connect(db_uri, autocommit=True) as conn:
        with conn.cursor() as cur:
            for cmd in commands:
                print(f"Executing: {cmd.strip()}")
                cur.execute(cmd)
    print("✅ Migration successful: system_alerts and user_profiles.quality_alert_at are ready.")
except Exception as e:
    print(f"❌ Error during migration: {e}")
