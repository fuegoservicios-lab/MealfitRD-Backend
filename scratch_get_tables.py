import sys
import os
sys.path.append(os.path.dirname(__file__))
from db_core import execute_sql_query
import json

try:
    tables = execute_sql_query("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
    print(json.dumps(tables, indent=2))
except Exception as e:
    print(f"Error: {e}")
