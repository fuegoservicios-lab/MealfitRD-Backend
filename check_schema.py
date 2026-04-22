import asyncio
from db_core import execute_sql_query
import json

def check():
    res = execute_sql_query("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'user_profiles'", fetch_all=True)
    print("USER_PROFILES:", json.dumps(res, indent=2))
    
    res2 = execute_sql_query("SELECT routine_name FROM information_schema.routines WHERE routine_name LIKE 'match_%'", fetch_all=True)
    print("MATCH RPCs:", json.dumps(res2, indent=2))

if __name__ == "__main__":
    check()
