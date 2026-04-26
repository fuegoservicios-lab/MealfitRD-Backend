from db_core import execute_sql_query
import json

def main():
    res = execute_sql_query("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'plan_chunk_queue'", fetch_all=True)
    print(json.dumps(res, indent=2))
    
    res2 = execute_sql_query("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'meal_plans'", fetch_all=True)
    print(json.dumps(res2, indent=2))

if __name__ == "__main__":
    main()
